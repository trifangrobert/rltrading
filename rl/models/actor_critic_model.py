import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from datetime import datetime
import os
from tqdm import tqdm
from collections import deque
import yfinance as yf

from rl.trading_env import TradingEnv

torch.autograd.set_detect_anomaly(True)


class ActorCriticAgent:
    def __init__(self, lookback_window_size: int = 50, order_window_size: int = 10, lr: float = 0.00005, epochs: int = 1, batch_size: int = 32, model: str = "Dense") -> None:
        self.lookback_window_size = lookback_window_size
        self.model = model
        self.action_space = np.array([0, 1, 2])
        self.state_size = (self.lookback_window_size, order_window_size) # market_history and order_history

        self.log_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"_model"

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_shape = 256

        self.shared_layers = ActorCritic(self.state_size, self.output_shape, self.model)
        self.actor = Actor(self.shared_layers, input_shape=self.output_shape, output_shape=self.action_space.shape[0])
        self.critic = Critic(self.shared_layers, input_shape=self.output_shape)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)
        

    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True):
        '''
        generalized advantage estimation
        https://arxiv.org/abs/1506.02438
        '''  
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas).squeeze()
        gaes = np.copy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)
    
    def actor_loss(self, y_true, y_pred):
        '''
        proximal policy optimization
        https://arxiv.org/abs/1707.06347 
        '''
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space.shape[0]], y_true[:, 1+self.action_space.shape[0]:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = torch.clamp(prob, 1e-10, 1.0)
        old_prob = torch.clamp(old_prob, 1e-10, 1.0)

        ratio = torch.exp(torch.log(prob) - torch.log(old_prob))

        p1 = ratio * advantages
        # clip to prevent from exploding
        p2 = torch.clamp(ratio, min=1 - LOSS_CLIPPING, max=1 + LOSS_CLIPPING) * advantages 

        # check ???
        actor_loss = -torch.mean(torch.min(p1, p2))

        # entropy to encourage exploration
        entropy = -(y_pred * torch.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * torch.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss
    
    def critic_loss(self, y_true, y_pred):
        '''
        standard loss
        '''
        value_loss = torch.mean((y_true - y_pred) ** 2)
        return value_loss
    
    def train(self, env, train_episodes: int = 50, training_batch_size: int = 500, save_path: str = "experiments/") -> None:
        save_path = os.path.join(save_path, self.log_name)
        total_average = deque(maxlen=100)
        best_average = 0

        for episode in tqdm(range(train_episodes), desc="Training"):
            state = env.reset(env_step_size=training_batch_size)

            states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []

            for _ in range(training_batch_size):
                action, prediction = self.act(state)
                next_state, reward, done = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                predictions.append(prediction)
                dones.append(done)
                next_states.append(next_state)

                state = next_state

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            predictions = torch.tensor(predictions, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)

            curr_values = self.critic(states).detach().numpy()
            next_values = self.critic(next_states).detach().numpy()
    
            advantages, target = self.get_gaes(rewards, dones, curr_values, next_values)

            advantages = torch.tensor(advantages, dtype=torch.float32)
            actions = actions.unsqueeze(1)

            y_true = torch.cat([advantages, predictions, actions], dim=1)

            actor_loss = self.actor_loss(y_true, self.actor(states))
            target = torch.tensor(target, dtype=torch.float32)
            critic_loss = self.critic_loss(target, self.critic(states))

            # import pdb; pdb.set_trace()
            total_loss = actor_loss + critic_loss
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            total_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()


            total_average.append(env.net_worth)
            average = np.mean(total_average)

            if episode > len(total_average) and best_average < average:
                self.save(save_path)
                print("Saving model at episode {} with average net worth of {}".format(episode, average))
                best_average = average

            if episode % 10 == 0:
                print("Episode: {}, total average: {}, current episode average: {}".format(episode, average, env.net_worth))

    def test(self, env, test_episodes: int = 10) -> None:
        average_net_worth = 0
        average_orders = 0
        no_profit_episodes = 0
        for episode in tqdm(range(test_episodes), desc="Testing"):
            state = env.reset()
            while True:
                env.render(visualize=True)
                action, prediction = self.act(state)
                state, reward, done = env.step(action)

                if env.current_step == env.end_step:
                    average_net_worth += env.net_worth
                    average_orders += env.episode_orders
                    if env.net_worth < env.initial_balance:
                        no_profit_episodes += 1
                    print("The model made it to the end of episode {} with {}.".format(episode, env.net_worth))
                    break

        print("Average net worth: {}".format(average_net_worth / test_episodes))
        print("Average orders: {}".format(average_orders / test_episodes))
        print("No profit episodes: {}".format(no_profit_episodes))



    def act(self, state):
        state = state.astype(np.float32)
        state = torch.tensor(state)
        prediction = self.actor(state)
        dist = torch.distributions.Categorical(prediction)
        action = dist.sample().item()
        return action, prediction.detach().numpy()[0]
    
    def load(self, path: str) -> None:
        self.shared_layers.load_state_dict(torch.load(path + "_shared"))
        self.actor.load_state_dict(torch.load(path + "_actor"))
        self.critic.load_state_dict(torch.load(path + "_critic"))
        print("Loaded model from {}".format(path))

    def save(self, path: str) -> None:
        torch.save(self.shared_layers.state_dict(), path + "_shared")
        torch.save(self.actor.state_dict(), path + "_actor")
        torch.save(self.critic.state_dict(), path + "_critic")
        print("Saved model to {}".format(path))
    

class ActorCritic(nn.Module):
    def __init__(self, input_shape: tuple, output_shape: int = 256, model: str = "Dense") -> None:
        super(ActorCritic, self).__init__()
        self.input_shape = input_shape

        if model == "Dense":
            self.shared_layers = nn.Sequential(
                nn.Linear(input_shape[0] * input_shape[1], 512),
                nn.ReLU(),
                nn.Linear(512, output_shape),
                nn.ReLU(),
            )

    def forward(self, x):
        x = x.view(-1, self.input_shape[0] * self.input_shape[1])
        return self.shared_layers(x)


class Actor(nn.Module):
    def __init__(self, shared_layers: nn.Module, input_shape: int, output_shape: int) -> None:
        super(Actor, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.shared_layers = shared_layers

        self.actor_head = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_shape),
            nn.Softmax(dim=-1)
        )


    def forward(self, state):
        x = self.shared_layers(state)
        x = self.actor_head(x) # check shapes
        return x
        

class Critic(nn.Module):
    def __init__(self, shared_layers: nn.Module, input_shape: int) -> None:
        super(Critic, self).__init__()
        self.input_shape = input_shape
        self.shared_layers = shared_layers

        self.critic_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        x = self.shared_layers(state)
        x = self.critic_head(x) # check shapes
        return x
        

if __name__ == "__main__":
    # test actor critic model
    agent = ActorCriticAgent()
    print(agent.shared_layers)
    print(agent.actor)
    print(agent.critic)

    env = TradingEnv(stock_data=yf.Ticker("MSFT"), period="5y", initial_balance=1000, lookback_window_size=50)
    agent.train(env)
    agent.test(env)