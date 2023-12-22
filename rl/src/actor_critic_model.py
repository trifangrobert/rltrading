import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from datetime import datetime
import os
from tqdm import tqdm
from collections import deque
import yfinance as yf

from .trading_env import TradingEnv

torch.autograd.set_detect_anomaly(True)

class ActorCriticAgent:
    def __init__(self, lookback_window_size: int = 50, lr: float = 0.00005, model: str = "Dense", best_average: float = 0) -> None:
        self.lookback_window_size = lookback_window_size
        self.model = model
        self.action_space = np.array([0, 1, 2])
        self.state_size = (self.lookback_window_size, 10) # lookback_window_size x (market_history + order_history = 10)

        self.log_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"_model"

        self.lr = lr
        self.output_shape = 256

        self.shared_layers = ActorCritic(input_shape=self.state_size,  hidden_size=512, output_shape=self.output_shape, model=self.model)
        self.actor = Actor(self.shared_layers, input_shape=self.output_shape, hidden_size=128, output_shape=self.action_space.shape[0])
        self.critic = Critic(self.shared_layers, input_shape=self.output_shape, hidden_size=128)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        self.best_average = best_average
        

    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True):
        '''
        generalized advantage estimation
        https://arxiv.org/abs/1506.02438
        '''  
        # print(f"rewards: {rewards}")

        # check if the parameters do not have nan values
        assert not np.isnan(rewards).any()
        assert not np.isnan(dones).any()
        assert not np.isnan(values).any()
        assert not np.isnan(next_values).any()

        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas).squeeze()

        # print(f"deltas: {deltas}")
        gaes = np.copy(deltas)
        gaes = gaes.astype(np.float32)

        # print(f"gaes: {gaes}")
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        # print(f"gaes: {gaes}")
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
    
    def train(self, env, train_episodes: int = 10, training_batch_size: int = 500, save_path: str = "./experiments/") -> None:
        save_path = os.path.join(save_path, self.log_name)
        total_average = deque(maxlen=5)

        for episode in tqdm(range(train_episodes), desc="Training"):
            state = env.reset(env_step_size=training_batch_size)

            states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []

            for _ in range(training_batch_size):
                action, prediction = self.act(state) # action is either hold, buy or sell and prediction is the probability of each action

                next_state, reward, done = env.step(action) 

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                predictions.append(prediction)
                dones.append(done)
                next_states.append(next_state)

                state = next_state

            # import pdb ; pdb.set_trace()
            states = np.array(states, dtype=np.float32)
            states = torch.tensor(states)

            actions = np.array(actions, dtype=np.float32)
            actions = torch.tensor(actions)

            rewards = torch.tensor(rewards, dtype=torch.float32)

            predictions = np.array(predictions, dtype=np.float32)
            predictions = torch.tensor(predictions)

            dones = np.array(dones, dtype=np.float32)
            dones = torch.tensor(dones)

            next_states = np.array(next_states, dtype=np.float32)
            next_states = torch.tensor(next_states)

            curr_values = self.critic(states).detach().numpy()
            next_values = self.critic(next_states).detach().numpy()
    
            advantages, target = self.get_gaes(rewards, dones, curr_values, next_values)

            advantages = torch.tensor(advantages, dtype=torch.float32)
            actions = actions.unsqueeze(1)

            y_true = torch.cat([advantages, predictions, actions], dim=1)

            actor_loss = self.actor_loss(y_true, self.actor(states))
            target = torch.tensor(target, dtype=torch.float32)
            critic_loss = self.critic_loss(target, self.critic(states))

            total_loss = actor_loss + critic_loss
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            total_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_average.append(env.net_worth)
            average = np.mean(total_average)

            if self.best_average < average:
                self.save(save_path)
                print("Saving model at episode {} with average net worth of {}".format(episode, average))
                self.best_average = average

            if episode % 10 == 0:
                print("Episode: {}, total average: {}, current episode net worth: {}".format(episode, average, env.net_worth))

    def test(self, env, test_episodes: int = 10) -> None:
        average_net_worth = 0
        average_orders = 0
        no_profit_episodes = 0
        profit = 0
        for episode in tqdm(range(test_episodes), desc="Testing"):
            state = env.reset()
            while True:
                env.render(visualize=True)
                action, prediction = self.act(state)
                state, reward, done = env.step(action)

                if env.current_step == env.end_step:
                    profit = env.net_worth - env.initial_balance
                    average_net_worth += env.net_worth
                    average_orders += env.episode_orders
                    if env.net_worth < env.initial_balance:
                        no_profit_episodes += 1
                    print("The model made it to the end of episode {} with {}.".format(episode, env.net_worth))
                    break

        print("Average net worth: {}".format(average_net_worth / test_episodes))
        print("Average orders: {}".format(average_orders / test_episodes))
        print("No profit episodes: {}".format(no_profit_episodes))
        print("Profit: {}".format(profit))


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
    def __init__(self, input_shape: tuple, hidden_size: int = 512, output_shape: int = 256, model: str = "Dense") -> None:
        super(ActorCritic, self).__init__()
        self.input_shape = input_shape

        if model == "Dense":
            self.shared_layers = nn.Sequential(
                nn.Linear(input_shape[0] * input_shape[1], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_shape),
                nn.ReLU(),
            )

    def forward(self, x):
        x = x.view(-1, self.input_shape[0] * self.input_shape[1])
        return self.shared_layers(x)


class Actor(nn.Module):
    def __init__(self, shared_layers: nn.Module, input_shape: int, hidden_size: int, output_shape: int) -> None:
        super(Actor, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.shared_layers = shared_layers

        self.actor_head = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_shape),
            nn.Softmax(dim=-1)
        )


    def forward(self, state):
        x = self.shared_layers(state)
        x = self.actor_head(x) # check shapes
        return x
        

class Critic(nn.Module):
    def __init__(self, shared_layers: nn.Module, input_shape: int, hidden_size: int) -> None:
        super(Critic, self).__init__()
        self.input_shape = input_shape
        self.shared_layers = shared_layers

        self.critic_head = nn.Sequential(
            nn.Linear(self.input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        x = self.shared_layers(state)
        x = self.critic_head(x) # check shapes
        return x
        

if __name__ == "__main__":
    # test actor critic model
    lookback_window_size = 20

    agent = ActorCriticAgent(lookback_window_size=lookback_window_size)
    print(agent.shared_layers)
    print(agent.actor)
    print(agent.critic)

    env = TradingEnv(stock_data=yf.Ticker("MSFT"), period="5y", initial_balance=1000, lookback_window_size=lookback_window_size)
    agent.train(env, train_episodes=100)

    # agent.load("../experiments/2023-12-22-17-26-29_model")
    # agent.load("../experiments/2023-12-22-18-31-34_model")
    # agent.test(env, test_episodes=1)