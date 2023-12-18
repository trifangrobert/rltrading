import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from datetime import datetime

from copy import deepcopy

class ActorCriticAgent:
    def __init__(self, lookback_window_size: int = 50, order_window_size: int = 10, lr: float = 0.00005, epochs: int = 1, batch_size: int = 32, model: str = "Dense") -> None:
        self.lookback_window_size = lookback_window_size
        self.model = model

        self.action_space = np.array([0, 1, 2])

        self.log_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"_model"

        self.state_size = (self.lookback_window_size, order_window_size) # market_history and order_history

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_shape = 256

        self.shared_layers = ActorCritic(self.state_size, self.output_shape, self.model)
        self.actor = Actor(self.shared_layers, input_shape=self.output_shape, output_shape=self.action_space.shape[0])
        self.critic = Critic(self.shared_layers, input_shape=self.output_shape)
        

    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True):
        '''
        generalized advantage estimation
        https://arxiv.org/abs/1506.02438
        '''
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = deepcopy(deltas)
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

    def train(self, env, train_episode: int = 50, training_batch_size: int = 500, save_path: str = "../experiments") -> None:
        save_path = save_path + "/" + self.log_name + ".pt"


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

    