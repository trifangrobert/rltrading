import numpy as np
from torch import nn
from torch.optim import Adam

class ActorCritic(nn.Module):
    def __init__(self, input_shape: tuple, action_space: int, lr: float, model: str = "Dense") -> None:
        super(ActorCritic, self).__init__()
        self.input_shape = input_shape
        self.action_space = action_space
        self.lr = lr
        self.optimizer = Adam(self.parameters(), lr=self.lr)

        if model == "CNN":
            self.shared_layers = nn.Sequential(
                nn.Conv1d(input_shape[0], 64, kernel_size=6, padding='same'),
                nn.Tanh(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 32, kernel_size=3, padding='same'),
                nn.Tanh(),
                nn.MaxPool1d(2),
                nn.Flatten()
            )
        elif model == "LSTM":
            self.shared_layers = nn.Sequential(
                nn.LSTM(input_shape[0], 512, batch_first=True),
                nn.ReLU(),
                nn.LSTM(512, 256, batch_first=True),
                nn.ReLU(),
            )
        else:
            self.shared_layers = nn.Sequential(
                nn.Linear(*self.input_shape, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
            )
        



        

class Actor:
    def __init__(self, input_shape: tuple, action_space: int, lr: float, optimizer: str = "Adam") -> None:
        self.input_shape = input_shape
        self.action_space = action_space
        self.lr = lr
        

class Critic:
    def __init__(self, input_shape: tuple, action_space: int, lr: float, optimizer: str = "Adam") -> None:
        self.input_shape = input_shape
        self.action_space = action_space
        self.lr = lr