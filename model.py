import torch.nn as nn
import torch
import torch.nn.functional as F


class TicTacToeModel(nn.Module):
    def __init__(self):
        super(TicTacToeModel, self).__init__()


        self.fc1 = nn.Linear(9, 64)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 18)
        self.ac2 = nn.ReLU()
        self.fc3 = nn.Linear(18, 9)

    def forward(self, x):
        x = x.flatten()
        x = self.ac1(self.fc1(x))
        x = self.ac2(self.fc2(x))
        x = self.fc3(x)
        return x

