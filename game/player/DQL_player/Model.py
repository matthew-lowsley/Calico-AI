import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from ...constants import DEVICE

class QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        #self.linear1 = nn.Linear(input_size, hidden_size)
        #self.linear2 = nn.Linear(hidden_size, output_size)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 942),
            nn.ReLU(),
            nn.Linear(942, 471),
            nn.ReLU(),
            nn.Linear(471, 235),
            nn.ReLU(),
            nn.Linear(235, 117),
            nn.ReLU(),
            nn.Linear(117, 58),
            nn.ReLU(),
            nn.Linear(58, 32)
        )
        # Try multi-head network for action seperation inside net!
        #self.position = nn.Linear(58, 25)
        #self.hand = nn.Linear(58, 4)
        #self.pick = nn.Linear(58, 3)
    
    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

class QTrainer:

    def __init__(self, net, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.net = net
        self.optimizer = optim.Adam(net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, valid):

        # Convet lists/single values to tensors
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.float, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            valid = (valid, )

        pred = self.net(state)

        target = pred.clone()
        for idx in range(len(done)):
            print(valid[idx])
            if not valid[idx]:
                Q_new = -float('inf')
            elif not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.net(next_state[idx]))
            else:
                Q_new = reward[idx]

            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
