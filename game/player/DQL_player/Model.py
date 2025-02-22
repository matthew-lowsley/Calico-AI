import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import csv
import numpy as np

from ...constants import DEVICE
from ..score_plotter import Plotter

class QNet(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        #self.linear1 = nn.Linear(input_size, hidden_size)
        #self.linear2 = nn.Linear(hidden_size, output_size)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 864),
            nn.ReLU(),
            nn.Linear(864, 432),
            nn.ReLU(),
            nn.Linear(432, output_size)
        )
        # Try multi-head network for action seperation inside net!
        #self.position = nn.Linear(58, 25)
        #self.hand = nn.Linear(58, 4)
        #self.pick = nn.Linear(58, 3)
    
    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:

    def __init__(self, net, target_net, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.net = net
        self.target = target_net
        self.optimizer = optim.Adam(net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()

        self.plotter = Plotter(1, "Games", "Max Q", "Max Q Value per Game")
        self.max_q_average = []

    def train_step(self, state, action, reward, next_state, done):

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

        pred = self.net(state)
        #print(pred)
        target = pred.clone()
        for idx in range(len(done)):
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.target(next_state[idx]))
            else:
                Q_new = reward[idx]

            target[idx][torch.argmax(action[idx]).item()] = Q_new
            #print(Q_new)
            #self.max_q_current.append(float(Q_new))

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

    def update_target_net(self):
        self.target.load_state_dict(self.net.state_dict())
    
    def validate_and_plot(self):

        states = []

        with open('transitions.csv', mode='r') as file:
            transitions_file = csv.reader(file)
            next(transitions_file, None) # Skip header
            for lines in transitions_file:
                #print(lines[0])
                # state = []
                # for position in lines[0]:
                #     state.append(int(position))
                states.append(eval(lines[0]))
        
        max_q_current = []

        for state in states:
            state = torch.tensor(state, dtype=torch.float, device=DEVICE)
            pred = self.net(state)
            max_q_current.append(torch.max(pred).detach().cpu().numpy())

        self.max_q_average.append(sum(max_q_current)/len(states))

        self.plotter.plot_Q(self.max_q_average)

