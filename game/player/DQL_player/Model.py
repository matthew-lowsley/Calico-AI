import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import csv
import numpy as np
import time

from ...constants import BATCH_SIZE, CLIP_VALUE, DEVICE, TAU, VALIDATE_EVERY
from ..score_plotter import Plotter

class QNet(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        #self.linear1 = nn.Linear(input_size, hidden_size)
        #self.linear2 = nn.Linear(hidden_size, output_size)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
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
        self.validation_states = []
        self.load_validation_states()

        self.recent_scores = []
        self.highest_average_score = 0

        self.Qplotter = Plotter(1, "Games (K)", "Average Action Value (Q)", "Average Q per Game", "Average_Q_Values")
        self.max_q_average = []

        self.average_score_plotter = Plotter(1, f'Games ({VALIDATE_EVERY}s)', 'Scores', f'Average Score Every {VALIDATE_EVERY} Games', f'Average_Score_Every_{VALIDATE_EVERY}_games')

        #self.highest_score_plotter = Plotter(1, "Games (200s)", "Best Models so Far", "Average Score", "Best_Model_Scores")

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
                next_action = self.net(next_state[idx])
                next_action_masked = self.mask_action(next_action, next_state[idx])
                next_action_idx = torch.argmax(next_action_masked).item()
                #next_action_idx = torch.argmax(self.net(next_state[idx])).item()
                Q_new = reward[idx] + self.gamma * self.target(next_state[idx])[next_action_idx]
            else:
                Q_new = reward[idx]

            target[idx][torch.argmax(action[idx]).item()] = Q_new
            #print(Q_new)
            #self.max_q_current.append(float(Q_new))

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.net.parameters(), clip_value=CLIP_VALUE)

        self.optimizer.step()

    def update_target_net(self):
        #for target_parameters, main_parameters in zip(self.target.parameters(), self.net.parameters()):
        #        target_parameters.data.copy_(TAU * main_parameters.data + (1.0 - TAU) * target_parameters.data)
        self.target.load_state_dict(self.net.state_dict())

    def load_validation_states(self):

        with open('transitions.csv', mode='r') as file:
            transitions_file = csv.reader(file)
            next(transitions_file, None) # Skip header
            for lines in transitions_file:
                #print(lines[0])
                # state = []
                # for position in lines[0]:
                #     state.append(int(position))
                self.validation_states.append(eval(lines[0]))

    def mask_action(self, action, state):
        #placement_idxs = [8, 9, 10, 11, 12, 15, 16, 18, 19, 22, 23, 24, 26, 29, 31, 32, 33, 36, 37, 38, 39, 40]
        placement_idxs = [8, 9, 10, 11, 12, 15, 16, 17, 19, 22, 24, 25, 26, 29, 30, 32, 33, 36, 37, 38, 39, 40]
       #placement_idxs = [8, 9, 10, 11, 12, 15, 16, 17, 19, 22, 24, 25, 26, 29, 30, 32, 33, 36, 37, 38, 39, 40]
        # for i, idx in enumerate(placement_idxs):
        #     action[i+22] = -float('inf')
        #     if sum(state[((idx)*12):((idx)*12)+12]) > 0:
        #         action[i] = -float('inf')
        #         action[i+22] = -float('inf')
        #self.print_state(state)
        for i, idx in enumerate(placement_idxs):
            space_state = state[(idx)*12:(idx)*12+12]
            if torch.all(space_state == 0) == False:
                #print(f'Tile {idx} : {space_state} Position is taken!')
                action[i] = -float('inf')
                action[i+22] = -float('inf')
        return action
    
    def print_state(self, state):
        for i in range(47):
            print(f'Action Selecting - Tile {i} : {state[i*12:i*12+12]}')

    def get_action(self, state):
        action = self.net(state)
        return self.mask_action(action, state)

    def validate_and_plot(self):
        
        max_q_current = []

        for state in self.validation_states:
            state = torch.tensor(state, dtype=torch.float, device=DEVICE)
            pred = self.net(state)
            max_q_current.append(torch.max(pred).detach().cpu().numpy())

        self.max_q_average.append(sum(max_q_current)/len(self.validation_states))

        #print(self.max_q_average)

        self.average_score_plotter.plot_average_scores(self.recent_scores, VALIDATE_EVERY)

        if (sum(self.recent_scores)/len(self.recent_scores)) > self.highest_average_score:
            self.highest_average_score = (sum(self.recent_scores)/len(self.recent_scores))
            self.net.save(f'model-{str(self.highest_average_score)}-{str(time.time())}.pth')

        self.recent_scores = []

        self.Qplotter.plot_Q(self.max_q_average)

