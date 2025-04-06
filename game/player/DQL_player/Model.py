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

class CQNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d( in_channels=14 , out_channels=14, kernel_size=3, padding=0, bias=False)
        self.conv2 = nn.Conv2d( in_channels=14, out_channels=14, kernel_size=3, padding=0, bias=False)
        self.linear = nn.Linear((14*3*3) + 28 , 44)

        with torch.no_grad():
            weights = torch.tensor([[0., 1., 1.],
                                    [1., 1., 1.],
                                    [0., 1., 1.]]).unsqueeze(0).unsqueeze(0)
            weights.requires_grad = True
            weights = weights.view(1, 1, 3, 3).repeat(14, 14, 1, 1)
            self.conv1.weight = nn.Parameter(weights)
            self.conv2.weight = nn.Parameter(weights)

    def forward(self, board_state, hand_state):
        #print(f"Size 1: {board_state.shape}")
        board_state = F.relu(self.conv1(board_state))
        #print(f"Size 2: {x.shape}")
        board_state = F.relu(self.conv2(board_state))
        #print(f"Size 3: {x.shape}")
        board_state = torch.flatten(board_state, start_dim=1)
        #print(f"Size 4: {x.shape}")
        #print(board_state)
        #print(hand_state)
        if hand_state.dim() == 1:
            hand_state = torch.unsqueeze(hand_state, 0)
        state = torch.cat((board_state, hand_state), dim=-1)
        x = self.linear(state)
        #print(f"Size 5: {x.shape}")
        return x
    
    def preprocess_input(self, x):

        #states = []

        # for i in range(len(x)):
        #     current_state = x[i]
        #     if isinstance(current_state, torch.Tensor):
        #         print(f"state shape: {current_state.shape}")
        #         print(f"state dims: {current_state.dim()}")
        #         if current_state.dim() == 3:
        #             print("Trying!!!")
        #             current_state = torch.unsqueeze(current_state, 0)
        #         print(f"state dims again: {x[i].shape}")
        #         states.append(current_state.permute(0, 3, 1, 2))
        #     else:
        #         current_state = torch.tensor(x[i], dtype=torch.float32, device=DEVICE)
        #         if current_state.dim() == 3:
        #             current_state = current_state.unsqueeze(0)
        #         states.append(current_state.permute(0, 3, 1, 2))
        # return states

        #print(f"Start: {x}")

        #print(f"Start Shape: {x.shape}")

        states = torch.tensor([], device=DEVICE)

        for state in x:
            if isinstance(state, torch.Tensor) == False:
                state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            if state.dim() == 3:
                #print(f"Mid Shape: {state.shape}")
                state = torch.unsqueeze(state, dim=0)
            state = state.permute(0, 3, 1, 2)
            states = torch.cat((states, state), dim=0)

        #print(f"End: {x}")

        #states = torch.squeeze(states)

        #if states.dim() == 5:
        #    states = torch.squeeze(states) 

        #print(f"Ending Shape: {states.shape}")

        return states
    
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

    def train_step(self, board_state, hand_state, action, reward, next_board_state, next_hand_state, done):

        # Convet lists/single values to tensors
        board_state = torch.tensor(board_state, dtype=torch.float, device=DEVICE)
        hand_state = torch.tensor(hand_state, dtype=torch.float, device=DEVICE)
        next_board_state = torch.tensor(next_board_state, dtype=torch.float, device=DEVICE)
        next_hand_state = torch.tensor(next_hand_state, dtype=torch.float, device=DEVICE )
        action = torch.tensor(action, dtype=torch.float, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)

        #state = state.permute(2, 0, 1).unsqueeze(state)
        #next_state = state.permute(2, 0, 1).unsqueeze(next_state)

        if len(board_state.shape) == 1:
            board_state = torch.unsqueeze(board_state, 0)
            next_board_state = torch.unsqueeze(next_board_state, 0)
            hand_state = torch.unsqueeze(hand_state, 0)
            next_hand_state = torch.unsqueeze(next_hand_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        processed_states = self.net.preprocess_input(board_state)

        #print(f"Next States Shape: {next_state.shape}")

        processed_next_states = self.net.preprocess_input(next_board_state)
        pred = self.net(processed_states, hand_state)
        #print(pred)
        target = pred.clone()
        for idx in range(len(done)):
            if not done[idx]:
                #print(f"Single next state shape:  {processed_next_states[idx].shape}")
                next_action = self.net(torch.unsqueeze(processed_next_states[idx], 0), next_hand_state[idx])
                next_action_masked = self.mask_action(next_action[0], next_board_state[idx])
                next_action_idx = torch.argmax(next_action_masked).item()
                #print("Action Masked!")
                #next_action_idx = torch.argmax(self.net(next_state[idx])).item()
                Q_new = reward[idx] + self.gamma * self.target(torch.unsqueeze(processed_next_states[idx], 0), next_hand_state[idx])[0][next_action_idx]
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
                self.validation_states.append({'board_state' : eval(lines[0]), 'hand_state' : eval(lines[1])})

    def mask_action(self, action, state):
        placement_idxs = [8, 9, 10, 11, 12, 15, 16, 17, 19, 22, 24, 25, 26, 29, 30, 32, 33, 36, 37, 38, 39, 40]
        #print(f"Action state: {state}")
        #print(f"Action: {action}")
        for i, idx in enumerate(placement_idxs):
            col = (idx // 7)
            row = (idx % 7)
            #print(f"col: {col}, row: {row}")
            #print(state[col][row])
            if torch.all(state[col][row] == 0) == False:
                action[i] = -float('inf')
                action[i+22] = -float('inf')
        #print(action)
        return action
    
    def print_state(self, state):
        for i in range(47):
            print(f'Action Selecting - Tile {i} : {state[i*12:i*12+12]}')

    def get_action(self, board_state, hand_state):
        processed_board_state = self.net.preprocess_input(torch.unsqueeze(board_state, dim=0))
        #print(f"processed state: {processed_state}")
        action = self.net(processed_board_state, hand_state)
        return self.mask_action(action[0], board_state)

    def validate_and_plot(self):
        
        max_q_current = []

        for state in self.validation_states:
            board_state = torch.tensor(state['board_state'], dtype=torch.float, device=DEVICE)
            hand_state = torch.tensor(state['hand_state'], dtype=torch.float, device=DEVICE)
            #processed_state = self.net.preprocess_input(state)
            pred = self.get_action(board_state, hand_state)
            max_q_current.append(torch.max(pred).detach().cpu().numpy())

        self.max_q_average.append(sum(max_q_current)/len(self.validation_states))

        #print(self.max_q_average)

        self.average_score_plotter.plot_average_scores(self.recent_scores, VALIDATE_EVERY)

        if (sum(self.recent_scores)/len(self.recent_scores)) > self.highest_average_score:
            self.highest_average_score = (sum(self.recent_scores)/len(self.recent_scores))
            self.net.save(f'model-{str(self.highest_average_score)}-{str(time.time())}.pth')

        self.recent_scores = []

        self.Qplotter.plot_Q(self.max_q_average)

