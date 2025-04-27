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