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

class CQNet2(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d( in_channels=14 , out_channels=14, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d( in_channels=14, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d( in_channels=32, out_channels=64, kernel_size=3, padding=0, bias=False)
        self.linear1 = nn.Linear( (64*5*5) + 28, 302)
        self.linear2 = nn.Linear( 302, 151)
        self.linear3 = nn.Linear(151 , 76)
        self.linear4 = nn.Linear(76, 44)

        with torch.no_grad():
            weights = torch.tensor([[0., 1., 1.],
                                    [1., 1., 1.],
                                    [0., 1., 1.]]).unsqueeze(0).unsqueeze(0)
            weights.requires_grad = True
            # Setting the Weights on every single filter channel
            weights_conv1 = weights.view(1, 1, 3, 3).repeat(14, 14, 1, 1) # needs to repeat the weight matrix 14 time for all 14 channels
            weights_conv2 = weights.view(1, 1, 3, 3).repeat(32, 14, 1, 1) # 32 output, 14 input
            weights_conv3 = weights.view(1, 1, 3, 3).repeat(64, 32, 1, 1)
            self.conv1.weight = nn.Parameter(weights_conv1) # Apply the weights
            self.conv2.weight = nn.Parameter(weights_conv2)
            self.conv3.weight = nn.Parameter(weights_conv3)

    def forward(self, board_state, hand_state):
        # Place the board through each convolution layer
        board_state = F.relu(self.conv1(board_state))
        board_state = F.relu(self.conv2(board_state))
        board_state = F.relu(self.conv3(board_state))
        # Flatten the convolution matrix into a single dimension: matrix = 64x5x5 -> flattened = 1x1600
        board_state = torch.flatten(board_state, start_dim=1)
        # Add batch dimension to the hand state
        if hand_state.dim() == 1:
            hand_state = torch.unsqueeze(hand_state, 0)
        # Concatenate board and hand state together.
        state = torch.cat((board_state, hand_state), dim=-1)
        # Run flatten board state + hand state through linear (fully connected) layers
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x
    
    def preprocess_input(self, x):

        states = torch.tensor([], device=DEVICE)

        for state in x:
            if isinstance(state, torch.Tensor) == False:
                state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            if state.dim() == 3:
                state = torch.unsqueeze(state, dim=0)
            state = state.permute(0, 3, 1, 2)
            states = torch.cat((states, state), dim=0)

        return states
    
    def save(self, file_name='model.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)