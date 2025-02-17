

from collections import deque
import copy
import random
import numpy as np
import pygame
import torch

from game.constants import BATCH_SIZE, LR, MAX_MEMORY, Pattern, DEVICE
from game.player.DQL_player.Model import QNet, QTrainer
from game.player.player import Player
from game.props.board import Board, Space
from game.props.cat import Almond, Callie, Cira, Coconut, Gwenivere, Leo, Oliver, Rumi, Tecolote, Tibbit
from game.props.tile import Shop

OBJECTIVES_SPACES = [Space(0, 4, -4), Space(2, 2, -4), Space(3, 3, -6)]

AVAILBABLE_SPACES = [Space(1,1,-2), Space(2,1,-3), Space(3,1,-4), Space(4,1,-5),
                     Space(5,1,-6), Space(0,2,-2), Space(1,2,-3),
                     Space(3,2,-5), Space(4,2,-6), Space(0,3,-3), Space(1,3,-4),
                     Space(2,3,-5), Space(4,3,-7), Space(-1,4,-3),
                     Space(1,4,-5), Space(2,4,-6), Space(3,4,-7),
                     Space(-1,5,-4), Space(0,5,-5), Space(1,5,-6), Space(2,5,-7),
                     Space(3,5,-8)]

class Agent(Player):

    def __init__(self):
        super().__init__()
        self.n_games = 0
        self.epsilon = 0.6
        self.epsilon_decay = 0.0001
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.net = QNet(1728, 44)
        self.target_net = QNet(1728, 44)
        self.net.to(DEVICE)
        self.target_net.to(DEVICE)
        self.trainer = QTrainer(self.net, self.target_net, lr=LR, gamma=self.gamma)

        self.turn = 0
        self.objectives_placed = False
        self.taken_spaces = []
        self.available_places = []

        self.reset()

    def reset(self):

        self.n_games += 1
        self.turn = 0
        self.objectives_placed = False
        self.points = 0
        self.hand = []
        self.taken_spaces = [0]*22
        self.available_places = copy.deepcopy(AVAILBABLE_SPACES)
    
    def get_state(self, board : Board):
        objectives_spaces = [tuple([0, 4, -4]), tuple([2, 2, -4]), tuple([3, 3, -6])]
        board_state = []
        for key in board.board.keys():
            space = board.board[key]
            space_state = []
            if key in objectives_spaces:
                pass
            else:
                space_state += [0]*36
                if space.tile != None:
                    colour = 0
                    pattern = space.tile.pattern.value
                    if len(space_state) == 36:
                        colour = space.tile.colour.value
                    space_state[(colour*6)+pattern] = 1
                board_state += space_state

        regular_hand_state = []
        if self.objectives_placed == False:
            regular_hand_state = [0]*72
        else:
            for i in range(2):
                tile_state = [0]*36
                colour = self.hand[i].colour.value
                pattern = self.hand[i].pattern.value
                tile_state[(colour*6)+pattern] = 1
                regular_hand_state += tile_state
    
        state = board_state + regular_hand_state
        #print(state)
        #print(len(state))
        return np.array(state)

    def remember(self, state, action, reward, next_state, done, valid):
        self.memory.append((state, action, reward, next_state, done, valid))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        # this loops through the sample of states and trains the model
        #states, actions , rewards, next_states, dones = zip(*mini_sample)
        #self.trainer.train_step(states, actions , rewards, next_states, dones)

        # only doing one at a time at the moment
        for state, action, reward, next_state, done, valid in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done, valid)

    def train_short_memory(self, state, action, reward, next_state, done, valid):
        self.trainer.train_step(state, action, reward, next_state, done, valid)
    
    def act(self, board, shop, events):

        epsilon = self.epsilon - (self.n_games * self.epsilon_decay)

        if self.objectives_placed == False:
            self.place_objectives(board, shop)
            return True
        
        state = self.get_state(board)

        action = self.get_action(state, random.random() < epsilon)
        done, points = self.perform_action(action, shop, board, state)

        self.points += points

        new_state = self.get_state(board)

        self.train_short_memory(state, action, points, new_state, done, True)

        if done:
            self.remember(state, action, points, new_state, done, True) 
            self.train_long_memory()
            print(f'Long Training! Epsilon: {epsilon}')
            if self.n_games % 20 == 0:
                print('Updating Target Net!')
                self.trainer.update_target_net()
            if self.n_games % 50 == 0:
                self.net.save()
        else:
            self.remember(state, action, points, new_state, done, True)
            
        return True
    
    def place(self, board, space, hand_idx):
        valid, points = board.insert_tile(space, self.hand[hand_idx])
        if valid:
            self.hand[hand_idx] = None
            #self.points += points
        return valid, points
    
    def place_objectives(self, board, shop):
        for i in range(3):
            valid, points = board.insert_tile(OBJECTIVES_SPACES[i], self.hand[i])
        self.objectives_placed = True

    def pick(self, shop, shop_idx):
        self.take_tile(shop.take_tile(index=shop_idx))

    def perform_action(self, action, shop, board, state):

        # converts an array (of 32 elements) into a placing and picking actions
        # element 0-24 = position to place
        # element 25-29 = which tile in hand to place
        # element 30-32 = which tile to pick from the shop

        action = np.array(action).argsort()

        points = 0
        i = -1

        while True:
            try:
                j = 0
                if action [i] > 21:
                    j = 1
                #print(action[i]-(j*22))
                valid, points = self.place(board, AVAILBABLE_SPACES[action[i]-(j*22)], j)
                if valid:
                    #print("Valid!")
                    self.taken_spaces[action[i]-(j*22)] = 1
                    break
                else:
                    #print("Invalid!")
                    self.remember(state, action, -10, state, False, False)
                    if i > -44:
                        i -= 1
                    else:
                        new_action = self.get_action(state, True)
                        return self.perform_action(new_action, board, state)
            except Exception as error:
                print(error)
                exit()

        if self.objectives_placed == True:
            self.pick(shop, random.randint(0,2))

        done = False

        self.turn += 1
        if self.turn >= 22:
            done = True

        return done, points
    
    def get_action(self, state, explore=False):

        final_move = [0]*32

        if explore:
            print("Random Move!")
            position = [0]*44
            valid_spaces = [i for i, x in enumerate(self.taken_spaces) if x == 0]
            tile_selection = random.randint(0, 1)
            random.shuffle(valid_spaces)
            position[valid_spaces[0]+(21*tile_selection)] = 1

            final_move = position
        else:
            print("Net Move!")
            state0 = torch.tensor(state, dtype=torch.float, device=DEVICE)
            q_values = self.net(state0)
            #position_q_values, hand_q_values, pick_q_values = self.net(state0)
            #q_values = torch.cat((position_q_values, hand_q_values, pick_q_values))
            move = torch.tensor(q_values, device=DEVICE)
            final_move = move.detach().cpu().numpy()

        #print(len(final_move))
        return final_move
