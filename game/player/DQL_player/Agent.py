

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
                     Space(5,1,-6), Space(0,2,-2), Space(1,2,-3), Space(2, 2, -4),
                     Space(3,2,-5), Space(4,2,-6), Space(0,3,-3), Space(1,3,-4),
                     Space(2,3,-5), Space(3, 3, -6), Space(4,3,-7), Space(-1,4,-3),
                     Space(0, 4, -4), Space(1,4,-5), Space(2,4,-6), Space(3,4,-7),
                     Space(-1,5,-4), Space(0,5,-5), Space(1,5,-6), Space(2,5,-7),
                     Space(3,5,-8)]

class Agent(Player):

    def __init__(self):
        super().__init__()
        self.n_games = 0
        self.epsilon = 1
        self.explore_chance = 0.4
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.net = QNet(1884, 256, 32)
        self.net.to(DEVICE)
        self.trainer = QTrainer(self.net, lr=LR, gamma=self.gamma)

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
        self.taken_spaces = [0]*25
        self.available_places = copy.deepcopy(AVAILBABLE_SPACES)
    
    def get_state(self, board : Board, shop : Shop, hand):
        objectives_spaces = [tuple([0, 4, -4]), tuple([2, 2, -4]), tuple([3, 3, -6])]
        board_state = []
        for key in board.board.keys():
            space = board.board[key]
            space_state = []
            if key in objectives_spaces:
                space_state += [0]*6
            else:
                space_state += [0]*36
            if space.tile != None:
                colour = 0
                pattern = space.tile.pattern.value
                if len(space_state) == 36:
                    colour = space.tile.colour.value
                space_state[(colour*6)+pattern] = 1
            board_state += space_state
        
        shop_state = []
        for tile in shop.tiles:
            tile_state = [0]*36
            colour = tile.colour.value
            pattern = tile.pattern.value
            tile_state[(colour*6)+pattern] = 1
            shop_state += tile_state

        cat_state = [0]*6
        for pattern in Pattern:
            cat = board.cats[pattern.name]
            #print(type(cat))
            match type(cat).__qualname__:
                case Oliver.__qualname__:
                    cat_state[pattern.value] = 1
                case Callie.__qualname__:
                    cat_state[pattern.value] = 2
                case Tibbit.__qualname__:
                    cat_state[pattern.value] = 3
                case Rumi.__qualname__:
                    cat_state[pattern.value] = 4
                case Coconut.__qualname__:
                    cat_state[pattern.value] = 5
                case Tecolote.__qualname__:
                    cat_state[pattern.value] = 6
                case Cira.__qualname__:
                    cat_state[pattern.value] = 7
                case Almond.__qualname__:
                    cat_state[pattern.value] = 8
                case Gwenivere.__qualname__:
                    cat_state[pattern.value] = 9
                case Leo.__qualname__:
                    cat_state[pattern.value] = 10

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
        
        starting_hand_state = []
        if self.objectives_placed == True:
            starting_hand_state = [0]*24
        else:
            for tile in hand:
                tile_state = [0]*6
                if tile != None:
                    pattern = tile.pattern.value
                    tile_state[pattern] = 1
                starting_hand_state += tile_state

        state = board_state + shop_state + cat_state + regular_hand_state + starting_hand_state
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

        epsilon = self.epsilon - (self.n_games / 10000)

        if self.objectives_placed == False:
            self.place_objectives(board, shop)
            return True
        
        state = self.get_state(board, shop, self.hand)

        action = self.get_action(state, random.random() < epsilon - 0.6)
        done, points = self.perform_action(action, board, shop, state)

        self.points += points

        new_state = self.get_state(board, shop, self.hand)

        self.train_short_memory(state, action, points, new_state, done, True)

        if done:
            self.remember(state, action, self.points * 2, new_state, done, True) 
            self.train_long_memory()
            print(f'Long Training! Epsilon: {epsilon}')
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
        epsilon = self.epsilon - (self.n_games / 1000)
        for i in range(3):
            state = self.get_state(board, shop, self.hand)
            action = self.get_action(state, random.random() < epsilon - 0.6)
            points = self.perform_action(action, board, shop, state)
        self.objectives_placed = True

    def pick(self, shop, shop_idx):
        self.take_tile(shop.take_tile(index=shop_idx))

    def perform_action(self, action, board, shop, state):

        # converts an array (of 32 elements) into a placing and picking actions
        # element 0-24 = position to place
        # element 25-29 = which tile in hand to place
        # element 30-32 = which tile to pick from the shop

        position = np.array(action[:25])
        #print(position)
        position = position.argsort() # list of indexes in the AVAILBABLE_SPACES sorted by Q-Value.
        #print("Position: "+str(position))

        hand = np.array(action[25:29])
        hand = hand.argsort() # list of indexes in hand sorted by Q-Value.
        #print("Hand: "+str(hand))

        shop_choice = np.array(action[29:])
        shop_choice = shop_choice.argsort() # list of indexes in shop sorted by Q-Value.
        #print("Shop: "+str(shop_choice))

        points = 0
        i = -1
        j = -1

        while True:
            try:
                valid, points = self.place(board, AVAILBABLE_SPACES[position[i]], hand[j])
                if valid:
                    #print("Valid!")
                    self.taken_spaces[position[i]] = 1
                    break
                else:
                    #print("Invalid!")
                    self.remember(state, action, -10, state, False, False)
                    if i > -25:
                        i -= 1
                    elif i == -25:
                        i = -1
                        j -= 1
                    elif j == -4:
                        new_action = self.get_action(state, True)
                        return self.perform_action(new_action, board, shop, state)
                    else:
                        i -= 1
            except:
                print("No position is valid for placing!!!")
                pygame.time.wait(99999)

        if self.objectives_placed == True:
            self.pick(shop, shop_choice[-1])

        done = False

        self.turn += 1
        if self.turn >= 25:
            done = True

        return done, points
    
    def get_action(self, state, explore=False):

        final_move = [0]*32

        if explore:
            print("Random Move!")
            position = [0]*25
            valid_spaces = [i for i, x in enumerate(self.taken_spaces) if x == 0]
            if self.objectives_placed == False:
                 if self.taken_spaces[7] == 0:
                     valid_spaces[0] = 7
                 elif self.taken_spaces[13] == 0:
                     valid_spaces[0] = 13
                 elif self.taken_spaces[16] == 0:
                     valid_spaces[0] = 16
            else:
                random.shuffle(valid_spaces)
            position[valid_spaces[0]] = 1

            hand = [0]*4
            if self.objectives_placed == False:
                hand[random.randint(0,3)] = 1
            else:
                hand[random.randint(0,1)] = 1

            pick = [0]*3
            if self.objectives_placed:
                pick[random.randint(0,2)] = 1

            final_move = position + hand + pick

        else:
            print("Net Move!")
            state0 = torch.tensor(state, dtype=torch.float, device=DEVICE)
            q_values = self.net(state0)
            #position_q_values, hand_q_values, pick_q_values = self.net(state0)
            #q_values = torch.cat((position_q_values, hand_q_values, pick_q_values))
            move = torch.tensor(q_values, device=DEVICE)
            final_move = move.detach().cpu().numpy()

        return final_move
