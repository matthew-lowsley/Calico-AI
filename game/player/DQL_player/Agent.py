

from collections import deque
import copy
import random
import numpy as np
import pygame
import torch

from game.constants import BATCH_SIZE, LR, MAX_MEMORY, REPLAY_RECENT, VALIDATE_EVERY, Pattern, DEVICE
from game.player.DQL_player.Memory import Memory
from game.player.DQL_player.Model import QNet
from game.player.DQL_player.Trainer import QTrainer
from game.player.player import Player
from game.props.board import Board, Space
from game.props.cat import Almond, Callie, Cira, Coconut, Gwenivere, Leo, Oliver, Rumi, Tecolote, Tibbit
from game.props.tile import Objective_Tile, Shop

OBJECTIVES_SPACES = [Space(0, 4, -4), Space(2, 2, -4), Space(3, 3, -6)]

#AVAILBABLE_SPACES = [Space(1,1,-2), Space(2,1,-3), Space(3,1,-4), Space(4,1,-5),
#                     Space(5,1,-6), Space(0,2,-2), Space(1,2,-3),
#                     Space(3,2,-5), Space(4,2,-6), Space(0,3,-3), Space(1,3,-4),
#                     Space(2,3,-5), Space(4,3,-7), Space(-1,4,-3),
#                     Space(1,4,-5), Space(2,4,-6), Space(3,4,-7),
#                     Space(-1,5,-4), Space(0,5,-5), Space(1,5,-6), Space(2,5,-7),
#                     Space(3,5,-8)]

AVAILBABLE_SPACES = [Space( 1, 1, -2), Space( 0, 2, -2), Space( 0, 3, -3), Space(-1, 4, -3),
                     Space(-1, 5, -4), Space( 2, 1, -3), Space( 1, 2, -3), Space(1, 3, -4),
                     Space(0, 5, -5), Space(3, 1, -4), Space(2, 3, -5), Space(1, 4, -5),
                     Space(1, 5, -6), Space(4, 1, -5), Space(3,2,-5), Space(2, 4, -6),
                     Space(2, 5, -7), Space(5, 1, -6), Space(4, 2, -6), Space(4, 3, -7),
                     Space(3, 4, -7), Space(3, 5, -8)]

class Agent(Player):

    def __init__(self, memory, trainer, is_head):
        super().__init__()
        self.n_games = -1
        self.epsilon = 0.0 #0.8
        self.epsilon_decay = 0.000 #0.001
        self.gamma = 0.95
        
        self.memory = memory
        self.trainer = trainer

        self.is_head = is_head

        #self.memory = Memory()
        #self.net = QNet(1728, 44)
        #self.target_net = QNet(1728, 44)
        #self.net.to(DEVICE)
        #self.target_net.to(DEVICE)
        #self.trainer = QTrainer(self.net, self.target_net, lr=LR, gamma=self.gamma)

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

    def get_state2(self, board : Board):

        state = {}

        board_state = []
        col = []
        i = 0
        # Each space is a 14 bit array where:
        # - the first 6 bits are colour
        # - the next 6 are pattern and 
        # - the final two are colour_used? and pattern_used?
        for key in board.board.keys():
            space = board.board[key]
            space_state = [0]*14
            if space.tile != None:
                if type(space.tile) is Objective_Tile:
                    space_state = space.tile.state
                    #print(f"{space.tile.objective} State : {space_state}")
                else:
                    colour = space.tile.colour.value
                    pattern = space.tile.pattern.value
                    space_state[colour] = 1
                    space_state[pattern+6] = 1
                    if space.tile.colour_used:
                        space_state[12] = 1
                    if space.tile.pattern_used:
                        space_state[13] = 1
            #print(f"{i} - {space.tile} - {space_state}")
            col.append(space_state)
            if len(col) == 7:
                board_state.append(col)
                col = []
            i += 1

        #print(f"Length of board: {len(board_state)}")
        #print(f"Length of board[0]: {len(board_state[0])}")

        state["board"] = np.array(board_state)

        hand_state = []
        if self.objectives_placed == False:
            hand_state = [0]*28
        else:
            for i in range(2):
                tile_state = [0]*14
                colour = self.hand[i].colour.value
                pattern = self.hand[i].pattern.value
                #print(f"Colour {colour}")
                #print(f"Pattern {pattern}")
                tile_state[colour] = 1
                tile_state[pattern+6] = 1
                hand_state += tile_state

        state["hand"] = np.array(hand_state)

        #print(board_state)

        return state

    def get_state(self, board : Board):
        objectives_spaces = [tuple([0, 4, -4]), tuple([2, 2, -4]), tuple([3, 3, -6])]
        board_state = []
        i = 0

        for key in board.board.keys():
            space = board.board[key]
            space_state = []
            if space.tile != None:
                if type(space.tile) is Objective_Tile:
                    space_state += [1]*12
                else:
                    space_state += [0]*12
                    colour = space.tile.colour.value
                    pattern = space.tile.pattern.value
                    space_state[colour] = 1
                    space_state[pattern+6] = 1
            else:
                space_state += [0]*12
            #print(f'State Creating - Tile {i} : {space_state}')
            i += 1
            board_state += space_state

        regular_hand_state = []
        if self.objectives_placed == False:
            regular_hand_state = [0]*24
        else:
            for i in range(2):
                tile_state = [0]*12
                colour = self.hand[i].colour.value
                pattern = self.hand[i].pattern.value
                tile_state[colour] = 1
                tile_state[pattern+6] = 1
                regular_hand_state += tile_state
    
        state = board_state + regular_hand_state
        #print(state)
        #print(len(state))
        return np.array(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # if len(self.memory) > BATCH_SIZE:
        #     mini_sample = random.sample(self.memory, BATCH_SIZE)
        # else:
        #     mini_sample = self.memory
        
    
        mini_sample = self.memory.sample(BATCH_SIZE)
        
        # this loops through the sample of states and trains the model
        board_states, hand_states, actions , rewards, next_board_states, next_hand_states, dones = zip(*mini_sample)
        self.trainer.train_step(np.array(board_states), np.array(hand_states),
                                np.array(actions), np.array(rewards), 
                                np.array(next_board_states), np.array(next_hand_states),
                                np.array(dones))

        # only doing one at a time at the moment
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def act(self, board, shop, events):

        # epsilon = 1
        epsilon = self.epsilon - (self.n_games * self.epsilon_decay)
        if epsilon < 0.01:
            epsilon = 0.01

        if self.objectives_placed == False:
            self.place_objectives(board, shop)
            return True
        
        state = self.get_state2(board)

        action = self.get_action(state, random.uniform(0, 1) < epsilon)
        #action = self.action_mask(action)
        done, points, reward = self.perform_action(action, shop, board)

        self.points += points

        new_state = self.get_state2(board)

        #self.train_short_memory(state, action, points, new_state, done)
        self.memory.push(state['board'], state['hand'], action, reward, new_state['board'], new_state['hand'], done)


        if self.is_head:
            if done:
                self.trainer.recent_scores.append(self.points)
                if self.n_games % 2 == 0:
                    self.trainer.update_target_net()
                if self.n_games % VALIDATE_EVERY == 0:
                    self.trainer.validate_and_plot()
            if self.turn % REPLAY_RECENT == 0:
                if len(self.memory.queue) == MAX_MEMORY:
                    print(f'Training from Long Term Memory! Epsilon at: {epsilon}')
                    self.train_long_memory()

        #self.get_state2(board)

        return True

    def action_mask(self, action):
        for i in range(len(self.taken_spaces)):
            if self.taken_spaces[i] == 1:
                action[i] = -float('inf')
                action[i+22] = -float('inf')
        return action
    
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

    def pick(self, shop, board):
        
        tiles = shop.tiles
        best_tile_idx = 0
        highest_reward = 0

        for i, tile in enumerate(tiles):
            tile_reward = 0
            for space in AVAILBABLE_SPACES:
                board_space = board.get_space(space)
                if board_space.tile == None:
                    reward = self.calculate_reward(board_space, tile, board)
                    if reward > 0:
                        tile_reward += reward
            if tile_reward > highest_reward:
                highest_reward = tile_reward
                best_tile_idx = i

        self.take_tile(shop.take_tile(index=best_tile_idx))

    def perform_action(self, action, shop, board):

        # converts an array (of 32 elements) into a placing action
        # elements 0-21 = placing left hand tile in one of the 22 spaces
        # elements 22-43 = placing right hand tile in one of the 22 spaces
        #print(f'Turn: {self.turn}')
        #print(action)
        #print(state)
        #print(board.board)
        action = np.array(action).argsort()

        #print(action)

        #print(action)

        points = 0
        i = -1

        #print(action[i])

        j = 0
        if action[i] > 21:
            j = 1

        reward = self.calculate_reward(AVAILBABLE_SPACES[action[i]-(j*22)], self.hand[j], board)
        #print(f"Reward: {reward}")
        
        valid, points = self.place(board, AVAILBABLE_SPACES[action[i]-(j*22)], j)
        if not valid:
            print("selected invalid action!!! Bad news bears!")
            #board.print_indexes_of_space(AVAILBABLE_SPACES)
            #print(f'Trying to place {self.hand[j]} into space {board.board[AVAILBABLE_SPACES[action[i]-(j*22)]]}')
            input("waiting for player input")
            exit()
        self.taken_spaces[action[i]-(j*22)] = 1

        if self.objectives_placed == True:
            self.pick(shop, board)

        done = False

        self.turn += 1
        if self.turn >= 22:
            done = True

        return done, points, reward
    
    def calculate_reward(self, space, tile, board):
        reward = 0
        neighbors = board.contains_tiles(board.find_existing_spaces(space.get_all_neighbors()))
        for neighbor in neighbors:
            if type(neighbor.tile) is Objective_Tile:
                colour_array = np.array(neighbor.tile.state[:6])
                pattern_array = np.array(neighbor.tile.state[6:12])
                colour_idxs = np.where(colour_array == 1)[0]
                pattern_idxs = np.where(pattern_array == 1)[0]
                if tile.colour.value in colour_idxs:
                    reward += 1
                if tile.pattern.value in pattern_idxs:
                    reward += 1
                continue
            if neighbor.tile.colour == tile.colour and neighbor.tile.colour_used == False:
                colour_chain = board.find_chain(neighbor, mode="colour")
                reward += len(colour_chain) #3
            if neighbor.tile.pattern == tile.pattern and neighbor.tile.pattern_used == False:
                pattern_chain = board.find_chain(neighbor, mode="pattern")
                reward += len(pattern_chain) #board.cats[tile.pattern.name].points
        if reward == 0:
            reward -= 1 #(board.cats[tile.pattern.name].points + 3)
        return reward

    def get_action(self, state, explore=False):

        final_move = [0]*32
        board_state = torch.tensor(state['board'], dtype=torch.float, device=DEVICE)
        hand_state = torch.tensor(state['hand'], dtype=torch.float, device=DEVICE)

        if explore:
            #print("Random Move!")
            #position = [0]*44
            #valid_spaces = [i for i, x in enumerate(self.taken_spaces) if x == 0]
            #tile_selection = random.randint(0, 1)
            #random.shuffle(valid_spaces)
            #position[valid_spaces[0]+(21*tile_selection)] = 1.0
            rng = np.random.default_rng()
            random_q_values = rng.uniform(low=0.0, high=1.0, size=44)
            final_move = self.trainer.mask_action(torch.tensor(random_q_values, dtype=torch.float, device=DEVICE), board_state)
            final_move = final_move.detach().cpu().numpy()
        else:
            #print("Net Move!")
            q_values = self.trainer.get_action(board_state, hand_state)
            #position_q_values, hand_q_values, pick_q_values = self.net(state0)
            #q_values = torch.cat((position_q_values, hand_q_values, pick_q_values))
            #move = torch.tensor(q_values, device=DEVICE)
            final_move = q_values.detach().cpu().numpy()

        #print(len(final_move))
        return final_move
