from .player import Player
from ..props.board import Space
import pygame
import numpy as np
import copy
import random

AVAILBABLE_SPACES = [Space(1,1,-2), Space(2,1,-3), Space(3,1,-4), Space(4,1,-5),
                     Space(5,1,-6), Space(0,2,-2), Space(1,2,-3), Space(3,2,-5), 
                     Space(4,2,-6), Space(0,3,-3), Space(1,3,-4),Space(2,3,-5), 
                     Space(4,3,-7), Space(-1,4,-3), Space(1,4,-5), Space(2,4,-6), 
                     Space(3,4,-7), Space(-1,5,-4), Space(0,5,-5), Space(1,5,-6), 
                     Space(2,5,-7),Space(3,5,-8)]

OBJECTIVES_SPACES = [Space(0, 4, -4), Space(2, 2, -4), Space(3, 3, -6)]

class Random_Player(Player):

    def __init__(self):
        super().__init__()
        self.objectives_placed = False
        self.available_places = []
        self.placed = False

    def place_objectives(self, board):
        hand_idx = np.array([0,1,2,3])
        np.random.shuffle(hand_idx)
        for i, space in enumerate(OBJECTIVES_SPACES):
            valid, points = board.insert_tile(space, self.hand[hand_idx[i]])
        self.objectives_placed = True

    def place(self, board):
        space = self.available_places.pop(0)
        hand_idx = random.randint(0,1)
        vaild, points = board.insert_tile(space, self.hand[hand_idx])
        if vaild:
            self.hand[hand_idx] = None
            self.points += points
            self.placed = True

    def pick(self, shop):
        shop_idx = random.randint(0,2)
        self.take_tile(shop.take_tile(index=shop_idx))

    def reset(self):
        self.objectives_placed = False
        self.points = 0
        self.hand = []
        self.placed = False
        self.available_places = self.get_available_places()

    def get_available_places(self):
        places = copy.deepcopy(AVAILBABLE_SPACES)
        random.shuffle(places)
        return places

    def act(self, board, shop, events):

        if self.objectives_placed == False:
            self.place_objectives(board)
            return True
        
        self.place(board)
        self.pick(shop)
        return True



