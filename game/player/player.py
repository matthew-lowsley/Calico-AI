import pygame

class Player:

    def __init__(self):
        self.hand = []
        self.points = 0

    def act(self, events):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    
    def take_tile(self, tile):
        for i in range(self, tile):
            if self.hand[i] == None:
                self.hand[i] = tile