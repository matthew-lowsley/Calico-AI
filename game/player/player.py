from ..constants import Vector2, hand_positions, HEX_SIZE

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
    
    def draw_hand(self, win):
        for i in range(len(self.hand)):
            if self.hand[i] != None:
                pygame.draw.rect(win, (128,128,128), pygame.Rect(hand_positions[i].x, hand_positions[i].y, HEX_SIZE.x*2, HEX_SIZE.y*2))
                self.hand[i].draw(win, Vector2(hand_positions[i].x+HEX_SIZE.x,hand_positions[i].y+HEX_SIZE.y), HEX_SIZE)
        