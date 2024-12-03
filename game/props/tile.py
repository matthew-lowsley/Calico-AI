from ..constants import Vector2, HEX_SIZE, Colour, Pattern

import pygame
import math
import copy
import numpy as np

class Tile:

    def __init__(self, colour : Colour, pattern : Pattern):
        self.colour : Colour = colour
        self.pattern : Pattern = pattern
        self.colour_used = False
        self.pattern_used = False
    
    def draw(self, win, position, size=HEX_SIZE):
        points = []
        for i in range(6):
            angle = math.radians(i * 60 - 30)
            x = position.x + HEX_SIZE.x * math.cos(angle)
            y = position.y + HEX_SIZE.y * math.sin(angle)
            points.append([x, y])
        pygame.draw.polygon(win, (0, 0, 255), points, 0)

class Shop: 
    
    def __init__(self):
        self.shop_areas = [pygame.Rect(320, 620, HEX_SIZE.x*2, HEX_SIZE.y*2), 
                           pygame.Rect(420, 620, HEX_SIZE.x*2, HEX_SIZE.y*2), 
                           pygame.Rect(520, 620, HEX_SIZE.x*2, HEX_SIZE.y*2)]
        self.tiles = [None, None, None]
        #self.stock_shop()
    
    def stock_shop(self):
        for i in range(len(self.tiles)):
            if self.tiles[i] == None:
                self.tiles[i] = Tile(Colour.DarkBlue, Pattern.CHURCHES)

    def take_tile(self, index : int):
        tile = self.tiles[index]
        self.tiles[index] = None
        self.stock_shop()
        return tile
    
    def draw(self, win):
        for i in range(len(self.tiles)):
            if self.tiles[i] != None:
                pygame.draw.rect(win, (128,128,128), self.shop_areas[i])
                self.tiles[i].draw(win, Vector2(self.shop_areas[i].centerx, self.shop_areas[i].centery))