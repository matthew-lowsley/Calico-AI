from ..constants import Vector2, HEX_SIZE, Colour, Pattern

import pygame
import math
import numpy as py

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