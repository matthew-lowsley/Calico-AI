import pygame
from enum import Enum

from .hex.hex import Vector2

class Colour(Enum):
    DarkBlue = 0
    Green = 1
    LightBlue = 2
    Pink = 3
    Purple = 4
    Yellow = 5

class Pattern(Enum):
    CHURCHES = 0
    FERNS = 1
    FLOWERS = 2
    SPOTS = 3
    STRIPES = 4
    VINES = 5

WIDTH, HEIGHT = 960, 720

hand_positions = [Vector2(700, 620), Vector2(800, 620), Vector2(700,670), Vector2(800, 670)]


FPS = 60
HEX_SIZE = Vector2(50, 50)
OFFSET = Vector2(200, 50)