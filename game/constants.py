import pygame
from enum import Enum

from .hex.hex import Vector2

pygame.font.init()

FONT = pygame.font.Font('freesansbold.ttf', 24)

class Colour(Enum):
    Objective = -1
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

class Objective(Enum):
    AAAABB = 0
    AAABBB = 1
    AAABBC = 2
    AABBCC = 3
    AABBCD = 4
    ABCDEF = 5


WIDTH, HEIGHT = 960, 720

hand_positions = [Vector2(700, 520), Vector2(800, 520), Vector2(700, 620), Vector2(800, 620)]


FPS = 60
HEX_SIZE = Vector2(50, 50)
OFFSET = Vector2(200, 50)