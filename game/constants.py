from collections import namedtuple
import pygame
import os
import torch
from enum import Enum

from .hex.hex import Vector2

pygame.font.init()

FONT = pygame.font.Font('freesansbold.ttf', 24)
BOARD_JSON = os.path.join("game", "boards", "boards.json")

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

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

WIDTH, HEIGHT = 960, 720

hand_positions = [Vector2(700, 520), Vector2(800, 520), Vector2(700, 620), Vector2(800, 620)]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Hard Device: " + str(DEVICE))

FPS = 60
HEX_SIZE = Vector2(50, 50)
OFFSET = Vector2(200, 50)

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001