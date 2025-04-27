from collections import namedtuple
import pygame
import os
import torch
from enum import Enum

from .hex.hex import Vector2

pygame.font.init()

FONT = pygame.font.Font('freesansbold.ttf', 24)
FONT_SMALL = pygame.font.Font('freesansbold.ttf', 12)
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

Transition = namedtuple('Transition', ('board_state', 'hand_state', 'action', 'reward', 'next_board_state', 'next_hand_state','done'))

WIDTH, HEIGHT = 1920, 1080 #960, 720

hand_positions = [Vector2(1400, 750), Vector2(1550, 750), Vector2(1400, 900), Vector2(1550, 900)]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Hardware Device: " + str(DEVICE))

FPS = 60
HEX_SIZE = Vector2(75, 75) #Vector2(50, 50)
OFFSET = Vector2(400, 100) #Vector2(200, 50)

MAX_MEMORY = 1000 #1000
BATCH_SIZE = 100
LR = 0.001
TAU = 0.005

REPLAY_RECENT = 4
CLIP_VALUE = 1
VALIDATE_EVERY = 100