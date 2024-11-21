import pygame

from .player import Player
from ..props.board import Board, Space
from ..hex.hex import pixel_to_hex, Vector2
from ..constants import HEX_SIZE, OFFSET, Colour, Pattern
from ..props.tile import Tile

class Human_Player(Player):

    def __init__(self):
        super().__init__()
        
    def reset(self):
        self.points = 0
        self.hand = []

    def act(self, board : Board, events):

        for event in events:

            if event.type == pygame.MOUSEBUTTONDOWN:

                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                space = pixel_to_hex(Vector2(mouse_x, mouse_y), OFFSET, HEX_SIZE)


                board.insert_tile(Space(space.x, space.y, space.z), Tile(Colour.DarkBlue, Pattern.CHURCHES))
