from ..hex.hex import Hex, hex_to_pixel, Vector2
from ..constants import HEX_SIZE, OFFSET
from .tile import Tile

import math
import pygame

class Space(Hex):

    def __init__(self, q_, r_, s_):
        super(Space, self).__init__(q_, r_, s_)
        self.tile = None

    def set_tile(self, tile):
        self.tile = tile

    def get_tile(self):
        return self.tile

class Board:

    def __init__(self):
        self.board = {}
    
    def insert_space(self, space : Space):
        self.board[tuple([space.q, space.r, space.s])] = space

    def insert_tile(self, space : Space, tile : Tile):
        space = self.get_space(space)
        print(space)
        if space == None or space.tile != None:
            return False
        space.tile = tile
        return True

    def get_space(self, space : Space):
        key = tuple([space.q, space.r, space.s])
        if key in self.board.keys():
            return self.board[key]
        return None

    def create_board(self):
        for i in range(7):
            for j in range(7):
                r = j
                q = i - math.floor(r/2)
                s = -q-r
                self.insert_space(Space(q, r, s))

    def print_board(self):
        for key in self.board.keys():
            hex = self.board[key]
            print(hex)

    def draw_space(self, win, position):
        points = []
        for i in range(6):
            angle = math.radians(i * 60 - 30)
            x = position.x + HEX_SIZE.x * math.cos(angle)
            y = position.y + HEX_SIZE.y * math.sin(angle)
            points.append([x, y])
        pygame.draw.polygon(win, (255, 0, 0), points, 5)

    def draw(self, win):
        for key in self.board.keys():
            space = self.board[key]
            position = hex_to_pixel(space, HEX_SIZE, OFFSET)
            self.draw_space(win, position)
            if space.tile != None:
                space.tile.draw(win, position, HEX_SIZE)