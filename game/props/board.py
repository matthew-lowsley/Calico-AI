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
            return False, 0
        space.tile = tile
        points = self.analyse_placement(space)
        return True, points

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

    def find_existing_spaces(self, spaces):
        existing_spaces = []
        for space in spaces:
            space = self.get_space(space)
            if space != None:
                existing_spaces.append(space)
        return existing_spaces

    def contains_tiles(self, spaces):
        contains_tiles = []
        for space in spaces:
            if space.tile != None:
                contains_tiles.append(space)
        return contains_tiles

    def find_chain(self, space):

        colour = space.tile.colour
        pattern = space.tile.pattern

        visited = set()
        chain = []

        visited.add(space)
        chain.append(space)

        def dfs(start : Space):
            neightbors = self.contains_tiles(self.find_existing_spaces(start.get_all_neighbors()))
            for neightbor in neightbors:
                if neightbor not in visited:
                    visited.add(neightbor)
                    if neightbor.tile.colour != colour:
                        continue
                    chain.append(neightbor)
                    dfs(neightbor)
        
        dfs(space)

        return chain

    def analyse_colour(self, space):
        chain = self.find_chain(space)
        points = 0
        if len(chain) >= 3:
            points = 3
        return points
    
    def analyse_placement(self, space):

        if not space.tile:
            return 0
        
        if not self.get_space(space):
            return 0

        points = 0

        points += self.analyse_colour(space)

        return points