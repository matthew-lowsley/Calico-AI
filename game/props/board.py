from ..hex.hex import Hex, hex_to_pixel, Vector2

import math
import pygame

class Board:

    def __init__(self):
        self.board = {}
    
    def insert_hex(self, hex : Hex):
        self.board[tuple([hex.q, hex.r, hex.s])] = hex

    def create_board(self):
        for i in range(7):
            for j in range(7):
                r = j
                q = i - math.floor(r/2)
                s = -q-r
                self.insert_hex(Hex(q, r, s))

    def print_board(self):
        for key in self.board.keys():
            hex = self.board[key]
            print(hex)

    def draw_hex(self, win, position, size):
        points = []
        for i in range(6):
            angle = math.radians(i * 60 - 30)
            x = position.x + size.x * math.cos(angle)
            y = position.y + size.y * math.sin(angle)
            points.append([x, y])
        pygame.draw.polygon(win, (255, 0, 0), points, 5)


    def draw(self, size, offset,  win):
        for key in self.board.keys():
            hex = self.board[key]
            position = hex_to_pixel(hex, size, offset)
            self.draw_hex(win, position, size)