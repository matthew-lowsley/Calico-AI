from game.player.random_player import Random_Player
from .props.board import Board
from .props.tile import Objective_Tile, Tile, Shop, Bag
from .player.human_player import Human_Player
from .constants import FONT, Colour, Objective, Pattern

import numpy as np
import copy
import pygame


objective_tiles = np.array([Objective_Tile(Objective.AAAABB), Objective_Tile(Objective.AAABBB), 
                            Objective_Tile(Objective.AAABBC), Objective_Tile(Objective.AABBCC),
                            Objective_Tile(Objective.AABBCD), Objective_Tile(Objective.ABCDEF)])

class Game_Manager:

    def __init__(self, win):
        self.win = win

        self.current_player = 0
        self.boards = [Board(), Board()]
        self.players = [Human_Player(), Random_Player()]
        self.turn = 0

        self.bag = Bag()
        self.shop = Shop(self.bag)

        self.points_areas = [pygame.Rect(50, 100, 50, 50), pygame.Rect(50, 150, 50, 50), pygame.Rect(50, 200, 50, 50), pygame.Rect(50, 250, 50, 50)]

        self.restart_game()

    def restart_game(self):

        self.player = 0
        self.turn = 0
        self.bag.fill_bag()
        self.shop.stock_shop()

        for player in self.players:
            np.random.shuffle(objective_tiles)
            player.reset()
            player.hand = copy.deepcopy(objective_tiles[:4])

        for board in self.boards:
            board.create_board()
        

    def give_starting_hand(self):

        for player in self.players:
            player.hand = [self.bag.take_tile(), self.bag.take_tile()]

    def draw(self):
        self.win.fill((255, 255, 255))

        self.boards[self.current_player].draw(self.win)
        self.players[self.current_player].draw_hand(self.win)

        for i in range(len(self.players)):
            points = FONT.render("Player "+str(i+1)+" - "+str(self.players[i].points), True, (0,0,0))
            self.win.blit(points, self.points_areas[i])

        self.shop.draw(self.win)

        pygame.display.update()

    def next_turn(self):
        self.turn += 1
        pygame.time.wait(1500)

    def step(self, events):

        self.current_player = self.turn % len(self.players)

        self.draw()

        if self.players[self.current_player].act(self.boards[self.current_player], self.shop, events):
            self.draw()

            if self.turn == len(self.players)-1:
                self.give_starting_hand()

            self.next_turn()