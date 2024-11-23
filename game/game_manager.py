from .props.board import Board
from .props.tile import Tile
from .player.human_player import Human_Player
from .constants import Colour, Pattern


import pygame

class Game_Manager:

    def __init__(self, win):
        self.win = win

        self.current_player = 0
        self.boards = [Board()]
        self.players = [Human_Player()]
        self.turn = 0

        self.restart_game()

    def restart_game(self):

        self.player = 0
        self.turn = 0

        for board in self.boards:
            board.create_board()
        
        self.give_starting_game()

    def give_starting_game(self):

        for player in self.players:
            player.hand = [Tile(Colour.DarkBlue, Pattern.CHURCHES), Tile(Colour.DarkBlue, Pattern.CHURCHES)]

    def draw(self):
        self.win.fill((255, 255, 255))

        self.boards[self.current_player].draw(self.win)
        self.players[self.current_player].draw_hand(self.win)

        pygame.display.update()

    def next_turn(self):
        self.turn += 1

    def step(self, events):

        self.current_player = self.turn % len(self.players)

        self.draw()

        if self.players[self.current_player].act(self.boards[self.current_player], events):
            self.draw()
            self.next_turn()