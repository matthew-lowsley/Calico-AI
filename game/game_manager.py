import os
from game.player.DQL_player.Agent import Agent
from game.player.DQL_player.Memory import Memory
from game.player.DQL_player.Model import CQNet, QNet
from game.player.DQL_player.Model2 import CQNet2
from game.player.DQL_player.Trainer import QTrainer
from game.player.random_player import Random_Player
from game.player.score_plotter import Plotter
from .props.board import Board
from .props.tile import DIR, TILE_TEXTURES_FOLDER, Objective_Tile, Tile, Shop, Bag
from .player.human_player import Human_Player
from .constants import DEVICE, FONT, FONT_SMALL, LR, VALIDATE_EVERY, Colour, Objective, Pattern, BOARD_JSON, WIDTH, HEIGHT
from .props.cat import *

import numpy as np
import copy
import pygame
import json


# objective_tiles = np.array([Objective_Tile(Objective.AAAABB), Objective_Tile(Objective.AAABBB), 
#                             Objective_Tile(Objective.AAABBC), Objective_Tile(Objective.AABBCC),
#                             Objective_Tile(Objective.AABBCD), Objective_Tile(Objective.ABCDEF)])
objective_tiles = np.array([Objective_Tile(Objective.AABBCC), Objective_Tile(Objective.AABBCD), 
                            Objective_Tile(Objective.AAABBC), Objective_Tile(Objective.AAAABB),
                            Objective_Tile(Objective.AAABBB), Objective_Tile(Objective.ABCDEF)])

lv1_cats = [Oliver, Callie, Tibbit, Rumi]
lv2_cats = [Coconut, Tecolote, Cira, Almond]
lv3_cats = [Gwenivere, Leo]

starting_cats = {
    'SPOTS': Oliver(Pattern.SPOTS),
    'CHURCHES': Oliver(Pattern.CHURCHES),
    'FLOWERS': Coconut(Pattern.FLOWERS),
    'STRIPES': Coconut(Pattern.STRIPES),
    'VINES': Gwenivere(Pattern.VINES),
    'FERNS': Gwenivere(Pattern.FERNS)
}

class Game_Manager:

    def __init__(self, win, players, disable_graphics=False, plot=False):
        self.win = win

        self.current_player = 0
        self.players = players
        self.boards = [Board() for _ in range(len(self.players))]
        self.scores = [[] for _ in range(len(self.players))]
        self.n_games = 0
        self.wins = [0 for _ in range(len(self.players))]
        self.highest_score_per_X_games = [0 for _ in range(len(self.players))]
        self.turn = 0
        self.cats = None
        self.cat_areas = [pygame.Rect(50, 640, 50, 50), pygame.Rect(50, 670, 50, 50), pygame.Rect(50, 700, 50, 50)]

        self.bag = Bag()
        self.shop = Shop(self.bag)
        self.final_turn = len(self.players) * 23

        self.points_areas = [pygame.Rect(50, 100, 50, 50), pygame.Rect(50, 150, 50, 50), pygame.Rect(50, 200, 50, 50), pygame.Rect(50, 250, 50, 50)]

        self.plotter = None
        self.average_score_plotter = None

        if plot:
            self.plotter = Plotter(len(self.players), "Games", "Mean Score", "Agents Scores", "Average_Scores")
            self.average_score_plotter = Plotter(len(self.players), f'Games ({VALIDATE_EVERY}s)', 'Scores', f'Average Score Every {VALIDATE_EVERY} Games', f'Average_Score_Every_{VALIDATE_EVERY}_games')  

        self.disable_graphics = disable_graphics

        background_texture = os.path.join(DIR, TILE_TEXTURES_FOLDER, "zwood2_background.jpg")
        self.background_img = pygame.image.load(background_texture)

        self.restart_game()

    def restart_game(self):

        self.player = 0
        self.turn = 0
        self.bag.fill_bag()
        self.shop.tiles = [None, None, None]
        self.shop.stock_shop()

        #CHANGE THIS TO HAVE RANDOM CATS AGAIN
        #self.cats = self.configure_cats()
        self.cats = starting_cats

        boards = self.read_json()
        board_colours = np.array(['blue', 'yellow', 'blue', 'blue'])

        for objective in objective_tiles:
            objective.reset()

        for player in self.players:
            #np.random.shuffle(objective_tiles)
            player.reset()
            player.hand = copy.deepcopy(objective_tiles[:4])

        #np.random.shuffle(board_colours)
        for i, board in enumerate(self.boards):
            board.create_board()
            board.create_perimeter(boards['boards'][board_colours[i]])
            board.cats = self.cats
    
    def configure_cats(self):
        
        cats = [lv1_cats, lv2_cats, lv3_cats]
        patterns = np.array([pattern for pattern in Pattern])
        np.random.shuffle(patterns)
        patterns = patterns.tolist()
        cats_pattern_dict = {}

        for i in range(0,6,2):
            cats[int(i/2)] = np.array(cats[int(i/2)])
            np.random.shuffle(cats[int(i/2)])
            cats_pattern_dict[patterns[i].name] = cats[int(i/2)][0](patterns[i])
            cats_pattern_dict[patterns[i+1].name] = cats[int(i/2)][0](patterns[i+1])

        return cats_pattern_dict

    def give_starting_hand(self):

        for player in self.players:
            player.hand = [self.bag.take_tile(), self.bag.take_tile(), None, None]
            #print(f'Player Starting Hand : {player.hand}')

    def draw(self):
        #self.win.fill((255, 255, 255))
        self.win.blit(self.background_img, (0,0))

        self.boards[self.current_player].draw(self.win)
        self.players[self.current_player].draw_hand(self.win)

        for i in range(len(self.players)):
            points = FONT.render("Player "+str(i+1)+" - "+str(self.players[i].points), True, (0,0,0))
            #player_type = FONT_SMALL.render(str(type(self.players[i])), True, (0,0,0))
            self.win.blit(points, self.points_areas[i])
            #self.win.blit(player_type, pygame.Rect(self.points_areas[i].x, self.points_areas[i].y+24, 50, 50))

        self.shop.draw(self.win)

        cats = list(self.cats.values())
        patterns = list(self.cats.keys())
        for i in range(0, 6, 2):
            cat_descriptions = FONT.render(str(cats[i].name)+": "+str(cats[i].description)+" - "+str(patterns[i])+" / "+str(patterns[i+1]), True, (0,0,0))
            self.win.blit(cat_descriptions, self.cat_areas[int(i/2)])

        pygame.display.update()

    def draw_end_screen(self, winner, highest):
        self.win.fill((255, 255, 255))
                
        message = FONT.render("Player "+str(winner+1)+" Wins! With "+str(highest)+" Points!", True, (0,0,0))
        self.win.blit(message, pygame.Rect(WIDTH/2, HEIGHT/2, 50, 50))

        pygame.display.update()

    def calculate_winner(self):

        winner = 0
        highest_score = 0
        for i in range(len(self.players)-1, -1, -1):
            if self.players[i].points >= highest_score:
                winner = i
                highest_score = self.players[i].points
        
        return winner, highest_score
    
    def calculate_scores(self, winner):

        self.wins[winner] += 1

        for i in range(len(self.players)):
            self.scores[i].append(self.players[i].points)
            #print(f"Player {i+1} Average Score: {sum(self.scores[i])/len(self.scores[i])}")
            #print(f"Player {i+1} Win Rate: {self.wins[i]/self.n_games}")

    def next_turn(self):
        self.turn += 1
        #input("Press Enter for Next Turn!")
        #pygame.time.wait(2000)

    def step(self, events):

        self.current_player = self.turn % len(self.players)

        if not self.disable_graphics: self.draw()

        if self.players[self.current_player].act(self.boards[self.current_player], self.shop, events):
            if not self.disable_graphics: self.draw()

            if self.turn == len(self.players)-1:
                self.give_starting_hand()

            self.next_turn()

        #print(f'Turn {self.turn} : {self.shop.tiles}')

        if self.turn >= self.final_turn:
            self.n_games += 1
            winner, highest = self.calculate_winner()
            self.calculate_scores(winner)
            if self.n_games % 100 == 0:
                last_100_scores = []
                for i in range(len(self.players)):
                    last_100_scores.append(self.scores[i][-100:])
                    average = sum(self.scores[i][-100:])/100
                    if average > self.highest_score_per_X_games[i]:
                        self.highest_score_per_X_games[i] = average
                    print(f"Player {i+1} Highest score per 100 game: {self.highest_score_per_X_games[i]}")
                if self.average_score_plotter: self.average_score_plotter.plot_average_scores(last_100_scores, 100)
            if not self.disable_graphics: 
                self.draw_end_screen(winner, highest)
            #self.restart_game()
            #print("Samples Collected: "+str(len(memory.queue)))
            return True
    
    def read_json(self):

        with open(BOARD_JSON, 'r') as file:
            boards = json.load(file)

        return boards