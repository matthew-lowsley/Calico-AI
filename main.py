import pygame
import math

import torch

from game.constants import WIDTH, HEIGHT, DEVICE
from game.game_manager import Game_Manager



PLOT = False
DISABLE_GRAPHICS = False
MAX_GAMES = math.inf

WIN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption('Calico')


def main():

    running = True
    game = Game_Manager(WIN, disable_graphics=DISABLE_GRAPHICS, plot=PLOT)
    n_games = 0
    
    while running:

        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False

        if game.step(events):

            n_games += 1
            
            if PLOT: game.plotter.plot_scores(game.scores, n_games)

            if n_games >= MAX_GAMES:
                exit()

            game.restart_game()
    
    pygame.QUIT()
    quit()

main()
            
    