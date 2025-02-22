import pygame
import math

import torch

from game.constants import WIDTH, HEIGHT, DEVICE
from game.game_manager import Game_Manager


WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Calico')

def main():

    running = True
    game = Game_Manager(WIN)
    n_games = 0
    
    while running:

        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False

        if game.step(events):

            n_games += 1

            game.plotter.plot_scores(game.scores, n_games)

            game.restart_game()
    
    pygame.QUIT()
    quit()

main()
            
    