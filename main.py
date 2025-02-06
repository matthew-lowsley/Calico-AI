import pygame
import math

import torch

from game.constants import WIDTH, HEIGHT, DEVICE
from game.game_manager import Game_Manager


FPS = 60
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Calico')

def main():

    running = True
    clock = pygame.time.Clock()
    game = Game_Manager(WIN)
    n_games = 0
    
    while running:

        clock.tick(FPS)

        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False

        if game.step(events):

            n_games += 1

            game.plotter.plot(game.scores, n_games)

            game.restart_game()
    
    pygame.QUIT()
    quit()

main()
            
    