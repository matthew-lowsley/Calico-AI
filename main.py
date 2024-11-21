import pygame
import math

from game.constants import WIDTH, HEIGHT
from game.game_manager import Game_Manager


FPS = 60
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Calico')

def main():

    running = True
    clock = pygame.time.Clock()
    game = Game_Manager(WIN)
    
    while running:

        clock.tick(FPS)

        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False

        game.step(events)
    
    pygame.QUIT()
    quit()

main()
            
    