import pygame
import math

from game.hex.hex import Hex, hex_to_pixel, Vector2
from game.props.board import Board

WIN = pygame.display.set_mode((960, 720))
pygame.display.set_caption('Calico')

FPS = 60
HEX_SIZE = Vector2(50, 50)
OFFSET = Vector2(200, 50)

#hex = Hex(0, 0, 0)
#hex.get_all_neighbors()

board = Board()

def draw_hex(hex : Hex):
        position = hex_to_pixel(hex, HEX_SIZE, OFFSET)
        points = []
        for i in range(6):
            angle = math.radians(i * 60 - 30)
            x = position.x + HEX_SIZE.x * math.cos(angle)
            y = position.y + HEX_SIZE.y * math.sin(angle)
            points.append([x, y])
        pygame.draw.polygon(WIN, (255, 0, 0), points, 5)

def draw():
    WIN.fill((255, 255, 255))

    board.draw(HEX_SIZE, OFFSET, WIN)

    pygame.display.update()

def main():

    running = True
    clock = pygame.time.Clock()
    board.create_board()
    board.print_board()
    
    while running:

        clock.tick(FPS)

        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False

        draw()
    
    pygame.QUIT()
    quit()

main()
            
    