import pygame
import math

from hex.hex import Hex, hex_to_pixel, Vector2

WIN = pygame.display.set_mode((960, 720))
pygame.display.set_caption('Calico')

FPS = 60
HEX_SIZE = Vector2(50, 50)
OFFSET = Vector2(960/2, 640/2)

hex = Hex(0, 0, 0)
hex.get_all_neighbors()

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

    draw_hex(hex)

    pygame.display.update()

def main():

    running = True
    clock = pygame.time.Clock()
    
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
            
    