import pygame

from .player import Player
from ..props.board import Board, Space
from ..hex.hex import pixel_to_hex, Vector2
from ..constants import HEX_SIZE, OFFSET, Colour, Pattern, hand_positions
from ..props.tile import Objective_Tile, Tile, Shop

OBJECTIVES_SPACES = [Space(0, 4, -4), Space(2, 2, -4), Space(3, 3, -6)]

class Human_Player(Player):

    def __init__(self):
        super().__init__()
        self.selected = None
        self.last_placed = None
        self.objectives_placed = 0
        self.hand_areas = [pygame.Rect(hand_positions[0].x, hand_positions[0].y, HEX_SIZE.x*2, HEX_SIZE.y*2), 
                           pygame.Rect(hand_positions[1].x, hand_positions[1].y, HEX_SIZE.x*2, HEX_SIZE.y*2), 
                           pygame.Rect(hand_positions[2].x, hand_positions[2].y, HEX_SIZE.x*2, HEX_SIZE.y*2),
                           pygame.Rect(hand_positions[3].x, hand_positions[3].y, HEX_SIZE.x*2, HEX_SIZE.y*2)]
        
        
    def reset(self):
        self.points = 0
        self.hand = []
        self.selected = None
        self.last_placed = None

    def select_from_hand(self, mouse_x, mouse_y):
        for i in range(len(self.hand)):
            if self.hand[i] != None:
                if self.hand_areas[i].collidepoint(mouse_x, mouse_y):
                    self.selected = i
    
    def select_from_shop(self, shop, mouse_x, mouse_y):
        for i in range(len(shop.tiles)):
            if shop.shop_areas[i].collidepoint(mouse_x, mouse_y):
                self.take_tile(shop.take_tile(index=i))
                return True
        return False

    def place(self, board, mouse_x, mouse_y):
        space = pixel_to_hex(Vector2(mouse_x, mouse_y), OFFSET, HEX_SIZE)
        valid, points = board.insert_tile(Space(space.x, space.y, space.z), self.hand[self.selected])
        if valid:
            self.points += points
            self.remove_from_hand()
            self.last_placed = space
            #print(self.points)
        else:
            self.selected = None
    
    def place_objective(self, board : Board, mouse_x, mouse_y):
        space_coords = pixel_to_hex(Vector2(mouse_x, mouse_y), OFFSET, HEX_SIZE)
        space = board.get_space(Space(space_coords.x, space_coords.y, space_coords.z))
        for obj_space in OBJECTIVES_SPACES:
            if space != None:
                if space.equal(obj_space):
                    self.place(board, mouse_x, mouse_y)
                    self.objectives_placed += 1
                    self.last_placed = None
                    self.selected = None
                    if self.objectives_placed == 3:
                        return True
                    return False

    def remove_from_hand(self):
        self.hand[self.selected] = None

    def act(self, board : Board, shop : Shop, events):

        for event in events:

            if event.type == pygame.MOUSEBUTTONDOWN:

                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                #space = pixel_to_hex(Vector2(mouse_x, mouse_y), OFFSET, HEX_SIZE)

                #board.insert_tile(Space(space.x, space.y, space.z), Tile(Colour.DarkBlue, Pattern.CHURCHES))

                if self.selected == None:
                    self.select_from_hand(mouse_x, mouse_y)
                
                else:
                    if self.last_placed == None:
                        if type(self.hand[self.selected]) is Objective_Tile:
                            return self.place_objective(board, mouse_x, mouse_y)
                        else:
                            self.place(board, mouse_x, mouse_y)
                    else:
                        if self.select_from_shop(shop, mouse_x, mouse_y):
                            self.selected = None
                            self.last_placed = None
                            return True
                        else:
                            break
        return False
    
                   
