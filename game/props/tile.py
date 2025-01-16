from ..constants import Vector2, HEX_SIZE, Colour, Pattern, Objective

import pygame
import math
import copy
import numpy as np
import os

DIR = os.getcwd()
TILE_TEXTURES_FOLDER = os.path.join("game", "tile_textures")

TILE_TEXTURES = {
    (-1,0) : "OT-AAAA-BB.png",(-1,1) : "OT-AAA-BBB.png",(-1,2) : "OT-AAA-BB-C.png",(-1,3) : "OT-AA-BB-CC.png",(-1,4) : "OT-AA-BB-C-D.png",(-1,5) : "OT-NTS.png",
    (0,0) : "DarkBlue-Churches.png",(1,0) : "Green-Churches.png",(2,0) : "LightBlue-Churches.png",(3,0) : "Pink-Churches.png",(4,0) : "Purple-Churches.png",(5,0) : "Yellow-Churches.png",
    (0,1) : "DarkBlue-Ferns.png",(1,1) : "Green-Ferns.png",(2,1) : "LightBlue-Ferns.png",(3,1) : "Pink-Ferns.png",(4,1) : "Purple-Ferns.png",(5,1) : "Yellow-Ferns.png",
    (0,2) : "DarkBlue-Flowers.png",(1,2) : "Green-Flowers.png",(2,2) : "LightBlue-Flowers.png",(3,2) : "Pink-Flowers.png",(4,2) : "Purple-Flowers.png",(5,2) : "Yellow-Flowers.png",
    (0,3) : "DarkBlue-Spots.png",(1,3) : "Green-Spots.png",(2,3) : "LightBlue-Spots.png",(3,3) : "Pink-Spots.png",(4,3) : "Purple-Spots.png",(5,3) : "Yellow-Spots.png",
    (0,4) : "DarkBlue-Stripes.png",(1,4) : "Green-Stripes.png",(2,4) : "LightBlue-Stripes.png",(3,4) : "Pink-Stripes.png",(4,4) : "Purple-Stripes.png",(5,4) : "Yellow-Stripes.png",
    (0,5) : "DarkBlue-Vines.png",(1,5) : "Green-Vines.png",(2,5) : "LightBlue-Vines.png",(3,5) : "Pink-Vines.png",(4,5) : "Purple-Vines.png",(5,5) : "Yellow-Vines.png"
}

class Tile:

    def __init__(self, colour : Colour, pattern : Pattern):
        self.colour : Colour = colour
        self.pattern : Pattern = pattern
        self.colour_used = False
        self.pattern_used = False
    
    def draw(self, win, position, size=HEX_SIZE):
        file_path = os.path.join(DIR, TILE_TEXTURES_FOLDER, TILE_TEXTURES[(self.colour.value, self.pattern.value)])
        texture = pygame.image.load(file_path)
        texture = pygame.transform.scale(texture, (size.x*2, size.y*2))
        texture_region = texture.get_rect()
        texture_region.center = (position.x, position.y)
        win.blit(texture, texture_region)

    def __repr__(self) -> str:
        return str(self.colour.name)+"-"+str(self.pattern.name)

class Colour_Pattern_Tile(Tile):

    def __init__(self, colour : Colour, pattern : Pattern):
        super().__init__(colour, pattern)

class Objective_Tile(Tile):
    
    def __init__(self, objective : Objective):
        super().__init__(colour=Colour.Objective, pattern=objective)
        self.objective = objective
        self.gold_points = 0
        self.blue_points = 0
        self.get_objective_rules()

    def get_objective_rules(self):
        match self.objective:
            case Objective.AAAABB:
                self.objective_rules = {'A':4, 'B':2}
                self.gold_points = 14
                self.blue_points = 7
            case Objective.AAABBB:
                self.objective_rules = {'A':3, 'B':3}
                self.gold_points = 13
                self.blue_points = 8
            case Objective.AAABBC:
                self.objective_rules = {'A':3, 'B':2, 'C':1}
                self.gold_points = 11
                self.blue_points = 7
            case Objective.AABBCC:
                self.objective_rules = {'A':2, 'B':2, 'C':2}
                self.gold_points = 11
                self.blue_points = 7
            case Objective.AABBCD:
                self.objective_rules = {'A':2, 'B':2, 'C':1, 'D':1}
                self.gold_points = 7
                self.blue_points = 5
            case Objective.ABCDEF:
                self.objective_rules ={'A':1, 'B':1, 'C':1, 'D':1, 'E':1, 'F':1}
                self.gold_points = 15
                self.blue_points = 10
            case _:
                self.objective_rules = {'A':2, 'B':2, 'C':2}
                self.gold_points = 0
                self.blue_points = 0

class Bag:

    def __init__(self):
        self.bag = []
        self.total_tiles_remaining = 0
        self.tiles_remaining = {}
        #self.fill_bag()

    def fill_bag(self):
        bag = []
        self.total_tiles_remaining = 0
        for i in range(6):
            colour = Colour(i)
            for pattern in (Pattern):
                self.tiles_remaining[str(colour.name)+"-"+str(pattern.name)] = 0
                for i in range(3):
                    bag.append(Colour_Pattern_Tile(colour, pattern))
                    self.tiles_remaining[str(colour.name)+"-"+str(pattern.name)] += 1
        bag = np.array(bag)
        np.random.shuffle(bag)
        self.bag = bag.tolist()
        self.total_tiles_remaining = len(self.bag)

    def take_tile(self):
        tile = self.bag.pop(0)
        self.tiles_remaining[str(tile.colour.name)+"-"+str(tile.pattern.name)] -= 1
        self.total_tiles_remaining = len(self.bag)
        #print(self.tiles_remaining)
        #print(self.total_tiles_remaining)
        return tile    

class Shop: 
    
    def __init__(self, bag : Bag):
        self.shop_areas = [pygame.Rect(320, 520, HEX_SIZE.x*2, HEX_SIZE.y*2), 
                           pygame.Rect(420, 520, HEX_SIZE.x*2, HEX_SIZE.y*2), 
                           pygame.Rect(520, 520, HEX_SIZE.x*2, HEX_SIZE.y*2)]
        self.tiles = [None, None, None]
        self.bag = bag
        #self.stock_shop()
    
    def stock_shop(self):
        for i in range(len(self.tiles)):
            if self.tiles[i] == None:
                self.tiles[i] = copy.deepcopy(self.bag.take_tile())

    def take_tile(self, index : int):
        tile = self.tiles[index]
        self.tiles[index] = None
        self.stock_shop()
        return tile
    
    def draw(self, win):
        for i in range(len(self.tiles)):
            if self.tiles[i] != None:
                pygame.draw.rect(win, (128,128,128), self.shop_areas[i])
                self.tiles[i].draw(win, Vector2(self.shop_areas[i].centerx, self.shop_areas[i].centery))