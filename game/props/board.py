from ..hex.hex import Hex, hex_to_pixel, Vector2
from ..constants import FONT, HEX_SIZE, OFFSET, Colour, Pattern
from .tile import Colour_Pattern_Tile, Objective_Tile, Tile

import math
import pygame

class Space(Hex):

    def __init__(self, q_, r_, s_):
        super(Space, self).__init__(q_, r_, s_)
        self.tile = None

    def set_tile(self, tile):
        self.tile = tile

    def get_tile(self):
        return self.tile
    
    def __repr__(self) -> str:
        return f'{str(self.q)}, {str(self.r)}, {str(self.s)} : {self.tile}'
    
OBJECTIVES_SPACES = [Space(0, 4, -4), Space(2, 2, -4), Space(3, 3, -6)]

class Board:

    def __init__(self):
        self.board = {}
        self.cats = None
    
    def insert_space(self, space : Space):
        self.board[tuple([space.q, space.r, space.s])] = space

    def insert_tile(self, space : Space, tile : Tile):
        space = self.get_space(space)
        if space == None or space.tile != None or tile == None:
            return False, 0
        if type(tile) is Objective_Tile and self.objective_placement_validation(space) == False:
            return False, 0
        space.tile = tile
        points = self.analyse_placement(space)
        return True, points
    
    def objective_placement_validation(self, placement : Hex):
        for space in OBJECTIVES_SPACES:
            if placement.equal(space):
                return True
        return False

    def get_space(self, space : Space):
        key = tuple([space.q, space.r, space.s])
        if key in self.board.keys():
            return self.board[key]
        return None

    def create_board(self):
        for i in range(7):
            for j in range(7):
                r = j
                q = i - math.floor(r/2)
                s = -q-r
                self.insert_space(Space(q, r, s))

    def print_board(self):
        for key in self.board.keys():
            hex = self.board[key]
            print(hex)

    def print_indexes_of_space(self, spaces):
        board_spaces = list(self.board.values())
        for i in range(len(board_spaces)):
            for j in range(len(spaces)):
                if board_spaces[i].equal(spaces[j]):
                    print(i)
                    break

    def draw_space(self, win, position, position_number):
        points = []
        for i in range(6):
            angle = math.radians(i * 60 - 30)
            x = position.x + HEX_SIZE.x * math.cos(angle)
            y = position.y + HEX_SIZE.y * math.sin(angle)
            points.append([x, y])
        pygame.draw.polygon(win, (255, 0, 0), points, 5)
        points = FONT.render(str(position_number), True, (0,0,0))
        win.blit(points, position)

    def draw(self, win):

        for i, key in enumerate(self.board.keys()):
            space = self.board[key]
            position = hex_to_pixel(space, HEX_SIZE, OFFSET)
            if space.tile != None:
                space.tile.draw(win, position, HEX_SIZE)
            self.draw_space(win, position, i)

    def find_existing_spaces(self, spaces):
        existing_spaces = []
        for space in spaces:
            space = self.get_space(space)
            if space != None:
                existing_spaces.append(space)
        return existing_spaces

    def contains_tiles(self, spaces):
        contains_tiles = []
        for space in spaces:
            if space.tile != None:
                contains_tiles.append(space)
        return contains_tiles

    def find_chain(self, space, mode="colour"):
        
        colour = space.tile.colour
        pattern = space.tile.pattern

        visited = set()
        chain = []

        visited.add(space)
        chain.append(space)
        chain_broken = False

        def dfs(start : Space):
            nonlocal chain_broken
            neighbors = self.contains_tiles(self.find_existing_spaces(start.get_all_neighbors()))
            for neighbor in neighbors:
                if (neighbor not in visited) and (type(neighbor.tile) is not Objective_Tile):
                    visited.add(neighbor)
                    if mode == "colour":
                        if(neighbor.tile.colour != colour):
                            continue
                        if(neighbor.tile.colour_used == True):
                            chain_broken = True
                            continue
                    if mode == "pattern":
                        if(neighbor.tile.pattern != pattern):
                            continue
                        if(neighbor.tile.pattern_used == True):
                            chain_broken = True
                            continue
                    chain.append(neighbor)
                    dfs(neighbor)
        dfs(space)

        if chain_broken:
            self.disable_chain(chain, mode=mode)
            return [space]

        return chain
    
    def disable_chain(self, chain, mode="colour"):

        #print("running!")

        for space in chain:
            if mode == "colour":
                space.tile.colour_used = True
            elif mode == "pattern":
                space.tile.pattern_used = True
    
    def complete_objective(self, objective):
        colour_dict = {}
        pattern_dict = {}
        colours_achieved = False
        patterns_achieved = False
        objective_rules_values = list(objective.tile.objective_rules.values())

        # Counts how many of a specific colour and how many of a specific pattern surround the objective tile.
        # The final result is an array contain the number of a same colour or pattern. E.g AAAABB = [4, 2], ABCDEF = [1, 1, 1, 1, 1, 1]
        neighbors = self.contains_tiles(self.find_existing_spaces(objective.get_all_neighbors()))
        for neighbor in neighbors:
            if neighbor.tile.colour not in colour_dict:
                colour_dict[neighbor.tile.colour] = 0
            if neighbor.tile.pattern not in pattern_dict:
                pattern_dict[neighbor.tile.pattern] = 0
            colour_dict[neighbor.tile.colour] += 1
            pattern_dict[neighbor.tile.pattern] += 1
        colour_dict_values_sorted = list(reversed(sorted(colour_dict.values())))
        pattern_dict_values_sorted = list(reversed(sorted(pattern_dict.values())))

        # This section checks if the colour and pattern arrays are the same as the objective rule on the tile.
        colours_achieved = (colour_dict_values_sorted == objective_rules_values)
        patterns_achieved = (pattern_dict_values_sorted == objective_rules_values)

        # This section checks how many points should be awarded.
        if colours_achieved and patterns_achieved:
            #print("Gold Objective Achieved "+"Colours: "+str(colour_dict_values_sorted)+" vs "+str(objective_rules_values)+" Patterns: "+str(pattern_dict_values_sorted)+" vs "+str(objective_rules_values))
            return objective.tile.gold_points
        elif colours_achieved ^ patterns_achieved:
            #print("Blue Objective Achieved "+"Colours: "+str(colour_dict_values_sorted)+" vs "+str(objective_rules_values)+" Patterns: "+str(pattern_dict_values_sorted)+" vs "+str(objective_rules_values))
            return objective.tile.blue_points
        else:
            #print("No Objective Achieved "+str(colour_dict_values_sorted)+" != "+str(objective_rules_values)+" and "+str(pattern_dict_values_sorted)+" != "+str(objective_rules_values))
            return 0
    
    def analyse_objectives(self, objectives):
        points = 0
        for objective in objectives:
            neighbors = self.contains_tiles(self.find_existing_spaces(objective.get_all_neighbors()))
            if len(neighbors) == 6:
                points += self.complete_objective(objective)
        return points

    def analyse_colour(self, space):
        chain = self.find_chain(space)
        points = 0
        if len(chain) >= 3:
            points = 3
            self.disable_chain(chain) 
        return points
    
    def analyse_pattern(self, space):
        chain = self.find_chain(space, mode="pattern")
        cat = self.cats[space.tile.pattern.name]
        points = cat.analyse_pattern(chain, self)
        if points > 0:
            self.disable_chain(chain, mode="pattern") 
            #print(str(cat)+" Got! "+ str(points) +" points!")
        return points
    
    def analyse_placement(self, space):

        if not space.tile:
            return 0
        
        if not self.get_space(space):
            return 0
        
        if type(space.tile) is Objective_Tile:
            return 0

        points = 0

        points += self.analyse_colour(space)
        points += self.analyse_pattern(space)

        neighbors = self.contains_tiles(self.find_existing_spaces(space.get_all_neighbors()))
        objectives = []
        for neighbor in neighbors:
            if type(neighbor.tile) is Objective_Tile:
                objectives.append(neighbor)
        if len(objectives) > 0:
            points += self.analyse_objectives(objectives)

        return points

    def create_perimeter(self, tiles):

            for tile in tiles:
                colour = Colour[tile['colour']]
                pattern = Pattern[tile['pattern']]
                self.board[tuple(tile['coordinates'])].tile = Colour_Pattern_Tile(colour, pattern)
        
        
