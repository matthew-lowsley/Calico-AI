from ..constants import Pattern
from ..hex.hex import Hex, Direction

def compare_configurations(new_config : list[int], true_config : list[int]):
    new_config_sorted = sorted(new_config)
    true_config_sorted = sorted(true_config)
    i = 0
    for num in true_config_sorted:
        while i < len(new_config_sorted):
            if new_config_sorted[i] < num:
                i += 1
            else:
                break
        if i >= len(new_config_sorted):
            return False        
        i += 1        
    return True

def find_new_configuration(chain, board, pattern):
    chain_config = []
    for tile in chain:
        count = 0
        tile_neighbors = tile.get_all_neighbors()
        for neighbor in tile_neighbors:
            neighbor = board.get_space(neighbor)
            if neighbor == None:
                continue
            if neighbor.tile == None:
                continue
            if neighbor.tile.pattern == pattern:
                count += 1
        chain_config.append(count)
    return chain_config

def find_straight_line(chain, board, pattern):
    for i in range(len(chain)):
        connections = []
        tile_neighbors = chain[i].get_all_neighbors()
        for j in range(len(tile_neighbors)):
            neighbor = board.get_space(tile_neighbors[j])
            if neighbor == None:
                continue
            if neighbor.tile == None:
                continue
            if neighbor.tile.pattern == pattern:
                connections.append(j)
        if len(connections) == 1:
            break
        if i >= len(chain):
            return 0
    if len(connections) < 1:
        return 0
    direction = list(Direction)[connections[0]]
    opp_direction = list(Direction)[(connections[0]+3)%6]
    #print("Direction: "+str(direction.name))
    #print("Opp Direction: "+str(opp_direction.name))
    straight_chain_length = 0
    for i in range(len(chain)):
        direction_space = board.get_space(chain[i].get_neighbor(direction))
        if direction_space != None:
            if direction_space.tile != None:
                if direction_space.tile.pattern == pattern:
                    straight_chain_length += 1
                    continue

        opp_direction_space = board.get_space(chain[i].get_neighbor(opp_direction))
        if opp_direction_space != None:
            if opp_direction_space.tile != None:
                if opp_direction_space.tile.pattern == pattern:
                    straight_chain_length += 1
    
    return straight_chain_length

def find_straight_line2(chain, board, pattern):
    q_count = {-3:0, -2:0, -1:0, 0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    r_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    s_count = {-9:0, -8:0, -7:0, -6:0, -5:0, -4:0, -3:0, -2:0, -1:0, 0:0}
    for i in range(len(chain)):
        q, r, s = chain[i].get_coors()
        q_count[q] += 1
        r_count[r] += 1
        s_count[s] += 1
    max_counts = [max(list(q_count.values())), max(list(r_count.values())), max(list(s_count.values()))]
    return max(max_counts)

class Cat:

    def __init__(self, pattern : Pattern = None):
        self.name = None
        self.points = 0
        self.pattern = pattern
        self.description = None
    
    def analyse_pattern(self, chain, board):
        #print("Oops No Cat!")
        return 0
    
    def __repr__(self) -> str:
        return self.name
    
class Oliver(Cat): # 3+ in any config

    def __init__(self, pattern):
        super().__init__(pattern)
        self.name = "Oliver"
        self.points = 3
        self.description = "3+"

    def analyse_pattern(self, chain, board):
        if len(chain) >= 3:
            return self.points
        return 0

class Callie(Cat): # 3 in a triangle config

    def __init__(self, pattern):
        super().__init__(pattern)
        self.name = "Callie"
        self.points = 3
        self.pattern_config = [2, 2, 2]
        self.description = "3 in Triangle"

    def analyse_pattern(self, chain, board):
        chain_config = find_new_configuration(chain, board, self.pattern)
        if compare_configurations(chain_config, self.pattern_config):
            return self.points
        return 0

class Tibbit(Cat): # 4+ any config

    def __init__(self, pattern):
        super().__init__(pattern)
        self.name = "Tibbit"
        self.points = 5
        self.description = "4+"

    def analyse_pattern(self, chain, board):
        if len(chain) >= 4:
            return self.points
        return 0

class Rumi(Cat): # 3 in a straight line

    def __init__(self, pattern):
        super().__init__(pattern)
        self.name = "Rumi"
        self.points = 5
        self.description = "3 in a Line"

    def analyse_pattern(self, chain, board):
        if find_straight_line2(chain, board, self.pattern) >= 3:
            return self.points
        return 0
        
class Coconut(Cat): # 5+ in any config

    def __init__(self, pattern):
        super().__init__(pattern)
        self.name = "Coconut"
        self.points = 7
        self.description = "5+"

    def analyse_pattern(self, chain, board):
        if len(chain) >= 5:
            return self.points
        return 0

class Tecolote(Cat): # 4 in a striaght line

    def __init__(self, pattern):
        super().__init__(pattern)
        self.name = "Tecolote"
        self.points = 7
        self.description = "4 in a Line"
    
    def analyse_pattern(self, chain, board):
        if find_straight_line2(chain, board, self.pattern) >= 4:
            return self.points
        return 0

class Cira(Cat): # 6+ in any config

    def __init__(self, pattern):
        super().__init__(pattern)
        self.name = "Cira"
        self.points = 9
        self.description = "6+"
    
    def analyse_pattern(self, chain, board):
        if len(chain) >= 6:
            return self.points
        return 0

class Almond(Cat): # 5 in a trapezium config

    def __init__(self, pattern):
        super().__init__(pattern)
        self.name = "Almond"
        self.points = 9
        self.pattern_config = [2, 2, 3, 3, 4]
        self.description = "5 in a Trapezium"

    def analyse_pattern(self, chain, board):
        chain_config = find_new_configuration(chain, board, self.pattern)
        if compare_configurations(chain_config, self.pattern_config):
            return self.points
        return 0

class Gwenivere(Cat): # 7+ in any config

    def __init__(self, pattern):
        super().__init__(pattern)
        self.name = "Gwenivere"
        self.points = 11
        self.description = "7+"

    def analyse_pattern(self, chain, board):
        if len(chain) >= 7:
            return self.points
        return 0

class Leo(Cat): # 5 in a striaght line

    def __init__(self, pattern):
        super().__init__(pattern)
        self.name = "Leo"
        self.points = 11
        self.description = "5 in a Line"

    def analyse_pattern(self, chain, board):
        if find_straight_line2(chain, board, self.pattern) >= 5:
            return self.points
        return 0

