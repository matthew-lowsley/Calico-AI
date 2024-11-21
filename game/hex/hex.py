import math
from enum import Enum
from collections import namedtuple

class Direction(Enum):
    E = [1, 0, -1] 
    NE = [1, -1, 0] 
    NW = [0, -1, 1] 
    W = [-1, 0, 1] 
    SW = [-1, 1, 0] 
    SE = [0, 1, -1]

Vector2 = namedtuple('Vector2', 'x, y')
Vector3 = namedtuple('Vector3', 'x, y, z')

class Hex:

    def __init__(self, q_, r_, s_):
        self.q = q_
        self.r = r_
        self.s = s_
        assert(self.q + self.r + self.s == 0)

    def equal(self, hex) -> bool:
        return self.q == hex.q and self.r == hex.r and self.s == hex.s
    
    def hex_add(self, hex):
        return Hex(self.q + hex.q, self.r + hex.r, self.s + hex.s)
    
    def hex_sub(self, hex):
        return Hex(self.q - hex.q, self.r - hex.r, self.s - hex.s)
    
    def hex_mult(self, num:int):
        return Hex(self.q * num, self.r * num, self.s * num)

    def get_neighbor(self, direction:Direction):
        return self.hex_add(Hex(direction.value[0],direction.value[1],direction.value[2]))
    
    def get_all_neighbors(self):
        neighbors = []
        for direction in Direction:
            neighbors.append(self.get_neighbor(direction))
        print(neighbors)
        return neighbors

    def __repr__(self) -> str:
        return f'{str(self.q)}, {str(self.r)}, {str(self.s)}'

def hex_to_pixel(hex : Hex, size : Vector2, offset : Vector2):
        x = (size.x * (math.sqrt(3) * hex.q + math.sqrt(3) / 2 * hex.r)) + offset.x
        y = (size.y * (3/2 * hex.r)) + offset.y
        return Vector2(x, y)
    
def cube_round(frac : Vector3):
    q = round(frac.x)
    r = round(frac.y)
    s = round(frac.z)

    q_diff = abs(q - frac.x)
    r_diff = abs(r - frac.y)
    s_diff = abs(s - frac.z)

    if q_diff > r_diff and q_diff > s_diff:
        q = -r-s
    elif r_diff > s_diff:
        r = -q-s
    else:
        s = -q-r
    return Vector3(q, r, s)

def pixel_to_hex(position : Vector2, offset : Vector2, size : Vector2):
    q = (math.sqrt(3)/3 * (position.x - offset.x) - 1.0/3 * (position.y - offset.y)) / size.x
    r = (2.0/3 * (position.y - offset.y) ) / size.y
    return cube_round(Vector3(q, r, -q-r))