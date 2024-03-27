import numpy as np
import matplotlib.pyplot as plt
from proj1 import GridAttrib, Grid, Alien
from numpy import random as nprd
import random as rd
from PIL import Image

D=35

class Grid2:
    def __init__(self, D, debug=1):
        self._grid = Grid(D)
        self.grid = self._grid.grid
        self.crew_pos = rd.choice(self._grid.get_open_indices())
    def distance(self, pos1, pos2):
        d = abs(pos1[1] - pos2[1])
        d += abs(pos1[0] - pos2[0])
        return d
    def distance_to_crew(self, pos):
        d = self.distance(self.crew_pos, pos)
        return d

class bot1:
    def __init__(self, grid, alpha = 0.1):
        self.grid = grid
        self.pos = None
        while self.pos == self.grid.crew_pos or self.pos is None:
            self.pos = rd.choice(grid.get_open_indices())
        self.alpha = alpha

    def crew_sensor(self):
        c = rd.random()
        return c <= np.exp(-self.alpha
                           * (self.grid.distance_to_crew(self.pos) - 1))

    def update_belief(self, beep):
        generative_fn = lambda x: np.exp(-self.alpha*(x - 1)) if beep else (1 - np.exp(-self.alpha*(x-1)))
        open_cells = self.grid.get_unoccupied_open_indices()
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].crew_belief *= generative_fn(self.grid.distance_to_crew(ci))

    def move(self):
        neighbors = self.grid.get_open_neighbors(self.pos)
        neighbors = [n for n in neighbors if self.grid.crew_pos == n]
        self.pos = rd.choice(neighbors)
        #possible_dir = self.grid.get_open_neighbors(self.pos)
        #possible_dir.sort(key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
        #possible_dir.so
print("hello")
