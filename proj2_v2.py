import numpy as np
import matplotlib.pyplot as plt
from proj1 import GridAttrib, Grid, Alien, PathTreeNode
from numpy import random as nprd
import random as rd
from PIL import Image
from collections import deque
from math import floor
import copy
import os

D=35
COMPUTE_LIMIT = 5000

'''
    # TODO
    after creating the divisions grid, we're going to need a few more things:
        - fix the moving in and out of the destination cell
        - maybe reduce the impact of the distance normalization?
        - path planning, that muzzammil should take care of
'''

class Alien:
    # This alien_id is used to keep track of every alien
    alien_id = 0
    def __init__(self, grid, indi=None):
        self.grid = grid
        indices = self.grid.get_unoccupied_open_indices()
        ind = rd.choice(indices)
        self.ind = ind if indi == None else indi
        self.alien_id = Alien.alien_id
        self.grid.place_alien(ind, Alien.alien_id)
        Alien.alien_id += 1

    def move(self):
        # Get all possible locations for the alien
        neighbors = self.grid.get_open_neighbors(self.ind)
        # Filter out the ones that are occupied by other aliens
        neighbors_without_aliens = [neighbor for neighbor in neighbors if self.grid.grid[neighbor[1]][neighbor[0]].alien_id == -1]
        # Randomly choose any of the locations
        if len(neighbors_without_aliens) > 0:
            rand_ind = np.random.randint(0, len( neighbors_without_aliens ))
            self.grid.remove_alien(self.ind)
            self.ind = neighbors_without_aliens[rand_ind]
            self.grid.place_alien(self.ind, self.alien_id)

class Grid2:
    def __init__(self, D=35, debug=1):
        self._grid = Grid(D, debug=debug - 1>0)
        self.D = D
        self.grid = self._grid.grid
        self.crew_pos = rd.choice(self._grid.get_open_indices())
        
        # let's divide the 35x35 grid into smaller 7x7 grids
        # we're going to keep a 2-d array for keeping track of the total probability of these cells
        self.divisions = [[1.0 for i in range(5)] for i in range(5)]

    def distance(self, pos1, pos2):
        d = abs(pos1[1] - pos2[1])
        d += abs(pos1[0] - pos2[0])
        return d
    
    def distance_to_crew(self, pos):
        d = self.distance(self.crew_pos, pos)
        return d

class bot1:
    def __init__(self, grid, alpha = 0.1, k=5, debug=1):
        self.grid = grid
        self.pos = None
        while self.pos == self.grid.crew_pos or self.pos is None:
            self.pos = rd.choice(self.grid._grid.get_open_indices())
        self.alpha = alpha
        self.debug=debug
        self.tick=0
        self.k=k

    def within_alien_sensor(self, pos):
        return abs(pos[0] - self.pos[0]) <= self.k and abs(pos[1] - self.pos[1]) <= self.k

    def alien_sensor_edge(self, pos, offset):
        return ( abs(pos[0] - self.pos[0]) == self.k + offset and abs(pos[1] - self.pos[1]) <= self.k ) or (abs(pos[0] - self.pos[0]) <= self.k and abs(pos[1] - self.pos[1]) == self.k + offset)

    def in_danger(self, offset=1):
        for i in range(-offset, offset):
            for j in range(-offset, offset):
                # Skip the current bot location
                if i == 0 and j == 0:
                    continue
                if self.grid.grid[j][i].open and self.grid.grid[j][i].alien_belief > 0.1/self.grid.D:                    
                    return True
        return False

    def crew_sensor(self):
        c = rd.random()
        return c <= np.exp(-self.alpha
                           * (self.grid.distance_to_crew(self.pos) - 1))
    
    def alien_sensor(self):
        found_alien = 0
        for j in range(-self.k, self.k + 1):
            if found_alien == 1:
                break
            for i in range(-self.k, self.k + 1):
                pos = [ self.pos[0] + i, self.pos[1] + j ]
                if pos[0] > self.grid.D - 1:
                    pos[0] = self.grid.D - 1
                elif pos[0] < 0:
                    pos[0] = 0
                if pos[1] > self.grid.D - 1:
                    pos[1] = self.grid.D - 1
                elif pos[1] < 0:
                    pos[1] = 0
                if self.grid.grid[pos[1]][pos[0]].alien_id != -1:
                    found_alien = 1
                    break
        return found_alien == 1
    
    def diffuse_alien_prob(self, alien_found):
        choose_fun = None
        if alien_found:
            choose_fun = lambda x: self.within_alien_sensor(x)
        else:
            choose_fun = lambda x: not self.within_alien_sensor(x)
        open_cells = self.grid._grid.get_open_indices()
        # Cells inside the alien sensor and just outside
        # The probability will diffuse among these
        filtered_open_cells = [oc for oc in open_cells if ( choose_fun(oc) or self.alien_sensor_edge(oc, 1 if alien_found else 0) )]
        alien_belief = np.zeros((self.grid.D, self.grid.D))

        # Diffuse through the edge cells
        for ci in filtered_open_cells:
            neighbors = self.grid._grid.get_neighbors(ci)
            neighbors = [n for n in neighbors if self.grid.grid[n[1]][n[0]].open and choose_fun(n) ]
            # Diffuse the probability at the current square into the
            # neighbors that the alien can move to
            for n in neighbors:
                alien_belief[n[1]][n[0]] += self.grid.grid[ci[1]][ci[0]].alien_belief/len(neighbors)
        # Normalizs
        total_belief = np.sum(alien_belief)
        for ci in open_cells:
            alien_belief[ci[1]][ci[0]] /= total_belief
        # Update the original probabilities
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief = alien_belief[ci[1]][ci[0]]

    def restrict_alien_prob(self, alien_found):
        choose_fun = None
        if alien_found:
            choose_fun = lambda x: self.within_alien_sensor(x)
        else:
            choose_fun = lambda x: not self.within_alien_sensor(x)

        open_cells = self.grid._grid.get_open_indices()
        filtered_open_cells = [oc for oc in open_cells if not choose_fun(oc)]
        # print(f"Cells to set to 0: {len(filtered_open_cells)}")
        for ci in filtered_open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief = 0.0
        # Normalize
        total_belief = 0
        for ci in open_cells:
            total_belief += self.grid.grid[ci[1]][ci[0]].alien_belief
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief /= total_belief

    def update_belief(self, beep, alien_found):
        # Crew Belief
        generative_fn = lambda x: np.exp(-self.alpha*(x - 1)) if beep else (1 - np.exp(-self.alpha*(x-1)))
        open_cells = self.grid._grid.get_open_indices()
        for ci in open_cells:
            if ci == self.pos:
                continue
            gen_res = generative_fn(self.grid.distance(ci, self.pos))
            if gen_res == 0:
                pass
                #print("DANGER!!!")
                #print(f"Distance: {self.grid.distance(ci, self.pos)}, Beep: {beep}")
            self.grid.grid[ci[1]][ci[0]].crew_belief *= gen_res
        # Normalize
        flat_beliefs = [self.grid.grid[ci[1]][ci[0]].crew_belief for ci in open_cells]
        belief_sum = sum(flat_beliefs)
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].crew_belief /= belief_sum

        # Alien Belief
        # The alien belief consists of two steps-
        # If there is no detection, we diffuse everything outside of the detection square
        # If there is a detection, we set everything outside the square to 0 and leave
        # everything inside the square as is
        alien_belief = np.zeros(( self.grid.D, self.grid.D ))
        self.diffuse_alien_prob(alien_found)
        self.restrict_alien_prob(alien_found)
        # print("Alien detected" if alien_found else "Alien Not Detected")
        #alien

    def plan_path(self, dest):
        if self.debug:
            print("Planning Path...")  # If path is empty we plan one
        self.path = deque([])
        self.grid._grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.pos
        path_deque = deque([path_tree])
        destination = None
        visited = set()
        compute_counter = 0
        while not captain_found:
            if len(path_deque) == 0 or compute_counter >= COMPUTE_LIMIT:
                self.grid._grid.remove_all_traversal()
                return
            compute_counter += 1
            node = path_deque.popleft()
            ind = node.data
            if ind in visited:
                continue
            visited.add(ind)
            self.grid._grid.set_traversed(ind)
            if ind == dest:
                destination = node
                break
            neighbors_ind = self.grid._grid.get_untraversed_open_neighbors(ind)
            for neighbor_ind in neighbors_ind:
                # Add all possible paths that start with no aliens nearby
                if (self.grid.grid[neighbor_ind[1]][neighbor_ind[0]].alien_belief == 0) or (compute_counter > 2):
                    new_node = PathTreeNode()
                    new_node.data = neighbor_ind
                    new_node.parent = node
                    node.children.append(new_node)
            path_deque.extend(node.children)
        self.grid._grid.remove_all_traversal()
        if self.debug:
            print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid._grid.set_traversed(ind)
        if self.debug:
            print("Planned Path")
    
    def move(self):
        self.update_belief(self.crew_sensor(), self.alien_sensor())

        neighbors = self.grid._grid.get_open_neighbors(self.pos)
        neighbors = [n for n in neighbors if not self.grid.crew_pos == n]
        neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
        open_cells = self.grid._grid.get_unoccupied_open_indices()

        self.grid._grid.remove_bot(self.pos)
        dest_cell = max(open_cells, key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
        self.plan_path(dest_cell)
        if len(self.path) != 0:
            self.pos = self.path[0]
        # If no path is found, we automatically shift to evasion, and the evasion strategy is basic
        # Go to the cell with the lowest alien probability
        else:
            if self.debug:
                print("Evasion!!")
            neighbors = self.grid._grid.get_neighbors(self.pos)
            open_neighbors = [n for n in neighbors if self.grid.grid[n[1]][n[0]].open]
            open_neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].alien_belief)
            self.pos = open_neighbors[0]
        
        self.grid._grid.place_bot(self.pos)
        
        if self.pos != self.grid.crew_pos:
            self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0

        if not self.grid._grid.has_alien(self.pos):
            self.grid.grid[self.pos[1]][self.pos[0]].alien_belief = 0.0
            self.tick += 1
            return False
        else:
            # bot ded
            return True

class bot2:
    def __init__(self, grid, alpha = 0.1, k=2, debug=1, bot_pos=None, crew=None):
        self.grid = grid
        self.pos = None
        while self.pos == self.grid.crew_pos or self.pos is None:
            self.pos = rd.choice(self.grid._grid.get_open_indices()) if \
                bot_pos == None else bot_pos
        self.grid.crew_pos = self.grid.crew_pos if crew == None else crew
        self.alpha = alpha
        self.debug=debug
        self.tick=0
        self.k=k
        self.divisions = self.grid.divisions
        self.grid_pos = (floor(self.pos[0] / 7), floor(self.pos[1] / 7))

    def within_alien_sensor(self, pos):
        return abs(pos[0] - self.pos[0]) <= self.k and abs(pos[1] - self.pos[1]) <= self.k

    def alien_sensor_edge(self, pos, offset):
        return ( abs(pos[0] - self.pos[0]) == self.k + offset and abs(pos[1] - self.pos[1]) <= self.k ) or (abs(pos[0] - self.pos[0]) <= self.k and abs(pos[1] - self.pos[1]) == self.k + offset)

    def in_danger(self, offset=1):
        for i in range(-offset, offset):
            for j in range(-offset, offset):
                # Skip the current bot location
                if i == 0 and j == 0:
                    continue
                if self.grid.grid[j][i].open and self.grid.grid[j][i].alien_belief > 0.1/self.grid.D:                    
                    return True
        return False

    def crew_sensor(self):
        c = rd.random()
        return c <= np.exp(-self.alpha
                           * (self.grid.distance_to_crew(self.pos) - 1))
    def alien_sensor(self):
        found_alien = 0
        for j in range(-self.k, self.k + 1):
            if found_alien == 1:
                break
            for i in range(-self.k, self.k + 1):
                pos = [ self.pos[0] + i, self.pos[1] + j ]
                if pos[0] > self.grid.D - 1:
                    pos[0] = self.grid.D - 1
                elif pos[0] < 0:
                    pos[0] = 0
                if pos[1] > self.grid.D - 1:
                    pos[1] = self.grid.D - 1
                elif pos[1] < 0:
                    pos[1] = 0
                if self.grid.grid[pos[1]][pos[0]].alien_id != -1:
                    found_alien = 1
                    break
        return found_alien == 1
    
    def find_upper_and_lower(self, grid_x, grid_y):
        upper_x, lower_x = (grid_x * 7) + 6, grid_x * 7
        upper_y, lower_y = (grid_y * 7) + 6, grid_y * 7

        return ((upper_x, lower_x), (upper_y, lower_y))
    
    def sub_max_belief(self):
        '''
            returns the next sub-grid to go to
        '''
        max_x, max_y = self.grid_pos[0], self.grid_pos[1]
        maxi = self.divisions[max_y][max_x]
        flag = False

        # TODO: maybe decrease the amount you normalize for distance?
        for j, y in enumerate(self.divisions):
            for i, x in enumerate(y):
                divs = self.divisions[j][i]

                (old_upper_x, old_lower_x), (old_upper_y, old_lower_y) = self.find_upper_and_lower(max_x, max_y)

                if flag == False:
                    old_mid_x, old_mid_y = (old_upper_x + old_lower_x) / 2, (old_upper_y + old_lower_y) / 2
                    old_distance = self.grid.distance((old_mid_x, old_mid_y), self.pos)
                    if old_distance == 0:
                        old_distance = 1
                    maxi /= (old_distance / 2)
                    flag = True

                (new_upper_x, new_lower_x), (new_upper_y, new_lower_y) = self.find_upper_and_lower(i, j)
                new_mid_x, new_mid_y = (new_upper_x + new_lower_x) / 2, (new_upper_y + new_lower_y) / 2

                new_distance = self.grid.distance((new_mid_x, new_mid_y), self.pos)

                if new_distance == 0:
                    new_distance = 1
                curr_iter_grid_prob = (divs / new_distance) * 2

                if curr_iter_grid_prob > maxi and divs != 0:
                    maxi = self.divisions[j][i]
                    max_x, max_y = i, j
                    flag = False

                elif divs == maxi:
                    # get the least traversed grid with the lowest neighbor belief
                    old_traversed, new_traversed = 0, 0

                    for x1 in range(old_lower_x, old_upper_x + 1):
                        for y1 in range(old_lower_y, old_upper_y + 1):
                            if self.grid.grid[y1][x1].traversed == True:
                                old_traversed += 1

                    for x1 in range(new_lower_x, new_upper_x + 1):
                        for y1 in range(new_lower_y, new_upper_y + 1):
                            if self.grid.grid[y1][x1].traversed == True:
                                new_traversed += 1

                    if new_traversed < old_traversed:
                        max_x, max_y = i, j
                        flag = False

                    elif new_traversed == old_traversed:
                        # i want to check the neighboring diagonal sub-grids for their probabilities
                        # let's check it for the current cell first
                        diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                        is_valid = lambda x, y: (0 <= x <= 4) and (0 <= y <= 4) 
                        new_diagonals = [(i + d[0], j + d[1]) for d in diagonals if is_valid(i + d[0], j + d[1])]
                        new_div_belief, curr_max_div_belief = 0, 0

                        for dx, dy in new_diagonals:
                            new_div_belief += self.divisions[dy][dx]
                        
                        curr_max_diagonals = [(max_x + d[0], max_y + d[1]) for d in diagonals \
                                              if is_valid(max_x + d[0], max_y + d[1])]
                        
                        for dx, dy in curr_max_diagonals:
                            curr_max_div_belief += self.divisions[dy][dx]

                        if curr_max_div_belief < new_div_belief:
                            max_x, max_y = i, j
                        elif curr_max_div_belief == new_div_belief:
                            max_x, max_y = rd.choice(((i, j), (max_x, max_y)))

                        flag = False
                    
        self.grid_pos = (max_x, max_y)
        return [maxi, (max_x, max_y)]

    def bot_division(self):
        '''
            returns (x, y) coordinates of the sub-grid to head to
            get the upper and lower bounds of the grid and head to the center of that grid
        '''
        max_belief, grid_coor = self.sub_max_belief()

        if max_belief == 0:
            # move to random neighbor cell
            neighbor = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            open_neighbors = [n for n in neighbor if self.grid.grid[self.pos[1] - n[0]][self.pos[0] - n[1]].open == True]
            max_x, max_y = rd.choice(open_neighbors)
            maxi = 0

            return [max, (max_x, max_y), grid_coor]

        (upper_x, lower_x), (upper_y, lower_y) = self.find_upper_and_lower(grid_coor[0], grid_coor[1])
        # print(f"upper_x: {upper_x}, lower_x: {lower_x}, upper_y: {upper_y}, lower_y: {lower_y}")

        maxi = 0
        max_x, max_y = self.pos[0], self.pos[1]
        for x in range(lower_x, upper_x + 1):
            for y in range(lower_y, upper_y + 1):
                curr_cell = self.grid.grid[y][x]
                if curr_cell.open == True and curr_cell.crew_belief > maxi:
                    maxi = self.grid.grid[y][x].crew_belief
                    max_x, max_y = x, y

        return [maxi, (max_x, max_y), grid_coor]  
    
    def diffuse_alien_prob(self, alien_found):
        choose_fun = None
        if alien_found:
            choose_fun = lambda x: self.within_alien_sensor(x)
        else:
            choose_fun = lambda x: not self.within_alien_sensor(x)
        open_cells = self.grid._grid.get_open_indices()
        # Cells inside the alien sensor and just outside
        # The probability will diffuse among these
        filtered_open_cells = [oc for oc in open_cells if ( choose_fun(oc) or self.alien_sensor_edge(oc, 1 if alien_found else 0) )]
        alien_belief = np.zeros((self.grid.D, self.grid.D))

        # Diffuse through the edge cells
        for ci in filtered_open_cells:
            neighbors = self.grid._grid.get_neighbors(ci)
            neighbors = [n for n in neighbors if self.grid.grid[n[1]][n[0]].open and choose_fun(n) ]
            # Diffuse the probability at the current square into the
            # neighbors that the alien can move to
            for n in neighbors:
                alien_belief[n[1]][n[0]] += self.grid.grid[ci[1]][ci[0]].alien_belief/len(neighbors)
        # Normalizs
        total_belief = np.sum(alien_belief)
        for ci in open_cells:
            alien_belief[ci[1]][ci[0]] /= total_belief
        # Update the original probabilities
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief = alien_belief[ci[1]][ci[0]]

    def restrict_alien_prob(self, alien_found):
        choose_fun = None
        if alien_found:
            choose_fun = lambda x: self.within_alien_sensor(x)
        else:
            choose_fun = lambda x: not self.within_alien_sensor(x)

        open_cells = self.grid._grid.get_open_indices()
        filtered_open_cells = [oc for oc in open_cells if not choose_fun(oc)]
        # print(f"Cells to set to 0: {len(filtered_open_cells)}")
        for ci in filtered_open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief = 0.0
        # Normalize
        total_belief = 0
        for ci in open_cells:
            total_belief += self.grid.grid[ci[1]][ci[0]].alien_belief
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief /= total_belief

    def update_belief(self, beep, alien_found):
        '''
            updates crew belief (depending on whether or not the beep is heard) every time its called (which 
            should be every time the bot moves).
        '''
        generative_fn = lambda x: np.exp(-self.alpha*(x - 1)) if beep else (1 - np.exp(-self.alpha*(x-1)))
        open_cells = self.grid._grid.get_unoccupied_open_indices()

        for ci in open_cells:
            if ci == self.pos:
                continue
            gen_res = generative_fn(self.grid.distance(ci, self.pos))
            self.grid.grid[ci[1]][ci[0]].crew_belief *= gen_res

        # Normalize
        flat_beliefs = [self.grid.grid[ci[1]][ci[0]].crew_belief for ci in open_cells]
        belief_sum = sum(flat_beliefs)
        # print(f"2. update belief function, sum of beliefs : {belief_sum}")
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].crew_belief /= belief_sum

        for j, y in enumerate(self.divisions):
            for i, x in enumerate(y):
                (old_upper_x, old_lower_x), (old_upper_y, old_lower_y) = self.find_upper_and_lower(i, j)
                probability = 0
                num_open_cells = 0

                for x1 in range(old_lower_x, old_upper_x + 1):
                    for y1 in range(old_lower_y, old_upper_y + 1):
                        if self.grid.grid[y1][x1].open == True:
                            num_open_cells += 1
                            probability += self.grid.grid[y1][x1].crew_belief

                self.divisions[j][i] = probability / num_open_cells

        alien_belief = np.zeros(( self.grid.D, self.grid.D ))
        
        self.diffuse_alien_prob(alien_found)
        self.restrict_alien_prob(alien_found)
        # print("Alien detected" if alien_found else "Alien Not Detected")

    def plan_path(self, dest):
        '''
            this function plans the path to the destination cell
        '''
        if self.debug:
            print("Planning Path...")  # If path is empty we plan one
        self.path = deque([])
        self.grid._grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.pos
        path_deque = deque([path_tree])
        destination = None
        visited = set()
        compute_counter = 0
        while not captain_found:
            if len(path_deque) == 0 or compute_counter >= COMPUTE_LIMIT:
                self.grid._grid.remove_all_traversal()
                return
            compute_counter += 1
            node = path_deque.popleft()
            ind = node.data
            if ind in visited:
                continue
            visited.add(ind)
            self.grid._grid.set_traversed(ind)
            if ind == dest:
                destination = node
                break
            neighbors_ind = self.grid._grid.get_untraversed_open_neighbors(ind)
            for neighbor_ind in neighbors_ind:
                # Add all possible paths that start with no aliens nearby
                if (self.grid.grid[neighbor_ind[1]][neighbor_ind[0]].alien_belief == 0) or (compute_counter > 2):
                    new_node = PathTreeNode()
                    new_node.data = neighbor_ind
                    new_node.parent = node
                    node.children.append(new_node)
            path_deque.extend(node.children)
        self.grid._grid.remove_all_traversal()
        if self.debug:
            print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid._grid.set_traversed(ind)
        if self.debug:
            print("Planned Path")

    def move(self):
        self.update_belief(self.crew_sensor(), self.alien_sensor())

        neighbors = self.grid._grid.get_open_neighbors(self.pos)
        neighbors = [n for n in neighbors if not self.grid.crew_pos == n]
        neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
        open_cells = self.grid._grid.get_unoccupied_open_indices()

        self.grid._grid.remove_bot(self.pos)
        _, dest_cell, grid_coor = self.bot_division()
        self.plan_path(dest_cell)
        if len(self.path) != 0:
            self.pos = self.path[0]
        # If no path is found, we automatically shift to evasion, and the evasion strategy is basic
        # Go to the cell with the lowest alien probability
        else:
            if self.debug:
                print("Evasion!!")
            neighbors = self.grid._grid.get_neighbors(self.pos)
            open_neighbors = [n for n in neighbors if self.grid.grid[n[1]][n[0]].open]
            open_neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].alien_belief)
            self.pos = open_neighbors[0]
        #elif self.grid.grid[neighbors[0][1]][neighbors[0][0]].crew_belief == self.grid.grid[neighbors[-1][1]][neighbors[-1][0]].crew_belief:
        #    self.pos = rd.choice(neighbors)
        #else:
        #    self.pos = neighbors[-1]
        
        self.grid._grid.place_bot(self.pos)
        
        if self.pos != self.grid.crew_pos:
            self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0

        if not self.grid._grid.has_alien(self.pos):
            self.grid.grid[self.pos[1]][self.pos[0]].alien_belief = 0.0
            self.tick += 1
            return [False, grid_coor]
        else:
            # bot ded
            return [True, grid_coor]

gif_coll = []
def plot_world_state(grid, bot, grid_coor=(0,0)):
    red = [1., 0., 0.]
    orange = [1.0, 0.7, 0.0]
    purple = [0.7, 0.0, 1.0]
    blue = [0., 0., 1.]
    green = [0., 1., 0.]
    yellow = [1., 1., 0.]
    white = [1., 1., 1.]
    black = [0., 0., 0.]
    grid_img = []
    grid_img2 = []
    grid_img3 = []
    open_cells = grid._grid.get_open_indices()
    beliefs_flat = [grid.grid[oc[1]][oc[0]].crew_belief for oc in open_cells]
    alien_beliefs_flat = [grid.grid[oc[1]][oc[0]].alien_belief for oc in open_cells]
    max_belief = max(beliefs_flat)
    max_alien_belief = max(alien_beliefs_flat)
    # print(f"Max Crew Belief: {max_belief}")
    # print(f"Max Alien Belief: {max_alien_belief}")
    for j in range(grid.D):
        grid_img.append([])
        grid_img2.append([])
        grid_img3.append([])
        for i in range(grid.D):
            if grid.crew_pos == (i, j):
                grid_img[-1].append(green)
            elif bot.pos == (i, j):
                grid_img[-1].append(yellow)
            elif grid._grid.has_alien((i,j)):
                grid_img[-1].append(red)
            elif grid.grid[j][i].traversed:
                grid_img[-1].append(purple)
            elif grid.grid[j][i].open:
                #grid_img[-1].append([c*grid.grid[j][i].crew_belief/max_belief for c in blue])
                #if grid.grid[j][i].crew_belief < 0:
                #    print("TOO LOW")
                grid_img[-1].append(black)
            else:
                grid_img[-1].append(white)

            if grid.grid[j][i].open:
                grid_img2[-1].append([c*grid.grid[j][i].crew_belief/max_belief for c in blue])
                if grid.grid[j][i].crew_belief < 0:
                    print("TOO LOW")
            else:
                grid_img2[-1].append(white)

            if grid.grid[j][i].open:
                grid_img3[-1].append([c*grid.grid[j][i].alien_belief/max_alien_belief for c in orange])
                if grid.grid[j][i].alien_belief < 0:
                    print("TOO LOW")
            else:
                grid_img3[-1].append(white)
    
    plt.figure(figsize=(18, 6))
    fig_manager = plt.get_current_fig_manager()
    # Set the size and position of the window using the window attribute
    fig_manager.window.geometry("+{x_position}+{y_position}".format(x_position=0, y_position=0))

    plt.subplot(131)
    plt.imshow(grid_img)

    plt.subplot(132)
    plt.imshow(grid_img2)
    for i in range(7, 35, 7):
        plt.axhline(i-0.5, color='white', linewidth=2)
        plt.axvline(i-0.5, color='white', linewidth=2)
    text = f"Fig 1. Subgrid the bot should head to: {grid_coor[0], grid_coor[1]}"
    x = 0.5 # horizontally centered
    y = -0.11 # near the bottom of the image
    plt.gca().text(x, y, text, ha='center', va='bottom', \
                   fontsize=10, transform=plt.gca().transAxes)
    
    plt.subplot(133)
    plt.imshow(grid_img3)

    plt.show()

    # plt.figure(figsize=(9, 8))

    # plt.imshow(grid_img)
    # plt.show()


for k in range(2, 10, 2):
    bot1_success = []
    bot1_deaths, bot2_deaths = 0, 0
    bot2_success = []

    for i in range(200):
        g = Grid2()
        b1 = bot1(g, k=k, debug=False)
        a1 = Alien(g)
        g2 = copy.deepcopy(g)

        bot_pos = b1.pos
        alien_pos = a1.ind
        a2 = Alien(g2, alien_pos)
        turns = 0
        dead = False

        for _ in range(500):
            dead = b1.move()
            a1.move()
            turns += 1

            if dead or b1.pos == g.crew_pos:
                break
        
        if dead:
            # this means that the bot died
            dead = False
            bot1_deaths += 1
        else:
            bot1_success.append(turns)

        plot_world_state(g, b1)
        turns = 0
        b2 = bot2(g2, k=k, bot_pos=bot_pos)

        for _ in range(500):
            dead, grid_coor = b2.move()
            a1.move()
            turns += 1
            plot_world_state(g2, b2, grid_coor)
            if dead or b2.pos == g.crew_pos:
                break
        
        if dead:
            # this means that the bot died
            dead = False
            bot2_deaths += 1
        else:
            bot2_success.append(turns)


    print(f"For k: {k}")
    print(f"Bot 1 success list: {bot1_success}")
    print(f"Bot 2 success list: {bot2_success}")
    print()

    bot1_success_rate = len(bot1_success) / 200
    bot2_success_rate = len(bot2_success) / 200

    bot1_avg = sum(bot1_success) / 200
    bot2_avg = sum(bot2_success) / 200

    print(f"The success rate of bot 1 is: {bot1_success_rate}")
    print(f"The success rate of bot 2 is: {bot2_success_rate}")
    print()

    print(f"Bot 1 died {bot1_deaths} times")
    print(f"Bot 2 died {bot2_deaths} times")
    print()

    print(f"The average number of steps taken by bot 1 are: {bot1_avg}")
    print(f"The average number of steps taken by bot 2 are: {bot2_avg}")
    print()
    print()


# print("Saving gif...")
# #gif_coll[0].save('animated.gif', save_all=True, append_images=gif_coll, duratin=len(gif_coll)*0.2, loop=0)
# os.system("ffmpeg -r 10 -i tmp%01d.png -vcodec mpeg4 -y -vb 400M movie.mp4")
# for _ in range(turns):
#     os.remove(f"tmp{_}.png")
# print("hello")
