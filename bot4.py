import numpy as np
import matplotlib.pyplot as plt
from proj1 import GridAttrib, Grid, PathTreeNode
from numpy import random as nprd
import random
import random as rd
from PIL import Image
from collections import deque
from math import exp
import os
import multiprocessing as mp
from multiprocessing import Pool

D = 35
COMPUTE_LIMIT = 5000

def grid_sum(D, num_g):
    s = 0
    for j in range(D):
        for i in range(D):
            s += num_g[j][i]
    return s

def zeros(x, y):
    ret = []
    for j in range(y):
        ret.append([])
        for i in range(x):
            ret[-1].append(0)
    return ret

class Alien:
    # This alien_id is used to keep track of every alien
    alien_id = 0
    def __init__(self, grid):
        self.grid = grid
        indices = self.grid.get_unoccupied_open_indices()
        ind = random.choice(indices)
        self.ind = ind
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
            rand_ind = rd.randint(0, len( neighbors_without_aliens ) - 1)
            self.grid.remove_alien(self.ind)
            self.ind = neighbors_without_aliens[rand_ind]
            self.grid.place_alien(self.ind, self.alien_id)


class Grid2:
    def __init__(self, D=35, debug=1):
        self._grid = Grid(D, debug=debug - 1>0)
        self.D = D
        self.grid = self._grid.grid

        self.beliefs = {}
        open_cells = self._grid.get_unoccupied_open_indices()

        sum = len(open_cells) * (len(open_cells) - 1) / 2
        for one_cell in open_cells:
            for two_cell in open_cells:
                if (two_cell, one_cell) in self.beliefs or one_cell == two_cell:
                    continue
                self.beliefs[(one_cell, two_cell)] = 1 / sum

        self.crew_pos = rd.choice(self._grid.get_open_indices())
        c = rd.choice(self._grid.get_open_indices())
        while self.crew_pos == c:
            c = rd.choice(self._grid.get_open_indices())
        self.crew_pos2 = c
        self.alpha = 0.1
        X = range(self.D)
        Y = range(self.D)
        prob_grid = []
        for j in range(self.D):
            prob_grid.append([])
            for i in range(self.D):
                p1 = exp(-self.alpha * self.distance(self.crew_pos, (i, j)))
                p2 = exp(-self.alpha * self.distance(self.crew_pos2, (i, j)))
                ptotal = p1 + p2 - p1*p2
                prob_grid[-1].append(ptotal)
        #prob_sum = np.sum(prob_grid)
        prob_sum = grid_sum(self.D, prob_grid)
        for j in Y:
            for i in X:
                prob_grid[j][i] /= prob_sum
        plt.imshow(prob_grid)
        plt.title("Probability of getting beeps")
        #plt.show()

    def distance(self, pos1, pos2):
        d = abs(pos1[1] - pos2[1])
        d += abs(pos1[0] - pos2[0])
        return d
    
    def distance_to_crew(self, pos):
        d1, d2 = None, None
        if self.crew_pos is not None:
            d1 = self.distance(self.crew_pos, pos)
        if self.crew_pos2 is not None:
            d2 = self.distance(self.crew_pos2, pos)

        return d1, d2
class bot4:
    def __init__(self, grid, alpha = 0.15, k=5, debug=1):
        self.grid = grid
        self.pos = None
        while self.pos == self.grid.crew_pos or self.pos == self.grid.crew_pos2 or self.pos is None:
            self.pos = rd.choice(self.grid._grid.get_open_indices())
        self.debug=debug
        if self.debug:
            print(self.pos)
        self.alpha = alpha
        
        self.tick=0
        self.k=k
        #self.found = False
        self.found_crew = None
        self.found1 = False
        self.found2 = False
        self.found_all_crew = False
        self.switch_to_single = False

    def crew_sensor(self):
        c1 = rd.random()
        c2 = rd.random()
        d1, d2 = self.grid.distance_to_crew(self.pos)
        a, b = False, False

        if d1 is not None:
            a = c1 <= exp(-self.alpha* (d1 - 1))
        if d2 is not None:
            b = c2 <= exp(-self.alpha* (d2 - 1))

        return a or b
    
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
        alien_belief = zeros(self.grid.D, self.grid.D)

        # Diffuse through the edge cells
        for ci in filtered_open_cells:
            neighbors = self.grid._grid.get_neighbors(ci)
            neighbors = [n for n in neighbors if self.grid.grid[n[1]][n[0]].open and choose_fun(n) ]
            # Diffuse the probability at the current square into the
            # neighbors that the alien can move to
            for n in neighbors:
                alien_belief[n[1]][n[0]] += self.grid.grid[ci[1]][ci[0]].alien_belief/len(neighbors)
        # Normalizs
        total_belief = grid_sum(self.grid.D, alien_belief)
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
        #print(f"Cells to set to 0: {len(filtered_open_cells)}")
        for ci in filtered_open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief = 0.0
        # Normalize
        total_belief = 0
        for ci in open_cells:
            total_belief += self.grid.grid[ci[1]][ci[0]].alien_belief
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief /= total_belief

    def update_helper(self, crew_member):
        '''
            this resets the probability after one of the crew members has been found
        '''
        # Use this switch to revert back to one crew behavior.
        # Could be done more elegantly with a list, lack of time forced this hacky solution
        self.switch_to_single = True
        crew = None
        if crew_member == 1:
            self.found_crew = self.grid.crew_pos
        elif crew_member == 2:
            self.found_crew = self.grid.crew_pos2
            
        # now we have to remove all the dict keys that don't have this crew coordinate
        new_dict = {}
        open_cells = self.grid._grid.get_open_indices()
        for oc in open_cells:
            max_belief = 0
            for k in self.grid.beliefs:
                if oc in k and self.grid.beliefs[k] > max_belief:
                    max_belief = self.grid.beliefs[k]
            new_dict[oc] = max_belief
            self.grid.grid[oc[1]][oc[0]].crew_belief = 1.0 #max_belief
        #total_belief = np.sum([v for _, v in new_dict.items()])
        #for i, k in new_dict:
        #    new_dict[k] = i/total_belief
        total_belief = sum([self.grid.grid[oc[1]][oc[0]].crew_belief for oc in open_cells])
        for oc in open_cells:
            self.grid.grid[oc[1]][oc[0]].crew_belief /= total_belief

    def update_belief(self, beep, alien_found):
        generative_fn = lambda x: exp(-self.alpha * (x - 1)) if beep else (1 - (np.exp(-self.alpha * (x - 1))))
        if self.switch_to_single:
            open_cells = self.grid._grid.get_open_indices()
            for ci in open_cells:
                if ci == self.pos:
                    continue
                dist = self.grid.distance(ci, self.pos)
                gen_res = generative_fn(dist)
                if not beep:
                    gen_res = 1.0 - gen_res
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

        else:
            # Crew Belief
            for key, _ in self.grid.beliefs.items():
                one_cell, two_cell = key
                gen_crew_one, gen_crew_two = 0, 0

                gen_crew_one = generative_fn(self.grid.distance(one_cell, self.pos))
                gen_crew_two = generative_fn(self.grid.distance(two_cell, self.pos))
                if beep:
                    total_prob = gen_crew_one + gen_crew_two - gen_crew_one * gen_crew_two
                else:
                    total_prob = gen_crew_one + gen_crew_two - gen_crew_one * gen_crew_two
                    total_prob = 1 - total_prob
                self.grid.beliefs[(one_cell, two_cell)] *= total_prob


            # Normalize
            sum_beliefs = sum(self.grid.beliefs.values())
            for key, value in self.grid.beliefs.items():
                self.grid.beliefs[key] = value / sum_beliefs

        # Alien Belief

        alien_belief = zeros(self.grid.D, self.grid.D)
        self.diffuse_alien_prob(alien_found)
        self.restrict_alien_prob(alien_found)
        #print("Alien detected" if alien_found else "Alien Not Detected")

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
                # Add all possible paths that start with no aliens nearby and go through paths with a low alien probability
                if (self.grid.grid[neighbor_ind[1]][neighbor_ind[0]].alien_belief == 0 ) or (compute_counter > 1):
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
    
    def test(self):
        consolidated_prob = {}
        for oc in self.grid._grid.get_open_indices():
            consolidated_prob[oc] = 0.0
        for k, v in self.grid.beliefs.items():
            consolidated_prob[k[0]] += v
            consolidated_prob[k[1]] += v
        overall_probs = list(consolidated_prob)
        overall_probs.sort(key=lambda x: consolidated_prob[x])
        print(f"Highest indices: {overall_probs[-1:-10:-1]}")

    def move(self):        
        beep = self.crew_sensor()
        self.update_belief(beep, self.alien_sensor())
        if self.debug:
            print("BEEP" if beep else "NO BEEP")

        neighbors = self.grid._grid.get_open_neighbors(self.pos)
        neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
        open_cells = self.grid._grid.get_unoccupied_open_indices()

        self.grid._grid.remove_bot(self.pos)

        if self.switch_to_single:
            dest_cell = max(open_cells, key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
            print(f"Dest Cell: {dest_cell}, Actual Crew: {self.grid.crew_pos}, Current Pos: {self.pos}, Distance to Crew: {self.grid.distance_to_crew(self.pos)}")
        else:
            max_belief = max(self.grid.beliefs.values())
            sorted_positions = [key for key in self.grid.beliefs.keys()]
            sorted_positions.sort(key=lambda x: self.grid.beliefs[x])
            position = sorted_positions[-1]
            dest_cell = min(position[0], position[1], 
                            key=lambda x: abs(x[0] - self.pos[0]) + abs(x[1] - self.pos[1])
                        ) if self.found_crew is None else (position[0] if self.found_crew == position[1] else position[1])
            if self.debug:
                print(f"No. of pairs: {len(self.grid.beliefs)}")
                print(f"Top 3 position pairs: {sorted_positions[-1]} : {self.grid.beliefs[sorted_positions[-1]]}, {sorted_positions[-2]} : {self.grid.beliefs[sorted_positions[-2]]}, {sorted_positions[-3]} : {self.grid.beliefs[sorted_positions[-3]]}")
                print(f"Dest Cell: {dest_cell}, actual crew: {self.grid.crew_pos}, {self.grid.crew_pos2}")
                self.test()
        self.plan_path(dest_cell)
        if len(self.path) != 0:
            self.pos = self.path[0]
        # elif self.grid.grid[neighbors[0][1]][neighbors[0][0]].crew_belief == self.grid.grid[neighbors[-1][1]][neighbors[-1][0]].crew_belief:
            # self.pos = rd.choice(neighbors)
        else:
            self.pos = neighbors[-1]
        self.grid._grid.place_bot(self.pos)

        # if self.pos != self.grid.crew_pos:
            # self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0

        if not self.grid._grid.has_alien(self.pos):
            self.grid.grid[self.pos[1]][self.pos[0]].alien_belief = 0.0

        if not self.switch_to_single:
            if (self.pos != self.found_crew) and \
                (self.pos != self.grid.crew_pos) and (self.pos != self.grid.crew_pos2):
                self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0
                keys_to_delete = []
                for key, _ in self.grid.beliefs.items():
                    if self.pos in key:
                        self.grid.beliefs[key] = 0
                        keys_to_delete.append(key)
                for k in keys_to_delete:
                    del self.grid.beliefs[k]
            
            if self.pos == self.grid.crew_pos:
                self.found1 = True
                self.update_helper(1)
                self.grid.crew_pos = self.grid.crew_pos2
                self.grid.crew_pos2 = None
            elif self.pos == self.grid.crew_pos2:
                self.found2 = True
                self.update_helper(2)
                self.grid.crew_pos2 = None
        else:
            if self.pos != self.grid.crew_pos:
                self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0
            else:
                self.found_all_crew = True

        self.tick += 1
        if self.grid.crew_pos == None and self.grid.crew_pos2 == None:
            print("Success!")
            pass
class World:
    def __init__(self, b):
        self.grid = Grid2()
        self.a = Alien(self.grid._grid)
        self.b = b(self.grid)
        self.b_fun = b

    def reset(self):
        self.grid = Grid2()
        self.a = Alien(self.grid._grid)
        self.b = self.b_fun(self.grid)

    def simulate(self):
        bot_steps = []
        for i in range(40):
            self.reset()
            MAX_TURNS = 1000
            turns = 0
        
            print(f"We're currently at iteration {i}")
            
            for _ in range(MAX_TURNS):
                # print(f"Turn {_}")
                self.b.move()
                self.a.move()
                turns += 1
                print(turns)
                #if self.grid.crew_pos == None and self.grid.crew_pos2 == None:
                if self.b.found_all_crew:
                    print(f"It took {_} steps to find both the crew members")
                    break
                    
            bot_steps.append(turns)

            print(bot_steps)

gif_coll = []
def plot_world_state(grid, bot):
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
    # print(f"Max Belief: {max_belief}")
    print(f"Max Alien Belief: {max_alien_belief}")
    for j in range(grid.D):
        grid_img.append([])
        grid_img2.append([])
        grid_img3.append([])
        for i in range(grid.D):
            if grid.crew_pos == (i, j):
                grid_img[-1].append(green)
            elif grid.crew_pos2 == (i, j):
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

w = World(bot4)
w.simulate()
