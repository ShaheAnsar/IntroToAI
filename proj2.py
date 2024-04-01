import numpy as np
import matplotlib.pyplot as plt
from proj1 import GridAttrib, Grid, Alien, PathTreeNode
from numpy import random as nprd
import random as rd
from PIL import Image
from collections import deque
import os

D = 35
COMPUTE_LIMIT = 5000

class Grid2:
    def __init__(self, D=35, debug=1):
        self._grid = Grid(D, debug=debug - 1>0)
        self.D = D
        self.grid = self._grid.grid

        midpoint_x = D // 2
        midpoint_y = D // 2

        self.quadrants = {
            'tl': [(0, 0), (midpoint_x, midpoint_y)],
            'tr': [(midpoint_x, 0), (D, midpoint_y)],
            'bl': [(0, midpoint_y), (midpoint_x, D)],
            'br': [(midpoint_x, midpoint_y), (D, D)]
        }

        # initialize the beliefs here
        self.beliefs = {}
        open_cells = self._grid.get_unoccupied_open_indices()
        sum = len(open_cells) * (len(open_cells) - 1) / 2
        for one_cell in open_cells:
            for two_cell in open_cells:
                if (two_cell, one_cell) in self.beliefs or one_cell == two_cell:
                    continue
                self.beliefs[(one_cell, two_cell)] = 1 / sum

        # choose crew member positions
        self.crew_pos = rd.choice(self._grid.get_open_indices())
        choice = rd.choice(self._grid.get_open_indices())
        while choice == self.crew_pos:
            choice = rd.choice(self._grid.get_open_indices())
        self.crew_pos2 = choice

    # manhattan distance calculator
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
        
class bot1:
    def __init__(self, grid, alpha = 0.1, k=2, debug=1):
        self.grid = grid
        self.pos = None
        while self.pos == self.grid.crew_pos or self.pos is None:
            self.pos = rd.choice(self.grid._grid.get_open_indices())
        self.alpha = alpha
        self.debug=debug
        self.tick=0
        self.k=k
        self.found_crew = None

    def crew_sensor(self):
        c = rd.random()
        d1, d2 = self.grid.distance_to_crew(self.pos)
        a, b = False, False

        if d1 is not None:
            a = c <= np.exp(-self.alpha * (d1 - 1))
        if d2 is not None:
            b = c <= np.exp(-self.alpha * (d2 - 1))

        return a or b
    
    def alien_sensor(self):
        found_alien = 0
        for j in range(-self.k, self.k + 1):
            for i in range(-self.k, self.k + 1):
                if self.grid.grid[j][i].alien_id != -1:
                    found_alien = 1
        return found_alien        
    
    def update_helper(self, crew_member: int):
        '''
            this resets the probability after one of the crew members has been found
        '''
        crew = None
        if crew_member == 1:
            self.found_crew = self.grid.crew_pos
        elif crew_member == 2:
            self.found_crew = self.grid.crew_pos2
            
        # now we have to remove all the dict keys that don't have this crew coordinate
        remove_keys_list = self.grid.beliefs.keys()
        remove_keys_list = [key for key in remove_keys_list if self.found_crew not in key]
        for key in remove_keys_list:
            del self.grid.beliefs[key]

        # now that we've removed the keys without the crew member, we need to normalize the probabilities remaining
        sum_beliefs = sum(self.grid.beliefs.values())
        print(f"the sum of beliefs is {sum_beliefs}")
        for key, _ in self.grid.beliefs.items():
            self.grid.beliefs[key] *= 1 / sum_beliefs
            print(f"beliefs after normalization: {self.grid.beliefs[key]}")
        sum_beliefs = sum(self.grid.beliefs.values())
        print(f"the sum of beliefs is {sum_beliefs}")
        
    def update_belief(self, beep, falien):
        # Crew Belief
        generative_fn = lambda x: np.exp(-self.alpha * (x - 1)) if beep else (1 - (np.exp(-self.alpha * (x - 1))))
        
        sum_beliefs = sum(self.grid.beliefs.values())
        print(f"1. update belief function, sum of beliefs : {sum_beliefs}")
        
        for key, _ in self.grid.beliefs.items():
            one_cell, two_cell = key
            gen_crew_one, gen_crew_two = 0, 0

            # probability of crew at one_cell
            if self.found_crew != one_cell:
                gen_crew_one = generative_fn(self.grid.distance(one_cell, self.pos)) \
                                
            # probability of crew at two_cell
            if self.found_crew != two_cell:
                gen_crew_two = generative_fn(self.grid.distance(two_cell, self.pos)) \

            # total_prob = generative_fn(self.grid.distance(one_cell, self.pos)) \
            #     * generative_fn(self.grid.distance(two_cell, self.pos)) \
            #     * gen_crew_one * gen_crew_two

            total_prob = gen_crew_one + gen_crew_two
            
            # TODO: MAKE SURE TO DOUBLE CHECK THE MULTIPLICATION HERE
            self.grid.beliefs[(one_cell, two_cell)] *= total_prob

        # let's normalize this
        sum_beliefs = sum(self.grid.beliefs.values())
        print(f"2. update belief function, sum of beliefs : {sum_beliefs}")
        for key, value in self.grid.beliefs.items():
            self.grid.beliefs[key] = value / sum_beliefs

    def move(self):
        self.update_belief(self.crew_sensor(), 1)

        neighbors = self.grid._grid.get_open_neighbors(self.pos)
        # neighbors = [n for n in neighbors if not self.grid.crew_pos == n]
        neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
        # open_cells = self.grid._grid.get_unoccupied_open_indices()

        self.grid._grid.remove_bot(self.pos)
        # dest_cell = max(open_cells, key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
        max_belief = max(self.grid.beliefs.values())
        print(f"The current position is {self.pos}, max belief is : {max_belief}")
        position = [key for key in self.grid.beliefs.keys() if self.grid.beliefs[key] == max_belief][0]
        dest_cell = min(position[0], position[1], 
                        key=lambda x: abs(x[0] - self.pos[0]) + abs(x[1] - self.pos[1])
                    ) if self.found_crew is None else (position[0] if self.found_crew == position[1] else position[1])

        self.plan_path(dest_cell)
        if len(self.path) != 0:
            self.pos = self.path[0]
        # elif self.grid.grid[neighbors[0][1]][neighbors[0][0]].crew_belief == self.grid.grid[neighbors[-1][1]][neighbors[-1][0]].crew_belief:
        #     self.pos = rd.choice(neighbors)
        else:
            self.pos = neighbors[-1]
        self.grid._grid.place_bot(self.pos)
        print(f"The new position is {self.pos}")

        if self.pos != self.found_crew and self.pos != self.grid.crew_pos and self.pos != self.grid.crew_pos2:
            self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0
            for key, _ in self.grid.beliefs.items():
                if self.pos in key:
                    self.grid.beliefs[key] = 0
        
        if self.pos == self.grid.crew_pos:
            self.update_helper(1)
            self.grid.crew_pos = None
        elif self.pos == self.grid.crew_pos2:
            self.update_helper(2)
            self.grid.crew_pos2 = None

        self.tick += 1
        #possible_dir = self.grid.get_open_neighbors(self.pos)
        #possible_dir.sort(key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
        #possible_dir.so

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
                self.grid.remove_all_traversal()
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
                # Add all possible paths that do not hit an alien
                if not self.grid._grid.has_alien(neighbor_ind):
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


gif_coll = []
def plot_world_state(grid, bot):
    red = [1., 0., 0.]
    blue = [0., 0., 1.]
    green = [0., 1., 0.]
    yellow = [1., 1., 0.]
    white = [1., 1., 1.]
    black = [0., 0., 0.]
    grid_img = []
    open_cells = grid._grid.get_unoccupied_open_indices()
    # beliefs_flat = [grid.grid[oc[1]][oc[0]].crew_belief for oc in open_cells]
    max_belief = max(grid.beliefs.values())
    position = [key for key in grid.beliefs.keys() if grid.beliefs[key] == max_belief][0]
    print(f"Max Belief: {max_belief}, position: {position}")
    for j in range(grid.D):
        grid_img.append([])
        for i in range(grid.D):
            if grid.crew_pos == (i, j):
                grid_img[-1].append(green)
            elif grid.crew_pos2 == (i, j):
                grid_img[-1].append(green)
            elif bot.pos == (i, j):
                grid_img[-1].append(yellow)
            elif grid.grid[j][i].open:
                grid_img[-1].append([c*grid.grid[j][i].crew_belief/max_belief for c in blue])
                if grid.grid[j][i].crew_belief < 0:
                    print("TOO LOW")
            else:
                grid_img[-1].append(white)
    plt.imshow(grid_img)
    #plt.show()
g = Grid2()
b = bot1(g)
MAX_TURNS = 200
turns = 0
for _ in range(MAX_TURNS):
    print(f"Turn {_}")
    b.move()
    plot_world_state(g, b)
    plt.savefig(f"tmp{_}.png", dpi=200)
    gif_coll.append(Image.open(f"tmp{_}.png"))
    turns += 1
    if g.crew_pos == None and g.crew_pos2 == None:
        print("SUCCES: Crew members reached!")
        break
print("Saving gif...")
#gif_coll[0].save('animated.gif', save_all=True, append_images=gif_coll, duratin=len(gif_coll)*0.2, loop=0)
os.system("ffmpeg -r 10 -i tmp%01d.png -vcodec mpeg4 -y -vb 400M movie.mp4")
for _ in range(turns):
    os.remove(f"tmp{_}.png")
print("hello")
