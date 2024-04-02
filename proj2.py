import numpy as np
import matplotlib.pyplot as plt
from proj1 import GridAttrib, Grid, Alien, PathTreeNode
from numpy import random as nprd
import random as rd
from PIL import Image
from collections import deque
import os

D=35
COMPUTE_LIMIT = 5000

'''
    # TODO
    after creating the divisions grid, we're going to need a few more things:
        - when the bot has to move, change the syntax to get the dest_cell from the created functions
        - for the step above, create more functions if needed
        - path planning, that muzzammil should take care of
        - normalize by distance (?)
'''

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
    def __init__(self, grid, alpha = 0.1, k=2, debug=1):
        self.grid = grid
        self.pos = None
        while self.pos == self.grid.crew_pos or self.pos is None:
            self.pos = rd.choice(self.grid._grid.get_open_indices())
        self.alpha = alpha
        self.debug=debug
        self.tick=0
        self.k=k
        self.divisions = self.grid.divisions

    def crew_sensor(self):
        c = rd.random()
        return c <= np.exp(-self.alpha
                           * (self.grid.distance_to_crew(self.pos) - 1))
    def alien_sensor(self):
        found_alien = 0
        for j in range(-self.k, self.k + 1):
            for i in range(-self.k, self.k + 1):
                if self.grid.grid[j][i].alien_id != -1:
                    found_alien = 1
        return found_alien
    
    def find_upper_and_lower(self, grid_x, grid_y):
        upper_x, lower_x = abs((grid_x * 5) - 1), grid_x * 4
        upper_y, lower_y = abs((grid_y * 5) - 1), grid_y * 4
        return ((upper_x, lower_x), (upper_y, lower_y))
    
    def sub_max_belief(self):
        '''
            returns the next sub-grid to go to
        '''
        max_x, max_y = -1, -1
        maxi = -1

        # TODO: normalize for distance
        for y in self.divisions:
            for x in y:
                divs = self.divisions[y][x]
                if divs != 0 and divs > maxi:
                    maxi = self.divisions[y][x]
                    max_x, max_y = x, y

                elif divs == maxi:
                    # get the least traversed grid with the lowest neighbor belief
                    old_traversed, new_traversed = 0, 0
                    
                    (old_upper_x, old_lower_x), (old_upper_y, old_lower_y) = self.find_upper_and_lower(max_x, max_y)
                    (new_upper_x, new_lower_x), (new_upper_y, new_lower_y) = self.find_upper_and_lower(x, y)

                    for x in range(old_lower_x, old_upper_x + 1):
                        for y in range(old_lower_y, old_upper_y + 1):
                            if self.grid.grid[y][x].traversed == True:
                                old_traversed += 1

                    for x in range(new_lower_x, new_upper_x + 1):
                        for y in range(new_lower_y, new_upper_y + 1):
                            if self.grid.grid[y][x].traversed == True:
                                new_traversed += 1

                    if new_traversed < old_traversed:
                        max_x, max_y = x, y
                    elif new_traversed == old_traversed:
                        # i want to check the neighboring diagonal sub-grids for their probabilities
                        # let's check it for the current cell first
                        diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                        is_valid = lambda x, y: (x, y) if (0 <= x <= 4) and (0 <= y <= 4) else None
                        new_diagonals = [(x + d[0], y + d[1]) for d in diagonals if is_valid(x + d[0], y + d[1]) is not None]
                        new_div_belief, curr_max_div_belief = 0, 0

                        for dx, dy in new_diagonals:
                            new_div_belief += self.divisions[dy][dx]
                        
                        curr_max_diagonals = [(max_x + d[0], max_y + d[1]) for d in diagonals \
                                              if is_valid(max_x + d[0], max_y + d[1]) is not None]
                        
                        for dx, dy in curr_max_diagonals:
                            curr_max_div_belief += self.divisions[dy][dx]

                        if curr_max_diagonals < new_div_belief:
                            max_x, max_y = x, y
                        elif curr_max_diagonals == new_div_belief:
                            max_x, max_y = rd.choice(((x, y), (max_x, max_y)))
                #     new_mid_x, new_mid_y = (new_upper_x + new_lower_x) / 2, (new_upper_y + new_lower_y) / 2
                #     new_dist = self.grid.distance((new_mid_x, new_mid_y), (self.pos[0], self.pos[1]))

                #     old_mid_x, old_mid_y = (old_upper_x + old_lower_x) / 2, (old_upper_y + old_lower_y) / 2
                #     old_dist = self.grid.distance((old_mid_x, old_mid_y), (self.pos[0], self.pos[1]))

                #     if new_dist < old_dist:
                #         max_x, max_y = x, y
                #     elif new_dist == old_dist:
                #         # i want a way to find out which one is more worth going to
                #         # maybe seeing the neighboring grids?
                #         pass
                    
        # a very rare edge case
        if maxi == -1:
            # move to random neighbor cell
            neighbor = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            open_neighbors = [n for n in neighbor if self.grid.grid[self.pos[1] - n[0]][self.pos[0] - n[1]].open == True]
            max_x, max_y = rd.choice(open_neighbors)
            maxi = 0
                    
        return [maxi, (max_x, max_y)]

    def bot_division(self):
        '''
            returns (x, y) coordinates of the sub-grid to head to
            get the upper and lower bounds of the grid and find the middle spot there(?)
        '''
        # n and n - 1, n = 5
        max_belief, grid_coor = self.sub_max_belief()
        (upper_x, lower_x), (upper_y, lower_y) = self.find_upper_and_lower(grid_coor[0], grid_coor[1])

        # what if the bot is already in the sub-grid?
        if (lower_x <= self.pos[1] <= upper_x) and (lower_y <= self.pos[0] <= upper_y):
            # this means that the bot is already in the sub-grid
            maxi = 0
            max_x, max_y = 0, 0
            for x in range(lower_x, upper_x + 1):
                for y in range(lower_y, upper_y + 1):
                    curr_cell = self.grid.grid[y][x]
                    if curr_cell.open == True and curr_cell.crew_belief > maxi:
                        maxi = self.grid.grid[y][x].crew_belief
                        max_x, max_y = x, y

            return [maxi, (max_x, max_y)]

        # now we have to find the center cell AND make sure that it's not a wall 
        mid_x, mid_y = (upper_x + lower_x) / 2, (upper_y + lower_y) / 2
        is_valid = lambda x, y: True if (0 <= x <= 34 and 0 <= y <= 34) else False
        deq = deque((mid_x, mid_y))

        while self.grid.grid[mid_x][mid_y].open == False:
            # keep finding other cells (go through the neighbors till you get something?)
            mid_x, mid_y = deq.popleft()
            neighbor = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            n_x, n_y = mid_x + neighbor[0], mid_y + neighbor[1]
            if is_valid(n_x, n_y):
                deq.append((n_x, n_y))

        return [max_belief, (mid_x, mid_y)]


    def update_belief(self, beep, falien):
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

    def move(self):
        self.update_belief(self.crew_sensor(), 1)

        neighbors = self.grid._grid.get_open_neighbors(self.pos)
        neighbors = [n for n in neighbors if not self.grid.crew_pos == n]
        neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
        open_cells = self.grid._grid.get_unoccupied_open_indices()

        self.grid._grid.remove_bot(self.pos)
        dest_cell = max(open_cells, key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
        self.plan_path(dest_cell)
        if len(self.path) != 0:
            self.pos = self.path[0]
        elif self.grid.grid[neighbors[0][1]][neighbors[0][0]].crew_belief == self.grid.grid[neighbors[-1][1]][neighbors[-1][0]].crew_belief:
            self.pos = rd.choice(neighbors)
        else:
            self.pos = neighbors[-1]
        self.grid._grid.place_bot(self.pos)

        if self.pos != self.grid.crew_pos:
            self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0

        self.tick += 1
        #possible_dir = self.grid.get_open_neighbors(self.pos)
        #possible_dir.sort(key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
        #possible_dir.so
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
    beliefs_flat = [grid.grid[oc[1]][oc[0]].crew_belief for oc in open_cells]
    max_belief = max(beliefs_flat)
    print(f"Max Belief: {max_belief}")
    for j in range(grid.D):
        grid_img.append([])
        for i in range(grid.D):
            if grid.crew_pos == (i, j):
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
    if g.crew_pos == b.pos:
        print("SUCCES: Crew member reached!")
        break
print("Saving gif...")
#gif_coll[0].save('animated.gif', save_all=True, append_images=gif_coll, duratin=len(gif_coll)*0.2, loop=0)
os.system("ffmpeg -r 10 -i tmp%01d.png -vcodec mpeg4 -y -vb 400M movie.mp4")
for _ in range(turns):
    os.remove(f"tmp{_}.png")
print("hello")
