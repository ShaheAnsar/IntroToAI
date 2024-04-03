import numpy as np
import matplotlib.pyplot as plt
from proj1 import GridAttrib, Grid, PathTreeNode
from numpy import random as nprd
import random
import random as rd
from PIL import Image
from collections import deque
import os

D=35
COMPUTE_LIMIT = 5000

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
        c = rd.choice(self._grid.get_open_indices())
        while self.crew_pos == c:
            c = rd.choice(self._grid.get_open_indices())
        self.crew_pos2 = c

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

class bot3:
    def __init__(self, grid, alpha = 0.1, k=5, debug=1):
        self.grid = grid
        self.pos = None
        while self.pos == self.grid.crew_pos or self.pos == self.grid.crew_pos2 or self.pos is None:
            self.pos = rd.choice(self.grid._grid.get_open_indices())
        self.alpha = alpha
        self.debug=debug
        self.tick=0
        self.k=k
        self.found = False

    def crew_sensor(self):
        c1 = rd.random()
        c2 = rd.random()
        d1, d2 = self.grid.distance_to_crew(self.pos)
        a, b = False, False

        if d1 is not None:
            a = c1 <= np.exp(-self.alpha* (d1 - 1))
        if d2 is not None:
            b = c2 <= np.exp(-self.alpha* (d2 - 1))

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
    
    # def diffuse_alien_prob(self, choose_fun):
        # open_cells = self.grid._grid.get_open_indices()
        # filtered_open_cells = [oc for oc in open_cells if choose_fun(oc)]
        # alien_belief = np.zeros((self.grid.D, self.grid.D))
        # for ci in filtered_open_cells:
            # neighbors = self.grid._grid.get_neighbors(ci)
            # neighbors = [n for n in neighbors if self.grid.grid[n[1]][n[0]].open and choose_fun(n)]
            # for n in neighbors:
                # alien_belief[n[1]][n[0]] += self.grid.grid[ci[1]][ci[0]].alien_belief/len(neighbors)
        # total_belief = np.sum(alien_belief)
        # for ci in open_cells:
            # alien_belief[ci[1]][ci[0]] /= total_belief
        # for ci in open_cells:
            # self.grid.grid[ci[1]][ci[0]].alien_belief = alien_belief[ci[1]][ci[0]]

    # def restrict_alien_prob(self, choose_fun):
        # open_cells = self.grid._grid.get_open_indices()
        # for i in open_cells:
            # if not choose_fun(i):
                # print("Seems to work")
        # filtered_open_cells = [oc for oc in open_cells if not choose_fun(oc)]
        # print(f"Cells to set to 0: {len(filtered_open_cells)}")
        # for ci in filtered_open_cells:
            # print("Setting to 0")
            # self.grid.grid[ci[1]][ci[0]].alien_belief = 0.0
        # total_belief = 0
        # for ci in open_cells:
            # total_belief += self.grid.grid[ci[1]][ci[0]].alien_belief
        # for ci in open_cells:
            # self.grid.grid[ci[1]][ci[0]].alien_belief /= total_belief

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
        print(f"Cells to set to 0: {len(filtered_open_cells)}")
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
            
        alien_belief = np.zeros(( self.grid.D, self.grid.D ))
        self.diffuse_alien_prob(alien_found)
        self.restrict_alien_prob(alien_found)
        print("Alien detected" if alien_found else "Alien Not Detected")

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
                if (self.grid.grid[neighbor_ind[1]][neighbor_ind[0]].alien_belief == 0 ) or (compute_counter > 2):
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
        else:
            if self.debug:
                print("Evasion!!")
            neighbors = self.grid._grid.get_neighbors(self.pos)
            open_neighbors = [n for n in neighbors if self.grid.grid[n[1]][n[0]].open]
            open_neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].alien_belief)
            self.pos = open_neighbors[0]
        # elif self.grid.grid[neighbors[0][1]][neighbors[0][0]].crew_belief == self.grid.grid[neighbors[-1][1]][neighbors[-1][0]].crew_belief:
            # self.pos = rd.choice(neighbors)
        # else:
            # self.pos = neighbors[-1]
        self.grid._grid.place_bot(self.pos)

        if self.pos != self.grid.crew_pos:
            self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0

        if not self.grid._grid.has_alien(self.pos):
            self.grid.grid[self.pos[1]][self.pos[0]].alien_belief = 0.0
        
        if self.pos == self.grid.crew_pos:
            self.grid.crew_pos = None
            self.found = True
        elif self.pos == self.grid.crew_pos2:
            self.grid.crew_pos2 = None
            self.found = True

        if self.found:
            open_cells = self.grid._grid.get_open_indices()
            open_cells = [cells for cells in open_cells if self.grid.grid[cells[1]][cells[0]].crew_belief != 0]
            count = 0
            for cell in open_cells:
                x, y = cell
                self.grid.grid[y][x].crew_belief = 1
                count += 1
            
            for cell in open_cells:
                x, y = cell
                self.grid.grid[y][x].crew_belief /= count

            self.found = False

        self.tick += 1

        if self.grid.crew_pos == None and self.grid.crew_pos2 == None:
            print("Success!")
            exit(1)

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
    print(f"Max Belief: {max_belief}")
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
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.imshow(grid_img)
    plt.subplot(132)
    plt.imshow(grid_img2)
    plt.subplot(133)
    plt.imshow(grid_img3)
    #plt.show()
g = Grid2()
b = bot3(g)
a = Alien(g._grid)
MAX_TURNS = 500
turns = 0
for _ in range(MAX_TURNS):
    print(f"Turn {_}")
    b.move()
    if g.grid[a.ind[1]][a.ind[0]].alien_belief == 0:
        print("Alien belief 0 at alien position!!!!")
    #plot_world_state(g, b)
    #plt.show()
    a.move()
    plot_world_state(g, b)
    plt.savefig(f"tmp{_}.png", dpi=200)
    plt.close()
    #plt.show()
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