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

class Grid2:
    def __init__(self, D=35, debug=1):
        self._grid = Grid(D, debug=debug - 1>0)
        self.D = D
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

    def update_belief(self, beep, falien):
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
        if falien:
            print("Detected Alien!")
            for j in range(0, self.grid.D):
                for i in range(0, self.grid.D):
                    # If not within the square, set it to 0
                    if not self.within_alien_sensor((i,j)):
                        self.grid.grid[j][i].alien_belief = 0.0
            # Normalize
            total_belief = 0
            for ci in open_cells:
                total_belief += self.grid.grid[ci[1]][ci[0]].alien_belief
            for ci in open_cells:
                self.grid.grid[ci[1]][ci[0]].alien_belief /= total_belief
        else:
            print("Did not Detect Alien!")
            alien_belief = np.zeros(( self.grid.D, self.grid.D ))
            for ci in open_cells:
                neighbors = self.grid._grid.get_neighbors(ci)
                neighbors = [n for n in neighbors if self.grid.grid[n[1]][n[0]].open and not self.within_alien_sensor(n)]
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
        elif self.grid.grid[neighbors[0][1]][neighbors[0][0]].crew_belief == self.grid.grid[neighbors[-1][1]][neighbors[-1][0]].crew_belief:
            self.pos = rd.choice(neighbors)
        else:
            self.pos = neighbors[-1]
        self.grid._grid.place_bot(self.pos)
        
        if self.pos != self.grid.crew_pos:
            self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0

        if not self.grid._grid.has_alien(self.pos):
            self.grid.grid[self.pos[1]][self.pos[0]].alien_belief = 0.0

        self.tick += 1
gif_coll = []
def plot_world_state(grid, bot):
    red = [1., 0., 0.]
    orange = [1.0, 0.5, 0.5]
    blue = [0., 0., 1.]
    green = [0., 1., 0.]
    yellow = [1., 1., 0.]
    white = [1., 1., 1.]
    black = [0., 0., 0.]
    grid_img = []
    grid_img2 = []
    open_cells = grid._grid.get_open_indices()
    beliefs_flat = [grid.grid[oc[1]][oc[0]].crew_belief for oc in open_cells]
    alien_beliefs_flat = [grid.grid[oc[1]][oc[0]].alien_belief for oc in open_cells]
    max_belief = max(beliefs_flat)
    max_alien_belief = max(alien_beliefs_flat)
    print(f"Max Crew Belief: {max_belief}")
    print(f"Max Alien Belief: {max_alien_belief}")
    for j in range(grid.D):
        grid_img.append([])
        grid_img2.append([])
        for i in range(grid.D):
            if grid._grid.has_alien((i,j)):
                grid_img2[-1].append(red)
            elif bot.pos == (i, j):
                grid_img2[-1].append(yellow)
            elif grid.grid[j][i].open:
                grid_img2[-1].append([c*grid.grid[j][i].alien_belief/max_alien_belief for c in orange])
                if grid.grid[j][i].alien_belief < 0:
                    print("TOO LOW")
            else:
                grid_img2[-1].append(white)

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
    plt.subplot(121)
    plt.imshow(grid_img)
    plt.subplot(122)
    plt.imshow(grid_img2)
    #plt.show()
g = Grid2()
b = bot1(g)
a = Alien(g._grid)
MAX_TURNS = 200
turns = 0
for _ in range(MAX_TURNS):
    print(f"Turn {_}")
    b.move()
    #plot_world_state(g, b)
    #plt.show()
    a.move()
    plot_world_state(g, b)
    plt.savefig(f"tmp{_}.png", dpi=200)
    plt.show()
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
