import numpy as np
from collections import namedtuple
from collections import deque
import copy
import random
from time import sleep
from termcolor import colored
from matplotlib import pyplot as plt
from multiprocessing import Pool
import time

GridPointAttrib = {}

class Grid:
    def __init__(self, D=30, debug=True):
        self.D = D
        self.grid = []
        self.debug = debug
        self.gen_grid()

    def valid_index(self, ind):
        if ind[0] >= self.D or ind[0] < 0 or ind[1] >= self.D or ind[1] < 0:
            return False
        return True

    def get_neighbors(self, ind):
        neighbors = []
        left = (ind[0] - 1, ind[1])
        right = (ind[0] + 1, ind[1])
        up = (ind[0], ind[1] + 1)
        down = (ind[0], ind[1] - 1)
        indices = [left, right, up, down]
        for index in indices:
            if self.valid_index(index):
                neighbors.append(index)
        return neighbors
    def get_open_neighbors(self, ind):
        neighbors = []
        left = (ind[0] - 1, ind[1])
        right = (ind[0] + 1, ind[1])
        up = (ind[0], ind[1] + 1)
        down = (ind[0], ind[1] - 1)
        indices = [left, right, up, down]
        for index in indices:
            if self.valid_index(index) and self.grid[index[1]][index[0]]['open'] == True:
                neighbors.append(index)
        return neighbors

    def get_untraversed_open_neighbors(self, ind):
        neighbors = []
        left = (ind[0] - 1, ind[1])
        right = (ind[0] + 1, ind[1])
        up = (ind[0], ind[1] + 1)
        down = (ind[0], ind[1] - 1)
        indices = [left, right, up, down]
        for index in indices:
            if self.valid_index(index) and self.grid[index[1]][index[0]]['open'] == True and self.grid[index[1]][index[0]]['traversed'] == False:
                neighbors.append(index)
        return neighbors

    def gen_grid_iterate(self):
        cells_to_open = []
        for j in range(self.D):
            for i in range(self.D):
                if self.grid[j][i]['open'] == True:
                    continue
                neighbors_ind = self.get_neighbors((i, j))
                open_neighbors = []
                for neighbor_ind in neighbors_ind:
                    if self.grid[neighbor_ind[1]][neighbor_ind[0]]['open'] is True:
                        open_neighbors.append(neighbor_ind)
                if len(open_neighbors) == 1:
                    cells_to_open.append((i, j))
        if len(cells_to_open) > 0:
            index = random.choice(cells_to_open)
            self.grid[index[1]][index[0]]['open'] = True
        if self.debug:
            print("After one iteration")
            print(self)
            print(f"Cells to open: {len(cells_to_open)}")
        return len(cells_to_open) != 0

    def gen_grid(self):
        for j in range(self.D):
            row = []
            for i in range(self.D):
                row.append({'open': False, 'traversed' : False, 'captain_slot': False, 'alien_id' : -1, 'bot_occupied': False})
            self.grid.append(row)
        # Open Random Cell
        rand_ind = np.random.randint(0, self.D, 2)
        self.grid[rand_ind[1]][rand_ind[0]]['open'] = True
        # Go through all cells in the grid 
        # Any cell with one open neigbor, add the index to 
        # And then select one at random
        if self.debug:
            print(self)
        while self.gen_grid_iterate():
            pass
        cells_to_open = []
        for j in range(self.D):
            for i in range(self.D):
                    all_neighbors = self.get_neighbors((i,j))
                    open_neighbors = [ind for ind in all_neighbors if self.grid[ind[1]][ind[0]]['open']]
                    closed_neighbors = [ind for ind in all_neighbors if not self.grid[ind[1]][ind[0]]['open']]
                    if self.grid[j][i]['open'] and random.randint(0, 1) == 1 and len(open_neighbors) == 1:
                        cells_to_open.append(random.choice(closed_neighbors))
        for ind in cells_to_open:
            self.grid[ind[1]][ind[0]]['open'] = True
        if self.debug:
            print("After dead end opening")
            print(self)

    def place_alien(self, ind, alien_id):
        self.grid[ind[1]][ind[0]]['alien_id'] = alien_id
    def remove_alien(self, ind):
        self.grid[ind[1]][ind[0]]['alien_id'] = -1
    # k tells us how deep to look from the index
    def has_alien(self, ind, k=1):
        if k == 1:
            return self.grid[ind[1]][ind[0]]['alien_id'] != -1
        elif k==2:
            ret = self.grid[ind[1]][ind[0]]['alien_id'] != -1
            neighbors = self.get_open_neighbors(ind)
            return ret or all([self.has_alien(neighbor) for neighbor in neighbors])
        else:
            #print(f"Has_Alien: {ind}")
            traversed = {}
            children = deque([])
            current = deque([ind])
            #print("Has_Alien: depth more than 1")
            while k >= 1:
                #print(f" At inverse depth of {k}")
                #print(f"Current Fringe: {current}")
                for ind in current:
                    traversed[ind] = 1
                    if self.grid[ind[1]][ind[0]]['alien_id'] != -1:
                        return True
                    neighbors = self.get_open_neighbors(ind)
                    #print(f"Neighbors before filter: {neighbors}")
                    neighbors = [neighbor for neighbor in neighbors if neighbor not in traversed]
                    #print(f"Neighbors after filter: {neighbors}")
                    children.extend(neighbors)
                current = children
                children = deque([])
                k -= 1
            return False
                
                    
                
    def place_bot(self, ind):
        self.grid[ind[1]][ind[0]]['bot_occupied'] = True
    def remove_bot(self, ind):
        self.grid[ind[1]][ind[0]]['bot_occupied'] = False
    def set_traversed(self, ind):
        self.grid[ind[1]][ind[0]]['traversed'] = True
    def remove_all_traversal(self):
        for j in range(self.D):
            for i in range(self.D):
                self.grid[j][i]['traversed'] = False

    def get_open_indices(self):
        return [(i, j) for i in range(self.D) for j in range(self.D) if self.grid[j][i]['open'] == True]

    def get_unoccupied_open_indices(self):
        return [(i, j) for i in range(self.D) for j in range(self.D) if self.grid[j][i]['open'] == True and self.grid[j][i]['alien_id'] == -1
                and self.grid[j][i]['bot_occupied'] == False]


    def __str__(self):
        s = ""
        for j in range(self.D):
            for i in range(self.D):
                if self.grid[j][i]['open'] == True:
                    if self.grid[j][i]['captain_slot']:
                        s += colored('C', 'magenta')
                    elif self.grid[j][i]['alien_id'] != -1:
                        s += colored('A', 'red')
                    elif self.grid[j][i]['bot_occupied']:
                        s += colored('B', 'yellow')
                    elif self.grid[j][i]['traversed']:
                        s += colored('P', 'blue')
                    else:
                        s += colored('O', 'green')
                else:
                    s += 'X'
            s += "\n"
        return s

class Alien:
    alien_id = 0
    aliens_ind = []
    def __init__(self, grid):
        self.grid = grid
        indices = self.grid.get_unoccupied_open_indices()
        ind = random.choice(indices)
        self.ind = ind
        Alien.aliens_ind.append(ind)
        self.alien_id = Alien.alien_id
        self.grid.place_alien(ind, Alien.alien_id)
        Alien.alien_id += 1
        #print(ind)

    def move(self):
        neighbors = self.grid.get_open_neighbors(self.ind)
        neighbors_without_aliens = [neighbor for neighbor in neighbors if self.grid.grid[neighbor[1]][neighbor[0]]['alien_id'] == -1]
        if len(neighbors_without_aliens) > 0:
            rand_ind = np.random.randint(0, len( neighbors_without_aliens ))
            self.grid.remove_alien(self.ind)
            self.ind = neighbors_without_aliens[rand_ind]
            self.grid.place_alien(self.ind, self.alien_id)


class PathTreeNode:
    def __init__(self):
        self.children = []
        self.parent = None
        self.data = None



class Bot1:
    def __init__(self, grid, captain_ind, debug=True):
        self.grid = grid
        self.captain_ind = captain_ind
        self.ind = random.choice(self.grid.get_open_indices())
        self.grid.place_bot(self.ind)
        self.path = deque([])
        self.debug = debug

    def plan_path(self):
        if self.debug:
            print("Planning Path...")  # If path is empty we plan one
        self.grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.ind
        #path_deque = deque([path_tree])
        path_deque = deque([self.ind])
        path_map = {self.ind: None}
        visited = set()
        destination = None
        while not captain_found:
            if len(path_deque) == 0:
                #raise RuntimeError("No Path Found!!!")
                return
            #node = path_deque.popleft()
            #ind = node.data
            ind = path_deque.popleft()
            visited.add(ind)
            if self.debug:
                print(f"Current Node: {ind}")
            #self.grid.set_traversed(ind)
            if ind == self.captain_ind:
                destination = ind
                break
            neighbors_ind = self.grid.get_open_neighbors(ind)
            neighbors_ind = [i for i in neighbors_ind if i not in visited]
            for neighbor_ind in neighbors_ind:
                # Add all possible paths that do not hit an alien
                if not self.grid.has_alien(neighbor_ind):
                    path_deque.append(neighbor_ind)
                    path_map[neighbor_ind] = ind
                    #new_node = PathTreeNode()
                    #new_node.data = neighbor_ind
                    #new_node.parent = node
                    #node.children.append(new_node)

            #path_deque.extend(node.children)
        #self.grid.remove_all_traversal()
        if self.debug:
            print("Planning Done!")
        reverse_path = []
        next_ind = destination
        while next_ind is not None:
            reverse_path.append(next_ind)
            next_ind = path_map[next_ind]
        #node = destination
        #while node.parent is not None:
        #    reverse_path.append(node.data)
        #    node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid.set_traversed(ind)
        if self.debug:
            print("Planned Path")
            print(self.grid)
    #def plan_path(self):
    #    if self.debug:
    #        print("Planning Path...")  # If path is empty we plan one
    #    self.path = deque([])
    #    self.grid.remove_all_traversal()
    #    captain_found = False
    #    path_tree = PathTreeNode()
    #    path_tree.data = self.ind
    #    path_deque = deque([path_tree])
    #    destination = None
    #    while not captain_found:
    #        if len(path_deque) == 0:
    #            self.grid.remove_all_traversal()
    #            #raise RuntimeError("No Path Found!!!")
    #            return
    #        node = path_deque.popleft()
    #        ind = node.data
    #        self.grid.set_traversed(ind)
    #        if ind == self.captain_ind:
    #            destination = node
    #            break
    #        neighbors_ind = self.grid.get_untraversed_open_neighbors(ind)
    #        for neighbor_ind in neighbors_ind:
    #            # Add all possible paths that do not hit an alien
    #            if not self.grid.has_alien(neighbor_ind):
    #                new_node = PathTreeNode()
    #                new_node.data = neighbor_ind
    #                new_node.parent = node
    #                node.children.append(new_node)
    #        path_deque.extend(node.children)
    #    self.grid.remove_all_traversal()
    #    if self.debug:
    #        print("Planning Done!")
    #    reverse_path = []
    #    node = destination
    #    while node.parent is not None:
    #        reverse_path.append(node.data)
    #        node = node.parent
    #    self.path.extend(reversed(reverse_path))
    #    for ind in self.path:
    #        self.grid.set_traversed(ind)
    #    if self.debug:
    #        print("Planned Path")
    #        print(self.grid)

    def move(self):
        if not self.path:
            self.plan_path()
        if len(self.path) == 0:
            if self.debug:
                print("No path found!")
            return

        next_dest = self.path.popleft()
        self.grid.remove_bot(self.ind)
        self.ind = next_dest
        self.grid.place_bot(self.ind)
            


class Bot2:
    def __init__(self, grid, captain_ind, debug = True):
        self.grid = grid
        self.captain_ind = captain_ind
        self.ind = random.choice(self.grid.get_open_indices())
        self.grid.place_bot(self.ind)
        self.path = deque([])
        self.debug = debug

    def plan_path(self):
        if self.debug:
            print("Planning Path...")  # If path is empty we plan one
        self.path = deque([])
        self.grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.ind
        path_deque = deque([path_tree])
        destination = None
        while not captain_found:
            if len(path_deque) == 0:
                self.grid.remove_all_traversal()
                #raise RuntimeError("No Path Found!!!")
                return
            node = path_deque.popleft()
            ind = node.data
            self.grid.set_traversed(ind)
            if ind == self.captain_ind:
                destination = node
                break
            neighbors_ind = self.grid.get_untraversed_open_neighbors(ind)
            for neighbor_ind in neighbors_ind:
                # Add all possible paths that do not hit an alien
                if not self.grid.has_alien(neighbor_ind):
                    new_node = PathTreeNode()
                    new_node.data = neighbor_ind
                    new_node.parent = node
                    node.children.append(new_node)
            path_deque.extend(node.children)
        self.grid.remove_all_traversal()
        if self.debug:
            print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid.set_traversed(ind)
        if self.debug:
            print("Planned Path")
            print(self.grid)

    def move(self):
        self.plan_path()
        if len(self.path) == 0:
            if self.debug:
                print("No path found!")
            return
        next_dest = self.path.popleft()
        self.grid.remove_bot(self.ind)
        self.ind = next_dest
        self.grid.place_bot(self.ind)


class Bot3:
    def __init__(self, grid, captain_ind, debug = True):
        self.grid = grid
        self.captain_ind = captain_ind
        self.ind = random.choice(self.grid.get_open_indices())
        self.grid.place_bot(self.ind)
        self.path = deque([])
        self.debug = debug

    def plan_path(self, k=2):
        if self.debug:
            print("Planning Path...")  # If path is empty we plan one
        self.path = deque([])
        self.grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.ind
        path_deque = deque([path_tree])
        destination = None
        while not captain_found:
            if len(path_deque) == 0:
                self.grid.remove_all_traversal()
                #raise RuntimeError("No Path Found!!!")
                return
            node = path_deque.popleft()
            ind = node.data
            self.grid.set_traversed(ind)
            if ind == self.captain_ind:
                destination = node
                break
            neighbors_ind = self.grid.get_untraversed_open_neighbors(ind)
            for neighbor_ind in neighbors_ind:
                # Add all possible paths that do not hit an alien
                if not self.grid.has_alien(neighbor_ind, k = k):
                    new_node = PathTreeNode()
                    new_node.data = neighbor_ind
                    new_node.parent = node
                    node.children.append(new_node)
            path_deque.extend(node.children)
        self.grid.remove_all_traversal()
        if self.debug:
            print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid.set_traversed(ind)
        if self.debug:
            print("Planned Path")
            print(self.grid)

    def move(self):
        self.plan_path(2)
        if len(self.path) == 0:
            if self.debug:
                print("Reverting...")
            self.plan_path(1)
            if len(self.path) == 0:
                if self.debug:
                    print("No path found")
                return
        next_dest = self.path.popleft()
        self.grid.remove_bot(self.ind)
        self.ind = next_dest
        self.grid.place_bot(self.ind)
class World:
    def __init__(self, debug=True, track_time = False, concurrent=False):
        self.debug = debug
    def gen_world(self, K):
        self.grid = Grid(debug=self.debug)
        self.captain_ind = random.choice(self.grid.get_open_indices())
        self.aliens = [Alien(self.grid) for _ in range(K)]
        self.captain_found = False
        self.bot_caught = False

    def gather_data(self, iters=20, K_end=20):
        self.data_dict = {}
        for b in range(3):
            self.data_dict[b] = [[], []]
            for K in range(K_end):
                print(f"Bot {b + 1}, K: {K}")
                start_time = time.perf_counter()
                successes = 0
                survivals = 0
                for _ in range(iters):
                    self.gen_world(K)
                    bot = None
                    if b == 0:
                        bot = Bot1(self.grid, self.captain_ind, debug=self.debug)
                    elif b == 1:
                        bot = Bot2(self.grid, self.captain_ind, debug=self.debug)
                    else:
                        bot = Bot3(self.grid, self.captain_ind, debug=self.debug)
                    ret = self.simulate_world(bot)
                    if ret == 0:
                        successes += 1
                        survivals += 1
                    elif ret == -1:
                        survivals += 1
                    elif ret == -2:
                        pass
                    else:
                        print("Ya fucked up bruv")
                end_time = time.perf_counter()
                print(f"Time to run: {end_time - start_time}")
                success_rate = successes/iters
                survival_rate = survivals/iters
                self.data_dict[b][0].append(success_rate)
                self.data_dict[b][1].append(survival_rate)
        for b in range(3):
            for i in range(2):
                self.data_dict[b][1] = np.array(self.data_dict[b][1])
                self.data_dict[b][0] = np.array(self.data_dict[b][0])
        print(self.data_dict)
    def plot_data(self):
        K = len(self.data_dict[0][0])
        x = np.arange(K)
        for b in range(3):
            plt.plot(x, self.data_dict[b][0], label=f"Bot {b + 1} Success")
            plt.plot(x, self.data_dict[b][1], label=f"Bot { b + 1} Survival")
        plt.legend()
        plt.ylim(0.001, 1.0)
        plt.show()

    def simulate_world(self, bot):
        for _ in range(1000):
            if self.bot_caught:
                break
            bot.move()
            if bot.ind == self.captain_ind:
                self.captain_found = True
                break
            for alien in self.aliens:
                if bot.ind == alien.ind:
                    self.bot_caught = True
                    break
                alien.move()
                if bot.ind == alien.ind:
                    self.bot_caught = True
                    if self.debug:
                        print("Failure")
                    return -2
                    break
            if self.debug:
                print("Next Iteration")
                print(self.grid)
                sleep(0.016)
        if self.captain_found:
            return 0
            if self.debug:
                print("Success")
        else:
            return -1
            if self.debug:
                print("Failure")

#debug = False
#grid = Grid(debug=debug)
#captain_ind = random.choice(grid.get_open_indices())
#grid.grid[captain_ind[1]][captain_ind[0]]['captain_slot'] = True
#bot = Bot1(grid, captain_ind, debug=debug)
#print(f"Bot index: {bot.ind}")
#aliens = [Alien(grid) for _ in range(30)]
#print("After placing 10 alien")
#print(grid)
#captain_found = False
#bot_caught = False
#for _ in range(1000):
#    if bot_caught:
#        break
#    bot.move()
#    if bot.ind == captain_ind:
#        captain_found = True
#        break
#    for alien in aliens:
#        if bot.ind == alien.ind:
#            bot_caught = True
#            break
#        alien.move()
#        if bot.ind == alien.ind:
#            bot_caught = True
#            print("Failure")
#            break
#    print("Next Iteration")
#    #for alien in aliens:
#    #    print(f"Alien {alien.alien_id} position: {alien.ind}")
#    print(grid)
#    sleep(0.016)
#if captain_found:
#    print("Success")
#else:
#    print("Failure")
plt.style.use('ggplot')
w = World(debug=False)
w.gather_data(iters=20, K_end=5)
w.plot_data()
