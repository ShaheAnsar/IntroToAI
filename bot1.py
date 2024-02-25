import numpy as np
from collections import namedtuple
from collections import deque
import copy
import random
from time import sleep
from termcolor import colored

GridPointAttrib = {}

class Grid:
    def __init__(self, D=30):
        self.D = D
        self.grid = []
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
        for index in cells_to_open:
            self.grid[index[1]][index[0]]['open'] = True
        print("After one iteration")
        print(self)
        print(f"Cells to open: {len(cells_to_open)}")
        return len(cells_to_open) != 0

    def gen_grid(self):
        for j in range(self.D):
            row = []
            for i in range(self.D):
                row.append({'open': False, 'traversed' : False, 'dead_end': False, 'alien_id' : -1, 'bot_occupied': False})
            self.grid.append(row)
        # Open Random Cell
        rand_ind = np.random.randint(0, self.D, 2)
        self.grid[rand_ind[1]][rand_ind[0]]['open'] = True
        # Go through all cells in the grid 
        # Any cell with one open neigbor, add the index to 
        # And then select one at random
        print(self)
        while self.gen_grid_iterate():
            pass
        print("After end")
        print(self)

    def place_alien(self, ind, alien_id):
        self.grid[ind[1]][ind[0]]['alien_id'] = alien_id
    def remove_alien(self, ind):
        self.grid[ind[1]][ind[0]]['alien_id'] = -1
    # k tells us how deep to look from the index
    def has_alien(self, ind, k=1):
        if k == 1:
            return self.grid[ind[1]][ind[0]]['alien_id'] != -1
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
                    if self.grid[j][i]['alien_id'] != -1:
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
        print(ind)

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
    def __init__(self, grid, captain_ind):
        self.grid = grid
        self.captain_ind = captain_ind
        self.ind = random.choice(self.grid.get_open_indices())
        self.grid.place_bot(self.ind)
        self.path = deque([])

    def plan_path(self):
        print("Planning Path...")  # If path is empty we plan one
        self.grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.ind
        path_deque = deque([path_tree])
        destination = None
        while not captain_found:
            if len(path_deque) == 0:
                #raise RuntimeError("No Path Found!!!")
                return
            node = path_deque.popleft()
            print(f"Current Node: {node.data}")
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
        print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid.set_traversed(ind)
        print("Planned Path")
        print(self.grid)

    def move(self):
        if not self.path:
            self.plan_path()
        if len(self.path) == 0:
            print("No path found!")
            return

        next_dest = self.path.popleft()
        self.grid.remove_bot(self.ind)
        self.ind = next_dest
        self.grid.place_bot(self.ind)
            


class Bot2:
    def __init__(self, grid, captain_ind):
        self.grid = grid
        self.captain_ind = captain_ind
        self.ind = random.choice(self.grid.get_open_indices())
        self.grid.place_bot(self.ind)
        self.path = deque([])

    def plan_path(self):
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
        print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid.set_traversed(ind)
        print("Planned Path")
        print(self.grid)

    def move(self):
        self.plan_path()
        if len(self.path) == 0:
            print("No path found!")
            return
        next_dest = self.path.popleft()
        self.grid.remove_bot(self.ind)
        self.ind = next_dest
        self.grid.place_bot(self.ind)


class Bot3:
    def __init__(self, grid, captain_ind):
        self.grid = grid
        self.captain_ind = captain_ind
        self.ind = random.choice(self.grid.get_open_indices())
        self.grid.place_bot(self.ind)
        self.path = deque([])

    def plan_path(self, k=2):
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
        print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid.set_traversed(ind)
        print("Planned Path")
        print(self.grid)

    def move(self):
        self.plan_path(2)
        if len(self.path) == 0:
            print("Reverting...")
            self.plan_path(1)
            if len(self.path) == 0:
                print("No path found")
                return
        next_dest = self.path.popleft()
        self.grid.remove_bot(self.ind)
        self.ind = next_dest
        self.grid.place_bot(self.ind)
grid = Grid()
captain_ind = random.choice(grid.get_open_indices())
bot = Bot3(grid, captain_ind)
print(f"Bot index: {bot.ind}")
aliens = [Alien(grid) for _ in range(3)]
print("After placing 10 alien")
print(grid)
captain_found = False
bot_caught = False
for _ in range(1000):
    if bot_caught:
        break
    bot.move()
    if bot.ind == captain_ind:
        captain_found = True
        break
    for alien in aliens:
        if bot.ind == alien.ind:
            bot_caught = True
            break
        alien.move()
        if bot.ind == alien.ind:
            bot_caught = True
            print("Failure")
            break
    print("Next Iteration")
    #for alien in aliens:
    #    print(f"Alien {alien.alien_id} position: {alien.ind}")
    print(grid)
    sleep(0.016)
if captain_found:
    print("Success")
else:
    print("Failure")
