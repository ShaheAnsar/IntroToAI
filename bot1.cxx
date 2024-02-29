#include <iostream>
#include <vector>
#include <array>
#include <functional>
#include <cstdlib>
#include <ctime>

struct GridAttrib{
	int open;
	int bot_occupied;
	int traversed;
	int alien_id;
	int captain_slot;
};

struct Grid{
	int D;
	std::vector<GridAttrib> grid;
	int debug;
	Grid(int D = 30, int debug_level = 1) :
		D(D), debug(debug_level)
	{
	}

	GridAttrib* get(int x, int y){
		return &grid[y * D + x];
	}
	int valid_index(int x, int y){
		return ((x >= 0) && (x < D) && (y >= 0) && (y < D));
	}

	std::vector<std::array<int, 2>> get_neighbors(int x, int y) {
		std::vector<std::array<int, 2>> all_indices;
		std::vector<std::array<int, 2>> ret;
		all_indices.push_back({x + 1, y});
		all_indices.push_back({x - 1, y});
		all_indices.push_back({x, y + 1});
		all_indices.push_back({x, y - 1});
		for(int i = 0; i < all_indices.size(); i++) {
			if(valid_index(all_indices[i][0], all_indices[i][1]))
				ret.push_back(all_indices[i]);
		}
		return ret;
	}
	std::vector<std::array<int, 2>> get_open_neighbors(int x, int y) {
		std::vector<std::array<int, 2>> all_neighbors = get_neighbors(x, y);
		std::vector<std::array<int, 2>> ret;
		for(int i = 0; i < all_neighbors.size(); i++) {
			int n_x = all_neighbors[i][0];
			int n_y = all_neighbors[i][1];
			GridAttrib* g = get(n_x, n_y);
			if(g->open) {
				ret.push_back(all_neighbors[i]);
			}
		}
		return ret;
	}

	void gen_grid_iterate() {
		std::vector<std::array<int, 2>> cells_to_open;
		for(int j = 0; j < D; j++) {
			for(int i = 0; i < D; i++) {

			}
		}
	}
	void gen_grid() {
		// Create the backbone of the grid
		GridAttrib init_attrib = GridAttrib{
			0, 0, 0, -1, 0
		};
		for(int j = 0; j < D; j++) {
			for(int i = 0; i < D; i++) {
				grid.push_back(init_attrib);
			}
		}
		//Randomly open a cell
		std::array<int, 2> rand_ind = {std::rand() % D, std::rand() % D};
		GridAttrib* g = get(rand_ind[0], rand_ind[1]);
		g->open = 1;

	}
};


int main(void) {
	std::srand(std::time(nullptr));

}
