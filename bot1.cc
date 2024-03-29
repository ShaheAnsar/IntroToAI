#include <iostream>
#include <vector>

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
		// Create the backbone of the grid
		GridAttrib init_attrib = GridAttrib{
			0, 0, 0, -1, 0
		};
		for(int j = 0; j < D; j++) {
			for(int i = 0; i < D; i++) {
				grid.push_back(init_attrib);
			}
		}
	}
	int valid_index(int x, int y){
		return ((x >= 0) && (x < D) && (y >= 0) && (y < D));
	}
	void gen_grid()
};


int main(void) {

}
