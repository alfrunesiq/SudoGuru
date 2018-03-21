#ifndef _SUDOKU_BOARD_
#define _SUDOKU_BOARD_

#include <vector>
#include "sudoku_brd_items.hpp"

class Board
{
public:
	
    std::vector<std::vector<Ticket *>> board;
    std::vector<Row> rows;
    std::vector<Column> cols;
    std::vector<std::vector<Box>> boxes;

	Board(int **grid);
    ~Board() {};
    int **solve();
};

#endif /* _SUDOKU_BOARD_ */
