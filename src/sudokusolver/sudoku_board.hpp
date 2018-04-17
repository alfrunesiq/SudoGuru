#ifndef _SUDOKU_BOARD_
#define _SUDOKU_BOARD_
#include <vector>

class Ticket;

class ticket_group
{
public:
    ticket_group() {}
    ~ticket_group() {}
    
    void add_member(Ticket *t);
    bool contains(int val);

private:
    std::vector<Ticket *> members;
};

class Box : public ticket_group
{
public:
    Box(){ }
    ~Box() { }
};

class Row : public ticket_group
{
public:
    Row(){ }
    ~Row(){ }
};

class Column : public ticket_group
{
public:
    Column(){ }
    ~Column(){ }
};

class Ticket
{
public:
    Ticket(int val, Row *r, Column *c, 
            Box *b, bool is_fixed):
        fixed (is_fixed),
        value (val),
        box (b),
        row (r),
        col (c)
         {}

    ~Ticket(){ }

    void get_possible_values(std::vector<int> &p);

    int get_value() {return value;}

    void set_value(int value);

    bool is_fixed() {return fixed;}
    
private:
    bool    fixed; // is the ticket fixed
    int     value; // the ticket value (0 if unassigned)
    Box     *box;  // reference to it's box
    Row     *row;  // reference to it's row
    Column  *col;  // reference to it's column
};


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

private: 
    Ticket *getFirstNonFixed();
};

#endif /* _SUDOKU_BOARD_ */