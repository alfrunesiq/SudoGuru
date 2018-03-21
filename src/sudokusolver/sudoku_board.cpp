#include <vector>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>

#include "sudoku_brd_items.hpp"
#include "sudoku_board.hpp"

#define MAX_SOLVE_ITER 10000000

bool ticket_group::contains(int val) {
    for (Ticket *t : members) {
        if (val == t->get_value()) {
            return true;
        }
    }
    return false;
}

void ticket_group::add_member(Ticket *t) {
    members.push_back(t);
}

void Ticket::set_value(int val) {
    if (!fixed) {
        value = val;
    }
}

void Ticket::get_possible_values(std::vector<int> &ret)
{
    if (fixed) {
        ret.push_back(value);
        return;
    }
    for (int i = 9; i >= 1; i--) {
        if (!row->contains(i) &&
            !col->contains(i) &&
            !box->contains(i)) {
            ret.push_back(i);
        }
    }
}

Board::Board(int **grid)
{
    for (int i = 0; i < 9; i++) {
        rows.push_back(Row());
        cols.push_back(Column());
        board.push_back(std::vector<Ticket *>());
    }
    for (int i = 0; i < 3; i++) {
        std::vector<Box> tmp;
        for (int j = 0; j < 3; j++) {
            tmp.push_back(Box());
        }
        boxes.push_back(tmp);
    }

    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            bool fixed = grid[i][j] == 0 ? false : true;
            Ticket *t = new Ticket(grid[i][j], 
                            &rows.at(i), &cols.at(j), 
                            &boxes.at(i/3).at(j/3), 
                            fixed);
            board.at(i).push_back(t);
            rows.at(i).add_member(t);
            cols.at(j).add_member(t);
            boxes.at(i/3).at(j/3).add_member(t);
        }
    }
}

/**
 * Board::solve
 * solves the containting sudoku and returns 
 * a c-type 9x9 array with the solution
 * the method is brute-force, and selects the first
 * solution it comes across. It is not fool-proof, so
 * a maximum number of iterations is defined. 
 * @return solution (9x9 array)
 */
int **Board::solve()
{
    int i = 0, j = 0;
    long count = 0;
    bool forward = true;
    std::vector<int> possible;

    while(++count < MAX_SOLVE_ITER) {
        possible.clear();
        if(!board.at(i).at(j)->is_fixed()) {
            board.at(i).at(j)->get_possible_values(possible);
            int ticket_val = board.at(i).at(j)->get_value();
            int num_possible = static_cast<int>(possible.size());

            for (int k = 0; k < num_possible; k++) {
                if (ticket_val < possible.at(k)) {
                    board.at(i).at(j)->set_value(possible.at(k));
                }
            }
            if (board.at(i).at(j)->get_value() == 0){
                forward = false;
            } else if (board.at(i).at(j)->get_value() == ticket_val &&
                    forward == false){
                board.at(i).at(j)->set_value(0);
            } else {
                forward = true;
            }
        }

        if (i == 8 && j == 8) {
            break;
        }
        if (forward) {
            i++;
            if (i >=9){
                j++;
                i=0;
            }
        } else {
            i--;
            if (i < 0) {
                i = 8;
                if (j == 0) {
                    return NULL;
                }
                j--;
            }
        }
    }
    int **ret = (int**) malloc(9*sizeof(int*));

    for (int i = 0; i < 9; i++) {
        ret[i] = (int *) malloc(9*sizeof(int));
        for (int j = 0; j < 9; j++){
            ret[i][j] = board.at(i).at(j)->get_value();
        }
    }
    return ret;
}
