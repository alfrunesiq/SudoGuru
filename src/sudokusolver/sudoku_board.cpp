#include <vector>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>

#include "sudoku_board.hpp"

#define MAX_SOLVE_ITER 1000000L

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
    value = val;
}

std::vector<int> Ticket::get_possible_values()
{
    std::vector<int> ret;
    if (fixed) {
        ret.push_back(value);
        return ret;
    }
    for (int i = 9; i >= 1; i--) {
        if (!row->contains(i) &&
            !col->contains(i) &&
            !box->contains(i)) {
            ret.push_back(i);
        }
    }
    return ret;
}

Board::Board()
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
            Ticket *t = new Ticket(0,
                            &rows.at(i), &cols.at(j),
                            &boxes.at(i/3).at(j/3),
                            false);
            board.at(i).push_back(t);
            rows.at(i).add_member(t);
            cols.at(j).add_member(t);
            boxes.at(i/3).at(j/3).add_member(t);
        }
    }
}


void Board::setBoard(std::vector<std::vector<int>> newBoard) {
    if (newBoard.size() != 9 || newBoard[0].size() != 9) {
        return;
    }
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (newBoard[i][j] == 0) {
                board[i][j]->set_value(0);
                board[i][j]->set_fixed(false);
            } else {
                board[i][j]->set_value(newBoard[i][j]);
                board[i][j]->set_fixed(true);
            }
        }
    }
}

void Board::setBoard(std::vector<std::vector<int>> *newBoard) {
    if ((*newBoard).size() != 9 || (*newBoard)[0].size() != 9) {
        return;
    }
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if ((*newBoard)[i][j] == 0) {
                board[i][j]->set_value(0);
                board[i][j]->set_fixed(false);
            } else {
                board[i][j]->set_value((*newBoard)[i][j]);
                board[i][j]->set_fixed(true);
            }
        }
    }
}


Ticket *Board::getFirstNonFixed() {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (!board[i][j]->is_fixed()) {
                return board[i][j];
            }
        }
    }
    return NULL;
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
std::vector<std::vector<int>> *Board::solve(
    std::vector<std::vector<int>> *brd)
{
    int i = 0, j = 0;
    long count = 0;
    bool forward = true;
    std::vector<int> possible;
    Ticket *firstChangeable = Board::getFirstNonFixed();

    while(++count < MAX_SOLVE_ITER) {
        if(!board[i][j]->is_fixed()) {
            possible = board[i][j]->get_possible_values();
            int ticket_val = board[i][j]->get_value();
            int num_possible = static_cast<int>(possible.size());

            for (int k = 0; k < num_possible; k++) {
                if (ticket_val < possible.at(k)) {
                    board[i][j]->set_value(possible.at(k));
                }
            }
            if (board[i][j]->get_value() == 0){
                forward = false;
            } else if (board[i][j]->get_value() == ticket_val &&
                    forward == false){
                if (board[i][j] != firstChangeable) {
                    board[i][j]->set_value(0);
                } else {
                    // if we reached here, we need to terminate
                    // else the loop will reach an oscillating state
                    return NULL;
                }
            } else {
                forward = true;
            }
        }

        if (i == 8 && j == 8 && forward == true) {
            break;
        }
        if (forward) {
            j++;
            if ( j >=9 ){
                i++;
                j=0;
            }
        } else {
            j--;
            if (j < 0) {
                j = 8;
                if (i == 0) {
                    return NULL;
                }
                i--;
            }
        }
    }
    if (count == MAX_SOLVE_ITER) {
        return NULL;
    }
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++){
            (*brd)[i][j] = board[i][j]->get_value();
        }
    }
    return brd;
}
