#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "sudoku_board.hpp"

int main(void) {
    int **brd;
    brd = (int **) malloc(9*sizeof(int*));
    for (int i = 0; i < 9; i++) {
        brd[i] = (int *) calloc(9, sizeof(int));
    }
    brd[2][2] = 1;
    brd[2][4] = 2;
    brd[1][5] = 3;
    brd[1][7] = 8;
    brd[1][8] = 5;
    brd[3][3] = 5;
    brd[3][5] = 7;
    brd[4][2] = 4;
    brd[5][1] = 9;
    brd[5][6] = 1;

    brd[8][6] = 1;
    brd[6][0] = 5;
    brd[7][2] = 2;
    brd[7][4] = 1;
    brd[8][4] = 4;
    brd[6][7] = 7;
    brd[7][8] = 3;
    brd[8][8] = 0;
    for (int i = 0; i < 9; i++) {
        if(!(i % 3)) { printf("%22s\n", 
            "+-----------------------+");}
        for (int j = 0; j < 9; j++) {
            if(!(j % 3)) { printf("| "); }
            printf("%d ", brd[i][j]);
   
        }
        printf("|\n");
    }
    printf("%22s\n", 
            "+-----------------------+");
    printf("\n\n");

    Board board(brd);
    printf("solving...\n");
    int **solution = board.solve();

    if (solution == NULL) {
        return -1; 
    }
    for (int i = 0; i < 9; i++) {
        if(!(i % 3)) { printf("%22s\n", 
            "+-----------------------+");}
        for (int j = 0; j < 9; j++) {
            if(!(j % 3)) { printf("| "); }
            printf("%d ", solution[i][j]);
   
        }
        printf("|\n");
    }
    printf("%22s\n", 
            "+-----------------------+");
    free(solution);
}
