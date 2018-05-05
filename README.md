# SudoGuru
This project implements a realtime Sudoku solver using OpenCV library. The project is my response to the computer vision assignment at the University of Oslo. 

## Program flow 
Below is an early drawing of the program flow. The actual program is slightly diffrent; mostly due to the use of a CNN classifier (instead of brute-force crosscorrelation as I initially planned).
![alt text](./doc/img/state-machine.png "State machine drawing")

## Instructions
1.  This program depends on OpenCV 3.3.0 or later (I use 3.4.1); so head over to [opencv](https://github.com/opencv/opencv/) github and follow their instructions. 
2.  Build this project using CMake. Make sure you are in `src` directory when you build with cmake (i.e. create a `build` directory within `src`) as I have assumed that you are in src directory when I load the tensorflow graph definition `pb` file.
3.  (optional) In order to make the feature matching as best as possible, you need to provide an xml file containing camera calibration parameters (camera matrix, and distortion coefficients). To generate a calibration file, use opencv's interactive calibration tool:
```
opencv_interactive-calibration -ci=0 -t=chessboard -sz=30 -w=8 -h=5 –pf=cameraParameters.xml
```
In order to run this tool, you need to print the [chessboard](https://github.com/opencv/opencv/blob/master/samples/data/chessboard.png) (you can find it under /path/to/your/opencv/samples/data/chessboard.png) and tape it to a rigid flat surface.
Put this file in the parent directory of your build directory.
