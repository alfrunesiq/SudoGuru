#ifndef _SUDOGURU_HPP_
#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "extractools.hpp"

#ifdef DEBUG
  #include <chrono>
  #include <thread>
#endif

#ifndef CAMERA_ID
  #define CAMERA_ID  0
#endif

#define BOARDSIZE    9*34 // roughly 32x32 per box

/**
 * Structure containing camera matrix and distortion coefficients
 * of the camera (if provided).
<*/
struct CameraModel
{
    bool isSet;
    cv::Matx33d K;
    cv::Vec<double, 5> dist_coeffs;
};

// main program function
int sudoguru(int argc, char **argv);

#endif /* _SUDOGURU_HPP_ */
