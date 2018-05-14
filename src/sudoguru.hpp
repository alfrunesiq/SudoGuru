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

struct CameraModel
{
    bool isSet;
    cv::Matx33d K;
    cv::Vec<double, 5> dist_coeffs;
};

int sudoguru(int argc, char **argv);

static cv::Point2f transformPoint(cv::Point2f pt, cv::Mat H);

static bool boardOutsideFrame(std::vector<cv::Point2f> corners,
                              cv::Mat H, cv::Size frame_size);

#endif /* _SUDOGURU_HPP_ */
