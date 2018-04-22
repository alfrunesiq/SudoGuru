#ifndef _SUDOGURU_HPP_
#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "extractools.hpp"

#ifdef DEBUG
  #include <chrono>
  #include <thread>
#endif

#ifndef CAMERA_ID_0
  #define CAMERA_ID_0 2
#endif

#define BOARDSIZE    9*34 // roughly 32x32 per box

int sudoguru(void);

#endif /* _SUDOGURU_HPP_ */
