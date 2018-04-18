#ifndef _EXTRACT_GRID_H_
#define _EXTRACT_GRID_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>


std::vector<cv::Vec2f> extractEdges (cv::Mat img_thr);

/**
 * @brief  find biggest connected component whitin segmented image
 * @param  binary image
 * @return new binary image with the component segmented out
 */
cv::Mat biggestConnectedComponents(cv::Mat binary_image);
#endif /* _EXTRACT_GRID_H_ */
