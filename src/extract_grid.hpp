#ifndef _EXTRACT_GRID_H_
#define _EXTRACT_GRID_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/**
 * @brief preprocess image for grid extraction
 * @param  frame -
 * @return       processed frame
 */
static cv::Mat preprocess_image(cv::Mat frame);

/**
 * @brief extracts sudoku grid from frame
 * @param  frame possibly containing a sudokuboard
 * @return       3x3 Homography mapping board boundingbox to
 *               NxN image corners. cv::Matx33d() if none found
 */
cv::Matx33d extract_grid(cv::Mat frame);


#endif /* _EXTRACT_GRID_H_ */