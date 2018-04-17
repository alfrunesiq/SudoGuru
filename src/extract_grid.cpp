#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/**
 * @brief preprocess image for grid extraction
 * @param  frame -
 * @return       processed frame
 */
static cv::Mat preprocess_image(cv::Mat frame) {
    return frame;
}

/**
 * @brief extracts sudoku grid from frame
 * @param  frame possibly containing a sudokuboard
 * @return       3x3 Homography mapping board boundingbox
 *               to NxN image corners. NULL if none found
 */
cv::Matx33f extract_grid(cv::Mat frame) {
    cv::Mat frame_processed = preprocess_image(frame);
    return cv::Matx33d();
}