#ifndef _EXTRACT_GRID_H_
#define _EXTRACT_GRID_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>


/**
 * @brief  extracts outermost edges in segmented plane
 * @param  img_thr  thresholded image containing plane edges
 * @return vector with (rho, theta) Hough lines
 */
std::vector<cv::Vec2f> extractEdges (cv::Mat img_thr);

/**
 * @brief  extracts corners where the four lines intersects
 * @param  edges   edge lines
 * @return intersecting points in orientation preserving order
 */
std::vector<cv::Point2f> extractCorners (std::vector<cv::Vec2f> edges);

/**
 * @brief  attempts to extract the 9x9 sudokugrid
 * TODO: As an initial attempt; try to use normalized cross correlation (template matching)
 * with (blurred) skeleton digit images. (first check if empty)
 * @param board   image of segmented board
 * @return        9x9 matrix of digits.
 */
std::vector<std::vector<int>> extractGrid (cv::Mat board);

/**
 * @brief  find biggest connected component whitin segmented image
 * TODO: Accelerated alternative: use flood fill from corner features to
 *       generate image of votes, and segment the region with most votes.
 *       (This might be worse, but worth a try)
 * @param  binary image
 * @return new binary image with the component segmented out
 */
cv::Mat biggestConnectedComponents(cv::Mat binary_image);

/**
 * @brief  finds the intersection between two lines in Hough description
 * @param  rt1 (rho, theta) description of line1
 * @param  rt2 (rho, theta) description of line2
 * @return     cv::Point intersection
 *             empty point if lines are parallel
 */
cv::Point2f getLineIntersection(cv::Vec2f rt1, cv::Vec2f rt2);

/**
 * @brief thresholds the hough transform to bundle and average together nearby lines
 * (TODO: performance improovement; this one is quite sluggish)
 * @param hough        hough lines (rho, theta)
 * @param thresh_rho   threshold wrt. rho
 * @param thresh_theta threshold wrt. theta
 */
std::vector<cv::Vec2f> bundleHough (std::vector<cv::Vec2f> hough,
                                    float thresh_rho, float thresh_theta);

#endif /* _EXTRACT_GRID_H_ */
