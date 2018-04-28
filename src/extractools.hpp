#ifndef _EXTRACT_GRID_H_
#define _EXTRACT_GRID_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <stdio.h>
#include <unistd.h>
#include <iostream>

#define NET_INPUT_SIZE     32

// TUNABLE PARAMETERS:
#define DIGIT_MIN_AREA     64    // 8*8
#define DIGIT_MAX_AREA     1156  // 34*34
#define DIGIT_HEIGHT_MIN   8
#define DIGIT_HEIGHT_MAX   28
#define DIGIT_WIDTH_MIN    8
#define DIGIT_WIDTH_MAX    28

#define HOUGH_THRESHOLD    BOARDSIZE-100

#define GRID_GAP_AVG       34
#define GRID_GAP_MIN       28
#define GRID_GAP_MAX       40


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
class Extractor
{
public:
    Extractor() {
        char path[258] = {0};
        getcwd(path, 258);
        std::string full_path = std::string(path);
        size_t idx = full_path.find("src");
        full_path.erase(full_path.begin() + idx, full_path.end());
        full_path.append("src/digitNet/digitNet.pb");

        net = cv::dnn::readNetFromTensorflow(full_path);
        if (net.empty()) {
            printf("ERROR: Network empty: \"%s\"", full_path.c_str());
            throw std::exception();
        }
    }
    Extractor(std::string pathToTfProto) {
        net = cv::dnn::readNetFromTensorflow(pathToTfProto);
    }
    ~Extractor() {}


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

private:
    cv::Vec2f grid[9][9];
    cv::dnn::Net net;
    cv::Mat _3x3Cross = cv::getStructuringElement(cv::MORPH_CROSS,
                                                  cv::Size(3,3));
    std::vector<cv::Rect> findDigits(cv::Mat board_thr);
};

#endif /* _EXTRACT_GRID_H_ */
