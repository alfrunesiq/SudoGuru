#include "extractools.hpp"
#include "sudoguru.hpp"
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <chrono>

#ifdef DEBUG
// Colormap used for showing detected digits
static std::vector<cv::Scalar> map = {
    CV_RGB(142, 69, 19),   // 1
    CV_RGB(255, 0, 0),     // 2
    CV_RGB(255, 150, 0),   // 3
    CV_RGB(255, 255, 0),   // 4
    CV_RGB(0, 255, 0),     // 5
    CV_RGB(0, 0, 255),     // 6
    CV_RGB(233, 16, 238),  // 7
    CV_RGB(132, 132, 132), // 8
    CV_RGB(255, 255, 255), // 9
    CV_RGB(255, 0, 255)
};
#endif

#define SEARCH_MARGIN    2.0f // used to relax constraint in exrtact edges

static cv::Mat lbl, stats, centroids;

/**
 * @brief  find biggest connected component within segmented image
 *         used for board detection
 * @param  binary image
 * @return new binary image with the component segmented out
 */
cv::Mat biggestConnectedComponents(cv::Mat binary_image)
{
    int num_comp = cv::connectedComponentsWithStats(binary_image, lbl,
                                                    stats, centroids,
                                                    4, CV_16U);
    int area, max_area_idx = -1, max_area = 0,
        max_allowable_area = binary_image.rows*binary_image.cols/4;
    unsigned char gray_level;
    for (int i = 1; i < num_comp; i++) {
        area = stats.at<int>(i, cv::CC_STAT_AREA);
        gray_level =
         binary_image.at<unsigned char>(stats.at<int>(i, cv::CC_STAT_TOP),
                                        stats.at<int>(i, cv::CC_STAT_LEFT));
        if ((max_area < area) && (gray_level == 0) &&
            (area < max_allowable_area)) {
            max_area = area;
            max_area_idx = i;
        }
    }

    // Paint contour in new image
    cv::Mat cc = cv::Mat::zeros(binary_image.size(), CV_8UC1);
    for (int i = 0; i < cc.rows; i++) {
        for (int j = 0; j < cc.cols; j++) {
            if (lbl.at<unsigned short>(i, j) == max_area_idx) {
                cc.at<unsigned char>(i, j) = 255;
            }
        }
    }

    return cc;
}

/**
 * @brief bundles together close lines by "averaging"
 * @param hough        hough lines (rho, theta)
 * @param thresh_rho   threshold on rho for which lines are grouped together
 * @param thresh_theta threshold on theta for which lines are grouped
 * @return  a trimmed set of hough lines
 */
std::vector<cv::Vec2f> bundleHough (std::vector<cv::Vec2f> hough,
                                           float thresh_rho, float thresh_theta)
{
    std::vector<std::vector<cv::Vec2f>> acc;
    std::vector<cv::Vec2f> ret;

    for (cv::Vec2f line : hough) {
        float rho = line[0],
            theta = line[1];
        bool isSet = false;
        for (size_t i = 0 ; i < acc.size() ; i++) {
            if ( ((cv::abs(theta - acc[i][0][1]) < thresh_theta) ||
                  cv::abs(theta - acc[i][0][1] - CV_PI) < thresh_theta) &&
                 (cv::abs(rho - acc[i][0][0]) < thresh_rho) ) {
                acc[i].push_back(line);
                isSet = true;
            }
        }
        if (!isSet) {
            acc.push_back(std::vector<cv::Vec2f>());
            acc.back().push_back(line);
        }
    }
    for (std::vector<cv::Vec2f> par : acc) {
        // not completely true naming; I am merely
        // averaging rho, but keeping rho from the
        // first (which defined threshold ref. above)
        cv::Vec2f median = par[0];
        float mean = 0.0f;
        for (size_t i = 0; i < par.size(); i++) {
            mean += cv::abs(par[i][0]);
        }
        mean /= static_cast<float>(par.size());
        median[0] = median[0] < 0 ? -mean : mean;
        ret.push_back(median);
    }

    return ret;
}

/**
 * @brief finds the four "most likely" edges of the segmented component
 * @param img_thr  thresholded image (biggest connected component)
 * @return vector of four edges in (rho, theta) coordinates
 */
std::vector<cv::Vec2f> Extractor::extractEdges (cv::Mat img_thr)
{
    std::vector<cv::Vec2f> lines;
    std::vector<cv::Vec2f> edges;
    cv::HoughLines(img_thr, lines, 1, CV_PI/180, HOUGH_THRESHOLD);
    /*
      edges indexing; might be mirrored / rotated arbitrarily in practice
      but the relative neighbours is always preserved
                0
            +------+
            |      |
          3 |      | 1
            |      |
            +------+
                2
       (might be mirrored along any of the axes, but extractCorners takes care of it)
         */
    /* intervall division:
                   \      /
                    \PI/2/
                     \  /
                 PI/4 \/ PI/4
           ----------------------------
           theta \in [0, PI]
           circularly wrap for the cases when first line is in
           the PI/4 part.
    */
    if (lines.size() > 3) {
        cv::Vec4f theta_intvl;
        edges.push_back(lines[0]);
        edges.push_back(cv::Vec2f());
        edges.push_back(lines[0]);
        edges.push_back(cv::Vec2f());
        // use first line to define orientation of line 0 and 2
        // and define intervalls that define relative "orthogonality"
        if (edges[0][1] < CV_PI/4.0f) {
            theta_intvl[0] = edges[0][1] + 3.0f*CV_PI/4.0f;
            theta_intvl[1] = CV_PI;
            theta_intvl[2] = 0.0f;
            theta_intvl[3] = edges[0][1] + CV_PI/4.0f;
        } else if (edges[0][1] > 3.0f*CV_PI/4.0f) {
            theta_intvl[0] = edges[0][1] - CV_PI/4.0f;
            theta_intvl[1] = CV_PI;
            theta_intvl[2] = 0.0f;
            theta_intvl[3] = edges[0][1] - 3.0f*CV_PI/4.0f;
        } else {
            theta_intvl[0] = edges[0][1] - CV_PI/4;
            theta_intvl[1] = edges[0][1] + CV_PI/4;
            theta_intvl[2] = theta_intvl[0];
            theta_intvl[3] = theta_intvl[1];
        }
        cv::Point intrsct1 = cv::Point(0,0);
        cv::Point intrsct2 = cv::Point(0,0);
        cv::Point intrsct;
        bool _13notSet = true;
        // find the set of lines that is farthest apart
        // current distance measure: magnitude of vanishing point (within a margin)
        // and relative difference of rho.
        for (int i = 1; i < static_cast<int>(lines.size()); i++) {
            float rho = cv::abs(lines[i][0]),
                theta = lines[i][1];
            if (( (theta >  theta_intvl[0]) && (theta <= theta_intvl[1]) ) ||
                ( (theta >= theta_intvl[2]) && (theta <  theta_intvl[3]) )) {
                float rho0 = cv::abs(edges[0][0]),
                      rho2 = cv::abs(edges[2][0]);
                intrsct = getLineIntersection(edges[2], lines[i]);
                // vanishing point further away than current config?
                // (rejects obvious detected non-parallel lines)
                // and if so is the lines further apart in terms of rho?
                if ((intrsct.dot(intrsct) + SEARCH_MARGIN > intrsct1.dot(intrsct1)) &&
                    (cv::abs(rho-rho2) > cv::abs(rho2-rho0)) ) {
                    edges[0] = lines[i];
                    intrsct1 = getLineIntersection(edges[0], edges[2]);
                    continue;
                }
                intrsct = getLineIntersection(edges[0], lines[i]);
                if ((intrsct.dot(intrsct) + SEARCH_MARGIN > intrsct1.dot(intrsct1)) &&
                    (cv::abs(rho-rho0) > cv::abs(rho2-rho0)) ) {
                    edges[2] = lines[i];
                    intrsct1 = getLineIntersection(edges[0], edges[2]);
                }
            } else { // line 1 or 3
                if (_13notSet) {
                    edges[1] = lines[i];
                    edges[3] = lines[i];
                    _13notSet = false;
                    continue;
                }
                intrsct = getLineIntersection(edges[3], lines[i]);
                float rho1 = cv::abs(edges[1][0]),
                      rho3 = cv::abs(edges[3][0]);
                if ((intrsct.dot(intrsct) + SEARCH_MARGIN > intrsct2.dot(intrsct2)) &&
                    (cv::abs(rho-rho3) > cv::abs(rho3-rho1)) ) {
                    edges[1] = lines[i];
                    intrsct2 = getLineIntersection(edges[1], edges[3]);
                    continue;
                }
                intrsct = getLineIntersection(edges[1], lines[i]);
                if ((intrsct.dot(intrsct) + SEARCH_MARGIN > intrsct2.dot(intrsct2)) &&
                    (cv::abs(rho-rho1) > cv::abs(rho1-rho3)) ) {
                    edges[3] = lines[i];
                    intrsct2 = getLineIntersection(edges[1], edges[3]);
                }
            }
        }
    }
    return edges;
}

/**
 * @brief extracts the four corners from the four lines in edges
 * @param edges  vector containing four (rho, theta) edges
 * @return  vector of the four corner-points
 */
std::vector<cv::Point2f> Extractor::extractCorners (std::vector<cv::Vec2f> edges)
{
    std::vector<cv::Point2f> pts;
    std::vector<cv::Point2f> ret;
    cv::Point2f mean;
    int min = 0;

    pts.push_back(getLineIntersection(edges[0], edges[1]));
    pts.push_back(getLineIntersection(edges[1], edges[2]));
    pts.push_back(getLineIntersection(edges[2], edges[3]));
    pts.push_back(getLineIntersection(edges[3], edges[0]));

    mean = (pts[0] + pts[1] + pts[2] + pts[3]) / 4;
    for (int i = 1; i < 4; i++) {
        if (cv::norm(pts[min]) > cv::norm(pts[i])) {
            min = i;
        }
    }
    // TODO: Make detection rotation invariant
    ret.push_back(pts[min]);
    pts.erase(pts.begin() + min);
    /*
    ret.push_back(pts.back());
    pts.pop_back();*/
    // orient points in positive angular direction
    for (int i = 1; i < 4; i++) {
        cv::Point2f e = ret.back() - mean;
        cv::Point2f next = pts[0];
        int idx = 0;
        e /= cv::norm(e);
        float theta_min = 360.0;
        for (size_t j = 0; j < pts.size(); j++) {
            cv::Point2f pt_ = pts[j] - mean;
            pt_ /= cv::norm(pt_);
            float theta = std::atan2(e.cross(pt_), e.dot(pt_));
            theta = theta < 0 ? theta + 2*CV_PI : theta;
            if (theta < theta_min) {
                next = pts[j];
                theta_min = theta;
                idx = j;
            }
        }
        ret.push_back(next);
        pts.erase(pts.begin()+idx);
    }

    return ret;
}

/**
 *  @brief simple comparison of rho values used for std::sort
 */
static bool compRho(cv::Vec2f l1, cv::Vec2f l2) {
    return (l1[0] < l2[0]);
}

/**
 *  @brief helper function to get boxes around digits
 *
 *  based on the idea used in finding the board
 *  @param board_thr  thresholded image of segmended board
 *
 *  @return           vector of rectangle boxes of detected digits
 *                    relative to board coordinates
 */
std::vector<cv::Rect> Extractor::findDigits(cv::Mat board_thr)
{
    std::vector<cv::Rect> digits;
    cv::Rect boundingBox;
    cv::Mat labels, stats, centroids, tmp;
    cv::Rect brd_rect = cv::Rect(0, 0, board_thr.cols, board_thr.rows);
    int ncomponents;

    // close in possibly fragmented numbers
    cv::dilate(board_thr, tmp, _3x3cross);

    // find boundingboxes around connected components
    ncomponents = cv::connectedComponentsWithStats(tmp, labels, stats,
                                     centroids, 8, CV_16U);

    for (int i = 0; i < ncomponents; i++) {
        boundingBox = cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT),
                               stats.at<int>(i, cv::CC_STAT_TOP),
                               stats.at<int>(i, cv::CC_STAT_WIDTH),
                               stats.at<int>(i, cv::CC_STAT_HEIGHT));
        // threshold away irrelevant boxes
        if (boundingBox.area() < DIGIT_MIN_AREA ||
            boundingBox.area() > DIGIT_MAX_AREA) {
            continue;
        } else if(boundingBox.height < DIGIT_HEIGHT_MIN ||
                  boundingBox.height > DIGIT_HEIGHT_MAX ||
                  boundingBox.width  < DIGIT_WIDTH_MIN  ||
                  boundingBox.width  > DIGIT_WIDTH_MAX) {
            continue;
        } else {
            // add border around box
            boundingBox.width  += 10;
            boundingBox.height += 6;
            boundingBox.x      -= 5;
            boundingBox.y      -= 3;
            boundingBox &= brd_rect;
            digits.push_back(boundingBox);
        }
    }
    /*
    for (size_t i = 0; i < digits.size(); i++) {
        for (size_t j = 0; j < digits.size(); j++) {
            if ((digits[i] & digits[j]).area() != 0) {
                if (digits[i].area() < digits[j].area()) {
                    digits.erase(digits.begin() + i);
                } else {
                    digits.erase(digits.begin() + j);
                }
            }
        }
    }*/

    return digits;
}


static void interpolateLines(std::vector<cv::Vec2f> *lines)
{
    for (size_t i = 1; i < (*lines).size(); i++) {
        float diff = cv::abs(cv::abs((*lines)[i][0]) -
                             cv::abs((*lines)[i-1][0]));
        if (diff < GRID_GAP_MIN) {
            // gap too tight: average and erase
            (*lines)[i-1][0] = ((*lines)[i-1][0] + (*lines)[i][0]) / 2.0f;
            (*lines).erase((*lines).begin() + i--);
        } else if (diff > 2*GRID_GAP_MIN && diff < 2*GRID_GAP_MAX) {
            // one missing line
            (*lines).insert((*lines).begin() + i, (*lines)[i]);
            (*lines)[i][0] -= diff/2.0f;
            i++;
        } else if (diff > 2*GRID_GAP_MAX && diff < 3*GRID_GAP_MAX) {
            // two missing lines
            diff = diff/3.0f;
            (*lines).insert((*lines).begin() + i, (*lines)[i-1]);
            (*lines)[i-1][0] += diff;
            i++;
            (*lines).insert((*lines).begin() + i, (*lines)[i]);
            (*lines)[i+1][0] -= diff;
            i++;
        }
    }
}

/**
 * @brief  attempts to extract the 9x9 sudokugrid
 * @param board   image of segmented board
 * @return        9x9 matrix of digits.
 */
std::vector<std::vector<int>> *Extractor::extractGrid (cv::Mat board) {
    cv::Mat buf, bufHough, board_;
    cv::Mat grad, gradY;
    std::vector<cv::Vec2f> lines;
    std::vector<cv::Vec2f> horizontal, vertical;
    double max, min;

    cv::cvtColor(board, board, cv::COLOR_BGR2GRAY);

    // pre-process gray-image by subtracting prewitt xy-
    // gradients, such that vertical and horizontal lines
    // get more pronounced
    board.convertTo(board_, CV_32F);
    cv::minMaxLoc(board_, &min, &max);
    board_ = (board_ - min)/(max-min);
    cv::filter2D(board_, grad, CV_32F, _prewitKrnl);
    cv::filter2D(board_, gradY, CV_32F, _prewitKrnl.t());
    grad = cv::abs(grad) + cv::abs(gradY);
    grad = 255.0f*(board_ - grad);
    grad.convertTo(bufHough, CV_8UC1);

#ifdef DEBUG
    cv::imshow("bufHough", bufHough);
#endif
    // Threshold two versions; one used for finding lines,
    // and one for finding digits
    cv::adaptiveThreshold(board, buf, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV, 19, 8.0);
    cv::adaptiveThreshold(bufHough, bufHough, 255,
                          cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY_INV, 19, 5.0);
    cv::HoughLines(bufHough, lines, 1, CV_PI/180, BOARDSIZE-50);

    // Thin lines first by removing non -vertical/-horizontal
    for (int i = static_cast<int>(lines.size()-1) ; i >= 0 ; i--) {
        float theta = (lines[i][1] > CV_PI) ? lines[i][1] - 1 :
            lines[i][1];
        if ( ((theta > CV_PI/16.0f) &&
              (theta < (CV_PI/2.0f - CV_PI/16.0f))) ||
             ((theta > (CV_PI/2.0f + CV_PI/16.0f)) &&
              (theta < (15.0f*CV_PI/16.0f)))) {
            lines.erase(lines.begin() + i);
        }
    }
    lines = bundleHough(lines, 8.0, CV_PI/16.0f);

    // sort out vertical and horizontal lines and force theta to mean
    for (cv::Vec2f line : lines) {
        if (line[1] < CV_PI/4.0f || line[1] > 3.0f*CV_PI/4.0f) {
            vertical.push_back(line);
            if (line[0] < 0) {
                vertical.back()[0] = -line[0];
                vertical.back()[1] += CV_PI;
            }
        } else {
            horizontal.push_back(line);
            if (line[0] < 0) {
                horizontal.back()[0] = -line[0];
                horizontal.back()[1] += CV_PI;
            }
        }
    }


    // SANITY CHECK 1: check if we have a minimum number of lines
    if (vertical.size() < 5 || horizontal.size() < 5) {
#ifdef DEBUG
        std::cout << "DEBUG: Insufiicient number of lines\n";
#endif
        return NULL;
    }

    // sort arrays in distance from origin
    std::sort(vertical.begin(), vertical.end(), compRho);
    std::sort(horizontal.begin(), horizontal.end(), compRho);

    // (try to) interpolate missing lines
    // check for outer lines
    if (vertical[0][0] > GRID_GAP_MIN) {
        vertical.insert(vertical.begin(), cv::Vec2f(0.0f, vertical[0][1]));
    }
    if (vertical.back()[0] < (BOARDSIZE - GRID_GAP_MIN)) {
        vertical.push_back(cv::Vec2f(BOARDSIZE, vertical.back()[1]));
    }
    if (horizontal[0][0] > GRID_GAP_MIN) {
        horizontal.insert(horizontal.begin(), cv::Vec2f(0.0f, horizontal[0][1]));
    }
    if (horizontal.back()[0] < (BOARDSIZE - GRID_GAP_MIN)) {
        horizontal.push_back(cv::Vec2f(BOARDSIZE, horizontal.back()[1]));
    }
    // interpolate intermediate lines
    interpolateLines(&horizontal);
    interpolateLines(&vertical);

#ifdef DEBUG // draw lines
    cv::Mat lineBuf = cv::Mat::zeros(bufHough.size(), CV_8UC1);
    for (cv::Vec2f line : vertical)
    {
        float y = line[0] / std::sin(line[1]),
            x = -1 / std::tan(line[1]);
        cv::line(lineBuf, cv::Point(0, y),
                 cv::Point(lineBuf.cols, lineBuf.cols*x+y), CV_RGB(0,0,255));
    }
    for (cv::Vec2f line : horizontal)
    {
        float y = line[0] / std::sin(line[1]),
            x = -1 / std::tan(line[1]);
        cv::line(lineBuf, cv::Point(0, y),
                 cv::Point(lineBuf.cols, lineBuf.cols*x+y), CV_RGB(0,0,255));
    }
    cv::imshow("Sanity lines", lineBuf);
#endif

    // SANITY CHECK 2: Did we resolve all 10 lines?
    if (vertical.size() != 10 && horizontal.size() != 10) {
#ifdef DEBUG
        std::cout << "DEBUG: extractGrid: could not resolve lines\n";
#endif
        return NULL;
    }

    // SANITY CHECK 3: does line gaps make sense?
    for (size_t i = 1; i < vertical.size(); i++) {
        float diff = cv::abs(cv::abs(vertical[i][0]) -
                             cv::abs(vertical[i-1][0]));
        if (diff > GRID_GAP_MAX || diff < GRID_GAP_MIN) {
#ifdef DEBUG
            std::cout << "DEBUG:" << diff << "Grid gaps doesn't fit model\n";
#endif
            return NULL;
        }
        diff = cv::abs(cv::abs(horizontal[i][0]) -
                       cv::abs(horizontal[i-1][0]));
        if (diff > GRID_GAP_MAX|| diff < GRID_GAP_MIN) {
#ifdef DEBUG
            std::cout << "DEBUG:" << diff << "Grid gaps doesn't fit model\n";
#endif
            return NULL;
        }
    }

    // zero out grid
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            grid[i][j] = 0;
        }
    }

    // segment out and classify digits

#ifdef DEBUG
    // show detected digits colorcoded
    cv::Mat buf_col;
    cv::cvtColor(buf, buf_col, cv::COLOR_GRAY2BGR);
#endif
    std::vector<cv::Rect> digits = findDigits(buf);
    cv::Mat croped, input_img, blob, out;
    int argmax[2];
    for (cv::Rect digit : digits) {
        croped = buf(digit);
        cv::resize(croped, input_img, cv::Size(NET_INPUT_SIZE, NET_INPUT_SIZE));
        cv::dnn::blobFromImage(input_img, blob, 1.0, cv::Size(NET_INPUT_SIZE,
                                                              NET_INPUT_SIZE));
        net.setInput(blob);
        out = net.forward();
        cv::minMaxIdx(out.reshape(1), 0, 0, 0, argmax);
        if (out.at<float>(argmax[1]) > CONFIDENCE_THRESH) {
            int x = digit.x + digit.width/2;
            int y = digit.y + digit.height/2;
            x /= GRID_GAP_AVG;
            y /= GRID_GAP_AVG;
            if (x > 8) {
                x = 8;
            }
            if (y > 8) {
                y = 8;
            }
            //* grid is indexed matrix-wise
            grid[y][x] = argmax[1]+1;
#ifdef DEBUG
            cv::rectangle(buf_col, digit, map[argmax[1]], 1);
#endif
        }
    }
#ifdef DEBUG
    cv::imshow("digits", buf_col);
#endif
    return &grid;
}


/**
 * @brief  finds the intersection between two lines in Hough description
 * @param  rt1 (rho, theta) description of line1
 * @param  rt2 (rho, theta) description of line2
 * @return     cv::Point intersection
 *             empty point if lines are parallel
 */
cv::Point2f getLineIntersection(cv::Vec2f rt1, cv::Vec2f rt2)
{
    // The projective reasoning:
    // r1 = x1 cos (theta1) + y1 sin (theta1)
    // r2 = x2 cos (theta2) + y2 sin (theta2)
    // ax + by + c = 0
    // l = [a, b, c] ; ~x = [x, y, w] ; l.T * ~x = 0
    // l1.T ~x = 0, ~x.T l2 = 0
    // l1.T ~x = l2.T ~x
    // ~x_intrsct = l2 x l1;
    // x_intersct = [xi/wi, yi/wi]
    // l_i = [cos(theta_i), sin(theta_i), r_i]
    cv::Vec3f l1 = cv::Vec3f(std::cos(rt1[1]), std::sin(rt1[1]), rt1[0]);
    cv::Vec3f l2 = cv::Vec3f(std::cos(rt2[1]), std::sin(rt2[1]), rt2[0]);

    cv::Vec3f intersect = l1.cross(l2);
    if (std::abs(intersect[2]) < 1e-8f) {
        // lines are (close to) parallel to the normal plane
        float x = intersect[0] < 0 ? -1e10f : 1e10f;
        float y = intersect[1] < 0 ? -1e10f : 1e10f;
        return cv::Point2f(x, y);
    } else {
        return cv::Point2f(std::abs(intersect[0]/intersect[2]),
                           std::abs(intersect[1]/intersect[2]));
    }
}
