#include "extractools.hpp"
#include <iostream>

static cv::Mat lbl, stats, centroids;
std::vector<cv::Vec2f> lines;

/**
 * @brief  find biggest connected component whitin segmented image
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

std::vector<cv::Vec2f> bundleHough (std::vector<cv::Vec2f> hough,
                                           float thresh_rho, float thresh_theta)
{
    std::vector<std::vector<cv::Vec2f>> acc;
    std::vector<cv::Vec2f> ret;

    for (cv::Vec2f line : hough) {
        float rho = line[0],
            theta = line[1];
        bool isSet = false;
        for (int i = 0 ; i < static_cast<int>(acc.size()) ; i++) {
            if ((theta > 3.0f*CV_PI/4.0f) &&
                (acc[i][0][0] < CV_PI/4.0f)) {
                theta -= CV_PI;
                rho = -rho;
            }
            if ((cv::abs(theta - acc[i][0][1]) < thresh_theta) &&
                (cv::abs(rho - acc[i][0][0]) < thresh_rho)) {
                acc[i].push_back(line);
                isSet = true;
            }
        }
        if (!isSet) {
            std::vector<cv::Vec2f> tmp;
            tmp.push_back(line);
            acc.push_back(tmp);
        }
    }
    for (std::vector<cv::Vec2f> par : acc) {
        cv::Vec2f mean;
        for (size_t i = 0; i < par.size(); i++) {
            if (par[i][0] < 0) {
                par[i][1] = (par[i][1] > 3.0f*CV_PI/4.0f) ? par[i][1] - CV_PI :
                                                            par[i][1] + CV_PI;
                par[i][0] = -par[i][0];
            }
            mean += par[i];
        }
        mean /= static_cast<float>(par.size());
        mean[1] = (mean[1] > CV_PI) ? mean[1] - CV_PI : mean[1];
        ret.push_back(mean);
    }
    return ret;
}

std::vector<cv::Vec2f> extractEdges (cv::Mat img_thr)
{
    cv::HoughLines(img_thr, lines, 1, CV_PI/180, 100);
    /*
      edges indexing (might be rotated arbitrarily in practice):
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
    std::vector<cv::Vec2f> edges;
    if (lines.size() > 3) {
        edges.push_back(lines[0]);
        edges.push_back(cv::Vec2f());
        edges.push_back(lines[0]);
        edges.push_back(cv::Vec2f());
        cv::Vec4f theta_intvl;
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
        /* TODO: define a (more) projectively invariant distance measure */
        for (int i = 1; i < static_cast<int>(lines.size()); i++) {
            float rho = cv::abs(lines[i][0]),
                theta = lines[i][1];
            if (( (theta >  theta_intvl[0]) && (theta <= theta_intvl[1]) ) ||
                ( (theta >= theta_intvl[2]) && (theta <  theta_intvl[3]) )) {
                float rho0 = cv::abs(edges[0][0]),
                      rho2 = cv::abs(edges[2][0]);
                intrsct = getLineIntersection(edges[2], lines[i]);
                if ( ((intrsct == cv::Point(0,0))  ||
                      (intrsct.dot(intrsct) > intrsct1.dot(intrsct1))) &&
                     (cv::abs(rho-rho2) > cv::abs(rho2-rho0)) ) {
                    edges[0] = lines[i];
                    intrsct1 = getLineIntersection(edges[0], edges[2]);
                    continue;
                }
                intrsct = getLineIntersection(edges[0], lines[i]);
                if ( ((intrsct == cv::Point(0,0)) ||
                      (intrsct.dot(intrsct) > intrsct1.dot(intrsct1))) &&
                         (cv::abs(rho-rho0) > cv::abs(rho2-rho0)) ) {
                    edges[2] = lines[i];
                    intrsct1 = getLineIntersection(edges[0], edges[2]);
                }
            } else {
                if (_13notSet) {
                    edges[1] = lines[i];
                    edges[3] = lines[i];
                    _13notSet = false;
                    continue;
                }
                intrsct = getLineIntersection(edges[3], lines[i]);
                float rho1 = cv::abs(edges[1][0]),
                      rho3 = cv::abs(edges[3][0]);
                if ( ((intrsct == cv::Point(0,0)) ||
                     (intrsct.dot(intrsct) > intrsct2.dot(intrsct2)) ) &&
                     (cv::abs(rho-rho3) > cv::abs(rho3-rho1)) ) {
                    edges[1] = lines[i];
                    intrsct2 = getLineIntersection(edges[1], edges[3]);
                    continue;
                }
                intrsct = getLineIntersection(edges[1], lines[i]);
                if ( ((intrsct == cv::Point(0,0)) ||
                      (intrsct.dot(intrsct) > intrsct2.dot(intrsct2))) &&
                     (cv::abs(rho-rho1) > cv::abs(rho1-rho3)) ) {
                    edges[3] = lines[i];
                    intrsct2 = getLineIntersection(edges[1], edges[3]);
                }
            }
        }
    }
    return edges;
}

std::vector<cv::Point2f> extractCorners (std::vector<cv::Vec2f> edges)
{
    std::vector<cv::Point2f> pts;
    std::vector<cv::Point2f> ret;

    pts.push_back(getLineIntersection(edges[0], edges[1]));
    pts.push_back(getLineIntersection(edges[1], edges[2]));
    pts.push_back(getLineIntersection(edges[2], edges[3]));
    pts.push_back(getLineIntersection(edges[3], edges[0]));

    cv::Point2f mean = (pts[0] + pts[1] + pts[2] + pts[3]) / 4;
    ret.push_back(pts.back());
    pts.pop_back();
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
 * @brief  attempts to extract the 9x9 sudokugrid
 * @param board   image of segmented board
 * @return        9x9 matrix of digits.
 */
std::vector<std::vector<int>> extractGrid (cv::Mat board) {
    cv::Mat buf;
    std::vector<std::vector<int>> grid;
    cv::cvtColor(board, board, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(board, buf, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV, 11, 4.0);
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(buf, lines, 1, CV_PI/180, 150.0);
    for (int i = static_cast<int>(lines.size()-1) ; i >= 0 ; i--) {
        if (lines[i][1] > 0.14 && lines[i][1] < (CV_PI/2.0f - 0.14)) {
            lines.pop_back();
        }
    }
    lines = bundleHough(lines, 17.0, 1.0);

    int num_horizontal = 0,
        num_vertical   = 0;
    for (cv::Vec2f line : lines) {
        if (line[1] < CV_PI/4.0f || line[1] > 3.0f*CV_PI/4.0f) {
            num_horizontal++;
        } else {
            num_vertical++;
        }
    }
    // sanity check
    if (num_vertical < 5 || num_horizontal < 5) {
        return grid;
    }

    /* TODO: Finish off the actual grid extraction */
    if (num_vertical < 8) {
        // Probably found edges + bold lines, and possibly a couple
        // thin lines;
    } else {
        // If 10: found all lines + edges
        // If less: figure out what's missing
    }

    if (num_horizontal < 8) {
        // same as above
    } else {
        // yup...
    }

    // loop over all boxes and template match I guess

    return grid;
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
        return cv::Point2f();
    } else {
        return cv::Point2f(std::abs(intersect[0]/intersect[2]),
                           std::abs(intersect[1]/intersect[2]));
    }
}
