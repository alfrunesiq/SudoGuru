#include "extractools.hpp"

static cv::Mat lbl, stats, centroids;
std::vector<cv::Vec2f> lines;

/**
 * @brief  find biggest connected component whitin segmented image
 * @param  binary image
 * @return new binary image with the component segmented out
 */
cv::Mat biggestConnectedComponents(cv::Mat binary_image) {

    int num_comp = cv::connectedComponentsWithStats(binary_image, lbl,
                                                    stats, centroids,
                                                    4, CV_16U);
    int area, max_area_idx = 0,
        max_area = 0;
    unsigned char gray_level;
    for (int i = 1; i < num_comp; i++) {
        area = stats.at<int>(i, cv::CC_STAT_AREA);
        gray_level =
         binary_image.at<unsigned char>(stats.at<int>(i, cv::CC_STAT_TOP),
                                        stats.at<int>(i, cv::CC_STAT_LEFT));
        if ((max_area < area) && (gray_level == 0)) {
            max_area = area;
            max_area_idx = i;
        }
    }

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
         */
    //////////////////////////////
    // This needs a closer look //
    //////////////////////////////
    /* move to function in extract_board */
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
    if (lines.size() > 0) {
        edges.push_back(lines[0]);
        edges.push_back({0, lines[0][1] + 
                        static_cast<float>(CV_PI)/2.0f});
        edges.push_back(lines[0]);
        edges.push_back({2.0f*img_thr.rows, lines[1][1] + 
                        static_cast<float>(CV_PI)/2.0f});
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

        for (int i = 0; i < static_cast<int>(lines.size()); i++) {
            float rho = cv::abs(lines[i][0]),
                theta = lines[i][1];
            if (( (theta >  theta_intvl[0]) && (theta <= theta_intvl[1]) ) ||
                ( (theta >= theta_intvl[2]) && (theta <  theta_intvl[3]) )) {
                if (rho > cv::abs(edges[0][0])) {
                    edges[0] = lines[i];
                } else if (rho < cv::abs(edges[2][0])) {
                    edges[2] = lines[i];
                }
            } else {
                if (rho > cv::abs(edges[2][0])) {
                    edges[1] = lines[i];
                } else if (rho < cv::abs(edges[3][0])) {
                    edges[3] = lines[i];
                }
            }
        }
    }
    // maybe substitute with comment below?

    return edges;
}

/*
            for (int i = 0; i < lines.size(); i++) {
                float rho = lines[i][0],
                    theta = lines[i][1];
                if (( (theta >  theta_intvl[0]) && (theta <= theta_intvl[1]) ) ||
                    ( (theta >= theta_intvl[2]) && (theta <  theta_intvl[3]) )) {
                    if (rho < 0) {
                        rho = -rho;
                        theta += CV_PI;
                    }

                    float theta0 = edges[0][0] < 0 ? edges[0][1] + CV_PI : edges[0][1];
                    float theta2 = edges[2][0] < 0 ? edges[2][1] + CV_PI : edges[2][1];
                    float rho2 = std::max(std::abs(edges[0][0]),
                                          std::abs(edges[2][0]));
                    float rho1 = std::min(std::abs(edges[0][0]),
                                          std::abs(edges[2][0]));
                    float alph = std::abs(theta2 - theta0);
                    float l02 = rho2*(1 + 1/std::cos(alph)) -
                                rho1*(std::cos(alph)+1/(std::cos(alph)*std::cos(alph)));
                    rho2 = std::max(rho, std::abs(edges[2][0]));
                    rho1 = std::min(rho, std::abs(edges[2][0]));
                    float l_2 = rho2*(1 + 1/std::cos(alph)) -
                                rho1*(std::cos(alph)+1/(std::cos(alph)*std::cos(alph)));
                    rho2 = std::max(rho, std::abs(edges[0][0]));
                    rho1 = std::min(rho, std::abs(edges[0][0]));
                    float l_0 = rho2*(1 + 1/std::cos(alph)) -
                                rho1*(std::cos(alph)+1/(std::cos(alph)*std::cos(alph)));
                    if ( l_2 > l02 && l_2 > l_0) {
                        edges[0] = lines[i];
                    } else if ( l_0 > l02 && l_0 > l_2) {
                        edges[2] = lines[i];
                    }
               } else {
                    if (rho < 0) {
                        rho = -rho;
                        theta += CV_PI;
                    }
                    float theta1 = edges[1][0] < 0 ? edges[1][1] + CV_PI : edges[1][1];
                    float theta3 = edges[3][0] < 0 ? edges[3][1] + CV_PI : edges[3][1];
                    float rho2 = std::max(std::abs(edges[3][0]),
                                          std::abs(edges[1][0]));
                    float rho1 = std::min(std::abs(edges[3][0]),
                                          std::abs(edges[1][0]));
                    float alph = std::abs(theta3 - theta1);
                    float l13 = rho2*(1 + 1/std::cos(alph)) -
                                rho1*(std::cos(alph)+1/(std::cos(alph)*std::cos(alph)));
                    rho2 = std::max(rho, std::abs(edges[3][0]));
                    rho1 = std::min(rho, std::abs(edges[3][0]));
                    float l_3 = rho2*(1 + 1/std::cos(alph)) -
                                rho1*(std::cos(alph)+1/(std::cos(alph)*std::cos(alph)));
                    rho2 = std::max(rho, std::abs(edges[1][0]));
                    rho1 = std::min(rho, std::abs(edges[1][0]));
                    float l_1 = rho2*(1 + 1/std::cos(alph)) -
                                rho1*(std::cos(alph)+1/(std::cos(alph)*std::cos(alph)));


                    if ( l_3 > l13 && l_3 > l_1) {
                        edges[1] = lines[i];
                    } else if ( l_1 > l13 && l_1 > l_3) {
                        edges[3] = lines[i];
                    }
                }
           }
*/