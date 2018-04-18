#include "sudoguru.hpp"
#include <chrono>
#include <thread>

#ifndef CAMERA_ID_0
    #define CAMERA_ID_0 1
#endif

int sudoguru (void)
{
#ifdef DEBUG
    cv::namedWindow("cam0");
    cv::namedWindow("Thresholded");
#endif
    cv::VideoCapture cap;
    if (!cap.open(CAMERA_ID_0)) {
        std::cerr << "sudoguru.cpp: Could not open camera " << CAMERA_ID_0 << "\n";
        return EXIT_FAILURE;
    }

    int height = static_cast<int> (cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int width = static_cast<int> (cap.get(cv::CAP_PROP_FRAME_WIDTH));

    cv::Mat frame(height, width, CV_8UC3);
    cv::Mat frame_buf(height, width, CV_8UC1),
            frame_bin(height, width, CV_8UC1);
    cv::Mat _3x3cross = cv::getStructuringElement(cv::MORPH_CROSS,
                                                  cv::Size(3,3)),
           _11x11circ = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                   cv::Size(11,11));
    cv::Mat buf[3];
    cv::Mat buf_comp;


    std::vector<cv::Vec2f> lines;
    for (;;)
    {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "sudoguru.cpp: Frame empty\n";
            return EXIT_FAILURE;
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        // Convert to grayscale and adaptively threshold; assuming board
        // is high contrast
        cv::cvtColor(frame, frame_buf, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(frame_buf, frame_buf, cv::Size(5,5), 0);
        cv::adaptiveThreshold(frame_buf, frame_bin, 255,
                              cv::ADAPTIVE_THRESH_MEAN_C,
                              cv::THRESH_BINARY_INV, 11, 5.0);
        cv::dilate(frame_bin, frame_bin, _3x3cross);
        frame_buf = biggestConnectedComponents(frame_bin);
        cv::erode(frame_buf, frame_buf, _3x3cross);

        /*
        cv::cornerMinEigenVal(frame_buf, buf[0], 5, 3);
        cv::cornerMinEigenVal(frame_buf, buf[1], 9, 7);
        cv::cornerMinEigenVal(frame_buf, buf[2], 15, 13);
        cv::max(buf[0], buf[1], buf[0]);
        cv::max(buf[0], buf[2], buf[0]);
        cv::dilate(buf[0], buf[1], _11x11circ);

        buf[2] = (buf[1] == buf[0]);
        buf[2].convertTo(buf[1], buf[0].type());
        buf[0] = buf[1].mul(buf[0]);
        double min, max;
        cv::minMaxLoc(buf[0], &min, &max);
        cv::threshold(buf[0], buf[0], 0.8*max, 255, cv::THRESH_BINARY);*/

        std::vector<cv::Vec2f> edges = extractEdges(frame_buf);
        if (edges.size() > 0) {
            // r1 = x1 cos (theta1) + y1 sin (theta1)
            // r2 = x2 cos (theta2) + y2 sin (theta2)
            // ax + by + c = 0
            // l = [a, b, c] ; ~x = [x, y, w] ; l.T * ~x = 0
            // l1.T ~x = 0, ~x.T l2 = 0
            // l1.T ~x = l2.T ~x
            // ~x_intrsct = l2 x l1;
            // x_intersct = [xi/wi, yi/wi]
            // l_i = [cos(theta_i), sin(theta_i), r_i]
            /* TODO: Wrap into function; cv::Point getLineIntersection(cv::Vec2f rt) */
        
            cv::Vec3f l1, l2, l3;
            std::vector<cv::Point> pts;
            l1[0] = std::cos(edges[0][1]), l1[1] = std::sin(edges[0][1]), l1[2] = edges[0][0];
            l2[0] = std::cos(edges[1][1]), l2[1] = std::sin(edges[1][1]), l2[2] = edges[1][0];
            l3 = l1.cross(l2);
            pts.push_back(cv::Point(cv::abs(l3[0]/l3[2]), cv::abs(l3[1]/l3[2])));
            l2[0] = std::cos(edges[3][1]), l2[1] = std::sin(edges[3][1]), l2[2] = edges[3][0];
            l3 = l1.cross(l2);
            pts.push_back(cv::Point(cv::abs(l3[0]/l3[2]), cv::abs(l3[1]/l3[2])));
            l1[0] = std::cos(edges[2][1]), l1[1] = std::sin(edges[2][1]), l1[2] = edges[2][0];
            l3 = l1.cross(l2);
            pts.push_back(cv::Point(cv::abs(l3[0]/l3[2]), cv::abs(l3[1]/l3[2])));

            l2[0] = std::cos(edges[1][1]), l2[1] = std::sin(edges[1][1]), l2[2] = edges[1][0];
            l3 = l1.cross(l2);
            pts.push_back(cv::Point(cv::abs(l3[0]/l3[2]), cv::abs(l3[1]/l3[2])));

#ifdef DEBUG
            cv::circle(frame, pts[0], 3, CV_RGB(0,0,255), 2);
            cv::circle(frame, pts[1], 3, CV_RGB(0,0,255), 2);
            cv::circle(frame, pts[2], 3, CV_RGB(0,0,255), 2);
            cv::circle(frame, pts[3], 3, CV_RGB(0,0,255), 2);
         // draw lines
            for (unsigned int i = 0; i < 4; i++) {
                if (edges[i][1] != 0) {
                    float y = edges[i][0] / std::sin(edges[i][1]),
                        x = -1 / std::tan(edges[i][1]);
                    cv::line(frame_buf, cv::Point(0, y),
                             cv::Point(frame_buf.cols, frame_buf.cols*x+y), CV_RGB(0,0,255));
                }
            }
#endif
        }
        /* TODO: implement DLT to estimate homography from determined points */
        /*          - sanity check ? */

#ifdef DEBUG

        cv::imshow("cam0", frame);
        cv::imshow("Thresholded", frame_bin);
        cv::imshow("CC", frame_buf);
#endif

        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                  << std::endl;
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }

    return EXIT_SUCCESS;
}
