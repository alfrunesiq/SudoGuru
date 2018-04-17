#include "sudoguru.hpp"
#include "extract_grid.hpp"
#include <chrono>
#include <thread>

#ifndef CAMERA_ID_0
    #define CAMERA_ID_0 1
#endif

int sudoguru (void)
{
#ifdef DEBUG 
    cv::namedWindow("cam0");
#endif
    cv::VideoCapture cap;
    if (!cap.open(CAMERA_ID_0)) {
        std::cerr << "sudoguru.cpp: Could not open camera " << CAMERA_ID_0 << "\n";
        return EXIT_FAILURE;
    }
    
    int height = static_cast<int> (cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int width = static_cast<int> (cap.get(cv::CAP_PROP_FRAME_WIDTH));

    cv::Mat frame(height, width, CV_8UC3);
    cv::Mat frame_gray, frame_bin, canny;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4.0);
    for (;;)
    {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "sudoguru.cpp: Frame empty\n";
            return EXIT_FAILURE;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        cv::GaussianBlur(frame, frame, cv::Size(5,5), 1);
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::cornerMinEigenVal(frame_gray, frame_gray, 5);
        double min, max;
        cv::minMaxLoc(frame_gray, &min, &max);
        std::cout << max << "  " << min << std::endl;
        frame_gray = (frame_gray - static_cast<float>(min))/static_cast<float>(max-min);
        cv::threshold(frame_gray, frame_gray, 0.75, 1.0, cv::THRESH_BINARY);
        cv::imshow("gray", frame_gray);  

        cv::dilate(frame_gray, canny, cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, 
            cv::Size(9,9)));
        cv::Mat tmp = 255.0f*(frame_gray == canny).mul(frame_gray > 0.0f);
        tmp.convertTo(frame_bin, CV_8UC1);
        std::cout << frame_bin.type();
        std::vector<cv::Point> pts;
        cv::findNonZero(frame_bin, pts);
        cv::Mat lines;
        std::vector<cv::Point3d> lines3d;
        cv::HoughLinesPointSet(pts, lines, 50, 1, 0.0f, static_cast<float>(cv::sqrt(height*width))/2.0f, 
                                1.0, 0.0f, CV_PI, CV_PI/180.0f);
        //cv::HoughLines(frame_bin, lines, 1, CV_PI/180, 8, 0, 0 );
        lines.copyTo(lines3d);
        std::cout << lines3d.size();
  for( int i = 0; i < lines.size().width; i++ )
  {
     double rho = lines.at<double>(i, 1), theta = lines.at<double>(i, 0);
     cv::Point pt1, pt2;
     double a = std::cos(theta), b = std::sin(theta);
     double x0 = a*rho, y0 = b*rho;
     pt1.x = cvRound(x0 + 1000*(-b));
     pt1.y = cvRound(y0 + 1000*(a));
     pt2.x = cvRound(x0 - 1000*(-b));
     pt2.y = cvRound(y0 - 1000*(a));
     cv::line( frame, pt1, pt2, cv::Scalar(0,0,255), 3, CV_AA);
  }
/*
        cv::Matx33d H = extract_grid(frame);
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        clahe->apply(frame_gray, frame_gray);
        cv::GaussianBlur(frame_gray, frame_gray, cv::Size(9,9), 1.5);
        cv::adaptiveThreshold(frame_gray, frame_bin, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 7, 5.0);
        cv::Canny(frame_gray, canny, 64.0, 132.0);
//      cv::getDerivKernels(canny, frame, 5, 5, 5);
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(frame_bin, lines, 1, CV_PI/180, 50, 50, 10 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        cv::Vec4i l = lines[i];
        cv::line( frame, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 1, cv::LINE_AA);
    }
        std::vector<std::vector<cv::Point>> contours; 
         std::vector<cv::Vec4i> hierarchy;
         cv::findContours(frame_bin, contours, hierarchy, CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE);

double area; double maxarea = 0;int p;
  for (size_t i = 0; i < contours.size(); i++)
  {
    area = contourArea(contours[i], false);
    if (area > maxarea)
    {
      maxarea = area;
      p = i;
    }
   }
   if (p < static_cast<int>(contours.size())) {
       cv::drawContours(frame, contours, p, cv::Scalar(0,255,0));
}*/

#ifdef DEBUG
        cv::imshow("cam0", frame);
        cv::imshow("Canny", frame_bin);    
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