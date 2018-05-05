#ifndef _HOMOGRAPHY_ESTIMATOR_H_
#define _HOMOGRAPHY_ESTIMATOR_H_

#define MIN_NUM_KEYPTS     50
#define MAX_NUM_KEYPTS     950
#define MAX_DISTANCE_RATIO 0.85f
#define MIN_NUM_INLIERS    8

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>


class HomographyEstimator
{
public:
    HomographyEstimator();

    /**
     * @brief Estimates the homography from current set reference frame to @frame
     * @param frame  undistorted image (src image)
     * @return  relative homography; or empty mat if too few inliers
     */
    cv::Mat estimateHomography(cv::Mat frame);

    /**
     * @brief sets new reference frame for relative homography estimation
     * @param frame  reference frame
     */
    void setReferenceFrame (cv::Mat frame);


private:
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> desc_extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    // features: features from reference frame
    cv::Mat feature_desc;
    std::vector<cv::Point2f> feature_points;

};

#endif /* _HOMOGRAPHY_ESTIMATOR_H_ */
