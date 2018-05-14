#include "homographyestimator.hpp"

#ifdef DEBUG
#include <iostream>
#endif

HomographyEstimator::HomographyEstimator()
{
    // initialize detector / descriptor and matcher
    detector = cv::ORB::create(576, 1.5f, 8, 32, 0, 2, cv::ORB::FAST_SCORE, 32, 20);
    desc_extractor = detector;
    matcher = cv::BFMatcher::create(desc_extractor->defaultNorm());
}


/**
 * @brief static helper function performing ratio test
 */
static std::vector<cv::KeyPoint> ratioTest (const std::vector<cv::KeyPoint> keypts,
                                             const cv::Mat featureDescriptors,
                                             const cv::Ptr<cv::DescriptorMatcher> matcher)
{
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::KeyPoint> goodKeypoints;
    matcher->knnMatch(featureDescriptors, featureDescriptors, matches, 2);
    for (std::vector<cv::DMatch> match : matches)
    {
        if (match[0].distance < match[1].distance * MAX_DISTANCE_RATIO) {
            goodKeypoints.push_back(keypts[match[0].queryIdx]);
        }
    }
    return goodKeypoints;
}


/**
 * @brief sets new reference frame for relative homography estimation
 * @param frame  reference frame
 */
void HomographyEstimator::setReferenceFrame (cv::Mat frame)
{
    cv::Mat frame_gray, frame_desc;
    std::vector<cv::KeyPoint> keypoints_ref;
    std::vector<std::vector<cv::DMatch>> matches;

    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    // detect featurepoints
    detector->detect(frame_gray, keypoints_ref);
    cv::KeyPointsFilter::retainBest(keypoints_ref, MAX_NUM_KEYPTS);
    // compute respective descriptors
    desc_extractor->compute(frame_gray, keypoints_ref, frame_desc);
    // refine descriptors to keep (ratio test)
    matcher->knnMatch(frame_desc, frame_desc, matches, 2);
    for (const std::vector<cv::DMatch> match : matches)
    {
        if (match[0].distance < match[1].distance * MAX_DISTANCE_RATIO)
        {
            feature_desc.push_back(frame_desc.row(match[0].queryIdx));
            feature_points.push_back(keypoints_ref[match[0].queryIdx].pt);
        }
    }
    feature_index = 0;
}


void HomographyEstimator::updateFeaturePoints (std::vector<cv::KeyPoint> keypoints,
                                               const cv::Mat frame_gray, const cv::Mat H)
{
    cv::Mat keypt_desc;
    std::vector<cv::KeyPoint> goodKeypoints;
    std::vector<std::vector<cv::DMatch>> matches;
    cv::KeyPointsFilter::retainBest(keypoints, 150);

    desc_extractor->compute(frame_gray, keypoints, keypt_desc);
    goodKeypoints = ratioTest(keypoints, keypt_desc, matcher);
    desc_extractor->compute(frame_gray, keypoints, keypt_desc);

    matcher->knnMatch(keypt_desc, feature_desc, matches, 1);

    if (feature_points.size() < MAX_NUM_KEYPTS) {
        for (const std::vector<cv::DMatch> &match : matches) {
            // note that we are now inserting NEW features, which means we want it NOT
            // to match features that's allready there i.e. we want to preserve the
            // ratio given by the two nearest neighbors
            if ( match[0].distance > DIST_THRESH ) {
                // keypt: transformed into reference image
                cv::Point2f keypt = keypoints[match[0].queryIdx].pt;
                float divisor = H.at<double>(2,0)*keypt.x +
                                H.at<double>(2,1)*keypt.y + H.at<double>(2,2);
                keypt.x = (H.at<double>(0,0)*keypt.x +
                           H.at<double>(0,1)*keypt.y +
                           H.at<double>(0,2)) / divisor;
                keypt.y = (H.at<double>(1,0)*keypt.x +
                           H.at<double>(1,1)*keypt.y +
                           H.at<double>(1,2)) / divisor;
                feature_points.push_back(keypt);
                feature_desc.push_back(keypt_desc.row(match[0].queryIdx));
                if (feature_points.size() >= MAX_NUM_KEYPTS) {
                    break;
                }
            }
        }
    } else {
        // circularly swap out old features with updated feature points
        for (const std::vector<cv::DMatch> &match : matches) {
            // note that we are now inserting NEW features, which means we want it NOT
            // to match features that's allready there
            if (match[0].distance > DIST_THRESH) {
                // keypt: transformed into reference image
                cv::Point2f keypt = keypoints[match[0].queryIdx].pt;
                float divisor = H.at<double>(2,0)*keypt.x +
                    H.at<double>(2,1)*keypt.y + H.at<double>(2,2);
                keypt.x = (H.at<double>(0,0)*keypt.x +
                           H.at<double>(0,1)*keypt.y +
                           H.at<double>(0,2)) / divisor;
                keypt.y = (H.at<double>(1,0)*keypt.x +
                           H.at<double>(1,1)*keypt.y +
                           H.at<double>(1,2)) / divisor;
                if (feature_index == MAX_NUM_KEYPTS) {
                    feature_index = 0;
                }
                feature_points[feature_index] = keypt;
                keypt_desc.row(match[0].queryIdx).copyTo(feature_desc.row(feature_index++));
            }
        }
    }

#ifdef DEBUG
    std::cout << "DEBUG: Added feature points, now got: " <<
        feature_points.size() << "number of featurepoints in reference\n";
#endif
}


/**
 * @brief Estimates the homography from current set reference frame to @frame
 * @param frame  undistorted image (src image)
 * @return  relative homography; or empty mat if too few inliers
 */
cv::Mat HomographyEstimator::estimateHomography(cv::Mat frame)
{
    cv::Mat frame_gray, frame_desc, H;

    for (;;) {
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

        // detect keypoints
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(frame_gray, keypoints);

        // compute their descriptors
        desc_extractor->compute(frame_gray, keypoints, frame_desc);

        // match keypoint feature and refine using ratio test
        std::vector<std::vector<cv::DMatch>> matches;
        matcher->knnMatch(frame_desc, feature_desc, matches, 2);

        std::vector<cv::Point2f> keypt_ref, keypt_frame;
        for (const auto& match : matches)
        {
            if (match[0].distance < match[1].distance * MAX_DISTANCE_RATIO)
            {
                keypt_frame.push_back(keypoints[match[0].queryIdx].pt);
                keypt_ref.push_back(feature_points[match[0].trainIdx]);
            }
        }

        if (keypt_frame.size() < 2*MIN_NUM_INLIERS) {
            return cv::Mat();
        }
        // findHomography from ref. image to current image
        std::vector<char> inliers;
        H = cv::findHomography(keypt_ref, keypt_frame, cv::RANSAC, 3, inliers, 2100, 0.995);

        if (inliers.size() < MIN_NUM_INLIERS) {
            return cv::Mat();
        }
        /* EXPERIMENTAL update featureset */
        else if (inliers.size() < keypoints.size()/3 &&
                 keypoints.size() > 200 &&
                 inliers.size()*5 > keypt_frame.size()*4 &&
                 H.rows == 3){
            updateFeaturePoints(keypoints, frame_gray, H.inv());
            return H;
        }
        else {
            return H;
        }
    }
}

