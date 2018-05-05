#include "homography_estimator.hpp"

HomographyEstimator::HomographyEstimator()
{
    // initialize detector / descriptor and matcher
    detector = cv::ORB::create(512, 1.4f, 10, 32, 0, 2, cv::ORB::FAST_SCORE, 32, 20);
    desc_extractor = detector;
    matcher = cv::BFMatcher::create(desc_extractor->defaultNorm());
}


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
    for (const auto& match : matches)
    {
        if (match[0].distance < match[1].distance * MAX_DISTANCE_RATIO)
        {
            feature_desc.push_back(frame_desc.row(match[0].queryIdx));
            feature_points.push_back(keypoints_ref[match[0].queryIdx].pt);
        }
    }
}


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
        } else {
            return H;
        }
    }
}

