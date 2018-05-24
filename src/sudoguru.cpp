#include "sudoguru.hpp"
#include "extractools.hpp"
#include "sudokusolver/sudoku_board.hpp"
#include "homographyestimator.hpp"

// declaration of helperfunctions
static cv::Point2f transformPoint(cv::Point2f pt, cv::Mat H);
static bool boardOutsideFrame(std::vector<cv::Point2f> corners,
                              cv::Mat H, cv::Size frame_size);


/**
 * @brief sudoguru main function.
 */
int sudoguru (int argc, char **argv)
{
#ifdef DEBUG
    cv::namedWindow("Thresholded");
    cv::namedWindow("Perspective", cv::WINDOW_AUTOSIZE);
#endif
    cv::namedWindow("Camera");
    cv::VideoCapture cap;
    if(!cap.open(CAMERA_ID)) {
        std::cerr << "sudoguru.cpp: Could not open camera " << CAMERA_ID << "\n";
        return EXIT_FAILURE;
    }

    Extractor *extractor;
    HomographyEstimator *estimator = new HomographyEstimator;
    if (argc > 2) {
        // interpret second argument as path to tensorflow graph-def file (*.pb)
        extractor = new Extractor(std::string(argv[2]));
    } else {
        extractor = new Extractor;
    }
    cv::FileStorage f;
    CameraModel camera;

    // check for camera parameters
    if (argc > 1) {
        f.open(argv[1], cv::FileStorage::READ);
    } else {
        f.open("../cameraParameters.xml", cv::FileStorage::READ);
    }

    // set camera paramaters
    if (f.isOpened()) {
        cv::Mat tmp;
        f["cameraMatrix"] >> tmp;
        camera.K = tmp;
        f["dist_coeffs"] >> tmp;
        camera.dist_coeffs = { tmp.at<double>(0), tmp.at<double>(1),
                               tmp.at<double>(2), tmp.at<double>(3),
                               tmp.at<double>(4) };
        camera.isSet = true;
        f.release();
    } else {
        camera.isSet = false;
    }

    int height = static_cast<int> (cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int width = static_cast<int> (cap.get(cv::CAP_PROP_FRAME_WIDTH));

    /** INITIALIZE DATASTRUCTURES **/
    // initialize frame buffers
    cv::Mat frame(height, width, CV_8UC3),
            frame_undist(height, width, CV_8UC3);
    cv::Mat frame_buf(height, width, CV_8UC1),
            frame_bin(height, width, CV_8UC1);
    cv::Mat extracted_brd(BOARDSIZE, BOARDSIZE, CV_8UC3);

    // morhological structuring element
    cv::Mat _3x3cross = cv::getStructuringElement(cv::MORPH_CROSS,
                                                  cv::Size(3,3));

    // Corner points and edge lines
    std::vector<cv::Point2f> pts;
    std::vector<cv::Vec2f> edges;

    Board sudokuBoard = Board();
    std::vector<std::vector<int>> *grid;
    std::vector<std::vector<int>> solution;
    cv::Mat board_template, frame_mask;

    // initialize solution matrix
    for (int i = 0; i < 9; i++) {
        solution.push_back(std::vector<int>());
        for (int j = 0; j < 9; j++) {
            solution[i].push_back(0);
        }
    }

    // main loop
    for (;;)
    {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "sudoguru.cpp: Frame empty\n";
            return EXIT_FAILURE;
        }

        if (camera.isSet) {
            cv::undistort(frame, frame_undist, camera.K, camera.dist_coeffs);
        } else {
            frame_undist = frame;
        }
	// Draw marker to indicate state of the loop cycles between red, yellow and green
	// depending on which state the program reaches
        cv::drawMarker(frame_undist, cv::Point(10, 10), CV_RGB(255, 0, 16), cv::MARKER_CROSS);
#ifdef DEBUG
        auto t1 = std::chrono::high_resolution_clock::now();
#endif

        // Convert to grayscale and adaptively threshold; assuming board
        // is high contrast
        cv::cvtColor(frame_undist, frame_buf, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(frame_buf, frame_buf, cv::Size(5,5), 0);
        cv::adaptiveThreshold(frame_buf, frame_bin, 255,
                              cv::ADAPTIVE_THRESH_MEAN_C,
                              cv::THRESH_BINARY_INV, 11, 5.0);
        /* morphological closing */
        cv::dilate(frame_bin, frame_bin, _3x3cross);
        cv::erode(frame_buf, frame_buf, _3x3cross);
        frame_buf = biggestConnectedComponents(frame_bin);

        edges = extractor->extractEdges(frame_buf);
        if (edges.size() > 3) {
            pts = extractor->extractCorners(edges);
#ifdef DEBUG
            // draw lines
            for (cv::Vec2f edge : edges) {
                float y = edge[0] / std::sin(edge[1]),
                    x = -1 / std::tan(edge[1]);
                cv::line(frame_buf, cv::Point(0, y),
                         cv::Point(frame_buf.cols, frame_buf.cols*x+y), CV_RGB(0,0,255));
            }
#endif
            std::vector<cv::Point2f> p = {cv::Point2f(0.0f,0.0f),
                                          cv::Point2f(BOARDSIZE,0.0f),
                                          cv::Point2f(BOARDSIZE,BOARDSIZE),
                                          cv::Point2f(0.0f,BOARDSIZE)};
            cv::Matx33f H = cv::getPerspectiveTransform(pts, p);
            cv::warpPerspective(frame_undist, extracted_brd, H,
                                cv::Size(BOARDSIZE,BOARDSIZE),
                                cv::INTER_LINEAR);
            grid = extractor->extractGrid(extracted_brd);

            if (grid != NULL) {
                cv::drawMarker(frame_undist, cv::Point(10, 10), CV_RGB(230, 232, 16), cv::MARKER_CROSS);
                sudokuBoard.setBoard(grid);
                if(sudokuBoard.solve(&solution)) {
                    board_template = cv::Mat::zeros(cv::Size(BOARDSIZE, BOARDSIZE),
                                                    CV_8UC3);
                    char buf[2]; buf[1] = '\0';
                    for (int i = 0; i < 9; i++) {
                        for (int j = 0; j < 9; j++) {
                            if ((*grid)[i][j] == 0) {
                                buf[0] = '0' + (char) solution[i][j];
                                cv::putText(board_template,
                                            cv::String(buf),
                                            cv::Point(j*GRID_GAP_AVG+8, (i+1)*GRID_GAP_AVG-8),
                                            cv::FONT_HERSHEY_PLAIN, 1.5,
                                            CV_RGB(255, 127, 55), 1);
                            }
                        }
                    }
                    cv::bitwise_not(board_template, board_template);
                    cv::warpPerspective(board_template, frame_mask, H.inv(),
                                        frame_undist.size(), cv::INTER_LINEAR,
                                        cv::BORDER_CONSTANT, CV_RGB(255,255,255));
                    estimator->setReferenceFrame(frame_undist);
                    // LOOP: homography tracking
                    cv::Mat buffer;
                    for (;;) {
                        cap >> frame;
                        if (camera.isSet) {
                            cv::undistort(frame, frame_undist, camera.K,
                                          camera.dist_coeffs);
                        } else {
                            frame_undist = frame;
                        }
                        cv::Mat H2 = estimator->estimateHomography(frame_undist);
                        if (H2.empty()) {
                            break;
                        } else if (boardOutsideFrame(pts, H2, frame_undist.size())) {
                            break;
                        }

                        cv::warpPerspective(frame_mask, buffer, H2, frame_undist.size(),
                                            cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                                            CV_RGB(255, 255, 255));
                        buffer = frame_undist & buffer;
                        cv::drawMarker(buffer, cv::Point(10, 10), CV_RGB(54, 220, 16), cv::MARKER_STAR);
                        cv::imshow("Camera", buffer);
                        if (cv::waitKey(1) >= 0) { return EXIT_SUCCESS; }
                    }
                }
            }

#ifdef DEBUG
            // draw markers for points used in transform
            cv::drawMarker(frame_undist, pts[1], CV_RGB(255, 128, 0), cv::MARKER_DIAMOND);
            cv::drawMarker(frame_undist, pts[2], CV_RGB(255, 255, 56), cv::MARKER_DIAMOND);
            cv::drawMarker(frame_undist, pts[0], CV_RGB(56, 128, 255), cv::MARKER_DIAMOND);
            cv::drawMarker(frame_undist, pts[3], CV_RGB(230,57, 255), cv::MARKER_DIAMOND);

            // some debug images
            cv::imshow("Perspective", extracted_brd);
            cv::imshow("Thresholded", frame_bin);
            cv::imshow("Biggest Connected Component", frame_buf);
#endif
        }

        // show result
        cv::imshow("Camera", frame_undist);

#ifdef DEBUG
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                  << std::endl;
#endif
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }

    return EXIT_SUCCESS;
}

/**
 * @brief helper function to projectively transform euclidian points
 * TODO: put these in a seperate library "projTools" or something, and reuse
 *       in homography estimator.
 * @param point  to be transformed
 * @param H      cv::Mat<double> homography
 */
static cv::Point2f transformPoint(cv::Point2f pt, cv::Mat H)
{
    float divisor = pt.x*H.at<double>(2,0) + pt.y*H.at<double>(2,1) + H.at<double>(2,2);
    float x = static_cast<float>(pt.x*H.at<double>(0,0) + pt.y*H.at<double>(0,1) +
                                 H.at<double>(0,2))/divisor;
    float y = static_cast<float>(pt.x*H.at<double>(1,0) + pt.y*H.at<double>(1,1) +
                                 H.at<double>(1,2))/divisor;
    return cv::Point2f(x, y);
}

/**
 * @brief helper function to check if board is outside frame
 * @param corners    cornerpoints of the sudokuboard
 * @param H          cv::Mat<double> homography relating current frame to refference
 * @param frame_size size of frame (camera resolution)
 */
static bool boardOutsideFrame(std::vector<cv::Point2f> corners,
                              cv::Mat H, cv::Size frame_size)
{
    int pts_outside = 0;
    for (cv::Point2f corner : corners)
    {
        cv::Point transformed = transformPoint(corner, H);
        if(transformed.x < 0 || transformed.y < 0 ||
           transformed.x > frame_size.height || transformed.y > frame_size.width) {
            pts_outside++;
        }
    }
    if (pts_outside == static_cast<int>(corners.size())) {
        return true;
    }
    return false;
}
