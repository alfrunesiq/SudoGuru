#include "sudoguru.hpp"
#include "extractools.hpp"
#include "sudokusolver/sudoku_board.hpp"

int sudoguru (void)
{
#ifdef DEBUG
    cv::namedWindow("Thresholded");
    cv::namedWindow("Perspective", cv::WINDOW_AUTOSIZE);
#endif
    cv::namedWindow("Camera");
    cv::VideoCapture cap;
    Extractor *extractor = new Extractor;
    if (!cap.open(CAMERA_ID_0)) {
        std::cerr << "sudoguru.cpp: Could not open camera " << CAMERA_ID_0 << "\n";
        return EXIT_FAILURE;
    }

    int height = static_cast<int> (cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int width = static_cast<int> (cap.get(cv::CAP_PROP_FRAME_WIDTH));

    // frame buffers
    cv::Mat frame(height, width, CV_8UC3);
    cv::Mat frame_buf(height, width, CV_8UC1),
            frame_bin(height, width, CV_8UC1);
//#ifdef DEBUG
    cv::Mat prspct;
    cv::Mat debug_buf;
//#endif

    // morhological structuring elements
    cv::Mat _3x3cross = cv::getStructuringElement(cv::MORPH_CROSS,
                                                  cv::Size(3,3)),
           _11x11circ = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                   cv::Size(11,11));


    // Corner points and edge lines
    std::vector<cv::Point2f> pts;
    std::vector<cv::Vec2f> edges;

    Board sudokuBoard = Board();
    std::vector<std::vector<int>> *grid;

    for (;;)
    {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "sudoguru.cpp: Frame empty\n";
            return EXIT_FAILURE;
        }
#ifdef DEBUG
        auto t1 = std::chrono::high_resolution_clock::now();
#endif

        // Convert to grayscale and adaptively threshold; assuming board
        // is high contrast
        cv::cvtColor(frame, frame_buf, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(frame_buf, frame_buf, cv::Size(5,5), 0);
        cv::adaptiveThreshold(frame_buf, frame_bin, 255,
                              cv::ADAPTIVE_THRESH_MEAN_C,
                              cv::THRESH_BINARY_INV, 11, 5.0);
        /* morphological closing */
        cv::dilate(frame_bin, frame_bin, _3x3cross);
        cv::erode(frame_buf, frame_buf, _3x3cross);
        frame_buf = biggestConnectedComponents(frame_bin);

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

        edges = extractor->extractEdges(frame_buf);

        for (cv::Vec2f edge : edges) {
            float y = edge[0] / std::sin(edge[1]),
                x = -1 / std::tan(edge[1]);
            cv::line(frame_buf, cv::Point(0, y),
                     cv::Point(frame_buf.cols, frame_buf.cols*x+y), CV_RGB(0,0,255));
        }
        if (edges.size() > 0) {
            pts = extractor->extractCorners(edges);
#ifdef DEBUG
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
            std::vector<cv::Point2f> p = {cv::Point2f(0.0f,0.0f),
                                          cv::Point2f(BOARDSIZE,0.0f),
                                          cv::Point2f(BOARDSIZE,BOARDSIZE),
                                          cv::Point2f(0.0f,BOARDSIZE)};
            cv::Matx33f H = cv::getPerspectiveTransform(pts, p);
            cv::warpPerspective(frame, prspct, H,
                                cv::Size(BOARDSIZE,BOARDSIZE),
                                cv::INTER_LINEAR);
            grid = extractor->extractGrid(prspct);
            cv::imshow("Perspective", prspct);

            if (grid != NULL) {
                std::cout << grid->size();
                sudokuBoard.setBoard(grid);
                if(sudokuBoard.solve(grid)){
                    for (int i = 0; i < 9; i++) {
                        if(!(i % 3)) { printf("%22s\n",
                                              "+-----------------------+");}
                        for (int j = 0; j < 9; j++) {
                            if(!(j % 3)) { printf("| "); }
                            printf("%d ", (*grid)[i][j]);
                        }
                        printf("|\n");
                    }
                    printf("%22s\n",
                           "+-----------------------+");
                    printf("\n\n");
                } else {
                    std::cout << "Board Size < 9 !!!\n";
                }
            }
            // draw markers for points used in transform
            cv::drawMarker(frame, pts[1], CV_RGB(255, 128, 0), cv::MARKER_DIAMOND);
            cv::drawMarker(frame, pts[2], CV_RGB(255, 255, 56), cv::MARKER_DIAMOND);
            cv::drawMarker(frame, pts[0], CV_RGB(56, 128, 255), cv::MARKER_DIAMOND);
            cv::drawMarker(frame, pts[3], CV_RGB(230, 57, 255), cv::MARKER_DIAMOND);
        }

        cv::imshow("Camera", frame);
#ifdef DEBUG
        auto t2 = std::chrono::high_resolution_clock::now();
        cv::imshow("Thresholded", frame_bin);
        cv::imshow("CC", frame_buf);
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                  << std::endl;
#endif

        if (cv::waitKey(1) >= 0) {
            break;
        }
    }

    return EXIT_SUCCESS;
}
