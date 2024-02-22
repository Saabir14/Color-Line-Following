#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <ctime>

#define DISPLAY 1
#define FPS 1

using namespace std;
using namespace cv;

#if DISPLAY
Mat concatenateFrames(const vector<Mat>& frames);
#endif

// Matrices to store the frames
Mat frame, mapped;

// Maps
using ScalarPair = pair<cv::Scalar, cv::Scalar>;
using ColorMap = map<string, vector<ScalarPair>>;
const ColorMap mapDictionary = {
    {"red", {
        {Scalar(0, 102, 20), Scalar(7, 255, 255)},
        {Scalar(176, 102, 20), Scalar(180, 255, 255)}
    }},
    {"green", {
        {Scalar(35, 102, 20), Scalar(88, 255, 255)}
    }},
    {"blue", {
        {Scalar(89, 102, 25), Scalar(125, 255, 255)}
    }},
    {"black", {
        {Scalar(0, 0, 0), Scalar(180, 102, 64)}
    }}
};

int main(void) {
    VideoCapture camera(0); // Initialize camera (change index if needed)

    if (!camera.isOpened()) {
        cout << "Error opening the camera" << endl;
        return -1;
    }

#if DISPLAY
    // Create a vector to store frames
    vector<Mat> frames;
    Mat concatenatedResult;
#endif

#if FPS
    double totalTime = 0;
    int frameCount = 0;
#endif
    
    do {
        // Read a frame from the camera
        if (!camera.read(frame)) {
            cout << "Could not read from the camera" << endl;
            continue;
        }
        
#if FPS
        clock_t start = clock(); // Start time
#endif
        
#if DISPLAY
        // Clear the vectors of frames
        frames.clear();
        // Add frame to vector (use .clone to make deep copy)
        frames.push_back(frame.clone());
#endif
        
//        // Convert the frame to HSV and equalise the Value channel
//        cvtColor(frame, frame, COLOR_BGR2HSV); // Convert to HSV
//        vector<Mat> channels; // Array for channels
//        split(frame, channels); // Split the HSV into separate channels
//        equalizeHist(channels[2], channels[2]); // Equalise the Value channel
//        merge(channels, frame); // Merge back into a single image
//        cvtColor(frame, frame, COLOR_HSV2BGR); // Convert back to BGR
//#if DISPLAY
//        frames.push_back(frame.clone());
//#endif
        
        // Apply a blur to the frame
        //GaussianBlur(frameEQ, frameBlur, Size(11, 11), 0, 0);
        blur(frame, frame, Size(11, 11), Point(-1, -1));
        //medianBlur(frameEQ, frameBlur, 11);
        //frameBlur = frameEQ.clone();
#if DISPLAY
        frames.push_back(frame.clone());
#endif
        
        // Convert the frame to HSV
        cvtColor(frame, frame, COLOR_BGR2HSV);
        //frames.push_back(frame.clone());
        
#if DISPLAY
        // Create Mat for showing lines
        Mat lineMat = frames[0].clone();
        frames.push_back(lineMat);
#endif
        // Create maps
        // Filter each color from the frame and decide which color to use
        // Set frame to that color
        for (const auto& colorEntry : mapDictionary) {
            const string& color = colorEntry.first;
            const vector<ScalarPair>& bounds = colorEntry.second;
            
            for (const auto& pair : bounds)
                inRange(frame, pair.first, pair.second, mapped);
            
#if DISPLAY
            frames.push_back(mapped.clone());
            // Display color along with 'maped'
            putText(
                    frames.back(),
                    color,
                    Point(80, 80),
                    FONT_HERSHEY_SIMPLEX,
                    1,
                    Scalar(255, 255, 255),
                    2
                    );
#endif
            
//            if (countNonZero(mapped) > pixels) {
//
//            }
            
            // Dialate 'maped' for simpler line detection
            dilate(mapped, mapped, Mat(), Point(-1, -1), 20);
#if DISPLAY
            frames.push_back(mapped.clone());
            // Display color along with 'maped'
            putText(
                    frames.back(),
                    "dilated color",
                    Point(80, 80),
                    FONT_HERSHEY_SIMPLEX,
                    1,
                    Scalar(255, 255, 255),
                    2
                    );
#endif
            
            // Find weighted average of the color
            vector<Vec4i> lines;
            HoughLinesP(mapped, lines, 1, CV_PI / 180, 25, mapped.rows / 2);
            
            if (lines.empty())
                continue;
            
            double sum_slope = 0, sum_intercept = 0;
            int count = 0;

            for (size_t i = 0; i < lines.size(); i++) {
                Vec4i l = lines[i];
                double slope = (double)(l[3] - l[1]) / (l[2] - l[0]);
                double intercept = l[1] - slope * l[0];
                
                sum_slope += slope;
                sum_intercept += intercept;
                count++;
            }

            double mean_slope = sum_slope / count;
            double mean_intercept = sum_intercept / count;
            
#if DISPLAY
            // Display the line
            int x1 = 0;  // x-coordinate of the leftmost point
            int y1 = mean_slope * x1 + mean_intercept;  // y-coordinate of the leftmost point

            int x2 = mapped.cols;  // x-coordinate of the rightmost point
            int y2 = mean_slope * x2 + mean_intercept;  // y-coordinate of the rightmost point
            
            for (Vec4i l : lines)
                line(lineMat, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 255), 1);
            
            Vec4i mean_line(x1, y1, x2, y2);
            line(lineMat, Point(mean_line[0], mean_line[1]), Point(mean_line[2], mean_line[3]), Scalar(255, 255, 0), 20);
#endif
            
            // Calculate the x-intercept
            int x_intercept = -(mapped.rows - mean_slope * mapped.cols) / 2 / mean_slope;
            if (!x_intercept)
                continue;
            
            // Calculate the bearing angle (rad)
            double bearing_angle_rad = atan(mean_slope);
            if (bearing_angle_rad > CV_PI)
                bearing_angle_rad -= 2 * CV_PI;
            
            cout << "Bearing angle: " << bearing_angle_rad << "\tHorizontal position: " << x_intercept << endl;
        }
        
#if DISPLAY
        putText(
                lineMat,
                "lines",
                Point(80, 80),
                FONT_HERSHEY_SIMPLEX,
                1,
                Scalar(255, 255, 255),
                2
                );
        
        // Display all frames
        concatenatedResult = concatenateFrames(frames);
        imshow("frames", concatenatedResult);
#endif
        
#if FPS
        clock_t end = clock(); // End time
        double frameTime = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        // Accumulate total time and frame count1
        totalTime += frameTime;
        frameCount++;
        // Print FPS every second
        if (totalTime >= 1.0) {
            double fps = static_cast<double>(frameCount) / totalTime;
            cout << "FPS: " << fps << endl;
            
            // Reset counters
            frameCount = 0;
            totalTime = 0.0;
        }
#endif
    }
    while (waitKey(30) < 0);

    // Release camera resources
    camera.release();
    return 0;
}

#if DISPLAY
Mat concatenateFrames(const vector<Mat>& frames) {
    // Check if the input vector is empty
    if (frames.empty()) {
        cerr << "Error: No frames provided for concatenation." << endl;
        return Mat(); // Return an empty matrix
    }

    // Calculate the number of rows and columns for the grid
    int numFrames = static_cast<int>(frames.size());
    int gridCols = ceil(sqrt(numFrames));
    int gridRows = ceil(static_cast<double>(numFrames) / gridCols);

    // Find the maximum frame width and height
    int maxFrameWidth = 0;
    int maxFrameHeight = 0;
    for (const auto& frame : frames) {
        maxFrameWidth = max(maxFrameWidth, frame.cols);
        maxFrameHeight = max(maxFrameHeight, frame.rows);
    }

    // Create an empty canvas to hold the concatenated frame
    Mat concatenatedFrame(maxFrameHeight * gridRows, maxFrameWidth * gridCols, CV_8UC3, Scalar::all(0));

    // Copy each frame to the appropriate position in the concatenated frame
    for (int i = 0; i < numFrames; ++i) {
        if (frames[i].empty()) {
            cerr << "Warning: Frame " << i << " is empty." << endl;
            continue;
        }

        // Convert the frame to CV_8UC3 if it is grayscale or binary
        Mat converted;
        if (frames[i].channels() == 1) {
            // If the image is binary, scale it to 0-255
            if (cv::countNonZero(frames[i]) == 0 || cv::countNonZero(frames[i]) == frames[i].total()) {
                frames[i].convertTo(converted, CV_8UC1, 255.0);
            } else {
                converted = frames[i];
            }
            cvtColor(converted, converted, cv::COLOR_GRAY2BGR);
        } else {
            frames[i].convertTo(converted, CV_8UC3);
        }

        int x_offset = (i % gridCols) * maxFrameWidth;
        int y_offset = (i / gridCols) * maxFrameHeight;
        converted.copyTo(concatenatedFrame(Rect(x_offset, y_offset, converted.cols, converted.rows)));
    }

    return concatenatedFrame;
}
#endif
