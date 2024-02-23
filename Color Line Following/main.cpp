#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <ctime>
#include <numeric>

#define DISPLAY 1
#define FPS 1

using namespace std;
using namespace cv;

vector<Mat> splitImage(Mat & image, int M, int N );
#if DISPLAY
Mat concatenateFrames(const vector<Mat>& frames);
#endif

// Matrices to store the frames
Mat frame;

// Number of horizontal splits
const int horizontal_splits = 30;

// parameters for the weighted average
const double d2a_ratio = .5;

// Maps
using ScalarPair = pair<Scalar, Scalar>;
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
        {Scalar(0, 0, 0), Scalar(180, 102, 120)}
    }}
};

// Order of Maps
const string colorOrder[] = {"black"};
const int colorOrderSize = sizeof(colorOrder) / sizeof(colorOrder[0]);

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
    
int colorIndex = 0;
    do {
        // Read a frame from the camera
        if (!camera.read(frame)) {
            cout << "Could not read from the camera" << endl;
            continue;
        }
        
#if FPS
        const clock_t start = clock(); // Start time
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
        //GaussianBlur(frame, frame, Size(11, 11), 0, 0);
        blur(frame, frame, Size(20, 20), Point(-1, -1));
        //medianBlur(frame, frame, 11);
#if DISPLAY
        frames.push_back(frame.clone());
#endif
        
        // Convert the frame to HSV
        cvtColor(frame, frame, COLOR_BGR2HSV);
        //frames.push_back(frame.clone());
        
        // Set the frame to the mask of the relevent color
        string color = colorOrder[colorIndex];
        static Mat mask;
        for (const auto& pair : mapDictionary.at(color))
            inRange(frame, pair.first, pair.second, mask);
        
        if (colorIndex + 1 < colorOrderSize)
        {
            static Mat newMask;
            const string nextColor = colorOrder[(colorIndex + 1)];
            for (const auto& pair : mapDictionary.at(nextColor))
                inRange(frame, pair.first, pair.second, newMask);
            
            if (countNonZero(newMask) > countNonZero(mask) >> 2)
            {
                mask = newMask;
                colorIndex++;
            }
        }
        frame = mask;
#if DISPLAY
        frames.push_back(frame.clone());
        // Display dialated color
        putText(
                frames.back(),
                colorOrder[colorIndex],
                Point(80, 80),
                FONT_HERSHEY_SIMPLEX,
                1,
                Scalar(255, 255, 255),
                2
                );
#endif
        
        // Dialate 'maped' for simpler line detection
        dilate(frame, frame, Mat(), Point(-1, -1), 40);
#if DISPLAY
        frames.push_back(frame.clone());
        // Display dialated color
        putText(
                frames.back(),
                "dilated" + colorOrder[colorIndex],
                Point(80, 80),
                FONT_HERSHEY_SIMPLEX,
                1,
                Scalar(255, 255, 255),
                2
                );
#endif
        
        // Find a line for weighted average using points
        static vector<Point> points;
        points.clear();
        int count = 0;
        for (Mat img : splitImage(frame, 1, horizontal_splits))
        {
            const Moments mu = moments(img, true);
            Point center(mu.m10 / mu.m00, mu.m01 / mu.m00);
            center.y += (frame.rows / horizontal_splits) * count;
            points.push_back(center);
            count++;
        }

#if DISPLAY
        for (Point point : points)
        {
            circle(frames[0], point, 5, Scalar(0, 0, 255), -1);
        }
#endif

        // Convert vector<Point> to Mat
        Mat pointsMat = Mat(points);

        // Calculate mean and standard deviation for x-coordinates
        Mat xCoords = pointsMat.col(0); // Extract x-coordinates
        Scalar mean, stddev;
        meanStdDev(xCoords, mean, stddev);

        // Create a mask for points within one standard deviation in x-direction
        Mat m = (abs(xCoords - mean) <= stddev / 3);

        // Apply the mask to get the points within one standard deviation
        vector<Point> filteredPoints;
        for(int i = 0; i < m.total(); i++) {
            if(m.at<uchar>(i)) {
                filteredPoints.push_back(points[i]);
            }
        }

#if DISPLAY
        for (Point point : filteredPoints)
        {
            circle(frames[0], point, 5, Scalar(0, 255, 0), -1);
        }
#endif
        if (filteredPoints.size() >= 2)
        {
            Vec4f line_of_best_fit;
            fitLine(filteredPoints, line_of_best_fit, DIST_L2, 0, .01, .01);
            
            // Extract line parameters
            const float vx = line_of_best_fit[0];
            const float vy = line_of_best_fit[1];
            const float x = line_of_best_fit[2];
            const float y = line_of_best_fit[3];
            
#if DISPLAY
            // Calculate the line points from parameters
            Point point1, point2;
            point1.x = x - vx*1000; // Point on the line furthest from the center
            point1.y = y - vy*1000;
            point2.x = x + vx*1000; // Point on the line closest to the center
            point2.y = y + vy*1000;
            
            // Draw the line of best fit
            line(frames[0], point1, point2, Scalar(255, 0, 0), 2, LINE_AA);
#endif
            if (!vx)
                continue;
            const float m = vy / vx;
            const float b = y - m * x;
            const float horizontal_distence = (y - b) / m - (frame.cols >> 1);
            float angle = atan(m);
            if (angle > CV_PI)
                angle -= CV_2PI;
            
            // Calculate weighted average using some multiplier of both the bearing angle and the horizontal position
            const double w = horizontal_distence / (frame.cols >> 1) * d2a_ratio + angle / (CV_PI/2) * (1 - d2a_ratio);
            cout << "Weighted average: " << w << endl;
        }
            
#if DISPLAY
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

vector<Mat> splitImage(Mat & image, int M, int N )
{
  // All images should be the same size ...
  int width  = image.cols / M;
  int height = image.rows / N;
  // ... except for the Mth column and the Nth row
  int width_last_column = width  + ( image.cols % width  );
  int height_last_row   = height + ( image.rows % height );

  vector<Mat> result;

  for( int i = 0; i < N; ++i )
  {
    for( int j = 0; j < M; ++j )
    {
      // Compute the region to crop from
      Rect roi( width  * j,
                    height * i,
                    ( j == ( M - 1 ) ) ? width_last_column : width,
                    ( i == ( N - 1 ) ) ? height_last_row   : height );

      result.push_back( image( roi ) );
    }
  }

  return result;
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
            if (countNonZero(frames[i]) == 0 || countNonZero(frames[i]) == frames[i].total()) {
                frames[i].convertTo(converted, CV_8UC1, 255.0);
            } else {
                converted = frames[i];
            }
            cvtColor(converted, converted, COLOR_GRAY2BGR);
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
