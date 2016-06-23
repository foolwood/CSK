#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

void getSubWindow(Mat &frame, Mat &subWindow, Point centraCoor, Size sz, Mat &cos_window);

void calculatehann(Mat &cos_window, Size sz);