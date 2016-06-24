#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>

using namespace std;
using namespace cv;

void getSubWindow(Mat &frame, Mat &subWindow, Point centraCoor, Size sz, Mat &cos_window);

void calculateHann(Mat &cos_window, Size sz);

void denseGaussKernel(float sigma, Mat x, Mat y, Mat &k);

cv::Mat getGaussian1(int n, double sigma, int ktype);

cv::Mat getGaussian2(Size sz, double sigma, int ktype);

cv::Mat fft(Mat x);