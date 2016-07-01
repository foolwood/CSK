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

cv::Mat complexMul(Mat x1, Mat x2);

cv::Mat complexDiv(Mat x1, Mat x2);

static inline cv::Point centerRect(const cv::Rect& r)
{
    return cv::Point(r.x+cvRound(float(r.width) / 2.0), r.y+cvRound(float(r.height) / 2.0));
}

static inline cv::Rect scale_rect(const cv::Rect& r, float scale)
{
    cv::Point m=centerRect(r);
    float width  = r.width  * scale;
    float height = r.height * scale;
    int x=cvRound(m.x - width/2.0);
    int y=cvRound(m.y - height/2.0);
    
    return cv::Rect(x, y, cvRound(width), cvRound(height));
}

static inline cv::Size scale_size(const cv::Size& r, float scale)
{
    float width  = r.width  * scale;
    float height = r.height * scale;
    
    return cv::Size(cvRound(width), cvRound(height));
}

static inline cv::Size scale_sizexy(const cv::Size& r, float scalex,float scaley)
{
    float width  = r.width  * scalex;
    float height = r.height * scaley;
    
    return cv::Size(cvRound(width), cvRound(height));
}