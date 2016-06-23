#include "get_subwindow.h"
#define PI 3.1415926
void getSubWindow(Mat &frame, Mat &subWindow, Point centraCoor, Size sz, Mat &cos_window){
	Point lefttop;
	lefttop.x = max(centraCoor.x - (sz.width>>1),1);
	lefttop.y = max(centraCoor.y - (sz.height>>1),1);
	Point rightbottom;
	rightbottom.x = min(centraCoor.x + (sz.width >>1),frame.cols);
	rightbottom.y = min(centraCoor.y + (sz.height>>1),frame.rows);
	Rect roiRect(lefttop, rightbottom);
	frame(roiRect).copyTo(subWindow);
	cv::resize(subWindow, subWindow, sz);
	subWindow.convertTo(subWindow, CV_32FC1,1.0/255.0,-0.5);
	subWindow.dot(cos_window);
}


void calculatehann(Mat &cos_window, Size sz){
	Mat temp1(Size(sz.width, 1), CV_32FC1);
	Mat temp2(Size(sz.height, 1), CV_32FC1);
	for (int i = 0; i < sz.width; ++i)
		temp1.at<float>(0, i) = 0.5*(1 - cos(2 * PI*i / (sz.width - 1)));
	for (int i = 0; i < sz.height; ++i)
		temp2.at<float>(0, i) = 0.5*(1 - cos(2 * PI*i / (sz.height - 1)));
	cos_window = temp2.t()*temp1;
}