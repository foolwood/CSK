#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "load_video_info.h"
#include "get_subwindow.h"

using namespace cv;
using namespace std;


string benchmarkPath = "E:/benchmark50/";
string videoName = "Basketball";
string videoPath = benchmarkPath + videoName;
vector<Rect> groundtruthRect;
vector<String>fileName;

int main(){

	if (load_video_info(videoPath, groundtruthRect,fileName) != 1)
		return -1;

	Mat frame;
	Mat frame_gray;

	namedWindow(videoName,WINDOW_NORMAL);
	double duration;
	for (int i = 0; i < fileName.size(); ++i)
	{
		
		frame = imread(fileName[i], IMREAD_COLOR);
		frame_gray = imread(fileName[i], IMREAD_GRAYSCALE);
		Mat subWindow;
		Point centraCoor(groundtruthRect[i].x + (groundtruthRect[i].width >> 1), groundtruthRect[i].y + (groundtruthRect[i].height >> 1));
		Size sz(groundtruthRect[i].width, groundtruthRect[i].height);
		Mat cos_window(sz, CV_32FC1, Scalar(0.0));
		calculatehann(cos_window, sz);
		getSubWindow(frame_gray, subWindow, centraCoor, sz, cos_window);
		
		rectangle(frame, groundtruthRect[i], Scalar(0, 255, 0), 2);
		putText(frame, to_string(i), Point(20, 40),6, 1, Scalar(0, 255, 255),2);
		imshow(videoName, frame);
		char key = waitKey(1);
		if (key == 27)
			break;
	}
	return 0;
}