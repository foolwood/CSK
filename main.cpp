#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "load_video_info.h"

using namespace cv;
using namespace std;


string benchmarkPath = "E:/benchmark50/";
string videoName = "David3";
string videoPath = benchmarkPath + videoName;
vector<Rect> groundtruthRect;
vector<String>fileName;

int main(){

	if (load_video_info(videoPath, groundtruthRect,fileName) != 1)
		return -1;

	Mat frame;
	namedWindow(videoName,WINDOW_NORMAL);
	for (int i = 0; i < fileName.size(); ++i)
	{
		frame = imread(fileName[i]);
		
		
		rectangle(frame, groundtruthRect[i], Scalar(0, 255, 0), 2);
		putText(frame, to_string(i), Point(20, 40),6, 1, Scalar(0, 255, 255),2);
		imshow(videoName, frame);
		char key = waitKey(1);
		if (key == 27)
			break;
	}
	return 0;
}