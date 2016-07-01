#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "load_video_info.h"
#include "csk.h"

using namespace cv;
using namespace std;


string benchmarkPath = "/users/wangqiang/Desktop/benchmark100/";
string videoName = "Boy";
string videoPath = benchmarkPath + videoName;
vector<Rect> groundtruthRect;
vector<String>fileName;



int main(){

	if (load_video_info(videoPath, groundtruthRect,fileName) != 1)
		return -1;

	float padding = 4;					
	float output_sigma_factor = 1.0/16.0;
	float sigma = 0.2;
	float lambda = 0.01;
	float interp_factor = 0.075;

	

	Size target_sz(groundtruthRect[0].width, groundtruthRect[0].height);
    Size sz = scale_size(target_sz, (1.0+padding));
    groundtruthRect[0].x -=1;
    groundtruthRect[0].y -=1;
    Point pos = centerRect(groundtruthRect[0]);
    
    
	
	float output_sigma = sqrt(float(target_sz.area())) * output_sigma_factor;
	Mat y = getGaussian2(sz, output_sigma, CV_32F);
	Mat yf = fft(y);

	Mat cos_window(sz, CV_32FC1);
	calculateHann(cos_window, sz);

	Mat im;
	Mat im_gray;
	Mat z,new_z;
	Mat alphaf, new_alphaf;
	Mat x;
	Mat k, kf;
	Mat response;
	double time = 0;
	int64 tic,toc;
	for (int frame = 0; frame < fileName.size(); ++frame)
	{
		
		im = imread(fileName[frame], IMREAD_COLOR);
		im_gray = imread(fileName[frame], IMREAD_GRAYSCALE);

		tic = getTickCount();
		getSubWindow(im_gray, x, pos, sz, cos_window);
		
		
		if (frame > 0)
		{
			denseGaussKernel(sigma, x, z, k);
			kf = fft(k);
			cv::idft(complexMul(alphaf,kf), response, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT
			Point maxLoc;
			minMaxLoc(response, NULL, NULL, NULL, &maxLoc);
			pos.x = pos.x - cvRound(float(sz.width) / 2.0) + maxLoc.x;
			pos.y = pos.y - cvRound(float(sz.height) / 2.0) + maxLoc.y;
		}
		

		//get subwindow at current estimated target position, to train classifer
		getSubWindow(im_gray, x, pos, sz, cos_window);

		denseGaussKernel(sigma, x, x, k);
		kf = fft(k);
		vector<Mat> planes;
		split(kf, planes);
		planes[0] = planes[0] + lambda;
		merge(planes, kf);
		new_alphaf = complexDiv(yf,kf);
		new_z = x;

		if (frame == 0)
		{
			alphaf = new_alphaf;
			z = x;
		}
		else
		{
			alphaf = (1.0 - interp_factor) * alphaf + interp_factor * new_alphaf;
			z = (1.0 - interp_factor) * z + interp_factor * new_z;
		}
		toc = getTickCount() - tic;
		time += toc;
		Rect rect_position(pos.x - target_sz.width /2, pos.y - target_sz.height/2, target_sz.width, target_sz.height);
		rectangle(im, rect_position, Scalar(0, 255, 0), 2);
		putText(im, to_string(frame), Point(20, 40), 6, 1, Scalar(0, 255, 255), 2);
		imshow(videoName, im);
		char key = waitKey(1);
		if (key == 27)
			break;
		
	}
	time = time / getTickFrequency();
	std::cout << "Time: " << time << "    fps:" << fileName.size()/time << endl;
    waitKey();
	return 0;
}