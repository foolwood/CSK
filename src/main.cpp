#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "csk.h"
#include "benchmark_info.h"

using namespace std;
using namespace cv;


string benchmark_path = "E:/100Benchmark/";
string video_name = "Tiger1";
string video_path = benchmark_path + video_name +"/";
vector<Rect> groundtruth_rect;
vector<String>img_files;

int main(){

  if (load_video_info(video_path, groundtruth_rect, img_files) != 1)
		return -1;

	double padding = 2;
	double output_sigma_factor = 1./16;
  double sigma = 0.2;
  double lambda = 1e-2;
  double interp_factor = 0.075;

  groundtruth_rect[0].x -= 1;     //cpp is zero based
  groundtruth_rect[0].y -= 1;
  Point pos = centerRect(groundtruth_rect[0]);
	Size target_sz(groundtruth_rect[0].width, groundtruth_rect[0].height);
  bool resize_image = false;
  if (std::sqrt(target_sz.area()) >= 1000){
    pos.x = cvFloor(double(pos.x) / 2);
    pos.y = cvFloor(double(pos.y) / 2);
    target_sz.width = cvFloor(double(target_sz.width) / 2);
    target_sz.height = cvFloor(double(target_sz.height) / 2);
    resize_image = true;
  }
  Size sz = scale_size(target_sz, (1.0+padding));
  
  
  double output_sigma = sqrt(double(target_sz.area())) * output_sigma_factor;
	Mat y = CreateGaussian2(sz, output_sigma, CV_64F);
  Mat yf;
  dft(y, yf, DFT_COMPLEX_OUTPUT);
  
	Mat cos_window(sz, CV_64FC1);
	CalculateHann(cos_window, sz);

	Mat im;
	Mat im_gray;
	Mat z,new_z;
	Mat alphaf, new_alphaf;
	Mat x;
	Mat k, kf;
	Mat response;
	double time = 0;
	int64 tic,toc;
  for (int frame = 0; frame < img_files.size(); ++frame)
	{
		
    im = imread(img_files[frame], IMREAD_COLOR);
    im_gray = imread(img_files[frame], IMREAD_GRAYSCALE);
    if (resize_image){
      resize(im, im, im.size() / 2, 0, 0, INTER_CUBIC);
      resize(im_gray, im_gray, im.size() / 2, 0, 0, INTER_CUBIC);
    }

		tic = getTickCount();
		GetSubWindow(im_gray, x, pos, sz, cos_window);
		
		
		if (frame > 0)
		{
			DenseGaussKernel(sigma, x, z, k);
      dft(k, kf, DFT_COMPLEX_OUTPUT);
			cv::idft(ComplexMul(alphaf,kf), response, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT
			Point maxLoc;
			minMaxLoc(response, NULL, NULL, NULL, &maxLoc);
			pos.x = pos.x - cvFloor(float(sz.width) / 2.0) + maxLoc.x+1;
      pos.y = pos.y - cvFloor(float(sz.height) / 2.0) + maxLoc.y+1;
		}
		

		//get subwindow at current estimated target position, to train classifer
		GetSubWindow(im_gray, x, pos, sz, cos_window);

		DenseGaussKernel(sigma, x, x, k);
    dft(k, kf, DFT_COMPLEX_OUTPUT);
    new_alphaf = ComplexDiv(yf, kf + Scalar(lambda, 0));
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
		imshow(video_name, im);
		char key = waitKey(1);
		if (key == 27)
			break;
		
	}
	time = time / getTickFrequency();
  std::cout << "Time: " << time << "    fps:" << img_files.size() / time << endl;
    waitKey();
	return 0;
}