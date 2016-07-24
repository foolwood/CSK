/*******************************************************************************
* Created by Qiang Wang on 16/7.24
* Copyright 2016 Qiang Wang.  [wangqiang2015-at-ia.ac.cn]
* Licensed under the Simplified BSD License
*******************************************************************************/
#include <iostream>
#include <numeric>      // std::accumulate
#include <string>
#include "opencv2/opencv.hpp"
#include "csk.h"
#include "benchmark_info.h"

using namespace std;
using namespace cv;

int tracker(string video_path, string video_name,double &precision,double &fps);
vector<double>PrecisionCalculate(vector<Rect>groundtruth_rect, vector<Rect>result_rect);

int main(int argc, char** argv){
  string benchmark_path = "E:\\50Benchmark\\";
  vector<string>video_path_list, video_name_list;

  getFiles(benchmark_path, video_path_list, video_name_list);
  
  string mode = "simple video";
  if (argc == 2)
  {
    mode = argv[1];

  }

  if (mode == "all")
  {
    cout << ">> run_tracker('all')"<<endl;
    vector<double>all_precision, all_fps;
    double precision, fps;
    for (int i = 0; i < video_name_list.size(); i++)
    {
      string video_name = video_name_list[i];
      tracker(benchmark_path, video_name, precision, fps);
      all_precision.push_back(precision);
      all_fps.push_back(fps);
    }
    double mean_precision = std::accumulate(all_precision.begin(), all_precision.end(), 0.0) / double(all_precision.size());
    double mean_fps = std::accumulate(all_fps.begin(), all_fps.end(), 0.0) / double(all_fps.size());
    printf("\nAverage precision (20px):%1.3f, Average FPS:%4.2f\n\n", mean_precision, mean_fps);
  }
  else{
    int choice = 0;
    for (int i = 0; i < video_name_list.size(); i++)
    {
      std::printf(" %02d  %12s\n", i, video_name_list[i]);
    }
    cout << "\n\nChoice One Video!!" << endl;
    cin >> choice;
    if (choice >= video_name_list.size() || choice < 0)
    {
      cout << "No such video" << endl;
      return 0;
    }
    string video_name = video_name_list[choice];
    double precision, fps;
    tracker(benchmark_path, video_name, precision, fps);
  }
  system("PAUSE");
  return 0;
}



int tracker(string video_path, string video_name, double &precision, double &fps){

  vector<Rect> groundtruth_rect;
  vector<String>img_files;
  if (load_video_info(video_path, video_name, groundtruth_rect, img_files) != 1)
    return -1;

  double padding = 1;
  double output_sigma_factor = 1. / 16;
  double sigma = 0.2;
  double lambda = 1e-2;
  double interp_factor = 0.075;

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
  Size sz = scale_size(target_sz, (1.0 + padding));
  vector<Rect>result_rect;

  double output_sigma = sqrt(double(target_sz.area())) * output_sigma_factor;
  Mat y = CreateGaussian2(sz, output_sigma, CV_64F);
  Mat yf;
  dft(y, yf, DFT_COMPLEX_OUTPUT);

  Mat cos_window(sz, CV_64FC1);
  CalculateHann(cos_window, sz);

  Mat im;
  Mat im_gray;
  Mat z, new_z;
  Mat alphaf, new_alphaf;
  Mat x;
  Mat k, kf;
  Mat response;
  double time = 0;
  int64 tic, toc;
  for (int frame = 0; frame < img_files.size(); ++frame)
  {

    im = imread(img_files[frame], IMREAD_COLOR);
    im_gray = imread(img_files[frame], IMREAD_GRAYSCALE);
    if (resize_image){
      resize(im, im, im.size() / 2, 0, 0, INTER_CUBIC);
      resize(im_gray, im_gray, im.size() / 2, 0, 0, INTER_CUBIC);
    }

    tic = getTickCount();

    if (frame > 0)
    {
      GetSubWindow(im_gray, x, pos, sz, cos_window);
      DenseGaussKernel(sigma, x, z, k);
      cv::dft(k, kf, DFT_COMPLEX_OUTPUT);
      cv::idft(ComplexMul(alphaf, kf), response, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT
      Point maxLoc;
      minMaxLoc(response, NULL, NULL, NULL, &maxLoc);
      pos.x = pos.x - cvFloor(float(sz.width) / 2.0) + maxLoc.x + 1;
      pos.y = pos.y - cvFloor(float(sz.height) / 2.0) + maxLoc.y + 1;
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
    Rect rect_position(pos.x - target_sz.width / 2, pos.y - target_sz.height / 2, target_sz.width, target_sz.height);
    if (resize_image)
      result_rect.push_back(Rect(rect_position.x * 2, rect_position.y * 2, rect_position.width * 2, rect_position.height * 2));
    else
      result_rect.push_back(rect_position);

    rectangle(im, rect_position, Scalar(0, 255, 0), 2);
    putText(im, to_string(frame), Point(20, 40), 6, 1, Scalar(0, 255, 255), 2);
    imshow(video_name, im);
    char key = waitKey(1);
    if (key == 27)
      break;

  }
  time = time / getTickFrequency();
  vector<double>precisions = PrecisionCalculate(groundtruth_rect, result_rect);
  printf("%12s - Precision (20px):%1.3f, FPS:%4.2f\n", video_name, precisions[20], double(img_files.size()) / time);
  destroyAllWindows();
  precision = precisions[20];
  fps = double(img_files.size()) / time;
  return 0;
}


vector<double>PrecisionCalculate(vector<Rect>groundtruth_rect, vector<Rect>result_rect){
  int max_threshold = 50;
  vector<double>precisions(max_threshold+1, 0);
  if (groundtruth_rect.size() != result_rect.size()){
    int n = min(groundtruth_rect.size(), result_rect.size());
    groundtruth_rect.erase(groundtruth_rect.begin() + n, groundtruth_rect.end());
    result_rect.erase(groundtruth_rect.begin() + n, groundtruth_rect.end());
  }
  vector<double>distances;
  for (int  i = 0; i < result_rect.size(); i++)
  {
    double distemp = sqrt(double(pow(result_rect[i].x + result_rect[i].width / 2 - groundtruth_rect[i].x - groundtruth_rect[i].width / 2, 2) +
      pow(result_rect[i].y + result_rect[i].height / 2 - groundtruth_rect[i].y - groundtruth_rect[i].height / 2, 2)));
    distances.push_back(distemp);
  }
  for (int i = 0; i <= max_threshold; i++)
  {
    for (int j = 0; j < distances.size(); j++)
    {
      if (distances[j] < double(i))
        precisions[i]++;

    }
    precisions[i] = precisions[i] / distances.size();
  }
  return precisions;
}