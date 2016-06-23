#include <opencv2/opencv.hpp>
using namespace cv;

#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;


int load_video_info(string videoPath, vector<Rect> &groundtruthRect,vector<String> &fileName);