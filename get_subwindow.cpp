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


void calculateHann(Mat &cos_window, Size sz){
	Mat temp1(Size(sz.width, 1), CV_32FC1);
	Mat temp2(Size(sz.height, 1), CV_32FC1);
	for (int i = 0; i < sz.width; ++i)
		temp1.at<float>(0, i) = 0.5*(1 - cos(2 * PI*i / (sz.width - 1)));
	for (int i = 0; i < sz.height; ++i)
		temp2.at<float>(0, i) = 0.5*(1 - cos(2 * PI*i / (sz.height - 1)));
	cos_window = temp2.t()*temp1;
}

void denseGaussKernel(float sigma, Mat x, Mat y, Mat &k){
	Mat xf = fft(x);
	Mat yf = fft(y);
	double xx = norm(x);
	double yy = norm(y);

	Mat xyf,yf_conj;
	vector<Mat> planes;
	split(yf, planes);
	planes[1] = planes[1] * -1;
	merge(planes,yf_conj);
	xyf = xf.dot(yf_conj);

	Mat xy;
	idft(xyf, xy);

	int cx = xy.cols / 2;
	int cy = xy.rows / 2;

	Mat q0(xy, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(xy, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(xy, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(xy, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
	split(xy, planes);
	xy = planes[0];
	exp(-1 / (sigma*sigma) * max(0, (xx + yy - 2 * xy) / (x.cols*x.rows)), k);
}


cv::Mat getGaussian(int n, double sigma, int ktype)
{
	const int SMALL_GAUSSIAN_SIZE = 7;
	static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] =
	{
		{ 1.f },
		{ 0.25f, 0.5f, 0.25f },
		{ 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f },
		{ 0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f }
	};

	const float* fixed_kernel = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0 ?
		small_gaussian_tab[n >> 1] : 0;

	CV_Assert(ktype == CV_32F || ktype == CV_64F);
	Mat kernel(n, 1, ktype);
	float* cf = kernel.ptr<float>();
	double* cd = kernel.ptr<double>();

	double sigmaX = sigma > 0 ? sigma : ((n - 1)*0.5 - 1)*0.3 + 0.8;
	double scale2X = -0.5 / (sigmaX*sigmaX);
	double sum = 0;

	int i;
	for (i = 0; i < n; i++)
	{
		double x = i - (n - 1)*0.5;
		double t = fixed_kernel ? (double)fixed_kernel[i] : std::exp(scale2X*x*x);
		if (ktype == CV_32F)
		{
			cf[i] = (float)t;
			sum += cf[i];
		}
		else
		{
			cd[i] = t;
			sum += cd[i];
		}
	}

	return kernel;
}

cv::Mat getGaussian2(Size sz, double sigma, int ktype)
{
	Mat a = getGaussian(sz.height, sigma, ktype);
	Mat b = getGaussian(sz.width, sigma, ktype);
	return a*b.t();
}

cv::Mat fft(Mat x)
{
	Mat planes[] = { Mat_<float>(x), Mat::zeros(x.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix
	return complexI;
}