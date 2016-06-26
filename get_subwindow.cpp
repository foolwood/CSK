#include "get_subwindow.h"
#define PI 3.141592653589793

void getSubWindow(Mat &frame, Mat &subWindow, Point centraCoor, Size sz, Mat &cos_window){
	Point lefttop;
	lefttop.x = max(centraCoor.x - (sz.width>>1), 0);
	lefttop.y = max(centraCoor.y - (sz.height>>1), 0);
	Point rightbottom;
	rightbottom.x = min(centraCoor.x + int(ceil(float(sz.width) / 2.0)), frame.cols - 1);
	rightbottom.y = min(centraCoor.y + int(ceil(float(sz.height) / 2.0)), frame.rows - 1);
	Rect roiRect(lefttop, rightbottom);
	frame(roiRect).copyTo(subWindow);
	cv::Rect border(-min(centraCoor.x - (sz.width >> 1), 0), -min(centraCoor.y - (sz.height >> 1), 0),
		max(centraCoor.x + int(ceil(float(sz.width) / 2.0)) - (frame.cols - 1), 0), max(centraCoor.y + int(ceil(float(sz.height) / 2.0)) - (frame.rows - 1), 0));

	if (border != Rect(0,0,0,0))
	{
		cv::copyMakeBorder(subWindow, subWindow, border.y, border.height, border.x, border.width, cv::BORDER_CONSTANT);
	}

	subWindow.convertTo(subWindow, CV_32FC1,1.0/255.0,-0.5);
	subWindow = subWindow.mul(cos_window);
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
	xx = xx*xx;
	double yy = norm(y);
	yy = yy*yy;

	Mat yf_conj;
	vector<Mat> planes;
	split(yf, planes);
	planes[1] = planes[1] * -1;
	merge(planes, yf_conj);
	Mat xyf = complexMul(xf, yf_conj);

	Mat xy;
	cv::idft(xyf, xy, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT
	double numelx1 = x.cols*x.rows;
	exp(-1 / (sigma*sigma) * abs((xx + yy - 2 * xy) / numelx1), k);
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
		double x = i - floor(n / 2);
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

cv::Mat complexMul(Mat x1, Mat x2)
{
	vector<Mat> planes1;
	split(x1, planes1);
	vector<Mat> planes2;
	split(x2, planes2);
	vector<Mat>complex(2);
	complex[0] = planes1[0].mul(planes2[0]) - planes1[1].mul(planes2[1]);
	complex[1] = planes1[0].mul(planes2[1]) + planes1[1].mul(planes2[0]);
	Mat result;
	merge(complex, result);
	return result;
}

cv::Mat complexDiv(Mat x1, Mat x2)
{
	vector<Mat> planes1;
	split(x1, planes1);
	vector<Mat> planes2;
	split(x2, planes2);
	vector<Mat>complex(2);
	Mat cc = planes2[0].mul(planes2[0]);
	Mat dd = planes2[1].mul(planes2[1]);

	complex[0] = (planes1[0].mul(planes2[0]) + planes1[1].mul(planes2[1])) / (cc + dd);
	complex[1] = (planes1[0].mul(planes2[1]) - planes1[1].mul(planes2[0])) / (cc + dd);
	Mat result;
	merge(complex, result);
	return result;
}