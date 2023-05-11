#include <opencv2\opencv.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>

#define PI 3.1415926

void GrayPicture(cv::Mat src, cv::Mat dst);
void GuassianKernel(float sigma, int size, cv::Mat kernel);
void GaussianSmooth(cv::Mat src, cv::Mat dst, float sigma, int size);
void CalGradient(cv::Mat src, cv::Mat dst, cv::Mat angle);
void NMS(cv::Mat src, cv::Mat angle, cv::Mat dst);
void ThresholdCheck(cv::Mat& src, cv::Mat& dst, int thresholdLow, int thresholdHigh);
void CannyTest(cv::Mat& src, cv::Mat& dst, float sigma, int size, int thresholdLow, int thresholdHigh);
