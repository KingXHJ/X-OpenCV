#include <opencv2\opencv.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>

#define PI 3.1415926

void H_GrayPicture(cv::Mat src, cv::Mat dst);
void H_GuassianKernel(float sigma, int size, cv::Mat kernel);
void H_GaussianSmooth(cv::Mat src, cv::Mat dst, float sigma, int size);
void CalGradientXY(cv::Mat src, cv::Mat dstx, cv::Mat dsty);
void CalResponse(cv::Mat srcx, cv::Mat srcy, cv::Mat dst, int windowSize, float k);
void NMSThreshold(cv::Mat src, int threshold, int windowSize, cv::Mat dst);
void HarrisTest(cv::Mat& src, cv::Mat& dst, float sigma, int size, int windowSize, float k, int threshold);
