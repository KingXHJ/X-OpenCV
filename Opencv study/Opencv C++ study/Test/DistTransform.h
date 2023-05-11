#include <opencv2\opencv.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>

float Euclidean_distance(int x1, int y1, int x2, int y2);
float CityBlock_distance(int x1, int y1, int x2, int y2);
float ChessBoard_distance(int x1, int y1, int x2, int y2);

void DistanceTransform(cv::Mat& src, cv::Mat& dst, int n, int padding);