#include"DistTransform.h"

float Euclidean_distance(int x1, int y1, int x2, int y2)
{
	float dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
	return dist;
}

float CityBlock_distance(int x1, int y1, int x2, int y2)
{
	float dist = abs(x1 - x2) + abs(y1 - y2);
	return dist;
}

float ChessBoard_distance(int x1, int y1, int x2, int y2)
{
	float dist = std::max(abs(x1 - x2), abs(y1 - y2));
	return dist;
}

void DistanceTransform(cv::Mat& src, cv::Mat& dst, int n, int padding)
{
	int pad = padding / 2;
	cv::Mat mid(src.rows + pad * 2, src.cols + pad * 2, CV_32F, cv::Scalar(0, 0, 0));
	src.convertTo(src, CV_32F);
	for (int i = 0; i < src.rows + pad * 2; i++)
	{
		for (int j = 0; j < src.cols + pad * 2; j++)
		{
			if (i < pad || j < pad || i > src.rows + pad - 1 || j > src.cols + pad - 1)
			{
				mid.at<float>(i, j) = INFINITY;
				continue;
			}
			if (src.at<float>(i - pad, j - pad) == 255)
			{
				mid.at<float>(i, j) = INFINITY;
			}
			else if (src.at<float>(i - pad, j - pad) == 0)
			{
				mid.at<float>(i, j) = 0;
			}
		}
	}
	switch (n)
	{
	case 1:
		/*for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				float ED1 = Euclidean_distance(i + pad, j + pad, i + pad - 1, j + pad - 1);
				float ED2 = Euclidean_distance(i + pad, j + pad, i + pad - 1, j + pad);
				float ED3 = Euclidean_distance(i + pad, j + pad, i + pad, j + pad - 1);
				float ED4 = Euclidean_distance(i + pad, j + pad, i + pad + 1, j + pad - 1);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), (ED1 + mid.at<float>(i + pad - 1, j + pad - 1))/1441*256);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), (ED2 + mid.at<float>(i + pad - 1, j + pad))/1441*256);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), (ED3 + mid.at<float>(i + pad, j + pad - 1))/1441*256);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), (ED4 + mid.at<float>(i + pad + 1, j + pad - 1))/1441*256);
			}
		}
		for (int i = src.rows - 1; i >= 0; i--)
		{
			for (int j = src.cols - 1; j >= 0; j--)
			{
				float ED1 = Euclidean_distance(i + pad, j + pad, i + pad - 1, j + pad + 1);
				float ED2 = Euclidean_distance(i + pad, j + pad, i + pad, j + pad + 1);
				float ED3 = Euclidean_distance(i + pad, j + pad, i + pad + 1, j + pad + 1);
				float ED4 = Euclidean_distance(i + pad, j + pad, i + pad + 1, j + pad);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), (ED1 + mid.at<float>(i + pad - 1, j + pad + 1))/1441*256);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), (ED2 + mid.at<float>(i + pad, j + pad + 1))/1441*256);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), (ED3 + mid.at<float>(i + pad + 1, j + pad + 1))/1441*256);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), (ED4 + mid.at<float>(i + pad + 1, j + pad))/1441*256);
			}
		}*/
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				float ED1 = Euclidean_distance(i + pad, j + pad, i + pad - 1, j + pad - 1);
				float ED2 = Euclidean_distance(i + pad, j + pad, i + pad - 1, j + pad);
				float ED3 = Euclidean_distance(i + pad, j + pad, i + pad, j + pad - 1);
				float ED4 = Euclidean_distance(i + pad, j + pad, i + pad + 1, j + pad - 1);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), ED1 + mid.at<float>(i + pad - 1, j + pad - 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), ED2 + mid.at<float>(i + pad - 1, j + pad));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), ED3 + mid.at<float>(i + pad, j + pad - 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), ED4 + mid.at<float>(i + pad + 1, j + pad - 1));
			}
		}
		for (int i = src.rows - 1; i >= 0; i--)
		{
			for (int j = src.cols - 1; j >= 0; j--)
			{
				float ED1 = Euclidean_distance(i + pad, j + pad, i + pad - 1, j + pad + 1);
				float ED2 = Euclidean_distance(i + pad, j + pad, i + pad, j + pad + 1);
				float ED3 = Euclidean_distance(i + pad, j + pad, i + pad + 1, j + pad + 1);
				float ED4 = Euclidean_distance(i + pad, j + pad, i + pad + 1, j + pad);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), ED1 + mid.at<float>(i + pad - 1, j + pad + 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), ED2 + mid.at<float>(i + pad, j + pad + 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), ED3 + mid.at<float>(i + pad + 1, j + pad + 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), ED4 + mid.at<float>(i + pad + 1, j + pad));
			}
		}
		break;
	case 2:
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				float CB1 = CityBlock_distance(i + pad, j + pad, i + pad - 1, j + pad - 1);
				float CB2 = CityBlock_distance(i + pad, j + pad, i + pad - 1, j + pad);
				float CB3 = CityBlock_distance(i + pad, j + pad, i + pad, j + pad - 1);
				float CB4 = CityBlock_distance(i + pad, j + pad, i + pad + 1, j + pad - 1);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB1 + mid.at<float>(i + pad - 1, j + pad - 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB2 + mid.at<float>(i + pad - 1, j + pad));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB3 + mid.at<float>(i + pad, j + pad - 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB4 + mid.at<float>(i + pad + 1, j + pad - 1));
			}
		}
		for (int i = src.rows - 1; i >= 0; i--)
		{
			for (int j = src.cols - 1; j >= 0; j--)
			{
				float CB1 = CityBlock_distance(i + pad, j + pad, i + pad - 1, j + pad + 1);
				float CB2 = CityBlock_distance(i + pad, j + pad, i + pad, j + pad + 1);
				float CB3 = CityBlock_distance(i + pad, j + pad, i + pad + 1, j + pad + 1);
				float CB4 = CityBlock_distance(i + pad, j + pad, i + pad + 1, j + pad);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB1 + mid.at<float>(i + pad - 1, j + pad + 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB2 + mid.at<float>(i + pad, j + pad + 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB3 + mid.at<float>(i + pad + 1, j + pad + 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB4 + mid.at<float>(i + pad + 1, j + pad));
			}
		}
		break;
	case 3:
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				float CB1 = ChessBoard_distance(i + pad, j + pad, i + pad - 1, j + pad - 1);
				float CB2 = ChessBoard_distance(i + pad, j + pad, i + pad - 1, j + pad);
				float CB3 = ChessBoard_distance(i + pad, j + pad, i + pad, j + pad - 1);
				float CB4 = ChessBoard_distance(i + pad, j + pad, i + pad + 1, j + pad - 1);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB1 + mid.at<float>(i + pad - 1, j + pad - 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB2 + mid.at<float>(i + pad - 1, j + pad));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB3 + mid.at<float>(i + pad, j + pad - 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB4 + mid.at<float>(i + pad + 1, j + pad - 1));
			}
		}
		for (int i = src.rows - 1; i >= 0; i--)
		{
			for (int j = src.cols - 1; j >= 0; j--)
			{
				float CB1 = ChessBoard_distance(i + pad, j + pad, i + pad - 1, j + pad + 1);
				float CB2 = ChessBoard_distance(i + pad, j + pad, i + pad, j + pad + 1);
				float CB3 = ChessBoard_distance(i + pad, j + pad, i + pad + 1, j + pad + 1);
				float CB4 = ChessBoard_distance(i + pad, j + pad, i + pad + 1, j + pad);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB1 + mid.at<float>(i + pad - 1, j + pad + 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB2 + mid.at<float>(i + pad, j + pad + 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB3 + mid.at<float>(i + pad + 1, j + pad + 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB4 + mid.at<float>(i + pad + 1, j + pad));
			}
		}
		break;
	default:
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				float CB1 = Euclidean_distance(i + pad, j + pad, i + pad - 1, j + pad - 1);
				float CB2 = Euclidean_distance(i + pad, j + pad, i + pad - 1, j + pad);
				float CB3 = Euclidean_distance(i + pad, j + pad, i + pad, j + pad - 1);
				float CB4 = Euclidean_distance(i + pad, j + pad, i + pad + 1, j + pad - 1);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB1 + mid.at<float>(i + pad - 1, j + pad - 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB2 + mid.at<float>(i + pad - 1, j + pad));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB3 + mid.at<float>(i + pad, j + pad - 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB4 + mid.at<float>(i + pad + 1, j + pad - 1));
			}
		}
		for (int i = src.rows - 1; i >= 0; i--)
		{
			for (int j = src.cols - 1; j >= 0; j--)
			{
				float CB1 = Euclidean_distance(i + pad, j + pad, i + pad - 1, j + pad + 1);
				float CB2 = Euclidean_distance(i + pad, j + pad, i + pad, j + pad + 1);
				float CB3 = Euclidean_distance(i + pad, j + pad, i + pad + 1, j + pad + 1);
				float CB4 = Euclidean_distance(i + pad, j + pad, i + pad + 1, j + pad);
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB1 + mid.at<float>(i + pad - 1, j + pad + 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB2 + mid.at<float>(i + pad, j + pad + 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB3 + mid.at<float>(i + pad + 1, j + pad + 1));
				mid.at<float>(i + pad, j + pad) = std::min(mid.at<float>(i + pad, j + pad), CB4 + mid.at<float>(i + pad + 1, j + pad));
			}
		}
		break;
	}
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<float>(i, j) = mid.at<float>(i + pad, j + pad);
		}
	}
}

// Opencv test
//#include <opencv2\opencv.hpp>
//#include <iostream>
//using namespace cv;
//using namespace std;
//
// // 注意断言
// 1. 数组越界
// 2. 类型不匹配
// 3. 图片坐标系和长宽定义混淆
// 4. 注意大于，小于，大于等于，小于等于的严谨性
// 5. 不能随便用CV_32F，想要维持原图，要用CV_8UC1
// 6. 注意除零
// 
//int main()
//{
//	//构建建议矩阵，用于求取像素之间的距离
//	Mat a = (Mat_<uchar>(5, 5) << 1, 1, 1, 1, 1,
//		1, 1, 1, 1, 1,
//		1, 1, 0, 1, 1,
//		1, 1, 1, 1, 1,
//		1, 1, 1, 1, 1);
//	Mat dist_L1, dist_L2, dist_C, dist_L12;
//
//	//计算街区距离
//	distanceTransform(a, dist_L1, 1, 3, CV_8U);
//	cout << "街区距离：" << endl << dist_L1 << endl;
//
//	//计算欧式距离
//	distanceTransform(a, dist_L2, 2, 3, CV_8U);
//	cout << "欧式距离：" << endl << dist_L2 << endl;
//
//	//计算棋盘距离
//	distanceTransform(a, dist_C, 3, 5, CV_8U);
//	cout << "棋盘距离：" << endl << dist_C << endl;
//
//	//对图像进行距离变换
//	Mat rice = imread("1.jpg", IMREAD_GRAYSCALE);
//	if (rice.empty())
//	{
//		cout << "请确认图像文件名称是否正确" << endl;
//		return -1;
//	}
//	Mat riceBW, riceBW_INV;
//
//	//将图像转成二值图像，同时把黑白区域颜色呼唤
//	threshold(rice, riceBW, 50, 255, THRESH_BINARY);
//	threshold(rice, riceBW_INV, 50, 255, THRESH_BINARY_INV);
//
//	//距离变换
//	Mat dist, dist_INV;
//	distanceTransform(riceBW, dist, 1, 3, CV_32F);  //为了显示清晰，将数据类型变成CV_32F
//	distanceTransform(riceBW_INV, dist_INV, 1, 3, CV_8U);
//
//	//显示变换结果
//	imshow("riceBW", riceBW);
//	imshow("dist", dist);
//	imshow("riceBW_INV", riceBW_INV);
//	imshow("dist_INV", dist_INV);
//
//	waitKey(0);
//	return 0;
//}

// 自主函数测试
//#include"DistTransform.h"
//using namespace cv;
//using namespace std;
//
//int main()
//{
//	//构建建议矩阵，用于求取像素之间的距离
//	Mat a = (Mat_<uchar>(5, 5) << 1, 1, 1, 1, 1,
//		1, 1, 1, 1, 1,
//		1, 1, 0, 1, 1,
//		1, 1, 1, 1, 1,
//		1, 1, 1, 1, 1);
//	Mat dist_L1(a.rows, a.cols, CV_32F, cv::Scalar(0, 0, 0));
//	Mat dist_L2(a.rows, a.cols, CV_32F, cv::Scalar(0, 0, 0));
//	Mat dist_C(a.rows, a.cols, CV_32F, cv::Scalar(0, 0, 0));
//	Mat dist_L12(a.rows, a.cols, CV_32F, cv::Scalar(0, 0, 0));
//
//	//计算街区距离
//	DistanceTransform(a, dist_L1, 1, 3);
//	cout << "欧式距离：" << endl << dist_L1 << endl;
//
//	//计算欧式距离
//	DistanceTransform(a, dist_L2, 2, 3);
//	cout << "街区距离：" << endl << dist_L2 << endl;
//
//	//计算棋盘距离
//	DistanceTransform(a, dist_C, 3, 5);
//	cout << "棋盘距离：" << endl << dist_C << endl;
//
//	waitKey(0);
//	return 0;
//}

//#include"DistTransform.h"
//using namespace cv;
//using namespace std;
//
//int main()
//{
//	//对图像进行距离变换
//	Mat rice = imread("1.jpg", IMREAD_GRAYSCALE);
//	if (rice.empty())
//	{
//		cout << "请确认图像文件名称是否正确" << endl;
//		return -1;
//	}
//	rice.convertTo(rice, CV_32F);
//	Mat riceBW(rice.rows, rice.cols, CV_32F, cv::Scalar(0, 0, 0));
//	Mat riceBW_INV(rice.rows, rice.cols, CV_32F, cv::Scalar(0, 0, 0));
//
//	//将图像转成二值图像，同时把黑白区域颜色呼唤
//	threshold(rice, riceBW, 50, 255, THRESH_BINARY);
//	threshold(rice, riceBW_INV, 50, 255, THRESH_BINARY_INV);
//
//	//距离变换
//	Mat dist(rice.rows, rice.cols, CV_32F, cv::Scalar(0, 0, 0));
//	Mat dist_INV(rice.rows, rice.cols, CV_32F, cv::Scalar(0, 0, 0));
//	DistanceTransform(riceBW, dist, 1, 3);  //为了显示清晰，将数据类型变成CV_32F
//	DistanceTransform(riceBW_INV, dist_INV, 1, 3);
//	dist_INV.convertTo(dist_INV, CV_8U); // CV_32F显示太过于清晰
//
//	//显示变换结果
//	imshow("riceBW", riceBW);
//	imshow("dist", dist);
//	imshow("riceBW_INV", riceBW_INV);
//	imshow("dist_INV", dist_INV);
//
//	waitKey(0);
//	return 0;
//}