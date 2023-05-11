#include"RotateSmooth.h"

void RotateSmooth(cv::Mat src, cv::Mat dst, int kernel)
{
	src.convertTo(src, CV_8U);
	dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8U);
	int pad = kernel - 1;
	int n = kernel * kernel;
	int div = 1000000000, lighten = 0, d = 0, sum = 0;
	cv::Mat mid(src.rows + pad * 2, src.cols + pad * 2, CV_8U, cv::Scalar(0));
	for (int i = pad; i < src.rows - pad; i++)
	{
		for (int j = pad; j < src.cols - pad; j++)
		{
			mid.at<uchar>(i, j) = src.at<uchar>(i - pad, j - pad);
		}
	}
	for (int i = pad; i < mid.rows - pad; i++)
	{
		for (int j = pad; j < mid.cols - pad; j++)
		{
			div = 1000000000;
			lighten = 0;
			for (int k = 1; k < 9; k++)
			{
				d = 0;
				sum = 0;
				switch (k)
				{
				case 1:
					for (int a = 0; a < kernel; a++)
					{
						for (int b = 0; b < kernel; b++)
						{
							sum = sum + mid.at<uchar>(i - a, j + b);
						}
					}
					sum = sum / n;
					for (int a = 0; a < kernel; a++)
					{
						for (int b = 0; b < kernel; b++)
						{
							d = d + (mid.at<uchar>(i - a, j + b) - sum) * (mid.at<uchar>(i - a, j + b) - sum);
						}
					}
					d = d / n;
				case 2:
					d = 0;
					sum = 0;
					for (int a = 0; a < kernel; a++)
					{
						for (int b = 0; b < kernel; b++)
						{
							sum = sum + mid.at<uchar>(i + a, j + b);
						}
					}
					sum = sum / n;
					for (int a = 0; a < kernel; a++)
					{
						for (int b = 0; b < kernel; b++)
						{
							d = d + (mid.at<uchar>(i + a, j + b) - sum) * (mid.at<uchar>(i + a, j + b) - sum);
						}
					}
					d = d / n;
				case 3:
					d = 0;
					sum = 0;
					for (int a = 0; a < kernel; a++)
					{
						for (int b = 0; b < kernel; b++)
						{
							sum = sum + mid.at<uchar>(i + a, j - b);
						}
					}
					sum = sum / n;
					for (int a = 0; a < kernel; a++)
					{
						for (int b = 0; b < kernel; b++)
						{
							d = d + (mid.at<uchar>(i + a, j - b) - sum) * (mid.at<uchar>(i + a, j - b) - sum);
						}
					}
					d = d / n;
				case 4:
					d = 0;
					sum = 0;
					for (int a = 0; a < kernel; a++)
					{
						for (int b = 0; b < kernel; b++)
						{
							sum = sum + mid.at<uchar>(i - a, j - b);
						}
					}
					sum = sum / n;
					for (int a = 0; a < kernel; a++)
					{
						for (int b = 0; b < kernel; b++)
						{
							d = d + (mid.at<uchar>(i - a, j - b) - sum) * (mid.at<uchar>(i - a, j - b) - sum);
						}
					}
					d = d / n;
				case 5:
					d = 0;
					sum = 0;
					for (int a = 0; a < kernel; a++)
					{
						for (int b = -kernel / 2; b < kernel / 2 + 1; b++)
						{
							sum = sum + mid.at<uchar>(i + a, j + b);
						}
					}
					sum = sum / n;
					for (int a = 0; a < kernel; a++)
					{
						for (int b = -kernel / 2; b < kernel / 2 + 1; b++)
						{
							d = d + (mid.at<uchar>(i + a, j + b) - sum) * (mid.at<uchar>(i + a, j + b) - sum);
						}
					}
					d = d / n;
				case 6:
					d = 0;
					sum = 0;
					for (int a = -kernel / 2; a < kernel / 2 + 1; a++)
					{
						for (int b = 0; b < kernel; b++)
						{
							sum = sum + mid.at<uchar>(i - a, j + b);
						}
					}
					sum = sum / n;
					for (int a = -kernel / 2; a < kernel / 2 + 1; a++)
					{
						for (int b = 0; b < kernel; b++)
						{
							d = d + (mid.at<uchar>(i - a, j + b) - sum) * (mid.at<uchar>(i - a, j + b) - sum);
						}
					}
					d = d / n;
				case 7:
					d = 0;
					sum = 0;
					for (int a = -kernel / 2; a < kernel / 2 + 1; a++)
					{
						for (int b = 0; b < kernel; b++)
						{
							sum = sum + mid.at<uchar>(i + a, j - b);
						}
					}
					sum = sum / n;
					for (int a = -kernel / 2; a < kernel / 2 + 1; a++)
					{
						for (int b = 0; b < kernel; b++)
						{
							d = d + (mid.at<uchar>(i + a, j - b) - sum) * (mid.at<uchar>(i + a, j - b) - sum);
						}
					}
					d = d / n;
				case 8:
					d = 0;
					sum = 0;
					for (int a = 0; a < kernel; a++)
					{
						for (int b = -kernel / 2; b < kernel / 2 + 1; b++)
						{
							sum = sum + mid.at<uchar>(i - a, j - b);
						}
					}
					sum = sum / n;
					for (int a = 0; a < kernel; a++)
					{
						for (int b = -kernel / 2; b < kernel / 2 + 1; b++)
						{
							d = d + (mid.at<uchar>(i - a, j - b) - sum) * (mid.at<uchar>(i - a, j - b) - sum);
						}
					}
					d = d / n;
				}
				if (d < div)
				{
					div = d;
					lighten = sum;
				}
			}
			dst.at<uchar>(i - pad, j - pad) = lighten;
		}
	}
}

// test
//#include"RotateSmooth.h"
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
//	Mat gray = imread("2.jpg", IMREAD_GRAYSCALE);
//	Mat dst = cv::Mat::zeros(cv::Size(gray.cols, gray.rows), CV_8U);
//	RotateSmooth(gray, dst, 3);
//	imshow("gray", gray);
//	imshow("dst", dst);
//	waitKey(0);
//	return 0;
//}