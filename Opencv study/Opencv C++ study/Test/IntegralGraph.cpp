#include"IntegralGraph.h"

void IntegralGraph(cv::Mat& src)
{
	cv::Mat mid(src.rows, src.cols, CV_8U, cv::Scalar(0));
	cv::Mat dst(src.rows, src.cols, CV_8U, cv::Scalar(0));
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (i == 0)
			{
				if (j == 0)
				{
					mid.at<uchar>(i, j) = src.at<uchar>(i, j);
					dst.at<uchar>(i, j) = mid.at<uchar>(i, j);
				}
				else
				{
					mid.at<uchar>(i, j) = mid.at<uchar>(i, j - 1) + src.at<uchar>(i, j);
					dst.at<uchar>(i, j) = mid.at<uchar>(i, j);
				}
			}
			else
			{
				if (j == 0)
				{
					mid.at<uchar>(i, j) = src.at<uchar>(i, j);
					dst.at<uchar>(i, j) = dst.at<uchar>(i - 1, j) + mid.at<uchar>(i, j);
				}
				else
				{
					mid.at<uchar>(i, j) = mid.at<uchar>(i, j - 1) + src.at<uchar>(i, j);
					dst.at<uchar>(i, j) = dst.at<uchar>(i - 1, j) + mid.at<uchar>(i, j);
				}
			}
		}
	}
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			std::cout << int(dst.at<uchar>(i, j)) << std::endl;
		}
	}
}

// test
//#include"IntegralGraph.h"
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
//	Mat a = (Mat_<uchar>(5, 5) << 1, 2, 2, 4, 1,
//		3, 4, 1, 5, 2,
//		2, 3, 3, 2, 4,
//		4, 1, 5, 4, 6,
//		6, 3, 2, 1, 3);
//	IntegralGraph(a);
//	return 0;
//}