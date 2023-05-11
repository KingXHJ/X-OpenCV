#include"Comatrix.h"

void CalComatrix(cv::Mat& src, int n, int start, int end)
{
	cv::Mat mid(end - start + 1, end - start + 1, CV_8U, cv::Scalar(0));
	switch (n)
	{
	case 1:
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (j == 0)
				{
					mid.at<uchar>(src.at<uchar>(i, j), src.at<uchar>(i, j + 1)) = mid.at<uchar>(src.at<uchar>(i, j), src.at<uchar>(i, j + 1)) + 1;
				}
				else if (j == src.cols - 1)
				{
					mid.at<uchar>(src.at<uchar>(i, j), src.at<uchar>(i, j - 1)) = mid.at<uchar>(src.at<uchar>(i, j), src.at<uchar>(i, j - 1)) + 1;
				}
				else
				{
					mid.at<uchar>(src.at<uchar>(i, j), src.at<uchar>(i, j + 1)) = mid.at<uchar>(src.at<uchar>(i, j), src.at<uchar>(i, j + 1)) + 1;
					mid.at<uchar>(src.at<uchar>(i, j), src.at<uchar>(i, j - 1)) = mid.at<uchar>(src.at<uchar>(i, j), src.at<uchar>(i, j - 1)) + 1;
				}
			}
		}
		for (int i = 0; i < end - start + 1; i++)
		{
			for (int j = 0; j < end - start + 1; j++)
			{
				std::cout << int(mid.at<uchar>(i, j)) << std::endl;
			}
		}
	}
}

// test
//#include"Comatrix.h"
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
//	Mat a = (Mat_<uchar>(4, 4) << 0, 0, 1, 1,
//		0, 0, 1, 1,
//		0, 2, 2, 2,
//		2, 2, 3, 3);
//	int n = 1, start = 0, end = 3;
//	CalComatrix(a, n, start, end);
//	return 0;
//}