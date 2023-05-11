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
// // ע�����
// 1. ����Խ��
// 2. ���Ͳ�ƥ��
// 3. ͼƬ����ϵ�ͳ��������
// 4. ע����ڣ�С�ڣ����ڵ��ڣ�С�ڵ��ڵ��Ͻ���
// 5. ���������CV_32F����Ҫά��ԭͼ��Ҫ��CV_8UC1
// 6. ע�����
// 
//int main()
//{
//	//�����������������ȡ����֮��ľ���
//	Mat a = (Mat_<uchar>(5, 5) << 1, 2, 2, 4, 1,
//		3, 4, 1, 5, 2,
//		2, 3, 3, 2, 4,
//		4, 1, 5, 4, 6,
//		6, 3, 2, 1, 3);
//	IntegralGraph(a);
//	return 0;
//}