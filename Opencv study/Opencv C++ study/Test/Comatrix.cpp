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
//	Mat a = (Mat_<uchar>(4, 4) << 0, 0, 1, 1,
//		0, 0, 1, 1,
//		0, 2, 2, 2,
//		2, 2, 3, 3);
//	int n = 1, start = 0, end = 3;
//	CalComatrix(a, n, start, end);
//	return 0;
//}