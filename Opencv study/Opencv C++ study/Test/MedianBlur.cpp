#include "MedianBlur.h"


// cv::Mat::at<type>(�к�,�к�)
// ��ע�⣬��ͨ���ı�����ôдunsigned char������д��int,�������Խ�����⵼������ֵ�������
// img.at<unsigned char>(y, x) = 255;
void MedianBlur(cv::Mat& src, cv::Mat& dst, int kernel)
{
	int t = kernel * kernel / 2;
	int pad = kernel / 2;
	dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8U);
	cv::Mat mid(src.rows + pad * 2, src.cols + pad * 2, CV_8U, cv::Scalar(0)); // ע����������������

	for (int i = pad; i < mid.rows - pad; i++)
	{
		for (int j = pad; j < mid.cols - pad; j++)
		{
			mid.at<uchar>(i, j) = src.at<uchar>(i - pad, j - pad);
		}
	}

	const int channels[] = { 0 };
	int dims = 1;//����ֱ��ͼά��
	const int histSize[] = { 256 }; //ֱ��ͼÿһ��ά�Ȼ��ֵ���������Ŀ
	//ÿһ��ά��ȡֵ��Χ
	float pranges[] = { 0, 255 };//ȡֵ����
	const float* ranges[] = { pranges };
	cv::Mat ROI;
	cv::Mat hist;
	int nm = 0;
	int median = 0;

	for (int i = pad; i < mid.rows - pad; i++)
	{
		cv::Rect rect(0, i - pad, kernel, kernel); // Opencv�е�x����ˮƽ����,Ҳ���ǿ���y����ֱ����Ҳ���Ǹ߷���
		ROI = mid(rect);
		hist = cv::Mat::zeros(cv::Size(256, 1), CV_8U);

		calcHist(&ROI, 1, channels, cv::Mat(), hist, dims, histSize, ranges, true, false);//����ֱ��ͼ

		// std::cout << hist.type() << std::endl;// hist�����CV_32F������һ��ʼ��hist���õ���CV_8U
		hist.convertTo(hist, CV_8U); 

		// cal median
		nm = 0;
		median = 0;
		for (int k = 0; k < 256; k++)
		{
			nm = nm + int(hist.at<uchar>(k));
			if (nm >= t)
			{
				median = k;
				break;
			}
		}

		for (int j = pad; j < mid.cols - pad; j++)
		{
			if (j == pad)
			{
				dst.at<uchar>(i - pad, j - pad) = median;
			}
			else
			{
				for (int a = - pad; a < pad + 1; a++)
				{
					hist.at<uchar>(int(mid.at<uchar>(i + a, j - pad - 1))) = hist.at<uchar>(int(mid.at<uchar>(i + a, j - pad - 1))) - 1;
					if (int(mid.at<uchar>(i + a, j - pad - 1)) <= median) // �ǳ��𾪣��������û�е��ںţ���ô�㷨�õ���ͼ���Ǻڰ����Ƶ�
					{
						nm = nm - 1;
					}
				}
				for (int a = - pad; a < pad + 1; a++)
				{
					hist.at<uchar>(int(mid.at<uchar>(i + a, j + pad))) = hist.at<uchar>(int(mid.at<uchar>(i + a, j + pad))) + 1;
					if (int(mid.at<uchar>(i + a, j + pad)) <= median) // �ǳ��𾪣��������û�е��ںţ���ô�㷨�õ���ͼ���Ǻڰ����Ƶ�
					{
						nm = nm + 1;
					}
				}
				if (nm == t)
				{
					dst.at<uchar>(i - pad, j - pad) = median;
				}
				else if (nm < t)
				{
					while (nm < t && median != 255) // �����Ա��
					{
						median = median + 1;
						nm = nm + int(hist.at<uchar>(median));
						if (nm >= t)
						{
							break;
						}
					}
					dst.at<uchar>(i - pad, j - pad) = median;
				}
				else if (nm > t)
				{
					while (nm > t && median != 0) // �����Ա��
					{
						nm = nm - int(hist.at<uchar>(median));
						median = median - 1;
						if (nm <= t)
						{
							break;
						}
					}
					dst.at<uchar>(i - pad, j - pad) = median;
				}
			}
		}
	}

}

// test
//#include"MedianBlur.h"
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
//	Mat gray = imread("2.jpg", IMREAD_GRAYSCALE);
//	Mat dst = cv::Mat::zeros(cv::Size(gray.cols, gray.rows), CV_8U);
//	Mat dst_test = cv::Mat::zeros(cv::Size(gray.cols, gray.rows), CV_8U);
//
//	medianBlur(gray, dst, 3); // Opencv
//	MedianBlur(gray, dst_test, 3);
//
//	imshow("gray", gray);
//	imshow("dst", dst);
//	imshow("dst_test", dst_test);
//
//	waitKey(0);
//	return 0;
//}