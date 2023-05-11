#include "MedianBlur.h"


// cv::Mat::at<type>(行号,列号)
// 请注意，单通道的必须这么写unsigned char，不能写成int,否则会有越界问题导致像素值变得紊乱
// img.at<unsigned char>(y, x) = 255;
void MedianBlur(cv::Mat& src, cv::Mat& dst, int kernel)
{
	int t = kernel * kernel / 2;
	int pad = kernel / 2;
	dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8U);
	cv::Mat mid(src.rows + pad * 2, src.cols + pad * 2, CV_8U, cv::Scalar(0)); // 注意这里是先行再列

	for (int i = pad; i < mid.rows - pad; i++)
	{
		for (int j = pad; j < mid.cols - pad; j++)
		{
			mid.at<uchar>(i, j) = src.at<uchar>(i - pad, j - pad);
		}
	}

	const int channels[] = { 0 };
	int dims = 1;//设置直方图维度
	const int histSize[] = { 256 }; //直方图每一个维度划分的柱条的数目
	//每一个维度取值范围
	float pranges[] = { 0, 255 };//取值区间
	const float* ranges[] = { pranges };
	cv::Mat ROI;
	cv::Mat hist;
	int nm = 0;
	int median = 0;

	for (int i = pad; i < mid.rows - pad; i++)
	{
		cv::Rect rect(0, i - pad, kernel, kernel); // Opencv中的x都是水平方向,也就是宽方向；y是竖直方向，也就是高方向
		ROI = mid(rect);
		hist = cv::Mat::zeros(cv::Size(256, 1), CV_8U);

		calcHist(&ROI, 1, channels, cv::Mat(), hist, dims, histSize, ranges, true, false);//计算直方图

		// std::cout << hist.type() << std::endl;// hist输出是CV_32F，哪怕一开始给hist设置的是CV_8U
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
					if (int(mid.at<uchar>(i + a, j - pad - 1)) <= median) // 非常震惊！！！如果没有等于号，那么算法得到的图像是黑白条纹的
					{
						nm = nm - 1;
					}
				}
				for (int a = - pad; a < pad + 1; a++)
				{
					hist.at<uchar>(int(mid.at<uchar>(i + a, j + pad))) = hist.at<uchar>(int(mid.at<uchar>(i + a, j + pad))) + 1;
					if (int(mid.at<uchar>(i + a, j + pad)) <= median) // 非常震惊！！！如果没有等于号，那么算法得到的图像是黑白条纹的
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
					while (nm < t && median != 255) // 保护性编程
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
					while (nm > t && median != 0) // 保护性编程
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