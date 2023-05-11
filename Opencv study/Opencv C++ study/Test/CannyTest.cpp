#include "CannyTest.h"

void GrayPicture(cv::Mat src, cv::Mat dst)
{
	// 方法一
	/*cv::Mat b(src.rows, src.cols, CV_8UC1);
	cv::Mat g(src.rows, src.cols, CV_8UC1);
	cv::Mat r(src.rows, src.cols, CV_8UC1);
	cv::Mat out[] = { b,g,r };
	cv::split(src, out);*/
	
	// 方法二
	cv::Mat different_Channels[3];//declaring a matrix with three channels//  
	cv::split(src, different_Channels);//splitting images into 3 different channels//  
	cv::Mat b = different_Channels[0];//loading blue channels//
	cv::Mat g = different_Channels[1];//loading green channels//
	cv::Mat r = different_Channels[2];//loading red channels//  

	// 灰度化
	dst = 0.299 * r + 0.587 * g + 0.114 * b;
}

void GuassianKernel(float sigma, int size, cv::Mat kernel)
{
	int pad = size / 2;
	kernel = cv::Mat::zeros(cv::Size(size, size), CV_32F);
	float cnt = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			kernel.at<float>(i, j) = 1 / (2 * PI * sigma * sigma) * exp(-((i - pad) * (i - pad) + (j - pad) * (j - pad)) / (2 * sigma * sigma)); // kernel的坐标是以中心为原点的，而不是左上角
			//std::cout << kernel.at<float>(i, j) << std::endl;
			//cnt = cnt + kernel.at<float>(i, j);
		}
	}
	//std::cout << cnt << std::endl;
}

void GaussianSmooth(cv::Mat src, cv::Mat dst, float sigma, int size)
{
	int pad = size / 2;
	cv::Mat kernel(size, size, CV_32F, cv::Scalar(0));
	cv::Mat mid(src.rows + pad * 2, src.cols + pad * 2, CV_32F, cv::Scalar(0));
	dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	src.convertTo(src, CV_32F);

	GuassianKernel(sigma, size, kernel);

	for (int i = pad; i < mid.rows - pad; i++)
	{
		for (int j = pad; j < mid.cols - pad; j++)
		{
			mid.at<float>(i, j) = src.at<float>(i - pad, j - pad);
		}
	}

	for (int i = pad; i < mid.rows - pad; i++)
	{
		for (int j = pad; j < mid.cols - pad; j++)
		{
			for (int a = - pad; a < pad + 1; a++)
			{
				for (int b = - pad; b < pad + 1; b++)
				{
					// 不能随便用CV_32F，想要维持原图，要用CV_8UC1
					dst.at<uchar>(i - pad, j - pad) = dst.at<uchar>(i - pad, j - pad) + int(mid.at<float>(i + a, j + b) * kernel.at<float>(a + pad, b + pad));
				}
			}
			//std::cout << dst.at<float>(i - pad, j - pad) << std::endl;
		}
	}

	//dst.convertTo(dst, CV_8UC1);
}

// Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。因此要使用32位浮点型，即CV_32F。
// 这里懒得改了
void CalGradient(cv::Mat src, cv::Mat dst, cv::Mat angle)
{
	int size = 3;
	int pad = size / 2;
	cv::Mat x(src.rows + pad * 2, src.cols + pad * 2, CV_8UC1, cv::Scalar(0));
	cv::Mat y(src.rows + pad * 2, src.cols + pad * 2, CV_8UC1, cv::Scalar(0));
	cv::Mat kernelx = (cv::Mat_<uchar>(size, size) << 1, 0, -1,
		2, 0, -2,
		1, 0, -1);
	cv::Mat kernely = (cv::Mat_<uchar>(size, size) << 1, 2, 1,
		0, 0, 0,
		-1, -2, -1);
	dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	angle = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_32F);
	src.convertTo(src, CV_8UC1);

	for (int i = pad; i < x.rows - pad; i++)
	{
		for (int j = pad; j < x.cols - pad; j++)
		{
			x.at<uchar>(i, j) = src.at<uchar>(i - pad, j - pad);
		}
	}
	for (int i = pad; i < y.rows - pad; i++)
	{
		for (int j = pad; j < y.cols - pad; j++)
		{
			y.at<uchar>(i, j) = src.at<uchar>(i - pad, j - pad);
		}
	}

	for (int i = pad; i < x.rows - pad; i++)
	{
		for (int j = pad; j < x.cols - pad; j++)
		{
			for (int a = -pad; a < pad + 1; a++)
			{
				for (int b = -pad; b < pad + 1; b++)
				{
					// 不能随便用CV_32F，想要维持原图，要用CV_8UC1
					x.at<uchar>(i - pad, j - pad) = x.at<uchar>(i - pad, j - pad) + x.at<uchar>(i + a, j + b) * kernelx.at<uchar>(a + pad, b + pad);
				}
			}
			//std::cout << dst.at<float>(i - pad, j - pad) << std::endl;
		}
	}
	for (int i = pad; i < y.rows - pad; i++)
	{
		for (int j = pad; j < y.cols - pad; j++)
		{
			for (int a = -pad; a < pad + 1; a++)
			{
				for (int b = -pad; b < pad + 1; b++)
				{
					// 不能随便用CV_32F，想要维持原图，要用CV_8UC1
					y.at<uchar>(i - pad, j - pad) = y.at<uchar>(i - pad, j - pad) + y.at<uchar>(i + a, j + b) * kernely.at<uchar>(a + pad, b + pad);
				}
			}
			//std::cout << dst.at<float>(i - pad, j - pad) << std::endl;
		}
	}

	for (int i = pad; i < y.rows - pad; i++)
	{
		for (int j = pad; j < y.cols - pad; j++)
		{
			dst.at<uchar>(i - pad, j - pad) = std::abs(x.at<uchar>(i, j)) + std::abs(y.at<uchar>(i, j));
			// 注意除数是零
			if (int(x.at<uchar>(i, j)) == 0)
			{
				if (int(y.at<uchar>(i, j)) > 0)
				{
					angle.at<float>(i - pad, j - pad) = PI / 2;
				}
				else if (int(y.at<uchar>(i, j)) < 0)
				{
					angle.at<float>(i - pad, j - pad) = - PI / 2;
				}
				else
				{
					angle.at<float>(i - pad, j - pad) = 0;
				}
			}
			else
			{
				angle.at<float>(i - pad, j - pad) = std::atan(y.at<uchar>(i, j) / x.at<uchar>(i, j)); // 弧度
			}
		}
	}
}

// 重点：Canny中的非极大值抑制是沿着梯度方向对幅值进行非极大值抑制，而非边缘方向。

void NMS(cv::Mat src, cv::Mat angle, cv::Mat dst)
{
	int pad = 1;
	int dTmp1 = 0;
	int dTmp2 = 0;
	src.convertTo(src, CV_8UC1);
	angle.convertTo(angle, CV_32F);
	dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	cv::Mat mid(src.rows + pad * 2, src.cols + pad * 2, CV_8UC1, cv::Scalar(0));

	for (int i = pad; i < mid.rows - pad; i++)
	{
		for (int j = pad; j < mid.cols - pad; j++)
		{
			mid.at<uchar>(i, j) = src.at<uchar>(i - pad, j - pad);
		}
	}
	
	int theta = 0;
	int g1 = 0, g2 = 0, g3 = 0, g4 = 0, g5 = 0, g6 = 0, g7 = 0, g8 = 0, g9 = 0;

	for (int i = pad; i < mid.rows - pad; i++)
	{
		for (int j = pad; j < mid.cols - pad; j++)
		{
			theta = angle.at<float>(i - pad, j - pad);
			g1 = mid.at<uchar>(i - pad, j - pad);
			g2 = mid.at<uchar>(i - pad, j);
			g3 = mid.at<uchar>(i - pad, j + pad);
			g4 = mid.at<uchar>(i, j - pad);
			g5 = mid.at<uchar>(i, j);
			g6 = mid.at<uchar>(i, j + pad);
			g7 = mid.at<uchar>(i + pad, j - pad);
			g8 = mid.at<uchar>(i + pad, j);
			g9 = mid.at<uchar>(i + pad, j + pad);
			if (std::tan(theta) < -1)
			{
				dTmp1 = (g1 - g2) / std::abs(std::tan(theta)) + g2;
				dTmp2 = (g9 - g8) / std::abs(std::tan(theta)) + g8;
			}
			else if (std::tan(theta) > 1)
			{
				dTmp1 = (g3 - g2) / std::abs(std::tan(theta)) + g2;
				dTmp2 = (g7 - g8) / std::abs(std::tan(theta)) + g8;
			}
			else if (std::tan(theta) > 0 && std::tan(theta) <= 1)
			{
				dTmp1 = (g3 - g6) * std::abs(std::tan(theta)) + g6;
				dTmp2 = (g7 - g4) * std::abs(std::tan(theta)) + g4;
			}
			else if (std::tan(theta) < 0 && std::tan(theta) >= -1)
			{
				dTmp1 = (g1 - g4) * std::abs(std::tan(theta)) + g4;
				dTmp2 = (g9 - g6) * std::abs(std::tan(theta)) + g6;
			}
			else if (std::tan(theta) == 0)
			{
				dTmp1 = g4;
				dTmp2 = g6;
			}
			if (g5 > dTmp1 && g5 > dTmp2)
			{
				dst.at<uchar>(i - pad, j - pad) = src.at<uchar>(i - pad, j - pad);
			}
			else
			{
				dst.at<uchar>(i - pad, j - pad) = 0;
			}
		}
	}

}

// 人工给定两个阈值，一个是低阈值TL，一个高阈值TH，
// 如果边缘像素的梯度值高于高阈值，则将其标记为强边缘像素，该位置的像素值置255；
// 如果边缘像素的梯度值小于高阈值并且大于低阈值，则将其标记为弱边缘像素；
// 如果边缘像素的梯度值小于低阈值，则会被抑制，该位置的像素值置0。

//算法步骤：
//1、选取系数TH和TL，比率为2 : 1或3 : 1。（一般取TH = 0.3或0.2, TL = 0.1）；
//2、将小于低阈值的点抛弃，赋0；将大于高阈值的点立即标记（这些点为确定边缘点），赋1或255；
//3、将小于高阈值，大于低阈值的点使用8连通区域确定（即：只有与TH像素连接时才会被接受，成为边缘点，赋1或255）具体的，当强边缘的8邻域内有弱边缘像素，则将如边缘像素变成强边缘，赋值1或255，或者反过来理解，只要弱边缘的8邻域内有强边缘，则如边缘变成强边缘，赋值1或255

void ThresholdCheck(cv::Mat& src, cv::Mat& dst, int thresholdLow, int thresholdHigh)
{
	int pad = 1;
	src.convertTo(src, CV_8UC1);
	dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	cv::Mat mid(src.rows + pad * 2, src.cols + pad * 2, CV_8UC1, cv::Scalar(0));

	for (int i = pad; i < mid.rows - pad; i++)
	{
		for (int j = pad; j < mid.cols - pad; j++)
		{
			mid.at<uchar>(i, j) = src.at<uchar>(i - pad, j - pad);
		}
	}

	for (int i = pad; i < mid.rows - pad; i++)
	{
		for (int j = pad; j < mid.cols - pad; j++)
		{
			if (int(mid.at<uchar>(i, j)) >= thresholdHigh)
			{
				dst.at<uchar>(i - pad, j - pad) = 255;
			}
			else if (int(mid.at<uchar>(i, j)) < thresholdLow)
			{
				dst.at<uchar>(i - pad, j - pad) = 0;
			}
			else
			{
				for (int a = - pad; a < pad + 1; a++)
				{
					for (int b = - pad; b < pad + 1; b++)
					{
						if (int(mid.at<uchar>(i + a, j + b)) >= thresholdHigh)
						{
							dst.at<uchar>(i - pad, j - pad) = 255;
							break;
						}
					}
				}

				if (int(dst.at<uchar>(i - pad, j - pad)) != 255)
				{
					dst.at<uchar>(i - pad, j - pad) = mid.at<uchar>(i, j);
				}
			}
		}
	}
}

void CannyTest(cv::Mat& src, cv::Mat& dst, float sigma, int size, int thresholdLow, int thresholdHigh)
{
	src.convertTo(src, CV_8UC1);
	dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	cv::Mat gray_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	cv::Mat gs_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	cv::Mat gd_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	cv::Mat ga_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_32F);
	cv::Mat gnms_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	cv::Mat gtc_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);

	GrayPicture(src, gray_test);
	GaussianSmooth(gray_test, gs_test, sigma, size);
	CalGradient(gs_test, gd_test, ga_test);
	NMS(gd_test, ga_test, gnms_test);
	ThresholdCheck(gnms_test, gtc_test, thresholdLow, thresholdHigh);
	dst = gtc_test;
}


// test
//#include"CannyTest.h"
//using namespace cv;
//using namespace std;
//
//// 注意断言
//// 1. 数组越界
//// 2. 类型不匹配
//// 3. 图片坐标系和长宽定义混淆
//// 4. 注意大于，小于，大于等于，小于等于的严谨性
//// 5. 不能随便用CV_32F，想要维持原图，要用CV_8UC1
//// 6. 注意除零
//int main()
//{
//	// test all pictures
//	//Mat src = imread("girl.jpg");
//	//Mat gray_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
//	//Mat gs_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
//	//Mat gd_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
//	//Mat ga_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_32F);
//	//Mat gnms_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
//	//Mat gtc_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
//	//float sigma = 5;
//	//int size = 5;
//	//int thresholdLow = 80;
//	//int thresholdHigh = 240;
//	//
//	//GrayPicture(src, gray_test);
//	//GaussianSmooth(gray_test, gs_test, sigma, size);
//	//CalGradient(gs_test, gd_test, ga_test);
//	//NMS(gd_test, ga_test, gnms_test);
//	//ThresholdCheck(gnms_test, gtc_test, thresholdLow, thresholdHigh);
//
//
//	//imshow("src", src);
//	//imshow("gray_test", gray_test);
//	//imshow("gs_test", gs_test);
//	//imshow("gd_test", gd_test);
//	//imshow("gnms_test", gnms_test);
//	//imshow("gtc_test", gtc_test);
//
//	//waitKey(0);
//	//return 0;
//
//	// test
//	Mat src = imread("girl.jpg");
//	Mat dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
//
//	float sigma = 5;
//	int size = 5;
//	int thresholdLow = 80;
//	int thresholdHigh = 240;
//
//	CannyTest(src, dst, sigma, size, thresholdLow, thresholdHigh);
//
//	imshow("src", src);
//	imshow("dst", dst);
//
//	waitKey(0);
//	return 0;
// 
//  Mat src = imread("girl.jpg");
//  Mat dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
//
//  Canny(src, dst, 80, 240, 5, false);
//
//  imshow("src", src);
//  imshow("dst", dst);
//
//  waitKey(0);
//  return 0;
//}

