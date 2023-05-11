#include "HarrisTest.h"

void H_GrayPicture(cv::Mat src, cv::Mat dst)
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

void H_GuassianKernel(float sigma, int size, cv::Mat kernel)
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

void H_GaussianSmooth(cv::Mat src, cv::Mat dst, float sigma, int size)
{
	int pad = size / 2;
	cv::Mat kernel(size, size, CV_32F, cv::Scalar(0));
	cv::Mat mid(src.rows + pad * 2, src.cols + pad * 2, CV_32F, cv::Scalar(0));
	dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	src.convertTo(src, CV_32F);

	H_GuassianKernel(sigma, size, kernel);

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
			for (int a = -pad; a < pad + 1; a++)
			{
				for (int b = -pad; b < pad + 1; b++)
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
void CalGradientXY(cv::Mat src, cv::Mat dstx, cv::Mat dsty)
{
	int size = 3;
	int pad = size / 2;
	cv::Mat x(src.rows + pad * 2, src.cols + pad * 2, CV_32F, cv::Scalar(0));
	cv::Mat y(src.rows + pad * 2, src.cols + pad * 2, CV_32F, cv::Scalar(0));
	cv::Mat kernelx = (cv::Mat_<float>(size, size) << 1, 0, -1,
		2, 0, -2,
		1, 0, -1);
	cv::Mat kernely = (cv::Mat_<float>(size, size) << 1, 2, 1,
		0, 0, 0,
		-1, -2, -1);
	dstx = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_32F);
	dsty = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_32F);
	src.convertTo(src, CV_8UC1);

	for (int i = pad; i < x.rows - pad; i++)
	{
		for (int j = pad; j < x.cols - pad; j++)
		{
			x.at<float>(i, j) = src.at<uchar>(i - pad, j - pad);
		}
	}
	for (int i = pad; i < y.rows - pad; i++)
	{
		for (int j = pad; j < y.cols - pad; j++)
		{
			y.at<float>(i, j) = src.at<uchar>(i - pad, j - pad);
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
					x.at<float>(i - pad, j - pad) = x.at<float>(i - pad, j - pad) + x.at<float>(i + a, j + b) * kernelx.at<float>(a + pad, b + pad);
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
					y.at<float>(i - pad, j - pad) = y.at<float>(i - pad, j - pad) + y.at<float>(i + a, j + b) * kernely.at<float>(a + pad, b + pad);
				}
			}
			//std::cout << dst.at<float>(i - pad, j - pad) << std::endl;
		}
	}

	for (int i = pad; i < y.rows - pad; i++)
	{
		for (int j = pad; j < y.cols - pad; j++)
		{
			dstx.at<float>(i - pad, j - pad) = x.at<float>(i, j);
			dsty.at<float>(i - pad, j - pad) = y.at<float>(i, j);
		}
	}
}


void CalResponse(cv::Mat srcx, cv::Mat srcy, cv::Mat dst, int windowSize, float k)
{
	int pad = windowSize / 2;
	int width = srcx.rows + pad * 2;
	int height = srcx.cols + pad * 2;
	cv::Mat dx_squared(srcx.rows + pad * 2, srcx.cols + pad * 2, CV_32F, cv::Scalar(0));
	cv::Mat dy_squared(srcy.rows + pad * 2, srcy.cols + pad * 2, CV_32F, cv::Scalar(0));
	cv::Mat dx_dy_squared(srcx.rows + pad * 2, srcx.cols + pad * 2, CV_32F, cv::Scalar(0));
	cv::Mat mid(2, 2, CV_32F, cv::Scalar(0));
	srcx.convertTo(srcx, CV_32F);
	srcy.convertTo(srcy, CV_32F);
	dst = cv::Mat::zeros(cv::Size(srcx.cols, srcx.rows), CV_32F); // R(A)

	for (int i = pad; i < dx_squared.rows - pad; i++)
	{
		for (int j = pad; j < dx_squared.cols - pad; j++)
		{
			dx_squared.at<float>(i, j) = srcx.at<float>(i - pad, j - pad) * srcx.at<float>(i - pad, j - pad);
			//std::cout << dx_squared.at<float>(i, j) << std::endl;
			dy_squared.at<float>(i, j) = srcy.at<float>(i - pad, j - pad) * srcy.at<float>(i - pad, j - pad);
			dx_dy_squared.at<float>(i, j) = srcx.at<float>(i - pad, j - pad) * srcy.at<float>(i - pad, j - pad);
		}
	}

	float det = 0, trace = 0;

	for (int i = pad; i < dx_squared.rows - pad; i++)
	{
		for (int j = pad; j < dx_squared.cols - pad; j++)
		{
			mid.at<float>(0, 0) = 0;
			mid.at<float>(0, 1) = 0;
			mid.at<float>(1, 0) = 0;
			mid.at<float>(1, 1) = 0;
			det = 0;
			trace = 0;

			for (int a = - pad; a < pad + 1; a++)
			{
				for (int b = - pad; b < pad + 1; b++)
				{
					mid.at<float>(0, 0) = mid.at<float>(0, 0) + dx_squared.at<float>(i + a, j + b);
					mid.at<float>(0, 1) = mid.at<float>(0, 1) + dx_dy_squared.at<float>(i + a, j + b);
					mid.at<float>(1, 1) = mid.at<float>(1, 1) + dy_squared.at<float>(i + a, j + b);
				}
			}
			mid.at<float>(1, 0) = mid.at<float>(0, 1);
			det = mid.at<float>(0, 0) * mid.at<float>(1, 1) - mid.at<float>(1, 0) * mid.at<float>(0, 1);
			trace = mid.at<float>(1, 0) + mid.at<float>(0, 1);
			dst.at<float>(i - pad, j - pad) = det - k * trace * trace;
		}
	}
}

// 重点：Canny中的非极大值抑制是沿着梯度方向对幅值进行非极大值抑制，而非边缘方向。
void NMSThreshold(cv::Mat src, int threshold, int windowSize, cv::Mat dst)
{
	int pad = windowSize / 2;
	int cnt = 0;
	src.convertTo(src, CV_32F);// R(A)
	dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	cv::Mat mid(src.rows + 2 * pad, src.cols + 2 * pad, CV_32F, cv::Scalar(0));

	for (int i = pad; i < mid.rows - pad; i++)
	{
		for (int j = pad; j < mid.cols - pad; j++)
		{
			mid.at<float>(i, j) = src.at<float>(i - pad, j - pad);
			//std::cout << src.at<float>(i - pad, j - pad) << std::endl;
		}
	}

	for (int i = pad; i < mid.rows - pad; i++)
	{
		for (int j = pad; j < mid.cols - pad; j++)
		{
			cnt = 0;
			if (mid.at<float>(i, j) < threshold)
			{
				continue;
			}
			for (int a = - pad; a < pad + 1; a++)
			{
				for (int b = - pad; b < pad + 1; b++)
				{
					if (mid.at<float>(i, j) > mid.at<float>(i + a, j + b))
					{
						cnt = cnt + 1;
					}
				}
			}
			if (cnt == windowSize * windowSize - 1)
			{
				dst.at<uchar>(i - pad, j - pad) = 255;
			}
		}
	}
}


void HarrisTest(cv::Mat& src, cv::Mat& dst, float sigma, int size, int windowSize, float k, int threshold)
{
	src.convertTo(src, CV_8UC1);
	dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	cv::Mat gray_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	cv::Mat gs_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	cv::Mat gd1_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_32F);
	cv::Mat gd2_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_32F);
	cv::Mat gr_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_32F);
	cv::Mat gnms_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);

	H_GrayPicture(src, gray_test);
	H_GaussianSmooth(gray_test, gs_test, sigma, size);
	CalGradientXY(gs_test, gd1_test, gd2_test);
	CalResponse(gd1_test, gd2_test, gr_test, windowSize, k);
	NMSThreshold(gr_test, threshold, windowSize, gnms_test);
	dst = gnms_test;
}

// Test
//#include"HarrisTest.h"
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
//	int sigma = 1.4;
//	int size = 3;
//	int windowSize = 3;
//	float k = 0.09;
//	int H_threshold = 1000000000;
//
//	//以灰度模式载入图像并显示
//	Mat src = imread("chapter3.png");
//
//	/*Mat src_opencv;
//	cvtColor(src, src_opencv, COLOR_BGR2GRAY);*/
//
//	Mat dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
//	cv::Mat gray_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
//	cv::Mat gs_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
//	cv::Mat gd1_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_32F);
//	cv::Mat gd2_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_32F);
//	cv::Mat gr_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_32F);
//	cv::Mat gnms_test = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
//
//
//	H_GrayPicture(src, gray_test);
//	H_GaussianSmooth(gray_test, gs_test, sigma, size);
//	CalGradientXY(gs_test, gd1_test, gd2_test);
//	CalResponse(gd1_test, gd2_test, gr_test, windowSize, k);
//	NMSThreshold(gr_test, H_threshold, windowSize, gnms_test);
//	dst = gnms_test;
//
//
//
//	////进行Harris角点检测找出角点
//	//Mat corner, dst_test;
//	//cornerHarris(src_opencv, corner, 2, 3, 0.09);
//	//HarrisTest(src, dst_test, sigma, size, windowSize, k, H_threshold);
//
//	////对灰度图进行阈值操作，得到二值图并显示  
//	//Mat dst;
//	//threshold(corner, dst, 0.00001, 255, THRESH_BINARY);
//
//	imshow("src", src);
//	imshow("gray_test", gray_test);
//	imshow("gs_test", gs_test);
//	imshow("gd1_test", gd1_test);
//	imshow("gd2_test", gd2_test);
//	imshow("gr_test", gr_test);
//	imshow("dst", dst);
//
//	//imshow("dst_test", dst_test);
//
//	waitKey(0);
//	return 0;
//}

