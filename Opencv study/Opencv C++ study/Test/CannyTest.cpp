#include "CannyTest.h"

void GrayPicture(cv::Mat src, cv::Mat dst)
{
	// ����һ
	/*cv::Mat b(src.rows, src.cols, CV_8UC1);
	cv::Mat g(src.rows, src.cols, CV_8UC1);
	cv::Mat r(src.rows, src.cols, CV_8UC1);
	cv::Mat out[] = { b,g,r };
	cv::split(src, out);*/
	
	// ������
	cv::Mat different_Channels[3];//declaring a matrix with three channels//  
	cv::split(src, different_Channels);//splitting images into 3 different channels//  
	cv::Mat b = different_Channels[0];//loading blue channels//
	cv::Mat g = different_Channels[1];//loading green channels//
	cv::Mat r = different_Channels[2];//loading red channels//  

	// �ҶȻ�
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
			kernel.at<float>(i, j) = 1 / (2 * PI * sigma * sigma) * exp(-((i - pad) * (i - pad) + (j - pad) * (j - pad)) / (2 * sigma * sigma)); // kernel��������������Ϊԭ��ģ����������Ͻ�
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
					// ���������CV_32F����Ҫά��ԭͼ��Ҫ��CV_8UC1
					dst.at<uchar>(i - pad, j - pad) = dst.at<uchar>(i - pad, j - pad) + int(mid.at<float>(i + a, j + b) * kernel.at<float>(a + pad, b + pad));
				}
			}
			//std::cout << dst.at<float>(i - pad, j - pad) << std::endl;
		}
	}

	//dst.convertTo(dst, CV_8UC1);
}

// Sobel�������굼������и�ֵ�����л����255��ֵ����ԭͼ����uint8����8λ�޷�����������Sobel������ͼ��λ�����������нضϡ����Ҫʹ��32λ�����ͣ���CV_32F��
// �������ø���
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
					// ���������CV_32F����Ҫά��ԭͼ��Ҫ��CV_8UC1
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
					// ���������CV_32F����Ҫά��ԭͼ��Ҫ��CV_8UC1
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
			// ע���������
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
				angle.at<float>(i - pad, j - pad) = std::atan(y.at<uchar>(i, j) / x.at<uchar>(i, j)); // ����
			}
		}
	}
}

// �ص㣺Canny�еķǼ���ֵ�����������ݶȷ���Է�ֵ���зǼ���ֵ���ƣ����Ǳ�Ե����

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

// �˹�����������ֵ��һ���ǵ���ֵTL��һ������ֵTH��
// �����Ե���ص��ݶ�ֵ���ڸ���ֵ��������Ϊǿ��Ե���أ���λ�õ�����ֵ��255��
// �����Ե���ص��ݶ�ֵС�ڸ���ֵ���Ҵ��ڵ���ֵ��������Ϊ����Ե���أ�
// �����Ե���ص��ݶ�ֵС�ڵ���ֵ����ᱻ���ƣ���λ�õ�����ֵ��0��

//�㷨���裺
//1��ѡȡϵ��TH��TL������Ϊ2 : 1��3 : 1����һ��ȡTH = 0.3��0.2, TL = 0.1����
//2����С�ڵ���ֵ�ĵ���������0�������ڸ���ֵ�ĵ�������ǣ���Щ��Ϊȷ����Ե�㣩����1��255��
//3����С�ڸ���ֵ�����ڵ���ֵ�ĵ�ʹ��8��ͨ����ȷ��������ֻ����TH��������ʱ�Żᱻ���ܣ���Ϊ��Ե�㣬��1��255������ģ���ǿ��Ե��8������������Ե���أ������Ե���ر��ǿ��Ե����ֵ1��255�����߷�������⣬ֻҪ����Ե��8��������ǿ��Ե�������Ե���ǿ��Ե����ֵ1��255

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
//// ע�����
//// 1. ����Խ��
//// 2. ���Ͳ�ƥ��
//// 3. ͼƬ����ϵ�ͳ��������
//// 4. ע����ڣ�С�ڣ����ڵ��ڣ�С�ڵ��ڵ��Ͻ���
//// 5. ���������CV_32F����Ҫά��ԭͼ��Ҫ��CV_8UC1
//// 6. ע�����
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

