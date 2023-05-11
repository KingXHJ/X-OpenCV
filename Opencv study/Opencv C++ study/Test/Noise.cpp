#include"Noise.h"

#define N 999 //三位小数。
#define PI 3.1415926

void Noise_Guassian(cv::Mat& src, cv::Mat& dst)
{
	int sigma = 100;
	cv::Mat mid(src.rows, src.cols, CV_32F, cv::Scalar(0, 0, 0));
	src.convertTo(src, CV_32F);
	dst.convertTo(dst, CV_32F);
	srand(time(NULL));//设置随机数种子，使每次获取的随机序列不同。
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols - 1; j++)
		{
			float r = rand() % (N + 1) / (float)(N + 1);//生成0-1间的随机数。
			float phi = rand() % (N + 1) / (float)(N + 1);//生成0-1间的随机数。
			//std::cout << "r:" << r << "      " << "phi:" << phi << std::endl;
			float z1 = sigma * std::cos(2 * PI * phi) * std::sqrt(-2 * std::log(r));
			float z2 = sigma * std::sin(2 * PI * phi) * std::sqrt(-2 * std::log(r));
			mid.at<float>(i, j) = src.at<float>(i, j) + z1;
			mid.at<float>(i, j + 1) = src.at<float>(i, j + 1) + z2;
			if (mid.at<float>(i, j) < 0)
			{
				mid.at<float>(i, j) = 0;
			}
			else if (mid.at<float>(i, j) > 255)
			{
				mid.at<float>(i, j) = 255;
			}
			if (mid.at<float>(i, j + 1) < 0)
			{
				mid.at<float>(i, j + 1) = 0;
			}
			else if (mid.at<float>(i, j + 1) > 255)
			{
				mid.at<float>(i, j + 1) = 255;
			}
			dst.at<float>(i, j) = mid.at<float>(i, j);
			dst.at<float>(i, j + 1) = mid.at<float>(i, j + 1);
		}
	}
	src.convertTo(src, CV_8U);
	dst.convertTo(dst, CV_8U);
}

// Test
//#include"Noise.h"
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
//	Mat img = imread("1.jpg", IMREAD_GRAYSCALE);
//
//	if (img.empty())
//	{
//		cout << "请确认图像文件名称是否正确" << endl;
//		return -1;
//	}
//	// Opencv
//	//生成与原图像同尺寸、数据类型和通道数的矩阵
//	//Mat img_noise = Mat::zeros(img.rows, img.cols, img.type());
//	//imshow("lena原图", img);
//	//RNG rng;                                   //创建一个RNG类
//	//rng.fill(img_noise, RNG::NORMAL, 10, 20);  //生成三通道的高斯分布随机数（10，20）表示均值和标准差
//	//imshow("三通道高斯噪声", img_noise);
//	//img = img + img_noise;                     //在彩色图像中添加高斯噪声	
//	//imshow("img添加噪声", img);                //显示添加高斯噪声后的图像
//
//	// 自创函数
//	Mat img_noise(img.rows, img.cols, CV_8U, cv::Scalar(0, 0, 0));
//	Noise_Guassian(img, img_noise);
//	imshow("lena原图", img);
//	imshow("img添加噪声", img_noise);
//	waitKey(0);
//	return 0;
//}