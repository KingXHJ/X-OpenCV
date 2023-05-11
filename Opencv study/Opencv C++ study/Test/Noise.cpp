#include"Noise.h"

#define N 999 //��λС����
#define PI 3.1415926

void Noise_Guassian(cv::Mat& src, cv::Mat& dst)
{
	int sigma = 100;
	cv::Mat mid(src.rows, src.cols, CV_32F, cv::Scalar(0, 0, 0));
	src.convertTo(src, CV_32F);
	dst.convertTo(dst, CV_32F);
	srand(time(NULL));//������������ӣ�ʹÿ�λ�ȡ��������в�ͬ��
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols - 1; j++)
		{
			float r = rand() % (N + 1) / (float)(N + 1);//����0-1����������
			float phi = rand() % (N + 1) / (float)(N + 1);//����0-1����������
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
//	Mat img = imread("1.jpg", IMREAD_GRAYSCALE);
//
//	if (img.empty())
//	{
//		cout << "��ȷ��ͼ���ļ������Ƿ���ȷ" << endl;
//		return -1;
//	}
//	// Opencv
//	//������ԭͼ��ͬ�ߴ硢�������ͺ�ͨ�����ľ���
//	//Mat img_noise = Mat::zeros(img.rows, img.cols, img.type());
//	//imshow("lenaԭͼ", img);
//	//RNG rng;                                   //����һ��RNG��
//	//rng.fill(img_noise, RNG::NORMAL, 10, 20);  //������ͨ���ĸ�˹�ֲ��������10��20����ʾ��ֵ�ͱ�׼��
//	//imshow("��ͨ����˹����", img_noise);
//	//img = img + img_noise;                     //�ڲ�ɫͼ������Ӹ�˹����	
//	//imshow("img�������", img);                //��ʾ��Ӹ�˹�������ͼ��
//
//	// �Դ�����
//	Mat img_noise(img.rows, img.cols, CV_8U, cv::Scalar(0, 0, 0));
//	Noise_Guassian(img, img_noise);
//	imshow("lenaԭͼ", img);
//	imshow("img�������", img_noise);
//	waitKey(0);
//	return 0;
//}