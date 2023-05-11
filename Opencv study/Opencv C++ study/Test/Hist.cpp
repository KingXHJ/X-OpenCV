#include"Hist.h"

void HistCal(cv::Mat& src, cv::Mat& cal)
{
	src.convertTo(src, CV_8U);
	cal = cv::Mat::zeros(cv::Size(256, 1), CV_32F);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			cal.at<float>(int(src.at<uchar>(i, j))) = cal.at<float>(int(src.at<uchar>(i, j))) + 1;
		}
	}
}

void EqualCal(cv::Mat& src, cv::Mat& dst)
{
	src.convertTo(src, CV_8U);
	dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8U); // ע�⣡����cv::Size���ȿ���

	cv::Mat equal_hist = cv::Mat::zeros(cv::Size(256, 1), CV_8U);
	cv::Mat cal = cv::Mat::zeros(cv::Size(256, 1), CV_32F);
	cv::Mat equal_cal = cv::Mat::zeros(cv::Size(256, 1), CV_32F);
	int cal_min = 300, equal_cal_min = 0;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (int(src.at<uchar>(i, j)) < cal_min)
			{
				cal_min = int(src.at<uchar>(i, j));
			}
			cal.at<float>(int(src.at<uchar>(i, j))) = cal.at<float>(int(src.at<uchar>(i, j))) + 1;
		}
	}

	equal_cal.at<float>(0) = cal.at<float>(0);
	for (int i = 1; i < 256; i++)
	{
		equal_cal.at<float>(i) = equal_cal.at<float>(i - 1) + cal.at<float>(i);
	}
	equal_cal_min = equal_cal.at<float>(cal_min);

	for (int i = 0; i < 256; i++)
	{
		equal_hist.at<uchar>(i) = (int)(((equal_cal.at<float>(i) - equal_cal_min) / (src.rows * src.cols - equal_cal_min)) * 255 + 0.5);
	}

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			/*std::cout << int(src.at<uchar>(i, j)) << std::endl;
			std::cout << int(equal_hist.at<uchar>(int(src.at<uchar>(i, j)))) << std::endl;*/
			dst.at<uchar>(i,j) = equal_hist.at<uchar>(int(src.at<uchar>(i, j)));
		}
	}
}




// Opencv����������test
//#include"Hist.h"
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
//	Mat gray = imread("1.jpg", IMREAD_GRAYSCALE);
//	//��Ҫ�����ͼ���ͨ�����Ҷ�ͼ��Ϊ0��BGRͼ����Ҫָ��B,G,R
//	const int channels[] = { 0 };
//	Mat hist = Mat::zeros(Size(256, 1), CV_32F);
//	int dims = 1;//����ֱ��ͼά��
//	const int histSize[] = { 256 }; //ֱ��ͼÿһ��ά�Ȼ��ֵ���������Ŀ
//	//ÿһ��ά��ȡֵ��Χ
//	float pranges[] = { 0, 255 };//ȡֵ����
//	const float* ranges[] = { pranges };
//
//	calcHist(&gray, 1, channels, Mat(), hist, dims, histSize, ranges, true, false);//����ֱ��ͼ
//	//HistCal(gray, hist);
//	int scale = 2;
//	int hist_height = 256;
//	Mat hist_img = Mat::zeros(hist_height, 256 * scale, CV_8UC3); //����һ���ڵ׵�8λ��3ͨ��ͼ�񣬸�256����256*2
//	double max_val;
//	minMaxLoc(hist, 0, &max_val, 0, 0);//����ֱ��ͼ���������ֵ
//	//�����صĸ������ϵ� ͼ������Χ��
//	//����ֱ��ͼ�õ�������
//	for (int i = 0; i < 256; i++)
//	{
//		float bin_val = hist.at<float>(i);   //����histԪ�أ�ע��hist����float���ͣ�
//		int intensity = cvRound(bin_val * hist_height / max_val);  //���Ƹ߶�
//		rectangle(hist_img, Point(i * scale, hist_height - 1), Point((i + 1) * scale - 1, hist_height - intensity), Scalar(255, 255, 255));//����ֱ��ͼ
//	}
//
//	imshow("hist_img", hist_img);
//	/*for (int i = 0; i < gray.rows; i++)
//	{
//		for (int j = 0; j < gray.cols; j++)
//		{
//			cout << int(gray.at<uchar>(i, j)) << endl;
//		}
//	}*/
//	waitKey(0);
//	return 0;
//}

// testֱ��ͼ���⻯
//#include"Hist.h"
//using namespace cv;
//using namespace std;
//
//#include"Hist.h"
//using namespace cv;
//using namespace std;
//
//int main()
//{
//	Mat gray = imread("1.jpg", IMREAD_GRAYSCALE);
//	//��Ҫ�����ͼ���ͨ�����Ҷ�ͼ��Ϊ0��BGRͼ����Ҫָ��B,G,R
//	const int channels[] = { 0 };
//	Mat hist = Mat::zeros(Size(256, 1), CV_32F);
//	Mat equal_hist = Mat::zeros(Size(256, 1), CV_32F);
//	Mat equal_img, equal_img_test;
//	int dims = 1;//����ֱ��ͼά��
//	const int histSize[] = { 256 }; //ֱ��ͼÿһ��ά�Ȼ��ֵ���������Ŀ
//	//ÿһ��ά��ȡֵ��Χ
//	float pranges[] = { 0, 255 };//ȡֵ����
//	const float* ranges[] = { pranges };
//
//	//calcHist(&gray, 1, channels, Mat(), hist, dims, histSize, ranges, true, false);//����ֱ��ͼ
//	////HistCal(gray, hist);
//
//	//int scale = 2;
//	//int hist_height = 256;
//	//Mat hist_img = Mat::zeros(hist_height, 256 * scale, CV_8UC3); //����һ���ڵ׵�8λ��3ͨ��ͼ�񣬸�256����256*2
//	//double max_val;
//	//minMaxLoc(hist, 0, &max_val, 0, 0);//����ֱ��ͼ���������ֵ
//	////�����صĸ������ϵ� ͼ������Χ��
//	////����ֱ��ͼ�õ�������
//	//for (int i = 0; i < 256; i++)
//	//{
//	//	float bin_val = hist.at<float>(i);   //����histԪ�أ�ע��hist����float���ͣ�
//	//	int intensity = cvRound(bin_val * hist_height / max_val);  //���Ƹ߶�
//	//	rectangle(hist_img, Point(i * scale, hist_height - 1), Point((i + 1) * scale - 1, hist_height - intensity), Scalar(255, 255, 255));//����ֱ��ͼ
//	//}
//
//	//imshow("hist_img", hist_img);
//
//	imshow("gray", gray);
//	equalizeHist(gray, equal_img);
//	imshow("equal_img", equal_img);
//	//calcHist(&equal_img, 1, channels, Mat(), equal_hist, dims, histSize, ranges, true, false);//����ֱ��ͼ
//
//	EqualCal(gray, equal_img_test);
//	imshow("equal_img_test", equal_img_test);
//	//int equal_scale = 2;
//	//int equal_hist_height = 256;
//	//Mat equal_hist_img = Mat::zeros(equal_hist_height, 256 * equal_scale, CV_8UC3); //����һ���ڵ׵�8λ��3ͨ��ͼ�񣬸�256����256*2
//	//double equal_max_val;
//	//minMaxLoc(equal_hist, 0, &equal_max_val, 0, 0);//����ֱ��ͼ���������ֵ
//	////�����صĸ������ϵ� ͼ������Χ��
//	////����ֱ��ͼ�õ�������
//	//for (int i = 0; i < 256; i++)
//	//{
//	//	float equal_bin_val = equal_hist.at<float>(i);   //����histԪ�أ�ע��hist����float���ͣ�
//	//	int equal_intensity = cvRound(equal_bin_val * equal_hist_height / equal_max_val);  //���Ƹ߶�
//	//	rectangle(equal_hist_img, Point(i * equal_scale, equal_hist_height - 1), Point((i + 1) * equal_scale - 1, equal_hist_height - equal_intensity), Scalar(255, 255, 255));//����ֱ��ͼ
//	//}
//	//imshow("equal_hist_img", equal_hist_img);
//
//	/*for (int i = 0; i < gray.rows; i++)
//	{
//		for (int j = 0; j < gray.cols; j++)
//		{
//			cout << int(gray.at<uchar>(i, j)) << endl;
//		}
//	}*/
//	waitKey(0);
//	return 0;
//}