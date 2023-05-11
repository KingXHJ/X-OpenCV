#include <opencv2\opencv.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>
using namespace cv;
using namespace std;

// ע�����
// 1. ����Խ��
// 2. ���Ͳ�ƥ��
// 3. ͼƬ����ϵ�ͳ��������
// 4. ע����ڣ�С�ڣ����ڵ��ڣ�С�ڵ��ڵ��Ͻ���
// 5. ���������CV_32F����Ҫά��ԭͼ��Ҫ��CV_8UC1
// 6. ע�����
int main()
{
    Mat gray = imread("PCB3.jpg", IMREAD_GRAYSCALE);
    //��Ҫ�����ͼ���ͨ�����Ҷ�ͼ��Ϊ0��BGRͼ����Ҫָ��B,G,R
    imshow("gray", gray);
    Mat hist_img;
    equalizeHist(gray, hist_img);
    
    imshow("hist_img", hist_img);

    Mat gaussian_img;
    GaussianBlur(hist_img, gaussian_img, Size(5, 5), 3, 3);
    /*for (int i = 0; i < gray.rows; i++)
    {
    	for (int j = 0; j < gray.cols; j++)
    	{
    		cout << int(gray.at<uchar>(i, j)) << endl;
    	}
    }*/

    imshow("gaussian_img", gaussian_img);
    imwrite("gray.jpg", gray);
    imwrite("hist_img.jpg", hist_img);
    imwrite("gaussian_img.jpg", gaussian_img);
    waitKey(0);
    return 0;
}