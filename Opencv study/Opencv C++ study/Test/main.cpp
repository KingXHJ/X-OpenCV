#include <opencv2\opencv.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>
using namespace cv;
using namespace std;

// 注意断言
// 1. 数组越界
// 2. 类型不匹配
// 3. 图片坐标系和长宽定义混淆
// 4. 注意大于，小于，大于等于，小于等于的严谨性
// 5. 不能随便用CV_32F，想要维持原图，要用CV_8UC1
// 6. 注意除零
int main()
{
    Mat gray = imread("PCB3.jpg", IMREAD_GRAYSCALE);
    //需要计算的图像的通道，灰度图像为0，BGR图像需要指定B,G,R
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