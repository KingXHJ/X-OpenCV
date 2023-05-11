#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace std;

Mat *img = 0;

/*
更改透射变换的目标图片大小
*/
Mat transImage(256, 256, CV_8UC3);

Point2f srcPoints[4];
Point2f dstPoints[4];

//控制鼠标次数
int cnt = 0;

void onMouse(int event, int x, int y, int flags, void *param)
{
    Mat *im = reinterpret_cast<Mat *>(param);
    switch (event)
    {
    case 1: //鼠标左键按下响应：返回坐标和灰度
        //存储透视变换坐标
        if (cnt == 4)
        {
            cout << "点击次数超过4次！！" << endl;
            cout << "重新操作！！" << endl;
            for (int j = 0; j < 4; j++)
            {
                srcPoints[j] = Point(0, 0);
            }
            break;
        }
        srcPoints[cnt] = Point(x, y);
        cnt++;

        std::cout << "at(" << x << "," << y << ")value is:"
                  << static_cast<int>(im->at<uchar>(cv::Point(x, y))) << std::endl;

        if (cnt == 3)
        {
            cout << "你的最后一次点击已经完成，请按任意键切换图片或退出！！" << endl;
        }

        break;
    case 2: //鼠标右键按下响应：输入坐标并返回该坐标的灰度
        std::cout << "input(x,y)" << endl;
        std::cout << "x =" << endl;
        cin >> x;
        std::cout << "y =" << endl;
        cin >> y;
        std::cout << "at(" << x << "," << y << ")value is:"
                  << static_cast<int>(im->at<uchar>(cv::Point(x, y))) << std::endl;
        break;
    }
}

int main()
{
    /*
    更改透射变换的目标图片大小
    */
    //初始化目标坐标
    dstPoints[0] = Point2f(0, 0);
    dstPoints[1] = Point2f(255, 0);
    dstPoints[2] = Point2f(255, 255);
    dstPoints[3] = Point2f(0, 255);

    Mat src;

    for (int i = 0; i < 2; i++)
    {
        //计算鼠标点击次数
        cnt = 0;

        /*
        更改原始图片路径
        */
        string img_name = "C:\\Users\\zlz\\Desktop\\opencv_cpp\\numbers_difflight\\IMG_" + to_string(8744 + i) + ".jpg";
        Mat srcimgae = imread(img_name);

        /*
        更改图片缩放大小
        */
        resize(srcimgae, src, src.size(), 0.1, 0.1, INTER_LINEAR);

        img = &src;
        Mat src2 = src.clone();
        namedWindow("original image", WINDOW_AUTOSIZE);
        cv::setMouseCallback("original image", onMouse, reinterpret_cast<void *>(img)); //注册鼠标操作(回调)函数
        imshow("original image", src);

        /*
        按任意键切换
        */
        waitKey(0);

        Mat transMat = getPerspectiveTransform(srcPoints, dstPoints);
        warpPerspective(src, transImage, transMat, transImage.size());

        /*
        更改保存图片路径
        */
        imwrite("C:\\Users\\zlz\\Desktop\\opencv_cpp\\numbers_difflight_handle\\" + to_string(i + 7) + ".jpg", transImage);
    }
    return 0;
}