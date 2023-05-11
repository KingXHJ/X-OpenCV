# X-OpenCV
Opencv study and develop

# 目录
- [OpenCV VScode environment](#opencv-vscode-environment)
- [Opencv tools code](#opencv-tools-code)
- [Opencv study](#opencv-study)
    1. [Opencv C++ study](#opencv-c-study)
    1. [Opencv Python Study](#opencv-python-study)

## OpenCV VScode environment
### 报错解决
1. 编译问题
出现
```
ERROR: Unable to start debugging. Unexpected GDB output from command "-exec-run". During startup program exited with code 0xc0000139.
```
进入你的minGW/bin目录下, 将libstdc++ -6.dll拷贝到需要执行文件目录下, 重新运行或调试就可以啦。
1. 找不到文件
重新对json文件更新一下路径

## Opencv tools code
My Opencv tools

numbers_get_pixel.cpp   通过鼠标点击照片，选择要进行透视变换的四个点，注意从左上角的点，顺时针进行选择。  

numbers_save_video.cpp  通过按任意键，一帧一帧读取视频，并且把每一帧存储下来。 

视频体积太大了，就没有上传到GitHub上


## Opencv study

### Opencv C++ study
跟着一本书《图像处理、分析与机器视觉》做算法学习、复现，同时用opencv内置函数去对比效果

### Opencv Python Study
跟着opencv的pyhton中文手册做了一边