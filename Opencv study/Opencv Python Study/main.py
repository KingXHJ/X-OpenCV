# import cv2 as cv
# # print opencv version
# print(cv.__version__)
#
# # read picture
# img1 = cv.imread('test.jpg', cv.IMREAD_COLOR)
# img0 = cv.imread('test.jpg', cv.IMREAD_GRAYSCALE)
# imgneg1 = cv.imread('test.jpg', cv.IMREAD_UNCHANGED)
#
# # # show pictures
# # cv.imshow('RGB',img1)
# # cv.imshow('gray',img0)
# # cv.imshow('alpha',imgneg1)
#
# cv.namedWindow('RGB', cv.WINDOW_AUTOSIZE) # 全屏窗口后，会有灰色的边框
# # cv.namedWindow('RGB',cv.WINDOW_NORMAL) # 全屏窗口后，图像会自动跟着拉伸
#
# cv.imshow('RGB', img1)
# # Indefinitely wait for a press of ant key
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # save picture
# cv.imwrite('test.png', img1)
#
#
#
#


# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.png', 1)
# cv.namedWindow('image_test',cv.WINDOW_NORMAL)
# cv.imshow('image_test', img)
# key = cv.waitKey(0) & 0XFF
# # 查阅资料我才知道，原来系统中按键对应的ASCII码值并不一定仅仅只有8位，同一按键对应的ASCII并不一定相同（但是后8位一定相同）
# # 为什么会有这个差别？是系统为了区别不同情况下的同一按键。
# # 比如说“q”这个按键
# # 当小键盘数字键“NumLock”激活时，“q”对应的ASCII值为100000000000001100011 。
# # 而其他情况下，对应的ASCII值为01100011。
# # 相信你也注意到了，它们的后8位相同，其他按键也是如此。
# # 为了避免这种情况，引用&0xff，正是为了只取按键对应的ASCII值后8位来排除不同按键的干扰进行判断按键是什么。
# if key == 23:
#     cv.destroyAllWindows()
# elif key == ord('s'):
#     cv.imwrite('test1.png', img)
#     cv.destroyAllWindows()


# import numpy as np
# import cv2 as cv
# cap = cv.VideoCapture(0) # 换成本地视频的路径
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# while True:
#     ret,frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     cv.imshow('frame', gray)
#     if cv.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
#
# img = np.zeros((512, 512, 3), np.uint8)
# cv.line(img, (0, 0), (511, 511), (255, 0, 0), 1, cv.LINE_AA, 2)
#
# cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
#
# cv.circle(img, (447, 63), 63, (0, 0, 255), -1)
#
# cv.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
#
# pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
# pts = pts.reshape((-1, 1, 2))
# cv.polylines(img, [pts], True, (0, 255, 255))
#
# font = cv.FONT_HERSHEY_COMPLEX
# cv.putText(img, 'KingXHJ', (10, 400), font, 2, (255, 255, 255), 2, cv.LINE_AA)
#
# cv.imshow('test', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 列出所有可用的可用事件
# import cv2 as cv
# events = [i for i in dir(cv) if 'EVENT' in i]
# print(events)

# # mouse operation
# import numpy as np
# import cv2 as cv
#
# # 创建鼠标回调函数具有特定的格式，该格式在所有地方都相同
# # callback function
# def draw_circle(event, x, y, flags, param):
#     if event == cv.EVENT_LBUTTONDBLCLK:
#         cv.circle(img, (x, y), 100, (255, 0, 0), -1)
#
#
# img = np.zeros((512, 512, 3), np.uint8)
# cv.namedWindow('image')
# cv.setMouseCallback('image', draw_circle)
# while (1):
#     cv.imshow('image', img)
#     if cv.waitKey(20) & 0xFF == 27:
#         break
# cv.destroyAllWindows()

# mouse operation advance
# import numpy as np
# import cv2 as cv
#
# drawing = False # 如果按下鼠标，则为真
# mode = True # 如果为真，绘制矩形。按 m 键可以切换到曲线
# ix, iy = -1, -1
#
#
# def draw_circle(event, x, y, flags, param):
#     global ix, iy, drawing, mode
#     if event == cv.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y
#     elif event == cv.EVENT_MOUSEMOVE:
#         if drawing == True:
#             if mode == True:
#                 cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#             else:
#                 cv.circle(img,(x,y),5,(0,0,255),-1)
#     elif event == cv.EVENT_LBUTTONUP:
#         drawing = False
#         if mode == True:
#             cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#         else:
#             cv.circle(img,(x,y),5,(0,0,255),-1)
#
# # 创建一个黑色的图像，一个窗口，并绑定到窗口的功能
# img = np.zeros((512,512,3), np.uint8)
# cv.namedWindow('image')
# cv.setMouseCallback('image',draw_circle)
# while(1):
#      cv.imshow('image',img)
#      if cv.waitKey(20) & 0xFF == 27:
#         break
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
#
#
# def nothing(x):
#     pass
#
#
# img = np.zeros((300, 512, 3), np.uint8)
# cv.namedWindow('image')
#
# cv.createTrackbar('R', 'image', 0, 255, nothing)
# cv.createTrackbar('G', 'image', 0, 255, nothing)
# cv.createTrackbar('B', 'image', 0, 255, nothing)
#
# switch = '0 : OFF \n 1 : ON'
# cv.createTrackbar(switch, 'image', 0, 1, nothing)
# while(1):
#     cv.imshow('image',img)
#     k = cv.waitKey(1) & 0xFF
#     if k ==27:
#         break
#     r=cv.getTrackbarPos('R','image')
#     g = cv.getTrackbarPos('G', 'image')
#     b = cv.getTrackbarPos('B', 'image')
#     s = cv.getTrackbarPos(switch, 'image')
#     if s == 0:
#         img[:]=0
#     else:
#         img[:]=[b,g,r]
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg')
#
# px = img[100, 100]
# print(px)
#
# # blue = img[100, 100, 0]
# # blue = img[100, 100, 1]
# blue = img[100, 100, 2]
# print(blue)
#
# img[100, 100] = [255, 255, 255]
# print(img[100, 100])
#
# img.item(10, 10, 2)
# print(img[10, 10, 2])
#
# img.itemset((10, 10, 2), 100)
# print(img[10, 10, 2])
#
# print(img.shape)
# print(img.size)
# print(img.dtype)
#
# something = img[200:400, 200:400]
# img[500:700, 1000:1200] = something
# cv.imshow('image', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# b, g, r = cv.split(img)
# cv.imshow('b', b)
# cv.imshow('g', g)
# cv.imshow('r', r)
# cv.waitKey(0)
# cv.destroyAllWindows()
# img = cv.merge((b, g, r))
#
# b = img[:, :, 0] = 0

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# BLUE = [255, 0, 0]
# img1 = cv.imread('test.png')
# replicate = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_REPLICATE)
# reflect = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_REFLECT)
# reflect101 = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_REFLECT_101)
# wrap = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_WRAP)
# constant = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=BLUE)
# plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
# plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
# plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
# plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
# plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
# plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
# plt.show()

# import numpy as np
# import cv2 as cv
#
# x = np.uint8([250])
# y = np.uint8([10])
# print(cv.add(x, y))  # OpenCV加法是饱和运算
# print(x + y)  # Numpy加法是模运算
#
# img1 = cv.imread('test.jpg')
# print(img1.shape)
# img2 = cv.imread('test1.jpg')
# dst = cv.addWeighted(img1, 0.2, img2, 0.8, 0)
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()


# import numpy as np
# import cv2 as cv
#
# e1 = cv.getTickCount()
#
# img1 = cv.imread('test.jpg')
# img2 = cv.imread('test2.jpg')
# rows, cols, channels = img2.shape
# roi = img1[0:rows, 0:cols]
# img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)  # 彩色的地方是非0
# mask_inv = cv.bitwise_not(mask)  # 彩色的地方是0
# img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
# img2_fg = cv.bitwise_and(img2, img2, mask=mask)
# dst = cv.add(img1_bg, img2_fg)
# img1[0:rows, 0:cols] = dst
# cv.imshow('res', img1)
#
# e2 = cv.getTickCount()
#
# t = (e2 - e1) / cv.getTickFrequency()
# print(t)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv
# flags = [i for i in dir(cv) if i.startswith('COLOR_')]
# print(flags)

# import cv2 as cv
# import numpy as np
#
# cap = cv.VideoCapture(0)
# while (1):
#     _, frame = cap.read()
#     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#     lower_blue = np.array([110, 50, 50])
#     upper_blue = np.array([130, 255, 255])
#
#     mask = cv.inRange(hsv, lower_blue, upper_blue)
#
#     res = cv.bitwise_and(frame,frame,mask=mask)
#
#     cv.imshow('frame',frame)
#     cv.imshow('mask',mask)
#     cv.imshow('res',res)
#     k = cv.waitKey(5) & 0xFF
#     if k == 27:
#         break
# cv.destroyAllWindows()

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg')
# res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
# height, width = img.shape[:2]
# res = cv.resize(img, (2 * width, 2 * height), interpolation=cv.INTER_CUBIC)
# cv.imshow('source', img)
# cv.imshow('scale', res)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg', 0)
# rows, cols = img.shape
# print(img.shape)
#
# M1 = np.float32([[1, 0, 100], [0, 1, 50]])
# dst1 = cv.warpAffine(img, M1, (cols, rows))
# # cols-1 和 rows-1 是坐标限制
# M2 = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
# dst2 = cv.warpAffine(img, M2, (cols, rows))
#
# img3 = cv.imread('test.jpg')
# rows, cols, channels = img3.shape
# pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
# pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
# M3 = cv.getAffineTransform(pts1, pts2)
# dst3 = cv.warpAffine(img3, M3, (cols, rows))
# cv.imshow('img1', dst1)
# cv.imshow('img2', dst2)
# cv.imshow('img3', img3)
# cv.imshow('dst3', dst3)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.png')
# rows, cols, ch = img.shape
# pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
# pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
# M = cv.getPerspectiveTransform(pts1, pts2)
# dst = cv.warpPerspective(img, M, (300, 300))
# plt.subplot(121), plt.imshow(img), plt.title('Input')
# plt.subplot(122), plt.imshow(dst), plt.title('Output')
# plt.show()

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv.imread('test.jpg',0)
# ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
# ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
# ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
# ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks()
# plt.show()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg', 0)
# ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# blur = cv.GaussianBlur(img, (5, 5), 0)
# ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv2.THRESH_OTSU)
# images = [img, 0, th1,
#           img, 0, th2,
#           blur, 0, th3]
# titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
#           'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
#           'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
# for i in range(3):
#     plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
#     plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
#     plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
#     plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
# plt.show()
#
# # Otsu
# img = cv.imread('test.jpg', 0)
# blur = cv.GaussianBlur(img, (5, 5), 0)
# hist = cv.calcHist([blur], [0], None, [256], [0, 256])
# hist_norm = hist.ravel() / hist.max()
# Q = hist_norm.cumsum()
# bins = np.arange(256)
# fn_min = np.inf
# thresh = -1
# for i in range(1, 256):
#     p1, p2 = np.hsplit(hist_norm, [i])  # 概率
#     q1, q2 = Q[i], Q[255] - Q[i]  # 对类求和
#     b1, b2 = np.hsplit(bins, [i])  # 权重
#     # 寻找均值和方差
#     m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
#     v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
#     # 计算最小化函数
#     fn = v1 * q1 + v2 * q2
#     if fn < fn_min:
#         fn_min = fn
#     thresh = i
# # 使用OpenCV函数找到otsu的阈值
# ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# print("{} {}".format(thresh, ret))

# import numpy as np
# import  cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('test.jpg')
# kernel = np.ones((5,5),np.float32)/25
# dst = cv.filter2D(img,-1,kernel)
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg')
# # blur = cv.blur(img, (5, 5))
# # blur = cv.GaussianBlur(img, (5, 5), 0)
# # blur = cv.medianBlur(img, 5) # 消除椒盐噪声
# blur = cv.bilateralFilter(img, 9, 75, 75)
# plt.subplot(121), plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(blur), plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()

# import cv2 as cv
# import numpy as np
#
# img = cv.imread('test.jpg', 0)
# kernel = np.ones((5, 5), np.uint8)
# erosion = cv.erode(img, kernel, iterations=1)  # 侵蚀白色
# dilation = cv.dilate(img, kernel, iterations=1)  # 膨胀黑色
# opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)  # 开运算，先侵蚀再膨胀，用于去白色噪
# closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)  # 闭运算，先膨胀再侵蚀，用于去黑色噪声
# gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)  # 膨胀图与腐蚀图之差
# tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)  # 原始图像与开运算之差
# blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)  # 原图像与闭运算之差
# cv.imshow('img', img)
# cv.imshow('erosion', erosion)
# cv.imshow('dilation', dilation)
# cv.imshow('opening', opening)
# cv.imshow('closing', closing)
# cv.imshow('gradient', gradient)
# cv.imshow('tophat', tophat)
# cv.imshow('blackhat', blackhat)
#
# print(cv.getStructuringElement(cv.MORPH_RECT, (5, 5)))
# print(cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
# print(cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)))
#
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg', 0)
# laplacian = cv.Laplacian(img, cv.CV_64F)
# sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
# sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
# plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# plt.show()
#
# # 在我们的最后一个示例中，输出数据类型为 cv.CV_8U 或 np.uint8 。但这有一个小问题。黑色到
# # 白色的过渡被视为正斜率（具有正值），而白色到黑色的过渡被视为负斜率（具有负值）。因
# # 此，当您将数据转换为np.uint8时，所有负斜率均设为零。简而言之，您会错过这一边缘信息。
# # 如果要检测两个边缘，更好的选择是将输出数据类型保留为更高的形式，例如 cv.CV_16S ，
# # cv.CV_64F 等，取其绝对值，然后转换回 cv.CV_8U 。 下面的代码演示了用于水平Sobel滤波器和
# # 结果差异的此过程。
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.png', 0)
# # Output dtype = cv.CV_8U
# sobelx8u = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=5)
# # Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
# sobelx64f = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
# abs_sobel64f = np.absolute(sobelx64f)
# sobel_8u = np.uint8(abs_sobel64f)
# plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
# plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
# plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
# plt.show()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('test.jpg',0)
# edges = cv.Canny(img,100,200)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

# import cv2 as cv
# import numpy as np
# # 先降低分辨率，再恢复分辨率，一定会丢失信息
# img = cv.imread('test.jpg')
# lower_reso = cv.pyrDown(img)
# higher_reso = cv.pyrUp(img)
# cv.imshow('source', img)
# cv.imshow('pyrDown', lower_reso)
# cv.imshow('pyrUp', higher_reso)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv
# import numpy as np, sys
#
# A = cv.imread('test.jpg')
# B = cv.imread('test1.jpg')
# # 生成A的高斯金字塔
# G = A.copy()
# gpA = [G]
# for i in range(6):
#     G = cv.pyrDown(G)
#     gpA.append(G)
# # 生成B的高斯金字塔
# G = B.copy()
# gpB = [G]
# for i in range(6):
#     G = cv.pyrDown(G)
#     gpB.append(G)
# # 生成A的拉普拉斯金字塔
# lpA = [gpA[5]]
# for i in range(5, 0, -1):
#     GE = cv.pyrUp(gpA[i])
#     L = cv.subtract(gpA[i - 1], GE)
#     lpA.append(L)
# # 生成B的拉普拉斯金字塔
# lpB = [gpB[5]]
# for i in range(5, 0, -1):
#     GE = cv.pyrUp(gpB[i])
#     L = cv.subtract(gpB[i - 1], GE)
#     lpB.append(L)
# # 现在在每个级别中添加左右两半图像
# LS = []
# for la, lb in zip(lpA, lpB):
#     rows, cols, dpt = la.shape
#     ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
#     LS.append(ls)
# # 现在重建
# ls_ = LS[0]
# for i in range(1, 6):
#     ls_ = cv.pyrUp(ls_)
#     ls_ = cv.add(ls_, LS[i])
# # 图像与直接连接的每一半
# real = np.hstack((A[:, :cols / 2], B[:, cols / 2:]))
# cv.imshow('Pyramid_blending2.jpg', ls_)
# cv.imshow('Direct_blending.jpg', real)

# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg')
# imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(imggray, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img, contours, -1, (0,255,0), 3)
# # cv.drawContours(img, contours, 3, (0,255,0), 3)
# # cnt = contours[4]
# # cv.drawContours(img, [cnt], 0, (0,255,0), 3)
# cv.imshow('img',img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg', 0)
# ret, thresh = cv.threshold(img, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, 1, 2)
# cnt = contours[0]
# M = cv.moments(cnt)
# cx = int(M['m10'] / M['m00'])
# cy = int(M['m01'] / M['m00'])
# area = cv.contourArea(cnt)
# perimeter = cv.arcLength(cnt, True)
#
# epsilon = perimeter * 0.1
# approx = cv.approxPolyDP(cnt, epsilon, True)
# img1 = cv.imread('test.jpg', 1)
# dst = cv.drawContours(img1, approx, -1, (0, 255, 0), 3)
#
# hull = cv.convexHull(cnt)
# dst2 = cv.drawContours(img1, hull, -1, (255, 0, 0), 3)
#
# k = cv.isContourConvex(cnt)
#
# x, y, w, h = cv.boundingRect(cnt)
# bound_rec = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# rect = cv.minAreaRect(cnt)
# box = cv.boxPoints(rect)
# box = np.int0(box)
# dst3 = cv.drawContours(img, [box], 0, (0, 0, 255), 2)
#
# (x, y), radius = cv.minEnclosingCircle(cnt)
# center = (int(x), int(y))
# radius = int(radius)
# dst4 = cv.circle(img, center, radius, (0, 255, 0), 2)
#
# ellipse = cv.fitEllipse(cnt)
# dst5 = cv.ellipse(img, ellipse, (0, 255, 0), 2)
#
# rows, cols = img.shape[:2]
# [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
# lefty = int((-x * vy / vx) + y)
# righty = int(((cols - x) * vy / vx) + y)
# dst6 = cv.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
#
# x, y, w, h = cv.boundingRect(cnt)
# aspect_ratio = float(w) / h
#
# area = cv.contourArea(cnt)
# x, y, w, h = cv.boundingRect(cnt)
# rect_area = w * h
# extent = float(area) / rect_area
#
# area = cv.contourArea(cnt)
# hull = cv.convexHull(cnt)
# hull_area = cv.contourArea(hull)
# solidity = float(area) / hull_area
#
# area = cv.contourArea(cnt)
# equi_diameter = np.sqrt(4 * area / np.pi)
#
# (x, y), (MA, ma), angle = cv.fitEllipse(cnt)
#
# mask = np.zeros(img.shape, np.uint8)
# cv.drawContours(mask, [cnt], 0, 255, -1)
# pixelpoints = np.transpose(np.nonzero(mask))
#
# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(img, mask=mask)
#
# mean_val = cv.mean(img, mask=mask)
#
# leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
# rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
# topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
# bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
#
# hull = cv.convexHull(cnt, returnPoints=False)
# defects = cv.convexityDefects(cnt, hull)
#
# print('M:', M)
# print('cx:', cx)
# print('cy:', cy)
# print('area:', area)
# print('perimeter:', perimeter)
# print('k:', k)
# print('aspect_ratio', aspect_ratio)
# print('extent', extent)
# print('solidity', solidity)
# print('equi_diameter', equi_diameter)
# print('pixelpoints', pixelpoints)
#
# cv.imshow('approx', dst)
# cv.imshow('hull', dst2)
# cv.imshow('bound_rec', bound_rec)
# cv.imshow('box', dst3)
# cv.imshow('circle', dst4)
# cv.imshow('ellipse', dst5)
# cv.imshow('line', dst6)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv
# import numpy as np
#
# img = cv.imread('test.jpg')
# img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(img_gray, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, 2, 1)
# cnt = contours[0]
# hull = cv.convexHull(cnt, returnPoints=False)
# defects = cv.convexityDefects(cnt, hull)
# for i in range(defects.shape[0]):
#     s, e, f, d = defects[i, 0]
#     start = tuple(cnt[s][0])
#     end = tuple(cnt[e][0])
#     far = tuple(cnt[f][0])
#     cv.line(img, start, end, [0, 255, 0], 2)
#     cv.circle(img, far, 5, [0, 0, 255], -1)
#
# dist = cv.pointPolygonTest(cnt, (50, 50), True)
#
# print('dist', dist)
#
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv
# import numpy as np
#
# img1 = cv.imread('test.jpg', 0)
# img2 = cv.imread('test1.jpg', 0)
# ret1, thresh1 = cv.threshold(img1, 127, 255, 0)
# ret2, thresh2 = cv.threshold(img2, 127, 255, 0)
# contours1, hierarchy1 = cv.findContours(thresh1, 2, 1)
# cnt1 = contours1[0]
# # contours2, hierarchy2 = cv.findContours(thresh2, cv.RETR_LIST, 1)
# # contours2, hierarchy2 = cv.findContours(thresh2, cv.RETR_EXTERNAL, 1)
# # contours2, hierarchy2 = cv.findContours(thresh2, cv.RETR_CCOMP, 1)
# contours2, hierarchy2 = cv.findContours(thresh2, cv.RETR_TREE, 1)
# cnt2 = contours2[0]
# ret = cv.matchShapes(cnt1, cnt2, 1, 0.0)
# print(ret)  # 结果越低，匹配越好
# print(hierarchy2)

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# # img = cv.imread('test.jpg', 0)
# # hist = cv.calcHist([img], [0], None, [256], [0, 256])
# # hist, bins = np.histogram(img.ravel(), 256, [0, 256])
# # hist = np.bincount(img.ravel(), minlength=256)
#
# # img.ravel()–把多维数组转化成一维数组，指向原数组的地址
# # img。flatten()-和ravel的作用几乎相同，但是是指向原数组拷贝的地址
# # plt.hist(img.ravel(),256,[0,256]); plt.show()
#
# # img = cv.imread('test.jpg')
# # color = ('b', 'g', 'r')
# # for i, col in enumerate(color):
# #     histr = cv.calcHist([img], [i], None, [256], [0, 256])
# #     plt.plot(histr,color = col)
# #     plt.xlim([0,256])
# # plt.show()
#
# img = cv.imread('test.jpg', 0)
# mask = np.zeros(img.shape[:2], np.uint8)
# # print(img.shape)
# mask[100:300, 100:400] = 255
# masked_img = cv.bitwise_and(img, img, mask=mask)
# hist_full = cv.calcHist([img], [0], None, [256], [0, 256])
# hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])
# plt.subplot(221), plt.imshow(img, 'gray')
# plt.subplot(222), plt.imshow(mask, 'gray')
# plt.subplot(223), plt.imshow(masked_img, 'gray')
# plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
# plt.xlim([0, 256])
# plt.show()

# # 一幅好的图像会有来自图像所有区域的像素。因此，您需要将这个直方图拉伸到两端
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg')
# hist, bins = np.histogram(img.flatten(), 256, [0, 256])
# # CDF 累积分布函数(cumulative distribution function)
# cdf = hist.cumsum()
# cdf_normalized = cdf * float(hist.max()) / cdf.max()
#
# cdf1_m = np.ma.masked_equal(cdf, 0)
# cdf1_m = (cdf1_m - cdf1_m.min()) * 255 / (cdf1_m.max() - cdf1_m.min())
# cdf1 = np.ma.filled(cdf1_m, 0).astype('uint8')
# img2 = cdf1[img]
# hist1, bins1 = np.histogram(img2.flatten(), 256, [0, 256])
# cdf2 = hist1.cumsum()
# cdf2_normalized = cdf2 * float(hist1.max()) / cdf2.max()
#
# plt.subplot(211)
# plt.plot(cdf_normalized, color='b')
# plt.hist(img.flatten(), 256, [0, 256], color='r')
# plt.xlim([0, 256])
# plt.legend(('cdf', 'histogram'), loc='upper left')
#
# plt.subplot(212)
# plt.plot(cdf1, color='b')
# plt.hist(img2.flatten(), 256, [0, 256], color='r')
# plt.xlim([0, 256])
# plt.legend(('cdf', 'histogram'), loc='upper left')
# plt.show()

# import cv2 as cv
#
# img = cv.imread('test.jpg', 0)
# equ = cv.equalizeHist(img)
# res = np.hstack((img, equ))
# cv.imshow('img', res)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 为了解决上述2个问题, 就有2方面的解决方法: 一是解决全局性问题, 二是解决背景噪声增强问题.
# #
# # 针对全局性问题: 有人提出了对图像分块的方法, 每块区域单独进行直方图均衡, 这样就可以利用局部信息来增强图像, 这样就可以解决全局性问题;
# # 针对背景噪声增强问题: 主要背景增强太过了, 因而有人提出了对对比度进行限制的方法, 这样就可以解决背景噪声增强问题;
# # 将上述二者相结合就是 CLAHE 方法, 其全称为: Contrast Limited Adaptive Histogram Equalization.
# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg', 0)
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cll = clahe.apply(img)
# cv.imshow('img', img)
# cv.imshow('cll', cll)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg')
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# # hist, xbins, ybins = np.histogram2d(h.ravel(), s.ravel(), [180, 256], [[0, 180], [0, 256]]
# plt.imshow(hist, interpolation='nearest')
# plt.show()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# roi = cv.imread('test.jpg')
# hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
#
# target = cv.imread('test1.jpg')
# hsvt = cv.cvtColor(target, cv.COLOR_BGR2HSV)
#
# M = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# I = cv.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
#
# R = M / I
# h, s, v = cv.split(hsvt)
# B = R[h.ravel(), s.ravel()]
# B = np.minimum(B, 1)
# B = B.reshape(hsvt.shape[:2])
#
# disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
# cv.filter2D(B, -1, disc, B)
# B = np.uint8(B)
# cv.normalize(B, B, 0, 255, cv.NORM_MINMAX)
#
# ret,thresh = cv.threshold(B,50,255,0)

# import numpy as np
# import cv2 as cv
#
# roi = cv.imread('test.jpg')
# roi = cv.pyrDown(roi)
# hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# target = cv.imread('test1.jpg')
# target = cv.pyrDown(target)
# hsvt = cv.cvtColor(target, cv.COLOR_BGR2HSV)
#
# roihist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
#
# cv.normalize(roihist, roihist, 0, 255, cv.NORM_MINMAX)
# dst = cv.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)
#
# disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
# cv.filter2D(dst, -1, disc, dst)
#
# ret, thresh = cv.threshold(dst, 50, 255, 0)
# thresh = cv.merge((thresh, thresh, thresh))
# res = cv.bitwise_and(target, thresh)
# res = np.vstack((target, thresh, res))
# cv.imshow('res', res)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg', 0)
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20 * np.log(np.abs(fshift))
#
# rows, cols = img.shape
# crow, ccol = rows // 2, cols // 2
# fshift[crow - 30:crow + 31, ccol - 30:ccol + 31] = 0
# f_ishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.real(img_back)
#
# # plt.subplot(121), plt.imshow(img, cmap='gray')
# # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# # plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
# # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# # plt.show()
#
# plt.subplot(131), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132), plt.imshow(img_back, cmap='gray')
# plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(img_back)
# plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
# plt.show()


# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg', 0)
# dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
# magnitude_spectum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
#
# # plt.subplot(121), plt.imshow(img, cmap='gray')
# # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# # plt.subplot(122), plt.imshow(magnitude_spectum, cmap='gray')
# # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# # plt.show()
#
# rows, cols = img.shape
# # / 在python3中是浮点数除法，// 是向下取整
# crow, ccol = rows // 2, cols // 2
#
# mask = np.zeros((rows, cols, 2), np.uint8)
# mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
#
# fshift = dft_shift * mask
# f_ishift = np.fft.ifftshift(fshift)
# img_back = cv.idft(f_ishift)
# img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(img_back, cmap='gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()
#
# print("{} {}".format(rows, cols))
#
# nrows = cv.getOptimalDFTSize(rows)
# ncols = cv.getOptimalDFTSize(cols)
# print("{} {}".format(nrows, ncols))
#
# nimg = np.zeros((nrows, ncols))
# nimg[:rows, :cols] = img

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# mean_filter = np.ones((3, 3))
#
# x = cv.getGaussianKernel(5, 10)
# gaussian = x * x.T
#
# scharr = np.array([[-3, 0, 3],
#                    [-10, 0, 10],
#                    [-3, 0, 3]])
#
# sobel_x = np.array([[-1, 0, 1],
#                     [-2, 0, 2],
#                     [-1, 0, 1]])
#
# sobel_y = np.array([[-1, -2, -1],
#                     [0, 0, 0],
#                     [1, 2, 1]])
#
# laplacian = np.array([[0, 1, 0],
#                       [1, -4, 1],
#                       [0, 1, 0]])
#
# filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
# filter_name = ['mean_filter', 'gaussian', 'laplacian', 'sobel_x', 'sobel_y', 'scharr_x']
# fft_filters = [np.fft.fft2(x) for x in filters]
# fft_shift = [np.fft.fftshift(y) for y in fft_filters]
# mag_spectrum = [np.log(np.abs(z) + 1) for z in fft_shift]
# for i in range(6):
#     plt.subplot(2, 3, i + 1), plt.imshow(mag_spectrum[i], cmap='gray')
#     plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
# plt.show()


# # 方差匹配方法：匹配度越高，值越接近于0。
# #
# #                归一化方差匹配方法：完全匹配结果为0。
# #
# #                相关性匹配方法：完全匹配会得到很大值，不匹配会得到一个很小值或0。
# #
# #                归一化的互相关匹配方法：完全匹配会得到1， 完全不匹配会得到0。
# #
# #                相关系数匹配方法：完全匹配会得到一个很大值，完全不匹配会得到0，完全负相关会得到很大的负数。
# #
# #       （此处与书籍以及大部分分享的资料所认为不同，研究公式发现，只有归一化的相关系数才会有[-1,1]的值域）
# #
# #                归一化的相关系数匹配方法：完全匹配会得到1，完全负相关匹配会得到-1，完全不匹配会得到0。
# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg', 0)
# img2 = img.copy()
# template = cv.imread('test1.jpg', 0)
# w, h = template.shape[::-1]
#
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF',
#            'cv.TM_SQDIFF_NORMED']
#
# for meth in methods:
#     img = img2.copy()
#     method = eval(meth) # 执行字符串表达的内容
#
#     res = cv.matchTemplate(img, template, method)
#     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
#
#     if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv.rectangle(img, top_left, bottom_right, 255, 2)
#     plt.subplot(121), plt.imshow(res, cmap='gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122), plt.imshow(img, cmap='gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# img_rgb = cv.imread('test.jpg')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# template = cv.imread('test1.jpg', 0)
# w, h = template.shape[::-1]
# res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
# threshold = 0.8
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]): # 加星号把列表里的元素取出来，这样就不会是把整个列表看成一个元素了
#     cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
# cv.imshow('res', res)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv
# import numpy as np
#
# img = cv.imread(cv.samples.findFile('test.png'))
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray, 50, 150, apertureSize=3)
# lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
# print(lines)
# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * (a))
#     cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
# cv.imshow('houghlines all', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# import cv2 as cv
# import numpy as np
# img = cv.imread(cv.samples.findFile('sudoku.png'))
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray,50,150,apertureSize = 3)
# lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
# for line in lines:
#  x1,y1,x2,y2 = line[0]
#  cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
# cv.imwrite('houghlines probability',img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg', 0)
# img = cv.medianBlur(img, 5)
# cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=70, param2=50, minRadius=50, maxRadius=200)
# circles = np.uint16(np.around(circles))
# for i in circles[0, :]:
#     cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
#     cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
# cv.imshow('img', cimg)
# cv.waitKey(0)
# cv.destroyAllWindows()


# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# cv.imshow('gray', gray)
#
# ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # ret/retval return value
#
# cv.imshow('thresh', thresh)
#
# kernel = np.ones((3, 3), np.uint8)
# opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
#
# cv.imshow('opening', opening)
#
# sure_bg = cv.dilate(opening, kernel, iterations=3)
#
# cv.imshow('sure_bg', sure_bg)
#
# dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
#
# cv.imshow('dist_transform', dist_transform)
#
# ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
#
# cv.imshow('sure_fg', sure_fg)
#
# sure_fg = np.uint8(sure_fg)
# unknown = cv.subtract(sure_bg, sure_fg)
#
# cv.imshow('unkown', unknown)
#
# ret, markers = cv.connectedComponents(sure_fg)
#
# markers = markers + 1
#
# markers[unknown == 255] = 0 # 通过marker = 0的操作，主要是标记处边界可能存在的位置，需要分水岭去解决
#
# markers = cv.watershed(img, markers)
# img[markers == -1] = [255, 0, 0]
#
# cv.imshow('img', img)
#
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg')
# mask = np.zeros(img.shape[:2], np.uint8)
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
# rect = (50, 50, 450, 290)
# cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# img = img * mask2[:, :, np.newaxis]
# plt.imshow(img)
# plt.colorbar()
# plt.show()

# import cv2 as cv
# import numpy as np
#
# filename = 'test.jpg'
# img = cv.imread(filename)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv.cornerHarris(gray, 2, 3, 0.04)
#
# dst = cv.dilate(dst, None)
#
# img[dst > 0.01 * dst.max()] = [0, 0, 255]
# cv.imshow('dst', img)
# if cv.waitKey(0) & 0xff == 27:
#     cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
#
# filename = 'test.jpg'
# img = cv.imread(filename)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv.cornerHarris(gray, 2, 3, 0.04)
# dst = cv.dilate(dst, None)
# ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
# dst = np.uint8(dst)
# # 连通域分析
# ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
# # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
#
# res = np.hstack((centroids, corners))
# res = np.int0(res)
# img[res[:, 1], res[:, 0]] = [0, 0, 255]
# img[res[:, 3],res[:, 2]] = [0, 255, 0]
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
# corners = np.int0(corners)
# for i in corners:
#     print(i)
#     x, y = i.ravel() # 拉成一维数组
#     cv.circle(img, (x, y), 3, (255, 0, 0), -1)
# plt.imshow(img)
# plt.show()

# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# sift = cv.xfeatures2d.SIFT_create()
# kp = sift.detect(gray, None)
# # kp, des = sift.detectAndCompute(gray, None)
# img = cv.drawKeypoints(gray, kp, img)
# cv.imshow('sift_keypoints', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg', 0)
#
# surf = cv.xfeatures2d.SURF_create(50)
# kp, des = surf.detectAndCompute(img, None)
#
# img2 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)
# print(surf.getUpright())
# surf.setUpright(True)
#
# kp3 = surf.detect(img, None)
# img3 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)
# plt.imshow(img2)
# plt.imshow(img3)
# plt.show()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg')
# fast = cv.FastFeatureDetector_create()
#
# kp = fast.detect(img, None)
# img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
#
# print("Threshold: {}".format(fast.getThreshold()))
# print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
# print("neighborhood: {}".format(fast.getType()))
# print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
#
# cv.imshow('img2', img2)
# fast.setNonmaxSuppression(0)
# kp = fast.detect(img, None)
# print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
#
# img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
# cv.imshow('img3', img3)
#
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg', 0)
#
# star = cv.xfeatures2d.StarDetector_create()
#
# brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
#
# kp = star.detect(img, None)
# kp, des = brief.compute(img, kp)
# print(brief.descriptorSize())
# print(des.shape)

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg', 0)
#
# orb = cv.ORB_create()
#
# kp = orb.detect(img, None)
#
# kp, des = orb.compute(img, kp)
#
# img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
# plt.imshow(img2)
# plt.show()

# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
#
# img1 = cv.imread('test.jpg', 0)
# img2 = cv.imread('test2.jpg', 0)
#
# orb = cv.ORB_create()
#
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)
#
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)
# img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3)
# plt.show()

# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
#
# img1 = cv.imread('test.jpg', cv.IMREAD_GRAYSCALE)
# img2 = cv.imread('test1.jpg', cv.IMREAD_GRAYSCALE)
#
# sift = cv.xfeatures2d.SIFT_create()
#
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
#
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)
#
# good = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good.append([m])
#
# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3)
# plt.show()

import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
#
# img1 = cv.imread('test.jpg', 0)
# img2 = cv.imread('test1.jpg', 0)
# sift = cv.xfeatures2d.SIFT_create()
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
#
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)
# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1, des2, k=2)
#
# matchesMask = [[0, 0] for i in range(len(matches))]
# for i, (m, n) in enumerate(matches):
#     if m.distance < 0.7 * n.distance:
#         matchesMask[i] = [1, 0]
# draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask,
#                    flags=cv.DrawMatchesFlags_DEFAULT)
#
# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
# plt.imshow(img3)
# plt.show()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# MIN_MATCH_COUNT = 10
#
# img1 = cv.imread('test.jpg', 0)
# img2 = cv.imread('test1.jpg', 0)
#
# sift = cv.xfeatures2d.SIFT_create()
#
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)
# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1, des2, k=2)
# good = []
# for m, n in matches:
#     if m.distance < 0.7 * n.distance:
#         good.append(m)
#
# if len(good) > MIN_MATCH_COUNT:
#     # 之前一直不明白match与knnmatch的返回值到底是什么，查阅了一些资料才理解。
#     #
#     # 其实二者都是返回的DMatch类型的数据结构。
#     # 那么这个这个DMatch数据结构究竟是什么呢？
#     # 它包含三个非常重要的数据分别是queryIdx，trainIdx，distance
#     # 先说一下这三个分别是什么在演示其用途：
#     # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
#     # trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
#     # distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
#     # 每个特征点本身也具有以下属性：.pt:关键点坐标，.angle：表示关键点方向，.response表示响应强度，.size:标书该点的直径大小。
#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()
#     h, w= img1.shape
#     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0], [w - 1, 0]]).reshape(-1, 1, 2)
#     dst = cv.perspectiveTransform(pts, M)
#     img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
# else:
#     print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
#     matchesMask = None
#
# draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
# img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
# plt.imshow(img3, 'gray')
# plt.show()

# from __future__ import print_function
# import cv2 as cv
# import argparse
#
# parser = argparse.ArgumentParser(
#     description='This program shows how to use background subtraction methods provided by OpenCV. You can process both videos and images.')
# parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
# parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
# args = parser.parse_args()
#
# if args.algo == 'MOG2':
#     backSub = cv.createBackgroundSubtractorMOG2()
# else:
#     backSub = cv.createBackgroundSubtractorKNN()
# capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
# if not capture.isOpened:
#     print('Unable to open: ' + args.input)
#     exit(0)
# while True:
#     ret, frame = capture.read()
#     if frame is None:
#         break
#
#     fgMask = backSub.apply(frame)
#
#     cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
#
#     cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#
#     cv.imshow('Frame', frame)
#     cv.imshow('FG Maks',fgMask)
#
#     keyboard = cv.waitKey(30)
#     if keyboard == 'q' or keyboard == 27:
#         break
#

# import numpy as np
# import cv2 as cv
# import argparse
#
# parser = argparse.ArgumentParser(
#     description='This sample demonstrates the meanshift algorithm. The example file can be downloaded from: https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# args = parser.parse_args()
# cap = cv.VideoCapture(args.image)
# # 视频的第一帧
# ret, frame = cap.read()
# # 设置窗口的初始位置
# x, y, w, h = 300, 200, 100, 50
# track_window = (x, y, w, h)
#
# roi = frame[y:y + h, x:x + w]
# hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2GHSV)
# # OpenCV中的inRange()函数可实现二值化功能（这点类似threshold()函数），更关键的是可以同时针对多通道进行操作，使用起来非常方便！主要是将在两个阈值内的像素值设置为白色（255），而不在阈值区间内的像素值设置为黑色（0），该功能类似于之间所讲的双阈值化操作。
# mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
# roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
# # 设置终止条件，可以是10次迭代，也可以至少移动1 pt
# term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
# while (1):
#     ret, frame = cap.read()
#     if ret == True:
#         hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#         dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
#         # 应用meanshift来获取新位置
#         ret, track_window = cv.meanShift(dst, track_window, term_crit)
#         # 在图像上绘制
#         x, y, w, h = track_window
#         img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
#         cv.imshow('img2', img2)
#         k = cv.waitKey(30) & 0xff
#         if k == 27:
#             break
#     else:
#         break

# import numpy as np
# import cv2 as cv
# import argparse
#
# parser = argparse.ArgumentParser(
#     description='This sample demonstrates the camshift algorithm. The example file can be downloaded from: https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# args = parser.parse_args()
# cap = cv.VideoCapture(args.image)
# # 获取视频第一帧
# ret, frame = cap.read()
# # 设置初始窗口
# x, y, w, h = 300, 200, 100, 50  # simply hardcoded the values
# track_window = (x, y, w, h)
# # 设置追踪的ROI窗口
# roi = frame[y:y + h, x:x + w]
# hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
# roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
# # 设置终止条件，可以是10次迭代，有可以至少移动1个像素
# term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
# while(1):
#      ret, frame = cap.read()
#      if ret == True:
#          hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#          dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#          # 应用camshift 到新位置
#          ret, track_window = cv.CamShift(dst, track_window, term_crit)
#          # 在图像上画出来
#          pts = cv.boxPoints(ret)
#          pts = np.int0(pts)
#          img2 = cv.polylines(frame,[pts],True, 255,2)
#          cv.imshow('img2',img2)
#          k = cv.waitKey(30) & 0xff
#          if k == 27:
#             break
#      else:
#         break

# import numpy as np
# import cv2 as cv
# import argparse
#
# parser = argparse.ArgumentParser(
#     description='This sample demonstrates LucasKanade Optical Flow calculation. The example file can be downloaded  from: https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# args = parser.parse_args()
# cap = cv.VideoCapture(args.image)
# # 用于ShiTomasi拐点检测的参数
# feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
# # lucas kanade光流参数
# lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# # 创建一些随机的颜色
# color = np.random.randint(0, 255, (100, 3))
# # 拍摄第一帧并在其中找到拐角
# ret, old_frame = cap.read()
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# # 创建用于作图的掩码图像
# mask = np.zeros_like(old_frame)
# while (1):
#     ret, frame = cap.read()
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # 计算光流
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#     # 选择良好点
#     good_new = p1[st == 1]
#     good_old = p0[st == 1]
#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         a, b = np.int0(new.ravel())
#         c, d = np.int0(new.ravel())
#         mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
#         frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
#     img = cv.add(frame, mask)
#     cv.imshow('frame', img)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1, 1, 2)

# import numpy as np
# import cv2 as cv
#
# cap = cv.VideoCapture('test.MP4')
# ret, frame1 = cap.read()
# prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame1)
# # 它是省略所有的冒号而用省略号代替。
# #
# # a[:, :, None]和a[..., None]的输出是一样的，就是因为...代替了前面两个冒号
# hsv[..., 1] = 255
# while (1):
#     ret, frame2 = cap.read()
#     next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
#     flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
#     hsv[..., 0] = ang * 180 / np.pi / 2
#     hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
#     bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
#     cv.imshow('frame2', bgr)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv.imshow('frame2', frame2)
#         cv.imshow('bgr', bgr)
#     prvs = next

# import numpy as np
# import cv2 as cv
# import glob
#
# # 终止条件
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# # 准备对象点， 如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6 * 7, 3), np.float32)
# objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
# # 用于存储所有图像的对象点和图像点的数组。
# objpoints = []  # 真实世界中的3d点
# imgpoints = []  # 图像中的2d点
# images = glob.glob('test4.png')
# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # cv.imshow('gray', gray)
#     # cv.waitKey(0)
#     # 找到棋盘角落
#     ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
#     print(ret)
#     # 如果找到，添加对象点，图像点（细化之后）
#     if ret == True:
#         objpoints.append(objp)
#         corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         imgpoints.append(corners)
#         ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#         h, w = img.shape[:2]
#         newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
#         # undistort 使用cv.undistort() 这是最简单的方法。只需调用该函数并使用上面获得的ROI裁剪结果即可。
#         dst = cv.undistort(img, mtx, dist, None, newcameramtx)
#         # 剪裁图像
#         x, y, w, h = roi
#         dst = dst[y:y + h, x:x + w]
#         cv.imshow('calibresult', dst)
#         cv.waitKey(0)
#         # undistort 使用remapping 该方式有点困难。首先，找到从扭曲图像到未扭曲图像的映射函数。然后使用重映射功能。
#         mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
#         dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
#         # 裁剪图像
#         x, y, w, h = roi
#         dst = dst[y:y + h, x:x + w]
#         cv.imwrite('calibresult2', dst)
#         cv.waitKey(0)
#         # 为了找到平均误差，我们计算为所有校准图像计算的误差的算术平均值。
#         mean_error = 0
#         for i in range(len(objpoints)):
#             imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#             error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
#             mean_error += error
#         print("total error: {}".format(mean_error / len(objpoints)))
#         # 绘制并显示拐角
#         cv.drawChessboardCorners(img, (7, 6), corners2, ret)
#         cv.imshow('img', img)
#         cv.waitKey(0)
# cv.destroyAllWindows()


# import numpy as np
# import cv2 as cv
# import glob
#
#
# def draw(img, corners, imgpts):
#     corner = tuple(corners[0].ravel())
#     img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
#     img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
#     img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
#     return img
#
# # 终止条件
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# # 准备对象点， 如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6 * 7, 3), np.float32)
# objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
# axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
# # 用于存储所有图像的对象点和图像点的数组。
# objpoints = []  # 真实世界中的3d点
# imgpoints = []  # 图像中的2d点
# images = glob.glob('test4.png')
# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # cv.imshow('gray', gray)
#     # cv.waitKey(0)
#     # 找到棋盘角落
#     ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
#     print(ret)
#     # 如果找到，添加对象点，图像点（细化之后）
#     if ret == True:
#         objpoints.append(objp)
#         corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         imgpoints.append(corners)
#         ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#         corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         # 找到旋转和平移矢量。
#         ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
#         # 将3D点投影到图像平面
#         imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
#         img = draw(img, corners2, imgpts)
#         cv.imshow('img', img)
#         k = cv.waitKey(0) & 0xff
#         if k == ord('s'):
#             cv.imshow(fname[:6] + '.png', img)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
#
# def drawlines(img1, img2, lines, pts1, pts2):
#     ''' img1 - 我们在img2相应位置绘制极点生成的图像
#      lines - 对应的极点 '''
#     r, c = img1.shape
#     img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
#     img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
#     for r, pt1, pt2 in zip(lines, pts1, pts2):
#         color = tuple(np.random.randint(0, 255, 3).tolist())
#         x0, y0 = map(int, [0, -r[2] / r[1]])
#         x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
#         img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
#         img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
#         img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
#     return img1, img2
#
#
# img1 = cv.imread('test.jpg', 0)
# img2 = cv.imread('test1.jpg', 0)
# sift = cv.SIFT()
# # 使用SIFT查找关键点和描述符
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
#
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)
# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1, des2, k=2)
# good = []
# pts1 = []
# pts2 = []
# # 根据Lowe的论文进行比率测试
# for i, (m, n) in enumerate(matches):
#     if m.distance < 0.8 * n.distance:
#         good.append(m)
#         pts2.append(kp2[m.trainIdx].pt)
#         pts1.append(kp1[m.queryIdx].pt)
#
# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)
# F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
# # 我们只选择内点
# pts1 = pts1[mask.ravel() == 1]
# pts2 = pts2[mask.ravel() == 1]
#
# # 在右图（第二张图）中找到与点相对应的极点，然后在左图绘制极线
# lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
# lines1 = lines1.reshape(-1, 3)
# img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
# # 在左图（第一张图）中找到与点相对应的Epilines，然后在正确的图像上绘制极线
# lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
# lines2 = lines2.reshape(-1, 3)
# img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
# plt.subplot(121)
# plt.imshow(img5)
# plt.subplot(122)
# plt.imshow(img3)
# plt.show()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# imgL = cv.imread('test.jpg', 0)
# imgR = cv.imread('test1.jpg', 0)
# stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(imgL, imgR)
# plt.imshow(disparity, 'gray')
# plt.show()

# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)
# responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)
#
# red = trainData[responses.ravel() == 0]
# plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')
#
# blue = trainData[responses.ravel() == 1]
# plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')
#
# newcomer = np.random.randint(0, 100, (10, 2)).astype(np.float32)
# plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o')
# knn = cv.ml.KNearest_create()
# knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
# ret, results, neighbours, dist = knn.findNearest(newcomer, 3)
# print("result: {}\n".format(results))
# print("neighbours: {}\n".format(neighbours))
# print("distance: {}\n".format(dist))
# plt.show()

# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
#
# x = np.array(cells)
#
# train = x[:, :50].reshape(-1, 400).astype(np.float32)
# test = x[:, 50:100].reshape(-1, 400).astype(np.float32)
#
# k = np.arange(10)
# train_labels = np.repeat(k, 250)[:, np.newaxis]
# test_labels = train_labels.copy()
#
# knn = cv.ml.KNearest_create()
# knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
# ret, result, neighbours, dist = knn.findNearest(test, k=5)
#
# matches = result == test_labels
# correct = np.count_nonzero(matches)
# accuracy = correct * 100.0 / result.size
# print(accuracy)
#
# np.savez('knn_data.npz', train=train, train_labels=train_labels)
#
# with np.load('knn_data.npz') as data:
#     print(data.files)
#     train = data['train']
#     train_labels = data['train_labels']

import cv2 as cv
# import numpy as np
#
# data = np.loadtxt('letter-recognition.data', dtype='float32', delimiter=',',
#                   converters={0: lambda ch: ord(ch) - ord('A')})
#
# train, test = np.vsplit(data, 2)
#
# responses, trainData = np.hsplit(train, [1])
# labels, testData = np.hsplit(test, [1])
#
# knn = cv.ml.KNearest_create()
# knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
# ret, result, neighbours, dist = knn.findNearest(testData, k=5)
# correct = np.count_nonzero(result == labels)
# accuracy = correct * 100.0 / 10000
# print(accuracy)

# import cv2 as cv
# import numpy as np
#
# SZ = 20
# bin_n = 16
# affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR
#
#
# def deskew(img):
# opencv中提供了moments()来计算图像中的中心矩(最高到三阶)，HuMoments()用于由中心矩计算Hu矩.同时配合函数contourArea函数计算轮廓面积和arcLength来计算轮廓或曲线长度
#     m = cv.moments(img)
#     if abs(m['mu02']) < 1e-2:
#         return img.copy()
#     skew = m['mu11'] / m['mu02']
#     M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
#     img = cv.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
#     return img
#
#
# def hog(img):
#     gx = cv.Sobel(img, cv.CV_32F, 1, 0)
#     gy = cv.Sobel(img, cv.CV_32F, 0, 1)
#     mag, ang = cv.cartToPolar(gx, gy)
#     bins = np.int32(bin_n * ang / (2 * np.pi))
#     bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
#     mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
#     hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
#     hist = np.hstack(hists)
#     return hist
#
#
# img = cv.imread('digits.png', 0)
# if img is None:
#     raise Exception("we need the digits.png image from samples/data here !")
# cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
#
# train_cells = [i[:50] for i in cells]
# test_cells = [i[50:] for i in cells]
# # map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
# deskewed = [list(map(deskew, row)) for row in train_cells]
# hogdata = [list(map(hog, row)) for row in deskewed]
# trainData = np.float32(hogdata).reshape(-1, 64)
# responses = np.repeat(np.arange(10), 250)[:, np.newaxis]
# svm = cv.ml.SVM_create()
# svm.setKernel(cv.ml.SVM_LINEAR)
# svm.setType(cv.ml.SVM_C_SVC)
# svm.setC(2.67)
# svm.setGamma(5.383)
# svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
# svm.save('svm_data.dat')
# deskewed = [list(map(deskew, row)) for row in test_cells]
# hogdata = [list(map(hog, row)) for row in deskewed]
# testData = np.float32(hogdata).reshape(-1, bin_n * 4)
# result = svm.predict(testData)[1]
# mask = result == responses
# correct = np.count_nonzero(mask)
# print(correct * 100.0 / result.size)

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# x = np.random.randint(25, 100, 25)
# y = np.random.randint(175, 255, 25)
# z = np.hstack((x, y))
# z = z.reshape((50, 1))
# z = np.float32(z)
# plt.hist(z, 256, [0, 256])
# plt.show()
#
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#
# flags = cv.KMEANS_RANDOM_CENTERS
#
# compactness, labels, centers = cv.kmeans(z, 2, None, criteria, 10, flags)
#
# A = z[labels == 0]
# B = z[labels == 1]
#
# plt.hist(A, 256, [0, 256], color='r')
# plt.hist(B, 256, [0, 256], color='b')
# plt.hist(centers, 32, [0, 256], color='y')
# plt.show()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# X = np.random.randint(25, 50, (25, 2))
# Y = np.random.randint(60, 85, (25, 2))
# Z = np.vstack((X, Y))
#
# Z = np.float32(Z)
#
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# ret, label, center = cv.kmeans(Z, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
#
# A = Z[label.ravel() == 0]
# B = Z[label.ravel() == 1]
#
# plt.scatter(A[:, 0], A[:, 1])
# plt.scatter(B[:, 0], B[:, 1], c='r')
# plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
# plt.xlabel('Height')
# plt.ylabel('Weight')
# plt.show()

# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg')
# Z = img.reshape((-1, 3))
#
# Z = np.float32(Z)
#
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 8
# ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
#
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((img.shape))
# cv.imshow('res2', res2)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('test.jpg')
# 图像中的蓝色补丁看起来很相似。绿色补丁看起来很相似。因此，我们获取一个像素，在其周围
# 获取一个小窗口，在图像中搜索相似的窗口，对所有窗口求平均，然后用得到的结果替换该像
# 素。此方法是“非本地均值消噪”。与我们之前看到的模糊技术相比，它花费了更多时间，但是效果
# 非常好。更多信息和在线演示可在其他资源的第一个链接中找到
# dst = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
# plt.subplot(121)
# plt.imshow(img)
# plt.subplot(122)
# plt.imshow(dst)
# plt.show()

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# cap = cv.VideoCapture('test.jpg')
#
# img = [cap.read()[1] for i in range(5)]
#
# gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in img]
# gray = [np.float64(i) for i in gray]
#
# noise = np.random.randn(*gray[1].shape) * 10
#
# noisy = [1 + noise for i in gray]
#
# noisy = [np.uint8(np.clip(i, 0, 255)) for i in noisy]
#
# dst = cv.fastNlMeansDenoisingColoredMulti(noisy, 2, 5, None, 4, 7, 35)
# plt.subplot(131),plt.imshow(gray[2],'gray')
# plt.subplot(132),plt.imshow(noisy[2],'gray')
# plt.subplot(133),plt.imshow(dst,'gray')
# plt.show()

# import numpy as np
# import cv2 as cv
#
# img = cv.imread('test.jpg')
# mask = cv.imread('test1.jpg', 0)
# # 基本思路很简单：用邻近的像素替换那些坏标记，使其看起来像是邻居
# dst = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv
# import numpy as np
#
# img_fn = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]
# img_list = [cv.imread(fn) for fn in img_fn]
# exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)
#
# merge_debevec = cv.createMergeDebevec()
# hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
# merge_robertson = cv.createMergeRobertson()
# hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())
#
# tonemap1 = cv.createTonemap(gamma=2.2)
# res_debevec = tonemap1.process(hdr_debevec.copy())
#
# merge_mertens = cv.createMergeMertens()
# res_mertens = merge_mertens.process(img_list)
#
# res_debevec_8bit = np.clip(res_debevec * 255, 0, 255).astype('uint8')
# hdr_robertson_8bit = np.clip(hdr_robertson * 255, 0, 255).astype('uint8')
# res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
# cv.imwrite("ldr_debevec.jpg", res_debevec_8bit)
# cv.imwrite("ldr_robertson.jpg", hdr_robertson_8bit)
# cv.imwrite("fusion_mertens.jpg", res_mertens_8bit)
#
# cal_debevec = cv.createCalibrateDebevec()
# crf_debevec = cal_debevec.process(img_list, times=exposure_times)
# hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy(), response=crf_debevec.copy())
#
# cal_robertson = cv.createCalibrateRobertson()
# crf_robertson = cal_robertson.process(img_list, times=exposure_times)
# hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy(), response=crf_robertson.copy())


# from __future__ import print_function
# import cv2 as cv
# import argparse
#
#
# def detectAndDisplay(frame):
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     frame_gray = cv.equalizeHist(frame_gray)
#     faces = face_cascade.detectMultiScale(frame_gray)
#     for (x, y, w, h) in faces:
#         center = (x + w // 2, y + h / 2)
#         frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
#         faceROI = frame_gray[y:y + h, x:x + w]
#
#         eyes = eyes_cascade.detectMultiScale(faceROI)
#         for (x2, y2, w2, h2) in eyes:
#             eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
#             radius = int(round((w2 + h2) * 0.25))
#             frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
#     cv.imshow('Capture - Face detection', frame)
#
#
# parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
# parser.add_argument('--face_cascade', help='Path to face cascade.',
#                     default='data/haarcascades/haarcascade_frontalface_alt.xml')
# parser.add_argument('--eyes_cascade', help='Path to eyes cascade.',
#                     default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
# parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
# args = =parser.parse_args()
#
# face_cascade_name = args.face_cascade
# eyes_cascade_name = args.eyes_cascade
# face_cascade = cv.CascadeClassifier()
# eyes_cascade = cv.CascadeClassifier()
#
# if not face_cascade.load(cv.samples.findFile(eyes_cascade_name)):
#     print('--(!)Error loading face cascade')
#     exit(0)
# if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
#     print('--(!)Error loading eyes cascade')
#     exit(0)
# camera_device = args.camera
#
# cap = cv.VideoCapture(camera_device)
# if not cap.isOpened:
#     print('--(!)Error opening video capture')
#     exit(0)
# while True:
#     ret, frame = cap.read()
#     if frame is None:
#         print('--(!) No captured frame -- Break!')
#         break
#     detectAndDisplay(frame)
#     if cv.waitKey(10) == 27:
#         break


