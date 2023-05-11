//#include"MSERTest.h"
//
//using namespace std;
//using namespace cv;
//// 自己写的
////struct ERinfo
////{
////	std::vector<cv::Point> p;
////	int graylevel;
////	int growhistory;
////};
////
////struct GHinfo
////{
////	std::vector<cv::Point> p;
////	int graylevel;
////	int parent;
////	int child;
////};
////
////void MSERTest(cv::Mat& src, int delta, int maxVariation, int minDiversity, int minArea, int maxArea, std::vector<std::vector<cv::Point> >& points, std::vector<cv::Rect>& rects)
////{
////	std::vector<std::vector<cv::Point> > BoundHeap;
////	std::vector<ERinfo > ERStack;
////	std::vector<GHinfo > GrowHistory;
////	cv::Mat mark(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
////	cv::Point current_pixel(0,0);
////	cv::Point next_pixel(0, 0);
////	cv::Point previos_pixel(0, 0);
////	bool is_next = false;
////	bool is_empty = true;
////	bool endup_merge_stack = false;
////
////	// init
////	int cnt = 0;
////	ERStack[0].graylevel = 256;
////
////	int H_cnt = 0;
////
////	// mark current pixel
////	mark.at<uchar>(current_pixel) = 1;
////	cnt = cnt + 1;
////	ERStack[cnt].graylevel = src.at<uchar>(current_pixel);
////
////	while (true)
////	{
////		// Find available neighbour
////		if (current_pixel.y + 1 < src.cols && mark.at<uchar>(current_pixel.x, current_pixel.y + 1) == 0)
////		{
////			next_pixel = cv::Point(current_pixel.x, current_pixel.y + 1);
////			is_next = true;
////		}
////		else if (current_pixel.x + 1 < src.rows && mark.at<uchar>(current_pixel.x + 1, current_pixel.y) == 0)
////		{
////			next_pixel = cv::Point(current_pixel.x + 1, current_pixel.y);
////			is_next = true;
////		}
////		else if (current_pixel.y - 1 > -1 && mark.at<uchar>(current_pixel.x, current_pixel.y - 1) == 0)
////		{
////			next_pixel = cv::Point(current_pixel.x, current_pixel.y - 1);
////			is_next = true;
////		}
////		else if (current_pixel.x - 1 > -1 && mark.at<uchar>(current_pixel.x - 1, current_pixel.y) == 0)
////		{
////			next_pixel = cv::Point(current_pixel.x - 1, current_pixel.y);
////			is_next = true;
////		}
////		else
////		{
////			is_next = false;
////		}
////
////		if (is_next)
////		{
////			// Have neighbours
////			// gray level < current?
////			if (src.at<uchar>(next_pixel) < src.at<uchar>(current_pixel))
////			{
////				BoundHeap[src.at<uchar>(current_pixel)].push_back(current_pixel);
////				cnt = cnt + 1;
////				ERStack[cnt].graylevel = src.at<uchar>(next_pixel);
////				current_pixel = next_pixel;
////				mark.at<uchar>(next_pixel) = 1;
////				// init
////				is_next = false;
////				continue;
////			}
////			else
////			{
////				BoundHeap[src.at<uchar>(next_pixel)].push_back(next_pixel);
////				continue;
////			}
////		}
////		else
////		{
////			// No neighbours
////			ERStack[cnt].p.push_back(current_pixel);
////
////			// Check if the heap is empty
////			for (int i = 0; i < 256; i++)
////			{
////				if (BoundHeap[i].size() == 0)
////				{
////					is_empty = true;
////				}
////				else
////				{
////					previos_pixel = current_pixel;
////					current_pixel = BoundHeap[i].at(BoundHeap[i].size() - 1);
////					BoundHeap[i].pop_back();
////					is_empty = false;
////					break;
////				}
////			}
////			
////			// is_empty
////			if (is_empty)
////			{
////				std::cout << "MSER algorithm stop!!!" << std::endl;
////				break;
////			}
////			else
////			{
////				// Same gray level as previous pixel?
////				if (src.at<uchar>(previos_pixel) == src.at<uchar>(current_pixel))
////				{
////					// Back to "Find available neighbour"
////					continue;
////				}
////				else
////				{
////					while (true)
////					{
////						// New grey level < second on stack
////						if (src.at<uchar>(current_pixel) < ERStack[cnt - 1].graylevel && ERStack[cnt - 1].graylevel != 256)
////						{
////							endup_merge_stack = true;
////							break;
////						}
////						else
////						{
////							// check
////							if (ERStack[cnt - 1].graylevel == 256)
////							{
////								// Merge ERStack
////								GrowHistory[H_cnt].p = ERStack[cnt].p;
////								GrowHistory[H_cnt].graylevel = ERStack[cnt].graylevel;
////								ERStack[cnt].graylevel = src.at<uchar>(current_pixel);
////								ERStack[cnt].growhistory = H_cnt;
////							}
////							else
////							{
////								// Merge ERStack
////								GrowHistory[H_cnt].p = ERStack[cnt].p;
////								GrowHistory[H_cnt].graylevel = ERStack[cnt].graylevel;
////								ERStack[cnt - 1].p.insert(ERStack[cnt - 1].p.end(), ERStack[cnt].p.begin(), ERStack[cnt].p.end());
////								ERStack[cnt - 1].graylevel = ERStack[cnt].graylevel;
////								ERStack[cnt - 1].growhistory = H_cnt;
////							}
////							if (H_cnt != 0 && GrowHistory[H_cnt].graylevel > GrowHistory[H_cnt - 1].graylevel)
////							{
////								GrowHistory[H_cnt].parent = H_cnt - 1;
////								GrowHistory[H_cnt - 1].child = H_cnt;
////							}
////							cnt = cnt - 1;
////							H_cnt = H_cnt + 1;
////
////							// New gray level > top of stack
////							if (src.at<uchar>(current_pixel) > ERStack[cnt].graylevel)
////							{
////								endup_merge_stack = false;
////								continue;
////							}
////							else
////							{
////								// Back to "Find available neighbour"
////								endup_merge_stack = true;
////								break;
////							}
////						}
////					}
////					if (endup_merge_stack)
////					{
////						continue;
////					}
////					
////				}
////			}
////
////		}
////	}
////
////
////}
//
//
//// 网上抄的Opencv 2.4.9
//// CvMat
//typedef struct CvMat
//{
//    int type;           // 数据类型
//    int step;           // 用字节表示行数据长度
//    int* refcount;      // 内部访问
//    int hdr_refcount;   // 内部使用
//    union {             // 指向数据区的指针
//        uchar* ptr;
//        short* s;
//        int* i;
//        float* fl;
//        double* db;
//    } data;    
//    union {             // 行数
//        int rows;
//        int height;
//    };
//    union {             // 列数
//        int cols;
//        int width;
//    };
//} CvMat; // 矩阵结构头
//// 基本结构
//// 节点指针链表
//typedef struct LinkedPoint
//{
//    struct LinkedPoint* prev;
//    struct LinkedPoint* next;
//    cv::Point pt;
//}
//LinkedPoint;
//
//// the history of region grown
//typedef struct MSERGrowHistory
//{
//    // 快捷路径，是指向以前历史的指针。因为不是一个一个连接的，所以不是parent。算法中是记录灰度差为delta的历史的指针。
//    // 例如：当前是灰度是10，delta=3，这个指针就指向灰度为7时候的历史
//    struct MSERGrowHistory* shortcut;
//    // 指向更新历史的指针，就是从这个历史繁衍的新历史，所以叫孩子
//    struct MSERGrowHistory* child;
//    // 大于零代表稳定，值是稳定是的像素数。这个值在不停的继承
//    int stable; // when it ever stabled before, record the size
//    // 灰度值
//    int val;
//    // 像素数
//    int size;
//}
//MSERGrowHistory;
//
//typedef struct MSERConnectedComp
//{
//    // 像素点链的头
//    LinkedPoint* head;
//    // 像素点链的尾
//    LinkedPoint* tail;
//    // 区域上次的增长历史，可以通过找个历史找到之前的记录
//    MSERGrowHistory* history;
//    // 灰度值
//    unsigned long grey_level;
//    // 像素数
//    int size;
//    int dvar; // the derivative of last var
//    float var; // the current variation (most time is the variation of one-step back)
//}
//MSERConnectedComp;
//
//// c++结构体指针，顾名思义就是指向结构体的一个指针，这篇博客作用是记录c++结构体指针的常用用法及我经常犯的一个错误。
//// 定义结构体：
////struct My {
////    My* left;
////    My* right;
////    int val;
////    My() {}
////    My(int val) :left(NULL), right(NULL), val(val) {}
////};
//// 一般结构体变量的访问方式：
////void test1() {
////    My m;
////    m.val = 1;
////    cout << m.val << endl;
////}
//// 可见，结构体中的变量，可以直接通过点操作符来访问。
//// 而对于结构体指针而言：必须通过->符号来访问指针所指结构体的变量。
////void test2() {
////    My m;
////    m.val = 1;
////    My* mm;
////    mm = &m;
////    cout << mm->val << endl;
////}
//// 声明一个结构体指针记得初始化，一定要初始化，不初始化会出事（重要的事情说三遍）
//// 如下：
////void test3() {
////    My* m;
////    m->val = 1;
////}
//// 这份代码会报一个错：空指针访问异常，这是因为m这个指针还没有初始化，因此他没有内存空间，自然就不存在有val这个参数。正确打开方式：
////void test3() {
////    My* m;
////    m = new My(3);
////    m->val = 4;
////    cout << m->val << endl;
////}
//// 以上代码用new申请了内存空间。问题即可解决。
//struct MSERParams
//{
//    MSERParams(int _delta, int _minArea, int _maxArea, double _maxVariation,
//        double _minDiversity, int _maxEvolution, double _areaThreshold,
//        double _minMargin, int _edgeBlurSize)
//        : delta(_delta), minArea(_minArea), maxArea(_maxArea), maxVariation(_maxVariation),
//        minDiversity(_minDiversity), maxEvolution(_maxEvolution), areaThreshold(_areaThreshold),
//        minMargin(_minMargin), edgeBlurSize(_edgeBlurSize)
//    {}
//
//    // MSER使用
//    int delta;                   // 两个区域间的灰度差
//    int minArea;                 // 区域最小像素数
//    int maxArea;                 // 区域最大像素数
//    double maxVariation;         // 两个区域的偏差
//    double minDiversity;         // 当前区域与稳定区域的变化率
//    // MSCR使用
//    int maxEvolution;
//    double areaThreshold;
//    double minMargin;
//    int edgeBlurSize;
//};
//
//// 数据预处理
//// to preprocess src image to following format
//    // 32-bit image
//    // > 0 is available, < 0 is visited
//    // 17~19 bits is the direction
//    // 8~11 bits is the bucket it falls to (for BitScanForward)
//    // 0~8 bits is the color
//    /** @brief 将所给原单通道灰度图和掩码图 预处理为一张方便遍历与记录数据的32位单通道图像；并且根据像素灰度值分配边缘栈。
//    * x64是小端机，低位字节在低位地址
//    * 32位格式如下：
//    * > 0 可用，< 0 已经被访问
//    * 16~18位用于记录下一个要探索的方向，5个值
//    * 8~11位 用于优化的二值搜索
//    * 0~7位用于记录灰度值
//    *@param heap_cur 边缘栈
//    *@param src 原单通道灰度图
//    *@param mask 掩码图
//    */
//static int* preprocessMSER_8UC1(CvMat* img, int*** heap_cur, CvMat* src, CvMat* mask)
//{
//    // 数据有效内容是在img中，由一圈-1包围着，靠左的区域。也就是被一圈-1的墙包围着。
//
//    // 原始数据跳转到下一行的偏移量。
//    int srccpt = src->step - src->cols;
//
//    // 跳转到下一行的偏移量，最后减一是因为，例如：xoooxxx，o是有效数据，x是扩充出来的。偏移量应该是3，就是ooo最
//    // 右边的xxx个数。为了计算，就需要减去ooo最左面的一个x。
//    int cpt_1 = img->cols - src->cols - 1;
//    int* imgptr = img->data.i;
//    int* startptr;
//
//    // 用于记录每个灰度有多少像素
//    int level_size[256];
//    for (int i = 0; i < 256; i++)
//        level_size[i] = 0;
//
//    // 设置第一行为-1
//    for (int i = 0; i < src->cols + 2; i++)
//    {
//        *imgptr = -1;
//        imgptr++;
//    }
//
//    // 偏移到第一个有效数据所在行的开头
//    imgptr += cpt_1 - 1;
//    uchar* srcptr = src->data.ptr;
//    if (mask)
//    {
//        // 有掩码
//        startptr = 0;            // 数据处理的开始位置，为最左上的位置。
//        uchar* maskptr = mask->data.ptr;
//        for (int i = 0; i < src->rows; i++)
//        {
//            // 最左面设置为-1
//            *imgptr = -1;
//            imgptr++;
//            for (int j = 0; j < src->cols; j++)
//            {
//                if (*maskptr)
//                {
//                    if (!startptr)
//                        startptr = imgptr;
//
//                    // 灰度值取反!!!!! !!!!! !!!!! !!!!!
//                    *srcptr = 0xff - *srcptr;
//
//                    // 所在灰度值个数自增
//                    level_size[*srcptr]++;
//
//                    // 写入0~8位，8~13位用作BitScanForward
//                    *imgptr = ((*srcptr >> 5) << 8) | (*srcptr);
//                }
//                else {
//                    // 标为-1，就是当作一个已经被发现的位置，和外围-1墙的原理一样
//                    *imgptr = -1;
//                }
//                imgptr++;
//                srcptr++;
//                maskptr++;
//            }
//
//            // 最右面设置为-1
//            *imgptr = -1;
//
//            // 都跳到下一行开始
//            imgptr += cpt_1;
//            srcptr += srccpt;
//            maskptr += srccpt;
//        }
//    }
//    else {
//        // 就是没有掩码的情况
//        startptr = imgptr + img->cols + 1;
//        for (int i = 0; i < src->rows; i++)
//        {
//            *imgptr = -1;
//            imgptr++;
//            for (int j = 0; j < src->cols; j++)
//            {
//                *srcptr = 0xff - *srcptr;
//                level_size[*srcptr]++;
//                *imgptr = ((*srcptr >> 5) << 8) | (*srcptr);
//                imgptr++;
//                srcptr++;
//            }
//            *imgptr = -1;
//            imgptr += cpt_1;
//            srcptr += srccpt;
//        }
//    }
//
//    // 设置最后一行为-1
//    for (int i = 0; i < src->cols + 2; i++)
//    {
//        *imgptr = -1;
//        imgptr++;
//    }
//
//    // 确定每个灰度在边界堆中的指针位置。0代表没有值。
//    heap_cur[0][0] = 0;
//    for (int i = 1; i < 256; i++)
//    {
//        heap_cur[i] = heap_cur[i - 1] + level_size[i - 1] + 1; // 很奇怪，为什么要做累加？？
//        heap_cur[i][0] = 0;
//    }
//    return startptr;
//}
//
//// 主流程及遍历方法
//static void extractMSER_8UC1_Pass(int* ioptr,
//    int* imgptr,
//    int*** heap_cur,                            // 边界栈的堆，里面是每一个灰度的栈
//    LinkedPoint* ptsptr,
//    MSERGrowHistory* histptr,
//    MSERConnectedComp* comptr,
//    int step,
//    int stepmask,
//    int stepgap,
//    MSERParams params,
//    int color,
//    CvSeq* contours,
//    CvMemStorage* storage)
//{
//    // ER栈第一项为结束的标识项，值为大于255的256
//    comptr->grey_level = 256;
//
//    // 将当前位置值入栈，并初始化
//    comptr++;
//    comptr->grey_level = (*imgptr) & 0xff;
//    initMSERComp(comptr);
//
//    // 设置为已经发现
//    *imgptr |= 0x80000000;
//
//    // 加上灰度偏移就将指针定位到了相应灰度的边界栈上
//    heap_cur += (*imgptr) & 0xff;
//
//    // 四个方向的偏移量，上下的偏移是隔行的步长
//    int dir[] = { 1, step, -1, -step };
//#ifdef __INTRIN_ENABLED__
//    unsigned long heapbit[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
//    unsigned long* bit_cur = heapbit + (((*imgptr) & 0x700) >> 8);
//#endif
//
//    // 循环
//    for (;;)
//    {
//        // take tour of all the 4 directions
//        // 提取当前像素的方向值，判断是否还有方向没有走过
//        while (((*imgptr) & 0x70000) < 0x40000)
//        {
//            // get the neighbor
//            // 通过方向对应的偏移获得相邻像素指针
//            int* imgptr_nbr = imgptr + dir[((*imgptr) & 0x70000) >> 16];
//
//            // 判断是否访问过
//            if (*imgptr_nbr >= 0) // if the neighbor is not visited yet
//            {
//                // 没有访问过，标记为访问过
//                *imgptr_nbr |= 0x80000000; // mark it as visited
//                if (((*imgptr_nbr) & 0xff) < ((*imgptr) & 0xff))
//                {
//                    // when the value of neighbor smaller than current
//                    // push current to boundary heap and make the neighbor to be the current one
//                    // create an empty comp
//                    // 如果相邻像素的灰度小于当前像素，将当前像素加入边界栈堆，并把相邻像素设置为当前像素，并新建ER栈项
//                    // 将当前加入边界栈堆
//                    (*heap_cur)++;
//                    **heap_cur = imgptr;
//
//                    // 转换方向
//                    *imgptr += 0x10000;
//
//                    // 将边界栈堆的指针调整为相邻的像素灰度所对应的位置
//                    heap_cur += ((*imgptr_nbr) & 0xff) - ((*imgptr) & 0xff);
//#ifdef __INTRIN_ENABLED__
//                    _bitset(bit_cur, (*imgptr) & 0x1f);
//                    bit_cur += (((*imgptr_nbr) & 0x700) - ((*imgptr) & 0x700)) >> 8;
//#endif
//                    // 将相邻像素设置为当前像素
//                    imgptr = imgptr_nbr;
//
//                    // 新建ER栈项，并设置灰度为当前像素灰度
//                    comptr++;
//                    initMSERComp(comptr);
//                    comptr->grey_level = (*imgptr) & 0xff;
//                    continue;
//                }
//                else {
//                    // otherwise, push the neighbor to boundary heap
//                    // 否则，将相邻像素添加到对应的边界帧堆中
//                    heap_cur[((*imgptr_nbr) & 0xff) - ((*imgptr) & 0xff)]++;
//                    *heap_cur[((*imgptr_nbr) & 0xff) - ((*imgptr) & 0xff)] = imgptr_nbr;
//#ifdef __INTRIN_ENABLED__
//                    _bitset(bit_cur + ((((*imgptr_nbr) & 0x700) - ((*imgptr) & 0x700)) >> 8), (*imgptr_nbr) & 0x1f);
//#endif
//                }
//            }
//
//            // 将当前像素的方向转换到下一个方向
//            *imgptr += 0x10000;
//        }
//
//        int imsk = (int)(imgptr - ioptr);
//
//        // 记录x&y，
//        ptsptr->pt = cvPoint(imsk & stepmask, imsk >> stepgap);
//        // get the current location
//        accumulateMSERComp(comptr, ptsptr);
//        ptsptr++;
//        // get the next pixel from boundary heap
//        // 从边界栈堆中获取一个像素用作当前像素
//        if (**heap_cur)
//        {
//            // 当前灰度的边界栈堆有值可以用，将当前边界栈堆值设置为当前像素，因为当前边界栈堆的灰度就是当前像素的灰度，所以可以直接拿出来用
//            imgptr = **heap_cur;
//
//            // 出栈
//            (*heap_cur)--;
//#ifdef __INTRIN_ENABLED__
//            if (!**heap_cur)
//                _bitreset(bit_cur, (*imgptr) & 0x1f);
//#endif
//        }
//        else {
//            // 当前灰度边界栈堆中没有值可以用
//#ifdef __INTRIN_ENABLED__
//            bool found_pixel = 0;
//            unsigned long pixel_val;
//            for (int i = ((*imgptr) & 0x700) >> 8; i < 8; i++)
//            {
//                if (_BitScanForward(&pixel_val, *bit_cur))
//                {
//                    found_pixel = 1;
//                    pixel_val += i << 5;
//                    heap_cur += pixel_val - ((*imgptr) & 0xff);
//                    break;
//                }
//                bit_cur++;
//            }
//            if (found_pixel)
//#else
//                // 从当前灰度后逐步提高灰度值，在边界堆中找到一个边界像素
//            heap_cur++;
//            unsigned long pixel_val = 0;
//            for (unsigned long i = ((*imgptr) & 0xff) + 1; i < 256; i++)
//            {
//                if (**heap_cur)
//                {
//                    // 不为零，指针指向了一个像素，这个灰度值还有边界
//                    pixel_val = i;
//                    break;
//                }
//
//                // 提高灰度值
//                heap_cur++;
//            }
//
//            // 判断边界中是否还有像素
//            if (pixel_val)
//#endif
//            {
//                // 将边界中的像素作为当前像素，并从边界中去除
//                imgptr = **heap_cur;
//                (*heap_cur)--;
//#ifdef __INTRIN_ENABLED__
//                if (!**heap_cur)
//                    _bitreset(bit_cur, pixel_val & 0x1f);
//#endif
//                // comptr[-1] == comptr--
//                if (pixel_val < comptr[-1].grey_level)
//                {
//                    // 刚从边界获得灰度如果小于上一个MSER组件灰度值，需要提高当前水位到边界的灰度值
//                    // check the stablity and push a new history, increase the grey level
//                    if (MSERStableCheck(comptr, params))
//                    {
//                        CvContour* contour = MSERToContour(comptr, storage);
//                        contour->color = color;
//                        cvSeqPush(contours, &contour);
//                    }
//
//                    // 由于水位要有变化了，添加一个历史
//                    MSERNewHistory(comptr, histptr);
//
//                    // 提高水位到边界的水位
//                    comptr[0].grey_level = pixel_val;
//
//                    // 指向下一个未使用历史空间
//                    histptr++;
//                }
//                else {
//                    // 刚从边界获得灰度如果不小于上一个MSER组件灰度值，其实就是和上一个灰度值一样。
//                    // 例如：当前水位2，上一个水位3，从边界出栈的水位为3.
//
//                    // keep merging top two comp in stack until the grey level >= pixel_val
//                    for (;;)
//                    {
//                        // 合并MSER组件，里面也随带完成了一个历史
//                        comptr--;
//                        MSERMergeComp(comptr + 1, comptr, comptr, histptr);
//                        histptr++;
//
//                        if (pixel_val <= comptr[0].grey_level)
//                            break;
//
//                        // 到这里，等于comptr[0].grey_level < pixel_val，也是当前像素的灰度与MSER组件的不一致，要提高MSER组件灰度
//                        if (pixel_val < comptr[-1].grey_level)
//                        {
//                            // 其实就是comptr[0].grey_level < pixel_val < comptr[-1].grey_level
//                            // 当前灰度大于当前MSER灰度小于上一个MSER组件灰度。同上面的代码情况一样。
//                            // check the stablity here otherwise it wouldn't be an ER
//                            if (MSERStableCheck(comptr, params))
//                            {
//                                CvContour* contour = MSERToContour(comptr, storage);
//                                contour->color = color;
//                                cvSeqPush(contours, &contour);
//                            }
//                            MSERNewHistory(comptr, histptr);
//                            comptr[0].grey_level = pixel_val;
//                            histptr++;
//                            break;
//                        }
//                    }
//                }
//            }
//            else
//                break;
//        }
//    }
//}
//
///** @brief 通过8UC1类型的图像提取MSER
//*@param mask 掩码
//*@param contours 轮廓结果
//*@param storage 轮廓内存空间
//*@param params 参数
//*/
//static void extractMSER_8UC1(CvMat* src,
//    CvMat* mask,
//    CvSeq* contours,
//    CvMemStorage* storage,
//    MSERParams params)
//{
//    // 为了加速计算，将每行数据大小扩展为大于原大小的第一个2的整指数。
//    // 这样在后面计算y时，只要右移stepgap就算除以2^stepgap了
//    int step = 8;
//    int stepgap = 3;
//    while (step < src->step + 2)
//    {
//        step <<= 1;
//        stepgap++;
//    }
//    int stepmask = step - 1;
//
//    // to speedup the process, make the width to be 2^N
//    CvMat* img = cvCreateMat(src->rows + 2, step, CV_32SC1);
//    int* ioptr = img->data.i + step + 1;        // 数据在扩展后的最开始位置
//    int* imgptr;                                        // 用于指向mser遍历的当前像素（所有数据）
//
//    // pre-allocate boundary heap
//    // 预分配边界堆和每个灰度指向堆的指针数组
//    // 堆大小就是像素数+所有灰度值（一个标志数据，用来表明这个灰度没有数据了）
//    int** heap = (int**)cvAlloc((src->rows * src->cols + 256) * sizeof(heap[0]));
//    int** heap_start[256];
//    heap_start[0] = heap;
//
//    // pre-allocate linked point and grow history
//    // 预分配连接像素点，用于将区域中的像素连接起来，大小就为所有像素个数
//    LinkedPoint* pts = (LinkedPoint*)cvAlloc(src->rows * src->cols * sizeof(pts[0]));
//    // 预分配增长历史，用于记录区域在太高水位后的父子关系，最大个数为所有像素个数。
//    MSERGrowHistory* history = (MSERGrowHistory*)cvAlloc(src->rows * src->cols * sizeof(history[0]));
//    // 预分配区域，用于记录每个区域的数据，大小为所有灰度值+1个超大灰度值代表顶
//    MSERConnectedComp comp[257];
//
//    // darker to brighter (MSER-)
//    // 提取mser亮区域（preprocessMSER_8UC1中将灰度值取反）
//    imgptr = preprocessMSER_8UC1(img, heap_start, src, mask);
//    extractMSER_8UC1_Pass(ioptr, imgptr, heap_start, pts, history, comp, step, stepmask, stepgap, params, -1, contours, storage);
//    // brighter to darker (MSER+)
//    // 提取mser暗区域
//    imgptr = preprocessMSER_8UC1(img, heap_start, src, mask);
//    extractMSER_8UC1_Pass(ioptr, imgptr, heap_start, pts, history, comp, step, stepmask, stepgap, params, 1, contours, storage);
//
//    // clean up
//    cvFree(&history);
//    cvFree(&heap);
//    cvFree(&pts);
//    cvReleaseMat(&img);
//}
//
//// 条件判断和生成结果
//// clear the connected component in stack
//static void
//initMSERComp(MSERConnectedComp* comp)
//{
//    comp->size = 0;
//    comp->var = 0;
//    comp->dvar = 1;
//    comp->history = NULL;
//}
//
//// add history of size to a connected component
//static void
///** @brief 通过当前ER项构建一个对应的历史，也就是说找个ER项要准备改变了
//*/
//MSERNewHistory(MSERConnectedComp* comp, MSERGrowHistory* history)
//{
//    // 初始时将下一条历史设置为自己
//    history->child = history;
//    if (NULL == comp->history)
//    {
//        // 从来没有历史过，将快捷路径也设置为自己，稳定的像素数为0
//        history->shortcut = history;
//        history->stable = 0;
//    }
//    else {
//        // 有历史，将当前历史设置为上一个历史的下个历史
//        comp->history->child = history;
//
//        // 快捷路径与稳定值继承至上一个历史
//        history->shortcut = comp->history->shortcut;
//        history->stable = comp->history->stable;
//    }
//
//    // 记录这时的ER项的灰度值与像素数
//    history->val = comp->grey_level;
//    history->size = comp->size;
//
//    // 设置ER项的历史为找个最新的历史
//    comp->history = history;
//}
//
//// merging two connected component
//static void
//MSERMergeComp(MSERConnectedComp* comp1,
//    MSERConnectedComp* comp2,
//    MSERConnectedComp* comp,
//    MSERGrowHistory* history)
//{
//    LinkedPoint* head;
//    LinkedPoint* tail;
//    comp->grey_level = comp2->grey_level;
//    history->child = history;
//    // select the winner by size
//    if (comp1->size >= comp2->size)
//    {
//        if (NULL == comp1->history)
//        {
//            history->shortcut = history;
//            history->stable = 0;
//        }
//        else {
//            comp1->history->child = history;
//            history->shortcut = comp1->history->shortcut;
//            history->stable = comp1->history->stable;
//        }
//
//        // 如果组件2有stable，并且大于1的，则stable使用2的值
//        if (NULL != comp2->history && comp2->history->stable > history->stable)
//            history->stable = comp2->history->stable;
//
//        // 使用数量多的
//        history->val = comp1->grey_level;
//        history->size = comp1->size;
//        // put comp1 to history
//        comp->var = comp1->var;
//        comp->dvar = comp1->dvar;
//
//        // 如果组件1和2都有像素点，将两个链按照1->2连接在一起
//        if (comp1->size > 0 && comp2->size > 0)
//        {
//            comp1->tail->next = comp2->head;
//            comp2->head->prev = comp1->tail;
//        }
//
//        // 确定头尾
//        head = (comp1->size > 0) ? comp1->head : comp2->head;
//        tail = (comp2->size > 0) ? comp2->tail : comp1->tail;
//        // always made the newly added in the last of the pixel list (comp1 ... comp2)
//    }
//    else {
//        // 与上面的正好相反
//        if (NULL == comp2->history)
//        {
//            history->shortcut = history;
//            history->stable = 0;
//        }
//        else {
//            comp2->history->child = history;
//            history->shortcut = comp2->history->shortcut;
//            history->stable = comp2->history->stable;
//        }
//        if (NULL != comp1->history && comp1->history->stable > history->stable)
//            history->stable = comp1->history->stable;
//        history->val = comp2->grey_level;
//        history->size = comp2->size;
//        // put comp2 to history
//        comp->var = comp2->var;
//        comp->dvar = comp2->dvar;
//        if (comp1->size > 0 && comp2->size > 0)
//        {
//            comp2->tail->next = comp1->head;
//            comp1->head->prev = comp2->tail;
//        }
//
//        head = (comp2->size > 0) ? comp2->head : comp1->head;
//        tail = (comp1->size > 0) ? comp1->tail : comp2->tail;
//        // always made the newly added in the last of the pixel list (comp2 ... comp1)
//    }
//    comp->head = head;
//    comp->tail = tail;
//    comp->history = history;
//
//    // 新ER的像素数量是两个ER项的和
//    comp->size = comp1->size + comp2->size;
//}
//
///** @brief 通过delta计算给定ER项的偏差
//*/
//static float MSERVariationCalc(MSERConnectedComp* comp, int delta)
//{
//    MSERGrowHistory* history = comp->history;
//    int val = comp->grey_level;
//    if (NULL != history)
//    {
//        // 从快捷路径开始往回找历史，找到灰度差大于delta的历史
//        MSERGrowHistory* shortcut = history->shortcut;
//        while (shortcut != shortcut->shortcut && shortcut->val + delta > val)
//            shortcut = shortcut->shortcut;
//
//        // 由于快捷路径是直接跳过一些历史的，要找到最准确的历史还要从以前历史往当前找
//        MSERGrowHistory* child = shortcut->child;
//        while (child != child->child && child->val + delta <= val)
//        {
//            shortcut = child;
//            child = child->child;
//        }
//        // get the position of history where the shortcut->val <= delta+val and shortcut->child->val >= delta+val
//        // 更新快捷路径
//        history->shortcut = shortcut;
//
//        // 返回(R-R(-delta)) / (R-delta)
//        return (float)(comp->size - shortcut->size) / (float)shortcut->size;
//        // here is a small modification of MSER where cal ||R_{i}-R_{i-delta}||/||R_{i-delta}||
//        // in standard MSER, cal ||R_{i+delta}-R_{i-delta}||/||R_{i}||
//        // my calculation is simpler and much easier to implement
//    }
//
//    // 没有历史，结果为1。也就是没有-delta对应的值。
//    // 如果按照(R-R(-delta)) / R(-delta) = 1公式推导:
//    // R = 2R(-delta)
//    // 就面积来说，怎么两倍这种关系都比较奇怪，因为是xy两个维度的，每个维度提高sqrt(2)倍
//    return 1.;
//}
//
///** @brief 检查是否为最稳定极值区域
//*/
//static bool MSERStableCheck(MSERConnectedComp* comp, MSERParams params)
//{
//    // 检查就是要确定水位的底是否是稳定的
//    // tricky part: it actually check the stablity of one-step back
//    // 稳定区域都是由比较而来的，不能没有上一个历史。
//    if (comp->history == NULL || comp->history->size <= params.minArea || comp->history->size >= params.maxArea)
//        return 0;
//
//    // diversity : (R(-1) - R(stable)) / R(-1)
//    // 使用水位的底与稳定时大小做比较
//    float div = (float)(comp->history->size - comp->history->stable) / (float)comp->history->size;
//
//    // variation
//    float var = MSERVariationCalc(comp, params.delta);
//
//    // 现在的variation要大于以前的variation，就是以前的更稳定
//    // 灰度值差是否大于1
//    int dvar = (comp->var < var || (unsigned long)(comp->history->val + 1) < comp->grey_level);
//    int stable = (dvar && !comp->dvar && comp->var < params.maxVariation&& div > params.minDiversity);
//    comp->var = var;
//    comp->dvar = dvar;
//    if (stable)
//        // 如果稳定的话，稳定值就是像素数
//        comp->history->stable = comp->history->size;
//    return stable != 0;
//}
//
//// add a pixel to the pixel list
///** @brief 添加像素到给定的MSER项中
//*/
//static void accumulateMSERComp(MSERConnectedComp* comp, LinkedPoint* point)
//{
//    if (comp->size > 0)
//    {
//        // 之前有像素，连接到原来像素的链上
//        point->prev = comp->tail;
//        comp->tail->next = point;
//        point->next = NULL;
//    }
//    else {
//        // 第一个像素
//        point->prev = NULL;
//        point->next = NULL;
//        comp->head = point;
//    }
//
//    // 新加入的点作为尾巴
//    comp->tail = point;
//
//    // 像素数自增
//    comp->size++;
//}
//
//// convert the point set to CvSeq
//static CvContour* MSERToContour(MSERConnectedComp* comp, CvMemStorage* storage)
//{
//    CvSeq* _contour = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage);
//    CvContour* contour = (CvContour*)_contour;
//
//    // 上次历史就是水位的底，将水位的底都添加到轮廓中
//    cvSeqPushMulti(_contour, 0, comp->history->size);
//    LinkedPoint* lpt = comp->head;
//    for (int i = 0; i < comp->history->size; i++)
//    {
//        CvPoint* pt = CV_GET_SEQ_ELEM(CvPoint, _contour, i);
//        pt->x = lpt->pt.x;
//        pt->y = lpt->pt.y;
//        lpt = lpt->next;
//    }
//    cvBoundingRect(contour);
//    return contour;
//}
