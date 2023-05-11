//#include"MSERTest.h"
//
//using namespace std;
//using namespace cv;
//// �Լ�д��
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
//// ���ϳ���Opencv 2.4.9
//// CvMat
//typedef struct CvMat
//{
//    int type;           // ��������
//    int step;           // ���ֽڱ�ʾ�����ݳ���
//    int* refcount;      // �ڲ�����
//    int hdr_refcount;   // �ڲ�ʹ��
//    union {             // ָ����������ָ��
//        uchar* ptr;
//        short* s;
//        int* i;
//        float* fl;
//        double* db;
//    } data;    
//    union {             // ����
//        int rows;
//        int height;
//    };
//    union {             // ����
//        int cols;
//        int width;
//    };
//} CvMat; // ����ṹͷ
//// �����ṹ
//// �ڵ�ָ������
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
//    // ���·������ָ����ǰ��ʷ��ָ�롣��Ϊ����һ��һ�����ӵģ����Բ���parent���㷨���Ǽ�¼�ҶȲ�Ϊdelta����ʷ��ָ�롣
//    // ���磺��ǰ�ǻҶ���10��delta=3�����ָ���ָ��Ҷ�Ϊ7ʱ�����ʷ
//    struct MSERGrowHistory* shortcut;
//    // ָ�������ʷ��ָ�룬���Ǵ������ʷ���ܵ�����ʷ�����Խк���
//    struct MSERGrowHistory* child;
//    // ����������ȶ���ֵ���ȶ��ǵ������������ֵ�ڲ�ͣ�ļ̳�
//    int stable; // when it ever stabled before, record the size
//    // �Ҷ�ֵ
//    int val;
//    // ������
//    int size;
//}
//MSERGrowHistory;
//
//typedef struct MSERConnectedComp
//{
//    // ���ص�����ͷ
//    LinkedPoint* head;
//    // ���ص�����β
//    LinkedPoint* tail;
//    // �����ϴε�������ʷ������ͨ���Ҹ���ʷ�ҵ�֮ǰ�ļ�¼
//    MSERGrowHistory* history;
//    // �Ҷ�ֵ
//    unsigned long grey_level;
//    // ������
//    int size;
//    int dvar; // the derivative of last var
//    float var; // the current variation (most time is the variation of one-step back)
//}
//MSERConnectedComp;
//
//// c++�ṹ��ָ�룬����˼�����ָ��ṹ���һ��ָ�룬��ƪ���������Ǽ�¼c++�ṹ��ָ��ĳ����÷����Ҿ�������һ������
//// ����ṹ�壺
////struct My {
////    My* left;
////    My* right;
////    int val;
////    My() {}
////    My(int val) :left(NULL), right(NULL), val(val) {}
////};
//// һ��ṹ������ķ��ʷ�ʽ��
////void test1() {
////    My m;
////    m.val = 1;
////    cout << m.val << endl;
////}
//// �ɼ����ṹ���еı���������ֱ��ͨ��������������ʡ�
//// �����ڽṹ��ָ����ԣ�����ͨ��->����������ָ����ָ�ṹ��ı�����
////void test2() {
////    My m;
////    m.val = 1;
////    My* mm;
////    mm = &m;
////    cout << mm->val << endl;
////}
//// ����һ���ṹ��ָ��ǵó�ʼ����һ��Ҫ��ʼ��������ʼ������£���Ҫ������˵���飩
//// ���£�
////void test3() {
////    My* m;
////    m->val = 1;
////}
//// ��ݴ���ᱨһ������ָ������쳣��������Ϊm���ָ�뻹û�г�ʼ���������û���ڴ�ռ䣬��Ȼ�Ͳ�������val�����������ȷ�򿪷�ʽ��
////void test3() {
////    My* m;
////    m = new My(3);
////    m->val = 4;
////    cout << m->val << endl;
////}
//// ���ϴ�����new�������ڴ�ռ䡣���⼴�ɽ����
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
//    // MSERʹ��
//    int delta;                   // ���������ĻҶȲ�
//    int minArea;                 // ������С������
//    int maxArea;                 // �������������
//    double maxVariation;         // ���������ƫ��
//    double minDiversity;         // ��ǰ�������ȶ�����ı仯��
//    // MSCRʹ��
//    int maxEvolution;
//    double areaThreshold;
//    double minMargin;
//    int edgeBlurSize;
//};
//
//// ����Ԥ����
//// to preprocess src image to following format
//    // 32-bit image
//    // > 0 is available, < 0 is visited
//    // 17~19 bits is the direction
//    // 8~11 bits is the bucket it falls to (for BitScanForward)
//    // 0~8 bits is the color
//    /** @brief ������ԭ��ͨ���Ҷ�ͼ������ͼ Ԥ����Ϊһ�ŷ���������¼���ݵ�32λ��ͨ��ͼ�񣻲��Ҹ������ػҶ�ֵ�����Եջ��
//    * x64��С�˻�����λ�ֽ��ڵ�λ��ַ
//    * 32λ��ʽ���£�
//    * > 0 ���ã�< 0 �Ѿ�������
//    * 16~18λ���ڼ�¼��һ��Ҫ̽���ķ���5��ֵ
//    * 8~11λ �����Ż��Ķ�ֵ����
//    * 0~7λ���ڼ�¼�Ҷ�ֵ
//    *@param heap_cur ��Եջ
//    *@param src ԭ��ͨ���Ҷ�ͼ
//    *@param mask ����ͼ
//    */
//static int* preprocessMSER_8UC1(CvMat* img, int*** heap_cur, CvMat* src, CvMat* mask)
//{
//    // ������Ч��������img�У���һȦ-1��Χ�ţ����������Ҳ���Ǳ�һȦ-1��ǽ��Χ�š�
//
//    // ԭʼ������ת����һ�е�ƫ������
//    int srccpt = src->step - src->cols;
//
//    // ��ת����һ�е�ƫ����������һ����Ϊ�����磺xoooxxx��o����Ч���ݣ�x����������ġ�ƫ����Ӧ����3������ooo��
//    // �ұߵ�xxx������Ϊ�˼��㣬����Ҫ��ȥooo�������һ��x��
//    int cpt_1 = img->cols - src->cols - 1;
//    int* imgptr = img->data.i;
//    int* startptr;
//
//    // ���ڼ�¼ÿ���Ҷ��ж�������
//    int level_size[256];
//    for (int i = 0; i < 256; i++)
//        level_size[i] = 0;
//
//    // ���õ�һ��Ϊ-1
//    for (int i = 0; i < src->cols + 2; i++)
//    {
//        *imgptr = -1;
//        imgptr++;
//    }
//
//    // ƫ�Ƶ���һ����Ч���������еĿ�ͷ
//    imgptr += cpt_1 - 1;
//    uchar* srcptr = src->data.ptr;
//    if (mask)
//    {
//        // ������
//        startptr = 0;            // ���ݴ���Ŀ�ʼλ�ã�Ϊ�����ϵ�λ�á�
//        uchar* maskptr = mask->data.ptr;
//        for (int i = 0; i < src->rows; i++)
//        {
//            // ����������Ϊ-1
//            *imgptr = -1;
//            imgptr++;
//            for (int j = 0; j < src->cols; j++)
//            {
//                if (*maskptr)
//                {
//                    if (!startptr)
//                        startptr = imgptr;
//
//                    // �Ҷ�ֵȡ��!!!!! !!!!! !!!!! !!!!!
//                    *srcptr = 0xff - *srcptr;
//
//                    // ���ڻҶ�ֵ��������
//                    level_size[*srcptr]++;
//
//                    // д��0~8λ��8~13λ����BitScanForward
//                    *imgptr = ((*srcptr >> 5) << 8) | (*srcptr);
//                }
//                else {
//                    // ��Ϊ-1�����ǵ���һ���Ѿ������ֵ�λ�ã�����Χ-1ǽ��ԭ��һ��
//                    *imgptr = -1;
//                }
//                imgptr++;
//                srcptr++;
//                maskptr++;
//            }
//
//            // ����������Ϊ-1
//            *imgptr = -1;
//
//            // ��������һ�п�ʼ
//            imgptr += cpt_1;
//            srcptr += srccpt;
//            maskptr += srccpt;
//        }
//    }
//    else {
//        // ����û����������
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
//    // �������һ��Ϊ-1
//    for (int i = 0; i < src->cols + 2; i++)
//    {
//        *imgptr = -1;
//        imgptr++;
//    }
//
//    // ȷ��ÿ���Ҷ��ڱ߽���е�ָ��λ�á�0����û��ֵ��
//    heap_cur[0][0] = 0;
//    for (int i = 1; i < 256; i++)
//    {
//        heap_cur[i] = heap_cur[i - 1] + level_size[i - 1] + 1; // ����֣�ΪʲôҪ���ۼӣ���
//        heap_cur[i][0] = 0;
//    }
//    return startptr;
//}
//
//// �����̼���������
//static void extractMSER_8UC1_Pass(int* ioptr,
//    int* imgptr,
//    int*** heap_cur,                            // �߽�ջ�Ķѣ�������ÿһ���Ҷȵ�ջ
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
//    // ERջ��һ��Ϊ�����ı�ʶ�ֵΪ����255��256
//    comptr->grey_level = 256;
//
//    // ����ǰλ��ֵ��ջ������ʼ��
//    comptr++;
//    comptr->grey_level = (*imgptr) & 0xff;
//    initMSERComp(comptr);
//
//    // ����Ϊ�Ѿ�����
//    *imgptr |= 0x80000000;
//
//    // ���ϻҶ�ƫ�ƾͽ�ָ�붨λ������Ӧ�Ҷȵı߽�ջ��
//    heap_cur += (*imgptr) & 0xff;
//
//    // �ĸ������ƫ���������µ�ƫ���Ǹ��еĲ���
//    int dir[] = { 1, step, -1, -step };
//#ifdef __INTRIN_ENABLED__
//    unsigned long heapbit[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
//    unsigned long* bit_cur = heapbit + (((*imgptr) & 0x700) >> 8);
//#endif
//
//    // ѭ��
//    for (;;)
//    {
//        // take tour of all the 4 directions
//        // ��ȡ��ǰ���صķ���ֵ���ж��Ƿ��з���û���߹�
//        while (((*imgptr) & 0x70000) < 0x40000)
//        {
//            // get the neighbor
//            // ͨ�������Ӧ��ƫ�ƻ����������ָ��
//            int* imgptr_nbr = imgptr + dir[((*imgptr) & 0x70000) >> 16];
//
//            // �ж��Ƿ���ʹ�
//            if (*imgptr_nbr >= 0) // if the neighbor is not visited yet
//            {
//                // û�з��ʹ������Ϊ���ʹ�
//                *imgptr_nbr |= 0x80000000; // mark it as visited
//                if (((*imgptr_nbr) & 0xff) < ((*imgptr) & 0xff))
//                {
//                    // when the value of neighbor smaller than current
//                    // push current to boundary heap and make the neighbor to be the current one
//                    // create an empty comp
//                    // ����������صĻҶ�С�ڵ�ǰ���أ�����ǰ���ؼ���߽�ջ�ѣ�����������������Ϊ��ǰ���أ����½�ERջ��
//                    // ����ǰ����߽�ջ��
//                    (*heap_cur)++;
//                    **heap_cur = imgptr;
//
//                    // ת������
//                    *imgptr += 0x10000;
//
//                    // ���߽�ջ�ѵ�ָ�����Ϊ���ڵ����ػҶ�����Ӧ��λ��
//                    heap_cur += ((*imgptr_nbr) & 0xff) - ((*imgptr) & 0xff);
//#ifdef __INTRIN_ENABLED__
//                    _bitset(bit_cur, (*imgptr) & 0x1f);
//                    bit_cur += (((*imgptr_nbr) & 0x700) - ((*imgptr) & 0x700)) >> 8;
//#endif
//                    // ��������������Ϊ��ǰ����
//                    imgptr = imgptr_nbr;
//
//                    // �½�ERջ������ûҶ�Ϊ��ǰ���ػҶ�
//                    comptr++;
//                    initMSERComp(comptr);
//                    comptr->grey_level = (*imgptr) & 0xff;
//                    continue;
//                }
//                else {
//                    // otherwise, push the neighbor to boundary heap
//                    // ���򣬽�����������ӵ���Ӧ�ı߽�֡����
//                    heap_cur[((*imgptr_nbr) & 0xff) - ((*imgptr) & 0xff)]++;
//                    *heap_cur[((*imgptr_nbr) & 0xff) - ((*imgptr) & 0xff)] = imgptr_nbr;
//#ifdef __INTRIN_ENABLED__
//                    _bitset(bit_cur + ((((*imgptr_nbr) & 0x700) - ((*imgptr) & 0x700)) >> 8), (*imgptr_nbr) & 0x1f);
//#endif
//                }
//            }
//
//            // ����ǰ���صķ���ת������һ������
//            *imgptr += 0x10000;
//        }
//
//        int imsk = (int)(imgptr - ioptr);
//
//        // ��¼x&y��
//        ptsptr->pt = cvPoint(imsk & stepmask, imsk >> stepgap);
//        // get the current location
//        accumulateMSERComp(comptr, ptsptr);
//        ptsptr++;
//        // get the next pixel from boundary heap
//        // �ӱ߽�ջ���л�ȡһ������������ǰ����
//        if (**heap_cur)
//        {
//            // ��ǰ�Ҷȵı߽�ջ����ֵ�����ã�����ǰ�߽�ջ��ֵ����Ϊ��ǰ���أ���Ϊ��ǰ�߽�ջ�ѵĻҶȾ��ǵ�ǰ���صĻҶȣ����Կ���ֱ���ó�����
//            imgptr = **heap_cur;
//
//            // ��ջ
//            (*heap_cur)--;
//#ifdef __INTRIN_ENABLED__
//            if (!**heap_cur)
//                _bitreset(bit_cur, (*imgptr) & 0x1f);
//#endif
//        }
//        else {
//            // ��ǰ�Ҷȱ߽�ջ����û��ֵ������
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
//                // �ӵ�ǰ�ҶȺ�����߻Ҷ�ֵ���ڱ߽�����ҵ�һ���߽�����
//            heap_cur++;
//            unsigned long pixel_val = 0;
//            for (unsigned long i = ((*imgptr) & 0xff) + 1; i < 256; i++)
//            {
//                if (**heap_cur)
//                {
//                    // ��Ϊ�㣬ָ��ָ����һ�����أ�����Ҷ�ֵ���б߽�
//                    pixel_val = i;
//                    break;
//                }
//
//                // ��߻Ҷ�ֵ
//                heap_cur++;
//            }
//
//            // �жϱ߽����Ƿ�������
//            if (pixel_val)
//#endif
//            {
//                // ���߽��е�������Ϊ��ǰ���أ����ӱ߽���ȥ��
//                imgptr = **heap_cur;
//                (*heap_cur)--;
//#ifdef __INTRIN_ENABLED__
//                if (!**heap_cur)
//                    _bitreset(bit_cur, pixel_val & 0x1f);
//#endif
//                // comptr[-1] == comptr--
//                if (pixel_val < comptr[-1].grey_level)
//                {
//                    // �մӱ߽��ûҶ����С����һ��MSER����Ҷ�ֵ����Ҫ��ߵ�ǰˮλ���߽�ĻҶ�ֵ
//                    // check the stablity and push a new history, increase the grey level
//                    if (MSERStableCheck(comptr, params))
//                    {
//                        CvContour* contour = MSERToContour(comptr, storage);
//                        contour->color = color;
//                        cvSeqPush(contours, &contour);
//                    }
//
//                    // ����ˮλҪ�б仯�ˣ����һ����ʷ
//                    MSERNewHistory(comptr, histptr);
//
//                    // ���ˮλ���߽��ˮλ
//                    comptr[0].grey_level = pixel_val;
//
//                    // ָ����һ��δʹ����ʷ�ռ�
//                    histptr++;
//                }
//                else {
//                    // �մӱ߽��ûҶ������С����һ��MSER����Ҷ�ֵ����ʵ���Ǻ���һ���Ҷ�ֵһ����
//                    // ���磺��ǰˮλ2����һ��ˮλ3���ӱ߽��ջ��ˮλΪ3.
//
//                    // keep merging top two comp in stack until the grey level >= pixel_val
//                    for (;;)
//                    {
//                        // �ϲ�MSER���������Ҳ��������һ����ʷ
//                        comptr--;
//                        MSERMergeComp(comptr + 1, comptr, comptr, histptr);
//                        histptr++;
//
//                        if (pixel_val <= comptr[0].grey_level)
//                            break;
//
//                        // ���������comptr[0].grey_level < pixel_val��Ҳ�ǵ�ǰ���صĻҶ���MSER����Ĳ�һ�£�Ҫ���MSER����Ҷ�
//                        if (pixel_val < comptr[-1].grey_level)
//                        {
//                            // ��ʵ����comptr[0].grey_level < pixel_val < comptr[-1].grey_level
//                            // ��ǰ�Ҷȴ��ڵ�ǰMSER�Ҷ�С����һ��MSER����Ҷȡ�ͬ����Ĵ������һ����
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
///** @brief ͨ��8UC1���͵�ͼ����ȡMSER
//*@param mask ����
//*@param contours �������
//*@param storage �����ڴ�ռ�
//*@param params ����
//*/
//static void extractMSER_8UC1(CvMat* src,
//    CvMat* mask,
//    CvSeq* contours,
//    CvMemStorage* storage,
//    MSERParams params)
//{
//    // Ϊ�˼��ټ��㣬��ÿ�����ݴ�С��չΪ����ԭ��С�ĵ�һ��2����ָ����
//    // �����ں������yʱ��ֻҪ����stepgap�������2^stepgap��
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
//    int* ioptr = img->data.i + step + 1;        // ��������չ����ʼλ��
//    int* imgptr;                                        // ����ָ��mser�����ĵ�ǰ���أ��������ݣ�
//
//    // pre-allocate boundary heap
//    // Ԥ����߽�Ѻ�ÿ���Ҷ�ָ��ѵ�ָ������
//    // �Ѵ�С����������+���лҶ�ֵ��һ����־���ݣ�������������Ҷ�û�������ˣ�
//    int** heap = (int**)cvAlloc((src->rows * src->cols + 256) * sizeof(heap[0]));
//    int** heap_start[256];
//    heap_start[0] = heap;
//
//    // pre-allocate linked point and grow history
//    // Ԥ�����������ص㣬���ڽ������е�����������������С��Ϊ�������ظ���
//    LinkedPoint* pts = (LinkedPoint*)cvAlloc(src->rows * src->cols * sizeof(pts[0]));
//    // Ԥ����������ʷ�����ڼ�¼������̫��ˮλ��ĸ��ӹ�ϵ��������Ϊ�������ظ�����
//    MSERGrowHistory* history = (MSERGrowHistory*)cvAlloc(src->rows * src->cols * sizeof(history[0]));
//    // Ԥ�����������ڼ�¼ÿ����������ݣ���СΪ���лҶ�ֵ+1������Ҷ�ֵ����
//    MSERConnectedComp comp[257];
//
//    // darker to brighter (MSER-)
//    // ��ȡmser������preprocessMSER_8UC1�н��Ҷ�ֵȡ����
//    imgptr = preprocessMSER_8UC1(img, heap_start, src, mask);
//    extractMSER_8UC1_Pass(ioptr, imgptr, heap_start, pts, history, comp, step, stepmask, stepgap, params, -1, contours, storage);
//    // brighter to darker (MSER+)
//    // ��ȡmser������
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
//// �����жϺ����ɽ��
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
///** @brief ͨ����ǰER���һ����Ӧ����ʷ��Ҳ����˵�Ҹ�ER��Ҫ׼���ı���
//*/
//MSERNewHistory(MSERConnectedComp* comp, MSERGrowHistory* history)
//{
//    // ��ʼʱ����һ����ʷ����Ϊ�Լ�
//    history->child = history;
//    if (NULL == comp->history)
//    {
//        // ����û����ʷ���������·��Ҳ����Ϊ�Լ����ȶ���������Ϊ0
//        history->shortcut = history;
//        history->stable = 0;
//    }
//    else {
//        // ����ʷ������ǰ��ʷ����Ϊ��һ����ʷ���¸���ʷ
//        comp->history->child = history;
//
//        // ���·�����ȶ�ֵ�̳�����һ����ʷ
//        history->shortcut = comp->history->shortcut;
//        history->stable = comp->history->stable;
//    }
//
//    // ��¼��ʱ��ER��ĻҶ�ֵ��������
//    history->val = comp->grey_level;
//    history->size = comp->size;
//
//    // ����ER�����ʷΪ�Ҹ����µ���ʷ
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
//        // ������2��stable�����Ҵ���1�ģ���stableʹ��2��ֵ
//        if (NULL != comp2->history && comp2->history->stable > history->stable)
//            history->stable = comp2->history->stable;
//
//        // ʹ���������
//        history->val = comp1->grey_level;
//        history->size = comp1->size;
//        // put comp1 to history
//        comp->var = comp1->var;
//        comp->dvar = comp1->dvar;
//
//        // ������1��2�������ص㣬������������1->2������һ��
//        if (comp1->size > 0 && comp2->size > 0)
//        {
//            comp1->tail->next = comp2->head;
//            comp2->head->prev = comp1->tail;
//        }
//
//        // ȷ��ͷβ
//        head = (comp1->size > 0) ? comp1->head : comp2->head;
//        tail = (comp2->size > 0) ? comp2->tail : comp1->tail;
//        // always made the newly added in the last of the pixel list (comp1 ... comp2)
//    }
//    else {
//        // ������������෴
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
//    // ��ER����������������ER��ĺ�
//    comp->size = comp1->size + comp2->size;
//}
//
///** @brief ͨ��delta�������ER���ƫ��
//*/
//static float MSERVariationCalc(MSERConnectedComp* comp, int delta)
//{
//    MSERGrowHistory* history = comp->history;
//    int val = comp->grey_level;
//    if (NULL != history)
//    {
//        // �ӿ��·����ʼ��������ʷ���ҵ��ҶȲ����delta����ʷ
//        MSERGrowHistory* shortcut = history->shortcut;
//        while (shortcut != shortcut->shortcut && shortcut->val + delta > val)
//            shortcut = shortcut->shortcut;
//
//        // ���ڿ��·����ֱ������һЩ��ʷ�ģ�Ҫ�ҵ���׼ȷ����ʷ��Ҫ����ǰ��ʷ����ǰ��
//        MSERGrowHistory* child = shortcut->child;
//        while (child != child->child && child->val + delta <= val)
//        {
//            shortcut = child;
//            child = child->child;
//        }
//        // get the position of history where the shortcut->val <= delta+val and shortcut->child->val >= delta+val
//        // ���¿��·��
//        history->shortcut = shortcut;
//
//        // ����(R-R(-delta)) / (R-delta)
//        return (float)(comp->size - shortcut->size) / (float)shortcut->size;
//        // here is a small modification of MSER where cal ||R_{i}-R_{i-delta}||/||R_{i-delta}||
//        // in standard MSER, cal ||R_{i+delta}-R_{i-delta}||/||R_{i}||
//        // my calculation is simpler and much easier to implement
//    }
//
//    // û����ʷ�����Ϊ1��Ҳ����û��-delta��Ӧ��ֵ��
//    // �������(R-R(-delta)) / R(-delta) = 1��ʽ�Ƶ�:
//    // R = 2R(-delta)
//    // �������˵����ô�������ֹ�ϵ���Ƚ���֣���Ϊ��xy����ά�ȵģ�ÿ��ά�����sqrt(2)��
//    return 1.;
//}
//
///** @brief ����Ƿ�Ϊ���ȶ���ֵ����
//*/
//static bool MSERStableCheck(MSERConnectedComp* comp, MSERParams params)
//{
//    // ������Ҫȷ��ˮλ�ĵ��Ƿ����ȶ���
//    // tricky part: it actually check the stablity of one-step back
//    // �ȶ��������ɱȽ϶����ģ�����û����һ����ʷ��
//    if (comp->history == NULL || comp->history->size <= params.minArea || comp->history->size >= params.maxArea)
//        return 0;
//
//    // diversity : (R(-1) - R(stable)) / R(-1)
//    // ʹ��ˮλ�ĵ����ȶ�ʱ��С���Ƚ�
//    float div = (float)(comp->history->size - comp->history->stable) / (float)comp->history->size;
//
//    // variation
//    float var = MSERVariationCalc(comp, params.delta);
//
//    // ���ڵ�variationҪ������ǰ��variation��������ǰ�ĸ��ȶ�
//    // �Ҷ�ֵ���Ƿ����1
//    int dvar = (comp->var < var || (unsigned long)(comp->history->val + 1) < comp->grey_level);
//    int stable = (dvar && !comp->dvar && comp->var < params.maxVariation&& div > params.minDiversity);
//    comp->var = var;
//    comp->dvar = dvar;
//    if (stable)
//        // ����ȶ��Ļ����ȶ�ֵ����������
//        comp->history->stable = comp->history->size;
//    return stable != 0;
//}
//
//// add a pixel to the pixel list
///** @brief ������ص�������MSER����
//*/
//static void accumulateMSERComp(MSERConnectedComp* comp, LinkedPoint* point)
//{
//    if (comp->size > 0)
//    {
//        // ֮ǰ�����أ����ӵ�ԭ�����ص�����
//        point->prev = comp->tail;
//        comp->tail->next = point;
//        point->next = NULL;
//    }
//    else {
//        // ��һ������
//        point->prev = NULL;
//        point->next = NULL;
//        comp->head = point;
//    }
//
//    // �¼���ĵ���Ϊβ��
//    comp->tail = point;
//
//    // ����������
//    comp->size++;
//}
//
//// convert the point set to CvSeq
//static CvContour* MSERToContour(MSERConnectedComp* comp, CvMemStorage* storage)
//{
//    CvSeq* _contour = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage);
//    CvContour* contour = (CvContour*)_contour;
//
//    // �ϴ���ʷ����ˮλ�ĵף���ˮλ�ĵ׶���ӵ�������
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
