#include <opencv2/opencv.hpp>

// SS, SE, ES, EE表示线段起点终点的方向
#define SS 0
#define SE 1
#define ES 2
#define EE 3

// 定义边缘方向的标识
#define EDGE_VERTICAL   1     // 垂直边缘
#define EDGE_HORIZONTAL 2     // 水平边缘

// 锚点和边缘点的标识
#define ANCHOR_PIXEL  254     // 锚点像素
#define EDGE_PIXEL    255     // 边缘像素

// 定义四个方向常量，用于链连接方向
#define LEFT  1
#define RIGHT 2
#define UP    3
#define DOWN  4

// 定义梯度算子的枚举类型，选择使用不同的梯度算子进行边缘检测
enum GradientOperator {
    PREWITT_OPERATOR = 101,  // Prewitt算子
    SOBEL_OPERATOR = 102,    // Sobel算子
    SCHARR_OPERATOR = 103    // Scharr算子
};

// 栈节点结构体，用于深度优先搜索时表示像素的起点、方向和父节点
struct StackNode {
    int r, c;   // 起始像素行列坐标
    int parent; // 父节点索引（-1表示没有父节点）
    int dir;    // 方向标识
};

// 链结构体，用于存储边缘链信息
struct Chain {
    int dir;                   // 链的方向
    int len;                   // 链的像素数量
    int parent;                // 父节点索引（-1表示没有父节点）
    int children[2];           // 子节点（-1表示没有子节点）
    cv::Point *pixels;         // 指向像素数组的指针
};

// 线段结构体，用于表示检测到的线段
struct LS {
    cv::Point2d start;         // 线段的起点
    cv::Point2d end;           // 线段的终点

    LS(cv::Point2d _start, cv::Point2d _end) {
        start = _start;
        end = _end;
    }
};

// 线段信息结构体，用于存储线段的方程参数及其起点和终点
struct LineSegment {
    double a, b;               // 线段方程的参数
    int invert;                // 是否反转线段坐标系

    double sx, sy;             // 线段起点坐标
    double ex, ey;             // 线段终点坐标

    int segmentNo;             // 所属段编号
    int firstPixelIndex;       // 线段中的第一个像素的索引
    int len;                   // 线段的像素长度
    std::vector<std::pair<double, double>> keyPoints; // 线段上的关键点

    LineSegment(double _a, double _b, int _invert, double _sx, double _sy, double _ex, double _ey, int _segmentNo, int _firstPixelIndex, int _len, std::vector<std::pair<double, double>> _keyPoints) {
        a = _a;
        b = _b;
        invert = _invert;
        sx = _sx;
        sy = _sy;
        ex = _ex;
        ey = _ey;
        segmentNo = _segmentNo;
        firstPixelIndex = _firstPixelIndex;
        len = _len;
        keyPoints = _keyPoints; // 存储关键点
    }
};


// EDLines类，用于检测和处理线段
class EDLines {
public:
    // 构造函数，用于初始化EDLines的各项参数
    EDLines(cv::Mat _srcImage, GradientOperator _op = PREWITT_OPERATOR, int _gradThresh = 20,
            int _anchorThresh = 0, int _scanInterval = 1, int _minPathLen = 10,
            double _sigma = 1.5, bool _sumFlag = true, double _line_error = 1.0,
            int _min_line_len = -1, double _max_distance_between_two_lines = 6.0, double _max_error = 1.3);

    // 在原始图像上绘制检测出的线段
    cv::Mat drawOnImage();

    // 获取线段段的数量
    int getSegmentNo();
    // 获取锚点的数量
    int getAnchorNo();

    // 获取锚点列表
    std::vector<cv::Point> getAnchorPoints();
    // 获取段集合
    std::vector<std::vector<cv::Point>> getSegments();
    // 获取排序后的段集合
    std::vector<std::vector<cv::Point>> getSortedSegments();

    // 在图像上绘制特定的段
    cv::Mat drawParticularSegments(std::vector<int> list);

    // 获取检测出的线段
    std::vector<LineSegment> getLines();


protected:
    int width;  // 图像的宽度
    int height; // 图像的高度
    uchar *srcImg; // 原始图像数据
    std::vector<std::vector<cv::Point>> segmentPoints; // 存储段的像素点集合
    double sigma; // 高斯模糊的sigma值
    cv::Mat smoothImage; // 平滑后的图像
    uchar *edgeImg; // 边缘图像数据
    uchar *smoothImg; // 平滑图像数据
    int segmentNos; // 段的数量
    int minPathLen; // 最小路径长度
    cv::Mat srcImage; // 原始图像

private:
    // 计算图像梯度
    void ComputeGradient();
    // 计算锚点位置
    void ComputeAnchorPoints();
    // 使用排序后的锚点连接成段
    void JoinAnchorPointsUsingSortedAnchors();
    // 按照梯度值对锚点进行排序
    int* sortAnchorsByGradValue1();

    // 计算最长的链
    static int LongestChain(Chain *chains, int root);
    // 获取链的数量
    static int RetrieveChainNos(Chain *chains, int root, int chainNos[]);

    int anchorNos; // 锚点的数量
    std::vector<cv::Point> anchorPoints; // 锚点的坐标集合
    std::vector<cv::Point> edgePoints; // 边缘点的坐标集合

    cv::Mat edgeImage; // 存储边缘图像
    cv::Mat gradImage; // 存储梯度图像
    cv::Mat threshImage; // 存储阈值图像

    uchar *dirImg; // 像素梯度方向数据
    short *gradImg; // 像素梯度值数据

    GradientOperator gradOperator; // 使用的梯度算子
    int gradThresh; // 梯度阈值
    int anchorThresh; // 锚点阈值
    int scanInterval; // 扫描间隔
    bool sumFlag; // 是否使用累加模式

    std::vector<LineSegment> lines; // 存储有效线段
    std::vector<LineSegment> invalidLines; // 存储无效线段
    std::vector<LS> linePoints; // 存储线段的端点
    int linesNo; // 线段的数量
    int min_line_len; // 最小线段长度
    double line_error; // 线段允许的误差
    double max_distance_between_two_lines; // 两条线段之间的最大距离
    double max_error; // 最大误差
    double prec; // 精度

    // 计算最小线段长度
    int ComputeMinLineLength();
    // 将段分割成线段
    void SplitSegment2Lines(double *x, double *y, int noPixels, int segmentNo);

    // 连接共线线段
    void JoinCollinearLines();
    // 清理垂直方向的邻近锚点
    void cleanVerticalNeighbors(int r, int c);
    // 清理水平方向的邻近锚点
    void cleanHorizontalNeighbors(int r, int c);
    // 寻找水平方向的下一个像素
    bool findNextHorizontalPixel(int &r, int &c, int dir);
    // 寻找垂直方向的下一个像素
    bool findNextVerticalPixel(int &r, int &c, int dir);
    // 将水平链扩展推入栈中
    void pushHorizontalChainsToStack(StackNode *stack, int &top, int r, int c, int chainNo);
    // 将垂直链扩展推入栈中
    void pushVerticalChainsToStack(StackNode *stack, int &top, int r, int c, int chainNo);
    // 处理链条并删除冗余像素
    void processChainsAndRemoveRedundantPixels(Chain *chains, int *chainNos, int noChains);

    // 尝试连接两条共线的线段
    bool TryToJoinTwoLineSegments(LineSegment *ls1, LineSegment *ls2, int changeIndex);

    // 计算点到线段的最小距离
    static double ComputeMinDistance(double x1, double y1, double a, double b, int invert);
    // 计算最近的点
    static void ComputeClosestPoint(double x1, double y1, double a, double b, int invert, double &xOut, double &yOut);
    // 对线段进行线性拟合
    static void LineFit(double *x, double *y, int count, double &a, double &b, int invert);
    static void LineFit(double *x, double *y, int count, double &a, double &b, double &e, int &invert);
    // 计算两条线段之间的最小距离
    static double ComputeMinDistanceBetweenTwoLines(LineSegment *ls1, LineSegment *ls2, int *pwhich);
    // 更新线段的方程参数
    static void UpdateLineParameters(LineSegment *ls);
};

