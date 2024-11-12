#include "../include/line_feature.h"

using namespace cv;
using namespace std;

EDLines::EDLines(Mat _srcImage, GradientOperator _op, int _gradThresh, int _anchorThresh,
                 int _scanInterval, int _minPathLen, double _sigma, bool _sumFlag,
                 double _line_error, int _min_line_len, double _max_distance_between_two_lines, double _max_error)
{
	// 检查参数的合法性
	if (_gradThresh < 1) _gradThresh = 1;
	if (_anchorThresh < 0) _anchorThresh = 0;
	if (_sigma < 1.0) _sigma = 1.0;

	// 初始化原始图像
	srcImage = _srcImage;
	height = srcImage.rows;   // 图像高度
	width = srcImage.cols;    // 图像宽度

	gradOperator = _op;
	gradThresh = _gradThresh;
	anchorThresh = _anchorThresh;
	scanInterval = _scanInterval;
	minPathLen = _minPathLen;
	sigma = _sigma;
	sumFlag = _sumFlag;

	segmentNos = 0;
	segmentPoints.push_back(vector<Point>()); // 创建空的段点集合

	edgeImage = Mat(height, width, CV_8UC1, Scalar(0)); // 初始化边缘图像，像素值设为0
	smoothImage = Mat(height, width, CV_8UC1);          // 初始化平滑图像
	gradImage = Mat(height, width, CV_16SC1);           // 初始化梯度图像

	smoothImg = smoothImage.data;
	gradImg = (short*)gradImage.data;
	edgeImg = edgeImage.data;

	srcImg = srcImage.data;
	dirImg = new unsigned char[width * height]; // 动态分配梯度方向数组

	// 应用高斯模糊对图像进行平滑处理
	if (sigma == 1.0) {
		GaussianBlur(srcImage, smoothImage, Size(5, 5), sigma); // 使用5x5内核
	} else {
		GaussianBlur(srcImage, smoothImage, Size(), sigma); // 根据 sigma 计算内核大小
	}

	// 计算图像梯度和边缘方向
	ComputeGradient();

	// 计算锚点
	ComputeAnchorPoints();

	// 使用排序后的锚点连接线段
	JoinAnchorPointsUsingSortedAnchors();

	delete[] dirImg; // 释放内存

	min_line_len = _min_line_len;
	line_error = _line_error;
	max_distance_between_two_lines = _max_distance_between_two_lines;
	max_error = _max_error;

	if (min_line_len == -1) {
		min_line_len = ComputeMinLineLength(); // 计算最小线段长度
	}
	if (min_line_len < 9) {
		min_line_len = 9;
	}

	std::vector<double> x((width + height) * 8);
	std::vector<double> y((width + height) * 8);

	linesNo = 0;

	// 遍历段，处理每个段
	for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {
		std::vector<Point> segment = segmentPoints[segmentNumber]; // 获取当前段

		// 将当前段的像素点坐标存入 x 和 y 向量
		for (int k = 0; k < segment.size(); k++) {
			x[k] = segment[k].x;
			y[k] = segment[k].y;
		}

		// 将段分割成线段
		SplitSegment2Lines(x.data(), y.data(), segment.size(), segmentNumber);
	}

	// 连接共线线段
	JoinCollinearLines();

	// 将线段起点和终点存入线段点集合
	for (int i = 0; i < linesNo; i++) {
		Point2d start(lines[i].sx, lines[i].sy);
		Point2d end(lines[i].ex, lines[i].ey);
		linePoints.push_back(LS(start, end));
	}
}

// 返回段的数量
int EDLines::getSegmentNo()
{
	return segmentNos;
}

// 返回锚点的数量
int EDLines::getAnchorNo()
{
	return anchorNos;
}

// 返回锚点的坐标点集合
std::vector<Point> EDLines::getAnchorPoints()
{
	return anchorPoints;
}

// 返回所有段的像素点集合
std::vector<std::vector<Point>> EDLines::getSegments()
{
	return segmentPoints;
}
// 返回按长度排序后的段集合
std::vector<std::vector<Point>> EDLines::getSortedSegments()
{
		// сортируем сегметы по убыванию длины
		std::sort(segmentPoints.begin(), segmentPoints.end(), [](const std::vector<Point> & a, const std::vector<Point> & b) { return a.size() > b.size(); });

		return segmentPoints;
}

// 在图像上绘制指定的段
Mat EDLines::drawParticularSegments(std::vector<int> list)
{
	Mat segmentsImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));

	std::vector<Point>::iterator it;
	std::vector<int>::iterator itInt;

	for (itInt = list.begin(); itInt != list.end(); itInt++)
		for (it = segmentPoints[*itInt].begin(); it != segmentPoints[*itInt].end(); it++)
			segmentsImage.at<uchar>(*it) = 255;

	return segmentsImage;
}

// 计算图像的梯度和边缘方向
void EDLines::ComputeGradient()
{
	// 初始化边界梯度，避免越界访问
	for (int j = 0; j < width; j++) {
		gradImg[j] = gradImg[(height - 1) * width + j] = gradThresh - 1; // 设置顶部和底部边界
	}
	for (int i = 1; i < height - 1; i++) {
		gradImg[i * width] = gradImg[(i + 1) * width - 1] = gradThresh - 1; // 设置左右边界
	}

	// 缓存行指针，减少对 smoothImg 的重复访问
	uchar* rowPrev, * rowCurr, * rowNext;

	for (int i = 1; i < height - 1; i++) {
		// 缓存当前行和上下行
		rowPrev = &smoothImg[(i - 1) * width]; // 上一行
		rowCurr = &smoothImg[i * width];       // 当前行
		rowNext = &smoothImg[(i + 1) * width]; // 下一行

		for (int j = 1; j < width - 1; j++) {
			// 计算对角差和横纵差
			int com1 = rowNext[j + 1] - rowPrev[j - 1]; // 左上和右下差值
			int com2 = rowPrev[j + 1] - rowNext[j - 1]; // 右上和左下差值

			int gx, gy;

			// 根据选择的算子计算梯度
			switch (gradOperator) {
			case PREWITT_OPERATOR:
				gx = abs(com1 + com2 + (rowCurr[j + 1] - rowCurr[j - 1])); // 水平梯度
				gy = abs(com1 - com2 + (rowNext[j] - rowPrev[j]));         // 垂直梯度
				break;
			case SOBEL_OPERATOR:
				gx = abs(com1 + com2 + 2 * (rowCurr[j + 1] - rowCurr[j - 1]));
				gy = abs(com1 - com2 + 2 * (rowNext[j] - rowPrev[j]));
				break;
			case SCHARR_OPERATOR:
				gx = abs(3 * (com1 + com2) + 10 * (rowCurr[j + 1] - rowCurr[j - 1]));
				gy = abs(3 * (com1 - com2) + 10 * (rowNext[j] - rowPrev[j]));
				break;
			}

			// 根据标志 sumFlag，决定使用简单加和还是平方和计算梯度幅值
			int sum = sumFlag ? (gx + gy) : (int)sqrt((double)gx * gx + gy * gy);

			int index = i * width + j;// 当前像素的索引
			gradImg[index] = sum;// 将计算的梯度幅值存储到 gradImg 中

			// 如果当前像素的梯度幅值大于等于阈值，则判断其边缘方向
			if (sum >= gradThresh) {
				dirImg[index] = (gx >= gy) ? EDGE_VERTICAL : EDGE_HORIZONTAL; // 如果水平方向梯度大于垂直方向梯度，标记为垂直边缘， 否则标记为水平边缘
			}
		}
	}
}

// 计算锚点
void EDLines::ComputeAnchorPoints()
{
	// 遍历图像的每一行，跳过上下2个像素的边界，避免越界访问
	for (int i = 2; i<height - 2; i++) {
		int start = 2;// 默认从当前行的第2个像素开始遍历
		int inc = 1;// 默认步长为1
		// 根据扫描间隔 scanInterval，调整扫描的起始位置和步长
		if (i%scanInterval != 0) { start = scanInterval; inc = scanInterval; }

		// 遍历图像的每一列，跳过左右2个像素的边界
		for (int j = start; j<width - 2; j += inc) {
			// 如果当前像素的梯度值小于阈值，则跳过该像素
			if (gradImg[i*width + j] < gradThresh) continue;

			// 检查边缘方向，如果是垂直边缘
			if (dirImg[i*width + j] == EDGE_VERTICAL) {
				// 垂直边缘：与左右相邻像素的梯度比较
				int diff1 = gradImg[i*width + j] - gradImg[i*width + j - 1];
				int diff2 = gradImg[i*width + j] - gradImg[i*width + j + 1];
				// 如果当前像素的梯度明显大于左右相邻像素的梯度，且差值超过锚点阈值
				if (diff1 >= anchorThresh && diff2 >= anchorThresh) {
					edgeImg[i*width + j] = ANCHOR_PIXEL;
					anchorPoints.push_back(Point(j, i));
				}

			}
			else {
				// 如果是水平边缘, 与上下相邻像素的梯度比较
				int diff1 = gradImg[i*width + j] - gradImg[(i - 1)*width + j];
				int diff2 = gradImg[i*width + j] - gradImg[(i + 1)*width + j];
				if (diff1 >= anchorThresh && diff2 >= anchorThresh) {
					edgeImg[i*width + j] = ANCHOR_PIXEL;
					anchorPoints.push_back(Point(j, i));
				}
			}
		}
	}
	// 统计总锚点数量，并将其存入 anchorNos
	anchorNos = anchorPoints.size();
}

void EDLines::JoinAnchorPointsUsingSortedAnchors()
{
	int *chainNos = new int[(width + height) * 8];

	Point *pixels = new Point[width*height];
	StackNode *stack = new StackNode[width*height];
	Chain *chains = new Chain[width*height];

	// сортируем опорные точки по убыванию градиента в них
	int *A = sortAnchorsByGradValue1();

	// соединяем опорные точки начиная с наибольших значений градиента
	int totalPixels = 0;

	for (int k = anchorNos - 1; k >= 0; k--) {
		int pixelOffset = A[k];

		int i = pixelOffset / width;
		int j = pixelOffset % width;


		if (edgeImg[i*width + j] != ANCHOR_PIXEL) continue;

		chains[0].len = 0;
		chains[0].parent = -1;
		chains[0].dir = 0;
		chains[0].children[0] = chains[0].children[1] = -1;
		chains[0].pixels = NULL;


		int noChains = 1;
		int len = 0;
		int duplicatePixelCount = 0;
		int top = -1;  // вершина стека

		if (dirImg[i*width + j] == EDGE_VERTICAL) {
			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = DOWN;
			stack[top].parent = 0;

			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = UP;
			stack[top].parent = 0;

		}
		else {
			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = RIGHT;
			stack[top].parent = 0;

			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = LEFT;
			stack[top].parent = 0;
		} //end-else

		  // пока стек не пуст
	StartOfWhile:
		while (top >= 0) {
			int r = stack[top].r;
			int c = stack[top].c;
			int dir = stack[top].dir;
			int parent = stack[top].parent;
			top--;

			if (edgeImg[r*width + c] != EDGE_PIXEL) duplicatePixelCount++;

			chains[noChains].dir = dir;   // traversal direction
			chains[noChains].parent = parent;
			chains[noChains].children[0] = chains[noChains].children[1] = -1;


			int chainLen = 0;

			chains[noChains].pixels = &pixels[len];

			pixels[len].y = r;
			pixels[len].x = c;
			len++;
			chainLen++;

			if (dir == LEFT) {
				while (dirImg[r*width + c] == EDGE_HORIZONTAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;

					// Грань горизонтальная. Направлена влево
					//
					//   A
					//   B x
					//   C
					//
					// очищаем верхний и нижний пиксели
					if (edgeImg[(r - 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r - 1)*width + c] = 0;
					if (edgeImg[(r + 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r + 1)*width + c] = 0;

					// ищем пиксель на грани среди соседей
					if (edgeImg[r*width + c - 1] >= ANCHOR_PIXEL) { c--; }
					else if (edgeImg[(r - 1)*width + c - 1] >= ANCHOR_PIXEL) { r--; c--; }
					else if (edgeImg[(r + 1)*width + c - 1] >= ANCHOR_PIXEL) { r++; c--; }
					else {
						// иначе -- идем в максимальный по градиенту пиксель СЛЕВА
						int A = gradImg[(r - 1)*width + c - 1];
						int B = gradImg[r*width + c - 1];
						int C = gradImg[(r + 1)*width + c - 1];

						if (A > B) {
							if (A > C) r--;
							else       r++;
						}
						else  if (C > B) r++;
						c--;
					}

					if (edgeImg[r*width + c] == EDGE_PIXEL || gradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[0] = noChains;
							noChains++;
						}
						goto StartOfWhile;
					}


					pixels[len].y = r;
					pixels[len].x = c;
					len++;
					chainLen++;
				}

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = DOWN;
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = UP;
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[0] = noChains;
				noChains++;

			}
			else if (dir == RIGHT) {
				while (dirImg[r*width + c] == EDGE_HORIZONTAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;

					// Грань горизонтальная. Направлена вправо
					//
					//     A
					//   x B
					//     C
					//
					// очищаем верхний и нижний пиксели
					if (edgeImg[(r + 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r + 1)*width + c] = 0;
					if (edgeImg[(r - 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r - 1)*width + c] = 0;

					// ищем пиксель на грани среди соседей
					if (edgeImg[r*width + c + 1] >= ANCHOR_PIXEL) { c++; }
					else if (edgeImg[(r + 1)*width + c + 1] >= ANCHOR_PIXEL) { r++; c++; }
					else if (edgeImg[(r - 1)*width + c + 1] >= ANCHOR_PIXEL) { r--; c++; }
					else {
						// иначе -- идем в максимальный по градиенту пиксель СПРАВА
						int A = gradImg[(r - 1)*width + c + 1];
						int B = gradImg[r*width + c + 1];
						int C = gradImg[(r + 1)*width + c + 1];

						if (A > B) {
							if (A > C) r--;       // A
							else       r++;       // C
						}
						else if (C > B) r++;  // C
						c++;
					}

					if (edgeImg[r*width + c] == EDGE_PIXEL || gradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[1] = noChains;
							noChains++;
						}
						goto StartOfWhile;
					}


					pixels[len].y = r;
					pixels[len].x = c;
					len++;
					chainLen++;
				}

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = DOWN;
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = UP;
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[1] = noChains;
				noChains++;

			}
			else if (dir == UP) {
				while (dirImg[r*width + c] == EDGE_VERTICAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;

					// Грань вертикальная. Направлена вверх
					//
					//   A B C
					//     x
					//
					// очищаем левый и правый пиксели
					if (edgeImg[r*width + c - 1] == ANCHOR_PIXEL) edgeImg[r*width + c - 1] = 0;
					if (edgeImg[r*width + c + 1] == ANCHOR_PIXEL) edgeImg[r*width + c + 1] = 0;

					// ищем пиксель на грани среди соседей
					if (edgeImg[(r - 1)*width + c] >= ANCHOR_PIXEL) { r--; }
					else if (edgeImg[(r - 1)*width + c - 1] >= ANCHOR_PIXEL) { r--; c--; }
					else if (edgeImg[(r - 1)*width + c + 1] >= ANCHOR_PIXEL) { r--; c++; }
					else {
						// иначе -- идем в максимальный по градиенту пиксель ВВЕРХ
						int A = gradImg[(r - 1)*width + c - 1];
						int B = gradImg[(r - 1)*width + c];
						int C = gradImg[(r - 1)*width + c + 1];

						if (A > B) {
							if (A > C) c--;
							else       c++;
						}
						else if (C > B) c++;
						r--;
					}

					if (edgeImg[r*width + c] == EDGE_PIXEL || gradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[0] = noChains;
							noChains++;
						}
						goto StartOfWhile;
					}


					pixels[len].y = r;
					pixels[len].x = c;

					len++;
					chainLen++;
				}

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = RIGHT;
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = LEFT;
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[0] = noChains;
				noChains++;

			}
			else {
				while (dirImg[r*width + c] == EDGE_VERTICAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;

					// Грань вертикальная. Направлена вниз
					//
					//     x
					//   A B C
					//
					// очищаем пиксле слева и справа
					if (edgeImg[r*width + c + 1] == ANCHOR_PIXEL) edgeImg[r*width + c + 1] = 0;
					if (edgeImg[r*width + c - 1] == ANCHOR_PIXEL) edgeImg[r*width + c - 1] = 0;

					// ищем пиксель на грани среди соседей
					if (edgeImg[(r + 1)*width + c] >= ANCHOR_PIXEL) { r++; }
					else if (edgeImg[(r + 1)*width + c + 1] >= ANCHOR_PIXEL) { r++; c++; }
					else if (edgeImg[(r + 1)*width + c - 1] >= ANCHOR_PIXEL) { r++; c--; }
					else {
						// иначе -- идем в максимальный по градиенту пиксель ВНИЗУ
						int A = gradImg[(r + 1)*width + c - 1];
						int B = gradImg[(r + 1)*width + c];
						int C = gradImg[(r + 1)*width + c + 1];

						if (A > B) {
							if (A > C) c--;       // A
							else       c++;       // C
						}
						else if (C > B) c++;  // C
						r++;
					}

					if (edgeImg[r*width + c] == EDGE_PIXEL || gradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[1] = noChains;
							noChains++;
						}
						goto StartOfWhile;
					}

					pixels[len].y = r;
					pixels[len].x = c;

					len++;
					chainLen++;
				}

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = RIGHT;
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = LEFT;
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[1] = noChains;
				noChains++;
			}

		}


		if (len - duplicatePixelCount < minPathLen) {
			for (int k = 0; k<len; k++) {

				edgeImg[pixels[k].y*width + pixels[k].x] = 0;
				edgeImg[pixels[k].y*width + pixels[k].x] = 0;

			}

		}
		else {

			int noSegmentPixels = 0;

			int totalLen = LongestChain(chains, chains[0].children[1]);

			if (totalLen > 0) {
				int count = RetrieveChainNos(chains, chains[0].children[1], chainNos);

				// копируем пиксели в обратном порядке
				for (int k = count - 1; k >= 0; k--) {
					int chainNo = chainNos[k];

                    /* Пробуем удалить лишние пиксели */

                    int fr = chains[chainNo].pixels[chains[chainNo].len - 1].y;
                    int fc = chains[chainNo].pixels[chains[chainNo].len - 1].x;

                    int index = noSegmentPixels - 2;
                    while (index >= 0) {
                        int dr = abs(fr - segmentPoints[segmentNos][index].y);
                        int dc = abs(fc - segmentPoints[segmentNos][index].x);

                        if (dr <= 1 && dc <= 1) {
                            // neighbors. Erase last pixel
                            segmentPoints[segmentNos].pop_back();
                            noSegmentPixels--;
                            index--;
                        }
                        else break;
                    } //end-while

                    if (chains[chainNo].len > 1 && noSegmentPixels > 0) {
                        fr = chains[chainNo].pixels[chains[chainNo].len - 2].y;
                        fc = chains[chainNo].pixels[chains[chainNo].len - 2].x;

                        int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
                        int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

                        if (dr <= 1 && dc <= 1) chains[chainNo].len--;
                    }

					for (int l = chains[chainNo].len - 1; l >= 0; l--) {
						segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
						noSegmentPixels++;
					}

					chains[chainNo].len = 0;  // помечаем скопированной
				}
			}

			totalLen = LongestChain(chains, chains[0].children[0]);
			if (totalLen > 1) {

				int count = RetrieveChainNos(chains, chains[0].children[0], chainNos);

                // копируем цепочку в прямом порядке. пропускаем первый пиксель в цепи
				int lastChainNo = chainNos[0];
				chains[lastChainNo].pixels++;
				chains[lastChainNo].len--;

				for (int k = 0; k<count; k++) {
					int chainNo = chainNos[k];

					/* Пробуем удалить лишние пиксели */
					int fr = chains[chainNo].pixels[0].y;
					int fc = chains[chainNo].pixels[0].x;

					int index = noSegmentPixels - 2;
					while (index >= 0) {
						int dr = abs(fr - segmentPoints[segmentNos][index].y);
						int dc = abs(fc - segmentPoints[segmentNos][index].x);

						if (dr <= 1 && dc <= 1) {
							segmentPoints[segmentNos].pop_back();
							noSegmentPixels--;
							index--;
						}
						else break;
					}

					int startIndex = 0;
					int chainLen = chains[chainNo].len;
					if (chainLen > 1 && noSegmentPixels > 0) {
						int fr = chains[chainNo].pixels[1].y;
						int fc = chains[chainNo].pixels[1].x;

						int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
						int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

						if (dr <= 1 && dc <= 1) { startIndex = 1; }
					}

					for (int l = startIndex; l<chains[chainNo].len; l++) {
						segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
						noSegmentPixels++;
					}

					chains[chainNo].len = 0;  // помечаем скопированной
				}
			}


			  //  Пробуем удалить лишние пиксели
			int fr = segmentPoints[segmentNos][1].y;
			int fc = segmentPoints[segmentNos][1].x;


			int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
			int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);


			if (dr <= 1 && dc <= 1) {
				segmentPoints[segmentNos].erase(segmentPoints[segmentNos].begin());
				noSegmentPixels--;
			} //end-if

			segmentNos++;
			segmentPoints.push_back(vector<Point>());

													  // копируем оставшиеся цепочки сюда
			for (int k = 2; k<noChains; k++) {
				if (chains[k].len < 2) continue;

				totalLen = LongestChain(chains, k);

				if (totalLen >= 10) {

					int count = RetrieveChainNos(chains, k, chainNos);

					// копируем пиксели
					noSegmentPixels = 0;
					for (int k = 0; k<count; k++) {
						int chainNo = chainNos[k];

						/* Пробуем удалить лишние пиксели */
						int fr = chains[chainNo].pixels[0].y;
						int fc = chains[chainNo].pixels[0].x;

						int index = noSegmentPixels - 2;
						while (index >= 0) {
							int dr = abs(fr - segmentPoints[segmentNos][index].y);
							int dc = abs(fc - segmentPoints[segmentNos][index].x);

							if (dr <= 1 && dc <= 1) {
								// удаляем последний пиксель т к соседи
								segmentPoints[segmentNos].pop_back();
								noSegmentPixels--;
								index--;
							}
							else break;
						}

						int startIndex = 0;
						int chainLen = chains[chainNo].len;
						if (chainLen > 1 && noSegmentPixels > 0) {
							int fr = chains[chainNo].pixels[1].y;
							int fc = chains[chainNo].pixels[1].x;

							int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
							int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

							if (dr <= 1 && dc <= 1) { startIndex = 1; }
						}
						for (int l = startIndex; l<chains[chainNo].len; l++) {
							segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
							noSegmentPixels++;
						}

						chains[chainNo].len = 0;  // помечаем скопироавнной
					}
					segmentPoints.push_back(vector<Point>());
					segmentNos++;
				}
			}

		}

	}

    // удаляем последний сегмент из массива, т.к. он пуст
	segmentPoints.pop_back();

	delete[] A;
	delete[] chains;
	delete[] stack;
	delete[] chainNos;
	delete[] pixels;
}


// 根据像素的梯度值对锚点进行排序
int* EDLines::sortAnchorsByGradValue1() {
	int SIZE = 128 * 256;
	std::vector<int> C(SIZE, 0); // 使用 std::vector 进行统计

	// 统计每个梯度值的出现次数
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (edgeImg[i * width + j] != ANCHOR_PIXEL) continue;
			int grad = gradImg[i * width + j];
			C[grad]++;
		}
	}

	// 计算累积计数
	for (int i = 1; i < SIZE; i++) {
		C[i] += C[i - 1];
	}

	int noAnchors = C[SIZE - 1];

	// 手动分配动态数组来存储排序结果
	int* A = new int[noAnchors];

	// 根据梯度值排序锚点
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (edgeImg[i * width + j] != ANCHOR_PIXEL) continue;
			int grad = gradImg[i * width + j];
			int index = --C[grad];
			A[index] = i * width + j;
		}
	}

	return A; // 返回排序后的动态数组
}

/**
 * @brief 计算链条树中从指定节点开始的最长链条长度
 *
 * 该函数通过递归遍历链条的左右子节点，找到最长的链条路径，并返回该链条的总长度。
 * 如果当前节点的子节点链条较短，则剪枝该子链，以优化后续计算。
 *
 * @param chains 链条数组，存储链条的各个节点信息
 * @param root 当前递归计算的链条节点的索引，表示树的根节点
 * @return int 返回从当前链条节点开始的最长链条的长度
 */
int EDLines::LongestChain(Chain *chains, int root) {
	// 如果根节点为 -1 或者当前链的长度为 0，说明已经到达了末端，返回 0
	if (root == -1 || chains[root].len == 0) return 0;

	int len0 = 0;
	// 递归计算左子节点的最长链条长度，如果存在左子节点
	if (chains[root].children[0] != -1) len0 = LongestChain(chains, chains[root].children[0]);

	int len1 = 0;
	// 递归计算右子节点的最长链条长度，如果存在右子节点
	if (chains[root].children[1] != -1) len1 = LongestChain(chains, chains[root].children[1]);

	int max = 0;
	// 比较左右子节点的链条长度，选择最长的一条路径
	if (len0 >= len1) {
		// 如果左链条更长，则保存左子节点，剪枝右子节点
		max = len0;
		chains[root].children[1] = -1; // 剪枝右子节点
	} else {
		// 如果右链条更长，则保存右子节点，剪枝左子节点
		max = len1;
		chains[root].children[0] = -1; // 剪枝左子节点
	}

	// 返回当前节点的链条长度加上最长子节点链条的长度
	return chains[root].len + max;
}

/**
 * @brief 从链条树的根节点开始，遍历链条，并将节点编号存入数组中
 *
 * 该函数从指定的根节点 `root` 开始，沿着链条的子节点（优先选择左子节点）向下遍历，将每个链条节点的索引存储到数组 `chainNos` 中。
 * 如果左子节点存在，则继续向左遍历；如果不存在，则向右子节点遍历，直到到达链条末端。
 *
 * @param chains 链条数组，存储每个链条节点的相关信息
 * @param root 当前遍历的根节点索引
 * @param chainNos 输出数组，用于存储遍历过的链条节点索引
 * @return int 返回遍历的链条节点数量
 */
int EDLines::RetrieveChainNos(Chain *chains, int root, int chainNos[]) {
	int count = 0; // 计数器，用于记录遍历过的节点数量

	// 从根节点开始，逐步遍历链条，直到到达链条末端（即 root 为 -1）
	while (root != -1) {
		chainNos[count] = root; // 将当前节点的索引存入结果数组中
		count++; // 增加计数器，表示已遍历的节点数量

		// 优先选择左子节点继续遍历，如果没有左子节点则转向右子节点
		if (chains[root].children[0] != -1)
			root = chains[root].children[0]; // 如果存在左子节点，继续向左遍历
		else
			root = chains[root].children[1]; // 如果没有左子节点，则转向右子节点
	}

	// 返回遍历过的节点数量
	return count;
}

// 返回 linePoints，即所有检测到的线段
vector<LineSegment> EDLines::getLines()
{
    return lines;
}

// 将检测到的线段绘制在原始图像上，并返回彩色图像
Mat EDLines::drawOnImage()
{
    Mat colorImage = Mat(height, width, CV_8UC1, srcImg);
    cvtColor(colorImage, colorImage, COLOR_GRAY2BGR);
    for (int i = 0; i < linesNo; i++) {
        line(colorImage, linePoints[i].start, linePoints[i].end, Scalar(0, 255, 0), 1, LINE_AA, 0); // draw lines as green on image
    }

    return colorImage;
}

/**
 * @brief 计算最小线段长度
 *
 * 该函数使用 NFA（错误报警数）的公式来计算图像中可以被检测到的最小线段长度。
 * 通过结合图像的宽度和高度，以及固定的经验参数，估算在随机噪声中找到与检测到的线段相似的错误线段的可能性。
 *
 * @return int 返回最小线段的长度，线段长度小于该值的线段可能是由噪声引起的。
 */
int EDLines::ComputeMinLineLength() {

	// logNT 是与图像尺寸相关的一个常数，用于调整线段长度的检测标准
	double logNT = 2.0 * (log10((double)width) + log10((double)height));

	// 计算并返回最小线段长度，round 用于四舍五入，log10(0.125) 是经验参数
	// todo:缩放因子0.5改为3.0
	return (int) round((-logNT / log10(0.125)) * 0.5);
}

/**
 * @brief 将像素序列分割成线段
 *
 * 该函数用于将给定的一段像素点序列（段）分割成多个线段。首先尝试拟合最小长度的线段，
 * 如果拟合成功，则尝试延长线段，直到误差超出限制或没有更多的像素可以延长为止。
 *
 * @param x 像素点的 x 坐标数组
 * @param y 像素点的 y 坐标数组
 * @param noPixels 该段中的像素点数量
 * @param segmentNo 段的编号
 */
void EDLines::SplitSegment2Lines(double *x, double *y, int noPixels, int segmentNo) {
    // 第一个像素的索引
    int firstPixelIndex = 0;

    // 当剩余像素数量大于等于最小线段长度时，开始分割
    while (noPixels >= min_line_len) {
        bool valid = false;
        double lastA, lastB, error;
        int lastInvert;

        std::vector<std::pair<double, double>> keyPoints;  // 用于存储关键点

        while (noPixels >= min_line_len) {
            LineFit(x, y, min_line_len, lastA, lastB, error, lastInvert);
            if (error <= 0.5) {
                valid = true;
                break;
            }

            noPixels -= 1;
            x += 1; y += 1;
            firstPixelIndex += 1;
        }

        if (!valid) return;

        // 尝试延长线段
        int index = min_line_len;
        int len = min_line_len;

        while (index < noPixels) {
            int startIndex = index;
            int lastGoodIndex = index - 1;
            int goodPixelCount = 0;
            int badPixelCount = 0;

            // 检查是否可以继续延长线段
            while (index < noPixels) {
                double d = ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert);

                if (d <= line_error) {
                    lastGoodIndex = index;
                    goodPixelCount++;
                    badPixelCount = 0;

                    // 如果当前点与拟合线段的距离超过一定阈值，存储为关键点
                    if (d > 0.3) {  // 这里设置一个较小的阈值用于筛选关键点
                        keyPoints.push_back({x[index], y[index]});
                    }

                } else {
                    badPixelCount++;
                    if (badPixelCount >= 5) break;
                }

                index++;
            }

            if (goodPixelCount >= 2) {
                len += lastGoodIndex - startIndex + 1;
                LineFit(x, y, len, lastA, lastB, lastInvert);
                index = lastGoodIndex + 1;
            }

            if (goodPixelCount < 2 || index >= noPixels) {
                double sx, sy, ex, ey;

                // 计算线段的起点
                int index = 0;
                while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error) index++;
                ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, sx, sy);
                int noSkippedPixels = index;

                // 计算线段的终点
                index = lastGoodIndex;
                while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error) index--;
                ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, ex, ey);

                // 添加起点和终点为关键点
                keyPoints.insert(keyPoints.begin(), {sx, sy});
                keyPoints.push_back({ex, ey});

                /*// 自适应调整关键点数量
                size_t totalPoints = keyPoints.size();
                size_t maxPoints = std::max(5, int(len / 10)); // 根据线段长度自适应关键点数量, 最少5个点

                if (totalPoints > maxPoints) {
                    std::vector<std::pair<double, double>> sampledKeyPoints;
                    double step = double(totalPoints) / maxPoints;
                    for (size_t i = 0; i < maxPoints; ++i) {
                        sampledKeyPoints.push_back(keyPoints[int(i * step)]);
                    }
                    keyPoints = sampledKeyPoints; // 替换为均匀采样后的关键点
                }*/
            	// 按照 x 坐标从小到大排序
            	std::sort(keyPoints.begin(), keyPoints.end(), [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
            		return a.first < b.first; });

                // 创建 LineSegment 对象并保存关键点
                lines.push_back(LineSegment(lastA, lastB, lastInvert, sx, sy, ex, ey, segmentNo,
                                            firstPixelIndex + noSkippedPixels, index - noSkippedPixels + 1, keyPoints));
                linesNo++;
                len = index + 1;
                break;
            }
        }

        noPixels -= len;
        x += len;
        y += len;
        firstPixelIndex += len;
    }
}

/**
 * @brief 连接共线的线段
 *
 * 该函数遍历所有检测到的线段，检查同一段内是否有共线的线段，并尝试将这些线段合并为一条更长的线段。
 * 如果可以连接，则合并两条线段，否则保留原线段。处理完所有线段后，更新线段的总数。
 */
void EDLines::JoinCollinearLines() {
	int lastLineIndex = -1; // 用于跟踪上一个处理过的线段索引
	int i = 0;

	// 遍历所有线段
	while (i < linesNo) {
		int segmentNo = lines[i].segmentNo; // 获取当前线段的段号

		lastLineIndex++;
		// 如果当前线段和之前的线段不是同一个，则移动到新的位置
		if (lastLineIndex != i)
			lines[lastLineIndex] = lines[i];

		int firstLineIndex = lastLineIndex; // 记录当前段的第一个线段索引

		int count = 1;
		// 遍历当前段中的其他线段，尝试连接
		for (int j = i + 1; j < linesNo; j++) {
			// 如果不是同一段的线段，则停止处理
			if (lines[j].segmentNo != segmentNo) break;

			// 尝试将当前线段与上一条线段连接
			if (TryToJoinTwoLineSegments(&lines[lastLineIndex], &lines[j], lastLineIndex) == false) {
				lastLineIndex++;
				// 如果不能连接，保留当前线段到新的位置
				if (lastLineIndex != j)
					lines[lastLineIndex] = lines[j];
			}

			count++;
		}

		// 尝试将段的第一个线段与最后一个线段连接，形成闭环
		if (firstLineIndex != lastLineIndex) {
			if (TryToJoinTwoLineSegments(&lines[firstLineIndex], &lines[lastLineIndex], firstLineIndex)) {
				lastLineIndex--; // 如果成功连接，减少线段数量
			}
		}

		// 跳过已经处理过的线段
		i += count;
	}

	// 更新线段总数
	linesNo = lastLineIndex + 1;
}

/**
 * @brief 计算点 (x1, y1) 到直线的最小距离
 *
 * 直线由方程 y = a + bx 或 x = a + by 表示，函数根据 invert 参数选择是处理水平线还是垂直线。
 * 通过计算垂直线与直线的交点，得到点到直线的最小距离。
 *
 * @param x1 点的 x 坐标
 * @param y1 点的 y 坐标
 * @param a 直线方程中的参数 a
 * @param b 直线方程中的参数 b
 * @param invert 反转标志，0 表示水平线，1 表示垂直线
 * @return double 返回点到直线的最小距离
 */
double EDLines::ComputeMinDistance(double x1, double y1, double a, double b, int invert) {
	double x2, y2; // 交点的坐标

	// 处理水平线 (y = a + bx)
	if (invert == 0) {
		if (b == 0) {
			// 当 b == 0 时，直线为水平线 y = a
			x2 = x1; // 最近的 x 坐标为 x1
			y2 = a;  // 最近的 y 坐标为 a
		} else {
			// 当 b != 0 时，计算垂直于给定直线的线，并求交点
			double d = -1.0 / (b);    // 垂直线的斜率
			double c = y1 - d * x1;   // 垂直线的截距
			x2 = (a - c) / (d - b);   // 计算交点的 x 坐标
			y2 = a + b * x2;          // 计算交点的 y 坐标
		}
	}
	// 处理垂直线 (x = a + by)
	else {
		if (b == 0) {
			// 当 b == 0 时，直线为垂直线 x = a
			x2 = a;  // 最近的 x 坐标为 a
			y2 = y1; // 最近的 y 坐标为 y1
		} else {
			// 当 b != 0 时，计算垂直于给定直线的线，并求交点
			double d = -1.0 / (b);    // 垂直线的斜率
			double c = x1 - d * y1;   // 垂直线的截距
			y2 = (a - c) / (d - b);   // 计算交点的 y 坐标
			x2 = a + b * y2;          // 计算交点的 x 坐标
		}
	}

	// 返回点 (x1, y1) 到直线的最小距离
	return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

/**
 * @brief 计算点 (x1, y1) 到直线的最近点
 *
 * 该函数根据给定点 (x1, y1) 和直线方程，计算最近的垂直交点，并将最近点的坐标存储在 xOut 和 yOut 中。
 *
 * @param x1 点的 x 坐标
 * @param y1 点的 y 坐标
 * @param a 直线方程中的参数 a
 * @param b 直线方程中的参数 b
 * @param invert 反转标志，0 表示水平线，1 表示垂直线
 * @param xOut 输出，最近点的 x 坐标
 * @param yOut 输出，最近点的 y 坐标
 */
void EDLines::ComputeClosestPoint(double x1, double y1, double a, double b, int invert, double &xOut, double &yOut) {
	double x2, y2; // 存储最近点的临时变量

	// 处理水平线 (y = a + bx)
	if (invert == 0) {
		if (b == 0) {
			// 直线为水平线 y = a，最近点的 y 坐标为 a，x 坐标为 x1
			x2 = x1;
			y2 = a;
		} else {
			// 计算垂直线的斜率 d，交点 (x2, y2)
			double d = -1.0 / (b);    // 垂直线的斜率
			double c = y1 - d * x1;   // 垂直线的截距
			x2 = (a - c) / (d - b);   // 计算交点的 x 坐标
			y2 = a + b * x2;          // 计算交点的 y 坐标
		}
	}
	// 处理垂直线 (x = a + by)
	else {
		if (b == 0) {
			// 直线为垂直线 x = a，最近点的 x 坐标为 a，y 坐标为 y1
			x2 = a;
			y2 = y1;
		} else {
			// 计算垂直线的斜率 d，交点 (x2, y2)
			double d = -1.0 / (b);    // 垂直线的斜率
			double c = x1 - d * y1;   // 垂直线的截距
			y2 = (a - c) / (d - b);   // 计算交点的 y 坐标
			x2 = a + b * y2;          // 计算交点的 x 坐标
		}
	}

	// 输出最近点的坐标
	xOut = x2;
	yOut = y2;
}

/**
 * @brief 拟合直线方程 y = a + bx 或 x = a + by
 *
 * 根据输入的点集，拟合出一条直线方程。函数可以通过 `invert` 参数来选择是拟合水平线（y = a + bx）
 * 还是垂直线（x = a + by）。此函数不计算拟合误差。
 *
 * @param x 要拟合的点的 x 坐标数组
 * @param y 要拟合的点的 y 坐标数组
 * @param count 点的数量，必须大于等于 2
 * @param a 输出直线方程的截距（a）
 * @param b 输出直线方程的斜率（b）
 * @param invert 方向标志，0 表示水平线（y = a + bx），1 表示垂直线（x = a + by）
 */
void EDLines::LineFit(double *x, double *y, int count, double &a, double &b, int invert) {
	if (count < 2) return; // 点数小于2时无法拟合

	// 初始化累加变量
	double S = count, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
	for (int i = 0; i < count; i++) {
		Sx += x[i];
		Sy += y[i];
	}

	// 如果是垂直线，交换 x 和 y
	if (invert) {
		double *t = x;
		x = y;
		y = t;
		std::swap(Sx, Sy); // 交换 Sx 和 Sy
	}

	// 计算 Sxx 和 Sxy
	for (int i = 0; i < count; i++) {
		Sxx += x[i] * x[i];
		Sxy += x[i] * y[i];
	}

	// 使用最小二乘法计算 a 和 b
	double D = S * Sxx - Sx * Sx;
	a = (Sxx * Sy - Sx * Sxy) / D;
	b = (S * Sxy - Sx * Sy) / D;
}
/**
 * @brief 拟合直线方程 y = a + bx 或 x = a + by，并计算拟合误差
 *
 * 根据输入的点集，拟合出一条直线方程，同时自动检测线段的方向（水平或垂直），并计算拟合误差。
 * 当 dx < dy 时，认为线段是垂直的，并交换 x 和 y。
 *
 * @param x 要拟合的点的 x 坐标数组
 * @param y 要拟合的点的 y 坐标数组
 * @param count 点的数量，必须大于等于 2
 * @param a 输出直线方程的截距（a）
 * @param b 输出直线方程的斜率（b）
 * @param e 输出拟合误差
 * @param invert 输出方向标志，0 表示水平线，1 表示垂直线
 */
void EDLines::LineFit(double *x, double *y, int count, double &a, double &b, double &e, int &invert) {
	if (count < 2) return; // 点数小于2时无法拟合

	// 初始化累加变量
	double S = count, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
	for (int i = 0; i < count; i++) {
		Sx += x[i];
		Sy += y[i];
	}

	// 计算 x 和 y 的平均值
	double mx = Sx / count;
	double my = Sy / count;

	// 计算 dx 和 dy，用于判断是水平线还是垂直线
	double dx = 0.0, dy = 0.0;
	for (int i = 0; i < count; i++) {
		dx += (x[i] - mx) * (x[i] - mx);
		dy += (y[i] - my) * (y[i] - my);
	}

	// 如果 dx < dy，认为线段是垂直的，交换 x 和 y
	if (dx < dy) {
		invert = 1;
		std::swap(x, y);
		std::swap(Sx, Sy);
	} else {
		invert = 0;
	}

	// 计算 Sxx 和 Sxy
	for (int i = 0; i < count; i++) {
		Sxx += x[i] * x[i];
		Sxy += x[i] * y[i];
	}

	// 使用最小二乘法计算 a 和 b
	double D = S * Sxx - Sx * Sx;
	a = (Sxx * Sy - Sx * Sxy) / D;
	b = (S * Sxy - Sx * Sy) / D;

	// 计算拟合误差
	if (b == 0.0) {
		// 如果斜率为 0，计算直线和点之间的误差
		double error = 0.0;
		for (int i = 0; i < count; i++) {
			error += fabs(a - y[i]);
		}
		e = error / count;
	} else {
		// 斜率不为 0 时，计算所有点到直线的最小距离
		double error = 0.0;
		for (int i = 0; i < count; i++) {
			double d = -1.0 / b;
			double c = y[i] - d * x[i];
			double x2 = (a - c) / (d - b);
			double y2 = a + b * x2;
			double dist = (x[i] - x2) * (x[i] - x2) + (y[i] - y2) * (y[i] - y2);
			error += dist;
		}
		e = sqrt(error / count); // 计算均方根误差
	}
}

/**
 * @brief 检查两条线段是否共线，并尝试将其合并
 *
 * 该函数检测 `ls1` 和 `ls2` 是否可以合并为一条共线的线段。如果可以合并，`ls1` 的起点或终点
 * 会更新为合并后的线段，并返回 true。合并过程中，`ls2` 保持不变。
 *
 * @param ls1 第一条线段（将更新）
 * @param ls2 第二条线段（保持不变）
 * @param changeIndex 更新 `lines` 数组中对应线段的索引
 * @return bool 如果线段合并成功，则返回 true，否则返回 false
 */
bool EDLines::TryToJoinTwoLineSegments(LineSegment *ls1, LineSegment *ls2, int changeIndex) {
    int which;
    // 计算两条线段之间的最小距离
    double dist = ComputeMinDistanceBetweenTwoLines(ls1, ls2, &which);
    if (dist > max_distance_between_two_lines) return false; // 如果距离大于最大允许距离，则不合并

    // 计算两条线段的长度
    double dx = ls1->sx - ls1->ex;
    double dy = ls1->sy - ls1->ey;
    double prevLen = sqrt(dx * dx + dy * dy);

    dx = ls2->sx - ls2->ex;
    dy = ls2->sy - ls2->ey;
    double nextLen = sqrt(dx * dx + dy * dy);

    // 确定较短和较长的线段
    LineSegment *shorter = ls1;
    LineSegment *longer = ls2;
    if (prevLen > nextLen) {
        shorter = ls2;
        longer = ls1;
    }

    // 使用三点法检查共线性
    dist = ComputeMinDistance(shorter->sx, shorter->sy, longer->a, longer->b, longer->invert);
    dist += ComputeMinDistance((shorter->sx + shorter->ex) / 2.0, (shorter->sy + shorter->ey) / 2.0, longer->a, longer->b, longer->invert);
    dist += ComputeMinDistance(shorter->ex, shorter->ey, longer->a, longer->b, longer->invert);
    dist /= 3.0;

    if (dist > max_error) return false; // 如果误差超过最大允许误差，则不合并

    // 选择四种连接方式之一
    double max = fabs(ls1->sx - ls2->sx) + fabs(ls1->sy - ls2->sy);
    which = 1;

    double d = fabs(ls1->sx - ls2->ex) + fabs(ls1->sy - ls2->ey);
    if (d > max) { max = d; which = 2; }

    d = fabs(ls1->ex - ls2->sx) + fabs(ls1->ey - ls2->sy);
    if (d > max) { max = d; which = 3; }

    d = fabs(ls1->ex - ls2->ex) + fabs(ls1->ey - ls2->ey);
    if (d > max) { which = 4; }

    // 根据选择的方式更新线段端点
    if (which == 1) {
        ls1->ex = ls2->sx;
        ls1->ey = ls2->sy;
    } else if (which == 2) {
        ls1->ex = ls2->ex;
        ls1->ey = ls2->ey;
    } else if (which == 3) {
        ls1->sx = ls2->sx;
        ls1->sy = ls2->sy;
    } else {
        ls1->sx = ls1->ex;
        ls1->sy = ls1->ey;
        ls1->ex = ls2->ex;
        ls1->ey = ls2->ey;
    }

    // 更新线段长度
    if (ls1->firstPixelIndex + ls1->len + 5 >= ls2->firstPixelIndex) {
        ls1->len += ls2->len;
    } else if (ls2->len > ls1->len) {
        ls1->firstPixelIndex = ls2->firstPixelIndex;
        ls1->len = ls2->len;
    }

    // 更新线段参数
    UpdateLineParameters(ls1);
    lines[changeIndex] = *ls1; // 更新 lines 数组中的线段

    return true; // 合并成功
}

/**
 * @brief 计算两条线段的端点之间的最小距离
 *
 * 该函数计算 `ls1` 和 `ls2` 的四对端点之间的距离（起点-起点、起点-终点、终点-起点、终点-终点），
 * 并返回最小距离。函数通过 `pwhich` 指针返回是哪一对端点之间的距离最小。
 *
 * @param ls1 第一条线段
 * @param ls2 第二条线段
 * @param pwhich 用于返回哪对端点距离最小的指针（SS 表示起点-起点，SE 表示起点-终点，ES 表示终点-起点，EE 表示终点-终点）
 * @return double 两条线段之间的最小距离
 */
double EDLines::ComputeMinDistanceBetweenTwoLines(LineSegment *ls1, LineSegment *ls2, int *pwhich) {
	// 计算第一对端点之间的距离（起点-起点）
	double dx = ls1->sx - ls2->sx;
	double dy = ls1->sy - ls2->sy;
	double d = sqrt(dx * dx + dy * dy); // 起点-起点的距离
	double min = d; // 初始化最小距离
	int which = SS; // 起点-起点（SS）

	// 计算第二对端点之间的距离（起点-终点）
	dx = ls1->sx - ls2->ex;
	dy = ls1->sy - ls2->ey;
	d = sqrt(dx * dx + dy * dy); // 起点-终点的距离
	if (d < min) { // 如果距离更小，更新最小距离和标志
		min = d;
		which = SE; // 起点-终点（SE）
	}

	// 计算第三对端点之间的距离（终点-起点）
	dx = ls1->ex - ls2->sx;
	dy = ls1->ey - ls2->sy;
	d = sqrt(dx * dx + dy * dy); // 终点-起点的距离
	if (d < min) { // 如果距离更小，更新最小距离和标志
		min = d;
		which = ES; // 终点-起点（ES）
	}

	// 计算第四对端点之间的距离（终点-终点）
	dx = ls1->ex - ls2->ex;
	dy = ls1->ey - ls2->ey;
	d = sqrt(dx * dx + dy * dy); // 终点-终点的距离
	if (d < min) { // 如果距离更小，更新最小距离和标志
		min = d;
		which = EE; // 终点-终点（EE）
	}

	// 如果 pwhich 不为空，返回最小距离对应的端点组合
	if (pwhich) *pwhich = which;

	// 返回最小的距离
	return min;
}

/**
 * @brief 更新线段的参数，根据起点和终点计算线段的方程
 *
 * 该函数根据线段的起点和终点，计算线段的斜率和截距，更新线段的方程形式。
 * 如果线段更接近水平线，则使用 y = a + bx 形式；如果线段更接近垂直线，则使用 x = a + by 形式。
 *
 * @param ls 指向需要更新参数的线段
 */
void EDLines::UpdateLineParameters(LineSegment *ls) {
    // 计算线段两端点的水平和垂直距离
    double dx = ls->ex - ls->sx;
    double dy = ls->ey - ls->sy;

    // 判断是否更接近水平线
    if (fabs(dx) >= fabs(dy)) {
        ls->invert = 0; // 使用 y = a + bx 形式

        // 如果线段几乎水平
        if (fabs(dy) < 1e-3) {
            ls->b = 0; // 斜率为 0，表示完全水平
            ls->a = (ls->sy + ls->ey) / 2; // 取 y 坐标的平均值作为截距
        }
        // 否则，计算斜率和截距
        else {
            ls->b = dy / dx; // 计算斜率 b = dy / dx
            ls->a = ls->sy - (ls->b) * ls->sx; // 计算截距 a = sy - b * sx
        }
    }
    // 如果更接近垂直线
    else {
        ls->invert = 1; // 使用 x = a + by 形式

        // 如果线段几乎垂直
        if (fabs(dx) < 1e-3) {
            ls->b = 0; // 斜率为 0，表示完全垂直
            ls->a = (ls->sx + ls->ex) / 2; // 取 x 坐标的平均值作为截距
        }
        // 否则，计算斜率和截距
        else {
            ls->b = dx / dy; // 计算斜率 b = dx / dy
            ls->a = ls->sx - (ls->b) * ls->sy; // 计算截距 a = sx - b * sy
        }
    }
}