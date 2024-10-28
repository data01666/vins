#include "../include/feature_tracker.h"

int FeatureTracker::n_id = 0;//特征点 ID 从 0 开始递增
int FeatureTracker::line_n_id = 0;//特征点 ID 从 0 开始递增

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}
void reduceVector(vector<LineSegment> &v, vector<uchar> &status)
{
    // 使用 std::remove_if 和 lambda 函数移除无效的线段
    v.erase(std::remove_if(v.begin(), v.end(),
                [&status, i = 0](const LineSegment&) mutable {
                    return !status[i++];  // 通过 status[i] 判断是否要移除
                }),
            v.end());
}

// the following function belongs to FeatureTracker Class
FeatureTracker::FeatureTracker() {}

void FeatureTracker::equalize(const cv::Mat &_img,cv::Mat &img)
{
    if (EQUALIZE)
    {
        // 创建CLAHE对象，对比度限制参数3.0，网格尺寸8x8，将图像划分为多个小块，对每个小块进行直方图均衡化
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        // 创建一个计时器对象，用于统计CLAHE操作所耗费的时间
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;
}

void FeatureTracker::flowTrack()
{
    // 判断是否有特征点
    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        // calcOpticalFlowPyrLK函数，它是Pyramidal Lucas-Kanade光流算法的实现
        // cur_img：当前图像帧; forw_img：下一帧; cur_pts：当前特征点; forw_pts：输出参数：下一帧特征点;
        // status：输出参数，特征点是否被成功跟踪; err：输出参数，特征点跟踪误差
        // cv::Size(21, 21)：光流窗口大小; 3：金字塔的层数
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }
}

void FeatureTracker::lineflowTrack()
{
    if (PUB_THIS_FRAME)
    {
        if (cur_line_segments.size() > 0)
    {
        TicToc t_o;
        std::vector<uchar> lstatus(cur_line_segments.size(), 1); // 初始化线段的状态，默认值为1（有效）

        for (int i = 0; i < cur_line_segments.size(); i++)
        {
            std::vector<cv::Point2f> cur_keypoints;

            // 将当前帧线段关键点从 std::pair 转换为 cv::Point2f
            for (const auto& kp : cur_line_segments[i].keyPoints)
            {
                cur_keypoints.push_back(cv::Point2f(static_cast<float>(kp.first), static_cast<float>(kp.second)));
            }

            std::vector<cv::Point2f> forw_keypoints;
            std::vector<float> err;
            std::vector<uchar> status; // 用于记录线段每个关键点的状态

            // 对当前帧的线段关键点进行光流跟踪
            cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_keypoints, forw_keypoints, status, err, cv::Size(21, 21), 3);

            // 判断关键点跟踪成功率
            int tracked_count = 0;
            for (int j = 0; j < status.size(); j++)
            {
                if (status[j])
                    tracked_count++;
            }

            double success_ratio = (double)tracked_count / cur_keypoints.size();
            /*
            if (success_ratio > 0.7)  // 成功率高，至少70%的关键点被成功跟踪
            {
                std::vector<std::pair<double, double>> tracked_keypoints;
                for (int j = 0; j < status.size(); j++)
                {
                    if (status[j])
                    {
                        tracked_keypoints.push_back({ forw_keypoints[j].x, forw_keypoints[j].y });
                    }
                }

                // 将新的线段关键点添加到 forward 线段容器
                LineSegment new_segment = cur_line_segments[i];
                new_segment.keyPoints = tracked_keypoints;
                forw_line_segments.push_back(new_segment);
            }
            else
            {
                // 如果成功率太低，使用DTW进行验证
                LineSegment new_segment = cur_line_segments[i];

                // 对当前帧的线段关键点与前一帧进行DTW匹配
                double dtw_distance = computeDTW(cur_line_segments[i].keyPoints, new_segment.keyPoints);
                double DTW_THRESHOLD = 30;

                if (dtw_distance > DTW_THRESHOLD)
                {
                    // 如果DTW距离较大，标记为无效
                    lstatus[i] = 0;
                }
                else
                {
                    // 如果DTW验证通过，则将该线段添加到forward线段集合
                    forw_line_segments.push_back(new_segment);
                }
            }*/

            if (success_ratio < 0.3)  // 成功率低，至多30%的关键点被成功跟踪
            {
                lstatus[i] = 0;
            }
            else
            {
                std::vector<std::pair<double, double>> tracked_keypoints;
                for (int j = 0; j < status.size(); j++)
                {
                    if (status[j])
                    {
                        tracked_keypoints.push_back({ forw_keypoints[j].x, forw_keypoints[j].y });
                    }
                }

                // 将新的线段关键点添加到 forward 线段容器
                LineSegment new_segment = cur_line_segments[i];
                new_segment.keyPoints = tracked_keypoints;

                // 对当前帧的线段关键点与前一帧进行DTW匹配
                double dtw_distance = computeDTW(cur_line_segments[i].keyPoints, new_segment.keyPoints);
                double DTW_THRESHOLD = 30;

                if (dtw_distance > DTW_THRESHOLD)
                {
                    // 如果DTW距离较大，标记为无效
                    lstatus[i] = 0;
                }
                else
                {
                    // 如果DTW验证通过，则将该线段添加到forward线段集合
                    forw_line_segments.push_back(new_segment);
                }
            }
        }

        // 根据跟踪状态更新线段
        reduceVector(prev_line_segments, lstatus);
        reduceVector(cur_line_segments, lstatus);
        reduceVector(forw_line_segments, lstatus);
        reduceVector(line_ids, lstatus);

        ROS_DEBUG("line optical flow tracking costs: %fms", t_o.toc());
    }
    }

}

double FeatureTracker::computeDTW(const std::vector<std::pair<double, double>>& cur_keypoints,
                                  const std::vector<std::pair<double, double>>& forw_keypoints)
{
    int n = cur_keypoints.size();
    int m = forw_keypoints.size();

    // 创建 DTW 距离矩阵
    std::vector<std::vector<double>> dtw(n + 1, std::vector<double>(m + 1, std::numeric_limits<double>::infinity()));

    // 初始化起点
    dtw[0][0] = 0.0;

    // 计算 DTW 距离
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            double cost = sqrt(pow(cur_keypoints[i - 1].first - forw_keypoints[j - 1].first, 2) +
                               pow(cur_keypoints[i - 1].second - forw_keypoints[j - 1].second, 2));

            dtw[i][j] = cost + std::min({ dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1] });
        }
    }
    // 返回最终的 DTW 距离
    return dtw[n][m];
}

void FeatureTracker::trackNew()
{
    // 当需要在当前帧中发布数据时才执行基础的特征点筛选与跟踪操作，减少计算负担、避免冗余数据、提高特征点跟踪的效率
    if (PUB_THIS_FRAME)
    {
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        // 需要检测的特征点数量等于最大特征点数量减去前向帧中已有的特征点数量
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            // 检查遮罩是否已初始化
            if(mask.empty())
                cout << "mask is empty " << endl;
            // 检查遮罩是否为单通道的8位图像
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            // 检查遮罩的尺寸是否与当前处理帧的尺寸相同
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            // 调用 OpenCV 函数 cv::goodFeaturesToTrack 来检测新特征点,该函数是基于Shi-Tomasi角点检测算法
            // forw_img：前向图像帧; n_pts：输出参数，检测到的新特征点; n_max_cnt：需要检测的特征点数量;
            // 0.01：角点检测的质量水平，值越大，检测到的角点越多; MIN_DIST：角点之间的最小距离
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
            addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
}

// 设置线特征的遮罩，并且添加新的线段
void FeatureTracker::trackNewlines()
{
    // 当需要在当前帧中发布数据时才执行基础的特征线筛选与跟踪操作
    if (PUB_THIS_FRAME)
    {
        // 遮罩初始化
        if (FISHEYE)
            mask = fisheye_mask.clone();
        else
            mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255)); // 初始化遮罩

        ROS_DEBUG("detect new line features begins");
        TicToc t_l;

        // 使用 EDLines 检测新线段
        EDLines lines = EDLines(forw_img);
        line_pts = lines.getLines(); // 将检测到的新线段加入到 line_pts 中

        // 设置遮罩并筛选线段
        for (const auto &prev_line : prev_line_map)
        {
            const LineSegment &line = prev_line.second; // 获取线段
            cv::Point2f start(static_cast<float>(line.sx), static_cast<float>(line.sy));
            cv::Point2f end(static_cast<float>(line.ex), static_cast<float>(line.ey));

            // 检查起点和终点是否在图像范围内
            if (start.x >= 0 && start.x < mask.cols && start.y >= 0 && start.y < mask.rows &&
                end.x >= 0 && end.x < mask.cols && end.y >= 0 && end.y < mask.rows)
            {
                // 在遮罩中对该线段的起点和终点占用此区域
                cv::line(mask, start, end, cv::Scalar(0), MIN_DIST);
            }
        }

        for (auto &new_line : line_pts)
        {
            cv::Point2f start(static_cast<float>(new_line.sx), static_cast<float>(new_line.sy));
            cv::Point2f end(static_cast<float>(new_line.ex), static_cast<float>(new_line.ey));

            // 检查起点和终点是否在图像范围内
            if (start.x >= 0 && start.x < mask.cols && start.y >= 0 && start.y < mask.rows &&
                end.x >= 0 && end.x < mask.cols && end.y >= 0 && end.y < mask.rows)
            {
                // 判断起点和终点是否在遮罩区域内
                if (mask.at<uchar>(start) == 255 && mask.at<uchar>(end) == 255)
                {
                    // 如果该线段的起点和终点均未被遮罩占用，将其加入当前帧的线段集合
                    forw_line_segments.push_back(new_line);
                    // 在遮罩中对该线段的起点和终点占用此区域
                    cv::line(mask, start, end, cv::Scalar(0), MIN_DIST);
                    // 记录线段的 ID
                    line_ids.push_back(-1);
                }
            }
            else
            {
                // 如果起点或终点不在图像范围内，可以选择记录日志或跳过处理
                std::cout << "Line start or end point out of bounds" << std::endl;
            }
        }
        ROS_DEBUG("detect new line features costs: %fms", t_l.toc());
    }
}

// 主要目的是根据跟踪特征点的历史信息设置遮罩区域，防止特征点过于密集。
// 同时它还优先保留那些已经被跟踪较长时间的特征点。
void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    // 创建了一个 vector 对象 cnt_pts_id，用于存储特征点的跟踪次数、特征点坐标和特征点 ID
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    // 遍历 forw_pts 中的每一个特征点，将其跟踪次数、特征点坐标和特征点 ID 存储到 cnt_pts_id 中
    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        // 检查特征点是否在遮罩区域内。
        // 初始时，遮罩区域内的值为255，表示该区域可用。
        // 我们按照跟踪次数对特征点进行排序，优先保留那些跟踪次数多的特征点。
        if (mask.at<uchar>(it.second.first) == 255)
        {
            // 如果该特征点位置的遮罩值为255，表示该区域尚未被占用。
            // 将该特征点的坐标添加到前向特征点集合(forw_pts)中。
            forw_pts.push_back(it.second.first);

            // 将该特征点的ID添加到ID集合(ids)中。
            ids.push_back(it.second.second);

            // 将该特征点的跟踪次数添加到跟踪计数集合(track_cnt)中。
            track_cnt.push_back(it.first);

            // 使用cv::circle()在特征点的位置画一个半径为MIN_DIST的圆，
            // 将该圆内的像素值设为0，表示该区域已占用，不再允许检测新特征点。
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    // 图像直方图均衡化
    equalize(_img, img);

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    // 清空当前帧的特征点和线段
    forw_pts.clear();
    forw_line_segments.clear();

    // 点特征的光流跟踪
    flowTrack();

    // 已跟踪的点特征计数器加一
    for (auto &n : track_cnt)
        n++;

    // 线特征的光流跟踪
    lineflowTrack();

    // 提取新的点和线特征
    trackNew();
    trackNewlines();

    // 更新上一帧的图像和特征信息
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_line_segments = cur_line_segments;
    cur_img = forw_img;
    cur_pts = forw_pts;
    cur_line_segments = forw_line_segments;

    // 特征点线去畸变
    undistortedPoints();
    undistortedLines();
    prev_time = cur_time;
}

// 通过基础矩阵（Fundamental Matrix）和RANSAC算法对特征点匹配进行筛选
void FeatureTracker::rejectWithF()
{
    // 先检查 forw_pts.size() 是否大于或等于 8，因为计算基础矩阵需要至少 8 对匹配点。
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        // 遍历当前帧和前向帧的特征点，将它们转换为归一化图像坐标系下的特征点
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            // 将当前帧特征点从二维像素坐标投影到三维归一化平面
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        // 使用RANSAC算法计算基础矩阵并筛选特征点，status存储每个点是否符合基础矩阵模型
        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        // 保存原始特征点数量，用于后续比较
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        // 输出RANSAC筛选前后特征点数量的变化及通过率
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i, bool isLineSegment)
{
    if (isLineSegment)
    {
        // 如果是处理线段的情况
        if (i < line_ids.size())
        {
            if (line_ids[i] == -1)  // 线段 ID 为 -1 时进行全局 ID 分配
            {
                line_ids[i] = line_n_id++;  // 使用全局线段 ID 自增
                return true;  // 更新成功，返回 true
            }
        }
        else
            return false;
    }
    else
    {
        // 处理点特征的情况
        if (i < ids.size())
        {
            if (ids[i] == -1)  // 点特征 ID 为 -1 时进行分配
                ids[i] = n_id++;
            // 更新成功，返回 true
            return true;
        }
        else
            return false;
    }
    //return false;  // 如果索引超出范围或 ID 已经分配，返回 false

}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    // 创建一个大小为 ROW + 600, COL + 600 的灰度图，初始化所有像素值为 0
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    // 定义两个向量分别存储畸变和去畸变后的二维坐标点
    vector<Eigen::Vector2d> distortedp, undistortedp;
    // 遍历图像中的每个像素点（列i和行j），将其从像素坐标系投影到归一化平面坐标系
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            // 将当前像素点的坐标存储到二维向量 a 中
            Eigen::Vector2d a(i, j);
            // 创建三维向量 b 用于存储投影到去畸变平面的坐标，X 和 Y 是归一化坐标，Z 通常用来保持齐次坐标的形式
            Eigen::Vector3d b;
            // 调用相机模型的 liftProjective 方法将像素点从畸变坐标系投影到去畸变的归一化平面上
            m_camera->liftProjective(a, b);
            // 将畸变前和去畸变后的点分别存储到两个向量中
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));// 归一化坐标，去畸变后的点
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    // 遍历所有去畸变后的点，将其投影到图像坐标系，并将其像素值赋给去畸变图像
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        // 创建一个 3x1 的矩阵 pp，用于存储去畸变后的点在相机图像坐标中的位置
        cv::Mat pp(3, 1, CV_32FC1);
        // 通过焦距 FOCAL_LENGTH 将去畸变后的归一化坐标转换为像素坐标
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;// 齐次坐标的第三个分量为 1
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));

        // 判断转换后的点是否落在 undistortedImg 图像的有效区域内，如果是，则将其像素值赋给去畸变图像
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    // 清空之前的无畸变点和映射表
    cur_un_pts.clear();// 存储当前帧的无畸变特征点的集合
    cur_un_pts_map.clear();// 存储每个特征点 ID 与对应无畸变特征点的映射关系，用于后续速度计算
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    // 如果存在上一帧的无畸变特征点映射表，则可以根据时间差计算当前帧特征点的速度
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;// 计算时间差
        pts_velocity.clear();// 清空上一帧特征点速度集合
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end()) // prev_un_pts_map.end()表示在上一帧的映射中没有这个特征点的记录
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else // 如果特征点ID无效，速度设为0
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else // 如果不存在上一帧的无畸变特征点映射表，则将速度设为0
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}

void FeatureTracker::undistortedLines() {
    // 清空之前的无畸变线段和映射表
    std::vector<LineSegment> cur_un_lines; // 存储当前帧的无畸变线段集合
    std::map<int, LineSegment> cur_un_lines_map; // 存储每个线段 ID 与对应无畸变线段的映射关系

    // 处理当前帧线段
    for (unsigned int i = 0; i < cur_line_segments.size(); i++) {
        // 创建一个无畸变线段，直接用现有的线段数据初始化
        LineSegment undistorted_line = cur_line_segments[i]; // 直接复制

        // 处理线段的起点和终点
        Eigen::Vector2d start_a(cur_line_segments[i].sx, cur_line_segments[i].sy);
        Eigen::Vector3d start_b;
        m_camera->liftProjective(start_a, start_b);
        undistorted_line.sx = start_b.x() / start_b.z();
        undistorted_line.sy = start_b.y() / start_b.z();

        Eigen::Vector2d end_a(cur_line_segments[i].ex, cur_line_segments[i].ey);
        Eigen::Vector3d end_b;
        m_camera->liftProjective(end_a, end_b);
        undistorted_line.ex = end_b.x() / end_b.z();
        undistorted_line.ey = end_b.y() / end_b.z();

        // 插入到映射中
        cur_un_lines_map.insert({line_ids[i], undistorted_line});
        cur_un_lines.push_back(undistorted_line); // 添加到当前无畸变线段集合
    }

    // 计算线段速度
    if (!prev_line_map.empty()) {
        double dt = cur_time - prev_time; // 计算时间差
        line_velocity.clear(); // 清空上一帧线段速度集合
        for (unsigned int i = 0; i < cur_line_segments.size(); i++) {
            if (line_ids[i] != -1)
            {
                auto it = prev_line_map.find(line_ids[i]);
                if (it != prev_line_map.end()) {
                    // 计算起点和终点的速度
                    double v_x_start = (cur_un_lines[i].sx - it->second.sx) / dt;
                    double v_y_start = (cur_un_lines[i].sy - it->second.sy) / dt;

                    double v_x_end = (cur_un_lines[i].ex - it->second.ex) / dt;
                    double v_y_end = (cur_un_lines[i].ey - it->second.ey) / dt;

                    // 计算线段的平均速度
                    cv::Point2f avg_velocity((v_x_start + v_x_end) / 2, (v_y_start + v_y_end) / 2);
                    line_velocity.push_back(avg_velocity);
                } else {
                    // 如果上一帧没有对应线段，速度设为0
                    line_velocity.push_back(cv::Point2f(0, 0));
                }
            }
        }
    } else {
        // 如果没有上一帧线段，所有速度设为0
        line_velocity.resize(cur_line_segments.size(), cv::Point2f(0, 0));
    }
    // 更新上一帧的映射表
    prev_line_map = cur_un_lines_map;
}



