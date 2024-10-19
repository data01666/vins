#include "../include/feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {
        // 更新每个特征点的使用次数
        it.used_num = it.feature_per_frame.size();

        // 如果特征点在多个帧中被使用并且起始帧在窗口大小范围内，则计数
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

int FeatureManager::getLineFeatureCount()
{
    int cnt = 0;

    // 遍历所有线特征
    for (auto &it : line_feature)
    {
        // 记录每个线特征被使用的次数
        it.used_num = it.feature_per_frame.size();

        // 如果线特征至少被使用 2 次且起始帧不在窗口末端，计数
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }

    // 只返回线特征的数量
    return cnt;
}

bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("输入特征数量: %d", (int)image.size()); // 打印当前帧的特征点数量
    ROS_DEBUG("特征点总数: %d", getFeatureCount());    // 打印所有特征点的数量
    double parallax_sum = 0; // 累计视差和
    int parallax_num = 0;    // 累计计算视差的特征数
    last_track_num = 0;      // 当前帧中继续被追踪的特征点数

    // 遍历当前帧的每个特征点
    for (auto &id_pts : image)
    {
        // 将特征点信息存储到 FeaturePerFrame 对象中，包括归一化坐标和其他信息
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        int feature_id = id_pts.first; // 提取特征点的ID
        // 在滑窗中查找当前特征点是否已经存在
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          { return it.feature_id == feature_id; });

        // 如果当前特征点是新特征，添加到特征列表中
        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        // 如果特征点已存在，更新特征信息并增加追踪计数
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
        }
    }

    // 如果帧数少于2没达到滑窗总数或追踪到的特征点少于20，那么说明次新帧是关键帧，marg_old
    if (frame_count < 2 || last_track_num < 20)
        return true;

    // 计算相邻帧之间的视差，用于判断是否边缘化
    for (auto &it_per_id : feature)
    {
        // 如果当前特征点在当前帧-2以前出现过而且至少在当前帧-1还在，那么他就是平行特征点
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    // 如果没有足够的视差特征点或平均视差小于阈值，则选择边缘化
    if (parallax_num == 0) // 如果没有符合条件的视差特征点
    {
        return true;
    }
    else
    {
        ROS_DEBUG("视差总和: %lf, 视差特征点数: %d", parallax_sum, parallax_num);
        ROS_DEBUG("当前平均视差: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num < MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}
void FeatureManager::setLineDepth(const VectorXd &x)
{
    int line_feature_index = -1;
    for (auto &it_per_id : line_feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // 设置起点和终点的深度
        line_feature_index++;  // 递增一次索引
        it_per_id.estimated_depth_start = 1.0 / x(2 * line_feature_index);      // 更新起点的深度
        it_per_id.estimated_depth_end = 1.0 / x(2 * line_feature_index + 1);    // 更新终点的深度

        // 检查起点和终点的深度是否有效
        if (it_per_id.estimated_depth_start < 0 || it_per_id.estimated_depth_end < 0)
        {
            it_per_id.solve_flag = 2;  // 线特征解算失败
        }
        else
        {
            it_per_id.solve_flag = 1;  // 线特征解算成功
        }
    }
}


void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::removeLineFailures()// todo:新增线特征移除失败函数
{
    for (auto it = line_feature.begin(), it_next = line_feature.begin();
         it != line_feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2) // 判断线特征的求解状态是否为失败
        {
            line_feature.erase(it); // 移除失败的线特征
        }
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

VectorXd FeatureManager::getLineDepthVector()
{
    // 获取线特征的总数，每个线特征包含两个端点
    VectorXd dep_vec(2 * getLineFeatureCount()); // 只考虑每个线特征的数量

    int line_feature_index = -1;

    // 遍历所有线特征
    for (auto &it_per_id : line_feature)
    {
        // 记录线特征被使用的次数
        it_per_id.used_num = it_per_id.feature_per_frame.size();

        // 如果线特征被使用至少 2 次且起始帧不在窗口末端，则处理该线特征
        if (it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)
            continue;
#if 1
        dep_vec(++line_feature_index) = 1. / it_per_id.estimated_depth_start;
        dep_vec(++line_feature_index) = 1. / it_per_id.estimated_depth_end;
#else
        dep_vec(++line_feature_index) = it_per_id->estimated_depth_start;
        dep_vec(++line_feature_index) = it_per_id->estimated_depth_end;
#endif
    }
    return dep_vec;
}

void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    // 遍历所有特征点
    for (auto &it_per_id : feature)
    {
        // 计算特征点被观测到的次数
        it_per_id.used_num = it_per_id.feature_per_frame.size();

        // 如果该特征点至少被两帧以上观测到，且起始帧在窗口中（非末两帧），继续处理
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // 如果已经估计了深度，则跳过该特征点
        if (it_per_id.estimated_depth > 0)
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        // 系统默认只有一个相机，所以这里是断言只有一个相机
        ROS_ASSERT(NUM_OF_CAM == 1);

        // 构建 SVD 矩阵，用于三角化求解特征点的深度
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        // 设置第一帧的投影矩阵 P0
        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        // 遍历该特征点被观测到的所有帧
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            // 设置每一帧的投影矩阵
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;

            // 获取特征点在该帧下的归一化坐标
            Eigen::Vector3d f = it_per_frame.point.normalized();

            // 构建 SVD 矩阵的行
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }

        // 断言 SVD 矩阵的行数是否正确
        ROS_ASSERT(svd_idx == svd_A.rows());

        // 通过 SVD 求解特征点的深度
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];

        // 将求得的深度值存储到特征点中
        it_per_id.estimated_depth = svd_method;

        // 如果深度小于某个阈值，则将深度初始化为默认深度
        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }
    }
}
void FeatureManager::triangulateLinepoint(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : line_feature)
    {
        // 起点三角化
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        // Triangulate start point
        Eigen::MatrixXd svd_A_start(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx_start = 0;

        Eigen::Matrix<double, 3, 4> P0_start;
        Eigen::Vector3d t0_start = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0_start = Rs[imu_i] * ric[0];
        P0_start.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0_start.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d t1_start = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1_start = Rs[imu_j] * ric[0];
            Eigen::Vector3d t_start = R0_start.transpose() * (t1_start - t0_start);
            Eigen::Matrix3d R_start = R0_start.transpose() * R1_start;

            Eigen::Matrix<double, 3, 4> P_start;
            P_start.leftCols<3>() = R_start.transpose();
            P_start.rightCols<1>() = -R_start.transpose() * t_start;

            Eigen::Vector3d f_start = it_per_frame.start_point.normalized();
            svd_A_start.row(svd_idx_start++) = f_start[0] * P_start.row(2) - f_start[2] * P_start.row(0);
            svd_A_start.row(svd_idx_start++) = f_start[1] * P_start.row(2) - f_start[2] * P_start.row(1);
        }

        Eigen::Vector4d svd_V_start = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A_start, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_depth_start = svd_V_start[2] / svd_V_start[3];
        it_per_id.estimated_depth_start = svd_depth_start;

        // 终点三角化（类似于起点）
        imu_j = imu_i - 1;
        Eigen::MatrixXd svd_A_end(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx_end = 0;

        Eigen::Matrix<double, 3, 4> P0_end;
        Eigen::Vector3d t0_end = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0_end = Rs[imu_i] * ric[0];
        P0_end.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0_end.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d t1_end = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1_end = Rs[imu_j] * ric[0];
            Eigen::Vector3d t_end = R0_end.transpose() * (t1_end - t0_end);
            Eigen::Matrix3d R_end = R0_end.transpose() * R1_end;

            Eigen::Matrix<double, 3, 4> P_end;
            P_end.leftCols<3>() = R_end.transpose();
            P_end.rightCols<1>() = -R_end.transpose() * t_end;

            Eigen::Vector3d f_end = it_per_frame.end_point.normalized();
            svd_A_end.row(svd_idx_end++) = f_end[0] * P_end.row(2) - f_end[2] * P_end.row(0);
            svd_A_end.row(svd_idx_end++) = f_end[1] * P_end.row(2) - f_end[2] * P_end.row(1);
        }

        Eigen::Vector4d svd_V_end = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A_end, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_depth_end = svd_V_end[2] / svd_V_end[3];
        it_per_id.estimated_depth_end = svd_depth_end;
    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    // 选择滑窗内倒数第二帧（frame_i）和倒数第三帧（frame_j），
    // 用于计算它们之间的视差
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0; // 用于存储计算得到的视差值
    Vector3d p_j = frame_j.point;

    // 提取倒数第三帧中点的归一化坐标 (u_j, v_j)
    double u_j = p_j(0);
    double v_j = p_j(1);

    // 提取倒数第二帧的点
    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    // 对于单目情况，此处不进行视差补偿，直接将 p_i 赋给 p_i_comp
    p_i_comp = p_i;

    // 计算归一化坐标 u_i 和 v_i
    double dep_i = p_i(2); // 深度值（假设深度值为 1）
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j; // 原始视差计算

    // 计算经过补偿后的归一化坐标 u_i_comp 和 v_i_comp
    double dep_i_comp = p_i_comp(2); // 补偿后的深度值
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j; // 补偿后的视差计算

    // 取两者中较小的视差值，求平方和的平方根，更新 ans
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans; // 返回计算得到的视差值
}
