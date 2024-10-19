#include "../include/initial.h"
#include "../include/estimator.h"

bool Initial::initialStructure(Estimator *estimator)
{
    // 1. 检查IMU可观测性
    // 如果IMU不可观测，则返回false，跳过初始化
    // TicToc t_sfm;
    // if (!checkIMUObservibility())
    //     return false;

    // 2. 构建初始化所需的SFM（结构从运动）特征
    vector<SFMFeature> sfm_f;
    buildSFMFeature(estimator, sfm_f);

    // 3. 在滑动窗口中找到平均视差大于阈值的帧i
    // 并计算其与最新帧（第11帧）之间的相对旋转R和平移T
    Matrix3d relative_R;
    Vector3d relative_T;
    int l; // 满足与最新帧视差关系的那一帧的编号
    if (!relativePose(estimator, relative_R, relative_T, l))
    {
        // 如果没有足够的特征或视差不够大，则提醒用户移动设备，以增加视差
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }

    // 4. 解决SFM问题，获取所有帧和路标在帧l坐标系中的旋转和平移
    // 容量是frame_count + 1，滑窗容量为10，加上最新帧共11帧
    GlobalSFM sfm;
    Quaterniond Q[estimator->frame_count + 1];
    Vector3d T[estimator->frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;

    // 如果SFM构建失败，返回false，标记边缘化旧帧
    if(!sfm.construct(estimator->frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        estimator->marginalization_flag = estimator->MARGIN_OLD;
        return false;
    }

    // 5. 为所有帧解决PNP问题，优化相机位姿
    if(!solvePnPForAllFrame(estimator, Q, T, sfm_tracked_points))
        return false;

    // 6. 对齐视觉与IMU数据，完成初始化
    if (visualInitialAlign(estimator))
        return true;
    else
    {
        // 如果对齐失败，返回false
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
}

bool Initial::checkIMUObservibility(Estimator *estimator)
{
    map<double, ImageFrame>::iterator frame_it;
    // 第一次循环，求出滑窗内的平均线加速度
    Vector3d sum_g;
    for (frame_it = estimator->all_image_frame.begin(), frame_it++; frame_it != estimator->all_image_frame.end(); frame_it++)
    {
        double dt = frame_it->second.pre_integration->sum_dt;
        Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
        sum_g += tmp_g;
    }
    Vector3d aver_g;
    aver_g = sum_g * 1.0 / ((int)estimator->all_image_frame.size() - 1);
    // 第二次循环，求出滑窗内的线加速度的标准差
    double var = 0;
    for (frame_it = estimator->all_image_frame.begin(), frame_it++; frame_it != estimator->all_image_frame.end(); frame_it++)
    {
        double dt = frame_it->second.pre_integration->sum_dt;
        Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
        var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);//计算加速度的方差
        //cout << "frame g " << tmp_g.transpose() << endl;
    }
    var = sqrt(var / ((int)estimator->all_image_frame.size() - 1));//计算加速度的标准差
    //ROS_WARN("IMU variation %f!", var);
    if(var < 0.25)
    {
        ROS_INFO("IMU excitation not enouth!");
        return false;
    }
    return true;
}

void Initial::buildSFMFeature(Estimator *estimator, vector<SFMFeature> &sfm_f)
{
    // 遍历滑动窗口中的所有特征点
    for (auto &it_per_id : estimator->f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1; // 获取特征点在滑动窗口中的起始帧编号
        SFMFeature tmp_feature;
        tmp_feature.state = false; // 初始化状态，标记特征点尚未三角化
        tmp_feature.id = it_per_id.feature_id; // 设置特征点ID

        // 遍历该特征点在不同帧的观测
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++; // 增加帧编号
            Vector3d pts_j = it_per_frame.point; // 获取特征点在当前帧的归一化坐标
            tmp_feature.observation.push_back(
                make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()})); // 将当前帧编号及对应坐标加入观测

            // 在tmp_feature.observation中，每个pair是该特征在不同帧的归一化坐标
        }

        sfm_f.push_back(tmp_feature); // 将当前特征点的观测数据加入到sfm_f中
    }
    /*
    * 为什么需要构造一个新的数据结构sfm_f？
    * f_manager包含了所有的特征数据及大量像素信息，而sfm_f专门用于SfM，因此更加简化，仅包含SfM所需的数据。
    * 作者设计了一个新的数据结构sfm_f，以便为SfM任务提供更专业化的数据支持。
    */
}

bool Initial::relativePose(Estimator *estimator, Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // 在滑动窗口中寻找具有足够对应点和视差的前一帧
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        // 找到第i帧与当前帧之间的特征点对应关系，返回归一化坐标配对
        corres = estimator->f_manager.getCorresponding(i, WINDOW_SIZE);

        // 如果找到的对应特征点数目超过20，继续计算
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;

            // 计算平均视差
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1)); // 第i帧中的特征点坐标
                Vector2d pts_1(corres[j].second(0), corres[j].second(1)); // 当前帧中的特征点坐标
                double parallax = (pts_0 - pts_1).norm(); // 计算两个特征点间的视差
                sum_parallax += parallax;
            }

            // 计算并判断平均视差是否满足初始化条件
            average_parallax = sum_parallax / int(corres.size());
            if (average_parallax * 460 > 30 &&
                estimator->m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                // 如果满足视差条件，使用solveRelativeRT求解相对旋转和平移
                l = i; // 设置参考帧的编号为i
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true; // 找到合适的参考帧并返回
            }
        }
    }
    return false; // 若未找到满足条件的参考帧，返回false
}

bool Initial::solvePnPForAllFrame(Estimator *estimator, Quaterniond Q[], Vector3d T[], map<int, Vector3d> &sfm_tracked_points)
{
    // 遍历所有图像帧
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = estimator->all_image_frame.begin();
    for (int i = 0; frame_it != estimator->all_image_frame.end(); frame_it++)
    {
        // 提供初始估计值
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == estimator->Headers[i].stamp.toSec())
        {
            // 当前帧时间戳匹配，则直接设置为关键帧
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > estimator->Headers[i].stamp.toSec())
        {
            i++;
        }

        // 计算初始旋转和位移矩阵
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec); // 使用罗德里格斯公式将旋转矩阵转换为旋转向量
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;

        // 将每个特征点的3D世界坐标和图像坐标添加到向量中
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);

                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }

        // 检查特征点数目，至少需要6个点
        if(pts_3_vector.size() < 6)
        {
            ROS_DEBUG("Not enough points for solve pnp!");
            return false;
        }

        // 使用PnP算法求解位姿
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, true))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }

        // 将旋转向量转换为旋转矩阵并计算相机位姿
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();

        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);

        // 存储PnP求解后的位姿到图像帧中
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    return true;
}

bool Initial::visualInitialAlign(Estimator *estimator)
{
    // 计算陀螺仪偏置、尺度、重力加速度和速度
    TicToc t_g;
    VectorXd x;

    // 通过视觉-惯性对齐来求解尺度因子和重力加速度
    bool result = VisualIMUAlignment(estimator->all_image_frame, estimator->Bgs, estimator->g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false; // 如果失败，则返回false
    }

    // 从初始化结果恢复状态值（包括速度和其他状态量）
    recoverStatusValuesFromInitial(estimator, x);

    ROS_DEBUG_STREAM("g0     " << estimator->g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(estimator->Rs[0]).transpose());

    return true;
}

void Initial::recoverStatusValuesFromInitial(Estimator *estimator, const VectorXd &x)
{
    for (int i = 0; i <= estimator->frame_count; i++)
    {
        Matrix3d Ri = estimator->all_image_frame[estimator->Headers[i].stamp.toSec()].R;
        Vector3d Pi = estimator->all_image_frame[estimator->Headers[i].stamp.toSec()].T;
        estimator->Ps[i] = Pi;
        estimator->Rs[i] = Ri;
        estimator->all_image_frame[estimator->Headers[i].stamp.toSec()].is_key_frame = true;
    }

    VectorXd dep = estimator->f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    estimator->f_manager.clearDepth(dep);

    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    estimator->ric[0] = RIC[0];
    estimator->f_manager.setRic(estimator->ric);
    estimator->f_manager.triangulate(estimator->Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        estimator->pre_integrations[i]->repropagate(Vector3d::Zero(), estimator->Bgs[i]);
    }

    for (int i = estimator->frame_count; i >= 0; i--)
        estimator->Ps[i] = s * estimator->Ps[i] - estimator->Rs[i] * TIC[0] - (s * estimator->Ps[0] - estimator->Rs[0] * TIC[0]);

    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = estimator->all_image_frame.begin(); frame_i != estimator->all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            estimator->Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    for (auto &it_per_id : estimator->f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    Matrix3d R0 = Utility::g2R(estimator->g);
    double yaw = Utility::R2ypr(R0 * estimator->Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    estimator->g = R0 * estimator->g;

    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= estimator->frame_count; i++)
    {
        estimator->Ps[i] = rot_diff * estimator->Ps[i];
        estimator->Rs[i] = rot_diff * estimator->Rs[i];
        estimator->Vs[i] = rot_diff * estimator->Vs[i];
    }
}