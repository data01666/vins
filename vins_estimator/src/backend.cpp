#include "../include/backend.h"
#include "../include/estimator.h" 

Backend::Backend() {clearState();}

void Backend::clearState()
{
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();
}

void Backend::backend(Estimator *estimator)
{
    TicToc t_solve;
    // todo:创建计时器并调用位姿求解函数
    solveOdometry(estimator);
    ROS_DEBUG("solver costs: %fms", t_solve.toc());

    if (failureDetection(estimator))
    {
        ROS_WARN("failure detection!");
        estimator->failure_occur = 1;

        estimator->clearState();
        clearState();

        estimator->setParameter();
        ROS_WARN("system reboot!");
        return;
    }
    // 滑动窗口，用于边缘化旧的状态并保留最新的关键帧信息
    //  todo:需要考虑线段信息
    TicToc t_margin;
    slideWindow(estimator);

    estimator->f_manager.removeFailures();
    // todo:移除在边缘化过程中检测到的无效线特征
    ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
    // prepare output of VINS
    estimator->key_poses.clear();
    for (int i = 0; i <= WINDOW_SIZE; i++)
        estimator->key_poses.push_back(estimator->Ps[i]);

    estimator->last_R = estimator->Rs[WINDOW_SIZE];
    estimator->last_P = estimator->Ps[WINDOW_SIZE];
    estimator->last_R0 = estimator->Rs[0];
    estimator->last_P0 = estimator->Ps[0];
}

void Backend::solveOdometry(Estimator *estimator)
{
    if (estimator->frame_count < WINDOW_SIZE)
        return;
    if (estimator->solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        estimator->f_manager.triangulate(estimator->Ps, estimator->tic, estimator->ric);
        // todo:进行线特征的三角化
        estimator->f_manager.triangulateLinepoint(estimator->Ps, estimator->tic, estimator->ric); // 三角化线特征点
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        optimization(estimator);
    }
}

void Backend::vector2double(Estimator *estimator)
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = estimator->Ps[i].x();
        para_Pose[i][1] = estimator->Ps[i].y();
        para_Pose[i][2] = estimator->Ps[i].z();
        Quaterniond q{estimator->Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = estimator->Vs[i].x();
        para_SpeedBias[i][1] = estimator->Vs[i].y();
        para_SpeedBias[i][2] = estimator->Vs[i].z();

        para_SpeedBias[i][3] = estimator->Bas[i].x();
        para_SpeedBias[i][4] = estimator->Bas[i].y();
        para_SpeedBias[i][5] = estimator->Bas[i].z();

        para_SpeedBias[i][6] = estimator->Bgs[i].x();
        para_SpeedBias[i][7] = estimator->Bgs[i].y();
        para_SpeedBias[i][8] = estimator->Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = estimator->tic[i].x();
        para_Ex_Pose[i][1] = estimator->tic[i].y();
        para_Ex_Pose[i][2] = estimator->tic[i].z();
        Quaterniond q{estimator->ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = estimator->f_manager.getDepthVector();
    for (int i = 0; i < estimator->f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    // 将线特征的起点和终点的逆深度赋值到 para_Line_Feature 数组中
    VectorXd line_dep = estimator->f_manager.getLineDepthVector();
    for (int i = 0; i < estimator->f_manager.getLineFeatureCount(); i++)  // 遍历每个线特征
    {
        // 分别存储起点和终点的逆深度
        para_LineFeature[i][0] = line_dep(2 * i);       // 线特征起点的逆深度
        para_LineFeature[i][1] = line_dep(2 * i + 1);   // 线特征终点的逆深度
    }
   estimator->f_manager.setLineDepth(line_dep);

    if (ESTIMATE_TD)
        para_Td[0][0] = estimator->td;
}

void Backend::double2vector(Estimator *estimator)
{
    Vector3d origin_R0 = Utility::R2ypr(estimator->Rs[0]);
    Vector3d origin_P0 = estimator->Ps[0];

    if (estimator->failure_occur)
    {
        origin_R0 = Utility::R2ypr(estimator->last_R0);
        origin_P0 = estimator->last_P0;
        estimator->failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = estimator->Rs[0] * Quaterniond(para_Pose[0][6],
                                                  para_Pose[0][3],
                                                  para_Pose[0][4],
                                                  para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        estimator->Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        estimator->Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        estimator->Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        estimator->Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        estimator->Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        estimator->tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        estimator->ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = estimator->f_manager.getDepthVector();
    for (int i = 0; i < estimator->f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    estimator->f_manager.setDepth(dep);
    // 处理线特征的逆深度：
    VectorXd line_dep = estimator->f_manager.getLineDepthVector();
    for (int i = 0; i < estimator->f_manager.getLineFeatureCount(); i++)
    {
        line_dep(2 * i) = para_LineFeature[i][0];       // 起点深度
        line_dep(2 * i + 1) = para_LineFeature[i][1];   // 终点深度
    }
    estimator->f_manager.setLineDepth(line_dep);

    if (ESTIMATE_TD)
        estimator->td = para_Td[0][0];

    // relative info between two loop frame
    if(estimator->relocalization_info)
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(estimator->relo_Pose[6], estimator->relo_Pose[3], estimator->relo_Pose[4], estimator->relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(estimator->relo_Pose[0] - para_Pose[0][0],
                                     estimator->relo_Pose[1] - para_Pose[0][1],
                                     estimator->relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(estimator->prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        estimator->drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        estimator->drift_correct_t = estimator->prev_relo_t - estimator->drift_correct_r * relo_t;   
        estimator->relo_relative_t = relo_r.transpose() * (estimator->Ps[estimator->relo_frame_local_index] - relo_t);
        estimator->relo_relative_q = relo_r.transpose() * estimator->Rs[estimator->relo_frame_local_index];
        estimator->relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(estimator->Rs[estimator->relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << estimator->relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<estimator->relo_relative_yaw << endl;
        estimator->relocalization_info = 0;    

    }
}

//TODO
bool Backend::failureDetection(Estimator *estimator)
{
    if (estimator->f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", estimator->f_manager.last_track_num);
        //return true;
    }
    if (estimator->Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", estimator->Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (estimator->Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", estimator->Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = estimator->Ps[WINDOW_SIZE];
    if ((tmp_P - estimator->last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - estimator->last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = estimator->Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * estimator->last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

void Backend::nonLinearOptimization(Estimator *estimator,ceres::Problem &problem,ceres::LossFunction *loss_function)
{
    vector2double(estimator);// 将 Estimator 的变量转换为 Ceres 需要的数组格式

    // 1. 添加参数块（位姿和速度偏置）
    for (int i = 0; i < WINDOW_SIZE + 1; i++)//还包括最新的第11帧
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    // 2. 添加相机外参块
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);

        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);// 若不估计外参，则设置为常量
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    // 3. 添加时间延迟参数块（若需要）
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

    TicToc t_prepare;

    // 4. 添加边缘化因子
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    // 5. 添加 IMU 因子
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (estimator->pre_integrations[j]->sum_dt > 10.0)
            continue;// 跳过异常数据
        IMUFactor* imu_factor = new IMUFactor(estimator->pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    // 6. 添加视觉残差（点特征）
    int f_m_cnt = 0;// 记录添加的视觉测量数量
    int feature_index = -1;// 特征点索引，初始化为 -1
    for (auto &it_per_id : estimator->f_manager.feature)
    {
        // 计算特征点被观测到的次数
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 如果特征点未在至少 2 帧中观测到，或特征点的起始帧接近窗口末端，则跳过该特征点
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        ++feature_index;// 增加特征点索引

        int imu_i = it_per_id.start_frame;  // 特征点的起始帧索引
        int imu_j = imu_i - 1;  // 初始化为上一帧

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;// 获取特征点在起始帧的坐标

        // 遍历特征点在后续各帧中的观测数据
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++; // 增加当前帧索引
            if (imu_i == imu_j)
            {
                // 跳过起始帧，因为它不与自身计算投影误差
                continue;
            }
            Vector3d pts_j = it_per_frame.point;// 获取特征点在当前帧中的坐标
            if (ESTIMATE_TD)
            {
                // 如果需要估计时间延迟，则使用带时间延迟的投影因子
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else
            {
                // 不考虑时间延迟的投影因子
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++; // 增加视觉测量计数
        }
    }
    // 6-1. 添加线特征残差
    // 遍历线特征，将起点和终点作为两个独立的点特征来处理
    int line_feature_index = 0;
    for (auto &it_per_id : estimator->f_manager.line_feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        ++line_feature_index;
        int imu_i = it_per_id.start_frame;
        int imu_j = imu_i - 1;

        // 获取线段的起点和终点
        Vector3d pts_start = it_per_id.feature_per_frame[0].start_point;
        Vector3d pts_end = it_per_id.feature_per_frame[0].end_point;

        for (auto &line_frame : it_per_id.feature_per_frame) {
            imu_j++;
            if (imu_i == imu_j)
                continue;

            const Eigen::Vector3d &start_j = line_frame.start_point;
            const Eigen::Vector3d &end_j = line_frame.end_point;

            if (ESTIMATE_TD) {
                // 对起点计算带时间延迟的重投影误差
                ProjectionTdFactor *start_td_factor = new ProjectionTdFactor(
                    pts_start, start_j,
                    it_per_id.feature_per_frame[0].velocity, line_frame.velocity,
                    it_per_id.feature_per_frame[0].cur_td, line_frame.cur_td,
                    it_per_id.feature_per_frame[0].start_uv.y(), line_frame.start_uv.y()
                );
                problem.AddResidualBlock(start_td_factor, loss_function,
                                         para_Pose[imu_i], para_Pose[imu_j],
                                         para_Ex_Pose[0], &para_Feature[line_feature_index][0], para_Td[0]);

                // 对终点计算带时间延迟的重投影误差
                ProjectionTdFactor *end_td_factor = new ProjectionTdFactor(
                    pts_end, end_j,
                    it_per_id.feature_per_frame[0].velocity, line_frame.velocity,
                    it_per_id.feature_per_frame[0].cur_td, line_frame.cur_td,
                    it_per_id.feature_per_frame[0].end_uv.y(), line_frame.end_uv.y()
                );
                problem.AddResidualBlock(end_td_factor, loss_function,
                                         para_Pose[imu_i], para_Pose[imu_j],
                                         para_Ex_Pose[0], &para_Feature[line_feature_index][1], para_Td[0]);
            } else {
                // 对起点计算普通重投影误差
                ProjectionFactor *start_factor = new ProjectionFactor(pts_start, start_j);
                problem.AddResidualBlock(start_factor, loss_function,
                                         para_Pose[imu_i], para_Pose[imu_j],
                                         para_Ex_Pose[0], &para_Feature[line_feature_index][0]);

                // 对终点计算普通重投影误差
                ProjectionFactor *end_factor = new ProjectionFactor(pts_end, end_j);
                problem.AddResidualBlock(end_factor, loss_function,
                                         para_Pose[imu_i], para_Pose[imu_j],
                                         para_Ex_Pose[0], &para_Feature[line_feature_index][1]);
            }
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    // 7. 添加重定位因子（若有重定位信息）
    if(estimator->relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(estimator->relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : estimator->f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if(start <= estimator->relo_frame_local_index)
            {
                while((int)estimator->match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                if((int)estimator->match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(estimator->match_points[retrive_feature_index].x(), estimator->match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], estimator->relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }
            }
        }
    }

    // 8. 进行 Ceres 优化
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (estimator->marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    double2vector(estimator);// 将优化结果从 Ceres 数组格式转换回 Estimator 格式
}

void Backend::margOld(Estimator *estimator,ceres::LossFunction *loss_function)
{
    MarginalizationInfo *marginalization_info = new MarginalizationInfo();
    
    vector2double(estimator);

    if (last_marginalization_info)
    {
        vector<int> drop_set;
        for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
        {
            if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                drop_set.push_back(i);
        }
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                       last_marginalization_parameter_blocks,
                                                                       drop_set);

        marginalization_info->addResidualBlockInfo(residual_block_info);
    }

    {
        if (estimator->pre_integrations[1]->sum_dt < 10.0)
        {
            IMUFactor* imu_factor = new IMUFactor(estimator->pre_integrations[1]);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }
    }

    {
        int feature_index = -1;
        for (auto &it_per_id : estimator->f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            if (imu_i != 0)
                continue;

            Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i == imu_j)
                    continue;

                Vector3d pts_j = it_per_frame.point;
                if (ESTIMATE_TD)
                {
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                      it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                      it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                   vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                   vector<int>{0, 3});
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
                else
                {
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                   vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                   vector<int>{0, 3});
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }
        }
    }

    TicToc t_pre_margin;
    marginalization_info->preMarginalize();
    ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
    TicToc t_margin;
    marginalization_info->marginalize();
    ROS_DEBUG("marginalization %f ms", t_margin.toc());

    std::unordered_map<long, double *> addr_shift;
    for (int i = 1; i <= WINDOW_SIZE; i++)
    {
        addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
        addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
    if (ESTIMATE_TD)
    {
        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
    }
    vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

    if (last_marginalization_info)
        delete last_marginalization_info;
    last_marginalization_info = marginalization_info;
    last_marginalization_parameter_blocks = parameter_blocks;
}

void Backend::margNew(Estimator *estimator)
{
    if (last_marginalization_info &&
        std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
    {

        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        
        vector2double(estimator);

        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        TicToc t_pre_margin;
        ROS_DEBUG("begin marginalization");
        marginalization_info->preMarginalize();
        ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

        TicToc t_margin;
        ROS_DEBUG("begin marginalization");
        marginalization_info->marginalize();
        ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
        std::unordered_map<long, double *> addr_shift;
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            if (i == WINDOW_SIZE - 1)
                continue;
            else if (i == WINDOW_SIZE)
            {
                addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
            }
            else
            {
                addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
            }
        }

        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
            
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;            
    }
}

void Backend::optimization(Estimator *estimator)
{
    TicToc t_whole;
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    // 设置损失函数为 Cauchy Loss，以处理异常值并提高鲁棒性
    loss_function = new ceres::CauchyLoss(1.0);
    nonLinearOptimization(estimator,problem,loss_function);

    TicToc t_whole_marginalization;
    if (estimator->marginalization_flag == MARGIN_OLD)
        margOld(estimator,loss_function);   
    else
        margNew(estimator);
    
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

void Backend::slideWindow(Estimator *estimator)
{
    TicToc t_margin;
    if (estimator->marginalization_flag == MARGIN_OLD)
    {
        double t_0 = estimator->Headers[0].stamp.toSec();
        estimator->back_R0 = estimator->Rs[0];
        estimator->back_P0 = estimator->Ps[0];
        if (estimator->frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                estimator->Rs[i].swap(estimator->Rs[i + 1]);

                std::swap(estimator->pre_integrations[i], estimator->pre_integrations[i + 1]);

                estimator->dt_buf[i].swap(estimator->dt_buf[i + 1]);
                estimator->linear_acceleration_buf[i].swap(estimator->linear_acceleration_buf[i + 1]);
                estimator->angular_velocity_buf[i].swap(estimator->angular_velocity_buf[i + 1]);

                estimator->Headers[i] = estimator->Headers[i + 1];
                estimator->Ps[i].swap(estimator->Ps[i + 1]);
                estimator->Vs[i].swap(estimator->Vs[i + 1]);
                estimator->Bas[i].swap(estimator->Bas[i + 1]);
                estimator->Bgs[i].swap(estimator->Bgs[i + 1]);
            }
            estimator->Headers[WINDOW_SIZE] = estimator->Headers[WINDOW_SIZE - 1];
            estimator->Ps[WINDOW_SIZE] = estimator->Ps[WINDOW_SIZE - 1];
            estimator->Vs[WINDOW_SIZE] = estimator->Vs[WINDOW_SIZE - 1];
            estimator->Rs[WINDOW_SIZE] = estimator->Rs[WINDOW_SIZE - 1];
            estimator->Bas[WINDOW_SIZE] = estimator->Bas[WINDOW_SIZE - 1];
            estimator->Bgs[WINDOW_SIZE] = estimator->Bgs[WINDOW_SIZE - 1];

            delete estimator->pre_integrations[WINDOW_SIZE];
            estimator->pre_integrations[WINDOW_SIZE] = new IntegrationBase{estimator->acc_0, estimator->gyr_0, estimator->Bas[WINDOW_SIZE], estimator->Bgs[WINDOW_SIZE]};

            estimator->dt_buf[WINDOW_SIZE].clear();
            estimator->linear_acceleration_buf[WINDOW_SIZE].clear();
            estimator->angular_velocity_buf[WINDOW_SIZE].clear();

            if (estimator->solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = estimator->all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
 
                for (map<double, ImageFrame>::iterator it = estimator->all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                estimator->all_image_frame.erase(estimator->all_image_frame.begin(), it_0);
                estimator->all_image_frame.erase(t_0);
            }
            slideWindowOld(estimator);
        }
    }
    else
    {
        if (estimator->frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < estimator->dt_buf[estimator->frame_count].size(); i++)
            {
                double tmp_dt = estimator->dt_buf[estimator->frame_count][i];
                Vector3d tmp_linear_acceleration = estimator->linear_acceleration_buf[estimator->frame_count][i];
                Vector3d tmp_angular_velocity = estimator->angular_velocity_buf[estimator->frame_count][i];

                estimator->pre_integrations[estimator->frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                estimator->dt_buf[estimator->frame_count - 1].push_back(tmp_dt);
                estimator->linear_acceleration_buf[estimator->frame_count - 1].push_back(tmp_linear_acceleration);
                estimator->angular_velocity_buf[estimator->frame_count - 1].push_back(tmp_angular_velocity);
            }

            estimator->Headers[estimator->frame_count - 1] = estimator->Headers[estimator->frame_count];
            estimator->Ps[estimator->frame_count - 1] = estimator->Ps[estimator->frame_count];
            estimator->Vs[estimator->frame_count - 1] = estimator->Vs[estimator->frame_count];
            estimator->Rs[estimator->frame_count - 1] = estimator->Rs[estimator->frame_count];
            estimator->Bas[estimator->frame_count - 1] = estimator->Bas[estimator->frame_count];
            estimator->Bgs[estimator->frame_count - 1] = estimator->Bgs[estimator->frame_count];

            delete estimator->pre_integrations[WINDOW_SIZE];
            estimator->pre_integrations[WINDOW_SIZE] = new IntegrationBase{estimator->acc_0, estimator->gyr_0, estimator->Bas[WINDOW_SIZE], estimator->Bgs[WINDOW_SIZE]};

            estimator->dt_buf[WINDOW_SIZE].clear();
            estimator->linear_acceleration_buf[WINDOW_SIZE].clear();
            estimator->angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew(estimator);
        }
    }
}

// real marginalization is removed in solve_ceres()
void Backend::slideWindowNew(Estimator *estimator)
{
    //estimator->sum_of_front++;
    estimator->f_manager.removeFront(estimator->frame_count);
}

// real marginalization is removed in solve_ceres()
void Backend::slideWindowOld(Estimator *estimator)
{
    //estimator->sum_of_back++;

    bool shift_depth = estimator->solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = estimator->back_R0 * estimator->ric[0];
        R1 = estimator->Rs[0] * estimator->ric[0];
        P0 = estimator->back_P0 + estimator->back_R0 * estimator->tic[0];
        P1 = estimator->Ps[0] + estimator->Rs[0] * estimator->tic[0];
        estimator->f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        estimator->f_manager.removeBack();
}