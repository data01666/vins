//#include "backend.h"
#include "../include/estimator.h" 

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

void Estimator::setParameter()
{
    // 遍历所有相机
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        // 将相机的平移向量 tic 和旋转矩阵 ric 初始化为给定的 TIC 和 RIC 值
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }

    // 设置相机的旋转矩阵
    f_manager.setRic(ric);

    // 重投影误差部分的信息矩阵
    // 设置投影因子的平方根信息矩阵，使用焦距调整后的单位矩阵
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();

    // 设置时间差投影因子的平方根信息矩阵，使用焦距调整后的单位矩阵
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();

    // 设置时间差 TD
    td = TD;
}


void Estimator::clearState()
{
    // 遍历窗口内的所有状态
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // 将旋转矩阵设为单位矩阵
        Rs[i].setIdentity();
        // 将位置向量、速度向量、加速度偏差、角速度偏差清零
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();

        // 清空时间差和加速度、角速度缓冲区
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        // 删除预积分对象并将指针设为 nullptr
        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    // 重置相机参数
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero(); // 平移向量设为零
        ric[i] = Matrix3d::Identity(); // 旋转矩阵设为单位矩阵
    }

    // 遍历所有图像帧，清理预积分对象
    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration; // 删除预积分对象
            it.second.pre_integration = nullptr; // 将指针设为 nullptr
        }
    }

    // 重置求解器状态和其他标志
    solver_flag = INITIAL; // 设置求解器标志为初始状态
    first_imu = false; // 重置第一次 IMU 标志
    frame_count = 0; // 重置帧计数
    initial_timestamp = 0; // 重置初始时间戳
    all_image_frame.clear(); // 清空所有图像帧
    td = TD; // 恢复时间差参数

    // 清理临时预积分对象
    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    tmp_pre_integration = nullptr; // 将指针设为 nullptr

    // 清理状态管理器
    f_manager.clearState();

    // 重置故障和重定位信息
    failure_occur = 0;
    relocalization_info = 0;

    // 重置漂移修正矩阵和向量
    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}


void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    /* 这部分代码的作用是为以下数据提供初始值或进行初始化：
     * pre_integrations[frame_count]  - 预积分对象
     * dt_buf[frame_count]            - 时间差缓存
     * linear_acceleration_buf[frame_count]  - 线性加速度缓存
     * angular_velocity_buf[frame_count]    - 角速度缓存
     * Rs[frame_count]               - 旋转矩阵
     * Ps[frame_count]               - 位置
     * Vs[frame_count]               - 速度
     * TODO: frame_count 的更新目前只在 `process_img` 中的 `solver_flag == INITIAL` 条件下看到
     */

    // 判断是否为第一帧 IMU 数据，如果是，则将当前加速度和角速度作为初始值
    if (!first_imu)
    {
        first_imu = true; // 标记已接收到第一帧 IMU 数据
        acc_0 = linear_acceleration; // 将当前加速度作为初始值
        gyr_0 = angular_velocity;    // 将当前角速度作为初始值
    }

    // 如果当前帧的预积分对象未初始化，则进行初始化
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }

    // 如果 `frame_count` 不为 0，进行 IMU 数据的预积分
    if (frame_count != 0)
    {
        // 将当前 IMU 数据推入预积分对象中
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);

        // 如果求解器状态不为非线性优化，更新临时预积分
        if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        // 将时间差、线性加速度和角速度存入缓存
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        //注意啊，这块对j的操作看似反智，是因为j时刻的值都拷贝了j-1时刻的值！！
        //第一次使用实际上就是使用的是j-1时刻的值，所以在这些地方写上j-1是没有关系的！
        //noise是zero mean Gauss，在这里忽略了
        //TODO 把j改成j-1，看看效果是一样

        // 使用前一时刻的旋转、位置、速度和加速度来更新当前时刻的状态
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g; // 计算未补偿的初始加速度
        //下面都采用的是中值积分的传播方式，noise被忽略了
        //TODO 把j改成j-1，看看效果是一样
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j]; // 计算未补偿的角速度

        // 更新旋转矩阵，使用四元数进行姿态的增量更新
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();

        // 计算未补偿的当前加速度
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;

        // 使用中值积分法计算平均加速度
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

        // 根据加速度更新位置 Ps[j] 和速度 Vs[j]
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }

    // 更新上一时刻的加速度和角速度，为下次计算做准备
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}


//void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,const map<int, vector<pair<int, Eigen::Matrix<double, 12, 1>>>> &lines, const std_msgs::Header &header)
{
    // 1. 根据视差检查是否需要边缘化旧帧或次新帧，并将特征添加到特征管理器
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());

    //true：上一帧是关键帧，marg_old; false:上一帧不是关键帧marg_second_new
    //TODO frame_count指的是次新帧还是最新帧？
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;
    addline(lines);

    // 2. 为初始化阶段构建所有图像帧的数据结构
    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    if (solver_flag != NON_LINEAR)
    {
        // 初始化时，创建图像帧结构并保存 IMU 预积分数据
        // 数据结构: imageframe是ImageFrame的一个实例，定义在initial / initial_alignment.h里,它是用于融合IMU和视觉信息的数据结构
        // 包括了某一帧的全部信息位姿，特征点信息，预积分信息，是否是关键帧等。
        ImageFrame imageframe(image, header.stamp.toSec());
        imageframe.pre_integration = tmp_pre_integration;
        all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
        // 更新临时预积分初始值
        tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }

    // 3. 校准从相机到 IMU 的旋转与平移
    calibrationExRotation();

    // 4. 系统初始化
    if (solver_flag == INITIAL)
        initial(header);

    // 5. 进入非线性优化阶段
    else
        backend.backend(this);
}

void Estimator::addline(const map<int, vector<pair<int, Eigen::Matrix<double, 12, 1>>>> &lines)
{
    // 遍历所有线特征
    for (const auto &line : lines)
    {
        int line_id = line.first;  // 获取线特征 ID
        LineFeaturePerFrame line_frame(line.second[0].second, td);  // 获取该特征的观测信息

        // 查找当前线特征 ID 是否已经存在于特征管理器中
        auto it = find_if(f_manager.line_feature.begin(), f_manager.line_feature.end(), [line_id](const LineFeaturePerId &lf)
        {
            return lf.line_id == line_id;
        });

        if (it == f_manager.line_feature.end())
        {
            // 如果是新线特征，则将其添加到特征管理器中
            LineFeaturePerId new_line(line_id, frame_count);
            new_line.feature_per_frame.push_back(line_frame);
            f_manager.line_feature.push_back(new_line);  // 将新特征添加到特征管理器
        }
        else
        {
            // 如果线特征已存在，则更新当前帧的数据
            it->feature_per_frame.push_back(line_frame);
        }
    }
}

void Estimator::calibrationExRotation()
{
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }
}

void Estimator::initial(const std_msgs::Header &header)
{
    if (frame_count == WINDOW_SIZE)
    {
        bool result = false;
        if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
        {
            result = intializer.initialStructure(this);
            initial_timestamp = header.stamp.toSec();
        }
        if(result)
        {
            solver_flag = NON_LINEAR;
            backend.solveOdometry(this);
            backend.slideWindow(this);
            f_manager.removeFailures();
            ROS_INFO("Initialization finish!");
            last_R = Rs[WINDOW_SIZE];
            last_P = Ps[WINDOW_SIZE];
            last_R0 = Rs[0];
            last_P0 = Ps[0];               
        }
        else
            backend.slideWindow(this);
    }
    else
        frame_count++;
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = backend.para_Pose[i][j];
        }
    }
}

