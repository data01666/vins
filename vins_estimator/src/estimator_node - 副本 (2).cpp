/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Original Author: Qin Tong (qintonguav@gmail.com)
 * Remodified Author: Hu(rhuag@connect.ust.hk) at HKUST, https://blog.csdn.net/iwanderu
 *******************************************************/

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "../include/estimator.h"
#include "../include/parameters.h"
#include "../include/utility/visualization.h"

// @param main vio operator
Estimator estimator;

// @param buffer
std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

// @param mutex for buf, status value and vio processing
std::mutex m_buf;
std::mutex m_state;
//std::mutex i_buf;   // TODO seems like useless
std::mutex m_estimator;

// @param temp status values
double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;

// @param flags
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

// @brief predict status values: Ps/Vs/Rs
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // 获取当前 IMU 数据的时间戳
    double t = imu_msg->header.stamp.toSec();

    // 初始化 IMU 的标志
    if (init_imu)
    {
        // 如果是第一次接收 IMU 数据，则记录当前时间并标记初始化完成
        latest_time = t;
        init_imu = 0;
        return;
    }

    // 计算当前时间与上一次时间的时间差 dt
    double dt = t - latest_time;
    latest_time = t; // 更新最新的时间戳

    // 获取线性加速度
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz}; // 将线性加速度存入 Eigen 向量

    // 获取角速度
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz}; // 将角速度存入 Eigen 向量

    // 计算未补偿的初始加速度，考虑当前姿态（tmp_Q）和加速度偏置（tmp_Ba）
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    // 计算平均角速度，考虑当前的角速度偏置
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;

    // 根据角速度更新四元数 tmp_Q（更新姿态）
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    // 计算当前时刻的未补偿加速度
    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    // 计算平均未补偿加速度
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    // 根据加速度更新位置 tmp_P
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;

    // 根据加速度更新速度 tmp_V
    tmp_V = tmp_V + dt * un_acc;

    // 更新上一时刻的加速度和角速度
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}


// @brief update status values: Ps/Vs/Rs
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

// 获取并对齐特征帧和IMU测量数据
// IMU数据时间戳位于图像帧之前
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        // 边界判断：如果 IMU 缓冲区或特征缓冲区为空，说明配对完成
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        // 检查 IMU 缓冲区最后一条数据的时间戳是否晚于特征缓冲区第一条数据的时间戳
        // 如果不晚，说明当前 IMU 数据不足，需要等待更多 IMU 数据
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            // 统计等待次数
            sum_of_wait++;
            // 返回当前的测量结果
            return measurements;
        }

        // 检查 IMU 缓冲区第一条数据的时间戳是否早于特征缓冲区第一条数据的时间戳
        // 如果不早，说明特征帧过多，需要丢弃特征帧
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            // 发出警告信息，提示丢弃特征帧
            ROS_WARN("throw img, only should happen at the beginning");
            // 丢弃过多的特征帧
            feature_buf.pop();
            // 继续下一个循环
            continue;
        }

        // 核心操作：获取视觉帧信息
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop(); // 从特征缓冲区中移除该帧

        // 核心操作：提取与当前视觉帧时间戳对齐的 IMU 信息
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front()); // 将 IMU 数据添加到数组中
            imu_buf.pop(); // 从 IMU 缓冲区中移除该数据
        }

        // 注意：将最后一帧 IMU 数据放回到 imu_buf 中
        // 原因：最后一帧 IMU 信息是被相邻两个视觉帧共享的
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image"); // 如果没有 IMU 数据，发出警告

        measurements.emplace_back(IMUs, img_msg); // 将当前的 IMU 数据和图像帧信息添加到测量结果中
    }
    return measurements; // 返回最终的测量结果
}


// @brief put IMU measurement in buffer and publish status values: Ps/Vs/Rs
/**
 * imu_callback 主要干了 3 件事：
 * 1. 将 IMU 数据缓存到 imu_buf 中；
 * 2. 通过 IMU 预积分计算当前时刻的 PVQ（位置、速度、四元数）；
 * 3. 如果当前处于非线性优化阶段，将计算得到的 PVQ 发布到 rviz 中。
 */
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // 检查 IMU 消息的时间戳是否按序
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!"); // 警告：IMU 消息顺序错误
        return; // 返回，忽略该消息
    }

    // 更新最后处理的 IMU 时间戳
    last_imu_t = imu_msg->header.stamp.toSec();

    // 锁定缓冲区以安全地处理 IMU 数据
    m_buf.lock();
    imu_buf.push(imu_msg); // 将 IMU 消息推入缓冲区
    m_buf.unlock(); // 解锁缓冲区
    con.notify_one(); // 通知一个等待线程，有新数据可用

    // 再次更新最后处理的 IMU 时间戳
    last_imu_t = imu_msg->header.stamp.toSec();

    {
        // 使用 std::lock_guard 自动管理互斥锁
        std::lock_guard<std::mutex> lg(m_state); // TODO: 确认该锁是否必要
        predict(imu_msg); // 进行 IMU 预积分，更新预测的 PVQ

        std_msgs::Header header = imu_msg->header; // 创建消息头
        header.frame_id = "world"; // 设置帧 ID 为 "world"

        // 如果当前处于非线性优化阶段，发布最新的位姿
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header); // 发布 PVQ
    }
}

// 把当前帧的所有特征点放到 feature_buf
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        // 跳过第一个检测到的特征，因为它不包含光流速度
        init_feature = 1;
        return;
    }
    // 锁定缓冲区以安全地处理特征数据
    m_buf.lock();
    // 将特征消息推入缓冲区
    feature_buf.push(feature_msg);
    // 解锁缓冲区
    m_buf.unlock();
    // 通知一个等待线程，有新数据可用
    con.notify_one();
}

// 重置所有状态量并清空缓冲区
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        // 发出警告信息，提示重置估计器
        ROS_WARN("restart the estimator!");

        // 锁定缓冲区以安全地清空特征和IMU数据
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop(); // 清空特征缓冲区
        while(!imu_buf.empty())
            imu_buf.pop(); // 清空IMU缓冲区
        m_buf.unlock(); // 解锁缓冲区

        // 锁定估计器以安全地重置状态
        m_estimator.lock();
        estimator.clearState(); // 清除估计器状态
        estimator.setParameter(); // 重新设置参数
        m_estimator.unlock(); // 解锁估计器

        // 重置当前时间和最后IMU时间戳
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

// @brief put relocalization flag in buffer
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// IMU预积分并获取预优化的状态值 Ps/Vs/Rs
void processIMU(sensor_msgs::ImuConstPtr &imu_msg,
                sensor_msgs::PointCloudConstPtr &img_msg)
{
    double dx = 0, dy = 0, dz = 0; // 初始化线性加速度
    double rx = 0, ry = 0, rz = 0; // 初始化角速度
    double t = imu_msg->header.stamp.toSec(); // 获取 IMU 数据时间戳
    double img_t = img_msg->header.stamp.toSec() + estimator.td; // 获取图像帧时间戳

    // 检查 IMU 时间戳是否早于图像时间戳
    if (t <= img_t)
    {
        // 如果当前时间小于 0，更新当前时间为 IMU 时间
        if (current_time < 0)
            current_time = t;

        double dt = t - current_time; // 计算时间增量
        ROS_ASSERT(dt >= 0); // 确保时间增量非负
        current_time = t; // 更新当前时间

        // 获取线性加速度和角速度
        dx = imu_msg->linear_acceleration.x;
        dy = imu_msg->linear_acceleration.y;
        dz = imu_msg->linear_acceleration.z;
        rx = imu_msg->angular_velocity.x;
        ry = imu_msg->angular_velocity.y;
        rz = imu_msg->angular_velocity.z;

        // 调用估计器处理 IMU 数据
        estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
        //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
    }
    else //对于处于边界位置的IMU数据，是被相邻两帧共享的，而且对前一帧的影响会大一些，在这里，对数据线性分配
    {
        // 计算 IMU 数据时间戳与图像时间戳之间的时间差
        double dt_1 = img_t - current_time; // 图像时间与当前时间的差
        double dt_2 = t - img_t; // IMU 时间与图像时间的差
        current_time = img_t; // 更新当前时间

        ROS_ASSERT(dt_1 >= 0); // 确保时间差非负
        ROS_ASSERT(dt_2 >= 0); // 确保时间差非负
        ROS_ASSERT(dt_1 + dt_2 > 0); // 确保时间差和大于零

        // 计算加权系数
        double w1 = dt_2 / (dt_1 + dt_2); // IMU 数据的权重
        double w2 = dt_1 / (dt_1 + dt_2); // 图像数据的权重

        // 加权融合线性加速度和角速度
        dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
        dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
        dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
        rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
        ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
        rz = w1 * rz + w2 * imu_msg->angular_velocity.z;

        // 调用估计器处理 IMU 数据
        estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
        //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
    }
}


// 设置重定位帧数据
void setReloFrame(sensor_msgs::PointCloudConstPtr &relo_msg)
{
    // 清空当前重定位消息（从缓存中取出重定位帧）
    while (!relo_buf.empty())
    {
        relo_msg = relo_buf.front(); // 获取重定位缓冲区的第一个消息
        relo_buf.pop(); // 从缓冲区移除该消息
    }

    // 检查重定位消息是否有效
    if (relo_msg != NULL)
    {
        vector<Vector3d> match_points; // 存储匹配点的容器
        double frame_stamp = relo_msg->header.stamp.toSec(); // 获取重定位帧的时间戳

        // 提取点云中的所有点
        for (unsigned int i = 0; i < relo_msg->points.size(); i++)
        {
            Vector3d u_v_id; // 创建一个三维向量
            u_v_id.x() = relo_msg->points[i].x; // 设置 x 坐标
            u_v_id.y() = relo_msg->points[i].y; // 设置 y 坐标
            u_v_id.z() = relo_msg->points[i].z; // 设置 z 坐标
            match_points.push_back(u_v_id); // 将点添加到匹配点容器中
        }

        // 从通道中提取重定位平移和旋转信息
        Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]); // 提取平移向量
        Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]); // 提取四元数
        Matrix3d relo_r = relo_q.toRotationMatrix(); // 将四元数转换为旋转矩阵

        int frame_index;
        frame_index = relo_msg->channels[0].values[7]; // 获取帧索引
        // 调用估计器设置重定位帧
        estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
    }
}


// @brief 主 VIO 函数，包括初始化和优化
void processVIO(sensor_msgs::PointCloudConstPtr& img_msg)
{
    // 创建一个 map 用于存储图像特征点
    // key 是特征点的 ID，value 是一个包含相机 ID 和特征信息的向量
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;

    // 遍历图像中的每一个点
    for (unsigned int i = 0; i < img_msg->points.size(); i++)
    {
        // 从第一个通道中提取 v 值并转换为整数
        int v = img_msg->channels[0].values[i] + 0.5;

        // 根据相机数量计算出特征点的 ID 和相机的 ID
        int feature_id = v / NUM_OF_CAM;  // 计算特征点 ID
        int camera_id = v % NUM_OF_CAM;   // 计算相机 ID

        // 提取点的 x, y, z 坐标
        double x = img_msg->points[i].x;
        double y = img_msg->points[i].y;
        double z = img_msg->points[i].z;

        // 提取像素坐标 (p_u, p_v)
        double p_u = img_msg->channels[1].values[i];
        double p_v = img_msg->channels[2].values[i];

        // 提取特征点的速度 (velocity_x, velocity_y)
        double velocity_x = img_msg->channels[3].values[i];
        double velocity_y = img_msg->channels[4].values[i];

        // 确保 z 坐标等于 1（在处理 2D 图像时通常将 z 坐标设为 1）
        ROS_ASSERT(z == 1);

        // 使用 Eigen 矩阵将 x, y, z 坐标、像素坐标和速度信息打包成一个 7 维向量
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;

        // 将相机 ID 和该特征点的信息存储到 map 中
        image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }

    // 调用估计器的 processImage 函数，传入图像特征点和图像的头信息
    estimator.processImage(image, img_msg->header);
}

// @brief visualization
void visualize(sensor_msgs::PointCloudConstPtr &relo_msg,std_msgs::Header &header)
{
    pubOdometry(estimator, header);
    pubKeyPoses(estimator, header);
    pubCameraPose(estimator, header);
    pubPointCloud(estimator, header);
    pubTF(estimator, header);
    pubKeyframe(estimator);
    if (relo_msg != NULL)
        pubRelocalization(estimator);
    //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
}

// 主 VIO 函数，包括初始化和优化
void processMeasurement(vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>& measurements)
{
    // 遍历每一个测量对
    for (auto &measurement : measurements)
    {
        auto img_msg = measurement.second; // 获取图像帧信息

        // 处理与当前图像帧相关的所有 IMU 数据
        for (auto &imu_msg : measurement.first)
            processIMU(imu_msg, img_msg); // 处理 IMU 数据与图像帧的结合

        // 设置重定位帧
        sensor_msgs::PointCloudConstPtr relo_msg = NULL; // 初始化重定位消息指针
        setReloFrame(relo_msg); // 设置重定位帧
        ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec()); // 打印调试信息

        // 进行 VIO 处理的主函数
        TicToc t_s; // 开始计时
        processVIO(img_msg); // 处理 VIO 计算

        double whole_t = t_s.toc(); // 记录处理总时间
        printStatistics(estimator, whole_t); // 打印统计信息

        // 设置消息头
        std_msgs::Header header = img_msg->header;
        header.frame_id = "world"; // 设置帧 ID

        // 在 rviz 中可视化结果
        visualize(relo_msg, header); // 显示重定位信息
    }
}


// @brief main vio function, including initialization and optimization
// thread: visual-inertial odometry
void process()
{
    // 无限循环，处理传感器数据
    while (true)
    {
        // 获取对齐的 IMU 和图像测量数据
        // IMU 的频率高于视觉帧的发布频率，所以每一帧图像都会配对多个 IMU 数据
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

        // 加锁数据缓冲区，防止并发问题
        std::unique_lock<std::mutex> lk(m_buf);

        // 等待并检查是否有对齐的数据可用
        // 一旦 `getMeasurements()` 获取到数据，跳出等待
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });

        // 解锁缓冲区，允许其他线程继续访问
        lk.unlock();

        // VIO 的主处理函数
        // 估计器加锁，确保估计器处理过程中不发生并发问题
        m_estimator.lock();
        processMeasurement(measurements); // 处理测量数据，包括 IMU 和视觉数据的优化
        m_estimator.unlock(); // 解锁估计器

        // 更新状态值 Rs（旋转矩阵）、Ps（位置向量）、Vs（速度向量）
        m_buf.lock(); // 锁定数据缓冲区，防止并发访问
        m_state.lock(); // 锁定状态，防止并发问题
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) // 如果估计器处于非线性优化阶段
            update(); // 更新当前状态（位置、速度、姿态等）
        m_state.unlock(); // 解锁状态
        m_buf.unlock(); // 解锁数据缓冲区
    }
}

// @brief main function
int main(int argc, char **argv)
{
    // 初始化 ROS 节点，节点名称为 "vins_estimator"
    ros::init(argc, argv, "vins_estimator");
    // 创建一个私有的节点句柄，用于参数的读取和话题的管理
    ros::NodeHandle n("~");
    // 设置 ROS 控制台的日志级别为 Info
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    // 读取 ROS 参数服务器中的参数并设置
    readParameters(n);
    // 初始化估计器的参数
    estimator.setParameter();

    // 检查是否定义了 EIGEN_DONT_PARALLELIZE 宏，如果有则输出调试信息
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    // 输出警告信息，表示系统正在等待图像和 IMU 数据
    ROS_WARN("waiting for image and imu...");

    // 注册发布器，用于发布估计器的相关输出
    registerPub(n);

    // 订阅 IMU 数据的话题，回调函数为 imu_callback，消息队列大小为 2000，使用 tcpNoDelay 来防止延迟
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    // 订阅特征跟踪器的特征数据，回调函数为 feature_callback
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    // 订阅重新启动特征跟踪器的话题，回调函数为 restart_callback
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    // 订阅重定位匹配点的话题，回调函数为 relocalization_callback
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);
    // 创建一个线程处理测量数据的对齐和处理函数，调用 process 函数
    std::thread measurement_process{process};
    // 进入 ROS 事件循环，等待回调函数处理消息
    ros::spin();

    return 0; // 程序正常结束
}