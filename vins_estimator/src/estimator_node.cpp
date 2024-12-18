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
bool init_line = 0;
bool init_imu = 1;
double last_imu_t = 0;

// @brief predict status values: Ps/Vs/Rs
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

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

// @brief take and align measurement from feature frames and IMU measurement
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    if (!init_line)
        {
         // 初始化未完成，丢弃buf里所有id大于10000的特征
    std::queue<sensor_msgs::PointCloudConstPtr> filtered_buf; // 用于存储过滤后的特征点云

    while (!feature_buf.empty())
    {
        sensor_msgs::PointCloudConstPtr feature = feature_buf.front();
        feature_buf.pop();

        // 创建一个新的点云消息用于存储过滤后的点
        sensor_msgs::PointCloud::Ptr filtered_feature(new sensor_msgs::PointCloud);
        filtered_feature->header = feature->header;

        sensor_msgs::ChannelFloat32 id_of_point_filtered;
        sensor_msgs::ChannelFloat32 u_of_point_filtered;
        sensor_msgs::ChannelFloat32 v_of_point_filtered;
        sensor_msgs::ChannelFloat32 velocity_x_of_point_filtered;
        sensor_msgs::ChannelFloat32 velocity_y_of_point_filtered;

        // 遍历点云中的每个点，过滤掉 ID > 10000 的点
        for (size_t i = 0; i < feature->points.size(); ++i)
        {
            // 从 id_of_point 的 values 中获取点的 ID
            int v = feature->channels[0].values[i] + 0.5; // 四舍五入
            int id = v / NUM_OF_CAM;                     // 根据 NUM_OF_CAM 计算 ID

            if (id <= 10000)
            {
                // 保留符合条件的点和其相关的 channel 数据
                filtered_feature->points.push_back(feature->points[i]);
                id_of_point_filtered.values.push_back(feature->channels[0].values[i]);
                u_of_point_filtered.values.push_back(feature->channels[1].values[i]);
                v_of_point_filtered.values.push_back(feature->channels[2].values[i]);
                velocity_x_of_point_filtered.values.push_back(feature->channels[3].values[i]);
                velocity_y_of_point_filtered.values.push_back(feature->channels[4].values[i]);
            }
        }

        // 只有当过滤后的点云非空时才存入缓冲区
        if (!filtered_feature->points.empty())
        {
            filtered_feature->channels.push_back(id_of_point_filtered);
            filtered_feature->channels.push_back(u_of_point_filtered);
            filtered_feature->channels.push_back(v_of_point_filtered);
            filtered_feature->channels.push_back(velocity_x_of_point_filtered);
            filtered_feature->channels.push_back(velocity_y_of_point_filtered);

            filtered_buf.push(filtered_feature); // 存入过滤后的队列
        }
    }

    // 用过滤后的点云队列替换原始队列
    feature_buf = std::move(filtered_buf);

        }

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }

        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

// @brief put IMU measurement in buffer and publish status values: Ps/Vs/Rs
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);// TODO useless?
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

// @brief put feature measurement in buffer
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}
void line_feature_start_callback(const sensor_msgs::PointCloudConstPtr &line_feature_start_msg)
{
    m_buf.lock();
    feature_buf.push(line_feature_start_msg);
    m_buf.unlock();
    con.notify_one();
}
void line_feature_end_callback(const sensor_msgs::PointCloudConstPtr &line_feature_end_msg)
{
    m_buf.lock();
    feature_buf.push(line_feature_end_msg);
    m_buf.unlock();
    con.notify_one();
}
// @brief restart
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
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

// @brief IMU pre-integration and get pre-optimized status values Ps/Vs/Rs
void processIMU(sensor_msgs::ImuConstPtr &imu_msg,
                sensor_msgs::PointCloudConstPtr &img_msg)
{
    double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
    double t = imu_msg->header.stamp.toSec();
    double img_t = img_msg->header.stamp.toSec() + estimator.td;
    if (t <= img_t)
    { 
        if (current_time < 0)
            current_time = t;
        double dt = t - current_time;
        ROS_ASSERT(dt >= 0);
        current_time = t;
        dx = imu_msg->linear_acceleration.x;
        dy = imu_msg->linear_acceleration.y;
        dz = imu_msg->linear_acceleration.z;
        rx = imu_msg->angular_velocity.x;
        ry = imu_msg->angular_velocity.y;
        rz = imu_msg->angular_velocity.z;
        estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));                    
        //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
    }
    else
    {
        double dt_1 = img_t - current_time;
        double dt_2 = t - img_t;
        current_time = img_t;
        ROS_ASSERT(dt_1 >= 0);
        ROS_ASSERT(dt_2 >= 0);
        ROS_ASSERT(dt_1 + dt_2 > 0);
        double w1 = dt_2 / (dt_1 + dt_2);
        double w2 = dt_1 / (dt_1 + dt_2);
        dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
        dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
        dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
        rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
        ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
        rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
        estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
        //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
    }
}

// @brief setReloFrame
void setReloFrame(sensor_msgs::PointCloudConstPtr &relo_msg)
{
    //sensor_msgs::PointCloudConstPtr relo_msg = NULL;
    while (!relo_buf.empty())
    {
        relo_msg = relo_buf.front();
        relo_buf.pop();
    }
    if (relo_msg != NULL)
    {
        vector<Vector3d> match_points;
        double frame_stamp = relo_msg->header.stamp.toSec();
        for (unsigned int i = 0; i < relo_msg->points.size(); i++)
        {
            Vector3d u_v_id;
            u_v_id.x() = relo_msg->points[i].x;
            u_v_id.y() = relo_msg->points[i].y;
            u_v_id.z() = relo_msg->points[i].z;
            match_points.push_back(u_v_id);
        }
        Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
        Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
        Matrix3d relo_r = relo_q.toRotationMatrix();
        int frame_index;
        frame_index = relo_msg->channels[0].values[7];
        estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
    }    
}

// @brief main vio function, including initialization and optimization
void processVIO(sensor_msgs::PointCloudConstPtr& img_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
    for (unsigned int i = 0; i < img_msg->points.size(); i++)
    {
        int v = img_msg->channels[0].values[i] + 0.5;
        int feature_id = v / NUM_OF_CAM;
        int camera_id = v % NUM_OF_CAM;
        double x = img_msg->points[i].x;
        double y = img_msg->points[i].y;
        double z = img_msg->points[i].z;
        double p_u = img_msg->channels[1].values[i];
        double p_v = img_msg->channels[2].values[i];
        double velocity_x = img_msg->channels[3].values[i];
        double velocity_y = img_msg->channels[4].values[i];
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    //estimator.processImage(image, img_msg->header);
    estimator.processImage(image, img_msg->header,init_line);
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

// @brief main vio function, including initialization and optimization
void processMeasurement(vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>& measurements)
{
    for (auto &measurement : measurements)
    {
        auto img_msg = measurement.second;

        // get status value:Rs/Vs/Ps
        for (auto &imu_msg : measurement.first)
            processIMU(imu_msg,img_msg);

        // set relocalization frame
        sensor_msgs::PointCloudConstPtr relo_msg = NULL;
        setReloFrame(relo_msg);
        ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

        // main function for vio
        TicToc t_s;
        processVIO(img_msg);

        double whole_t = t_s.toc();
        printStatistics(estimator, whole_t);
        std_msgs::Header header = img_msg->header;
        header.frame_id = "world";

        // show in rviz
        visualize(relo_msg, header);
    }
}

// @brief main vio function, including initialization and optimization
// thread: visual-inertial odometry
void process()
{
    while (true)
    {
        // get measurement in buf and make them aligned
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);        
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();

        // main function of vio
        m_estimator.lock();
        processMeasurement(measurements);
        m_estimator.unlock();

        // update status value Rs/Ps/Vs
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

// @brief main function 
int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    // 订阅线特征起点数据，回调函数为 line_feature_start_callback
    //ros::Subscriber sub_line_start = n.subscribe("/feature_tracker/line_feature_start", 20000, line_feature_start_callback);
    // 订阅线特征终点数据，回调函数为 line_feature_end_callback
    //ros::Subscriber sub_line_end = n.subscribe("/feature_tracker/line_feature_end", 20000, line_feature_end_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
