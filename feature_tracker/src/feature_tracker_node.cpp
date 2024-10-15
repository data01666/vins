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

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "../include/feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match,pub_line_start,pub_line_end;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];

double first_image_time;
int pub_count = 1;  // 每隔delta_t = 1/FREQ 时间内连续(没有中断/没有报错)发布的帧数
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;  // 0:第一帧不把特征发布到buf里    1:发布到buf里


void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    // first image process
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }

    // detect unstable camera stream
    // 时间戳变化异常（超过1秒或时间戳倒退）, 重置特征跟踪器
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }

    // renew time stamp
    last_image_time = img_msg->header.stamp.toSec();

    // frequency control
    // 如果当前帧的发布频率小于设定的频率 FREQ（单位：帧/秒），则允许发布当前帧数据
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        // 如果当前帧的发布频率接近设定的频率 FREQ，则重置频率控制
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    // change the format of the image
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    // read image and track features
    cv::Mat show_img = ptr->image;
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)// 单目或者不进行立体跟踪
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif

    }

    // renew global feature ids
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
        {
            if (j != 1 || !STEREO_TRACK)
            {
                // 更新点特征的 ID
                completed |= trackerData[j].updateID(i, false);  // false 表示处理点特征
                // 更新线段特征的 ID
                completed |= trackerData[j].updateID(i, true);  // true 表示处理线段特征
            }
        }
        if (!completed)
            break;
    }

    // publish features
   if (PUB_THIS_FRAME)
    {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        // 线段相关数据结构
        sensor_msgs::ChannelFloat32 id_of_line;
        sensor_msgs::PointCloudPtr feature_line_start(new sensor_msgs::PointCloud);
        sensor_msgs::PointCloudPtr feature_line_end(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 start_x;
        sensor_msgs::ChannelFloat32 start_y;
        sensor_msgs::ChannelFloat32 end_x;
        sensor_msgs::ChannelFloat32 end_y;
        sensor_msgs::ChannelFloat32 velocity_x_of_line;
        sensor_msgs::ChannelFloat32 velocity_y_of_line;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";
        feature_line_start->header = img_msg->header; // 确保线段点云的头部信息
        feature_line_start->header.frame_id = "world";
        feature_line_end->header = img_msg->header;
        feature_line_end->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;

            auto &cur_un_line = trackerData[i].cur_un_lines;
            auto &line_segments = trackerData[i].cur_line_segments; // 当前线段
            auto &line_ids = trackerData[i].line_ids; // 线段 ID
            auto &line_velocity = trackerData[i].line_velocity; // 线段速度

            // 处理点特征
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    // p存贮的是归一化平面坐标
                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    // u_of_point存储的是像素坐标
                    u_of_point.values.push_back(cur_pts[j].x);
                    // v_of_point存储的是像素坐标
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }

            // 处理线段特征
            for (unsigned int j = 0; j < line_ids.size(); j++)
            {

                geometry_msgs::Point32 start_point, end_point;

                start_point.x = cur_un_line[j].sx;
                start_point.y = cur_un_line[j].sy;
                start_point.z = 1;
                end_point.x = cur_un_line[j].ex;
                end_point.y = cur_un_line[j].ey;
                end_point.z = 1;

                feature_line_start->points.push_back(start_point);
                feature_line_end->points.push_back(end_point);
                id_of_line.values.push_back(line_ids[j] * NUM_OF_CAM + i); // 假设 line_ids[j] 是有效的线段 ID
                velocity_x_of_line.values.push_back(line_velocity[j].x); // 假设有线段速度
                velocity_y_of_line.values.push_back(line_velocity[j].y);

                // 填充 start 和 end 通道
                const auto &line = line_segments[j];
                start_x.values.push_back(line.sx);
                start_y.values.push_back(line.sy);
                end_x.values.push_back(line.ex);
                end_y.values.push_back(line.ey);
            }
        }

        // 将点特征信息添加到消息中
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);

        // 将线段特征信息添加到消息中
        feature_line_start->channels.push_back(start_x);
        feature_line_start->channels.push_back(start_y);
        feature_line_end->channels.push_back(id_of_line);
        feature_line_end->channels.push_back(end_x);
        feature_line_end->channels.push_back(end_y);
        feature_line_end->channels.push_back(velocity_x_of_line);
        feature_line_end->channels.push_back(velocity_y_of_line);

        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());

        // skip the first image; since no optical speed on first image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
        {
            pub_img.publish(feature_points);
            pub_line_start.publish(feature_line_start); // 发布线段特征
            pub_line_end.publish(feature_line_end);
        }

        // show track
        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                // 绘制点特征
                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);

                    // 绘制速度线
                    Vector2d tmp_cur_un_pts(trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity(trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255, 0, 0), 1, 8, 0);
                }

                // 绘制线段特征
                for (const auto &line : trackerData[i].cur_line_segments)
                {
                    cv::line(tmp_img, cv::Point2f(line.sx,line.sy), cv::Point2f(line.ex,line.ey), cv::Scalar(0, 255, 0), 2); // 绘制线段
                }
            }

            pub_match.publish(ptr->toImageMsg());
        }
    }

    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_line_start = n.advertise<sensor_msgs::PointCloud>("line_feature_start", 1000);
    pub_line_end = n.advertise<sensor_msgs::PointCloud>("line_feature_end", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?