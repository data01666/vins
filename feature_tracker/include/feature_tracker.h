#ifndef _FEATURE_TRACKER_H_
#define _FEATURE_TRACKER_H_

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"
#include "line_feature.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;


/**
 * @brief 检查跟踪特征点的像素坐标是否在边界内
 * @param 输入 cv::Point2f &pt
 */
bool inBorder(const cv::Point2f &pt);

/**
 * @brief 根据输入值删除输出值中的项目
 * @param 输入  vector<uchar> status
 * @param 输出  vector<cv::Point2f> &v
 */
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);

/**
 * @brief 根据输入值删除输出值中的项目
 * @param 输入  vector<uchar> status
 * @param 输出  vector<int> &v
 */
void reduceVector(vector<int> &v, vector<uchar> status);


/**
 * @brief Class of FeatureTracker
 */
class FeatureTracker
{
  public:
    FeatureTracker();

    /**
     * @brief 均衡图像
     * @param 输入  const cv::Mat &_img
     * @param 输出  cv::Mat &img
     */
    void equalize(const cv::Mat &_img, cv::Mat &img);

    /**
     * @brief 跟踪现有特征并删除失败的特征
     */
    void flowTrack();
    void lineflowTrack();
    double computeDTW(const std::vector<std::pair<double, double>>& prev_keypoints,
                      const std::vector<std::pair<double, double>>& forw_keypoints);
 std::vector<std::pair<double, double>> normalizeKeypoints(const std::vector<std::pair<double, double>> &keypoints);
    void trackNewlines();
    void addLines();

    /**
     * @brief 在前向图像中跟踪新特征
     */
    void trackNew();

    /**
     * @brief 特征跟踪器的主要处理过程
     * @param 输入  const cv::Mat &_img
     * @param 输入  double _cur_time
     */
    void readImage(const cv::Mat &_img, double _cur_time);

    /**
     * @brief 根据特征存在的时间对其进行排序，并为高排名特征设置掩码，以避免特征布局过于密集
     */
    void setMask();

    /**
     * @brief 将新跟踪的特征添加到缓冲区
     */
    void addPoints();

    /**
     * @brief 更新特征的ID
     */
    bool updateID(unsigned int i, bool line);

    /**
     * @brief 读取相机参数
     * @param 输入  const string &calib_file
     */
    void readIntrinsicParameter(const string &calib_file);

    /**
     * @brief 显示去畸变效果
     * TODO 不重要，可以删除
     * @param 输入  const string &name
     */
    void showUndistortion(const string &name);

    /**
     * @brief 使用基本矩阵删除离群点
     */
    void rejectWithF();

    /**
     * @brief 获取特征的去畸变归一化坐标
     * @param 输入  const string &name
     */
    void undistortedPoints();
    void undistortedLines();

    cv::Mat mask; // 图像掩码
    cv::Mat fisheye_mask; // 鱼眼相机mask，用来去除边缘噪点

    cv::Mat prev_img, cur_img, forw_img; // 上一次发布出去的帧的图像数据/光流跟踪的上一帧的图像数据/光流跟踪的当前的图像数据
    // 完成跟踪后，把 forw 中的特征（点或线段）赋值给 cur

    vector<cv::Point2f> n_pts; // 每一帧中新提取的特征点
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts; // 对应的图像特征点
    vector<cv::Point2f> prev_un_pts, cur_un_pts; // 归一化相机坐标系下的坐标
    vector<cv::Point2f> pts_velocity; // 当前帧相对前一帧特征点沿x,y方向的像素移动速度

    vector<int> ids; // 能够被跟踪到的特征点的id
    vector<int> track_cnt; // 当前帧forw_img中每个特征点被追踪的时间次数

    map<int, cv::Point2f> cur_un_pts_map; // 构建id与归一化坐标的映射，见undistortedPoints()
    map<int, cv::Point2f> prev_un_pts_map; // 上一帧的id与归一化坐标的映射

    camodocal::CameraPtr m_camera; // 相机模型

    double cur_time; // 当前时间
    double prev_time; // 上一帧时间

    static int n_id; // 用来作为特征点id，每检测到一个新的特征点，就将++n_id作为该特征点的id

    std::vector<LineSegment> prev_line_segments,cur_line_segments,forw_line_segments; // 对应的线特征
 std::vector<LineSegment> prev_un_lines,cur_un_lines; // 归一化相机坐标系下的线特征
    std::vector<LineSegment> line_pts; // 每一帧新提取的线特征
    // 线段的ID
    vector<int> line_ids;       // 线段的ID
    static int line_n_id;      // 线段的n_id
 vector<int> line_track_cnt; // 当前帧forw_img中每个线段被追踪的时间次数
    map<int, LineSegment> cur_line_map;
    map<int, LineSegment> prev_line_map;
    // 线段的速度（可以根据关键点位置计算）
    vector<cv::Point2f> start_velocity; // 每一条线段的移动速度
    vector<cv::Point2f> end_velocity;
};

#endif