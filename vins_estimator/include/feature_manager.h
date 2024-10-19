#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

// 特征点在某一帧上的信息
class FeaturePerFrame
{
public:
  // 构造函数，初始化特征点的归一化坐标、像素坐标、速度和时间延迟
  FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
  {
    // 归一化坐标
    point.x() = _point(0);
    point.y() = _point(1);
    point.z() = _point(2);

    // 像素坐标
    uv.x() = _point(3);
    uv.y() = _point(4);

    // 速度
    velocity.x() = _point(5);
    velocity.y() = _point(6);

    // 时间延迟
    cur_td = td;
  }

  double cur_td;          // 时间延迟，用于校正特征点数据的时间差异
  Vector3d point;         // 归一化坐标（3D），即相机坐标系中的方向向量
  Vector2d uv;            // 像素坐标（2D）
  Vector2d velocity;      // 速度信息（2D），用于光流跟踪
  double z;               // 深度（初始化时未赋值）
  bool is_used;           // 标记该特征点是否被使用
  double parallax;        // 视差，用于判断关键帧
  MatrixXd A;             // 矩阵A，用于特征点优化中的信息矩阵
  VectorXd b;             // 向量b，用于特征点优化中的信息向量
  double dep_gradient;    // 深度梯度，用于深度估计优化
};

// 每个特征点的总体信息，包含在不同帧中的观测
class FeaturePerId
{
public:
  const int feature_id;                      // 特征点的唯一ID
  int start_frame;                           // 特征点首次出现的帧索引
  vector<FeaturePerFrame> feature_per_frame; // 该特征点在不同帧的观测数据

  int used_num;                              // 记录该特征点被使用的次数
  bool is_outlier;                           // 标记该特征点是否为异常点
  bool is_margin;                            // 标记该特征点是否在边缘化时被移除
  double estimated_depth;                    // 估计的深度值
  int solve_flag;                            // 解算状态，0表示未解算，1表示解算成功，2表示解算失败

  Vector3d gt_p;                             // 真值坐标，用于对比验证（一般用于测试）

  // 构造函数，初始化特征点ID及其初始帧索引
  FeaturePerId(int _feature_id, int _start_frame)
      : feature_id(_feature_id), start_frame(_start_frame),
        used_num(0), estimated_depth(-1.0), solve_flag(0)
  {
  }

  // 返回特征点的最后一个观测帧的索引
  int endFrame();
};

class LineFeaturePerFrame
{
public:
  LineFeaturePerFrame(const Eigen::Matrix<double, 12, 1> &_line, double td)
  {
    start_point.x() = _line(0);
    start_point.y() = _line(1);
    start_point.z() = _line(2);

    end_point.x() = _line(3);
    end_point.y() = _line(4);
    end_point.z() = _line(5);

    start_uv.x() = _line(6);
    start_uv.y() = _line(7);

    end_uv.x() = _line(8);
    end_uv.y() = _line(9);

    velocity.x() = _line(10);
    velocity.y() = _line(11);

    cur_td = td;
  }

  double cur_td;         // 时间延迟
  Vector3d start_point;  // 起点在相机坐标系中的归一化坐标
  Vector3d end_point;    // 终点在相机坐标系中的归一化坐标
  Vector2d start_uv;     // 起点像素坐标
  Vector2d end_uv;       // 终点像素坐标
  Vector2d velocity;     // 线特征速度信息
  double start_z;        // 起点深度
  double end_z;          // 终点深度

  MatrixXd A_start, A_end; // 优化中的信息矩阵
  VectorXd b_start, b_end; // 优化中的信息向量
  double dep_gradient_start, dep_gradient_end; // 深度梯度
};

class LineFeaturePerId
{
public:
  const int line_id;                         // 线特征的唯一ID
  int start_frame;                           // 线特征首次出现的帧索引
  vector<LineFeaturePerFrame> feature_per_frame; // 该线特征在不同帧的观测数据

  int used_num;                              // 记录该线特征被使用的次数
  bool is_outlier;                           // 标记该线特征是否为异常
  bool is_margin;                            // 标记该线特征是否被边缘化
  double estimated_depth_start;              // 起点估计深度
  double estimated_depth_end;                // 终点估计深度
  int solve_flag;                            // 解算状态（0未解算，1成功，2失败）

  Vector3d gt_p_start;                       // 起点真值坐标
  Vector3d gt_p_end;                         // 终点真值坐标

  LineFeaturePerId(int _line_id, int _start_frame)
      : line_id(_line_id), start_frame(_start_frame),
        used_num(0), solve_flag(0)
  {
  }

  int endFrame();
  void addFrame(int frame_count, int camera_id, const LineFeaturePerFrame &line_frame)
  {
    feature_per_frame.push_back(line_frame);
  }
};

class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();
    int getLineFeatureCount();// 新增线特征计数

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void setLineDepth(const VectorXd &x);// 新增线特征的深度设置
    void removeFailures();
    void removeLineFailures();// todo:新增线特征的移除失败
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    VectorXd getLineDepthVector();// todo:新增线特征的深度向量

    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void triangulateLinepoint(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);// todo:新增线特征的三角化

    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature;
    list<LineFeaturePerId> line_feature; // 新增线特征列表
    int last_track_num;
    int last_line_track_num; // 新增线特征计数

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    double compensatedLineParallax2(const LineFeaturePerId &it_per_id, int frame_count); // 新增线特征的平行度补偿
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif