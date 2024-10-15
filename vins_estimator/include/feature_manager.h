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

class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
    }
    double cur_td;
    Vector3d point;
    Vector2d uv;
    Vector2d velocity;
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame;

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

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

    double cur_td;
    Vector3d start_point;
    Vector3d end_point;
    Vector2d start_uv;
    Vector2d end_uv;
    Vector2d velocity;
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
  };

class LineFeaturePerId
{
public:
  const int line_id;
  int start_frame;
  vector<LineFeaturePerFrame> feature_per_frame;

  int used_num;
  bool is_outlier;
  bool is_margin;
  int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

  Vector3d gt_p;//todo:什么意思？

  LineFeaturePerId(int _line_id, int _start_frame)
      : line_id(_line_id), start_frame(_start_frame),
        used_num(0), solve_flag(0)
  {
  }

  int endFrame();
  void addFrame(int frame_count, int camera_id, const LineFeaturePerFrame &line_frame)
  {
    // 根据相机ID或其他信息来选择性地添加帧
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
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
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