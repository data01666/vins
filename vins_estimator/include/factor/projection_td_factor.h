#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"

class ProjectionTdFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1>
{
  public:
    ProjectionTdFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
    				   const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
    				   const double _td_i, const double _td_j, const double _row_i, const double _row_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Vector3d velocity_i, velocity_j;
    double td_i, td_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    double row_i, row_j;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};
/*
class lineProjectionTdFactor : public ceres::SizedCostFunction<4, 7, 7, 7, 1, 1>
{
public:
	// 构造函数
	lineProjectionTdFactor(const Eigen::Vector3d &_start_i, const Eigen::Vector3d &_end_i,
						   const Eigen::Vector3d &_start_j, const Eigen::Vector3d &_end_j,
						   const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
						   const double _td_i, const double _td_j,
						   const double _row_i, const double _row_j);

	// 误差计算和雅可比矩阵计算
	virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

	// 成员变量
	Eigen::Vector3d start_i, end_i;     // 起点和终点在帧i的3D坐标
	Eigen::Vector3d start_j, end_j;     // 起点和终点在帧j的3D坐标
	Eigen::Vector2d velocity_i, velocity_j;	// 线段速度
	double td_i, td_j;					// 时间延迟
	Eigen::Matrix<double, 4, 3> line_tangent_base; // 单位球面误差时的切平面基，用于降维处理
	double row_i, row_j;				// 行数（图像行号）

	// 误差的权重信息矩阵
	static Eigen::Matrix2d line_sqrt_info;
};
*/
