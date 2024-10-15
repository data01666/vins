#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"


class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1>
{
public:
    // 构造函数，初始化类并设置特征点
    // 参数：
    //   _pts_i - 特征点在关键帧i中的坐标（归一化平面）
    //   _pts_j - 特征点在关键帧j中的坐标（归一化平面）
    ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);

    // Evaluate函数，计算投影误差和雅可比矩阵（如果需要）
    // 参数：
    //   parameters - 输入的优化变量，包括位姿i和j，IMU到相机的外参，特征点的逆深度
    //   residuals  - 输出的误差向量
    //   jacobians  - 输出的雅可比矩阵
    // 返回值：
    //   bool - 表示计算是否成功
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    // check函数，用于验证解析雅可比矩阵是否正确
    // 使用数值方法计算雅可比矩阵，并与解析方法的结果进行对比
    // 参数：
    //   parameters - 输入的优化变量
    void check(double **parameters);

    // 成员变量：
    Eigen::Vector3d pts_i;             // 特征点在关键帧i中的3D坐标（归一化平面）
    Eigen::Vector3d pts_j;             // 特征点在关键帧j中的3D坐标（归一化平面）
    Eigen::Matrix<double, 2, 3> tangent_base; // 单位球面误差时的切平面基，用于降维处理

    // 静态成员变量：
    static Eigen::Matrix2d sqrt_info;  // 误差加权的平方根信息矩阵
    static double sum_t;               // 累积计时器，用于统计Evaluate函数的总运行时间
};

