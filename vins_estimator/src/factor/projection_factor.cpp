#include "../../include/factor/projection_factor.h"

Eigen::Matrix2d ProjectionFactor::sqrt_info;
double ProjectionFactor::sum_t;

// 构造函数，接受两个特征点的3D坐标
ProjectionFactor::ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j) : pts_i(_pts_i), pts_j(_pts_j)
{
#ifdef UNIT_SPHERE_ERROR
    // 定义两个辅助向量 b1 和 b2，用于计算单位球面上的切线基底
    Eigen::Vector3d b1, b2;

    // 将特征点 pts_j 归一化为单位向量 a
    Eigen::Vector3d a = pts_j.normalized();

    // 定义临时向量 tmp，初始值为 (0, 0, 1) 对应 z 轴
    Eigen::Vector3d tmp(0, 0, 1);

    // 判断 a 是否与 z 轴向量相同，如果相同则将 tmp 设置为 (1, 0, 0) 对应 x 轴
    if(a == tmp)
        tmp << 1, 0, 0;

    // 计算 b1 向量，将 tmp 在 a 方向上的投影去除并归一化
    b1 = (tmp - a * (a.transpose() * tmp)).normalized();

    // 计算 b2 向量，与向量 a 和 b1 正交，使用叉乘计算
    b2 = a.cross(b1);

    // 将 b1 和 b2 分别作为切线基底的第一行和第二行（2x3矩阵）
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
#endif
};

/*
 *接受优化器提供的参数，计算误差残差并存储在 residuals 中，同时计算相对于各个变量的雅可比矩阵并存储在 jacobians 中（如果需要）
 */
bool ProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc; // 计时器，用于记录函数执行时间

    // 提取关键帧 i 的平移和旋转
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]); // 关键帧 i 的平移向量
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]); // 关键帧 i 的旋转四元数

    // 提取关键帧 j 的平移和旋转
    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]); // 关键帧 j 的平移向量
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]); // 关键帧 j 的旋转四元数

    // 提取相机与 IMU 之间的外参（平移和旋转）
    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]); // 相机到 IMU 的平移向量
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]); // 相机到 IMU 的旋转四元数

    // 提取特征点的逆深度
    double inv_dep_i = parameters[3][0]; // 特征点的逆深度

    // 计算特征点在各个坐标系下的位置
    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i; // 从逆深度恢复特征点在关键帧 i 相机坐标系下的3D坐标
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic; // 将特征点从相机坐标系转换到 IMU 坐标系
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi; // 将特征点从 IMU 坐标系转换到世界坐标系
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj); // 将特征点从世界坐标系转换到关键帧 j 的 IMU 坐标系
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic); // 将特征点从 IMU 坐标系转换到关键帧 j 的相机坐标系

    // 使用 Eigen::Map 映射 residuals 指针为 Eigen 向量，方便操作
    Eigen::Map<Eigen::Vector2d> residual(residuals);

#ifdef UNIT_SPHERE_ERROR
    // 使用单位球面误差：计算特征点在单位球面上的投影误差
    // 先将特征点坐标归一化并转换到单位球面，计算与观测值的差异
    residual = tangent_base * (pts_camera_j.normalized() - pts_j.normalized());
#else
    // 使用普通重投影误差：计算特征点在相机归一化平面的投影误差
    double dep_j = pts_camera_j.z(); // 获取特征点在 j 相机坐标系下的深度
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>(); // 计算归一化后的投影误差
#endif

    // 使用平方根信息矩阵对误差进行加权
    residual = sqrt_info * residual;

    if (jacobians) // 如果要求计算雅可比矩阵
    {
        // 将四元数 Qi、Qj、qic 转换为旋转矩阵
        Eigen::Matrix3d Ri = Qi.toRotationMatrix(); // 关键帧 i 的旋转矩阵
        Eigen::Matrix3d Rj = Qj.toRotationMatrix(); // 关键帧 j 的旋转矩阵
        Eigen::Matrix3d ric = qic.toRotationMatrix(); // 相机到 IMU 的旋转矩阵

        // 定义降维矩阵，用于将 3D 误差向量投影到 2D 平面
        Eigen::Matrix<double, 2, 3> reduce(2, 3);

#ifdef UNIT_SPHERE_ERROR
        // 计算单位球面误差的降维矩阵
        double norm = pts_camera_j.norm(); // 计算点到原点的距离（单位球面半径）
        Eigen::Matrix3d norm_jaco;
        double x1 = pts_camera_j(0), x2 = pts_camera_j(1), x3 = pts_camera_j(2);
        norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), -x1 * x2 / pow(norm, 3), -x1 * x3 / pow(norm, 3),
                     -x1 * x2 / pow(norm, 3), 1.0 / norm - x2 * x2 / pow(norm, 3), -x2 * x3 / pow(norm, 3),
                     -x1 * x3 / pow(norm, 3), -x2 * x3 / pow(norm, 3), 1.0 / norm - x3 * x3 / pow(norm, 3);
        reduce = tangent_base * norm_jaco; // 投影到单位球面的切平面
#else
        // 计算普通重投影误差的降维矩阵
        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
                  0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
#endif
        reduce = sqrt_info * reduce; // 将降维矩阵与信息矩阵相乘，加权误差

        // 计算相对于关键帧 i 位姿的雅可比矩阵
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose(); // 平移部分
            jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i); // 旋转部分
            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero(); // 第 7 列填充为 0，用于保持与其他代码一致
        }

        // 计算相对于关键帧 j 位姿的雅可比矩阵
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose(); // 平移部分
            jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j); // 旋转部分
            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero(); // 第 7 列填充为 0
        }

        // 计算相对于相机与 IMU 外参的雅可比矩阵
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity()); // 外参平移部分
            Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
            jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                                     Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic)); // 外参旋转部分
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero(); // 第 7 列填充为 0
        }

        // 计算相对于逆深度的雅可比矩阵
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i * -1.0 / (inv_dep_i * inv_dep_i); // 逆深度影响
        }
    }
    sum_t += tic_toc.toc(); // 更新累加计算时间
    return true; // 返回 true 表示函数执行成功
}

// 验证雅可比矩阵的计算正确性,通过将解析雅可比与数值雅可比进行对比,用于调试
void ProjectionFactor::check(double **parameters)
{
    double *res = new double[15]; // 误差向量
    double **jaco = new double *[4]; // 解析雅可比矩阵
    jaco[0] = new double[2 * 7];
    jaco[1] = new double[2 * 7];
    jaco[2] = new double[2 * 7];
    jaco[3] = new double[2 * 1];
    Evaluate(parameters, res, jaco);
    puts("check begins");

    puts("my");

    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 1>>(res).transpose() << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[0]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[1]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[2]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Vector2d>(jaco[3]) << std::endl
              << std::endl;

    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
    double inv_dep_i = parameters[3][0];

    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);


    Eigen::Vector2d residual;
#ifdef UNIT_SPHERE_ERROR 
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j.normalized());
#else
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();
#endif
    residual = sqrt_info * residual;

    puts("num");
    std::cout << residual.transpose() << std::endl;

    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 19> num_jacobian;
    for (int k = 0; k < 19; k++)
    {
        // 重置关键帧 i 和 j 的位置、旋转，及相机与 IMU 的外参和逆深度
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
        double inv_dep_i = parameters[3][0];

        int a = k / 3, b = k % 3;
        Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

        // 根据参数索引选择性地为 Pi、Qi、Pj、Qj、tic、qic、inv_dep_i 添加微小扰动
        if (a == 0)
            Pi += delta;
        else if (a == 1)
            Qi = Qi * Utility::deltaQ(delta);
        else if (a == 2)
            Pj += delta;
        else if (a == 3)
            Qj = Qj * Utility::deltaQ(delta);
        else if (a == 4)
            tic += delta;
        else if (a == 5)
            qic = qic * Utility::deltaQ(delta);
        else if (a == 6)
            inv_dep_i += delta.x();

        // 计算扰动后特征点在各坐标系中的位置及残差
        Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
        Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
        Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
        Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
        Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);

        Eigen::Vector2d tmp_residual;
#ifdef UNIT_SPHERE_ERROR 
        tmp_residual =  tangent_base * (pts_camera_j.normalized() - pts_j.normalized());
#else
        double dep_j = pts_camera_j.z();
        tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();
#endif
        tmp_residual = sqrt_info * tmp_residual;
        // 计算数值雅可比：以扰动后的残差与原始残差之差除以 eps
        num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }
    std::cout << num_jacobian << std::endl;
}

/*
Eigen::Matrix2d lineProjectionFactor::line_sqrt_info;
// 构造函数，接受两个线段的起点和终点
lineProjectionFactor::lineProjectionFactor(const Eigen::Vector3d &_start_i, const Eigen::Vector3d &_end_i,
    const Eigen::Vector3d &_start_j, const Eigen::Vector3d &_end_j)
    : start_i(_start_i), end_i(_end_i), start_j(_start_j), end_j(_end_j)
{
    #ifdef UNIT_SPHERE_ERROR
    // 定义两个辅助向量 b1 和 b2，用于计算单位球面上的切线基底
    Eigen::Vector3d b1, b2;
    Eigen::Vector3d a1 = _start_j.normalized();
    Eigen::Vector3d a2 = _end_j.normalized();
    Eigen::Vector3d tmp1(0, 0, 1);
    Eigen::Vector3d tmp2(0, 0, 1);
    if(a1 == tmp)
        tmp << 1, 0, 0;
    if(a2 == tmp)
        tmp << 1, 0, 0;
    b1 = (tmp1 - a1 * (a1.transpose() * tmp1)).normalized();
    b2 = a1.cross(b1);
    line_tangent_base.block<1, 3>(0, 0) = b1.transpose();
    line_tangent_base.block<1, 3>(1, 0) = b2.transpose();
    b1 = (tmp2 - a2 * (a2.transpose() * tmp2)).normalized();
    b2 = a2.cross(b1);
    line_tangent_base.block<1, 3>(2, 0) = b1.transpose();
    line_tangent_base.block<1, 3>(3, 0) = b2.transpose();
    #endif
}

// 计算投影误差和雅可比矩阵
// 计算投影误差和雅可比矩阵
bool lineProjectionFactor::Evaluate(double const *const *parameters, double *residuals,double **jacobians) const
{
    // 提取关键帧 i 的平移和旋转
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    // 提取关键帧 j 的平移和旋转
    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    // 提取相机与 IMU 之间的外参（平移和旋转）
    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    // 提取特征点的逆深度
    double start_dep_i = parameters[4][0];
    double end_dep_i = parameters[4][1];

    // 计算特征点在各个坐标系下的位置
    Eigen::Vector3d start_camera_i = start_i / start_dep_i;
    Eigen::Vector3d end_camera_i = end_i / end_dep_i;
    Eigen::Vector3d start_imu_i = qic * start_camera_i + tic;
    Eigen::Vector3d end_imu_i = qic * end_camera_i + tic;
    Eigen::Vector3d start_w = Qi * start_imu_i + Pi;
    Eigen::Vector3d end_w = Qi * end_imu_i + Pi;
    Eigen::Vector3d start_imu_j = Qj.inverse() * (start_w - Pj);
    Eigen::Vector3d end_imu_j = Qj.inverse() * (end_w - Pj);
    Eigen::Vector3d start_camera_j = qic.inverse() * (start_imu_j - tic);
    Eigen::Vector3d end_camera_j = qic.inverse() * (end_imu_j - tic);

    // 使用 Eigen::Map 映射 residuals 指针为 Eigen 向量，方便操作
    Eigen::Map<Eigen::Vector4d> residual(residuals);  // 4D残差向量

#ifdef UNIT_SPHERE_ERROR
    // 使用单位球面误差：计算特征点在单位球面上的投影误差
    residual.head<2>() = line_tangent_base.block<2, 3>(0, 0) * (start_camera_j.normalized() - start_j.normalized());
    residual.tail<2>() = line_tangent_base.block<2, 3>(2, 0) * (end_camera_j.normalized() - end_j.normalized());
#else
    // 使用普通重投影误差：计算特征点在相机归一化平面的投影误差
    double start_dep_j = start_camera_j.z();
    double end_dep_j = end_camera_j.z();
    residual.head<2>() = (start_camera_j / start_dep_j).head<2>() - start_j.head<2>();
    residual.tail<2>() = (end_camera_j / end_dep_j).head<2>() - end_j.head<2>();
#endif

    // 使用平方根信息矩阵对误差进行加权
    residual = line_sqrt_info * residual;

    // 雅可比矩阵计算（这里根据具体情况补充）
    if (jacobians)
    {
        // 处理关键帧i和j位姿的雅可比矩阵，以及逆深度的雅可比矩阵
    }

    return true;
}
*/