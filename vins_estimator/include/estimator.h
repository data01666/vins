#ifndef _ESTIMATOR_H_
#define _ESTIMATOR_H_

//#include "parameters.h"
#include "feature_manager.h"
//#include "utility/utility.h"
//#include "utility/tic_toc.h"
//#include "initial/solve_5pts.h"
//#include "initial/initial_sfm.h"
//#include "initial/initial_alignment.h"
//#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include "backend.h"
#include "initial.h"

//#include <unordered_map>
//#include <queue>
//#include <opencv2/core/eigen.hpp>

class Backend;

class Estimator
{
  public:
    Estimator();
    /**
     * @brief 从 yaml 配置文件中设置参数
     */
    void setParameter();

    // ********************接口函数********************
    /**
     * @brief 进行 IMU 的预积分处理
     * @param 输入参数：double t，线性加速度（linear_acceleration），角速度（angular_velocity）
     */
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);

    /**
     * @brief VIO 的主处理函数
     * @param 输入参数：点特征数据（image），线特征数据（line），消息头（header）
     */
    //void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,  const std_msgs::Header &header);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const map<int, vector<pair<int, Eigen::Matrix<double, 12, 1>>>> &line, const std_msgs::Header &header);

    /**
     * @brief 添加线特征数据
     * @param 输入参数：线特征数据（line），消息头（header）
     */
    void addline(const map<int, vector<pair<int, Eigen::Matrix<double, 12, 1>>>> &line);

    /**
     * @brief 设置重定位帧
     * @param 输入参数：帧时间戳（_frame_stamp），帧索引（_frame_index），匹配点集合（_match_points），重定位位移（_relo_t）和旋转（_relo_r）
     */
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // ********************常规函数********************
    /**
     * @brief 清除状态，用于系统重启
     */
    void clearState();

    /**
     * @brief 校准相机到 IMU 的旋转
     */
    void calibrationExRotation();

    // ********************初始化函数组********************
    /**
     * @brief 初始化系统
     * @param 输入参数：消息头（header）
     */
    void initial(const std_msgs::Header &header);

  public:
    Backend backend;
    Initial intializer;

    // @param 状态标志
    enum SolverFlag
    {
        INITIAL,        // 初始化状态
        NON_LINEAR      // 非线性优化状态
    };

    // @param 边缘化标志
    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,         // 边缘化旧帧
        MARGIN_SECOND_NEW = 1   // 边缘化次新帧
    };

    // 状态标志
    SolverFlag solver_flag;
    MarginalizationFlag marginalization_flag;

    // 状态标志
    bool first_imu; // 是否为第一帧 IMU 数据
    bool failure_occur; // 系统是否发生错误

    // 当前的重力方向，初始化为0，IMU对齐后为 gc0，初始化后为 gw
    Vector3d g;

    // 外参：相机到 IMU 的旋转和平移
    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    // 系统状态：相机在世界坐标系中的位置、速度、姿态，IMU 偏置
    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];
    double td; // 时间偏移

    // 临时缓存
    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;

    // IMU 预积分信息
    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    // IMU 数据缓存
    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    // 当前帧计数（最大值为10）
    int frame_count;

    // 特征管理器
    FeatureManager f_manager;

    // 运动估计器，用于估计运动信息
    MotionEstimator m_estimator;

    // 外参初始化
    InitialEXRotation initial_ex_rotation;

    // 关键帧位姿
    vector<Vector3d> key_poses;
    double initial_timestamp;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    // 重定位变量
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];

    // 漂移矫正相关
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
};

#endif