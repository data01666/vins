#!/bin/bash

# 设置输出目录、数据集路径、实验名称和一些参数的默认值
export EPOC_DIR=$1;                      # 结果输出根目录
export DATASET_LOCATION=$2;               # 数据集存放路径
export EXP_NAME=$3;                       # 实验名称，用于命名输出目录
export CAPTURE_SCREEN=${4:-0}             # 是否捕获屏幕视频，默认0（不捕获）
export LOG_DATA=${5:-0}                   # 是否记录数据，默认0（不记录）
export LOG_DUR=${6:-300}                  # 日志记录持续时间，默认300秒
export FUSE_UWB=${7:-0}                   # 是否融合UWB数据，默认0（不融合）
export FUSE_VIS=${8:-0}                   # 是否融合视觉信息，默认0（不融合）
export UWB_BIAS=${9:-0}                   # UWB偏置，默认0
export ANC_ID_MAX=${10:-4}                # 锚点ID的最大值，默认4

# 获取bag文件的时长，并设置日志记录时长
export BAG_DUR=$(rosbag info $DATASET_LOCATION/$EXP_NAME/$EXP_NAME.bag | grep 'duration' | sed 's/^.*(//' | sed 's/s)//');
let LOG_DUR=BAG_DUR+20                    # 记录时长为bag文件时长加20秒

echo "BAG文件时长:" $BAG_DUR "=> 日志记录时长:" $LOG_DUR;

# 计算ANC_MAX值，增加1
let ANC_MAX=ANC_ID_MAX+1

# 设置实验结果输出目录
export EXP_OUTPUT_DIR=$EPOC_DIR/result_${EXP_NAME};
mkdir -p $EXP_OUTPUT_DIR;
if ((FUSE_VIS==1))
then
export EXP_OUTPUT_DIR=${EXP_OUTPUT_DIR}_vis;  # 如果FUSE_VIS为1，则在目录名后加_vis
fi
echo 输出目录: $EXP_OUTPUT_DIR;

# 启动VINS-Mono，指定日志输出目录、数据集路径和自动运行选项
roslaunch vins_estimator run_ntuviral.launch log_dir:=$EXP_OUTPUT_DIR \
  autorun:=true \
  bag_file:=$DATASET_LOCATION/$EXP_NAME/$EXP_NAME.bag \
&

# 如果需要捕获屏幕视频，启动ffmpeg进行录屏
if [[ $CAPTURE_SCREEN -eq 1 ]]
then
    echo 开启屏幕捕获;
    sleep 1;
    ffmpeg -video_size 1920x1080 -framerate 1 -f x11grab -i :0.0+1920,0 \
    -loglevel quiet -t $LOG_DUR -y $EXP_OUTPUT_DIR/$EXP_NAME.mp4 \
    &
else
    echo 关闭屏幕捕获;
    sleep 1;
fi

# 如果需要记录数据，记录各个话题的数据到CSV文件
if [[ $LOG_DATA -eq 1 ]]
then
    echo 开始记录数据;
    sleep 5;
    rosparam dump $EXP_OUTPUT_DIR/allparams.yaml;                   # 保存所有参数到yaml文件
    timeout $LOG_DUR rostopic echo -p --nostr --noarr /vins_estimator/imu_propagate \
    > $EXP_OUTPUT_DIR/vio_odom.csv  \
    & \
    timeout $LOG_DUR rostopic echo -p --nostr --noarr /vins_estimator/odometry \
    > $EXP_OUTPUT_DIR/opt_odom.csv \
    & \
    timeout $LOG_DUR rostopic echo -p --nostr --noarr /leica/pose/relative \
    > $EXP_OUTPUT_DIR/leica_pose.csv \
    & \
    timeout $LOG_DUR rostopic echo -p --nostr --noarr /dji_sdk/imu \
    > $EXP_OUTPUT_DIR/dji_sdk_imu.csv \
    & \
    timeout $LOG_DUR rostopic echo -p --nostr --noarr /imu/imu \
    > $EXP_OUTPUT_DIR/vn100_imu.csv \
    ;
else
    sleep $LOG_DUR;
fi