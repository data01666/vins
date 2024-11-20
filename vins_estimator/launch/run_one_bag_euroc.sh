#!/bin/bash

# 检查参数是否提供
if [ $# -ne 3 ]; then
  echo "用法: $0 <输出目录> <数据集路径> <实验名称>"
  exit 1
fi

export EPOC_DIR=$1
export DATASET_LOCATION=$2
export EXP_NAME=$3

export EXP_OUTPUT_DIR=$EPOC_DIR/result_${EXP_NAME}
echo "输出目录: $EXP_OUTPUT_DIR"

BAG_FILE="$DATASET_LOCATION/$EXP_NAME.bag"
if [ ! -f "$BAG_FILE" ]; then
  echo "错误: 数据集文件 $BAG_FILE 不存在"
  exit 1
fi

mkdir -p $EXP_OUTPUT_DIR || { echo "无法创建输出目录 $EXP_OUTPUT_DIR"; exit 1; }

# 清理旧进程和日志
pkill -f roscore
pkill -f roslaunch
rm -rf /root/.ros/log/*

# 启动 roscore
roscore &
ROSCORE_PID=$!
sleep 5

trap "echo '清理进程...'; kill $ROSCORE_PID $VINS_LAUNCH_PID $RVIZ_LAUNCH_PID; exit" SIGINT SIGTERM

# 启动 VINS-Mono 和 RViz
roslaunch vins_estimator euroc.launch &
VINS_LAUNCH_PID=$!
roslaunch vins_estimator vins_rviz.launch &
RVIZ_LAUNCH_PID=$!

sleep 5  # 等待 VINS-Mono 启动

# 播放数据集
rosbag play "$BAG_FILE" --clock

# 播放完成后延迟 10 秒
sleep 15
echo "播放完成，停止所有 ROS 节点..."
kill $VINS_LAUNCH_PID 2>/dev/null || echo "VINS-Mono 已退出"
kill $RVIZ_LAUNCH_PID 2>/dev/null || echo "RViz 已退出"

# 检查结果文件
RESULT_LOOP="/root/result/vins-mono-improved/euroc/vins_result_loop.csv"
RESULT_NO_LOOP="/root/result/vins-mono-improved/euroc/vins_result_no_loop.csv"

if [ ! -f "$RESULT_LOOP" ] || [ ! -f "$RESULT_NO_LOOP" ]; then
  echo "错误: 结果文件未生成"
  kill $ROSCORE_PID
  exit 1
fi

# 移动结果文件
mv "$RESULT_LOOP" "$EXP_OUTPUT_DIR"
mv "$RESULT_NO_LOOP" "$EXP_OUTPUT_DIR"

# 停止 roscore
kill $ROSCORE_PID 2>/dev/null || echo "ROS Master 已退出"

echo "实验完成，结果已保存到 $EXP_OUTPUT_DIR"
