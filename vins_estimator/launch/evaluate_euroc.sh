#!/bin/bash

# 设置路径
RESULT_ROOT_DIR=~/result/vins-mono-improved/euroc
DATASET_DIR=~/dataset/EUROC

# 遍历结果文件夹
for RESULT_DIR in "$RESULT_ROOT_DIR"/result_*; do
  if [ -d "$RESULT_DIR" ]; then
    echo "处理目录: $RESULT_DIR"
    
    # 提取数据集名称（如 MH_01_easy）
    DATASET_NAME=$(basename "$RESULT_DIR" | sed 's/result_//')
    echo "检测数据集: $DATASET_NAME"

    # 将 MH_01_easy 转换为 MH01 的格式
    FORMATTED_DATASET_NAME=$(echo "$DATASET_NAME" | sed 's/_//g' | sed 's/easy//g' | sed 's/medium//g' | sed 's/difficult//g')
    echo "格式化后的数据集名称: $FORMATTED_DATASET_NAME"

    # 找到对应的 groundtruth 文件路径
    GROUNDTRUTH_FILE="$DATASET_DIR/$FORMATTED_DATASET_NAME/mav0/state_groundtruth_estimate0/data.csv"
    if [ ! -f "$GROUNDTRUTH_FILE" ]; then
      echo "错误: Groundtruth 文件 $GROUNDTRUTH_FILE 不存在，跳过 $RESULT_DIR"
      continue
    fi

    # 进入结果目录
    cd "$RESULT_DIR" || { echo "无法进入目录 $RESULT_DIR"; continue; }

    # 转换 groundtruth 为 TUM 格式
    echo "转换 Groundtruth 为 TUM 格式..."
    evo_traj euroc "$GROUNDTRUTH_FILE" --save_as_tum || { echo "Groundtruth 转换失败"; continue; }

    # 无回环评估
    if [ -f "vins_result_no_loop.csv" ]; then
      echo "无回环评估..."
      # ATE
      evo_ape euroc "$GROUNDTRUTH_FILE" vins_result_no_loop.csv -va --save_results vins_no_loop_ate.zip
      # RPE
      evo_rpe euroc "$GROUNDTRUTH_FILE" vins_result_no_loop.csv -va --save_results vins_no_loop_rpe.zip
      # 轨迹图
      evo_traj tum vins_result_no_loop.csv --ref=data.tum --plot_mode=xyz --align --correct_scale --save_plot vins_no_loop_traj_plot.pdf
    else
      echo "跳过无回环评估 (vins_result_no_loop.csv 不存在)"
    fi

    # 有回环评估
    if [ -f "vins_result_loop.csv" ]; then
      echo "有回环评估..."
      # ATE
      evo_ape euroc "$GROUNDTRUTH_FILE" vins_result_loop.csv -va --save_results vins_loop_ate.zip
      # RPE
      evo_rpe euroc "$GROUNDTRUTH_FILE" vins_result_loop.csv -va --save_results vins_loop_rpe.zip
      # 轨迹图
      evo_traj tum vins_result_loop.csv --ref=data.tum --plot_mode=xyz --align --correct_scale --save_plot vins_loop_traj_plot.pdf
    else
      echo "跳过有回环评估 (vins_result_loop.csv 不存在)"
    fi

    # 返回上一级目录
    cd - > /dev/null || exit
  else
    echo "跳过无效目录: $RESULT_DIR"
  fi
done

echo "评估完成！"

