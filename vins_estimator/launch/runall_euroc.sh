#!/bin/bash

# 编译工作空间并设置环境变量
catkin_make -C /root/code/vins-improved || { echo "catkin_make 失败"; exit 1; }
source /root/code/vins-improved/devel/setup.bash

export EPOC_DIR=/root/result/vins-mono-improved/euroc
export DATASET_LOCATION=/root/dataset/EUROC

# 定义数据集列表
DATASETS=("MH_01_easy" "MH_02_easy" "MH_03_medium" "MH_04_difficult" "MH_05_difficult" "V1_01_easy" "V1_02_medium" "V1_03_difficult" "V2_01_easy" "V2_02_medium" "V2_03_difficult")

# 函数：运行单个bag文件
run_one_bag() {
  local dataset=$1
  echo "开始运行数据集: $dataset"
  ./run_one_bag_euroc.sh "$EPOC_DIR" "$DATASET_LOCATION" "$dataset" || {
    echo "运行 $dataset 失败";
    return 1;
  }
  echo "完成运行数据集: $dataset"
}

# 遍历数据集并运行
for dataset in "${DATASETS[@]}"; do
  run_one_bag "$dataset" || { echo "运行中断"; exit 1; }
  wait
done

echo "所有数据集运行完成"