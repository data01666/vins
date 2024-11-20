catkin_make -C /root/code/vins-improved ;
source /root/code/vins-improved/devel/setup.bash


# Get the current directory
CURR_DIR=$(pwd)
# Get the location of the viral package
roscd vins_estimator
PACKAGE_DIR=$(pwd)
# Return to the current dir, print the directions
cd $CURR_DIR
echo CURRENT DIR: $CURR_DIR
echo VINS DIR:    $PACKAGE_DIR

export EPOC_DIR=/root/result/vins-mono-improved/ntuviral/result
export DATASET_LOCATION=/root/dataset/ntuviral

#region Run each dataset with VINS ------------------------------------------------------------------------------------

wait;
./run_one_bag_ntuviral.sh $EPOC_DIR $DATASET_LOCATION eee_01 0 0 450 0 1 0.75 -1;
mkdir -p $EPOC_DIR/result_eee_01;
mv $EPOC_DIR/extrinsic_parameter.csv $EPOC_DIR/result_eee_01;
mv $EPOC_DIR/vins_result_loop.csv $EPOC_DIR/result_eee_01;
mv $EPOC_DIR/vins_result_no_loop.csv $EPOC_DIR/result_eee_01;

wait;
./run_one_bag_ntuviral.sh $EPOC_DIR $DATASET_LOCATION eee_02 0 0 450 0 1 0.75 -1;
mkdir -p $EPOC_DIR/result_eee_02;
mv $EPOC_DIR/extrinsic_parameter.csv $EPOC_DIR/result_eee_02;
mv $EPOC_DIR/vins_result_loop.csv $EPOC_DIR/result_eee_02;
mv $EPOC_DIR/vins_result_no_loop.csv $EPOC_DIR/result_eee_02;

wait;
./run_one_bag_ntuviral.sh $EPOC_DIR $DATASET_LOCATION eee_03 0 0 450 0 1 0.75 -1;
mkdir -p $EPOC_DIR/result_eee_03;
mv $EPOC_DIR/extrinsic_parameter.csv $EPOC_DIR/result_eee_03;
mv $EPOC_DIR/vins_result_loop.csv $EPOC_DIR/result_eee_03;
mv $EPOC_DIR/vins_result_no_loop.csv $EPOC_DIR/result_eee_03;

wait;
./run_one_bag_ntuviral.sh $EPOC_DIR $DATASET_LOCATION nya_01 0 0 450 0 1 0.75 -1;
mkdir -p $EPOC_DIR/result_nya_01;
mv $EPOC_DIR/extrinsic_parameter.csv $EPOC_DIR/result_nya_01;
mv $EPOC_DIR/vins_result_loop.csv $EPOC_DIR/result_nya_01;
mv $EPOC_DIR/vins_result_no_loop.csv $EPOC_DIR/result_nya_01;

wait;
./run_one_bag_ntuviral.sh $EPOC_DIR $DATASET_LOCATION nya_02 0 0 450 0 1 0.75 -1;
mkdir -p $EPOC_DIR/result_nya_02;
mv $EPOC_DIR/extrinsic_parameter.csv $EPOC_DIR/result_nya_02;
mv $EPOC_DIR/vins_result_loop.csv $EPOC_DIR/result_nya_02;
mv $EPOC_DIR/vins_result_no_loop.csv $EPOC_DIR/result_nya_02;

wait;
./run_one_bag_ntuviral.sh $EPOC_DIR $DATASET_LOCATION nya_03 0 0 450 0 1 0.75 -1;
mkdir -p $EPOC_DIR/result_nya_03;
mv $EPOC_DIR/extrinsic_parameter.csv $EPOC_DIR/result_nya_03;
mv $EPOC_DIR/vins_result_loop.csv $EPOC_DIR/result_nya_03;
mv $EPOC_DIR/vins_result_no_loop.csv $EPOC_DIR/result_nya_03;

wait;
./run_one_bag_ntuviral.sh $EPOC_DIR $DATASET_LOCATION sbs_01 0 0 450 0 1 0.75 -1;
mkdir -p $EPOC_DIR/result_sbs_01;
mv $EPOC_DIR/extrinsic_parameter.csv $EPOC_DIR/result_sbs_01;
mv $EPOC_DIR/vins_result_loop.csv $EPOC_DIR/result_sbs_01;
mv $EPOC_DIR/vins_result_no_loop.csv $EPOC_DIR/result_sbs_01;

wait;
./run_one_bag_ntuviral.sh $EPOC_DIR $DATASET_LOCATION sbs_02 0 0 450 0 1 0.75 -1;
mkdir -p $EPOC_DIR/result_sbs_02;
mv $EPOC_DIR/extrinsic_parameter.csv $EPOC_DIR/result_sbs_02;
mv $EPOC_DIR/vins_result_loop.csv $EPOC_DIR/result_sbs_02;
mv $EPOC_DIR/vins_result_no_loop.csv $EPOC_DIR/result_sbs_02;

wait;
./run_one_bag_ntuviral.sh $EPOC_DIR $DATASET_LOCATION sbs_03 0 0 450 0 1 0.75 -1;
mkdir -p $EPOC_DIR/result_sbs_03;
mv $EPOC_DIR/extrinsic_parameter.csv $EPOC_DIR/result_sbs_03;
mv $EPOC_DIR/vins_result_loop.csv $EPOC_DIR/result_sbs_03;
mv $EPOC_DIR/vins_result_no_loop.csv $EPOC_DIR/result_sbs_03;

#endregion Run each dataset with VINS ---------------------------------------------------------------------------------



#region ## Poweroff ---------------------------------------------------------------------------------------------------

wait;
# poweroff;

#endregion ## Poweroff ------------------------------------------------------------------------------------------------
