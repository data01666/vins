#include "../../include/initial/initial_sfm.h"

GlobalSFM::GlobalSFM(){}

void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{//这一部分代码涉及到了三角化求深度
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++)//feature_num = sfm_f.size() line121
	{//要把待求帧i上所有特征点的归一化坐标和3D坐标(l系上)都找出来
		if (sfm_f[j].state != true)//这个特征点没有被三角化为空间点，跳过这个点的PnP
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)//依次遍历特征j在每一帧中的归一化坐标
		{
			if (sfm_f[j].observation[k].first == i)//如果该特征在帧i上出现过
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);//把在待求帧i上出现过的特征的归一化坐标放到容器中
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);//把在待求帧i上出现过的特征在参考系l的空间坐标放到容器中
				break;//因为一个特征在帧i上只会出现一次，一旦找到了就没有必要再继续找了
			}
		}
	}
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	// 套用openCV的公式，进行PnP求解。
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);//转换成solvePnP能处理的格式
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);//得到了第l帧到第i帧的旋转平移
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);//转换成原有格式
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;//覆盖原先的旋转平移
	P_initial = T_pnp;
	return true;

}

void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	for (int j = 0; j < feature_num; j++)//在所有特征里面依次寻找
	{
		if (sfm_f[j].state == true)//如果这个特征已经三角化过了，那就跳过
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)//如果这个特征在frame0出现过
			{
				point0 = sfm_f[j].observation[k].second;//把他的归一化坐标提取出来
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)//如果这个特征在frame1出现过
			{
				point1 = sfm_f[j].observation[k].second;//如果这个特征在frame1出现过
				has_1 = true;
			}
		}
		if (has_0 && has_1)//如果这两个归一化坐标都存在
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);//根据他们的位姿和归一化坐标，输出在参考系l下的的空间坐标
			sfm_f[j].state = true;// 已经完成三角化，状态更改为true
			sfm_f[j].position[0] = point_3d(0);//把参考系l下的的空间坐标赋值给这个特征点的对象
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

// todo：修改了一下可能有问题
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
                          const Matrix3d relative_R, const Vector3d relative_T,
                          vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
    // 1. 初始化参考帧(l帧)和当前帧的位姿
    feature_num = sfm_f.size();
    q[l].w() = 1; q[l].x() = 0; q[l].y() = 0; q[l].z() = 0; // 参考帧设为原点
    T[l].setZero(); // 参考帧的平移为零
    q[frame_num - 1] = q[l] * Quaterniond(relative_R); // 当前帧的旋转
    T[frame_num - 1] = relative_T; // 当前帧的平移

    // 2. 准备其他帧的旋转和平移容器
    Matrix3d c_Rotation[frame_num];
    Vector3d c_Translation[frame_num];
    Quaterniond c_Quat[frame_num];
    double c_rotation[frame_num][4];
    double c_translation[frame_num][3];
    Eigen::Matrix<double, 3, 4> Pose[frame_num];

    // 3. 设置参考帧和当前帧在相机坐标系下的旋转和平移
    c_Quat[l] = q[l].inverse();
    c_Rotation[l] = c_Quat[l].toRotationMatrix();
    c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
    c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
    c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
    Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
    Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

    // 4. 使用PnP逐帧恢复参考帧后的位姿，并进行三角化
    for (int i = l; i < frame_num - 1 ; i++)
    {
        if (i > l)
        {
            Matrix3d R_initial = c_Rotation[i - 1];
            Vector3d P_initial = c_Translation[i - 1];
            if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) return false;
            c_Rotation[i] = R_initial;
            c_Translation[i] = P_initial;
            c_Quat[i] = c_Rotation[i];
            Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
            Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        }
        triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
    }

    // 5. 三角化参考帧后的所有帧
    for (int i = l + 1; i < frame_num - 1; i++)
        triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);

    // 6. 使用PnP和三角化恢复参考帧之前的位姿
    for (int i = l - 1; i >= 0; i--)
    {
        Matrix3d R_initial = c_Rotation[i + 1];
        Vector3d P_initial = c_Translation[i + 1];
        if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) return false;
        c_Rotation[i] = R_initial;
        c_Translation[i] = P_initial;
        c_Quat[i] = c_Rotation[i];
        Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
        Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
    }

    // 7. 三角化滑窗内所有未被三角化的特征点
    for (int j = 0; j < feature_num; j++)
    {
    	if (sfm_f[j].state == true)
    		continue;
    	if ((int)sfm_f[j].observation.size() >= 2)
    	{
    		Vector2d point0, point1;
    		int frame_0 = sfm_f[j].observation[0].first;
    		point0 = sfm_f[j].observation[0].second;
    		int frame_1 = sfm_f[j].observation.back().first;
    		point1 = sfm_f[j].observation.back().second;
    		Vector3d point_3d;
    		triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
    		sfm_f[j].state = true;
    		sfm_f[j].position[0] = point_3d(0);
    		sfm_f[j].position[1] = point_3d(1);
    		sfm_f[j].position[2] = point_3d(2);
    		//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
    	}
    }

    // 8. 全局BA优化所有帧的位姿和特征点的3D位置
    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();

    for (int i = 0; i < frame_num; i++)
    {
        c_translation[i][0] = c_Translation[i].x();
        c_translation[i][1] = c_Translation[i].y();
        c_translation[i][2] = c_Translation[i].z();
        c_rotation[i][0] = c_Quat[i].w();
        c_rotation[i][1] = c_Quat[i].x();
        c_rotation[i][2] = c_Quat[i].y();
        c_rotation[i][3] = c_Quat[i].z();

        problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
        problem.AddParameterBlock(c_translation[i], 3);
        if (i == l || i == frame_num - 1) problem.SetParameterBlockConstant(c_translation[i]);
        if (i == l) problem.SetParameterBlockConstant(c_rotation[i]);
    }

    for (int i = 0; i < feature_num; i++)
    {
        if (!sfm_f[i].state) continue;
    	for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
    	{
    		int l = sfm_f[i].observation[j].first;
    		ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());

    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l],
									sfm_f[i].position);
    	}
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (summary.termination_type != ceres::CONVERGENCE && summary.final_cost >= 5e-3)
        return false;

    // 9. 将优化后的位姿和特征点存储到输出变量中
    for (int i = 0; i < frame_num; i++)
    {
        q[i].w() = c_rotation[i][0];
        q[i].x() = c_rotation[i][1];
        q[i].y() = c_rotation[i][2];
        q[i].z() = c_rotation[i][3];
        q[i] = q[i].inverse();

        T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
    }

	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}

    return true;
}



