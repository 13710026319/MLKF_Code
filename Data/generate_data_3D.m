	% generate_data_3D.m 
	% 3D多车 UWB/IMU 协同定位仿真数据生成脚本
	% 空间范围：20m x 40m x 8m (高度严格控制在 2m ~ 7m)
	clc; clear; close all;
	%% 1. 全局参数设置与保存路径
	F_imu = 100;                % 想要生成的 IMU 频率
	F_uwb = 10;                 % 想要生成的 UWB 频率
	dt_imu = 1 / F_imu;        
	dt_uwb = 1 / F_uwb;        
	uwb_downsample_factor = round(dt_uwb / dt_imu); 
	if mod(dt_uwb, dt_imu) ~= 0
	    error('警告：UWB 采样周期必须是 IMU 采样周期的整数倍！');
	end
	t_end = 300;                
	N_steps = round(t_end / dt_imu) + 1; 
	% ==================================================
	% 在此处自由修改车辆数(4-12)与基站数(4-20)
	Vehicle_num = 12;            
	Anchor_num = 6;             
	% ==================================================
	% 保存路径
	save_dir = 'E:\SE3_MLKF\Data\diff_V_6Anc'; 
	trajectories_mat_name = sprintf('Trj_data_Veh%d_Anc%d_3D_1.mat', Vehicle_num, Anchor_num);
	% 噪声参数
	IMU_noise_params.sigma_na = 0.03;      
	IMU_noise_params.sigma_nw = 0.003;     
	IMU_noise_params.sigma_ba = 0.001;     
	IMU_noise_params.sigma_bw = 0.0001;    
	UWB_noise_params.sigma_anc = 0.3;     
	UWB_noise_params.sigma_rel = 0.3;  
	%% 2. 环境与真值轨迹生成 (调用外部 env_setup.m)
	[trajectories, anchors] = env_setup(Vehicle_num, Anchor_num, N_steps, dt_imu, t_end);
	%% 3. 生成含有零偏与白噪声的 100Hz 3D IMU 测量信号
	for n = 1:Vehicle_num
	    v_name = sprintf('V%d', n);
	    veh = trajectories.(v_name);
	    % 计算理想 3D 角速度
	    theta_unwrapped = unwrap(veh.Theta_true);
	    wz_true = gradient(theta_unwrapped, dt_imu);
	    omega_body_ideal = [zeros(N_steps, 2), wz_true];
	    % 计算理想特定力：f = R^T * (a - g)
	    a_body_ideal = zeros(N_steps, 3);
	    g_vec = [0; 0; -9.81];
	    for k = 1:N_steps
	        R_k = veh.R_true(:, :, k);
	        a_world = [veh.A_true(k, 1); veh.A_true(k, 2); veh.A_true(k, 3)];
	        a_body_ideal(k, :) = (R_k' * (a_world - g_vec))';
	    end
	    % 模拟高保真零偏游走
	    ba_init = (rand(1, 3) - 0.5) * 0.1;    
	    bw_init = (rand(1, 3) - 0.5) * 0.01;
	    b_a = ba_init + cumsum(randn(N_steps, 3) * IMU_noise_params.sigma_ba * sqrt(dt_imu), 1);
	    b_w = bw_init + cumsum(randn(N_steps, 3) * IMU_noise_params.sigma_bw * sqrt(dt_imu), 1);
	    % 叠加噪声
	    acc_noise = randn(N_steps, 3) * IMU_noise_params.sigma_na;
	    gyro_noise = randn(N_steps, 3) * IMU_noise_params.sigma_nw;
	    acc_m = a_body_ideal + b_a + acc_noise;
	    gyro_m = omega_body_ideal + b_w + gyro_noise;
	    % 保存 IMU 数据
	    trajectories.(v_name).IMU_Time = veh.Time_true;
	    trajectories.(v_name).IMU_acc_m = acc_m;
	    trajectories.(v_name).IMU_gyro_m = gyro_m;
	    trajectories.(v_name).IMU_bias_a_true = b_a;
	    trajectories.(v_name).IMU_bias_w_true = b_w;
	end
	%% 4. 生成 10Hz 同步 UWB 测距信号
	idx_uwb = 1:uwb_downsample_factor:N_steps;
	t_uwb = trajectories.V1.Time_true(idx_uwb);
	N_uwb = length(t_uwb);
	pos_true_uwb = zeros(N_uwb, 3, Vehicle_num);
	for n = 1:Vehicle_num
	    v_name = sprintf('V%d', n);
	    pos_true_uwb(:, 1, n) = trajectories.(v_name).X_true(idx_uwb);
	    pos_true_uwb(:, 2, n) = trajectories.(v_name).Y_true(idx_uwb);
	    pos_true_uwb(:, 3, n) = trajectories.(v_name).Z_true(idx_uwb);
	end
	for n = 1:Vehicle_num
	    v_name = sprintf('V%d', n);
	    % A. UWB 基站 3D 测距
	    UWB_Anchor = zeros(N_uwb, 1 + Anchor_num);
	    UWB_Anchor(:, 1) = t_uwb;
	    for a_idx = 1:Anchor_num
	        dx = pos_true_uwb(:, 1, n) - anchors(a_idx, 1);
	        dy = pos_true_uwb(:, 2, n) - anchors(a_idx, 2);
	        dz = pos_true_uwb(:, 3, n) - anchors(a_idx, 3);
	        dist_true = sqrt(dx.^2 + dy.^2 + dz.^2);
	        UWB_Anchor(:, 1 + a_idx) = dist_true + randn(N_uwb, 1) * UWB_noise_params.sigma_anc;
	    end
	    trajectories.(v_name).UWB_Anchor = UWB_Anchor;
	    % B. UWB 3D 车间相对测距
	    UWB_Relative = zeros(N_uwb, 1 + Vehicle_num);
	    UWB_Relative(:, 1) = t_uwb;
	    for j = 1:Vehicle_num
	        if n == j
	            UWB_Relative(:, 1 + j) = NaN;
	        else
	            dx = pos_true_uwb(:, 1, n) - pos_true_uwb(:, 1, j);
	            dy = pos_true_uwb(:, 2, n) - pos_true_uwb(:, 2, j);
	            dz = pos_true_uwb(:, 3, n) - pos_true_uwb(:, 3, j);
	            dist_true = sqrt(dx.^2 + dy.^2 + dz.^2);
	            UWB_Relative(:, 1 + j) = dist_true + randn(N_uwb, 1) * UWB_noise_params.sigma_rel;
	        end
	    end
	    trajectories.(v_name).UWB_Relative = UWB_Relative;
	end
	%% 5. 轨迹可视化 (三维立体作图以确认高度不共面)
	figure('Name', 'Multi-Agent 最优 3D 协同定位拓扑轨迹', 'Position', [100, 100, 850, 650]);
	hold on; grid on; axis equal;
	xlabel('X 轴位置'); ylabel('Y 轴位置'); zlabel('高度 Z (m)');
	title(sprintf('%d机动态立体分层轨迹与最优不共面基站布设', Vehicle_num));
	% 绘制边界立体框
	line([0, 20, 20, 0, 0], [0, 0, 40, 40, 0], [0, 0, 0, 0, 0], 'Color', [0.5,0.5,0.5], 'LineStyle', '--');
	line([0, 20, 20, 0, 0], [0, 0, 40, 40, 0], [8, 8, 8, 8, 8], 'Color', [0.5,0.5,0.5], 'LineStyle', '--');
	for corner = [0, 20]
	    for side = [0, 40]
	        line([corner, corner], [side, side], [0, 8], 'Color', [0.5,0.5,0.5], 'LineStyle', '--');
	    end
	end
	% 绘制最优不共面基站
	h_anchor = plot3(anchors(:,1), anchors(:,2), anchors(:,3), '^', 'MarkerSize', 13, ...
	    'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', '最优立体基站 (0m~8m)');
	colors = lines(Vehicle_num);
	h_traj = zeros(1, Vehicle_num);
	for n = 1:Vehicle_num
	    v_data = trajectories.(sprintf('V%d', n));
	    h_traj(n) = plot3(v_data.X_true, v_data.Y_true, v_data.Z_true, 'Color', colors(n,:), 'LineWidth', 2.5, ...
	        'DisplayName', sprintf('无人机 V%d (3D动态斜线)', n));
	    % 标记起始点和终点
	    plot3(v_data.X_true(1), v_data.Y_true(1), v_data.Z_true(1), 'o', 'MarkerSize', 8, 'MarkerFaceColor', colors(n,:), 'Color', colors(n,:));
	    plot3(v_data.X_true(end), v_data.Y_true(end), v_data.Z_true(end), '*', 'MarkerSize', 10, 'Color', colors(n,:));
	end
	view(3); 
	legend([h_anchor, h_traj], 'Location', 'northeastoutside');
	hold off;
	%% 6. 数据保存
	if ~exist(save_dir, 'dir')
	    mkdir(save_dir);
	end
	save_path = fullfile(save_dir, trajectories_mat_name);
	save(save_path, 'trajectories', 'anchors', 'IMU_noise_params', 'UWB_noise_params', 'Vehicle_num', 'Anchor_num');
	fprintf('CMLKF 优化版 3D 仿真数据生成成功，已保存至：%s\n', save_path);