	function [trajectories, anchors] = env_setup(Vehicle_num, Anchor_num, N_steps, dt_imu, t_end)
	    % env_setup.m - 环境配置与真值轨迹生成函数
	    % 输入：
	    %   Vehicle_num : 车辆数量 (4-10)
	    %   Anchor_num  : 基站数量 (4-20)
	    %   N_steps     : 总采样步数
	    %   dt_imu      : IMU 采样步长
	    %   t_end       : 运行总时间
	    %% 1. 基站拓扑池定义 (4-20)
	    all_anchors_pool = [
	         0,  0, 0;      % 1  
	        20,  0, 7.5;    % 2  
	        20, 40, 0;      % 3  
	         0, 40, 7.5;    % 4  
	         0, 40, 0;      % 5  
	        20,  0, 7.5;    % 6  
	        20, 40, 7.5;    % 7  
	        20,  0, 0;      % 8  
	         0, 20, 0;      % 9  
	        20, 20, 0;      % 10 
	         0, 20, 7.5;    % 11 
	        20, 20, 7.5;    % 12 
	        20, 30, 7.5;    % 13 
	         0, 10, 0;      % 14 
	        20, 30, 0;      % 15 
	         0, 30, 7.5;    % 16 
	         0, 30, 0;      % 17 
	        20, 33, 0;      % 18 
	        10, 0, 7.5;     % 19 
	        10, 0, 0;       % 20 
	    ];
	    if Anchor_num > size(all_anchors_pool, 1) || Anchor_num < 4
	        error('输入的 Anchor_num 超出预设范围(4-20)，请检查！');
	    end
	    anchors = all_anchors_pool(1:Anchor_num, :);
	    %% 2. 车辆初始配置池定义 (扩展至10辆)
	    % 初始状态规划：[X, Y, Z, 初始速率, 初始航向角]
	    all_veh_configs = [
	        3,   5,  2.5,  0.10,  0;      % V1
	       17,   3,  6.5,  0.10,  pi;     % V2
	        3,  39,  3.8,  0.10,  0;      % V3
	       18,  35,  5.2,  0.10,  pi;     % V4
	        5,  10,  4.0,  0.10,  0;      % V5: 扩展
	       15,  10,  5.5,  0.10,  pi;     % V6: 扩展
	        5,  30,  4.5,  0.10,  0;      % V7: 扩展
	       15,  30,  6.0,  0.10,  pi;     % V8: 扩展
	        8,  20,  3.0,  0.10,  0;      % V9: 扩展
	       12,  20,  6.8,  0.10,  pi;     % V10: 扩展
	    ];
	    % 转弯参数：[转弯方向(1左/-1右), 目标航向角]
	    veh_turn_params = [
	        1,  pi/2;     % V1: 左转 -> 朝北
	       -1,  pi/2;     % V2: 右转 -> 朝北
	       -1, -pi/2;     % V3: 右转 -> 朝南
	        1,  3*pi/2;   % V4: 左转 -> 朝南
	        1,  pi/2;     % V5: 左转 -> 朝北
	       -1,  pi/2;     % V6: 右转 -> 朝北
	       -1, -pi/2;     % V7: 右转 -> 朝南
	        1,  3*pi/2;   % V8: 左转 -> 朝南
	        1,  pi/2;     % V9: 左转 -> 朝北
	       -1,  pi/2;     % V10: 右转 -> 朝北
	    ];
	    if Vehicle_num > size(all_veh_configs, 1) || Vehicle_num < 4
	        error('输入的 Vehicle_num 超出预设范围(4-10)，请检查！');
	    end
	    init_configs = all_veh_configs(1:Vehicle_num, :);
	    %% 3. 通用动力学规划 (消除冗长的 switch-case)
	    trajectories = struct();
	    for n = 1:Vehicle_num
	        v_name = sprintf('V%d', n);
	        cfg = init_configs(n, :);
	        turn_cfg = veh_turn_params(n, :);
	        P_true = zeros(N_steps, 3);
	        V_true = zeros(N_steps, 3);
	        A_true = zeros(N_steps, 3);
	        Theta_true = zeros(N_steps, 1);
	        % 初始条件
	        P_true(1, :) = cfg(1:3);
	        v_mag = cfg(4);
	        start_th = cfg(5);
	        V_true(1, :) = [v_mag * cos(start_th), v_mag * sin(start_th), 0];
	        Theta_true(1) = start_th;
	        % 转弯过渡段的时间窗
	        t_turn_start = 120 + (n-1)*15; 
	        t_turn_end = t_turn_start + 3;
	        turn_dir = turn_cfg(1);
	        target_th = turn_cfg(2);
	        for k = 2:N_steps
	            t = (k-1) * dt_imu;
	            a_curr = [0; 0; 0]; 
	            th_curr = Theta_true(k-1);
	            if t < t_turn_start
	                th_curr = start_th; 
	                a_curr = [0; 0; 0];
	            elseif t >= t_turn_start && t <= t_turn_end
	                a_curr = [0; 0; 0]; 
	                V_true(k-1, 1:2) = 0; % 原地转弯，水平速度置零
	                th_curr = start_th + turn_dir * (pi/2)/3 * (t - t_turn_start);
	            elseif t > t_turn_end && t <= t_turn_end + 10
	                th_curr = target_th; 
	                % 依据目标航向角自动判定加速方向
	                a_curr = [0; 0.01 * sin(target_th); 0]; 
	            else
	                th_curr = target_th; 
	                a_curr = [0; 0; 0];
	            end
	            % 严格物理积分更新
	            A_true(k-1, :) = a_curr';
	            V_true(k, :)   = V_true(k-1, :) + A_true(k-1, :) * dt_imu;
	            P_true(k, :)   = P_true(k-1, :) + V_true(k-1, :) * dt_imu + 0.5 * A_true(k-1, :) * dt_imu^2;
	            Theta_true(k)  = th_curr;
	        end
	        A_true(end, :) = [0, 0, 0];
	        % 保存真值分量
	        trajectories.(v_name).Time_true = (0:dt_imu:t_end)';
	        trajectories.(v_name).X_true = P_true(:, 1);
	        trajectories.(v_name).Y_true = P_true(:, 2);
	        trajectories.(v_name).Z_true = P_true(:, 3);
	        trajectories.(v_name).Vx_true = V_true(:, 1);
	        trajectories.(v_name).Vy_true = V_true(:, 2);
	        trajectories.(v_name).Vz_true = V_true(:, 3);
	        trajectories.(v_name).Theta_true = Theta_true;
	        trajectories.(v_name).A_true = A_true; % 保存加速度供主脚本使用
	        % 旋转矩阵序列 R_true
	        R_true = zeros(3, 3, N_steps);
	        for k = 1:N_steps
	            th = Theta_true(k);
	            R_true(:, :, k) = [
	                cos(th), -sin(th), 0;
	                sin(th),  cos(th), 0;
	                0,        0,       1
	            ];
	        end
	        trajectories.(v_name).R_true = R_true;
	    end
	end