	function [trajectories, anchors] = env_setup(Vehicle_num, Anchor_num, N_steps, dt_imu, t_end)
	    % env_setup.m - 环境配置与真值轨迹生成函数 (固定轨迹可复现版)
	    % 输入：
	    %   Vehicle_num : 车辆数量 (4-12)
	    %   Anchor_num  : 基站数量 (4-20)
	    %   N_steps     : 总采样步数
	    %   dt_imu      : IMU 采样步长
	    %   t_end       : 运行总时间
	    %% 1. 基站拓扑池定义 (4-35)
	    all_anchors_pool = [
         0,  0, 0;      % 1  (4基站基础：四角错层交叉，提供最大非共面基座)
        20,  0, 7.5;    % 2  
        20, 40, 0;      % 3  
         0, 40, 7.5;    % 4  
         0, 40, 0;      % 5  (5-8基站：补齐四个角的另一高度，形成8角立体包围盒)
        20,  0, 0;      % 6  
        20, 40, 7.5;    % 7  
         0,  0, 7.5;    % 8  
        10,  0, 0;      % 9  (9-12基站：补充短边中点，强化短边观测)
        10,  0, 7.5;    % 10 
        10, 40, 0;      % 11 
        10, 40, 7.5;    % 12 
         0, 20, 0;      % 13 (13-16基站：补充长边中点，强化长边观测)
         0, 20, 7.5;    % 14 
        20, 20, 0;      % 15 
        20, 20, 7.5;    % 16 
         0, 10, 0;      % 17 (17-20基站：补充长边1/4与3/4点，实现边缘高密度均匀覆盖)
        20, 10, 7.5;    % 18 
         0, 30, 7.5;    % 19 
        20, 30, 0;      % 20 
        % ----------------- 以下为新增的 21~35 号基站 -----------------
        20, 10, 0;      % 21 (21-24基站：补齐长边 1/4 与 3/4 的对侧高低交叉对称)
         0, 10, 7.5;    % 22 
        20, 30, 7.5;    % 23 
         0, 30, 0;      % 24 
         5,  0, 0;      % 25 (25-28基站：补充下侧短墙的 1/4 与 3/4 观测点)
        15,  0, 7.5;    % 26 
         5,  0, 7.5;    % 27 
        15,  0, 0;      % 28 
         5, 40, 7.5;    % 29 (29-32基站：补充上侧短墙的 1/4 与 3/4 观测点)
        15, 40, 0;      % 30 
         5, 40, 0;      % 31 
        15, 40, 7.5;    % 32 
         0, 20, 3.75;   % 33 (33-35基站：引入中层高度打破垂直高度方向简并，强化 3D 观测)
        20, 20, 3.75;   % 34 
        10,  0, 3.75;   % 35 
    ];
	    if Anchor_num > size(all_anchors_pool, 1) || Anchor_num < 4
	        error('输入的 Anchor_num 超出预设范围(4-35)，请检查！');
	    end
	    anchors = all_anchors_pool(1:Anchor_num, :);
	    %% 2. 12车固定轨迹参数池定义 (严格数学计算确保不越界)
	    % 参数格式: [X0, Y0, Z0, 初始航向th0, 转弯方向turn_dir(1左/-1右), 转弯开始时间t_start]
	    % 速度固定为 0.10 m/s。所有轨迹终点均经过严格计算限制在 X(0~20), Y(0~40) 内。
	    all_veh_configs = [
	        3,  3, 3.0,  0,      1, 100;     % 车1: 起点西南角，朝东，左转北 (终点: 13, 22.7)
	       17,  3, 3.3,  pi,    -1, 110;     % 车2: 起点东南角，朝西，右转北 (终点: 6, 21.7)
	        3, 37, 3.6,  0,     -1, 120;     % 车3: 起点西北角，朝东，右转南 (终点: 15, 19.3)
	       17, 37, 3.9,  pi,     1, 130;     % 车4: 起点东北角，朝西，左转南 (终点: 4, 20.3)
	       10,  3, 4.2,  pi/2,   1, 200;     % 车5: 起点南正中，朝北，左转西 (终点: 0.3, 23)
	       10, 37, 4.5, -pi/2,  -1, 210;     % 车6: 起点北正中，朝南，右转西 (终点: 1.3, 16)
	        3, 20, 4.8,  0,      1, 140;     % 车7: 起点西中，朝东，左转北 (终点: 17, 35.7)
	       17, 20, 5.1,  pi,    -1, 150;     % 车8: 起点东中，朝西，右转北 (终点: 2, 34.7)
	        7, 13, 5.4,  pi/2,  -1, 180;     % 车9: 起点内偏西南，朝北，右转东 (终点: 18.7, 31)
	       13, 13, 5.7,  0,      1,  70;     % 车10:起点内偏东南，朝东，左转北 (终点: 20, 35.7)
	        7, 27, 6.0,  0,     -1,  80;     % 车11:起点内偏西北，朝东，右转南 (终点: 15, 5.3)
	       13, 27, 4.5,  pi,     1,  90;     % 车12:起点内偏东北，朝西，左转南 (终点: 4, 6.3)
	    ];
	    max_veh = 12;
	    if Vehicle_num > max_veh || Vehicle_num < 4
	        error('输入的 Vehicle_num 超出预设范围(4-12)，请检查！');
	    end
	    %% 3. 固定轨迹物理积分生成
	    trajectories = struct();
	    v_mag = 0.10; % 巡航速度
	    for n = 1:Vehicle_num
	        cfg = all_veh_configs(n, :);
	        X0 = cfg(1); Y0 = cfg(2); Z0 = cfg(3);
	        th0 = cfg(4); turn_dir = cfg(5); t_turn_start = cfg(6);
	        t_turn_end = t_turn_start + 3; % 严格保证3秒完成转弯
	        % 计算目标航向角
	        th_t = th0 + turn_dir * pi/2;
	        if th_t > pi; th_t = th_t - 2*pi; end
	        if th_t < -pi; th_t = th_t + 2*pi; end
	        P_true = zeros(N_steps, 3);
	        V_true = zeros(N_steps, 3);
	        A_true = zeros(N_steps, 3);
	        Theta_true = zeros(N_steps, 1);
	        % 初始状态
	        P_true(1, :) = [X0, Y0, Z0];
	        V_true(1, :) = [v_mag * cos(th0), v_mag * sin(th0), 0];
	        Theta_true(1) = th0;
	        % 物理积分更新
	        for k = 2:N_steps
	            t = (k-1) * dt_imu;
	            a_curr = [0; 0; 0];
	            if t < t_turn_start
	                th_curr = th0;
	                V_true(k, :) = [v_mag * cos(th0), v_mag * sin(th0), 0];
	            elseif t >= t_turn_start && t <= t_turn_end
	                a_curr = [0; 0; 0];
	                V_true(k, :) = [0, 0, 0]; % 原地转弯，水平速度置零
	                th_curr = th0 + turn_dir * (pi/2)/3 * (t - t_turn_start);
	            else
	                th_curr = th_t;
	                V_true(k, :) = [v_mag * cos(th_t), v_mag * sin(th_t), 0]; % 恢复匀速
	            end
	            % 严格物理积分更新 (显式欧拉)
	            A_true(k-1, :) = a_curr';
	            P_true(k, :) = P_true(k-1, :) + V_true(k-1, :) * dt_imu + 0.5 * A_true(k-1, :) * dt_imu^2;
	            Theta_true(k) = th_curr;
	        end
	        A_true(end, :) = [0, 0, 0];
	        % 保存真值分量
	        v_name = sprintf('V%d', n);
	        trajectories.(v_name).Time_true = (0:dt_imu:t_end)';
	        trajectories.(v_name).X_true = P_true(:, 1);
	        trajectories.(v_name).Y_true = P_true(:, 2);
	        trajectories.(v_name).Z_true = P_true(:, 3);
	        trajectories.(v_name).Vx_true = V_true(:, 1);
	        trajectories.(v_name).Vy_true = V_true(:, 2);
	        trajectories.(v_name).Vz_true = V_true(:, 3);
	        trajectories.(v_name).Theta_true = Theta_true;
	        trajectories.(v_name).A_true = A_true;
	        % 旋转矩阵序列 R_true
	        R_true = zeros(3, 3, N_steps);
	        for k = 1:N_steps
	            th = Theta_true(k);
	            R_true(:, :, k) = [cos(th), -sin(th), 0; sin(th), cos(th), 0; 0, 0, 1];
	        end
	        trajectories.(v_name).R_true = R_true;
	    end
	end