% =========================================================================
% generate_data_3D.m (CMLKF 几何最优化版)
% 3D多车(无人机行为) UWB/IMU 协同定位仿真数据生成脚本
% 空间范围：20m x 40m x 8m (高度严格控制在 2m ~ 7m)
% 优化重点：重构基站最大立体非对称拓扑，以及无人机群多高度层动态斜线协同轨迹
% =========================================================================
clc; clear; close all;

%% 1. 全局参数设置与保存路径
dt_imu = 0.01;              % IMU 采样时间 100Hz (0.01s)
dt_uwb = 0.1;               % UWB 采样时间 10Hz (0.1s)
t_end = 300;                % 运行时间 300秒
N_steps = round(t_end / dt_imu) + 1; % 30001 个采样点
Vehicle_num = 4;            % 车辆数量
Anchor_num = 20;             % 基站数量

% 基站高度在 0~8m 范围内实现非对称立体最大化错落

% 18个最优立体基站总表（严格按照你原代码的递增拓扑顺序排列）
all_anchors_pool = [
     0,  0, 0;      % 1  (4基站基础)
    20,  0, 7.5;    % 2
    20, 40, 0;      % 3
     0, 40, 7.5;    % 4
     0, 40, 0;      % 5  (5基站新增)
    20,  0, 7.5;    % 6  (6基站新增)
    20, 40, 7.5;    % 7  (7基站新增)
    20,  0, 0;      % 8  (8基站新增)
     0, 20, 0;      % 9  (9基站新增)
    20, 20, 0;      % 10 (10基站新增)
     0, 20, 7.5;    % 11 (11基站新增)
    20, 20, 7.5;    % 12 (12基站新增)
    20, 30, 7.5;    % 13 (13基站新增)
     0, 10, 0;      % 14 (14基站新增)
    20, 30, 0;      % 15 (15基站新增)
     0, 30, 7.5;    % 16 (16基站新增)
     0, 30, 0;      % 17 (17基站新增)
    20, 33, 0;      % 18 (18基站新增)
    10, 0, 7.5;     % 19 (19基站新增)
    10, 0, 0;       % 20 (20基站新增)
];

if Anchor_num > size(all_anchors_pool, 1) || Anchor_num < 4
    error('输入的 Anchor_num 超出预设范围，请检查！');
end
anchors = all_anchors_pool(1:Anchor_num, :);

% 保存路径
save_dir = 'E:\SE3_MLKF\Data'; 
trajectories_mat_name = sprintf('Trj_data_Veh%d_Anc%d_3D.mat', Vehicle_num, Anchor_num);
   
% 噪声参数
IMU_noise_params.sigma_na = 0.03;      
IMU_noise_params.sigma_nw = 0.003;     
IMU_noise_params.sigma_ba = 0.001;     
IMU_noise_params.sigma_bw = 0.0001;    
UWB_noise_params.sigma_anc = 0.3;     
UWB_noise_params.sigma_rel = 0.3;  

%% 2. 独立车辆动力学规划 (3秒平滑原地转弯 + 持续三维斜线飞行)
% 初始状态规划：[X, Y, Z, 初始速率, 初始航向角]
% 高度均限制在 2m ~ 7m 范围内，各自占据不同的起始高度层
init_configs = [
    3,   5,  2.5,  0.10,  0;      % V1: 低空层(2.5m)， 偏南，朝东，后续北转
   17,   3,  6.5,  0.10,  pi;     % V2: 高空层(6.5m)， 偏南，朝西，后续北转
    3,  39,  3.8,  0.10,  0;      % V3: 中低层(3.8m)， 偏北，朝东，后续南转
   18,  35,  5.2,  0.10,  pi;     % V4: 中高层(5.2m)， 偏北，朝西，后续南转
];

trajectories = struct();
for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);
    cfg = init_configs(n, :);
    
    P_true = zeros(N_steps, 3);
    V_true = zeros(N_steps, 3);
    A_true = zeros(N_steps, 3);
    Theta_true = zeros(N_steps, 1);
    
    % 初始条件
    P_true(1, :) = cfg(1:3);
    v_mag = cfg(4);
    th_curr = cfg(5);
    V_true(1, :) = [v_mag * cos(th_curr), v_mag * sin(th_curr), 0];
    Theta_true(1) = th_curr;
    
    % 计算转弯过渡段的时间窗
    t_turn_start = 120 + (n-1)*15; % 错开转弯时间：V1:120s, V2:135s, V3:150s, V4:165s
    t_turn_end = t_turn_start + 3;
    
    for k = 2:N_steps
        t = (k-1) * dt_imu;
        a_curr = [0; 0; 0]; % 默认 Z 轴加速度全程为 0
        th_curr = Theta_true(k-1);
        
        switch n
            case 1  % V1: 高度恒定 2.5m。东行巡航 -> 120s时原地左转 -> 向北巡航
                if t < t_turn_start
                    th_curr = 0; a_curr = [0; 0; 0];
                elseif t >= t_turn_start && t <= t_turn_end
                    a_curr = [0; 0; 0]; V_true(k-1, 1:2) = 0; % 原地转弯，水平速度置零
                    th_curr = 0 + (pi/2)/3 * (t - t_turn_start); % 3秒平滑左转90度(朝北)
                elseif t > t_turn_end && t <= t_turn_end + 10
                    th_curr = pi/2; a_curr = [0; 0.01; 0]; % 转弯后水平加速 10s 恢复航速
                else
                    th_curr = pi/2; a_curr = [0; 0; 0];    % 匀速北行
                end
                
            case 2  % V2: 高度恒定 6.5m。西行巡航 -> 135s时原地右转 -> 向北巡航
                if t < t_turn_start
                    th_curr = pi; a_curr = [0; 0; 0];
                elseif t >= t_turn_start && t <= t_turn_end
                    a_curr = [0; 0; 0]; V_true(k-1, 1:2) = 0;
                    th_curr = pi - (pi/2)/3 * (t - t_turn_start); % 3秒平滑右转90度(朝北)
                elseif t > t_turn_end && t <= t_turn_end + 10
                    th_curr = pi/2; a_curr = [0; 0.01; 0]; % 转弯后水平加速 10s
                else
                    th_curr = pi/2; a_curr = [0; 0; 0];
                end
                
            case 3  % V3: 高度恒定 3.8m。东行巡航 -> 150s时原地右转 -> 向南巡航
                if t < t_turn_start
                    th_curr = 0; a_curr = [0; 0; 0];
                elseif t >= t_turn_start && t <= t_turn_end
                    a_curr = [0; 0; 0]; V_true(k-1, 1:2) = 0;
                    th_curr = 0 - (pi/2)/3 * (t - t_turn_start); % 3秒平滑右转90度(朝南)
                elseif t > t_turn_end && t <= t_turn_end + 10
                    th_curr = -pi/2; a_curr = [0; -0.01; 0]; % 转弯后水平加速 10s
                else
                    th_curr = -pi/2; a_curr = [0; 0; 0];
                end
                
            case 4  % V4: 高度恒定 5.2m。西行巡航 -> 165s时原地左转 -> 向南巡航
                if t < t_turn_start
                    th_curr = pi; a_curr = [0; 0; 0];
                elseif t >= t_turn_start && t <= t_turn_end
                    a_curr = [0; 0; 0]; V_true(k-1, 1:2) = 0;
                    th_curr = pi + (pi/2)/3 * (t - t_turn_start); % 3秒平滑左转90度(朝南)
                elseif t > t_turn_end && t <= t_turn_end + 10
                    th_curr = 3*pi/2; a_curr = [0; -0.01; 0]; % 转弯后水平加速 10s
                else
                    th_curr = 3*pi/2; a_curr = [0; 0; 0];
                end
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
        a_world = [A_true(k, 1); A_true(k, 2); A_true(k, 3)];
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
idx_uwb = 1:10:N_steps;
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
xlabel('X 轴位置 (m)'); ylabel('Y 轴位置 (m)'); zlabel('高度 Z (m)');
title('4机动态立体分层轨迹与最优不共面基站布设');

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