% =========================================================================
% generate_Low.m (80s 无人机 3D 协同定位数据一键单次生成脚本)
% 空间物理约束：20m x 40m x 8m (高度严格锁定在 2m ~ 7m 内，实现立体分层)
% 运行时间：80秒 (100Hz IMU 预测/变频更新，10Hz 固定频 UWB 同步观测)
% 数据保存路径：E:\SE3_MLKF\Data\Low\
% =========================================================================
function generate_Low(target_anc)

% 【修改点】：使用 nargin 兼容 F5 手动直接运行本脚本
if nargin < 1
    clc; clear; close all;
    Anchor_num = 35; % 手动直接运行本脚本时的默认基站数
else
    % 如果是被主自动化控制脚本调用，则使用传入的基站数 [2]
    clc; close all;
    Anchor_num = target_anc;
end

%% 1. 用户指定单次生成的无人机和基站规模（可在此直接修改）
Vehicle_num = 6; % 指定单次生成的车辆/无人机规模 (范围: 4 - 12)
% 参数合法性检验
if Vehicle_num < 4 || Vehicle_num > 12
    error('Vehicle_num 必须在 4 到 12 之间！');
end
if Anchor_num < 4 || Anchor_num > 35
    error('Anchor_num 必须在 4 到 35 之间！');
end

F_imu = 100; % IMU 采样频率 (100Hz)
F_uwb = 10; % UWB 采样频率 (10Hz)
dt_imu = 1 / F_imu;
dt_uwb = 1 / F_uwb;
uwb_downsample_factor = round(dt_uwb / dt_imu);
if mod(dt_uwb, dt_imu) ~= 0
    error('UWB 采样周期必须是 IMU 采样周期的整数倍！');
end

t_end = 60; % 严格限定运行总时间为 60s
N_steps = round(t_end / dt_imu) + 1; % 共计 6001 步

% 物理保存根目录
save_dir = 'E:\SE3_MLKF\Data\Low';
trajectories_mat_name = sprintf('Trj_data_Veh%d_Anc%d_3D.mat', Vehicle_num, Anchor_num);
save_path = fullfile(save_dir, trajectories_mat_name);

% 传感器高精噪声参数
IMU_noise_params.sigma_na = 0.03;
IMU_noise_params.sigma_nw = 0.003;
IMU_noise_params.sigma_ba = 0.001;
IMU_noise_params.sigma_bw = 0.0001;
UWB_noise_params.sigma_anc = 0.3;
UWB_noise_params.sigma_rel = 0.3;

%% 2. 20点最优不共面三维立体基站拓扑池
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

%% 3. 12架无人机 40秒 空间均匀散乱分布的 3D 轨迹配置池
% 格式：[X0, Y0, Z0, th0, turn_dir, t_turn_start]
% 优化设计：对转弯时间进行了合理缩减与重分布 (区间 10s ~ 30s)，使其契合 40s 总仿真时长
all_veh_configs = [
    3, 6, 3.0, 0, 1, 12; % UAV1:  左下角区域 -> 朝东飞，12s后转弯向北
    17, 34, 3.9, pi, 1, 18; % UAV2:  右上角区域 -> 朝西飞，18s后转弯向南
    10, 15, 4.8, pi / 2, -1, 15; % UAV3:  中下部区域 -> 朝北飞，15s后转弯向西
    10, 28, 5.7, -pi / 2, -1, 20; % UAV4:  中上部区域 -> 朝南飞，20s后转弯向东
    3, 20, 3.3, 0, -1, 22; % UAV5:  左中部区域 -> 朝东飞，22s后转弯向南
    17, 20, 4.2, pi, -1, 10; % UAV6:  右中部区域 -> 朝西飞，10s后转弯向北
    3, 34, 3.6, 0, -1, 25; % UAV7:  左上角区域 -> 朝东飞，25s后转弯向南
    17, 6, 5.1, pi, 1, 28; % UAV8:  右下角区域 -> 朝西飞，28s后转弯向北
    7, 10, 5.4, pi / 2, 1, 14; % UAV9:  内左下区域 -> 朝北飞，14s后转弯向西
    13, 30, 4.5, -pi / 2, 1, 30; % UAV10: 内右上区域 -> 朝南飞，30s后转弯向东
    7, 30, 6.0, 0, -1, 11; % UAV11: 内左上区域 -> 朝东飞，11s后转弯向南
    13, 10, 4.2, pi, 1, 24; % UAV12: 内右下区域 -> 朝西飞，24s后转弯向北
    ];
%% 4. 单次数据生产内核
anchors = all_anchors_pool(1 : Anchor_num, :);

fprintf('>>> [正在生成数据集] UAV规模: %2d 架 | 基站数量: %2d 个...\n', Vehicle_num, Anchor_num);

% --- Step 4.1: 解析级 3D 轨迹真值生成 ---
trajectories = struct();
v_mag = 0.1; % 设定无人机巡航水平速度

for n = 1 : Vehicle_num
    cfg = all_veh_configs(n, :);
    X0 = cfg(1); Y0 = cfg(2); Z0 = cfg(3);
    th0 = cfg(4); turn_dir = cfg(5); t_turn_start = cfg(6);
    t_turn_end = t_turn_start + 3; % 3秒完成转弯

    % 计算目标航向角
    th_t = th0 + turn_dir * pi / 2;
    if th_t > pi, th_t = th_t - 2 * pi; end
    if th_t < -pi, th_t = th_t + 2 * pi; end

    P_true = zeros(N_steps, 3);
    V_true = zeros(N_steps, 3);
    A_true = zeros(N_steps, 3);
    Theta_true = zeros(N_steps, 1);

    % 解析物理递推
    for k = 1 : N_steps
        t = (k - 1) * dt_imu;

        % 1. 计算三维立体无人机高度 Z 轴柔和浮动真值
        % 满足 Z(t) = Z0 + 0.5 * sin(0.05 * t)
        z_k = Z0 + 0.5 * sin(0.05 * t);
        vz_k = 0.025 * cos(0.05 * t);
        az_k = -0.00125 * sin(0.05 * t);

        % 2. 区分水平分段运动
        if t < t_turn_start
            % A阶段：匀速直线
            th_k = th0;
            vx_k = v_mag * cos(th0);
            vy_k = v_mag * sin(th0);
            ax_k = 0; ay_k = 0;

            x_k = X0 + vx_k * t;
            y_k = Y0 + vy_k * t;

        elseif t >= t_turn_start && t <= t_turn_end
            % B阶段：原地高动态悬停转弯
            dt = t - t_turn_start;
            th_k = th0 + turn_dir * (pi / 2) / 3 * dt;
            vx_k = 0; vy_k = 0; % 原地转弯水平速度归零
            ax_k = 0; ay_k = 0;

            % 锁定转弯起始点
            x_k = X0 + (v_mag * cos(th0)) * t_turn_start;
            y_k = Y0 + (v_mag * sin(th0)) * t_turn_start;

        else
            % C阶段：转向后继续匀速直线
            dt_post = t - t_turn_end;
            th_k = th_t;
            vx_k = v_mag * cos(th_t);
            vy_k = v_mag * sin(th_t);
            ax_k = 0; ay_k = 0;

            % 累计转弯终点和后续直行距离
            x_turn_end = X0 + (v_mag * cos(th0)) * t_turn_start;
            y_turn_end = Y0 + (v_mag * sin(th0)) * t_turn_start;

            x_k = x_turn_end + vx_k * dt_post;
            y_k = y_turn_end + vy_k * dt_post;
        end

        % 3. 存储全维解析真值
        P_true(k, :) = [x_k, y_k, z_k];
        V_true(k, :) = [vx_k, vy_k, vz_k];
        A_true(k, :) = [ax_k, ay_k, az_k];
        Theta_true(k) = th_k;
    end

    % 4. 组装并保存真值
    v_name = sprintf('V%d', n);
    trajectories.(v_name).Time_true = (0 : dt_imu : t_end)';
    trajectories.(v_name).X_true = P_true(:, 1);
    trajectories.(v_name).Y_true = P_true(:, 2);
    trajectories.(v_name).Z_true = P_true(:, 3);
    trajectories.(v_name).Vx_true = V_true(:, 1);
    trajectories.(v_name).Vy_true = V_true(:, 2);
    trajectories.(v_name).Vz_true = V_true(:, 3);
    trajectories.(v_name).Theta_true = Theta_true;
    trajectories.(v_name).A_true = A_true;

    % 5. 组装解析姿态 R_true (3x3xN_steps)
    R_true = zeros(3, 3, N_steps);
    for k = 1 : N_steps
        th = Theta_true(k);
        R_true(:, :, k) = [cos(th), -sin(th), 0;
            sin(th), cos(th), 0;
            0, 0, 1];
    end
    trajectories.(v_name).R_true = R_true;
end

% --- Step 4.2: 生产变频 IMU 测量信息 ---
for n = 1 : Vehicle_num
    v_name = sprintf('V%d', n);
    veh = trajectories.(v_name);

    theta_unwrapped = unwrap(veh.Theta_true);
    wz_true = gradient(theta_unwrapped, dt_imu);
    omega_body_ideal = [zeros(N_steps, 2), wz_true];

    a_body_ideal = zeros(N_steps, 3);
    g_vec = [0; 0; -9.81];
    for k = 1 : N_steps
        R_k = veh.R_true(:, :, k);
        a_world = veh.A_true(k, :)';
        a_body_ideal(k, :) = (R_k' * (a_world - g_vec))';
    end

    % 高仿真零偏游走产生
    ba_init = (rand(1, 3) - 0.5) * 0.1;
    bw_init = (rand(1, 3) - 0.5) * 0.01;
    b_a = ba_init + cumsum(randn(N_steps, 3) * IMU_noise_params.sigma_ba * sqrt(dt_imu), 1);
    b_w = bw_init + cumsum(randn(N_steps, 3) * IMU_noise_params.sigma_bw * sqrt(dt_imu), 1);

    % 白噪声混合
    acc_m = a_body_ideal + b_a + randn(N_steps, 3) * IMU_noise_params.sigma_na;
    gyro_m = omega_body_ideal + b_w + randn(N_steps, 3) * IMU_noise_params.sigma_nw;

    % 整合输出
    trajectories.(v_name).IMU_Time = veh.Time_true;
    trajectories.(v_name).IMU_acc_m = acc_m;
    trajectories.(v_name).IMU_gyro_m = gyro_m;
    trajectories.(v_name).IMU_bias_a_true = b_a;
    trajectories.(v_name).IMU_bias_w_true = b_w;
end

% --- Step 4.3: 生产 10Hz 同步 UWB 相对测距信号 ---
idx_uwb = 1 : uwb_downsample_factor : N_steps;
t_uwb = trajectories.V1.Time_true(idx_uwb);
N_uwb = length(t_uwb);

pos_true_uwb = zeros(N_uwb, 3, Vehicle_num);
for n = 1 : Vehicle_num
    v_name = sprintf('V%d', n);
    pos_true_uwb(:, 1, n) = trajectories.(v_name).X_true(idx_uwb);
    pos_true_uwb(:, 2, n) = trajectories.(v_name).Y_true(idx_uwb);
    pos_true_uwb(:, 3, n) = trajectories.(v_name).Z_true(idx_uwb);
end

for n = 1 : Vehicle_num
    v_name = sprintf('V%d', n);

    % A. 绝对基站测距
    UWB_Anchor = zeros(N_uwb, 1 + Anchor_num);
    UWB_Anchor(:, 1) = t_uwb;
    for a_idx_sub = 1 : Anchor_num
        dx = pos_true_uwb(:, 1, n) - anchors(a_idx_sub, 1);
        dy = pos_true_uwb(:, 2, n) - anchors(a_idx_sub, 2);
        dz = pos_true_uwb(:, 3, n) - anchors(a_idx_sub, 3);
        dist_true = sqrt(dx .^ 2 + dy .^ 2 + dz .^ 2);
        UWB_Anchor(:, 1 + a_idx_sub) = dist_true + randn(N_uwb, 1) * UWB_noise_params.sigma_anc;
    end
    trajectories.(v_name).UWB_Anchor = UWB_Anchor;

    % B. 去中心化车间相对测距
    UWB_Relative = zeros(N_uwb, 1 + Vehicle_num);
    UWB_Relative(:, 1) = t_uwb;
    for j = 1 : Vehicle_num
        if n == j
            UWB_Relative(:, 1 + j) = NaN;
        else
            dx = pos_true_uwb(:, 1, n) - pos_true_uwb(:, 1, j);
            dy = pos_true_uwb(:, 2, n) - pos_true_uwb(:, 2, j);
            dz = pos_true_uwb(:, 3, n) - pos_true_uwb(:, 3, j);
            dist_true = sqrt(dx .^ 2 + dy .^ 2 + dz .^ 2);
            UWB_Relative(:, 1 + j) = dist_true + randn(N_uwb, 1) * UWB_noise_params.sigma_rel;
        end
    end
    trajectories.(v_name).UWB_Relative = UWB_Relative;
end

% --- Step 4.4: 保存当前配置的数据集 ---
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

save(save_path, 'trajectories', 'anchors', 'IMU_noise_params', 'UWB_noise_params', 'Vehicle_num', 'Anchor_num');
fprintf('  >> [生成成功] 无人机数据集已导出: %s\n', save_path);

%% 5. 轨迹可视化 (渲染当前单次仿真立体轨迹)
fprintf('\n正在进行三维立体轨迹可视化确认...\n');
figure('Name', 'Multi-UAV立体动态轨迹及立体不共面基站布设', 'Position', [150, 150, 850, 650]);
hold on; grid on; axis equal;
xlabel('X 轴 (m)'); ylabel('Y 轴 (m)'); zlabel('高度 Z (m)');
title(sprintf('%d架立体变高度无人机轨迹与最优非共面基座 (%d基站)', Vehicle_num, Anchor_num));

% 绘制 20m x 40m x 8m 立体边界框
line([0, 20, 20, 0, 0], [0, 0, 40, 40, 0], [0, 0, 0, 0, 0], 'Color', [0.6, 0.6, 0.6], 'LineStyle', '--');
line([0, 20, 20, 0, 0], [0, 0, 40, 40, 0], [8, 8, 8, 8, 8], 'Color', [0.6, 0.6, 0.6], 'LineStyle', '--');
for corner = [0, 20]
    for side = [0, 40]
        line([corner, corner], [side, side], [0, 8], 'Color', [0.6, 0.6, 0.6], 'LineStyle', '--');
    end
end

% 绘制立体基站
h_anchor = plot3(anchors(:, 1), anchors(:, 2), anchors(:, 3), '^', 'MarkerSize', 12, ...
    'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', '不共面立体基站 (0m~7.5m)');

colors = lines(Vehicle_num);
h_traj = zeros(1, Vehicle_num);
for n = 1 : Vehicle_num
    v_data = trajectories.(sprintf('V%d', n));
    h_traj(n) = plot3(v_data.X_true, v_data.Y_true, v_data.Z_true, 'Color', colors(n, :), 'LineWidth', 2.5, ...
        'DisplayName', sprintf('无人机 V%d (60s立体轨迹)', n));
    % 标出起点和终点
    plot3(v_data.X_true(1), v_data.Y_true(1), v_data.Z_true(1), 'o', 'MarkerSize', 8, 'MarkerFaceColor', colors(n, :), 'Color', colors(n, :));
    plot3(v_data.X_true(end), v_data.Y_true(end), v_data.Z_true(end), '*', 'MarkerSize', 10, 'Color', colors(n, :));
end

view(3);
legend([h_anchor, h_traj], 'Location', 'northeastoutside');
hold off;
end
