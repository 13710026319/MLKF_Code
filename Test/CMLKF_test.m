% =========================================================================
% CMLKF 算法运行测试脚本 (基于新版3D多车动力学流形文档)
% 运行机制：离散动力学状态预测 (100Hz) + UWB与高频预修正IMU联合观测更新 (10Hz) [1, 3]
% =========================================================================

clc; clear; close all;

%% 1. 加载路径与仿真数据
addpath(genpath('../Common'));
addpath(genpath('../Filter'));
addpath(genpath('../Data'));

% 指定的仿真数据集路径
data_file = 'E:\SE3_MLKF\Data\Trj_data_Veh4_Anc11_3D.mat';
if ~exist(data_file, 'file')
    data_file = '../Data/Trj_data_Veh4_Anc4_2D.mat'; % 相对路径备用
    if ~exist(data_file, 'file')
        error('未检测到指定仿真数据，请先运行 Data 里面的数据生成脚本！');
    end
end
load(data_file); % 加载 trajectories, anchors, IMU_noise_params, UWB_noise_params, Vehicle_num, Anchor_num

dt_imu = 0.01; % 100Hz 仿真步长 \tau [1]

%% 2. 状态真值重建与偏置已知设定
% 新算法将加速度 $a_t^i$ 和角速度 $\omega_t^i$ 作为了标称状态量的组成部分 [1]
% 我们直接通过对速度和角度进行数值微分，在测试脚本中重建这两项真值，实现对既有 .mat 文件的重用
for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);
    veh = trajectories.(v_name);
    N_steps = length(veh.Time_true);
    
    % A. 重建 3D 真实加速度序列 a_t^i
    v_true_matrix = [veh.Vx_true, veh.Vy_true, veh.Vz_true];
    a_true_matrix = zeros(N_steps, 3);
    a_true_matrix(:, 1) = gradient(v_true_matrix(:, 1), dt_imu);
    a_true_matrix(:, 2) = gradient(v_true_matrix(:, 2), dt_imu);
    a_true_matrix(:, 3) = gradient(v_true_matrix(:, 3), dt_imu);
    trajectories.(v_name).a_true = a_true_matrix;
    
    % B. 重建 3D 真实角速度序列 \omega_t^i (仅 Z 轴有旋转分量)
    theta_unwrapped = unwrap(veh.Theta_true);
    wz_true = gradient(theta_unwrapped, dt_imu);
    omega_true_matrix = [zeros(N_steps, 2), wz_true];
    trajectories.(v_name).omega_true = omega_true_matrix;
end

%% 3. 初始化 CMLKF 滤波器状态与协方差矩阵
init_states = struct('p', {}, 'v', {}, 'a', {}, 'R', {}, 'omega', {});
for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);
    veh = trajectories.(v_name);
    
    % 填充 3D 动力学流形状态初值 (位置, 速度, 重建的加速度, 姿态矩阵, 重建的角速度) [1]
    init_states(n).p     = [veh.X_true(1); veh.Y_true(1); veh.Z_true(1)];
    init_states(n).v     = [veh.Vx_true(1); veh.Vy_true(1); veh.Vz_true(1)];
    init_states(n).a     = veh.a_true(1, :)';
    init_states(n).R     = veh.R_true(:, :, 1);
    init_states(n).omega = veh.omega_true(1, :)';
end

% 构造单车误差状态初始协方差 P_n (15x15) [2, 3]
P_n_init = diag([ ...
    (0.01)^2 * ones(1, 3), ...     % 位置不确定度
    (0.01)^2 * ones(1, 3), ...     % 速度不确定度
    (0.05)^2 * ones(1, 3), ...     % 加速度状态不确定度
    (1*pi/180)^2 * ones(1, 3), ... % 姿态不确定度 (1 deg)
    (0.005)^2 * ones(1, 3) ...     % 角速度状态不确定度
]);
init_P = kron(eye(Vehicle_num), P_n_init);

% 设定动力学系统过程噪声标准差 (对应 w_t^i, 文档 1.1 节) [1]
Q_sigmas.sig_wp     = 0;  % 位置过程噪声
Q_sigmas.sig_wv     = 0;  % 速度过程噪声
Q_sigmas.sig_wa     = 0.0005;   % 加速度随机游走
Q_sigmas.sig_wR     = 0.00005;  % 姿态随机游走
Q_sigmas.sig_womega = 0.00005;  % 角速度随机游走

% 实例化新版 CMLKF
filter = CMLKF(init_states, init_P, Q_sigmas);

%% 4. 执行滤波主循环
pos_est = cell(Vehicle_num, 1);
for n = 1:Vehicle_num
    pos_est{n} = zeros(N_steps, 3);
    pos_est{n}(1, :) = init_states(n).p';
end
fprintf('开始执行新版多车 CMLKF 协同估计...\n');
tic;
for k = 2:N_steps
    % ==========================================
    % 核心修改 1：将 IMU 数据提取移到 if 外部（每步必做）
    % ==========================================
    imu_acc = zeros(Vehicle_num, 3);
    imu_gyro = zeros(Vehicle_num, 3);
    for n = 1:Vehicle_num
        v_name = sprintf('V%d', n);
        veh = trajectories.(v_name);
        
        % 读取真实的 IMU 零偏并预校正
        ba_true = veh.IMU_bias_a_true(k, :)';
        bw_true = veh.IMU_bias_w_true(k, :)';
        imu_acc(n, :)  = (veh.IMU_acc_m(k, :)'  - ba_true)';
        imu_gyro(n, :) = (veh.IMU_gyro_m(k, :)' - bw_true)';
    end
    
    % A. 系统标称状态与误差前向前向积分 (100Hz 纯预测步) [3]
    filter.propagate(dt_imu);
    
    % B. 观测更新步
    if mod(k - 1, 10) == 0
        % --- UWB与高频预修正IMU 联合观测更新时刻 (10Hz) ---
        uwb_idx = (k - 1) / 10 + 1;
        anc_meas = zeros(Vehicle_num, Anchor_num);
        rel_meas = zeros(Vehicle_num, Vehicle_num);
        
        for n = 1:Vehicle_num
            v_name = sprintf('V%d', n);
            veh = trajectories.(v_name);
            % 提取当前周期 UWB 测距 [8]
            anc_meas(n, :) = veh.UWB_Anchor(uwb_idx, 2:end);
            rel_meas(n, :) = veh.UWB_Relative(uwb_idx, 2:end);
        end
        
        % 执行联合最大似然观测融更新 [4]
        filter.update(imu_acc, imu_gyro, anchors, anc_meas, rel_meas, ...
                      IMU_noise_params.sigma_na, IMU_noise_params.sigma_nw, ...
                      UWB_noise_params.sigma_anc, UWB_noise_params.sigma_rel);
    else
        % ==========================================
        % 核心修改 2：补上 else 分支，执行高频 IMU 更新 (其余 90Hz 步)
        % ==========================================
        filter.update_imu_only(imu_acc, imu_gyro, ...
                               IMU_noise_params.sigma_na, IMU_noise_params.sigma_nw);
    end
    
    % C. 收集估计值
    for n = 1:Vehicle_num
        pos_est{n}(k, :) = filter.states(n).p';
    end
end
toc;
fprintf('滤波计算完毕！\n');

%% 5. 结果处理与 RMSE 数据评估表输出
pos_true = cell(Vehicle_num, 1);
for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);
    pos_true{n} = [trajectories.(v_name).X_true, ...
                   trajectories.(v_name).Y_true, ...
                   trajectories.(v_name).Z_true];
end

[errors, rmse] = calculate_position_errors(pos_est, pos_true);

% 组装表格对象输出
RowNames = cell(Vehicle_num, 1);
X_RMSE = zeros(Vehicle_num, 1);
Y_RMSE = zeros(Vehicle_num, 1);
Z_RMSE = zeros(Vehicle_num, 1);
Euc_RMSE = zeros(Vehicle_num, 1);

Mean_Euc_RMSE = 0;

for n = 1:Vehicle_num
    RowNames{n} = sprintf('Vehicle_%d', n);
    X_RMSE(n)   = rmse(n).axis_rmse(1);
    Y_RMSE(n)   = rmse(n).axis_rmse(2);
    Z_RMSE(n)   = rmse(n).axis_rmse(3);
    Euc_RMSE(n) = rmse(n).euc_rmse;
    Mean_Euc_RMSE = Mean_Euc_RMSE + rmse(n).euc_rmse;
end

rmse_table = table(X_RMSE, Y_RMSE, Z_RMSE, Euc_RMSE, 'RowNames', RowNames);
fprintf('\n================== 新版 CMLKF 定位 RMSE 评估表 ==================\n');
disp(rmse_table);
fprintf('=================================================================\n');

fprintf('全车欧氏误差均值: CMLKF: %.4fm \n', Mean_Euc_RMSE/Vehicle_num);
fprintf('===================================================================================================\n');

%% 6. 绘图 (欧氏定位误差曲线)
time_arr = trajectories.V1.IMU_Time;
figure('Name', 'CMLKF Position Euclidean Error', 'Position', [150, 150, 1000, 800]);
for n = 1:Vehicle_num
    subplot(2, 2, n);
    hold on; grid on;
    plot(time_arr, errors(n).euc_err, 'b-', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('Vehicle %d Euc Error', n));
    title(sprintf('Vehicle %d Euclidean Error', n));
    xlabel('Time (s)'); ylabel('Error (m)');
    xlim([0, time_arr(end)]);
    ylim([0, 1.5]);
    legend('Location', 'northeast');
    hold off;
end