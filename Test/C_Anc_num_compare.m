% =========================================================================
% C_Anc_num_compare.m (基站数量变动性能对比脚本)
% 3种集中式算法在不同基站数下的对比
% 评测维度：平均欧氏定位误差 (m) 以及相对于 EKF 的提升百分比
% =========================================================================

clc; clear; close all;

%% 1. 路径与评测参数配置
addpath(genpath('../Common'));
addpath(genpath('../Filter'));
addpath(genpath('../Data'));

Anc_list = 4:20;                % 评测的基站数量范围
N_anc_tests = length(Anc_list);

% 结果保存路径与文件名配置
save_dir = 'E:\SE3_MLKF\Result';
save_name = 'C_Anc_num_4V_compare.mat';
save_path = fullfile(save_dir, save_name);

% 智能检测：如果历史评测数据已存在，直接跳过计算进入绘图
if exist(save_path, 'file')
    fprintf('检测到历史评测数据 [%s]，正在直接加载并生成图表...\n', save_name);
    load(save_path); % 加载获得 Anc_list, rmse_all_ekf, rmse_all_iekf, rmse_all_cmlkf
    jump_to_plot = true;
else
    jump_to_plot = false;
    % 记录各个基站数量下，全车平均欧氏定位误差的数组
    rmse_all_ekf   = zeros(N_anc_tests, 1);
    rmse_all_iekf  = zeros(N_anc_tests, 1);
    rmse_all_cmlkf = zeros(N_anc_tests, 1);
    fprintf('未检测到历史数据，开始执行多数据集联合性能评测...\n');
end

%% 2. 核心评测循环
if ~jump_to_plot
    for idx_anc = 1:N_anc_tests
        anc_num = Anc_list(idx_anc);
        fprintf('\n>>> 当前评测数据集基站数量: %d <<<\n', anc_num);
        
        % A. 自动检测并加载数据集
        data_file = sprintf('E:\\SE3_MLKF\\Data\\diff_Anc_4V\\Trj_data_Veh4_Anc%d_3D.mat', anc_num);
        if ~exist(data_file, 'file')
            data_file = sprintf('../Data/Trj_data_Veh4_Anc%d_3D.mat', anc_num); 
            if ~exist(data_file, 'file')
                error('未检测到指定数据集，请确认数据文件是否存在于：%s', data_file);
            end
        end
        load(data_file); % 加载 trajectories, anchors, IMU_noise_params, UWB_noise_params, Vehicle_num, Anchor_num
        
        dt_imu = 0.01; % 100Hz
        
        % B. 真值重建
        for n = 1:Vehicle_num
            v_name = sprintf('V%d', n);
            veh = trajectories.(v_name);
            N_steps = length(veh.Time_true);
            
            v_true_matrix = [veh.Vx_true, veh.Vy_true, veh.Vz_true];
            a_true_matrix = zeros(N_steps, 3);
            a_true_matrix(:, 1) = gradient(v_true_matrix(:, 1), dt_imu);
            a_true_matrix(:, 2) = gradient(v_true_matrix(:, 2), dt_imu);
            a_true_matrix(:, 3) = gradient(v_true_matrix(:, 3), dt_imu);
            trajectories.(v_name).a_true = a_true_matrix;
            
            theta_unwrapped = unwrap(veh.Theta_true);
            wz_true = gradient(theta_unwrapped, dt_imu);
            omega_true_matrix = [zeros(N_steps, 2), wz_true];
            trajectories.(v_name).omega_true = omega_true_matrix;
        end
        
        % C. 初始化三大滤波器
        % 15维 CMLKF 标称状态与协方差
        init_states_15d = struct('p', {}, 'v', {}, 'a', {}, 'R', {}, 'omega', {});
        for n = 1:Vehicle_num
            v_name = sprintf('V%d', n);
            veh = trajectories.(v_name);
            init_states_15d(n).p     = [veh.X_true(1); veh.Y_true(1); veh.Z_true(1)];
            init_states_15d(n).v     = [veh.Vx_true(1); veh.Vy_true(1); veh.Vz_true(1)];
            init_states_15d(n).a     = veh.a_true(1, :)';
            init_states_15d(n).R     = veh.R_true(:, :, 1);
            init_states_15d(n).omega = veh.omega_true(1, :)';
        end
        P_n_init_15d = diag([ ...
            (0.01)^2 * ones(1, 3), ...     
            (0.01)^2 * ones(1, 3), ...     
            (0.05)^2 * ones(1, 3), ...     
            (1*pi/180)^2 * ones(1, 3), ... 
            (0.005)^2 * ones(1, 3) ...     
        ]);
        init_P_15d = kron(eye(Vehicle_num), P_n_init_15d);
        
        Q_sigmas_15d.sig_wp      = 0.0001;  
        Q_sigmas_15d.sig_wv      = 0.001;  
        Q_sigmas_15d.sig_wa      = 0.00025; 
        Q_sigmas_15d.sig_wR      = 0.0001;  
        Q_sigmas_15d.sig_womega  = 0.00025;
    
        
        % 9维 EKF/IEKF 标称状态与协方差
        init_states_9d = struct('p', {}, 'v', {}, 'R', {});
        for n = 1:Vehicle_num
            v_name = sprintf('V%d', n);
            veh = trajectories.(v_name);
            init_states_9d(n).p = [veh.X_true(1); veh.Y_true(1); veh.Z_true(1)];
            init_states_9d(n).v = [veh.Vx_true(1); veh.Vy_true(1); veh.Vz_true(1)];
            init_states_9d(n).R = veh.R_true(:, :, 1);
        end
        P_n_init_9d = diag([ ...
            (0.01)^2 * ones(1, 3), ...     
            (0.01)^2 * ones(1, 3), ...     
            (1*pi/180)^2 * ones(1, 3) ...  
        ]);
        init_P_9d = kron(eye(Vehicle_num), P_n_init_9d);
        
        % 实例化滤波器类
        filter_cmlkf = CMLKF(init_states_15d, init_P_15d, Q_sigmas_15d);
        filter_ekf   = CEKF(init_states_9d, init_P_9d); 
        filter_iekf  = CIEKF(init_states_9d, init_P_9d); 
        
        % D. 运行记录初始化
        pos_est_cmlkf = cell(Vehicle_num, 1);
        pos_est_ekf   = cell(Vehicle_num, 1);
        pos_est_iekf  = cell(Vehicle_num, 1);
        for n = 1:Vehicle_num
            pos_est_cmlkf{n} = zeros(N_steps, 3);
            pos_est_ekf{n}   = zeros(N_steps, 3);
            pos_est_iekf{n}  = zeros(N_steps, 3);
            pos_est_cmlkf{n}(1, :) = init_states_15d(n).p';
            pos_est_ekf{n}(1, :)   = init_states_9d(n).p';
            pos_est_iekf{n}(1, :)  = init_states_9d(n).p';
        end
        
        % E. 算法并行滤波循环
        for k = 2:N_steps
            imu_acc = zeros(Vehicle_num, 3);
            imu_gyro = zeros(Vehicle_num, 3);
            for n = 1:Vehicle_num
                v_name = sprintf('V%d', n);
                veh = trajectories.(v_name);
                ba_true = veh.IMU_bias_a_true(k, :)';
                bw_true = veh.IMU_bias_w_true(k, :)';
                imu_acc(n, :)  = (veh.IMU_acc_m(k, :)'  - ba_true)';
                imu_gyro(n, :) = (veh.IMU_gyro_m(k, :)' - bw_true)';
            end
            
            filter_cmlkf.propagate(dt_imu);
            filter_ekf.propagate(imu_acc, imu_gyro, dt_imu);
            filter_iekf.propagate(imu_acc, imu_gyro, dt_imu);
            
            if mod(k - 1, 10) == 0
                uwb_idx = (k - 1) / 10 + 1;
                anc_meas = zeros(Vehicle_num, Anchor_num);
                rel_meas = zeros(Vehicle_num, Vehicle_num);
                for n = 1:Vehicle_num
                    v_name = sprintf('V%d', n);
                    veh = trajectories.(v_name);
                    anc_meas(n, :) = veh.UWB_Anchor(uwb_idx, 2:end);
                    rel_meas(n, :) = veh.UWB_Relative(uwb_idx, 2:end);
                end
                
                filter_cmlkf.update(imu_acc, imu_gyro, anchors, anc_meas, rel_meas, ...
                                    IMU_noise_params.sigma_na, IMU_noise_params.sigma_nw, ...
                                    UWB_noise_params.sigma_anc, UWB_noise_params.sigma_rel);
                filter_ekf.update(anchors, anc_meas, rel_meas, ...
                                  UWB_noise_params.sigma_anc, UWB_noise_params.sigma_rel);
                filter_iekf.update(anchors, anc_meas, rel_meas, ...
                                   UWB_noise_params.sigma_anc, UWB_noise_params.sigma_rel);
            else
                filter_cmlkf.update_imu_only(imu_acc, imu_gyro, ...
                                             IMU_noise_params.sigma_na, IMU_noise_params.sigma_nw);
            end
            
            for n = 1:Vehicle_num
                pos_est_cmlkf{n}(k, :) = filter_cmlkf.states(n).p';
                pos_est_ekf{n}(k, :)   = filter_ekf.states(n).p';
                pos_est_iekf{n}(k, :)  = filter_iekf.states(n).p';
            end
        end
        
        % F. 计算多车定位误差
        pos_true = cell(Vehicle_num, 1);
        for n = 1:Vehicle_num
            v_name = sprintf('V%d', n);
            pos_true{n} = [trajectories.(v_name).X_true, ...
                           trajectories.(v_name).Y_true, ...
                           trajectories.(v_name).Z_true];
        end
        
        [~, rmse_cmlkf] = calculate_position_errors(pos_est_cmlkf, pos_true);
        [~, rmse_ekf]     = calculate_position_errors(pos_est_ekf, pos_true);
        [~, rmse_iekf]    = calculate_position_errors(pos_est_iekf, pos_true);
        
        % G. 累加并计算全车平均欧氏误差
        sum_euc_ekf = 0;
        sum_euc_iekf = 0;
        sum_euc_cmlkf = 0;
        for n = 1:Vehicle_num
            sum_euc_ekf   = sum_euc_ekf   + rmse_ekf(n).euc_rmse;
            sum_euc_iekf  = sum_euc_iekf  + rmse_iekf(n).euc_rmse;
            sum_euc_cmlkf = sum_euc_cmlkf + rmse_cmlkf(n).euc_rmse;
        end
        
        rmse_all_ekf(idx_anc)   = sum_euc_ekf   / Vehicle_num;
        rmse_all_iekf(idx_anc)  = sum_euc_iekf  / Vehicle_num;
        rmse_all_cmlkf(idx_anc) = sum_euc_cmlkf / Vehicle_num;
        
        fprintf('基站数 %d 运行结束。EKF=%.4fm, IEKF=%.4fm, CMLKF=%.4fm\n', ...
            anc_num, rmse_all_ekf(idx_anc), rmse_all_iekf(idx_anc), rmse_all_cmlkf(idx_anc));
    end
end
%% 3. 数据处理与性能表格输出
fprintf('\n======================================= 基站数量变动性能评估与提升对比表 =======================================\n');
fprintf('%-10s | %-16s | %-32s | %-32s\n', '基站数量', 'EKF 欧氏误差 (m)', 'IEKF 欧氏误差及提升百分比', 'CMLKF 欧氏误差及提升百分比');
fprintf('--------------------------------------------------------------------------------------------------------------\n');

for idx_anc = 1:N_anc_tests
    anc_num = Anc_list(idx_anc);
    err_ekf   = rmse_all_ekf(idx_anc);
    err_iekf  = rmse_all_iekf(idx_anc);
    err_cmlkf = rmse_all_cmlkf(idx_anc);
    
    % 计算以 EKF 为基准的提升幅度
    pct_imp_iekf  = (err_ekf - err_iekf)  / err_ekf * 100;
    pct_imp_cmlkf = (err_ekf - err_cmlkf) / err_ekf * 100;
    
    str_iekf  = sprintf('%.4f (%+.2f%%)', err_iekf, pct_imp_iekf);
    str_cmlkf = sprintf('%.4f (%+.2f%%)', err_cmlkf, pct_imp_cmlkf);
    
    fprintf('%-10d | %-16.4f | %-32s | %-32s\n', anc_num, err_ekf, str_iekf, str_cmlkf);
end
fprintf('==============================================================================================================\n');

%% 4. 生成包含误差与提升度的双子图 (全英文图例)
figure('Name', 'Multi-Algorithm Performance Analysis VS Anchor Number', 'Position', [100, 100, 1200, 500]);

% 计算提升百分比序列
pct_imp_all_iekf  = (rmse_all_ekf - rmse_all_iekf)  ./ rmse_all_ekf * 100;
pct_imp_all_cmlkf = (rmse_all_ekf - rmse_all_cmlkf) ./ rmse_all_ekf * 100;

% --- 子图 1：平均欧氏误差对比折线图 ---
subplot(1, 2, 1);
grid on; hold on;
plot(Anc_list, rmse_all_ekf, 'b-o', 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', 'EKF (9D, IMU Input)');
plot(Anc_list, rmse_all_iekf, 'g-^', 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'IEKF (9D, Iterative)');
plot(Anc_list, rmse_all_cmlkf, 'r--s', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'CMLKF (15D, 100Hz ML)');

xlabel('Anchor Number', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Mean Multi-Agent Euclidean Error (m)', 'FontSize', 11, 'FontWeight', 'bold');
title('Position Error Comparison', 'FontSize', 12, 'FontWeight', 'bold');
xticks(Anc_list);
xlim([Anc_list(1) - 0.5, Anc_list(end) + 0.5]);
ylim([0, max(rmse_all_ekf) * 1.2]);
legend('Location', 'northeast', 'FontSize', 10);
hold off;

% --- 子图 2：相较于 EKF 的精度提升量曲线图 ---
subplot(1, 2, 2);
grid on; hold on;
plot(Anc_list, pct_imp_all_iekf, 'g-^', 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'IEKF Improvement');
plot(Anc_list, pct_imp_all_cmlkf, 'r--s', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'CMLKF Improvement');

xlabel('Anchor Number', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Accuracy Improvement over EKF (%)', 'FontSize', 11, 'FontWeight', 'bold');
title('Filtering Accuracy Improvement Degree', 'FontSize', 12, 'FontWeight', 'bold');
xticks(Anc_list);
xlim([Anc_list(1) - 0.5, Anc_list(end) + 0.5]);
% 动态控制 Y 轴范围，防止单点导致画面死板
min_imp = min([pct_imp_all_iekf; pct_imp_all_cmlkf]);
max_imp = max([pct_imp_all_iekf; pct_imp_all_cmlkf]);
ylim([min(0, min_imp - 5), max_imp + 10]); 
legend('Location', 'southeast', 'FontSize', 10);
hold off;

fprintf('双波形对比图（误差 & 提升度）绘制完成。\n');

%% 5. 评测数据自动化保存
if ~jump_to_plot
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    % 将画图所需的关键核心轴保存
    save(save_path, 'Anc_list', 'rmse_all_ekf', 'rmse_all_iekf', 'rmse_all_cmlkf');
    fprintf('本次评测数据已成功固化保存至：%s\n', save_path);
else
    fprintf('当前图表基于历史缓存数据渲染生成。\n');
end