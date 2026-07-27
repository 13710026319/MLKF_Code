% =========================================================================
% UKF_Test.m (流形无迹卡尔曼滤波性能测试脚本 - 含数据集缓存免重复运行功能)
% =========================================================================
clc; clear; close all;

%% 1. 结果缓存检测与加载机制 [2]
save_dir = 'E:\SE3_MLKF\Result\Else';
save_file = fullfile(save_dir, 'DUKF_V_num_6Anc.mat');

if exist(save_file, 'file')
    fprintf('=========================================================================\n');
    fprintf('  [提示] 检测到已存在历史评测数据集: %s\n', save_file);
    fprintf('  直接加载并打印历史多维博弈报表...\n');
    fprintf('=========================================================================\n');
    load(save_file); % 载入已保存的 results_dukf 和 Summary_Table 等
    disp(Summary_Table);
    return; % 绕过后续耗时计算直接结束 [2]
end

% 未检测到缓存，加载环境并启动仿真
addpath(genpath('../Common'));
addpath(genpath('../Filter'));
addpath(genpath('../Data'));

%% 2. 评测超参数配置
imu_update_factors = [1];                 % IMU 更新分频因子 (对应100Hz, 50Hz, 20Hz, 10Hz)
CI_Weight          = [0.92, 0.90, 0.89, 0.88];       % 动态匹配权重 [2]
Veh_num_list       = 4 : 12;                         % 评测的车辆规模列表
dt_imu             = 0.01;                           % 100Hz 物理采样步长
uwb_downsample_factor = 10;                          % UWB 更新频率 (10Hz)

% 15维过程噪声设置
Q_sigmas_15d = [ ...
    0.0001 * ones(1, 3), ... % 位置过程噪声标准差
    0.001 * ones(1, 3), ...  % 速度过程噪声标准差
    0.00025 * ones(1, 3), ... % 加速度过程噪声标准差
    0.0001 * ones(1, 3), ...  % 姿态过程噪声标准差
    0.00025 * ones(1, 3)  ... % 角速度过程噪声标准差
    ];
Q_15d = diag(Q_sigmas_15d .^ 2);

% 结果数据矩阵
results_dukf = zeros(length(imu_update_factors), length(Veh_num_list));

%% 3. 双变量主循环仿真测试
for f_idx = 1 : length(imu_update_factors)
    imu_update_factor = imu_update_factors(f_idx);
    omega_ci = CI_Weight(f_idx);
    imu_freq = round(100 / imu_update_factor);
    
    fprintf('\n=========================================================================\n');
    fprintf('  [启动仿真组] IMU 更新频率: %3d Hz | 预测: 100 Hz | CI 权重: %.2f\n', ...
        imu_freq, omega_ci);
    fprintf('=========================================================================\n');
    
    for v_idx = 1 : length(Veh_num_list)
        veh_num = Veh_num_list(v_idx);
        
        % 数据集读取，含多级容错fallback
        data_file = sprintf('E:\\SE3_MLKF\\Data\\diff_V_6Anc\\Trj_data_Veh%d_Anc6_3D.mat', veh_num);
        if ~exist(data_file, 'file')
            data_file = sprintf('E:\\SE3_MLKF\\Data\\diff_V_6Anc_1\\Trj_data_Veh%d_Anc6_3D.mat', veh_num);
            if ~exist(data_file, 'file')
                data_file = sprintf('../Data/Trj_data_Veh%d_Anc6_3D.mat', veh_num);
                if ~exist(data_file, 'file')
                    error('未检测到数据集：%s', data_file);
                end
            end
        end
        load(data_file);
        
        % 真值重建
        for n = 1 : Vehicle_num
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
        
        % 实例化与空间预分配
        filters_dukf = cell(Vehicle_num, 1);
        pos_est_dukf = cell(Vehicle_num, 1);
        
        for n = 1 : Vehicle_num
            v_name = sprintf('V%d', n);
            veh = trajectories.(v_name);
            
            init_state = struct();
            init_state.p = [veh.X_true(1); veh.Y_true(1); veh.Z_true(1)];
            init_state.v = [veh.Vx_true(1); veh.Vy_true(1); veh.Vz_true(1)];
            init_state.a = veh.a_true(1, :)';
            
            th0 = veh.Theta_true(1);
            init_state.R = [cos(th0), -sin(th0), 0;
                            sin(th0),  cos(th0), 0;
                            0,         0,        1];
            init_state.omega = veh.omega_true(1, :)';
            
            init_cov = diag([ ...
                0.01 * ones(1, 3), ... 
                0.01 * ones(1, 3), ... 
                0.005 * ones(1, 3), ... 
                (1 * pi / 180) * ones(1, 3), ... 
                0.005 * ones(1, 3)  ... 
                ]);
            
            Sigma_a = diag(IMU_noise_params.sigma_na .^ 2 * ones(1, 3));
            Sigma_w = diag(IMU_noise_params.sigma_nw .^ 2 * ones(1, 3));
            
            filters_dukf{n} = DUKF(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu, omega_ci);
            pos_est_dukf{n} = zeros(N_steps, 3);
        end
        
        neighbors_map = cell(Vehicle_num, 1);
        for n = 1 : Vehicle_num
            if n == 1
                neighbors_map{n} = [Vehicle_num, 2];
            elseif n == Vehicle_num
                neighbors_map{n} = [Vehicle_num - 1, 1];
            else
                neighbors_map{n} = [n - 1, n + 1];
            end
        end
        
        % 运行递推
        for k = 1 : N_steps
            imu_acc = zeros(Vehicle_num, 3);
            imu_gyro = zeros(Vehicle_num, 3);
            for n = 1 : Vehicle_num
                v_name = sprintf('V%d', n);
                veh = trajectories.(v_name);
                ba_true = veh.IMU_bias_a_true(k, :)';
                bw_true = veh.IMU_bias_w_true(k, :)';
                imu_acc(n, :) = (veh.IMU_acc_m(k, :)' - ba_true)';
                imu_gyro(n, :) = (veh.IMU_gyro_m(k, :)' - bw_true)';
            end
            
            for n = 1 : Vehicle_num
                filters_dukf{n} = filters_dukf{n}.predict();
                if mod(k - 1, imu_update_factor) == 0
                    filters_dukf{n} = filters_dukf{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
                end
            end
            
            if mod(k - 1, uwb_downsample_factor) == 0
                k_uwb = (k - 1) / uwb_downsample_factor + 1;
                p_est_shared_dukf = cell(Vehicle_num, 1);
                Sigma_pos_shared_dukf = cell(Vehicle_num, 1);
                
                for n = 1 : Vehicle_num
                    [p_est_shared_dukf{n}, Sigma_pos_shared_dukf{n}] = ...
                        filters_dukf{n}.get_marginalized_position_info();
                end
                
                for n = 1 : Vehicle_num
                    v_name = sprintf('V%d', n); veh = trajectories.(v_name);
                    anchor_ranges_raw = veh.UWB_Anchor(k_uwb, 2 : end)';
                    anchor_positions_veh = anchors(1 : Anchor_num, :);
                    active_neighbors = neighbors_map{n};
                    M_neighbors = length(active_neighbors);
                    relative_ranges = zeros(M_neighbors, 1);
                    neigh_positions = zeros(M_neighbors, 3);
                    neigh_Sigma_pos = cell(M_neighbors, 1);
                    
                    for idx = 1 : M_neighbors
                        nid = active_neighbors(idx);
                        neigh_positions(idx, :) = p_est_shared_dukf{nid}';
                        neigh_Sigma_pos{idx} = Sigma_pos_shared_dukf{nid};
                        rel_val = veh.UWB_Relative(k_uwb, 1 + nid);
                        if isnan(rel_val) || isinf(rel_val)
                            rel_val = norm(p_est_shared_dukf{n} - p_est_shared_dukf{nid});
                        end
                        relative_ranges(idx) = rel_val;
                    end
                    for a_idx = 1 : length(anchor_ranges_raw)
                        if isnan(anchor_ranges_raw(a_idx)) || isinf(anchor_ranges_raw(a_idx))
                            anchor_ranges_raw(a_idx) = norm(p_est_shared_dukf{n} - anchor_positions_veh(a_idx, :)');
                        end
                    end
                    sigma_s = UWB_noise_params.sigma_anc;
                    sigma_z = UWB_noise_params.sigma_rel;
                    
                    filters_dukf{n} = filters_dukf{n}.apply_uwb_update([], ...
                        anchor_ranges_raw, anchor_positions_veh, ...
                        active_neighbors, neigh_positions, ...
                        neigh_Sigma_pos, relative_ranges, sigma_s, sigma_z);
                end
            end
            
            for n = 1 : Vehicle_num
                pos_est_dukf{n}(k, :) = filters_dukf{n}.state.p';
            end
        end
        
        % 误差计算
        pos_true = cell(Vehicle_num, 1);
        for n = 1 : Vehicle_num
            v_name = sprintf('V%d', n);
            pos_true{n} = [trajectories.(v_name).X_true, ...
                           trajectories.(v_name).Y_true, ...
                           trajectories.(v_name).Z_true];
        end
        [~, rmse_dukf] = calculate_position_errors(pos_est_dukf, pos_true);
        avg_rmse_val = mean([rmse_dukf.euc_rmse]);
        results_dukf(f_idx, v_idx) = avg_rmse_val;
        
        fprintf('  - 规模: %2d 车 | 运算完毕. 平均 RMSE = %.4f m\n', veh_num, avg_rmse_val);
    end
end

%% 4. 数据整理与 Table 构造
RowNames = cell(length(Veh_num_list) + 1, 1);
for v_idx = 1 : length(Veh_num_list)
    RowNames{v_idx} = sprintf('Vehicles_%d', Veh_num_list(v_idx));
end
RowNames{end} = 'Average (均值)';

col_100Hz = [results_dukf(1, :), mean(results_dukf(1, :))]';
col_50Hz  = [results_dukf(2, :), mean(results_dukf(2, :))]';
col_20Hz  = [results_dukf(3, :), mean(results_dukf(3, :))]';
col_10Hz  = [results_dukf(4, :), mean(results_dukf(4, :))]';

Summary_Table = table(col_100Hz, col_50Hz, col_20Hz, col_10Hz, ...
    'RowNames', RowNames, ...
    'VariableNames', {'Hz_100_w0_92', 'Hz_50_w0_90', 'Hz_20_w0_89', 'Hz_10_w0_88'});

fprintf('\n\n=========================================================================\n');
fprintf('         DUKF 多重博弈评测报表：IMU 更新速率退化与 CI 权重动态调节       \n');
fprintf('=========================================================================\n');
disp(Summary_Table);
fprintf('=========================================================================\n');

%% 5. 数据持久化保存 (防二次计算损耗)
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
    fprintf('已自动创建保存目录: %s\n', save_dir);
end

% 使用完整路径保存（关键修改）
try
    save(save_file, 'results_dukf', 'Summary_Table', 'Veh_num_list', ...
         'imu_update_factors', 'CI_Weight');
    fprintf('测试结果已自动保存至: %s\n', save_file);
catch ME
    fprintf('保存失败: %s\n', ME.message);
end