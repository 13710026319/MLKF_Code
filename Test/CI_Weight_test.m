% =========================================================================
% CI_Weight_test.m (CI 权重与 IMU 采样频率博弈评估脚本)
% 评测算法：DEKF, DIEKF
% 评估内容：当 IMU 采样率降低时，CI 权重从 0.88 提升到 0.94 对定位精度的影响
% =========================================================================
clc; clear; close all;

%% 1. 加载路径与仿真数据
addpath(genpath('../Common'));
addpath(genpath('../Filter'));
addpath(genpath('../Data'));

data_file = 'E:\SE3_MLKF\Data\diff_V_6Anc\Trj_data_Veh9_Anc6_3D.mat';
if ~exist(data_file, 'file')
    data_file = '../Data/Trj_data_Veh4_Anc5_3D.mat';
    if ~exist(data_file, 'file')
        error('未检测到指定数据文件，请先运行 Data 下数据生成脚本并指定生成对应基站数据！');
    end
end
load(data_file); % 载入 trajectories, anchors, IMU_noise_params, UWB_noise_params, Vehicle_num, Anchor_num
dt_imu = 0.01; % 100Hz 基础物理采样步长

%% 2. 状态真值重建与偏置已知设定
for n = 1 : Vehicle_num
    v_name = sprintf('V%d', n);
    veh = trajectories.(v_name);
    N_steps = length(veh.Time_true);
    % A. 重建 3D 真实加速度 a_t^i
    v_true_matrix = [veh.Vx_true, veh.Vy_true, veh.Vz_true];
    a_true_matrix = zeros(N_steps, 3);
    a_true_matrix(:, 1) = gradient(v_true_matrix(:, 1), dt_imu);
    a_true_matrix(:, 2) = gradient(v_true_matrix(:, 2), dt_imu);
    a_true_matrix(:, 3) = gradient(v_true_matrix(:, 3), dt_imu);
    trajectories.(v_name).a_true = a_true_matrix;
    % B. 重建 3D 真实角速度 \omega_t^i
    theta_unwrapped = unwrap(veh.Theta_true);
    wz_true = gradient(theta_unwrapped, dt_imu);
    omega_true_matrix = [zeros(N_steps, 2), wz_true];
    trajectories.(v_name).omega_true = omega_true_matrix;
end

%% 3. 测试超参数配置
weights = [0.88, 0.94];                     % 待评估的 CI 权重组
imu_update_factors = [1, 2, 5, 10];         % IMU 更新分频因子
uwb_downsample_factor = 10;                 % UWB 固定 10Hz 更新

% 15维过程噪声设置 (同 DMLKF)
Q_sigmas_15d = [ ...
    0.0001 * ones(1, 3), ... % 位置过程噪声标准差
    0.001 * ones(1, 3), ... % 速度过程噪声标准差
    0.00025 * ones(1, 3), ... % 加速度过程噪声标准差
    0.0001 * ones(1, 3), ... % 姿态(旋转)过程噪声标准差
    0.00025 * ones(1, 3)  ... % 角速度过程噪声标准差
    ];
Q_15d = diag(Q_sigmas_15d .^ 2);

% 保存最终平均 RMSE 结果的矩阵 (行对应权重 0.88/0.92，列对应 100Hz/50Hz/20Hz/10Hz)
results_dekf = zeros(length(weights), length(imu_update_factors));
results_diekf = zeros(length(weights), length(imu_update_factors));

%% 4. 双变量交叉评测主循环
for w_idx = 1 : length(weights)
    omega_ci = weights(w_idx);
    
    for f_idx = 1 : length(imu_update_factors)
        imu_update_factor = imu_update_factors(f_idx);
        imu_freq = round(100 / imu_update_factor);
        
        fprintf('正在评估：CI权重 = %.2f | IMU更新频率 = %d Hz...\n', omega_ci, imu_freq);
        
        % 实例化当前配置下的估值器容器
        filters_dekf = cell(Vehicle_num, 1);
        filters_diekf = cell(Vehicle_num, 1);
        pos_est_dekf = cell(Vehicle_num, 1);
        pos_est_diekf = cell(Vehicle_num, 1);
        
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
            
            % 将当前测试权重传入构造函数第 8 个参数
            filters_dekf{n} = DEKF(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu, omega_ci);
            filters_diekf{n} = DIEKF(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu, omega_ci);
            
            pos_est_dekf{n} = zeros(N_steps, 3);
            pos_est_diekf{n} = zeros(N_steps, 3);
        end
        
        % 建立全局环形双邻居拓扑
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
        
        % 仿真步进递推
        for k = 1 : N_steps
            % 提取原始 IMU 数据
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
            
            % IMU 预测（100Hz 物理驱动）与 IMU 滤波更新（变频）
            for n = 1 : Vehicle_num
                filters_dekf{n} = filters_dekf{n}.predict();
                filters_diekf{n} = filters_diekf{n}.predict();
                
                % 变频执行 IMU 更新约束
                if mod(k - 1, imu_update_factor) == 0
                    filters_dekf{n} = filters_dekf{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
                    filters_diekf{n} = filters_diekf{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
                end
            end
            
            % UWB 更新（固定 10Hz）
            if mod(k - 1, uwb_downsample_factor) == 0
                k_uwb = (k - 1) / uwb_downsample_factor + 1;
                
                p_est_shared_dekf = cell(Vehicle_num, 1);
                Sigma_pos_shared_dekf = cell(Vehicle_num, 1);
                p_est_shared_diekf = cell(Vehicle_num, 1);
                Sigma_pos_shared_diekf = cell(Vehicle_num, 1);
                
                for n = 1 : Vehicle_num
                    [p_est_shared_dekf{n}, Sigma_pos_shared_dekf{n}] = filters_dekf{n}.get_marginalized_position_info();
                    [p_est_shared_diekf{n}, Sigma_pos_shared_diekf{n}] = filters_diekf{n}.get_marginalized_position_info();
                end
                
                % DEKF UWB 更新
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
                        neigh_positions(idx, :) = p_est_shared_dekf{nid}';
                        neigh_Sigma_pos{idx} = Sigma_pos_shared_dekf{nid};
                        rel_val = veh.UWB_Relative(k_uwb, 1 + nid);
                        if isnan(rel_val) || isinf(rel_val)
                            rel_val = norm(p_est_shared_dekf{n} - p_est_shared_dekf{nid});
                        end
                        relative_ranges(idx) = rel_val;
                    end
                    for a_idx = 1 : length(anchor_ranges_raw)
                        if isnan(anchor_ranges_raw(a_idx)) || isinf(anchor_ranges_raw(a_idx))
                            anchor_ranges_raw(a_idx) = norm(p_est_shared_dekf{n} - anchor_positions_veh(a_idx, :)');
                        end
                    end
                    sigma_s = UWB_noise_params.sigma_anc;
                    sigma_z = UWB_noise_params.sigma_rel;
                    
                    filters_dekf{n} = filters_dekf{n}.apply_uwb_update([], ...
                        anchor_ranges_raw, anchor_positions_veh, ...
                        active_neighbors, neigh_positions, ...
                        neigh_Sigma_pos, relative_ranges, sigma_s, sigma_z);
                end
                
                % DIEKF UWB 更新
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
                        neigh_positions(idx, :) = p_est_shared_diekf{nid}';
                        neigh_Sigma_pos{idx} = Sigma_pos_shared_diekf{nid};
                        rel_val = veh.UWB_Relative(k_uwb, 1 + nid);
                        if isnan(rel_val) || isinf(rel_val)
                            rel_val = norm(p_est_shared_diekf{n} - p_est_shared_diekf{nid});
                        end
                        relative_ranges(idx) = rel_val;
                    end
                    for a_idx = 1 : length(anchor_ranges_raw)
                        if isnan(anchor_ranges_raw(a_idx)) || isinf(anchor_ranges_raw(a_idx))
                            anchor_ranges_raw(a_idx) = norm(p_est_shared_diekf{n} - anchor_positions_veh(a_idx, :)');
                        end
                    end
                    sigma_s = UWB_noise_params.sigma_anc;
                    sigma_z = UWB_noise_params.sigma_rel;
                    
                    filters_diekf{n} = filters_diekf{n}.apply_uwb_update([], ...
                        anchor_ranges_raw, anchor_positions_veh, ...
                        active_neighbors, neigh_positions, ...
                        neigh_Sigma_pos, relative_ranges, sigma_s, sigma_z);
                end
            end
            
            % 写入定位轨迹结果
            for n = 1 : Vehicle_num
                pos_est_dekf{n}(k, :) = filters_dekf{n}.state.p';
                pos_est_diekf{n}(k, :) = filters_diekf{n}.state.p';
            end
        end
        
        % 计算该配置下所有车辆的平均三维欧氏距离误差 RMSE
        pos_true = cell(Vehicle_num, 1);
        for n = 1 : Vehicle_num
            v_name = sprintf('V%d', n);
            pos_true{n} = [trajectories.(v_name).X_true, ...
                           trajectories.(v_name).Y_true, ...
                           trajectories.(v_name).Z_true];
        end
        
        [~, rmse_dekf] = calculate_position_errors(pos_est_dekf, pos_true);
        [~, rmse_diekf] = calculate_position_errors(pos_est_diekf, pos_true);
        
        results_dekf(w_idx, f_idx) = mean([rmse_dekf.euc_rmse]);
        results_diekf(w_idx, f_idx) = mean([rmse_diekf.euc_rmse]);
    end
end

%% 5. 数据后处理与控制台表格格式化输出
fprintf('\n仿真计算完毕。正在整理评测数据...\n');

% 计算由权重从 0.88 提升到 0.92 的提升比例 (%) 
% 提升比例 = (RMSE_0.88 - RMSE_0.92) / RMSE_0.88 * 100
improve_dekf = (results_dekf(1, :) - results_dekf(2, :)) ./ results_dekf(1, :) * 100;
improve_diekf = (results_diekf(1, :) - results_diekf(2, :)) ./ results_diekf(1, :) * 100;

% 组装表格变量
RowNames = {'100Hz (Factor=1)', '50Hz (Factor=2)', '20Hz (Factor=5)', '10Hz (Factor=10)'}';

% A. DEKF 评估表
DEKF_RMSE_0_88 = results_dekf(1, :)';
DEKF_RMSE_0_92 = results_dekf(2, :)';
DEKF_Improvement_Percent = improve_dekf';
table_dekf = table(DEKF_RMSE_0_88, DEKF_RMSE_0_92, DEKF_Improvement_Percent, ...
    'RowNames', RowNames);

% B. DIEKF 评估表
DIEKF_RMSE_0_88 = results_diekf(1, :)';
DIEKF_RMSE_0_94 = results_diekf(2, :)';
DIEKF_Improvement_Percent = improve_diekf';
table_diekf = table(DIEKF_RMSE_0_88, DIEKF_RMSE_0_92, DIEKF_Improvement_Percent, ...
    'RowNames', RowNames);

% 打印总报表
fprintf('\n=========================================================================\n');
fprintf('                CI 权重与 IMU 更新频率多重博弈测试报表                   \n');
fprintf('=========================================================================\n');

fprintf('\n>>> [1] DEKF 算法测试结果 (全维 CI EKF)\n');
disp(table_dekf);

fprintf('\n>>> [2] DIEKF 算法测试结果 (全维迭代 CI IEKF)\n');
disp(table_diekf);

