% =========================================================================
% DFilter_compare.m (分布式多车协同定位评测脚本)
% 评测算法：Decentralized DMLKF (15D分布式极大似然卡尔曼滤波)
% 数据拓扑规则：1 - 2 - 3 - 4 - 1 环形双邻居分布式通信网络
% =========================================================================

clc; clear; close all;

%% 1. 加载路径与 5基站 仿真数据
addpath(genpath('../Common'));
addpath(genpath('../Filter'));
addpath(genpath('../Data'));

data_file = 'E:\SE3_MLKF\Data\Trj_data_Veh4_Anc6_3D.mat';
if ~exist(data_file, 'file')
    data_file = '../Data/Trj_data_Veh4_Anc5_3D.mat'; 
    if ~exist(data_file, 'file')
        error('未检测到指定数据文件，请先运行 Data 下数据生成脚本并指定生成 5 基站数据！');
    end
end
load(data_file); % 载入 trajectories, anchors, IMU_noise_params, UWB_noise_params, Vehicle_num, Anchor_num

dt_imu = 0.01; % 100Hz 采样步长

%% 2. 状态真值重建与偏置已知设定
for n = 1:Vehicle_num
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

%% 3. 初始化分布式 DMLKF 状态估值器
% A. 【过程噪声设置】选用相对保守稳定的15维过程噪声标准差
Q_sigmas_15d = [ ...
    0.0005 * ones(1,3), ... % 位置过程噪声标准差
    0.0005 * ones(1,3), ... % 速度过程噪声标准差
    0.001 * ones(1,3), ... % 加速度过程噪声标准差
    0.0001 * ones(1,3), ... % 姿态(旋转)过程噪声标准差
    0.00025 * ones(1,3)  ... % 角速度过程噪声标准差
];



Q_15d = diag(Q_sigmas_15d.^2);

% 实例化各车的估值器
filters = cell(Vehicle_num, 1);
pos_est_dmlkf = cell(Vehicle_num, 1);

for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);
    veh = trajectories.(v_name);
    
    % 提取并构建初始状态估计
    init_state = struct();
    init_state.p = [veh.X_true(1); veh.Y_true(1); veh.Z_true(1)];
    init_state.v = [veh.Vx_true(1); veh.Vy_true(1); veh.Vz_true(1)];
    init_state.a = veh.a_true(1, :)';
    
    % 初始二维姿态转三维旋转矩阵
    th0 = veh.Theta_true(1);
    R_init = [cos(th0), -sin(th0), 0;
              sin(th0),  cos(th0), 0;
              0,         0,        1];
    init_state.R = R_init;
    init_state.omega = veh.omega_true(1, :)';
    
    % 初始状态协方差矩阵
    init_cov = diag([ ...
        0.01 * ones(1,3), ... % 位置
        0.01 * ones(1,3), ... % 速度
        0.005 * ones(1,3), ... % 加速度
        (1*pi/180) * ones(1,3), ... % 姿态
        0.005 * ones(1,3)  ... % 角速度
    ]);
    
    % IMU测量白噪声协方差
    Sigma_a = diag(IMU_noise_params.sigma_na.^2 * ones(1,3));
    Sigma_w = diag(IMU_noise_params.sigma_nw.^2 * ones(1,3));
    
    % 创建 DMLKF 对象
    filters{n} = DMLKF(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu);
    
    % 预分配定位结果空间
    pos_est_dmlkf{n} = zeros(N_steps, 3);
end

% B. 【全局通信拓扑规则】：1-2-3-4-1 环形双邻居网络
% 节点 1 邻居是 [2, 4]; 节点 2 邻居是 [1, 3]; 节点 3 邻居是 [2, 4]; 节点 4 邻居是 [3, 1]
neighbors_map = { [2, 4], [1, 3], [2, 4], [3, 1] };
uwb_downsample_factor = 10; % 10步(10Hz)触发一次 UWB 分布式更新

%% 4. 主循环仿真系统 (100Hz 级高频驱动)
fprintf('启动 DMLKF 分布式估计主循环仿真实时评测...\n');
for k = 1:N_steps
    % 进度提示
    if mod(k, 5000) == 0
        fprintf('  当前进度: %.1f%% (%d/%d 步)\n', (k/N_steps)*100, k, N_steps);
    end
    
    % A. 偏置已知时的原始 IMU 数据预加工
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
    
    % B. 执行高频局部惯性积分与 IMU 局部滤波更新 (100Hz)
    for n = 1:Vehicle_num
        % 1. 先验状态与分裂协方差传播
        filters{n} = filters{n}.predict();
        
        % 2. 局部非线性最大似然滤波估计姿态与加速度
        filters{n} = filters{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3,1), zeros(3,1));
    end
    
    % C. 执行分布式协同定位 UWB 一致性滤波更新 (10Hz)
    if mod(k-1, uwb_downsample_factor) == 0
        k_uwb = (k-1)/uwb_downsample_factor + 1;
        
        % --- Step C.1: 各车独立边缘化，准备待广播的先验 3D 位置及分裂信息组件 ---
        p_est_shared = cell(Vehicle_num, 1);
        I_pos_indep_shared = cell(Vehicle_num, 1);
        I_pos_dep_shared = cell(Vehicle_num, 1);
        for n = 1:Vehicle_num
            [p_est_shared{n}, I_pos_indep_shared{n}, I_pos_dep_shared{n}] = ...
                filters{n}.get_marginalized_position_info();
        end
        
        % --- Step C.2: 分布式一致性网络 ADMM 仿真架构 (在外层循环层级驱动) ---
        max_admm_iter = 2;
        rho = 1.4;
        
        % 新增：在开始本轮 ADMM 迭代前，重置所有车辆的拉格朗日乘子，保证切空间初值一致性
        for n = 1:Vehicle_num
            filters{n} = filters{n}.reset_dual_variables();
        end

        % 1. 为所有车辆初始化一阶误差状态
        s_admm_all = cell(Vehicle_num, 1);
        for n = 1:Vehicle_num
            M = length(neighbors_map{n});
            s_admm_all{n} = zeros(3 * (M + 1), 1);
        end
        
        % 2. 各节点本地一致性状态分发缓冲容器
        dp_neigh_neigh_all = cell(Vehicle_num, 1);
        dp_neigh_self_all = cell(Vehicle_num, 1);
        for n = 1:Vehicle_num
            M = length(neighbors_map{n});
            dp_neigh_neigh_all{n} = zeros(3, M);
            dp_neigh_self_all{n} = zeros(3, M);
        end
        
        % 3. 全局 ADMM 外层通信迭代开始
        for admm_k = 1:max_admm_iter
            s_admm_new = cell(Vehicle_num, 1);
            
            % Sub-Step 1: 所有车辆物理隔离，基于上一轮估计独立求解 Primal Update (内层GN)
            for n = 1:Vehicle_num
                v_name = sprintf('V%d', n);
                veh = trajectories.(v_name);
                
                % 读取该车观测到的基站测距数据与视线
                anchor_ranges_raw = veh.UWB_Anchor(k_uwb, 2:end)';
                anchor_positions_veh = anchors(1:Anchor_num, :);
                active_neighbors = neighbors_map{n};
                M_neighbors = length(active_neighbors);
                
                relative_ranges = zeros(M_neighbors, 1);
                neigh_positions = zeros(M_neighbors, 3);
                
                % 基站测距 NaN 净化防线
                for a_idx = 1:length(anchor_ranges_raw)
                    if isnan(anchor_ranges_raw(a_idx)) || isinf(anchor_ranges_raw(a_idx))
                        anchor_ranges_raw(a_idx) = norm(p_est_shared{n} - anchor_positions_veh(a_idx, :)');
                    end
                end
                
                % 相对协同邻居测距/状态 NaN 净化防线
                for idx = 1:M_neighbors
                    nid = active_neighbors(idx);
                    neigh_positions(idx, :) = p_est_shared{nid}';
                    
                    rel_val = veh.UWB_Relative(k_uwb, 1 + nid);
                    if isnan(rel_val) || isinf(rel_val)
                        rel_val = norm(p_est_shared{n} - p_est_shared{nid});
                    end
                    relative_ranges(idx) = rel_val;
                end
                
                sigma_s = UWB_noise_params.sigma_anc;
                sigma_z = UWB_noise_params.sigma_rel;
                
                % 各车本地调用求解器解算 Primal Update (Eq. 54)
                s_admm_new{n} = filters{n}.solve_primal_public(s_admm_all{n}, ...
                    anchor_ranges_raw, anchor_positions_veh, ...
                    neigh_positions, relative_ranges, sigma_s, sigma_z, ...
                    rho, active_neighbors, ...
                    dp_neigh_neigh_all{n}, dp_neigh_self_all{n});
            end
            s_admm_all = s_admm_new;
            
            % Sub-Step 2: 真实通信网络拓扑交换 (提取邻车 Primal 更新值分发给本车)
            for n = 1:Vehicle_num
                active_neighbors = neighbors_map{n};
                M_neighbors = length(active_neighbors);
                for idx = 1:M_neighbors
                    nid = active_neighbors(idx);
                    
                    % 邻车 nid 对自身的估计 (即该邻车 s_admm 的前3维)
                    dp_neigh_neigh_all{n}(:, idx) = s_admm_all{nid}(1:3);
                    
                    % 邻居 nid 对本车 n 的估计 (需要检索车 n 在邻车 nid 的邻居列表中的索引槽位)
                    idx_of_n_in_nid = find(neighbors_map{nid} == n);
                    if ~isempty(idx_of_n_in_nid)
                        % 邻车 nid 本地估计本车误差对应的槽位索引为 3*idx_of_n_in_nid + (1:3)
                        dp_neigh_self_all{n}(:, idx) = s_admm_all{nid}(3 * idx_of_n_in_nid + (1:3));
                    else
                        dp_neigh_self_all{n}(:, idx) = s_admm_all{n}(1:3); % 降级异常拦截
                    end
                end
            end
            
            % Sub-Step 3: 各车独立更新本地对偶乘子 (Eq. 56, 57)
            for n = 1:Vehicle_num
                active_neighbors = neighbors_map{n};
                filters{n} = filters{n}.update_dual(s_admm_all{n}, ...
                    active_neighbors, dp_neigh_neigh_all{n}, dp_neigh_self_all{n}, rho);
            end
        end
        
        % Step C.3: 全局ADMM收敛后，触发最终的信息融合与流形重回射更新
        for n = 1:Vehicle_num
            v_name = sprintf('V%d', n);
            veh = trajectories.(v_name);
            anchor_ranges_raw = veh.UWB_Anchor(k_uwb, 2:end)';
            anchor_positions_veh = anchors(1:Anchor_num, :);
            active_neighbors = neighbors_map{n};
            M_neighbors = length(active_neighbors);
            
            relative_ranges = zeros(M_neighbors, 1);
            neigh_positions = zeros(M_neighbors, 3);
            neigh_I_indep = cell(M_neighbors, 1);
            neigh_I_dep = cell(M_neighbors, 1);
            
            for idx = 1:M_neighbors
                nid = active_neighbors(idx);
                neigh_positions(idx, :) = p_est_shared{nid}';
                neigh_I_indep{idx} = I_pos_indep_shared{nid};
                neigh_I_dep{idx} = I_pos_dep_shared{nid};
                
                rel_val = veh.UWB_Relative(k_uwb, 1 + nid);
                if isnan(rel_val) || isinf(rel_val)
                    rel_val = norm(p_est_shared{n} - p_est_shared{nid});
                end
                relative_ranges(idx) = rel_val;
            end
            
            for a_idx = 1:length(anchor_ranges_raw)
                if isnan(anchor_ranges_raw(a_idx)) || isinf(anchor_ranges_raw(a_idx))
                    anchor_ranges_raw(a_idx) = norm(p_est_shared{n} - anchor_positions_veh(a_idx, :)');
                end
            end
            
            sigma_s = UWB_noise_params.sigma_anc;
            sigma_z = UWB_noise_params.sigma_rel;
            
            % 提取收敛值，做 Schur 补 SCI 边际化最终融合
            filters{n} = filters{n}.apply_uwb_update(s_admm_all{n}, ...
                anchor_ranges_raw, anchor_positions_veh, ...
                active_neighbors, neigh_positions, ...
                neigh_I_indep, neigh_I_dep, ...
                relative_ranges, sigma_s, sigma_z);
        end
    end
    
    % D. 记录最终定位状态
    for n = 1:Vehicle_num
        pos_est_dmlkf{n}(k, :) = filters{n}.state.p';
    end
end
fprintf('滤波解算主循环执行完毕。\n');

%% 5. 位置误差计算与控制台评估报表输出
pos_true = cell(Vehicle_num, 1);
for n = 1:Vehicle_num
    v_name = sprintf('V%d', n);
    pos_true{n} = [trajectories.(v_name).X_true, ...
                   trajectories.(v_name).Y_true, ...
                   trajectories.(v_name).Z_true];
end

[errors_dmlkf, rmse_dmlkf] = calculate_position_errors(pos_est_dmlkf, pos_true);

% 组装分布式定位报表并显示
RowNames = cell(Vehicle_num, 1);
X_RMSE = zeros(Vehicle_num, 1);
Y_RMSE = zeros(Vehicle_num, 1);
Z_RMSE = zeros(Vehicle_num, 1);
Euc_RMSE = zeros(Vehicle_num, 1);

for n = 1:Vehicle_num
    RowNames{n} = sprintf('Vehicle_%d_DMLKF_15D', n);
    X_RMSE(n)   = rmse_dmlkf(n).axis_rmse(1);
    Y_RMSE(n)   = rmse_dmlkf(n).axis_rmse(2);
    Z_RMSE(n)   = rmse_dmlkf(n).axis_rmse(3);
    Euc_RMSE(n) = rmse_dmlkf(n).euc_rmse;
end

rmse_report_table = table(X_RMSE, Y_RMSE, Z_RMSE, Euc_RMSE, 'RowNames', RowNames);
mean_euc_rmse = mean(Euc_RMSE);

fprintf('\n============================ DMLKF 算法性能评估对比表 (5 基站) ============================\n');
disp(rmse_report_table);
fprintf('----------------------------------------------------------------------------------------\n');
fprintf('4辆车辆全局平均欧氏定位误差 (Mean Euclidean RMSE): %.4f m\n', mean_euc_rmse);
fprintf('========================================================================================\n');

%% 6. 绘制多车欧氏定位误差曲线
time_arr = trajectories.V1.IMU_Time;
figure('Name', 'Euclidean Position Errors for Distributed DMLKF', 'Position', [100, 100, 1000, 700]);

for n = 1:Vehicle_num
    subplot(2, 2, n);
    hold on; grid on;
    
    % 绘制 DMLKF 的欧氏误差
    plot(time_arr, errors_dmlkf(n).euc_err, 'k-', 'LineWidth', 1.5);
    
    title(sprintf('Vehicle %d Euclidean Error (DMLKF)', n));
    xlabel('Time (s)'); ylabel('Error (m)');
    xlim([0, time_arr(end)]);
    ylim([0, 1.5]); 
    hold off;
end