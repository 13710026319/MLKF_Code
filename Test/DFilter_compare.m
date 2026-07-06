% =========================================================================
% DFilter_compare.m (分布式多车协同定位评测脚本)
% 评测算法对比：DMLKF,V1,V2,V3,DEKF,DIEKF

% 数据拓扑规则：1 - 2 - 3 - ... - N - 1 动态环形双邻居分布式通信网络
% =========================================================================
clc; clear; close all;

%% 1. 加载路径与 6基站 仿真数据
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
dt_imu = 0.01; % 100Hz 采样步长

%% 一些超参数
max_admm_iter = 2;

SCI_rho = 0.8;  % DMLKF,DMLKF_V1
CI_rho = 1.5;     % DMLKF_V2, 发现V2的要比V1的数值大

%% 算法运行开关 (Algorithm selection flags)
run_dmlkf = 1; % 原始 DMLKF (SCI+ADMM)
run_dmlkf_v1 = 1; % 基准 DMLKF_V1 (SCI+No Joint)
run_dmlkf_v2 = 0; % 基准 DMLKF_V2 (CI+ADMM)
run_dmlkf_v3 = 0; % 新增基准 DMLKF_V3 (CI+No Joint)

run_dekf = 0; % 经典 DEKF + CI 融合
run_diekf = 0; % DIEKF_V1 + 全维 CI 融合

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

%% 3. 初始化分布式 DMLKF, DMLKF_V1, DMLKF_V2 状态估值器
% A. 【过程噪声设置】选用相对保守稳定的15维过程噪声标准差
Q_sigmas_15d = [ ...
    0.0001 * ones(1, 3), ... % 位置过程噪声标准差
    0.001 * ones(1, 3), ... % 速度过程噪声标准差
    0.00025 * ones(1, 3), ... % 加速度过程噪声标准差
    0.0001 * ones(1, 3), ... % 姿态(旋转)过程噪声标准差
    0.00025 * ones(1, 3)  ... % 角速度过程噪声标准差
    ];
Q_15d = diag(Q_sigmas_15d .^ 2);


% 实例化各车的估值器
filters = cell(Vehicle_num, 1); % 原始 DMLKF (SCI) 容器
filters_v1 = cell(Vehicle_num, 1); % 基准 DMLKF_V1 (无联合) 容器
filters_v2 = cell(Vehicle_num, 1); % DMLKF_V2 (纯CI融合) 容器
filters_v3 = cell(Vehicle_num, 1); % DMLKF_V3 (纯CI融合,无联合) 容器
filters_dekf = cell(Vehicle_num, 1);
filters_diekf = cell(Vehicle_num, 1);

pos_est_dmlkf = cell(Vehicle_num, 1); % DMLKF 定位结果
pos_est_dmlkf_v1 = cell(Vehicle_num, 1); % DMLKF_V1 定位结果
pos_est_dmlkf_v2 = cell(Vehicle_num, 1); % DMLKF_V2 定位结果
pos_est_dmlkf_v3 = cell(Vehicle_num, 1); % DMLKF_V3 定位结果
pos_est_dekf = cell(Vehicle_num, 1);
pos_est_diekf = cell(Vehicle_num, 1);

for n = 1 : Vehicle_num
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
        sin(th0), cos(th0), 0;
        0, 0, 1];
    init_state.R = R_init;
    init_state.omega = veh.omega_true(1, :)';
    % 初始状态协方差矩阵
    init_cov = diag([ ...
        0.01 * ones(1, 3), ... % 位置
        0.01 * ones(1, 3), ... % 速度
        0.005 * ones(1, 3), ... % 加速度
        (1 * pi / 180) * ones(1, 3), ... % 姿态
        0.005 * ones(1, 3)  ... % 角速度
        ]);
    % IMU测量白噪声协方差
    Sigma_a = diag(IMU_noise_params.sigma_na .^ 2 * ones(1, 3));
    Sigma_w = diag(IMU_noise_params.sigma_nw .^ 2 * ones(1, 3));

    % 分别根据开关创建对象与预分配定位结果空间
    if run_dmlkf
        filters{n} = DMLKF(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu, 0.9);
        pos_est_dmlkf{n} = zeros(N_steps, 3);
    end
    if run_dmlkf_v1
        filters_v1{n} = DMLKF_V1(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu,0.9);
        pos_est_dmlkf_v1{n} = zeros(N_steps, 3);
    end
    if run_dmlkf_v2
        filters_v2{n} = DMLKF_V2(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu);
        pos_est_dmlkf_v2{n} = zeros(N_steps, 3);
    end
    if run_dmlkf_v3
        filters_v3{n} = DMLKF_V3(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu);
        pos_est_dmlkf_v3{n} = zeros(N_steps, 3);
    end
    if run_dekf
        % DEKF实例化
        filters_dekf{n} = DEKF(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu);
        pos_est_dekf{n} = zeros(N_steps, 3);
    end
    if run_diekf
        % 实例化流形迭代 EKF
        filters_diekf{n} = DIEKF(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu);
        pos_est_diekf{n} = zeros(N_steps, 3);
    end
end


% B. 【全局通信拓扑规则】：动态环形双邻居网络 1-2-3-...-N-1
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
uwb_downsample_factor = 10; % 10步(10Hz)触发一次 UWB 分布式更新

%% 4. 主循环仿真系统 (100Hz 级 high-frequency 驱动)
fprintf('启动分布式估计主循环仿真评测...\n');
for k = 1 : N_steps
    % 进度提示
    if mod(k, 5000) == 0
        fprintf('  当前进度: %.1f%% (%d/%d 步)\n', (k / N_steps) * 100, k, N_steps);
    end
    % A. 原始 IMU 数据预加工
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
    % B. 执行高频局部惯性积分与 IMU 局部滤波更新 (100Hz)
    for n = 1 : Vehicle_num
        % 1. 原始 DMLKF
        if run_dmlkf
            filters{n} = filters{n}.predict();
            filters{n} = filters{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
        end
        % 2. DMLKF_V1 基准
        if run_dmlkf_v1
            filters_v1{n} = filters_v1{n}.predict();
            filters_v1{n} = filters_v1{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
        end
        % 3. DMLKF_V2 纯CI版
        if run_dmlkf_v2
            filters_v2{n} = filters_v2{n}.predict();
            filters_v2{n} = filters_v2{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
        end
        % 4. DMLKF_V3 (CI+无联合)
        if run_dmlkf_v3
            filters_v3{n} = filters_v3{n}.predict();
            filters_v3{n} = filters_v3{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
        end
        % 5. 经典 DEKF 更新
        if run_dekf
            filters_dekf{n} = filters_dekf{n}.predict();
            filters_dekf{n} = filters_dekf{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
        end
        % 6. 经典流形迭代 DIEKF 更新 (内部自主进行 MAP 迭代收敛)
        if run_diekf
            filters_diekf{n} = filters_diekf{n}.predict();
            filters_diekf{n} = filters_diekf{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
        end
    end

    % C. 执行分布式协同定位 UWB 滤波更新 (10Hz)
    if mod(k - 1, uwb_downsample_factor) == 0
        k_uwb = (k - 1) / uwb_downsample_factor + 1;
        % --- Step C.1: 独立边缘化，准备待广播的先验 3D 位置及信息组件 ---
        % DMLKF (SCI) 广播变量
        if run_dmlkf
            p_est_shared = cell(Vehicle_num, 1);
            I_pos_indep_shared = cell(Vehicle_num, 1);
            I_pos_dep_shared = cell(Vehicle_num, 1);
        end
        % DMLKF_V1 广播变量
        if run_dmlkf_v1
            p_est_shared_v1 = cell(Vehicle_num, 1);
            I_pos_indep_shared_v1 = cell(Vehicle_num, 1);
            I_pos_dep_shared_v1 = cell(Vehicle_num, 1);
        end
        % DMLKF_V2 (CI) 广播变量
        if run_dmlkf_v2
            p_est_shared_v2 = cell(Vehicle_num, 1);
            Sigma_pos_shared_v2 = cell(Vehicle_num, 1);
        end
        % DMLKF_V3 (CI + No Joint) 广播变量
        if run_dmlkf_v3
            p_est_shared_v3 = cell(Vehicle_num, 1);
            Sigma_pos_shared_v3 = cell(Vehicle_num, 1);
        end
        if run_dekf
            p_est_shared_dekf = cell(Vehicle_num, 1);
            Sigma_pos_shared_dekf = cell(Vehicle_num, 1);
        end
        if run_diekf
            p_est_shared_diekf = cell(Vehicle_num, 1);
            Sigma_pos_shared_diekf = cell(Vehicle_num, 1);
        end


        % 只有用SCI的需要真的边缘化，其他算法仅是用于为邻车提供位置和协方差
        for n = 1 : Vehicle_num
            % DMLKF 边缘化
            if run_dmlkf
                [p_est_shared{n}, I_pos_indep_shared{n}, I_pos_dep_shared{n}] = ...
                    filters{n}.get_marginalized_position_info();
            end
            % DMLKF_V1 边缘化
            if run_dmlkf_v1
                [p_est_shared_v1{n}, I_pos_indep_shared_v1{n}, I_pos_dep_shared_v1{n}] = ...
                    filters_v1{n}.get_marginalized_position_info();
            end
            % DMLKF_V2 边缘化
            if run_dmlkf_v2
                [p_est_shared_v2{n}, Sigma_pos_shared_v2{n}] = ...
                    filters_v2{n}.get_marginalized_position_info();
            end
            % DMLKF_V3 边缘化
            if run_dmlkf_v3
                [p_est_shared_v3{n}, Sigma_pos_shared_v3{n}] = ...
                    filters_v3{n}.get_marginalized_position_info();
            end
            if run_dekf
                [p_est_shared_dekf{n}, Sigma_pos_shared_dekf{n}] = ...
                    filters_dekf{n}.get_marginalized_position_info();
            end
            % DIEKF_V1 位置与协方差提取
            if run_diekf
                [p_est_shared_diekf{n}, Sigma_pos_shared_diekf{n}] = ...
                    filters_diekf{n}.get_marginalized_position_info();
            end
        end

        % =================================================================
        %   分支1: 原始 DMLKF (含联合邻车ADMM优化 + SCI融合)
        % =================================================================
        if run_dmlkf
            for n = 1 : Vehicle_num
                filters{n} = filters{n}.reset_dual_variables();
            end
            s_admm_all = cell(Vehicle_num, 1);
            for n = 1 : Vehicle_num
                M = length(neighbors_map{n});
                s_admm_all{n} = zeros(3 * (M + 1), 1);
            end
            dp_neigh_neigh_all = cell(Vehicle_num, 1);
            dp_neigh_self_all = cell(Vehicle_num, 1);
            for n = 1 : Vehicle_num
                M = length(neighbors_map{n});
                dp_neigh_neigh_all{n} = zeros(3, M);
                dp_neigh_self_all{n} = zeros(3, M);
            end
            for admm_k = 1 : max_admm_iter
                s_admm_new = cell(Vehicle_num, 1);
                for n = 1 : Vehicle_num
                    v_name = sprintf('V%d', n); veh = trajectories.(v_name);
                    anchor_ranges_raw = veh.UWB_Anchor(k_uwb, 2 : end)';
                    anchor_positions_veh = anchors(1 : Anchor_num, :);
                    active_neighbors = neighbors_map{n};
                    M_neighbors = length(active_neighbors);
                    relative_ranges = zeros(M_neighbors, 1);
                    neigh_positions = zeros(M_neighbors, 3);
                    for a_idx = 1 : length(anchor_ranges_raw)
                        if isnan(anchor_ranges_raw(a_idx)) || isinf(anchor_ranges_raw(a_idx))
                            anchor_ranges_raw(a_idx) = norm(p_est_shared{n} - anchor_positions_veh(a_idx, :)');
                        end
                    end
                    for idx = 1 : M_neighbors
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
                    s_admm_new{n} = filters{n}.solve_primal_public(s_admm_all{n}, ...
                        anchor_ranges_raw, anchor_positions_veh, ...
                        neigh_positions, relative_ranges, sigma_s, sigma_z, ...
                        SCI_rho, active_neighbors, ...
                        dp_neigh_neigh_all{n}, dp_neigh_self_all{n});
                end
                s_admm_all = s_admm_new;
                for n = 1 : Vehicle_num
                    active_neighbors = neighbors_map{n};
                    M_neighbors = length(active_neighbors);
                    for idx = 1 : M_neighbors
                        nid = active_neighbors(idx);
                        dp_neigh_neigh_all{n}(:, idx) = s_admm_all{nid}(1 : 3);
                        idx_of_n_in_nid = find(neighbors_map{nid} == n);
                        if ~isempty(idx_of_n_in_nid)
                            dp_neigh_self_all{n}(:, idx) = s_admm_all{nid}(3 * idx_of_n_in_nid + (1 : 3));
                        else
                            dp_neigh_self_all{n}(:, idx) = s_admm_all{n}(1 : 3);
                        end
                    end
                end
                for n = 1 : Vehicle_num
                    active_neighbors = neighbors_map{n};
                    filters{n} = filters{n}.update_dual(s_admm_all{n}, ...
                        active_neighbors, dp_neigh_neigh_all{n}, dp_neigh_self_all{n}, SCI_rho);
                end
            end
            for n = 1 : Vehicle_num
                v_name = sprintf('V%d', n); veh = trajectories.(v_name);
                anchor_ranges_raw = veh.UWB_Anchor(k_uwb, 2 : end)';
                anchor_positions_veh = anchors(1 : Anchor_num, :);
                active_neighbors = neighbors_map{n};
                M_neighbors = length(active_neighbors);
                relative_ranges = zeros(M_neighbors, 1);
                neigh_positions = zeros(M_neighbors, 3);
                neigh_I_indep = cell(M_neighbors, 1);
                neigh_I_dep = cell(M_neighbors, 1);
                for idx = 1 : M_neighbors
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
                for a_idx = 1 : length(anchor_ranges_raw)
                    if isnan(anchor_ranges_raw(a_idx)) || isinf(anchor_ranges_raw(a_idx))
                        anchor_ranges_raw(a_idx) = norm(p_est_shared{n} - anchor_positions_veh(a_idx, :)');
                    end
                end
                sigma_s = UWB_noise_params.sigma_anc;
                sigma_z = UWB_noise_params.sigma_rel;
                filters{n} = filters{n}.apply_uwb_update(s_admm_all{n}, ...
                    anchor_ranges_raw, anchor_positions_veh, ...
                    active_neighbors, neigh_positions, ...
                    neigh_I_indep, neigh_I_dep, ...
                    relative_ranges, sigma_s, sigma_z);
            end
        end

        % =================================================================
        %   分支2: DMLKF_V1 (无联合状态/无迭代，测距噪声膨胀单步GN)
        % =================================================================
        if run_dmlkf_v1
            for n = 1 : Vehicle_num
                v_name = sprintf('V%d', n); veh = trajectories.(v_name);
                anchor_ranges_raw = veh.UWB_Anchor(k_uwb, 2 : end)';
                anchor_positions_veh = anchors(1 : Anchor_num, :);
                active_neighbors = neighbors_map{n};
                M_neighbors = length(active_neighbors);
                relative_ranges = zeros(M_neighbors, 1);
                neigh_positions = zeros(M_neighbors, 3);
                neigh_I_indep = cell(M_neighbors, 1);
                neigh_I_dep = cell(M_neighbors, 1);
                for idx = 1 : M_neighbors
                    nid = active_neighbors(idx);
                    neigh_positions(idx, :) = p_est_shared_v1{nid}';
                    neigh_I_indep{idx} = I_pos_indep_shared_v1{nid};
                    neigh_I_dep{idx} = I_pos_dep_shared_v1{nid};
                    rel_val = veh.UWB_Relative(k_uwb, 1 + nid);
                    if isnan(rel_val) || isinf(rel_val)
                        rel_val = norm(p_est_shared_v1{n} - p_est_shared_v1{nid});
                    end
                    relative_ranges(idx) = rel_val;
                end
                for a_idx = 1 : length(anchor_ranges_raw)
                    if isnan(anchor_ranges_raw(a_idx)) || isinf(anchor_ranges_raw(a_idx))
                        anchor_ranges_raw(a_idx) = norm(p_est_shared_v1{n} - anchor_positions_veh(a_idx, :)');
                    end
                end
                sigma_s = UWB_noise_params.sigma_anc;
                sigma_z = UWB_noise_params.sigma_rel;
                filters_v1{n} = filters_v1{n}.apply_uwb_update([], ...
                    anchor_ranges_raw, anchor_positions_veh, ...
                    active_neighbors, neigh_positions, ...
                    neigh_I_indep, neigh_I_dep, ...
                    relative_ranges, sigma_s, sigma_z);
            end
        end

        % =================================================================
        %   分支3: DMLKF_V2 (含联合邻车ADMM优化 + 纯CI融合)
        % =================================================================
        if run_dmlkf_v2
            for n = 1 : Vehicle_num
                filters_v2{n} = filters_v2{n}.reset_dual_variables();
            end
            s_admm_all_v2 = cell(Vehicle_num, 1);
            for n = 1 : Vehicle_num
                M = length(neighbors_map{n});
                s_admm_all_v2{n} = zeros(3 * (M + 1), 1);
            end
            dp_neigh_neigh_all_v2 = cell(Vehicle_num, 1);
            dp_neigh_self_all_v2 = cell(Vehicle_num, 1);
            for n = 1 : Vehicle_num
                M = length(neighbors_map{n});
                dp_neigh_neigh_all_v2{n} = zeros(3, M);
                dp_neigh_self_all_v2{n} = zeros(3, M);
            end
            for admm_k = 1 : max_admm_iter
                s_admm_new_v2 = cell(Vehicle_num, 1);
                for n = 1 : Vehicle_num
                    v_name = sprintf('V%d', n); veh = trajectories.(v_name);
                    anchor_ranges_raw = veh.UWB_Anchor(k_uwb, 2 : end)';
                    anchor_positions_veh = anchors(1 : Anchor_num, :);
                    active_neighbors = neighbors_map{n};
                    M_neighbors = length(active_neighbors);
                    relative_ranges = zeros(M_neighbors, 1);
                    neigh_positions = zeros(M_neighbors, 3);
                    for a_idx = 1 : length(anchor_ranges_raw)
                        if isnan(anchor_ranges_raw(a_idx)) || isinf(anchor_ranges_raw(a_idx))
                            anchor_ranges_raw(a_idx) = norm(p_est_shared_v2{n} - anchor_positions_veh(a_idx, :)');
                        end
                    end
                    for idx = 1 : M_neighbors
                        nid = active_neighbors(idx);
                        neigh_positions(idx, :) = p_est_shared_v2{nid}';
                        rel_val = veh.UWB_Relative(k_uwb, 1 + nid);
                        if isnan(rel_val) || isinf(rel_val)
                            rel_val = norm(p_est_shared_v2{n} - p_est_shared_v2{nid});
                        end
                        relative_ranges(idx) = rel_val;
                    end
                    sigma_s = UWB_noise_params.sigma_anc;
                    sigma_z = UWB_noise_params.sigma_rel;
                    s_admm_new_v2{n} = filters_v2{n}.solve_primal_public(s_admm_all_v2{n}, ...
                        anchor_ranges_raw, anchor_positions_veh, ...
                        neigh_positions, relative_ranges, sigma_s, sigma_z, ...
                        CI_rho, active_neighbors, ...
                        dp_neigh_neigh_all_v2{n}, dp_neigh_self_all_v2{n});
                end
                s_admm_all_v2 = s_admm_new_v2;
                for n = 1 : Vehicle_num
                    active_neighbors = neighbors_map{n};
                    M_neighbors = length(active_neighbors);
                    for idx = 1 : M_neighbors
                        nid = active_neighbors(idx);
                        dp_neigh_neigh_all_v2{n}(:, idx) = s_admm_all_v2{nid}(1 : 3);
                        idx_of_n_in_nid = find(neighbors_map{nid} == n);
                        if ~isempty(idx_of_n_in_nid)
                            dp_neigh_self_all_v2{n}(:, idx) = s_admm_all_v2{nid}(3 * idx_of_n_in_nid + (1 : 3));
                        else
                            dp_neigh_self_all_v2{n}(:, idx) = s_admm_all_v2{n}(1 : 3);
                        end
                    end
                end
                for n = 1 : Vehicle_num
                    active_neighbors = neighbors_map{n};
                    filters_v2{n} = filters_v2{n}.update_dual(s_admm_all_v2{n}, ...
                        active_neighbors, dp_neigh_neigh_all_v2{n}, dp_neigh_self_all_v2{n}, CI_rho);
                end
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
                    neigh_positions(idx, :) = p_est_shared_v2{nid}';
                    neigh_Sigma_pos{idx} = Sigma_pos_shared_v2{nid};
                    rel_val = veh.UWB_Relative(k_uwb, 1 + nid);
                    if isnan(rel_val) || isinf(rel_val)
                        rel_val = norm(p_est_shared_v2{n} - p_est_shared_v2{nid});
                    end
                    relative_ranges(idx) = rel_val;
                end
                for a_idx = 1 : length(anchor_ranges_raw)
                    if isnan(anchor_ranges_raw(a_idx)) || isinf(anchor_ranges_raw(a_idx))
                        anchor_ranges_raw(a_idx) = norm(p_est_shared_v2{n} - anchor_positions_veh(a_idx, :)');
                    end
                end
                sigma_s = UWB_noise_params.sigma_anc;
                sigma_z = UWB_noise_params.sigma_rel;
                filters_v2{n} = filters_v2{n}.apply_uwb_update(s_admm_all_v2{n}, ...
                    anchor_ranges_raw, anchor_positions_veh, ...
                    active_neighbors, neigh_positions, ...
                    neigh_Sigma_pos, relative_ranges, sigma_s, sigma_z);
            end
        end

        % =================================================================
        %   分支4: DMLKF_V3 (纯CI融合 + 无联合状态/无迭代，测距噪声膨胀单步GN)
        % =================================================================
        if run_dmlkf_v3
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
                    neigh_positions(idx, :) = p_est_shared_v3{nid}';
                    neigh_Sigma_pos{idx} = Sigma_pos_shared_v3{nid};
                    rel_val = veh.UWB_Relative(k_uwb, 1 + nid);
                    if isnan(rel_val) || isinf(rel_val)
                        rel_val = norm(p_est_shared_v3{n} - p_est_shared_v3{nid});
                    end
                    relative_ranges(idx) = rel_val;
                end
                for a_idx = 1 : length(anchor_ranges_raw)
                    if isnan(anchor_ranges_raw(a_idx)) || isinf(anchor_ranges_raw(a_idx))
                        anchor_ranges_raw(a_idx) = norm(p_est_shared_v3{n} - anchor_positions_veh(a_idx, :)');
                    end
                end
                sigma_s = UWB_noise_params.sigma_anc;
                sigma_z = UWB_noise_params.sigma_rel;
                filters_v3{n} = filters_v3{n}.apply_uwb_update([], ...
                    anchor_ranges_raw, anchor_positions_veh, ...
                    active_neighbors, neigh_positions, ...
                    neigh_Sigma_pos, relative_ranges, sigma_s, sigma_z);
            end
        end
        % =================================================================
        %   分支5: DEKF_V1 (15D 经典 DEKF + CI 联合测距更新)
        % =================================================================
        if run_dekf
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

                % EKF 更新不使用 primal 优化结果，传入 []
                filters_dekf{n} = filters_dekf{n}.apply_uwb_update([], ...
                    anchor_ranges_raw, anchor_positions_veh, ...
                    active_neighbors, neigh_positions, ...
                    neigh_Sigma_pos, relative_ranges, sigma_s, sigma_z);
            end
        end
        % =================================================================
        %   分支6: DIEKF_V1 (15D 经典迭代 DEKF + CI 联合测距更新)
        % =================================================================
        if run_diekf
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

                % 迭代 EKF 协同 MAP 求解不依赖 ADMM 对偶估计，第一个参数传入 []
                filters_diekf{n} = filters_diekf{n}.apply_uwb_update([], ...
                    anchor_ranges_raw, anchor_positions_veh, ...
                    active_neighbors, neigh_positions, ...
                    neigh_Sigma_pos, relative_ranges, sigma_s, sigma_z);
            end
        end


    end

    % D. 记录最终定位状态
    for n = 1 : Vehicle_num
        if run_dmlkf
            pos_est_dmlkf{n}(k, :) = filters{n}.state.p';
        end
        if run_dmlkf_v1
            pos_est_dmlkf_v1{n}(k, :) = filters_v1{n}.state.p';
        end
        if run_dmlkf_v2
            pos_est_dmlkf_v2{n}(k, :) = filters_v2{n}.state.p';
        end
        if run_dmlkf_v3
            pos_est_dmlkf_v3{n}(k, :) = filters_v3{n}.state.p';
        end
        if run_dekf
            pos_est_dekf{n}(k, :) = filters_dekf{n}.state.p';
        end
        if run_diekf
            pos_est_diekf{n}(k, :) = filters_diekf{n}.state.p';
        end
    end
end
fprintf('滤波解算主循环执行完毕。\n');

%% 5. 位置误差计算与控制台评估报表输出
pos_true = cell(Vehicle_num, 1);
for n = 1 : Vehicle_num
    v_name = sprintf('V%d', n);
    pos_true{n} = [trajectories.(v_name).X_true, ...
        trajectories.(v_name).Y_true, ...
        trajectories.(v_name).Z_true];
end

% 仅计算已运行算法的位置误差与 RMSE
if run_dmlkf
    [errors_dmlkf, rmse_dmlkf] = calculate_position_errors(pos_est_dmlkf, pos_true);
end
if run_dmlkf_v1
    [errors_dmlkf_v1, rmse_dmlkf_v1] = calculate_position_errors(pos_est_dmlkf_v1, pos_true);
end
if run_dmlkf_v2
    [errors_dmlkf_v2, rmse_dmlkf_v2] = calculate_position_errors(pos_est_dmlkf_v2, pos_true);
end
if run_dmlkf_v3
    [errors_dmlkf_v3, rmse_dmlkf_v3] = calculate_position_errors(pos_est_dmlkf_v3, pos_true);
end
if run_dekf
    [errors_dekf, rmse_dekf] = calculate_position_errors(pos_est_dekf, pos_true);
end
if run_diekf
    [errors_diekf, rmse_diekf] = calculate_position_errors(pos_est_diekf, pos_true);
end

% 组装分布式定位对照报表并显示
RowNames = cell(Vehicle_num + 1, 1);
for n = 1 : Vehicle_num
    RowNames{n} = sprintf('Vehicle_%d', n);
end
RowNames{Vehicle_num + 1} = 'Average';

% 动态构建 Table 的列变量与列名称
vars = {};
var_names = {};

if run_dmlkf
    DMLKF_Euc_RMSE = zeros(Vehicle_num + 1, 1);
    for n = 1 : Vehicle_num
        DMLKF_Euc_RMSE(n) = rmse_dmlkf(n).euc_rmse;
    end
    DMLKF_Euc_RMSE(Vehicle_num + 1) = mean(DMLKF_Euc_RMSE(1 : Vehicle_num));
    vars{end + 1} = DMLKF_Euc_RMSE;
    var_names{end + 1} = 'DMLKF_Euc_RMSE';
end

if run_dmlkf_v1
    V1_Euc_RMSE = zeros(Vehicle_num + 1, 1);
    for n = 1 : Vehicle_num
        V1_Euc_RMSE(n) = rmse_dmlkf_v1(n).euc_rmse;
    end
    V1_Euc_RMSE(Vehicle_num + 1) = mean(V1_Euc_RMSE(1 : Vehicle_num));
    vars{end + 1} = V1_Euc_RMSE;
    var_names{end + 1} = 'V1_Euc_RMSE';
end

if run_dmlkf_v2
    V2_Euc_RMSE = zeros(Vehicle_num + 1, 1);
    for n = 1 : Vehicle_num
        V2_Euc_RMSE(n) = rmse_dmlkf_v2(n).euc_rmse;
    end
    V2_Euc_RMSE(Vehicle_num + 1) = mean(V2_Euc_RMSE(1 : Vehicle_num));
    vars{end + 1} = V2_Euc_RMSE;
    var_names{end + 1} = 'V2_Euc_RMSE';
end

if run_dmlkf_v3
    V3_Euc_RMSE = zeros(Vehicle_num + 1, 1);
    for n = 1 : Vehicle_num
        V3_Euc_RMSE(n) = rmse_dmlkf_v3(n).euc_rmse;
    end
    V3_Euc_RMSE(Vehicle_num + 1) = mean(V3_Euc_RMSE(1 : Vehicle_num));
    vars{end + 1} = V3_Euc_RMSE;
    var_names{end + 1} = 'V3_Euc_RMSE';
end

if run_dekf
    DEKF_Euc_RMSE = zeros(Vehicle_num + 1, 1);
    for n = 1 : Vehicle_num
        DEKF_Euc_RMSE(n) = rmse_dekf(n).euc_rmse;
    end
    DEKF_Euc_RMSE(Vehicle_num + 1) = mean(DEKF_Euc_RMSE(1 : Vehicle_num));
    vars{end + 1} = DEKF_Euc_RMSE;
    var_names{end + 1} = 'DEKF_V1_Euc_RMSE';
end

if run_diekf
    DIEKF_Euc_RMSE = zeros(Vehicle_num + 1, 1);
    for n = 1 : Vehicle_num
        DIEKF_Euc_RMSE(n) = rmse_diekf(n).euc_rmse;
    end
    DIEKF_Euc_RMSE(Vehicle_num + 1) = mean(DIEKF_Euc_RMSE(1 : Vehicle_num));
    vars{end + 1} = DIEKF_Euc_RMSE;
    var_names{end + 1} = 'DIEKF_V1_Euc_RMSE';
end

if ~isempty(vars)
    rmse_report_table = table(vars{:}, 'RowNames', RowNames, 'VariableNames', var_names);
    fprintf('\n======================= 分布式协同定位性能评估表 (%d 基站) =======================\n', Anchor_num);
    disp(rmse_report_table);
    fprintf('-------------------------------------------------------------------------------------------------------------\n');

    if run_dmlkf
        fprintf(' DMLKF    (SCI融合 + 含联合ADMM优化)    全局平均欧氏定位误差: %.4f m\n', DMLKF_Euc_RMSE(Vehicle_num + 1));
    end
    if run_dmlkf_v1
        fprintf(' DMLKF_V1 (SCI融合 + 无联合邻车估计/GN) 全局平均欧氏定位误差: %.4f m\n', V1_Euc_RMSE(Vehicle_num + 1));
    end
    if run_dmlkf_v2
        fprintf(' DMLKF_V2 (纯CI融合 + 含联合ADMM优化)   全局平均欧氏定位误差: %.4f m\n', V2_Euc_RMSE(Vehicle_num + 1));
    end
    if run_dmlkf_v3
        fprintf(' DMLKF_V3 (纯CI融合 + 无联合邻车估计/GN)   全局平均欧氏定位误差: %.4f m\n', V3_Euc_RMSE(Vehicle_num + 1));
    end
    if run_dekf
        fprintf(' DEKF   (经典 DEKF + 全维 CI)         全局平均欧氏定位误差: %.4f m\n', DEKF_Euc_RMSE(Vehicle_num + 1));
    end
    if run_diekf
        fprintf(' DIEKF  (经典迭代 DEKF + 全维 CI)   全局平均欧氏定位误差: %.4f m\n', DIEKF_Euc_RMSE(Vehicle_num + 1));
    end

    % 根据开关状态自适应计算增益对比
    if run_dmlkf && run_dmlkf_v1
        improvement_v1 = (V1_Euc_RMSE(Vehicle_num + 1) - DMLKF_Euc_RMSE(Vehicle_num + 1)) / V1_Euc_RMSE(Vehicle_num + 1) * 100;
        fprintf(' [增益对比] 相比V1，状态联合优化使整体定位精度提升了: %.2f%%\n', improvement_v1);
    end
    if run_dmlkf && run_dmlkf_v2
        improvement_v2 = (V2_Euc_RMSE(Vehicle_num + 1) - DMLKF_Euc_RMSE(Vehicle_num + 1)) / V2_Euc_RMSE(Vehicle_num + 1) * 100;
        fprintf(' [增益对比] 相比V2，SCI使整体定位精度提升了: %.2f%%\n', improvement_v2);
    end
    if run_dmlkf && run_dmlkf_v3
        improvement_v3 = (V3_Euc_RMSE(Vehicle_num + 1) - DMLKF_Euc_RMSE(Vehicle_num + 1)) / V3_Euc_RMSE(Vehicle_num + 1) * 100;
        fprintf(' [增益对比] 相比V3，SCI+联合优化使整体定位精度提升了: %.2f%%\n', improvement_v3);
    end
    if run_dmlkf && run_dekf
        improvement_dekf = (DEKF_Euc_RMSE(Vehicle_num + 1) - DMLKF_Euc_RMSE(Vehicle_num + 1)) / DEKF_Euc_RMSE(Vehicle_num + 1) * 100;
        fprintf(' [增益对比] 相比经典DEKF，DMLKF (联合共识+SCI) 使定位精度提升了: %.2f%%\n', improvement_dekf);
    end
    if run_dmlkf && run_diekf
        improvement_diekf = (DIEKF_Euc_RMSE(Vehicle_num + 1) - DMLKF_Euc_RMSE(Vehicle_num + 1)) / DIEKF_Euc_RMSE(Vehicle_num + 1) * 100;
        fprintf(' [增益对比] 相比经典迭代DIEKF，DMLKF (联合共识+SCI) 使定位精度提升了: %.2f%%\n', improvement_diekf);
    end

    fprintf('=============================================================================================================\n');
else
    fprintf('警告: 未选择运行任何算法，无评估结果。\n');
end

%% 6. 绘制多车欧氏定位误差对比曲线
% if ~isempty(vars)
%     time_arr = trajectories.V1.IMU_Time;
%     n_rows = ceil(sqrt(Vehicle_num));
%     n_cols = ceil(Vehicle_num / n_rows);
%     figure('Name', 'Euclidean Position Errors Comparison', 'Position', [100, 100, 1200, 800]);
%     for n = 1 : Vehicle_num
%         subplot(n_rows, n_cols, n);
%         hold on; grid on;
% 
%         legend_entries = {};
% 
%         % 动态绘制曲线与添加图例
%         if run_dmlkf
%             plot(time_arr, errors_dmlkf(n).euc_err, 'k-', 'LineWidth', 1.5);
%             legend_entries{end + 1} = 'DMLKF (SCI+ADMM)';
%         end
%         if run_dmlkf_v1
%             plot(time_arr, errors_dmlkf_v1(n).euc_err, 'r--', 'LineWidth', 1.2);
%             legend_entries{end + 1} = 'DMLKF\_V1 (No Joint)';
%         end
%         if run_dmlkf_v2
%             plot(time_arr, errors_dmlkf_v2(n).euc_err, 'b-.', 'LineWidth', 1.2);
%             legend_entries{end + 1} = 'DMLKF\_V2 (CI+ADMM)';
%         end
% 
%         if run_dmlkf_v3
%             plot(time_arr, errors_dmlkf_v3(n).euc_err, 'g:', 'LineWidth', 1.2);
%             legend_entries{end + 1} = 'DMLKF\_V3 (CI+No Joint)';
%         end
% 
%         if run_dekf
%             plot(time_arr, errors_dekf(n).euc_err, 'm-.', 'LineWidth', 1.2); % 采用品红色点划线
%             legend_entries{end + 1} = 'DEKF (Classic CI)';
%         end
%         if run_diekf
%             plot(time_arr, errors_diekf(n).euc_err, 'c--', 'LineWidth', 1.2); % 品青色虚线
%             legend_entries{end + 1} = 'DIEKF (Iterated CI)';
%         end
% 
%         title(sprintf('Vehicle %d Euclidean Error', n));
%         xlabel('Time (s)'); ylabel('Error (m)');
%         legend(legend_entries, 'Location', 'northeast');
%         xlim([0, time_arr(end)]);
%         ylim([0, 1.5]);
%         hold off;
%     end
% end
