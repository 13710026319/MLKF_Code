% =========================================================================
% D_V_num_6Anc_compare.m
% 分布式多算法(DMLKF,V1,V2,V3,DEKF,DIEKF)在6基站、不同车辆数(4-12)下的表现对比
% 基准：DEKF，提升百分比相对于DEKF计算
% =========================================================================
clc; clear; close all;

%% 1. 路径与评测参数配置
addpath(genpath('../Common'));
addpath(genpath('../Filter'));
addpath(genpath('../Data'));

Veh_list = 4 : 12;
uwb_downsample_factor = 10;
imu_update_factors = [1, 2, 5, 10, 25]; % IMU update 1 = 100Hz, 2 = 50Hz, 5 = 20Hz, 10 = 10Hz, 25 = 4Hz 等

N_veh_tests = length(Veh_list);

save_dir = 'E:\SE3_MLKF\Result';

%% 2. 核心评测循环

max_admm_iter = 2;
SCI_rho = 0.8;
CI_rho = 1.5;

for f_idx = 1:length(imu_update_factors)
    imu_update_factor = imu_update_factors(f_idx);
    fprintf('\n========== 开始 IMU factor = %d (约 %.1f Hz) ==========\n', ...
            imu_update_factor, 100/imu_update_factor);
    
    % 为当前frequency生成文件名
    save_name = sprintf('D_V_num_6Anc_IMU_%dHZ.mat', 100/imu_update_factor);
    save_path = fullfile(save_dir, save_name);
    
    if exist(save_path, 'file')
        fprintf('检测到历史数据 [%s]，直接加载...\n', save_name);
        load(save_path);
        jump_to_plot = true;
    else
        jump_to_plot = false;
        fprintf('未找到结果，开始计算...\n');
    end

if ~jump_to_plot
    for idx_veh = 1 : N_veh_tests
        veh_num = Veh_list(idx_veh);
        fprintf('\n>>> 当前评测数据集车辆数量: %d <<<\n', veh_num);

        % 加载数据集
        data_file = sprintf('E:\\SE3_MLKF\\Data\\diff_V_6Anc\\Trj_data_Veh%d_Anc6_3D.mat', veh_num);
        if ~exist(data_file, 'file')
            error('未检测到指定数据集：%s', data_file);
        end
        load(data_file);
        dt_imu = 0.01;
        

        %% 状态真值重建
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

        %% 初始化所有滤波器
        Q_sigmas_15d = [0.0001 * ones(1, 3), 0.001 * ones(1, 3), 0.00025 * ones(1, 3), ...
            0.0001 * ones(1, 3), 0.00025 * ones(1, 3)];
        Q_15d = diag(Q_sigmas_15d .^ 2);

        init_cov = diag([0.01 * ones(1, 3), 0.01 * ones(1, 3), 0.005 * ones(1, 3), ...
            (1 * pi / 180) * ones(1, 3), 0.005 * ones(1, 3)]);

        Sigma_a = diag(IMU_noise_params.sigma_na .^ 2 * ones(1, 3));
        Sigma_w = diag(IMU_noise_params.sigma_nw .^ 2 * ones(1, 3));

        filters_dekf = cell(Vehicle_num, 1);
        filters_diekf = cell(Vehicle_num, 1);
        filters_v3 = cell(Vehicle_num, 1);
        filters_v2 = cell(Vehicle_num, 1);
        filters_v1 = cell(Vehicle_num, 1);
        filters_dmlkf = cell(Vehicle_num, 1);

        pos_est_dekf = cell(Vehicle_num, 1); pos_est_diekf = cell(Vehicle_num, 1);
        pos_est_v3 = cell(Vehicle_num, 1); pos_est_v2 = cell(Vehicle_num, 1);
        pos_est_v1 = cell(Vehicle_num, 1); pos_est_dmlkf = cell(Vehicle_num, 1);

        for n = 1 : Vehicle_num
            v_name = sprintf('V%d', n);
            veh = trajectories.(v_name);

            init_state.p = [veh.X_true(1); veh.Y_true(1); veh.Z_true(1)];
            init_state.v = [veh.Vx_true(1); veh.Vy_true(1); veh.Vz_true(1)];
            init_state.a = veh.a_true(1, :)';
            th0 = veh.Theta_true(1);
            init_state.R = [cos(th0), -sin(th0), 0; sin(th0), cos(th0), 0; 0, 0, 1];
            init_state.omega = veh.omega_true(1, :)';

            % 初始化各算法
            filters_dekf{n} = DEKF(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu);
            filters_diekf{n} = DIEKF(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu);
            filters_v3{n} = DMLKF_V3(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu);
            filters_v2{n} = DMLKF_V2(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu);
            filters_v1{n} = DMLKF_V1(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu);
            filters_dmlkf{n} = DMLKF(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu);

            % 预分配结果
            pos_est_dekf{n} = zeros(N_steps, 3);
            pos_est_diekf{n} = zeros(N_steps, 3);
            pos_est_v3{n} = zeros(N_steps, 3);
            pos_est_v2{n} = zeros(N_steps, 3);
            pos_est_v1{n} = zeros(N_steps, 3);
            pos_est_dmlkf{n} = zeros(N_steps, 3);
        end

        % 通信拓扑
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

        %% 主循环
        fprintf('开始主循环仿真...\n');
        for k = 1 : N_steps
            if mod(k, 5000) == 0
                fprintf('  进度: %.1f%% (%d/%d)\n', k / N_steps * 100, k, N_steps);
            end

            % IMU 数据
            imu_acc = zeros(Vehicle_num, 3);
            imu_gyro = zeros(Vehicle_num, 3);
            for n = 1 : Vehicle_num
                v_name = sprintf('V%d', n);
                veh = trajectories.(v_name);
                ba = veh.IMU_bias_a_true(k, :)';
                bw = veh.IMU_bias_w_true(k, :)';
                imu_acc(n, :) = (veh.IMU_acc_m(k, :)' - ba)';
                imu_gyro(n, :) = (veh.IMU_gyro_m(k, :)' - bw)';
            end

            % IMU 部分
            for n = 1 : Vehicle_num
                % IMU 预测
                filters_dekf{n} = filters_dekf{n}.predict();
                filters_diekf{n} = filters_diekf{n}.predict();
                filters_v3{n} = filters_v3{n}.predict();
                filters_v2{n} = filters_v2{n}.predict();
                filters_v1{n} = filters_v1{n}.predict();
                filters_dmlkf{n} = filters_dmlkf{n}.predict();

                % IMU 更新（使用当前时刻真实 IMU 数据）
                if mod(k - 1, imu_update_factor) == 0
                    filters_dekf{n} = filters_dekf{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
                    filters_diekf{n} = filters_diekf{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
                    filters_v3{n} = filters_v3{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
                    filters_v2{n} = filters_v2{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
                    filters_v1{n} = filters_v1{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
                    filters_dmlkf{n} = filters_dmlkf{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
                end
            end

            % UWB 更新 (10Hz)
            if mod(k - 1, uwb_downsample_factor) == 0

                k_uwb = (k - 1) / uwb_downsample_factor + 1;
                % --- Step C.1: 独立边缘化，准备待广播的先验 3D 位置及信息组件 ---
                % DMLKF (SCI) 广播变量

                p_est_shared = cell(Vehicle_num, 1);
                I_pos_indep_shared = cell(Vehicle_num, 1);
                I_pos_dep_shared = cell(Vehicle_num, 1);

                % DMLKF_V1 广播变量

                p_est_shared_v1 = cell(Vehicle_num, 1);
                I_pos_indep_shared_v1 = cell(Vehicle_num, 1);
                I_pos_dep_shared_v1 = cell(Vehicle_num, 1);

                % DMLKF_V2 (CI) 广播变量

                p_est_shared_v2 = cell(Vehicle_num, 1);
                Sigma_pos_shared_v2 = cell(Vehicle_num, 1);

                % DMLKF_V3 (CI + No Joint) 广播变量

                p_est_shared_v3 = cell(Vehicle_num, 1);
                Sigma_pos_shared_v3 = cell(Vehicle_num, 1);

                % DEKF 广播变量

                p_est_shared_dekf = cell(Vehicle_num, 1);
                Sigma_pos_shared_dekf = cell(Vehicle_num, 1);

                % DIEKF 广播变量

                p_est_shared_diekf = cell(Vehicle_num, 1);
                Sigma_pos_shared_diekf = cell(Vehicle_num, 1);


                % 只有用SCI的需要真的边缘化，其他算法仅是用于为邻车提供位置和协方差
                for n = 1 : Vehicle_num
                    % DMLKF 边缘化

                    [p_est_shared{n}, I_pos_indep_shared{n}, I_pos_dep_shared{n}] = ...
                        filters_dmlkf{n}.get_marginalized_position_info();

                    % DMLKF_V1 边缘化

                    [p_est_shared_v1{n}, I_pos_indep_shared_v1{n}, I_pos_dep_shared_v1{n}] = ...
                        filters_v1{n}.get_marginalized_position_info();

                    % DMLKF_V2 边缘化

                    [p_est_shared_v2{n}, Sigma_pos_shared_v2{n}] = ...
                        filters_v2{n}.get_marginalized_position_info();

                    % DMLKF_V3 边缘化

                    [p_est_shared_v3{n}, Sigma_pos_shared_v3{n}] = ...
                        filters_v3{n}.get_marginalized_position_info();

                    % DEKF 边缘化

                    [p_est_shared_dekf{n}, Sigma_pos_shared_dekf{n}] = ...
                        filters_dekf{n}.get_marginalized_position_info();

                    % DIEKF 边缘化

                    [p_est_shared_diekf{n}, Sigma_pos_shared_diekf{n}] = ...
                        filters_diekf{n}.get_marginalized_position_info();

                end

                % =================================================================
                %   分支1: 原始 DMLKF (含联合邻车ADMM优化 + SCI融合)
                % =================================================================

                for n = 1 : Vehicle_num
                    filters_dmlkf{n} = filters_dmlkf{n}.reset_dual_variables();
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
                        s_admm_new{n} = filters_dmlkf{n}.solve_primal_public(s_admm_all{n}, ...
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
                        filters_dmlkf{n} = filters_dmlkf{n}.update_dual(s_admm_all{n}, ...
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
                    filters_dmlkf{n} = filters_dmlkf{n}.apply_uwb_update(s_admm_all{n}, ...
                        anchor_ranges_raw, anchor_positions_veh, ...
                        active_neighbors, neigh_positions, ...
                        neigh_I_indep, neigh_I_dep, ...
                        relative_ranges, sigma_s, sigma_z);
                end


                % =================================================================
                %   分支2: DMLKF_V1 (无联合状态/无迭代，测距噪声膨胀单步GN)
                % =================================================================

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


                % =================================================================
                %   分支3: DMLKF_V2 (含联合邻车ADMM优化 + 纯CI融合)
                % =================================================================

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


                % =================================================================
                %   分支4: DMLKF_V3 (纯CI融合 + 无联合状态/无迭代，测距噪声膨胀单步GN)
                % =================================================================

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

                % =================================================================
                %   分支5: DEKF (经典 9D-CI 批量更新)
                % =================================================================

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


                % =================================================================
                %   分支6: DIEKF (经典 9D-CI 批量流形迭代更新)
                % =================================================================

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

            % 记录位置
            for n = 1 : Vehicle_num
                pos_est_dekf{n}(k, :) = filters_dekf{n}.state.p';
                pos_est_diekf{n}(k, :) = filters_diekf{n}.state.p';
                pos_est_v3{n}(k, :) = filters_v3{n}.state.p';
                pos_est_v2{n}(k, :) = filters_v2{n}.state.p';
                pos_est_v1{n}(k, :) = filters_v1{n}.state.p';
                pos_est_dmlkf{n}(k, :) = filters_dmlkf{n}.state.p';
            end
        end

        % 计算 RMSE（调用原有 calculate_position_errors）
        pos_true = cell(Vehicle_num, 1);
        for n = 1 : Vehicle_num
            pos_true{n} = [trajectories.(sprintf('V%d', n)).X_true, ...
                trajectories.(sprintf('V%d', n)).Y_true, ...
                trajectories.(sprintf('V%d', n)).Z_true];
        end

        [~, rmse_dekf] = calculate_position_errors(pos_est_dekf, pos_true);
        [~, rmse_diekf] = calculate_position_errors(pos_est_diekf, pos_true);
        [~, rmse_v3] = calculate_position_errors(pos_est_v3, pos_true);
        [~, rmse_v2] = calculate_position_errors(pos_est_v2, pos_true);
        [~, rmse_v1] = calculate_position_errors(pos_est_v1, pos_true);
        [~, rmse_dmlkf] = calculate_position_errors(pos_est_dmlkf, pos_true);

        % 计算平均值
        rmse_all_dekf(idx_veh) = mean([rmse_dekf(:).euc_rmse]);
        rmse_all_diekf(idx_veh) = mean([rmse_diekf(:).euc_rmse]);
        rmse_all_v3(idx_veh) = mean([rmse_v3(:).euc_rmse]);
        rmse_all_v2(idx_veh) = mean([rmse_v2(:).euc_rmse]);
        rmse_all_v1(idx_veh) = mean([rmse_v1(:).euc_rmse]);
        rmse_all_dmlkf(idx_veh) = mean([rmse_dmlkf(:).euc_rmse]);

        fprintf('车辆数 %d 完成: DEKF=%.4f, DIEKF=%.4f, V3=%.4f, V2=%.4f, V1=%.4f, DMLKF=%.4f\n', ...
            veh_num, rmse_all_dekf(idx_veh), rmse_all_diekf(idx_veh), rmse_all_v3(idx_veh), ...
            rmse_all_v2(idx_veh), rmse_all_v1(idx_veh), rmse_all_dmlkf(idx_veh));
    end
end

%% 3. 结果表格输出（按指定顺序）
fprintf('\n======================= 多车规模分布式算法性能对比表 (基准: DEKF) =======================\n');
fprintf('%-8s | %-12s | %-20s | %-20s | %-20s | %-20s | %-20s\n', ...
    '车辆数', 'DEKF', 'DIEKF(+%%)', 'V3(+%%)', 'V2(+%%)', 'V1(+%%)', 'DMLKF(+%%)');
fprintf('------------------------------------------------------------------------------------\n');

for i = 1 : N_veh_tests
    v = Veh_list(i);
    d = rmse_all_dekf(i);
    di = rmse_all_diekf(i); p_di = (d - di) / d * 100;
    v3 = rmse_all_v3(i); p_v3 = (d - v3) / d * 100;
    v2 = rmse_all_v2(i); p_v2 = (d - v2) / d * 100;
    v1 = rmse_all_v1(i); p_v1 = (d - v1) / d * 100;
    dm = rmse_all_dmlkf(i); p_dm = (d - dm) / d * 100;

    fprintf('%-8d | %-12.4f | %-20s | %-20s | %-20s | %-20s | %-20s\n', v, d, ...
        sprintf('%.4f (%+.2f%%)', di, p_di), ...
        sprintf('%.4f (%+.2f%%)', v3, p_v3), ...
        sprintf('%.4f (%+.2f%%)', v2, p_v2), ...
        sprintf('%.4f (%+.2f%%)', v1, p_v1), ...
        sprintf('%.4f (%+.2f%%)', dm, p_dm));
end
fprintf('====================================================================================\n');

%% 4. 双子图可视化
if exist('rmse_all_dekf', 'var')
    figure('Name', 'Multi-Vehicle Distributed Algorithms Comparison', 'Position', [100 100 1400 600]);

    pct_diekf = (rmse_all_dekf - rmse_all_diekf) ./ rmse_all_dekf * 100;
    pct_v3 = (rmse_all_dekf - rmse_all_v3) ./ rmse_all_dekf * 100;
    pct_v2 = (rmse_all_dekf - rmse_all_v2) ./ rmse_all_dekf * 100;
    pct_v1 = (rmse_all_dekf - rmse_all_v1) ./ rmse_all_dekf * 100;
    pct_dmlkf = (rmse_all_dekf - rmse_all_dmlkf) ./ rmse_all_dekf * 100;

    subplot(1, 2, 1);
    hold on; grid on;
    plot(Veh_list, rmse_all_dekf, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'DisplayName', 'DEKF');
    plot(Veh_list, rmse_all_diekf, 'c-^', 'LineWidth', 2, 'MarkerFaceColor', 'c', 'DisplayName', 'DIEKF');
    plot(Veh_list, rmse_all_v3, 'g-s', 'LineWidth', 2, 'MarkerFaceColor', 'g', 'DisplayName', 'DMLKF\_V3');
    plot(Veh_list, rmse_all_v2, 'm-d', 'LineWidth', 2, 'MarkerFaceColor', 'm', 'DisplayName', 'DMLKF\_V2');
    plot(Veh_list, rmse_all_v1, 'r-^', 'LineWidth', 2, 'MarkerFaceColor', 'r', 'DisplayName', 'DMLKF\_V1');
    plot(Veh_list, rmse_all_dmlkf, 'k-p', 'LineWidth', 2.5, 'MarkerFaceColor', 'k', 'DisplayName', 'DMLKF');
    xlabel('Vehicle Number'); ylabel('Mean Euclidean RMSE (m)');
    title('Position Error Comparison'); legend('Location', 'northwest');

    subplot(1, 2, 2);
    hold on; grid on;
    plot(Veh_list, pct_diekf, 'c-^', 'LineWidth', 2, 'MarkerFaceColor', 'c');
    plot(Veh_list, pct_v3, 'g-s', 'LineWidth', 2, 'MarkerFaceColor', 'g');
    plot(Veh_list, pct_v2, 'm-d', 'LineWidth', 2, 'MarkerFaceColor', 'm');
    plot(Veh_list, pct_v1, 'r-^', 'LineWidth', 2, 'MarkerFaceColor', 'r');
    plot(Veh_list, pct_dmlkf, 'k-p', 'LineWidth', 2.5, 'MarkerFaceColor', 'k');
    xlabel('Vehicle Number'); ylabel('Improvement over DEKF (%)');
    title('Accuracy Improvement'); legend({'DIEKF', 'V3', 'V2', 'V1', 'DMLKF'}, 'Location', 'southeast');

    sgtitle(sprintf('Distributed Multi-Vehicle Localization Performance (6 Anchors, IMU Update %.1f Hz)', 100/imu_update_factor));
end

%% 5. 保存
if ~jump_to_plot
    if ~exist(save_dir, 'dir'), mkdir(save_dir); end
    save(save_path, 'Veh_list', 'rmse_all_dekf', 'rmse_all_diekf', ...
        'rmse_all_v3', 'rmse_all_v2', 'rmse_all_v1', 'rmse_all_dmlkf');
    fprintf('结果已保存至：%s\n', save_path);
end

end
fprintf('评测完成！\n');
