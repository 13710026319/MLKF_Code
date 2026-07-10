% =========================================================================
% RBPF_Test.m (流形RBPF与DMLKF多车协同定位对比评测脚本)
% 测评维度：
%   1. 车辆规模与基站数多级交叉遍历 (Veh_list, Anc_list)
%   2. 高频IMU退化博弈：100Hz -> 50Hz -> 20Hz -> 10Hz
%   3. 分频自适应权重同步退退：0.9 -> 0.8 -> 0.7 -> 0.6
% 特色机制：分级实时存盘机制 + 历史结果旁路秒级载入机制
% =========================================================================
clc; clear; close all;

%% 1. 环境路径加载与保存路径设定
addpath(genpath('../Common'));
addpath(genpath('../Filter'));
addpath(genpath('../Data'));

save_dir = 'E:\SE3_MLKF\Result\RBPF';

%% 2. 评测自维参数组配置
Veh_list = 6;                         % 评测的车辆规模列表
Anc_list = [5];                            % 评测的基站数量列表

max_admm_iter = 2;
SCI_rho = 0.8;

imu_update_factors = [1];         % IMU更新退化因子
SCI_Weight         = [0.9, 0.8, 0.7, 0.6];  % 分频公用先验权重系数
dt_imu             = 0.01;                  % 100Hz 物理基础采样步长
uwb_downsample_factor = 10;                 % 10Hz UWB协同更新

% 算法运行控制开关
run_dmlkf = 0;                              % 置1运行DMLKF
run_rbpf  = 1;                              % 置1运行RBPF

% 15维过程噪声设置 (同全局 DMLKF 一致)
Q_sigmas_15d = [ ...
    0.0001 * ones(1, 3), ... % 位置过程噪声标准差
    0.001 * ones(1, 3), ...  % 速度过程噪声标准差
    0.00025 * ones(1, 3), ... % 加速度过程噪声标准差
    0.0001 * ones(1, 3), ...  % 姿态(旋转)过程噪声标准差
    0.00025 * ones(1, 3)  ... % 角速度过程噪声标准差
    ];
Q_15d = diag(Q_sigmas_15d .^ 2);

%% 3. 基站-车辆多重嵌套主测试循环
for a_idx = 1 : length(Anc_list)
    anc_num = Anc_list(a_idx);
    
    for v_idx = 1 : length(Veh_list)
        veh_num = Veh_list(v_idx);
        
        % 建立分级存盘目标文件名
        save_file = fullfile(save_dir, sprintf('RBPF_Compare_Veh%d_Anc%d.mat', veh_num, anc_num));
        
        % --- 旁路判定 (Bypass Check): 若存在历史数据直接打印并略过仿真 ---
        if (exist(save_file, 'file') && run_dmlkf && run_rbpf)
            fprintf('\n=========================================================================\n');
            fprintf('  [检测到缓存] 已存在该车辆/基站配置下的评测文件：%s\n', save_file);
            fprintf('  正在直接载入并打印历史多频博弈对比表...\n');
            fprintf('=========================================================================\n');
            load(save_file);
            disp(Summary_Table);
            fprintf('-------------------------------------------------------------------------\n');
            continue; % 直接跳入下一轮遍历
        end
        
        % 若无缓存，加载数据集并启动变频仿真组
        data_file = sprintf('E:\\SE3_MLKF\\Data\\Low\\Trj_data_Veh%d_Anc%d_3D.mat', veh_num, anc_num);
        if ~exist(data_file, 'file')
            % 鲁棒容错相对路径寻找
            data_file = sprintf('../Data/Trj_data_Veh%d_Anc%d_3D.mat', veh_num, anc_num);
            if ~exist(data_file, 'file')
                error('未检测到对应的仿真数据源文件：%s，请确认后运行！', data_file);
            end
        end
        load(data_file); % 载入 trajectories, anchors, IMU_noise_params, UWB_noise_params, Vehicle_num, Anchor_num
        
        % 结果记录结构初始化
        results_dmlkf_all = zeros(length(imu_update_factors), veh_num);
        results_rbpf_all  = zeros(length(imu_update_factors), veh_num);
        avg_rmse_dmlkf    = zeros(length(imu_update_factors), 1);
        avg_rmse_rbpf     = zeros(length(imu_update_factors), 1);
        
        % 变频与权重退化循环
        for f_idx = 1 : length(imu_update_factors)
            imu_update_factor = imu_update_factors(f_idx);
            weight = SCI_Weight(f_idx);
            imu_freq = round(100 / imu_update_factor);
            
            fprintf('  [正在仿真] 车数: %2d | 基站数: %d | IMU更新: %3d Hz | 权重 w = %.2f...\n', ...
                veh_num, anc_num, imu_freq, weight);
            
            % A. 惯性真值重建
            for n = 1 : Vehicle_num
                v_name = sprintf('V%d', n);
                veh = trajectories.(v_name);
                N_steps = length(veh.Time_true);
                
                % 加速度重建
                v_true_matrix = [veh.Vx_true, veh.Vy_true, veh.Vz_true];
                a_true_matrix = zeros(N_steps, 3);
                a_true_matrix(:, 1) = gradient(v_true_matrix(:, 1), dt_imu);
                a_true_matrix(:, 2) = gradient(v_true_matrix(:, 2), dt_imu);
                a_true_matrix(:, 3) = gradient(v_true_matrix(:, 3), dt_imu);
                trajectories.(v_name).a_true = a_true_matrix;
                
                % 角速度重建
                theta_unwrapped = unwrap(veh.Theta_true);
                wz_true = gradient(theta_unwrapped, dt_imu);
                omega_true_matrix = [zeros(N_steps, 2), wz_true];
                trajectories.(v_name).omega_true = omega_true_matrix;
            end
            
            % B. 实例化对应运行周期的滤波器
            filters_dmlkf = cell(Vehicle_num, 1);
            filters_rbpf  = cell(Vehicle_num, 1);
            pos_est_dmlkf = cell(Vehicle_num, 1);
            pos_est_rbpf  = cell(Vehicle_num, 1);
            
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
                
                if run_dmlkf
                    filters_dmlkf{n} = DMLKF(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu, weight);
                    pos_est_dmlkf{n} = zeros(N_steps, 3);
                end
                if run_rbpf
                    % 使用默认 100 粒子实例化高鲁棒 RBPF
                    filters_rbpf{n}  = RBPF(n, init_state, init_cov, Q_15d, Sigma_a, Sigma_w, dt_imu, weight, 150);
                    pos_est_rbpf{n}  = zeros(N_steps, 3);
                end
            end
            
            % 双邻居动态拓扑
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
            
            % C. 仿真步进时间递推 (100Hz 级)
            for k = 1 : N_steps
                % 提取传感器输入
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
                
                % 变频 IMU 更新
                for n = 1 : Vehicle_num
                    if run_dmlkf
                        filters_dmlkf{n} = filters_dmlkf{n}.predict();
                        if mod(k - 1, imu_update_factor) == 0
                            filters_dmlkf{n} = filters_dmlkf{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
                        end
                    end
                    if run_rbpf
                        filters_rbpf{n} = filters_rbpf{n}.predict();
                        if mod(k - 1, imu_update_factor) == 0
                            update_weight_flag = (mod(k - 1, 20) == 0);
                            filters_rbpf{n} = filters_rbpf{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1), update_weight_flag);
                        end
                    end
                end
                
                % UWB 观测更新 (10Hz 固定采样)
                if mod(k - 1, uwb_downsample_factor) == 0
                    k_uwb = (k - 1) / uwb_downsample_factor + 1;
                    
                    p_est_shared_dmlkf = cell(Vehicle_num, 1);
                    I_pos_indep_shared_dmlkf = cell(Vehicle_num, 1);
                    I_pos_dep_shared_dmlkf = cell(Vehicle_num, 1);
                    p_est_shared_rbpf = cell(Vehicle_num, 1);
                    Sigma_pos_shared_rbpf = cell(Vehicle_num, 1);
                    
                    % 广播准备
                    for n = 1 : Vehicle_num
                        if run_dmlkf
                            [p_est_shared_dmlkf{n}, I_pos_indep_shared_dmlkf{n}, I_pos_dep_shared_dmlkf{n}] = ...
                                filters_dmlkf{n}.get_marginalized_position_info();
                        end
                        if run_rbpf
                            [p_est_shared_rbpf{n}, Sigma_pos_shared_rbpf{n}] = ...
                                filters_rbpf{n}.get_marginalized_position_info();
                        end
                    end
                    
                    % C.1 触发 DMLKF 专属的 ADMM 分布式约束更新
                    if run_dmlkf
                        s_admm_all = cell(Vehicle_num, 1);
                        for n = 1 : Vehicle_num
                            M_neigh = length(neighbors_map{n});
                            s_admm_all{n} = zeros(3 * (M_neigh + 1), 1);
                        end
                        dp_neigh_neigh_all = cell(Vehicle_num, 1);
                        dp_neigh_self_all = cell(Vehicle_num, 1);
                        for n = 1 : Vehicle_num
                            M_neigh = length(neighbors_map{n});
                            dp_neigh_neigh_all{n} = zeros(3, M_neigh);
                            dp_neigh_self_all{n} = zeros(3, M_neigh);
                        end
                        
                        % ADMM 局域共识循环
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
                                for a_idx_sub = 1 : length(anchor_ranges_raw)
                                    if isnan(anchor_ranges_raw(a_idx_sub)) || isinf(anchor_ranges_raw(a_idx_sub))
                                        anchor_ranges_raw(a_idx_sub) = norm(p_est_shared_dmlkf{n} - anchor_positions_veh(a_idx_sub, :)');
                                    end
                                end
                                for idx = 1 : M_neighbors
                                    nid = active_neighbors(idx);
                                    neigh_positions(idx, :) = p_est_shared_dmlkf{nid}';
                                    rel_val = veh.UWB_Relative(k_uwb, 1 + nid);
                                    if isnan(rel_val) || isinf(rel_val)
                                        rel_val = norm(p_est_shared_dmlkf{n} - p_est_shared_dmlkf{nid});
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
                                neigh_positions(idx, :) = p_est_shared_dmlkf{nid}';
                                neigh_I_indep{idx} = I_pos_indep_shared_dmlkf{nid};
                                neigh_I_dep{idx} = I_pos_dep_shared_dmlkf{nid};
                                rel_val = veh.UWB_Relative(k_uwb, 1 + nid);
                                if isnan(rel_val) || isinf(rel_val)
                                    rel_val = norm(p_est_shared_dmlkf{n} - p_est_shared_dmlkf{nid});
                                end
                                relative_ranges(idx) = rel_val;
                            end
                            for a_idx_sub = 1 : length(anchor_ranges_raw)
                                if isnan(anchor_ranges_raw(a_idx_sub)) || isinf(anchor_ranges_raw(a_idx_sub))
                                    anchor_ranges_raw(a_idx_sub) = norm(p_est_shared_dmlkf{n} - anchor_positions_veh(a_idx_sub, :)');
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
                    end
                    
                    % C.2 RBPF 分布式协同更新
                    if run_rbpf
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
                                neigh_positions(idx, :) = p_est_shared_rbpf{nid}';
                                neigh_Sigma_pos{idx} = Sigma_pos_shared_rbpf{nid};
                                rel_val = veh.UWB_Relative(k_uwb, 1 + nid);
                                if isnan(rel_val) || isinf(rel_val)
                                    rel_val = norm(p_est_shared_rbpf{n} - p_est_shared_rbpf{nid});
                                end
                                relative_ranges(idx) = rel_val;
                            end
                            for a_idx_sub = 1 : length(anchor_ranges_raw)
                                if isnan(anchor_ranges_raw(a_idx_sub)) || isinf(anchor_ranges_raw(a_idx_sub))
                                    anchor_ranges_raw(a_idx_sub) = norm(p_est_shared_rbpf{n} - anchor_positions_veh(a_idx_sub, :)');
                                end
                            end
                            sigma_s = UWB_noise_params.sigma_anc;
                            sigma_z = UWB_noise_params.sigma_rel;
                            
                            % RBPF 协同更新
                            filters_rbpf{n} = filters_rbpf{n}.apply_uwb_update([], ...
                                anchor_ranges_raw, anchor_positions_veh, ...
                                active_neighbors, neigh_positions, ...
                                neigh_Sigma_pos, relative_ranges, sigma_s, sigma_z);
                        end
                    end
                end
                
                % 记录定位状态
                for n = 1 : Vehicle_num
                    if run_dmlkf
                        pos_est_dmlkf{n}(k, :) = filters_dmlkf{n}.state.p';
                    end
                    if run_rbpf
                        pos_est_rbpf{n}(k, :) = filters_rbpf{n}.state.p';
                    end
                end
            end % 主循环闭合
            
            % D. 精度与误差结算
            pos_true = cell(Vehicle_num, 1);
            for n = 1 : Vehicle_num
                v_name = sprintf('V%d', n);
                pos_true{n} = [trajectories.(v_name).X_true, ...
                               trajectories.(v_name).Y_true, ...
                               trajectories.(v_name).Z_true];
            end
            if run_dmlkf
                [~, rmse_dmlkf] = calculate_position_errors(pos_est_dmlkf, pos_true);
                results_dmlkf_all(f_idx, :) = [rmse_dmlkf.euc_rmse];
                avg_rmse_dmlkf(f_idx) = mean([rmse_dmlkf.euc_rmse]);
            end
            if run_rbpf
                [~, rmse_rbpf] = calculate_position_errors(pos_est_rbpf, pos_true);
                results_rbpf_all(f_idx, :) = [rmse_rbpf.euc_rmse];
                avg_rmse_rbpf(f_idx) = mean([rmse_rbpf.euc_rmse]);
            end
        end % 变频循环闭合
        
        % --- 数据后处理：构建多维对比 Summary Table ---
        RowNames = cell(length(imu_update_factors), 1);
        for i_f = 1 : length(imu_update_factors)
            RowNames{i_f} = sprintf('%dHz (w=%.2f)', round(100 / imu_update_factors(i_f)), SCI_Weight(i_f));
        end

        if run_dmlkf && run_rbpf
            % 提升百分比 = (DMLKF_RMSE - RBPF_RMSE) / DMLKF_RMSE * 100
            improvement_percent = (avg_rmse_dmlkf - avg_rmse_rbpf) ./ avg_rmse_dmlkf * 100;
            Summary_Table = table(avg_rmse_dmlkf, avg_rmse_rbpf, improvement_percent, ...
                'RowNames', RowNames, ...
                'VariableNames', {'DMLKF_Average_RMSE', 'RBPF_Average_RMSE', 'RBPF_Improvement_Percent'});
            if ~exist(save_dir, 'dir')
            mkdir(save_dir);
            end
        save(save_file, 'results_dmlkf_all', 'results_rbpf_all', ...
                         'avg_rmse_dmlkf', 'avg_rmse_rbpf', ...
                         'Summary_Table', 'veh_num', 'anc_num', ...
                         'imu_update_factors', 'SCI_Weight');
                     
        fprintf('\n=========================================================================\n');
        fprintf('  [完成并保存] 车辆规模: %d | 基站数: %d\n', veh_num, anc_num);
        fprintf('  存盘文件：%s\n', save_file);

        elseif run_dmlkf
            Summary_Table = table(avg_rmse_dmlkf, 'RowNames', RowNames, 'VariableNames', {'DMLKF_Average_RMSE'});
        elseif run_rbpf
            Summary_Table = table(avg_rmse_rbpf, 'RowNames', RowNames, 'VariableNames', {'RBPF_Average_RMSE'});
        end
        
        % --- 级联存盘机制：当前(Veh, Anc)组合完成，立刻保存以防突发中断 ---
        
        fprintf('=========================================================================\n');
        disp(Summary_Table);
        fprintf('-------------------------------------------------------------------------\n');
        
    end % 车辆循环闭合
end % 基站循环闭合
