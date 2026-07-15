% =========================================================================
% RBPF_Test.m (DMLKF、DEKF与RBPF多车协同定位对比评测脚本)
% 测评维度：
%   1. 车辆规模与基站数多级交叉遍历 (Veh_list, Anc_list)
%   2. 高频IMU退化博弈：100Hz -> 50Hz -> 20Hz -> 10Hz (仅对卡尔曼类算法降频)
%   3. 分频自适应权重同步退化：0.9 -> 0.8 -> 0.7 -> 0.6
% 特色机制：增量式全局缓存检索 + 自适应数据集命名 + 自动图表绘制与保存
% =========================================================================
clc; clear; close all;

%% 1. 环境路径加载与保存路径设定
addpath(genpath('../Common'));
addpath(genpath('../Filter'));
addpath(genpath('../Data'));

save_dir = 'E:\SE3_MLKF\Result\RBPF';       % 统一数据保存目录
save_dir_fig = 'E:\SE3_MLKF\Result\Figure'; % 统一图片保存目录

%% 2. 评测自维参数组配置
Veh_list = 6;     % 评测的车辆规模列表
Anc_list = 4:20;  % 评测的基站数量列表（支持 5:20, 9:11, 或单基站例如 9）

max_admm_iter = 2;
SCI_rho = 0.8;

imu_update_factors = [10]; % IMU更新退化因子 (仅影响DMLKF与DEKF)
SCI_Weight = [0.9]; % 分频公用先验权重系数
dt_imu = 0.01; % 100Hz 物理基础采样步长
uwb_downsample_factor = 10; % 10Hz UWB协同更新

% 算法运行控制开关 (1-开启, 0-关闭)
run_dmlkf = 1;    % 运行 DMLKF 算法 (ADMM 分裂协方差版)
run_dekf = 1;     % 运行 DEKF 算法 (全维 CI EKF 基准对照)
run_rbpf = 1;     % 运行 RBPF 算法

run_compare = 0;  % 强制运行算法仿真开关 (1-覆盖全部缓存重新运行, 0-读取缓存进行增量计算)

RBPF_num = 3;     % RBPF 重复运行次数（用于取均值减小随机性误差）
Particle_num = 200; % RBPF 的粒子参数（传给 RBPF 构造函数）

% DMLKF/DEKF 15维过程噪声设置 (同全局配置一致)
Q_sigmas_15d = [ ...
    0.0001 * ones(1, 3), ... % 位置过程噪声标准差
    0.001 * ones(1, 3), ...  % 速度过程噪声标准差
    0.00025 * ones(1, 3), ... % 加速度过程噪声标准差
    0.0001 * ones(1, 3), ...  % 姿态(旋转)过程噪声标准差
    0.00025 * ones(1, 3)  ... % 角速度过程噪声标准差
    ];
Q_15d = diag(Q_sigmas_15d .^ 2);

init_cov = diag([ ...
                0.01 * ones(1, 3), ... % 位置
                0.01 * ones(1, 3), ... % 速度
                0.005 * ones(1, 3), ... % 加速度
                (1 * pi / 180) * ones(1, 3), ... % 姿态
                0.005 * ones(1, 3)  ... % 角速度
                ]);

data_ratio = 1; % 仿真使用的数据比例

%% 3. 基站-车辆增量检测与仿真主循环
for v_idx = 1 : length(Veh_list)
    veh_num = Veh_list(v_idx);

    % 初始化内存中的统一数据存储结构
    unified_data = struct();
    unified_data.results_dmlkf_all = cell(100, 1);
    unified_data.results_dekf_all = cell(100, 1);
    unified_data.results_rbpf_all = cell(100, 1);
    unified_data.avg_rmse_dmlkf = cell(100, 1);
    unified_data.avg_rmse_dekf = cell(100, 1);
    unified_data.avg_rmse_rbpf = cell(100, 1);

    % 判断当前 Anc_list 是否为列表（若仅为单个基站，则不保存文件，也不作图）
    is_list = length(Anc_list) > 1;
    if is_list
        save_file = fullfile(save_dir, sprintf('RBPF_Compare_Veh%d_Anc%d_to_%d.mat', veh_num, Anc_list(1), Anc_list(end)));
        if exist(save_file, 'file')
            unified_data = load(save_file);
        end
    end

    % 自动识别在当前请求的 Anc_list 中，有哪些基站数据缺失
    missing_anc_list = [];
    
    if ~is_list
        % 若输入基站为单个，直接标记为仿真运行项，不检查、不读取
        missing_anc_list = Anc_list;
    else
        for a_idx = 1 : length(Anc_list)
            anc_num = Anc_list(a_idx);
            
            % 检查当前加载的数据集中是否已经存在该基站的数据
            if run_compare == 0 && anc_num <= length(unified_data.avg_rmse_dekf) && ~isempty(unified_data.avg_rmse_dekf{anc_num})
                continue; % 数据已存在，跳过
            else
                % 发现缺失，列入待仿真列表
                missing_anc_list = [missing_anc_list, anc_num];
            end
        end
    end

    % --- 仿真执行分支：若有缺失基站数据，在线计算 ---
    if ~isempty(missing_anc_list)
        fprintf('\n>>> [检测到数据缺失] 启动仿真计算，当前缺失且需仿真的基站: ');
        disp(missing_anc_list);

        for a_idx = 1 : length(missing_anc_list)
            anc_num = missing_anc_list(a_idx);
            
            % 加载对应基站仿真源文件
            data_file = sprintf('E:\\SE3_MLKF\\Data\\Low\\Trj_data_Veh%d_Anc%d_3D.mat', veh_num, anc_num);
            if ~exist(data_file, 'file')
                data_file = sprintf('../Data/Trj_data_Veh%d_Anc%d_3D.mat', veh_num, anc_num);
                if ~exist(data_file, 'file')
                    error('未检测到对应的仿真数据源文件：%s，请确认！', data_file);
                end
            end
            load(data_file); % 载入 trajectories, anchors, IMU_noise_params, UWB_noise_params, Vehicle_num, Anchor_num

            % ============ DMLKF/DEKF 专属参数实例化 ============
            Sigma_a_dmlkf = diag(IMU_noise_params.sigma_na .^ 2 * ones(1, 3));
            Sigma_w_dmlkf = diag(IMU_noise_params.sigma_nw .^ 2 * ones(1, 3));

            results_dmlkf_all = zeros(length(imu_update_factors), veh_num);
            results_dekf_all = zeros(length(imu_update_factors), veh_num);
            results_rbpf_all = zeros(length(imu_update_factors), veh_num);
            
            avg_rmse_dmlkf = zeros(length(imu_update_factors), 1);
            avg_rmse_dekf = zeros(length(imu_update_factors), 1);
            avg_rmse_rbpf = zeros(length(imu_update_factors), 1);

            % 变频与权重退化循环
            for f_idx = 1 : length(imu_update_factors)
                imu_update_factor = imu_update_factors(f_idx);
                weight = SCI_Weight(f_idx);
                imu_freq = round(100 / imu_update_factor);

                fprintf('    正在计算 -> 车数: %2d | 基站数: %d | IMU更新: %3d Hz | 权重 w = %.2f...\n', ...
                    veh_num, anc_num, imu_freq, weight);

                % A. 惯性真值重建与全局真值位置提取
                pos_true = cell(Vehicle_num, 1);
                for n = 1 : Vehicle_num
                    v_name = sprintf('V%d', n);
                    veh = trajectories.(v_name);
                    N_steps = floor(length(veh.Time_true) * data_ratio); 

                    % 加速度重建
                    v_true_matrix = [veh.Vx_true(1:N_steps), ...
                                     veh.Vy_true(1:N_steps), ...
                                     veh.Vz_true(1:N_steps)];
                    a_true_matrix = zeros(N_steps, 3);
                    a_true_matrix(:, 1) = gradient(v_true_matrix(:, 1), dt_imu);
                    a_true_matrix(:, 2) = gradient(v_true_matrix(:, 2), dt_imu);
                    a_true_matrix(:, 3) = gradient(v_true_matrix(:, 3), dt_imu);
                    trajectories.(v_name).a_true = a_true_matrix;

                    % 角速度重建
                    theta_unwrapped = unwrap(veh.Theta_true(1:N_steps));
                    wz_true = gradient(theta_unwrapped, dt_imu);
                    omega_true_matrix = [zeros(N_steps, 2), wz_true];
                    trajectories.(v_name).omega_true = omega_true_matrix;

                    % 提取真实位置用于精度结算
                    pos_true{n} = [trajectories.(v_name).X_true(1:N_steps), ...
                                   trajectories.(v_name).Y_true(1:N_steps), ...
                                   trajectories.(v_name).Z_true(1:N_steps)];
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

                % ==================== 运行 DMLKF 与 DEKF ====================
                if run_dmlkf || run_dekf
                    filters_dmlkf = cell(Vehicle_num, 1);
                    filters_dekf = cell(Vehicle_num, 1);
                    pos_est_dmlkf = cell(Vehicle_num, 1);
                    pos_est_dekf = cell(Vehicle_num, 1);

                    for n = 1 : Vehicle_num
                        v_name = sprintf('V%d', n);
                        veh = trajectories.(v_name);

                        init_state = struct();
                        init_state.p = [veh.X_true(1); veh.Y_true(1); veh.Z_true(1)];
                        init_state.v = [veh.Vx_true(1); veh.Vy_true(1); veh.Vz_true(1)];
                        init_state.a = veh.a_true(1, :)';

                        th0 = veh.Theta_true(1);
                        init_state.R = [cos(th0), -sin(th0), 0;
                            sin(th0), cos(th0), 0;
                            0, 0, 1];
                        init_state.omega = veh.omega_true(1, :)';

                        if run_dmlkf
                            filters_dmlkf{n} = DMLKF(n, init_state, init_cov, Q_15d, Sigma_a_dmlkf, Sigma_w_dmlkf, dt_imu, weight);
                            pos_est_dmlkf{n} = zeros(N_steps, 3);
                        end
                        if run_dekf
                            % DEKF 的 CI 参数固定传入 0.9
                            filters_dekf{n} = DEKF(n, init_state, init_cov, Q_15d, Sigma_a_dmlkf, Sigma_w_dmlkf, dt_imu, 0.9);
                            pos_est_dekf{n} = zeros(N_steps, 3);
                        end
                    end

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
                            if run_dmlkf
                                filters_dmlkf{n} = filters_dmlkf{n}.predict();
                                if mod(k - 1, imu_update_factor) == 0
                                    filters_dmlkf{n} = filters_dmlkf{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
                                end
                            end
                            if run_dekf
                                filters_dekf{n} = filters_dekf{n}.predict();
                                if mod(k - 1, imu_update_factor) == 0
                                    filters_dekf{n} = filters_dekf{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3, 1), zeros(3, 1));
                                end
                            end
                        end

                        if mod(k - 1, uwb_downsample_factor) == 0
                            k_uwb = (k - 1) / uwb_downsample_factor + 1;

                            p_est_shared_dmlkf = cell(Vehicle_num, 1);
                            I_pos_indep_shared_dmlkf = cell(Vehicle_num, 1);
                            I_pos_dep_shared_dmlkf = cell(Vehicle_num, 1);
                            p_est_shared_dekf = cell(Vehicle_num, 1);
                            Sigma_pos_shared_dekf = cell(Vehicle_num, 1);

                            for n = 1 : Vehicle_num
                                if run_dmlkf
                                    [p_est_shared_dmlkf{n}, I_pos_indep_shared_dmlkf{n}, I_pos_dep_shared_dmlkf{n}] = ...
                                        filters_dmlkf{n}.get_marginalized_position_info();
                                end
                                if run_dekf
                                    [p_est_shared_dekf{n}, Sigma_pos_shared_dekf{n}] = ...
                                        filters_dekf{n}.get_marginalized_position_info();
                                end
                            end

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

                            if run_dekf
                                for n = 1 : Vehicle_num
                                    v_name = sprintf('V%d', n); veh = trajectories.(v_name);
                                    anchor_ranges_raw = veh.UWB_Anchor(k_uwb, 2 : end)';
                                    anchor_positions_veh = anchors(1 : Anchor_num, :);
                                    active_neighbors = neighbors_map{n};
                                    M_neighbors = length(active_neighbors);
                                    relative_ranges = zeros(M_neighbors, 1);
                                    neigh_positions_dekf = zeros(M_neighbors, 3);
                                    neigh_Sigma_pos_dekf = cell(M_neighbors, 1);

                                    for idx = 1 : M_neighbors
                                        nid = active_neighbors(idx);
                                        neigh_positions_dekf(idx, :) = p_est_shared_dekf{nid}';
                                        neigh_Sigma_pos_dekf{idx} = Sigma_pos_shared_dekf{nid};
                                        rel_val = veh.UWB_Relative(k_uwb, 1 + nid);
                                        if isnan(rel_val) || isinf(rel_val)
                                            rel_val = norm(p_est_shared_dekf{n} - p_est_shared_dekf{nid});
                                        end
                                        relative_ranges(idx) = rel_val;
                                    end
                                    for a_idx_sub = 1 : length(anchor_ranges_raw)
                                        if isnan(anchor_ranges_raw(a_idx_sub)) || isinf(anchor_ranges_raw(a_idx_sub))
                                            anchor_ranges_raw(a_idx_sub) = norm(p_est_shared_dekf{n} - anchor_positions_veh(a_idx_sub, :)');
                                        end
                                    end
                                    sigma_s = UWB_noise_params.sigma_anc;
                                    sigma_z = UWB_noise_params.sigma_rel;

                                    filters_dekf{n} = filters_dekf{n}.apply_uwb_update(...
                                        [], ... 
                                        anchor_ranges_raw, anchor_positions_veh, ...
                                        active_neighbors, neigh_positions_dekf, ...
                                        neigh_Sigma_pos_dekf, relative_ranges, sigma_s, sigma_z);
                                end
                            end
                        end

                        for n = 1 : Vehicle_num
                            if run_dmlkf
                                pos_est_dmlkf{n}(k, :) = filters_dmlkf{n}.state.p';
                            end
                            if run_dekf
                                pos_est_dekf{n}(k, :) = filters_dekf{n}.state.p';
                            end
                        end
                    end 

                    if run_dmlkf
                        [~, rmse_dmlkf] = calculate_position_errors(pos_est_dmlkf, pos_true);
                        results_dmlkf_all(f_idx, :) = [rmse_dmlkf.euc_rmse];
                        avg_rmse_dmlkf(f_idx) = mean([rmse_dmlkf.euc_rmse]);
                    end
                    if run_dekf
                        [~, rmse_dekf] = calculate_position_errors(pos_est_dekf, pos_true);
                        results_dekf_all(f_idx, :) = [rmse_dekf.euc_rmse];
                        avg_rmse_dekf(f_idx) = mean([rmse_dekf.euc_rmse]);
                    end
                end

                % ==================== 运行 RBPF (多次独立运行取 RMSE 均值) ====================
                if run_rbpf
                    accum_rmse_rbpf = zeros(1, Vehicle_num);

                    for r_idx = 1 : RBPF_num
                        fprintf('        -> [RBPF 进度] 正在运行第 %d/%d 次独立随机仿真...\n', r_idx, RBPF_num);
                        
                        filters_rbpf = cell(Vehicle_num, 1);
                        pos_est_rbpf = cell(Vehicle_num, 1);

                        for n = 1 : Vehicle_num
                            v_name = sprintf('V%d', n);
                            veh = trajectories.(v_name);

                            init_state = struct();
                            init_state.p = [veh.X_true(1); veh.Y_true(1); veh.Z_true(1)];
                            init_state.v = [veh.Vx_true(1); veh.Vy_true(1); veh.Vz_true(1)];
                            init_state.a = veh.a_true(1, :)';

                            th0 = veh.Theta_true(1);
                            init_state.R = [cos(th0), -sin(th0), 0;
                                sin(th0), cos(th0), 0;
                                0, 0, 1];
                            init_state.omega = veh.omega_true(1, :)';

                            % 传入 Particle_num = 200 作为粒子参数
                            filters_rbpf{n} = RBPF(n, init_state, dt_imu, Particle_num);
                            pos_est_rbpf{n} = zeros(N_steps, 3);
                        end

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
                                filters_rbpf{n} = filters_rbpf{n}.predict(imu_acc(n, :)', imu_gyro(n, :)');
                            end

                            if mod(k - 1, uwb_downsample_factor) == 0
                                k_uwb = (k - 1) / uwb_downsample_factor + 1;

                                p_est_shared_rbpf = cell(Vehicle_num, 1);
                                Sigma_pos_shared_rbpf = cell(Vehicle_num, 1);

                                for n = 1 : Vehicle_num
                                    [p_est_shared_rbpf{n}, Sigma_pos_shared_rbpf{n}] = ...
                                        filters_rbpf{n}.get_marginalized_position_info();
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

                                    filters_rbpf{n} = filters_rbpf{n}.apply_uwb_update(...
                                        anchor_ranges_raw, anchor_positions_veh, sigma_s, ...
                                        active_neighbors, neigh_positions, ...
                                        neigh_Sigma_pos, relative_ranges, sigma_z, 1.3);
                                end
                            end   

                            for n = 1 : Vehicle_num
                                pos_est_rbpf{n}(k, :) = filters_rbpf{n}.state.p';
                            end
                        end 

                        [~, rmse_rbpf_run] = calculate_position_errors(pos_est_rbpf, pos_true);
                        accum_rmse_rbpf = accum_rmse_rbpf + [rmse_rbpf_run.euc_rmse];
                    end

                    mean_rmse_rbpf = accum_rmse_rbpf / RBPF_num;
                    results_rbpf_all(f_idx, :) = mean_rmse_rbpf;
                    avg_rmse_rbpf(f_idx) = mean(mean_rmse_rbpf);
                end

            end % 变频循环闭合

            % --- 将计算出的基站数据更新到内存的 unified_data 中 ---
            unified_data.results_dmlkf_all{anc_num} = results_dmlkf_all;
            unified_data.results_dekf_all{anc_num}  = results_dekf_all;
            unified_data.results_rbpf_all{anc_num}  = results_rbpf_all;
            unified_data.avg_rmse_dmlkf{anc_num}    = avg_rmse_dmlkf;
            unified_data.avg_rmse_dekf{anc_num}     = avg_rmse_dekf;
            unified_data.avg_rmse_rbpf{anc_num}     = avg_rmse_rbpf;

            % --- 多基站列表保存逻辑：仅在 Anc_list 为列表且核心算法均开启时存盘 ---
            if is_list && run_dekf && run_dmlkf && run_rbpf
                if ~exist(save_dir, 'dir')
                    mkdir(save_dir);
                end
                save(save_file, '-struct', 'unified_data');
                fprintf('    -> [增量存盘成功] 当前基站数 %d 下的结果已追加写入：%s\n', anc_num, save_file);
            end

        end
    else
        fprintf('\n=========================================================================\n');
        fprintf('  [检测到完整缓存] 所需基站 [%d 至 %d] 评测数据均已从缓存中加载完毕！\n', Anc_list(1), Anc_list(end));
        fprintf('=========================================================================\n');
    end

    % --- 最终展现 ---
    if is_list
        % 如果是列表，打印完整数据集并画图保存
        print_and_plot_results(unified_data, Anc_list, veh_num, save_dir_fig);
    else
        % 如果是单个基站，只进行控制台打印，不进行作图与保存
        anc_num = Anc_list(1);
        dekf_avg = mean(unified_data.avg_rmse_dekf{anc_num});
        dmlkf_avg = mean(unified_data.avg_rmse_dmlkf{anc_num});
        rbpf_avg = mean(unified_data.avg_rmse_rbpf{anc_num});
        
        dmlkf_imp = (dekf_avg - dmlkf_avg) / dekf_avg * 100;
        rbpf_imp = (dekf_avg - rbpf_avg) / dekf_avg * 100;
        
        fprintf('\n======================== 单基站评估结果 ========================\n');
        fprintf('基站数: %d\n', anc_num);
        fprintf('DEKF  (基准): %.4f m\n', dekf_avg);
        fprintf('DMLKF       : %.4f m (%.2f%%)\n', dmlkf_avg, dmlkf_imp);
        fprintf('RBPF        : %.4f m (%.2f%%)\n', rbpf_avg, rbpf_imp);
        fprintf('===============================================================\n');
    end

end 

%% =========================================================================
%  内部局部辅助函数区域 (必须置于脚本文件的最下方)
% =========================================================================

function print_and_plot_results(unified_data, Anc_list, veh_num, save_dir_fig)
    N_anc = length(Anc_list);
    RowNames = cell(N_anc, 1);
    
    DEKF_Col = zeros(N_anc, 1);
    DMLKF_Col = cell(N_anc, 1);
    RBPF_Col = cell(N_anc, 1);
    
    rmse_dekf_vec = zeros(N_anc, 1);
    rmse_dmlkf_vec = zeros(N_anc, 1);
    rmse_rbpf_vec = zeros(N_anc, 1);
    
    imp_dmlkf_vec = zeros(N_anc, 1);
    imp_rbpf_vec = zeros(N_anc, 1);
    
    for i = 1 : N_anc
        anc_num = Anc_list(i);
        RowNames{i} = sprintf('Anc_%d', anc_num);
        
        % 提取平均 RMSE（计算全部因子的均值）
        rmse_dekf_vec(i) = mean(unified_data.avg_rmse_dekf{anc_num});
        rmse_dmlkf_vec(i) = mean(unified_data.avg_rmse_dmlkf{anc_num});
        rmse_rbpf_vec(i) = mean(unified_data.avg_rmse_rbpf{anc_num});
        
        % 计算提升比例
        imp_dmlkf_vec(i) = (rmse_dekf_vec(i) - rmse_dmlkf_vec(i)) / rmse_dekf_vec(i) * 100;
        imp_rbpf_vec(i)  = (rmse_dekf_vec(i) - rmse_rbpf_vec(i)) / rmse_dekf_vec(i) * 100;
        
        % 格式化表格显示项
        DEKF_Col(i) = rmse_dekf_vec(i);
        DMLKF_Col{i} = sprintf('%.4f (%.2f%%)', rmse_dmlkf_vec(i), imp_dmlkf_vec(i));
        RBPF_Col{i}  = sprintf('%.4f (%.2f%%)', rmse_rbpf_vec(i), imp_rbpf_vec(i));
    end
    
    % 组装表格
    Summary_Table = table(DEKF_Col, DMLKF_Col, RBPF_Col, ...
        'RowNames', RowNames, ...
        'VariableNames', {'DEKF_RMSE', 'DMLKF_RMSE_Imp', 'RBPF_RMSE_Imp'});
        
    fprintf('\n======================== 全局平均定位精度与提升对比表 ========================\n');
    disp(Summary_Table);
    fprintf('-------------------------------------------------------------------------\n');
    
    % --- 绘制一图两子图 ---
    figure('Name', sprintf('Veh%d Multi-Anchor Comparison', veh_num), ...
           'Color', 'w', 'Position', [150, 150, 1050, 450]);
    
    % 子图 1：RMSE 数值变化趋势
    subplot(1, 2, 1);
    plot(Anc_list, rmse_dekf_vec, 'o-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'DEKF (Baseline)');
    hold on;
    plot(Anc_list, rmse_dmlkf_vec, 's-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'DMLKF');
    plot(Anc_list, rmse_rbpf_vec, '^-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'RBPF');
    grid on;
    xlabel('Anchors number', 'FontSize', 10);
    ylabel('Average RMSE (m)', 'FontSize', 10);
    title(sprintf('RMSE (Vehicle: %d)', veh_num), 'FontSize', 11);
    legend('Location', 'northeast');
    set(gca, 'XTick', Anc_list);
    
    % 子图 2：提升百分比趋势显示
    subplot(1, 2, 2);
    plot(Anc_list, imp_dmlkf_vec, 's--', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'DMLKF vs DEKF');
    hold on;
    plot(Anc_list, imp_rbpf_vec, '^--', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'RBPF vs DEKF');
    grid on;
    xlabel('Anchors number', 'FontSize', 10);
    ylabel('Improvement (%)', 'FontSize', 10);
    title('Relative to the improvement in accuracy of DEKF', 'FontSize', 11);
    legend('Location', 'northeast');
    set(gca, 'XTick', Anc_list);
    
    % --- 图片文件保存逻辑 ---
    if ~exist(save_dir_fig, 'dir')
        mkdir(save_dir_fig);
    end
    if length(Anc_list) > 1
        fig_name = sprintf('RBPF_Compare_Anc_%d_%d', Anc_list(1), Anc_list(end));
    else
        fig_name = sprintf('RBPF_Compare_Anc_%d', Anc_list(1));
    end

    % 显式指定 'png' 作为格式参数，规避底层 feval 中无法识别 'saveaspng' 的 dispatch 错误
    try
        % R2020a 及以上版本推荐的高质量导出接口（紧凑裁剪边缘）
        exportgraphics(gcf, fullfile(save_dir_fig, [fig_name, '.png']), 'Resolution', 300);
    catch
        % 兼容老版本的经典 print 底层输出接口（300 DPI 高清分辨率）
        print(gcf, fullfile(save_dir_fig, [fig_name, '.png']), '-dpng', '-r300');
    end
    fprintf('一图两子图分析图像已成功保存至：%s\n', fullfile(save_dir_fig, [fig_name, '.png']));
end