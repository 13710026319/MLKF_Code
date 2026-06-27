	% =========================================================================
	% DC_V_num_compare.m 
	% 分布式(DMLKF)与集中式算法在6基站、不同车辆数(4-12)下的表现对比
	% 评测维度：平均欧氏定位误差 以及相对于 EKF 的提升百分比
	% 通信网络：动态环形双邻居分布式通信网络 (1-2-...-N-1)
	% =========================================================================
	clc; clear; close all;
	%% 1. 路径与评测参数配置
	addpath(genpath('../Common'));
	addpath(genpath('../Filter'));
	addpath(genpath('../Data'));
	Veh_list = 4:12;                % 车辆数量范围：4到12
	N_veh_tests = length(Veh_list);
	% 结果保存路径与文件名配置
	save_dir = 'E:\SE3_MLKF\Result';
	save_name = 'DC_V_num_6Anc_compare.mat';
	save_path = fullfile(save_dir, save_name);
	% 智能检测历史数据
	if exist(save_path, 'file')
	    fprintf('检测到历史评测数据 [%s]，正在直接加载并生成图表...\n', save_name);
	    load(save_path); 
	    jump_to_plot = true;
	else
	    jump_to_plot = false;
	    % 预分配 4 算法误差记录数组
	    rmse_all_ekf   = zeros(N_veh_tests, 1);
	    rmse_all_iekf  = zeros(N_veh_tests, 1);
	    rmse_all_cmlkf = zeros(N_veh_tests, 1);
	    rmse_all_dmlkf = zeros(N_veh_tests, 1);
	    fprintf('未检测到历史数据，开始执行 4 算法多车数据集联合性能评测...\n');
	end
	%% 2. 核心评测循环
	if ~jump_to_plot
	    for idx_veh = 1:N_veh_tests
	        veh_num = Veh_list(idx_veh);
	        fprintf('\n>>> 当前评测数据集车辆数量: %d <<<\n', veh_num);
	        % A. 自动检测并加载数据集
	        data_file = sprintf('E:\\SE3_MLKF\\Data\\diff_V_6Anc\\Trj_data_Veh%d_Anc6d_3D.mat', veh_num);
	        if ~exist(data_file, 'file')
	            error('未检测到指定数据集，请确认数据文件是否存在于：%s', data_file);
	        end
	        load(data_file); 
	        dt_imu = 0.01; % 100Hz
	        uwb_downsample_factor = 10; % 10Hz
	        % B. 状态真值重建
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
	        % C. 初始化四大滤波器状态与参数
	        % (1) 15维 CMLKF 标称状态与协方差
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
	        P_n_init_15d = diag([ (0.01)^2*ones(1,3), (0.01)^2*ones(1,3), (0.05)^2*ones(1,3), (1*pi/180)^2*ones(1,3), (0.005)^2*ones(1,3) ]);
	        init_P_15d = kron(eye(Vehicle_num), P_n_init_15d);
	        Q_sigmas_15d.sig_wp = 0; Q_sigmas_15d.sig_wv = 0;  
	        Q_sigmas_15d.sig_wa = 0.0005; Q_sigmas_15d.sig_wR = 0.00005; Q_sigmas_15d.sig_womega = 0.00005;
	        % (2) 9维 EKF/IEKF 标称状态
	        init_states_9d = struct('p', {}, 'v', {}, 'R', {});
	        for n = 1:Vehicle_num
	            v_name = sprintf('V%d', n);
	            veh = trajectories.(v_name);
	            init_states_9d(n).p = [veh.X_true(1); veh.Y_true(1); veh.Z_true(1)];
	            init_states_9d(n).v = [veh.Vx_true(1); veh.Vy_true(1); veh.Vz_true(1)];
	            init_states_9d(n).R = veh.R_true(:, :, 1);
	        end
	        P_n_init_9d = diag([ (0.01)^2*ones(1,3), (0.01)^2*ones(1,3), (1*pi/180)^2*ones(1,3) ]);
	        init_P_9d = kron(eye(Vehicle_num), P_n_init_9d);
	        % (3) 分布式 DMLKF 独立实例化准备
	        Q_sigmas_dmlkf = [0.0005*ones(1,3), 0.0005*ones(1,3), 0.001*ones(1,3), 0.0001*ones(1,3), 0.00025*ones(1,3)];
	        Q_15d_dmlkf = diag(Q_sigmas_dmlkf.^2);
	        init_cov_dmlkf = diag([0.01*ones(1,3), 0.01*ones(1,3), 0.005*ones(1,3), (1*pi/180)*ones(1,3), 0.005*ones(1,3)]);
	        Sigma_a = diag(IMU_noise_params.sigma_na.^2 * ones(1,3));
	        Sigma_w = diag(IMU_noise_params.sigma_nw.^2 * ones(1,3));
	        % 实例化各滤波器类
	        filter_cmlkf = CMLKF(init_states_15d, init_P_15d, Q_sigmas_15d);
	        filter_ekf   = EKF(init_states_9d, init_P_9d); 
	        filter_iekf  = IEKF(init_states_9d, init_P_9d); 
	        filters_dmlkf = cell(Vehicle_num, 1);
	        for n = 1:Vehicle_num
	            th0 = trajectories.(sprintf('V%d', n)).Theta_true(1);
	            R_init = [cos(th0), -sin(th0), 0; sin(th0), cos(th0), 0; 0, 0, 1];
	            dmlkf_state = struct('p', init_states_15d(n).p, 'v', init_states_15d(n).v, 'a', init_states_15d(n).a, 'R', R_init, 'omega', init_states_15d(n).omega);
	            filters_dmlkf{n} = DMLKF(n, dmlkf_state, init_cov_dmlkf, Q_15d_dmlkf, Sigma_a, Sigma_w, dt_imu);
	        end
	        % 动态生成环形双向邻居通信拓扑 (1-2-3-...-N-1)
	        neighbors_map = cell(Vehicle_num, 1);
	        for n = 1:Vehicle_num
	            if n == 1
	                neighbors_map{n} = [Vehicle_num, 2];
	            elseif n == Vehicle_num
	                neighbors_map{n} = [Vehicle_num-1, 1];
	            else
	                neighbors_map{n} = [n-1, n+1];
	            end
	        end
	        % D. 运行记录初始化
	        pos_est_cmlkf = cell(Vehicle_num, 1);
	        pos_est_ekf   = cell(Vehicle_num, 1);
	        pos_est_iekf  = cell(Vehicle_num, 1);
	        pos_est_dmlkf = cell(Vehicle_num, 1);
	        for n = 1:Vehicle_num
	            pos_est_cmlkf{n} = zeros(N_steps, 3);
	            pos_est_ekf{n}   = zeros(N_steps, 3);
	            pos_est_iekf{n}  = zeros(N_steps, 3);
	            pos_est_dmlkf{n} = zeros(N_steps, 3);
	            pos_est_cmlkf{n}(1, :) = init_states_15d(n).p';
	            pos_est_ekf{n}(1, :)   = init_states_9d(n).p';
	            pos_est_iekf{n}(1, :)  = init_states_9d(n).p';
	            pos_est_dmlkf{n}(1, :) = init_states_15d(n).p';
	        end
	        % E. 算法并行积分与滤波循环
	        for k = 2:N_steps
	            % 提取当前步所有车辆经过零偏校正的 IMU 数据
	            imu_acc = zeros(Vehicle_num, 3); imu_gyro = zeros(Vehicle_num, 3);
	            for n = 1:Vehicle_num
	                v_name = sprintf('V%d', n); veh = trajectories.(v_name);
	                ba_true = veh.IMU_bias_a_true(k, :)'; bw_true = veh.IMU_bias_w_true(k, :)';
	                imu_acc(n, :)  = (veh.IMU_acc_m(k, :)'  - ba_true)';
	                imu_gyro(n, :) = (veh.IMU_gyro_m(k, :)' - bw_true)';
	            end
	            % 4算法时间状态预测外推
	            filter_cmlkf.propagate(dt_imu);
	            filter_ekf.propagate(imu_acc, imu_gyro, dt_imu);
	            filter_iekf.propagate(imu_acc, imu_gyro, dt_imu);
	            for n = 1:Vehicle_num
	                filters_dmlkf{n} = filters_dmlkf{n}.predict();
	                filters_dmlkf{n} = filters_dmlkf{n}.update_imu(imu_acc(n, :)', imu_gyro(n, :)', zeros(3,1), zeros(3,1));
	            end
	            % 协同观测更新时刻 (10Hz)
	            if mod(k - 1, 10) == 0
	                uwb_idx = (k - 1) / 10 + 1;
	                anc_meas = zeros(Vehicle_num, Anchor_num);
	                rel_meas = zeros(Vehicle_num, Vehicle_num);
	                for n = 1:Vehicle_num
	                    v_name = sprintf('V%d', n); veh = trajectories.(v_name);
	                    anc_meas(n, :) = veh.UWB_Anchor(uwb_idx, 2:end);
	                    rel_meas(n, :) = veh.UWB_Relative(uwb_idx, 2:end);
	                end
	                % (1) 集中式算法更新
	                filter_cmlkf.update(imu_acc, imu_gyro, anchors, anc_meas, rel_meas, ...
	                                    IMU_noise_params.sigma_na, IMU_noise_params.sigma_nw, ...
	                                    UWB_noise_params.sigma_anc, UWB_noise_params.sigma_rel);
	                filter_ekf.update(anchors, anc_meas, rel_meas, UWB_noise_params.sigma_anc, UWB_noise_params.sigma_rel);
	                filter_iekf.update(anchors, anc_meas, rel_meas, UWB_noise_params.sigma_anc, UWB_noise_params.sigma_rel);
	                % (2) 分布式 DMLKF 网络一致性 ADMM 更新架构
	                p_est_shared = cell(Vehicle_num, 1);
	                I_pos_indep_shared = cell(Vehicle_num, 1); I_pos_dep_shared = cell(Vehicle_num, 1);
	                for n = 1:Vehicle_num
	                    [p_est_shared{n}, I_pos_indep_shared{n}, I_pos_dep_shared{n}] = filters_dmlkf{n}.get_marginalized_position_info();
	                    filters_dmlkf{n} = filters_dmlkf{n}.reset_dual_variables();
	                end
	                max_admm_iter = 2; rho = 1.4;
	                s_admm_all = cell(Vehicle_num, 1);
	                dp_neigh_neigh_all = cell(Vehicle_num, 1); dp_neigh_self_all = cell(Vehicle_num, 1);
	                for n = 1:Vehicle_num
	                    M = length(neighbors_map{n});
	                    s_admm_all{n} = zeros(3 * (M + 1), 1);
	                    dp_neigh_neigh_all{n} = zeros(3, M); dp_neigh_self_all{n} = zeros(3, M);
	                end
	                for admm_k = 1:max_admm_iter
	                    s_admm_new = cell(Vehicle_num, 1);
	                    for n = 1:Vehicle_num
	                        v_name = sprintf('V%d', n); veh = trajectories.(v_name);
	                        anchor_ranges_raw = veh.UWB_Anchor(uwb_idx, 2:end)';
	                        anchor_positions_veh = anchors(1:Anchor_num, :);
	                        active_neighbors = neighbors_map{n}; M_neighbors = length(active_neighbors);
	                        relative_ranges = zeros(M_neighbors, 1); neigh_positions = zeros(M_neighbors, 3);
	                        for a_idx = 1:length(anchor_ranges_raw)
	                            if isnan(anchor_ranges_raw(a_idx)) || isinf(anchor_ranges_raw(a_idx))
	                                anchor_ranges_raw(a_idx) = norm(p_est_shared{n} - anchor_positions_veh(a_idx, :)');
	                            end
	                        end
	                        for idx = 1:M_neighbors
	                            nid = active_neighbors(idx); neigh_positions(idx, :) = p_est_shared{nid}';
	                            rel_val = veh.UWB_Relative(uwb_idx, 1 + nid);
	                            if isnan(rel_val) || isinf(rel_val), rel_val = norm(p_est_shared{n} - p_est_shared{nid}); end
	                            relative_ranges(idx) = rel_val;
	                        end
	                        s_admm_new{n} = filters_dmlkf{n}.solve_primal_public(s_admm_all{n}, anchor_ranges_raw, anchor_positions_veh, ...
	                            neigh_positions, relative_ranges, UWB_noise_params.sigma_anc, UWB_noise_params.sigma_rel, ...
	                            rho, active_neighbors, dp_neigh_neigh_all{n}, dp_neigh_self_all{n});
	                    end
	                    s_admm_all = s_admm_new;
	                    for n = 1:Vehicle_num
	                        active_neighbors = neighbors_map{n};
	                        for idx = 1:length(active_neighbors)
	                            nid = active_neighbors(idx);
	                            dp_neigh_neigh_all{n}(:, idx) = s_admm_all{nid}(1:3);
	                            idx_of_n_in_nid = find(neighbors_map{nid} == n);
	                            if ~isempty(idx_of_n_in_nid)
	                                dp_neigh_self_all{n}(:, idx) = s_admm_all{nid}(3 * idx_of_n_in_nid + (1:3));
	                            else
	                                dp_neigh_self_all{n}(:, idx) = s_admm_all{n}(1:3);
	                            end
	                        end
	                    end
	                    for n = 1:Vehicle_num
	                        filters_dmlkf{n} = filters_dmlkf{n}.update_dual(s_admm_all{n}, neighbors_map{n}, dp_neigh_neigh_all{n}, dp_neigh_self_all{n}, rho);
	                    end
	                end
	                % DMLKF 融合重回射
	                for n = 1:Vehicle_num
	                    v_name = sprintf('V%d', n); veh = trajectories.(v_name);
	                    anchor_ranges_raw = veh.UWB_Anchor(uwb_idx, 2:end)'; anchor_positions_veh = anchors(1:Anchor_num, :);
	                    active_neighbors = neighbors_map{n}; M_neighbors = length(active_neighbors);
	                    relative_ranges = zeros(M_neighbors, 1); neigh_positions = zeros(M_neighbors, 3);
	                    neigh_I_indep = cell(M_neighbors, 1); neigh_I_dep = cell(M_neighbors, 1);
	                    for idx = 1:M_neighbors
	                        nid = active_neighbors(idx); neigh_positions(idx, :) = p_est_shared{nid}';
	                        neigh_I_indep{idx} = I_pos_indep_shared{nid}; neigh_I_dep{idx} = I_pos_dep_shared{nid};
	                        rel_val = veh.UWB_Relative(uwb_idx, 1 + nid);
	                        if isnan(rel_val) || isinf(rel_val), rel_val = norm(p_est_shared{n} - p_est_shared{nid}); end
	                        relative_ranges(idx) = rel_val;
	                    end
	                    for a_idx = 1:length(anchor_ranges_raw)
	                        if isnan(anchor_ranges_raw(a_idx)) || isinf(anchor_ranges_raw(a_idx))
	                            anchor_ranges_raw(a_idx) = norm(p_est_shared{n} - anchor_positions_veh(a_idx, :)');
	                        end
	                    end
	                    filters_dmlkf{n} = filters_dmlkf{n}.apply_uwb_update(s_admm_all{n}, anchor_ranges_raw, anchor_positions_veh, ...
	                        active_neighbors, neigh_positions, neigh_I_indep, neigh_I_dep, relative_ranges, UWB_noise_params.sigma_anc, UWB_noise_params.sigma_rel);
	                end
	            else
	                % 非UWB时刻：集中式 CMLKF 对齐调用高频 IMU 更新步
	                filter_cmlkf.update_imu_only(imu_acc, imu_gyro, IMU_noise_params.sigma_na, IMU_noise_params.sigma_nw);
	            end
	            % 记录估计状态
	            for n = 1:Vehicle_num
	                pos_est_cmlkf{n}(k, :) = filter_cmlkf.states(n).p';
	                pos_est_ekf{n}(k, :)   = filter_ekf.states(n).p';
	                pos_est_iekf{n}(k, :)  = filter_iekf.states(n).p';
	                pos_est_dmlkf{n}(k, :) = filters_dmlkf{n}.state.p';
	            end
	        end
	        % F. 计算误差并取全车均值
	        pos_true = cell(Vehicle_num, 1);
	        for n = 1:Vehicle_num
	            v_name = sprintf('V%d', n);
	            pos_true{n} = [trajectories.(v_name).X_true, trajectories.(v_name).Y_true, trajectories.(v_name).Z_true];
	        end
	        [~, rmse_cmlkf] = calculate_position_errors(pos_est_cmlkf, pos_true);
	        [~, rmse_ekf]   = calculate_position_errors(pos_est_ekf, pos_true);
	        [~, rmse_iekf]  = calculate_position_errors(pos_est_iekf, pos_true);
	        [~, rmse_dmlkf] = calculate_position_errors(pos_est_dmlkf, pos_true);
	        sum_euc_ekf = 0; sum_euc_iekf = 0; sum_euc_cmlkf = 0; sum_euc_dmlkf = 0;
	        for n = 1:Vehicle_num
	            sum_euc_ekf   = sum_euc_ekf   + rmse_ekf(n).euc_rmse;
	            sum_euc_iekf  = sum_euc_iekf  + rmse_iekf(n).euc_rmse;
	            sum_euc_cmlkf = sum_euc_cmlkf + rmse_cmlkf(n).euc_rmse;
	            sum_euc_dmlkf = sum_euc_dmlkf + rmse_dmlkf(n).euc_rmse;
	        end
	        rmse_all_ekf(idx_veh)   = sum_euc_ekf   / Vehicle_num;
	        rmse_all_iekf(idx_veh)  = sum_euc_iekf  / Vehicle_num;
	        rmse_all_cmlkf(idx_veh) = sum_euc_cmlkf / Vehicle_num;
	        rmse_all_dmlkf(idx_veh) = sum_euc_dmlkf / Vehicle_num;
	        fprintf('车辆数 %d 运行完毕。EKF=%.4fm, IEKF=%.4fm, CMLKF=%.4fm, DMLKF=%.4fm\n', ...
	            veh_num, rmse_all_ekf(idx_veh), rmse_all_iekf(idx_veh), rmse_all_cmlkf(idx_veh), rmse_all_dmlkf(idx_veh));
	    end
	end
	%% 3. 数据处理与性能可视化表格输出
	fprintf('\n====================================================== 4算法车辆数量变动综合性能对比表 ======================================================\n');
	fprintf('%-8s | %-15s | %-28s | %-28s | %-28s\n', '车辆数', 'EKF误差', 'IEKF误差及提升(降幅)', 'CMLKF误差及提升(降幅)', 'DMLKF(分布式)误差及提升');
	fprintf('--------------------------------------------------------------------------------------------------------------------------------------------\n');
	for idx_veh = 1:N_veh_tests
	    veh_num = Veh_list(idx_veh);
	    err_ekf = rmse_all_ekf(idx_veh); err_iekf = rmse_all_iekf(idx_veh);
	    err_cmlkf = rmse_all_cmlkf(idx_veh); err_dmlkf = rmse_all_dmlkf(idx_veh);
	    pct_imp_iekf  = (err_ekf - err_iekf)  / err_ekf * 100;
	    pct_imp_cmlkf = (err_ekf - err_cmlkf) / err_ekf * 100;
	    pct_imp_dmlkf = (err_ekf - err_dmlkf) / err_ekf * 100;
	    fprintf('%-8d | %-15.4f | %-28s | %-28s | %-28s\n', ...
	        veh_num, err_ekf, ...
	        sprintf('%.4f (%+.2f%%)', err_iekf, pct_imp_iekf), ...
	        sprintf('%.4f (%+.2f%%)', err_cmlkf, pct_imp_cmlkf), ...
	        sprintf('%.4f (%+.2f%%)', err_dmlkf, pct_imp_dmlkf));
	end
	fprintf('================================================================================================--------------------------------------------\n');
	%% 4. 生成双子图曲线评估分析 (包含 DMLKF 分布式曲线)
	figure('Name', '4-Algorithms Performance Analysis VS Vehicle Number', 'Position', [80, 100, 1300, 550]);
	pct_imp_all_iekf  = (rmse_all_ekf - rmse_all_iekf)  ./ rmse_all_ekf * 100;
	pct_imp_all_cmlkf = (rmse_all_ekf - rmse_all_cmlkf) ./ rmse_all_ekf * 100;
	pct_imp_all_dmlkf = (rmse_all_ekf - rmse_all_dmlkf) ./ rmse_all_ekf * 100;
	% --- 子图 1：平均欧氏定位误差对比 ---
	subplot(1, 2, 1);
	grid on; hold on;
	plot(Veh_list, rmse_all_ekf,   'b-o',  'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', 'b', 'DisplayName', 'EKF (9D, Centralized)');
	plot(Veh_list, rmse_all_iekf,  'g-^',  'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', 'g', 'DisplayName', 'IEKF (9D, Iterative)');
	plot(Veh_list, rmse_all_cmlkf, 'r--s', 'LineWidth', 2.2, 'MarkerSize', 7, 'MarkerFaceColor', 'r', 'DisplayName', 'CMLKF (15D)');
	plot(Veh_list, rmse_all_dmlkf, 'k-.d', 'LineWidth', 2.2, 'MarkerSize', 7, 'MarkerFaceColor', 'k', 'DisplayName', 'DMLKF (15D, Distributed)');
	xlabel('Vehicle Number', 'FontSize', 11, 'FontWeight', 'bold');
	ylabel('Mean Multi-Agent Euclidean Error (m)', 'FontSize', 11, 'FontWeight', 'bold');
	title('Position Error Comparison', 'FontSize', 12, 'FontWeight', 'bold');
	xticks(Veh_list); xlim([Veh_list(1) - 0.5, Veh_list(end) + 0.5]);
	ylim([0, max(rmse_all_ekf) * 1.2]);
	legend('Location', 'northeast', 'FontSize', 9);
	hold off;
	% --- 子图 2：精度提升百分比曲线 ---
	subplot(1, 2, 2);
	grid on; hold on;
	plot(Veh_list, pct_imp_all_iekf,  'g-^',  'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', 'g', 'DisplayName', 'IEKF Improvement');
	plot(Veh_list, pct_imp_all_cmlkf, 'r--s', 'LineWidth', 2.2, 'MarkerSize', 7, 'MarkerFaceColor', 'r', 'DisplayName', 'CMLKF Improvement');
	plot(Veh_list, pct_imp_all_dmlkf, 'k-.d', 'LineWidth', 2.2, 'MarkerSize', 7, 'MarkerFaceColor', 'k', 'DisplayName', 'DMLKF Improvement'); 
	xlabel('Vehicle Number', 'FontSize', 11, 'FontWeight', 'bold');
	ylabel('Accuracy Improvement over EKF (%)', 'FontSize', 11, 'FontWeight', 'bold');
	title('Filtering Accuracy Improvement Degree', 'FontSize', 12, 'FontWeight', 'bold');
	xticks(Veh_list); xlim([Veh_list(1) - 0.5, Veh_list(end) + 0.5]);
	min_imp = min([pct_imp_all_iekf; pct_imp_all_cmlkf; pct_imp_all_dmlkf]);
	max_imp = max([pct_imp_all_iekf; pct_imp_all_cmlkf; pct_imp_all_dmlkf]);
	ylim([min(0, min_imp - 5), max_imp + 10]);
	legend('Location', 'southeast', 'FontSize', 9);
	hold off;
	fprintf('4 算法对比图表（误差 & 提升度分布）渲染完成。\n');
	%% 5. 评测数据自动化保存
	if ~jump_to_plot
	    if ~exist(save_dir, 'dir'), mkdir(save_dir); end
	    save(save_path, 'Veh_list', 'rmse_all_ekf', 'rmse_all_iekf', 'rmse_all_cmlkf', 'rmse_all_dmlkf');
	    fprintf('本次 4 算法评估结果已成功保存固化至：%s\n', save_path);
	else
	    fprintf('当前图表基于历史固化数据渲染生成。\n');
	end