	classdef DMLKF_V2
	    % DMLKF_V2: 15维分布式最大似然卡尔曼滤波器类 (纯CI融合版)
	    % 保留邻车状态联合ADMM优化，但先验与后验融合采用协方差交集(CI)防数据血缘污染
	    properties
	        id              % 节点ID (1, 2, 3, 4...)
	        state           % 状态结构体: .p(3x1), .v(3x1), .a(3x1), .R(3x3), .omega(3x1)
	        I_total         % 15x15 统一总信息矩阵 (无独立/相关分裂)
	        Q               % 15x15 过程噪声协方差矩阵
	        Sigma_a         % 3x3 加速度计噪声协方差
	        Sigma_w         % 3x3 陀螺仪噪声协方差
	        g_vec           % 3x1 重力向量 (通常为 [0; 0; -9.81])
	        tau             % IMU采样周期 (秒)
	        mu              % 高斯-牛顿正则化参数 (防御NaN)
	        omega_self      % 本车CI权重 (外部输入)
	        % ADMM 持久化对偶变量
	        lambda_local    % Map 容器: 存储该车对邻车的拉格朗日乘子
	        lambda_remote   % Map 容器: 存储邻车对该车的拉格朗日乘子
	    end
	    methods
	        %% 构造函数
	        function obj = DMLKF_V2(id, init_state, init_cov, Q_matrix, Sigma_a, Sigma_w, tau)
	            obj.id = id;
	            obj.state = init_state;
	            obj.state.R = obj.robust_orthonormalize(init_state.R);
	            % 保证初始协方差不奇异，防 NaN
	            init_cov_safe = init_cov + 1e-10 * eye(15);
	            obj.I_total = obj.safe_inv(init_cov_safe);
	            obj.Q = Q_matrix;
	            obj.Sigma_a = Sigma_a;
	            obj.Sigma_w = Sigma_w;
	            obj.tau = tau;
	            obj.g_vec = [0; 0; -9.81];
	            obj.mu = 1e-5; % 防止GN迭代中Hessian退化
                % obj.omega_self = 0.93;
                obj.omega_self = 0.85;
	            obj.lambda_local = containers.Map('KeyType', 'double', 'ValueType', 'any');
	            obj.lambda_remote = containers.Map('KeyType', 'double', 'ValueType', 'any');
	        end
	        %% 重置对偶变量 (新历元防爆机制)
	        function obj = reset_dual_variables(obj)
	            obj.lambda_local = containers.Map('KeyType', 'double', 'ValueType', 'any');
	            obj.lambda_remote = containers.Map('KeyType', 'double', 'ValueType', 'any');
	        end
	        %% 1.2 去中心化状态与协方差传播 (Prediction Step - 100Hz)
	        function obj = predict(obj)
	            % 提取当前名义状态
	            p_old = obj.state.p;
	            v_old = obj.state.v;
	            a_old = obj.state.a;
	            R_old = obj.state.R;
	            w_old = obj.state.omega;
	            t_step = obj.tau;
	            % --- 1. 名义状态传播 ---
	            obj.state.p = p_old + t_step * v_old + (t_step^2 / 2) * a_old;
	            obj.state.v = v_old + t_step * a_old;
	            obj.state.a = a_old;
	            exp_w = obj.so3_exp_safe(t_step * w_old);
	            obj.state.R = obj.robust_orthonormalize(R_old * exp_w);
	            obj.state.omega = w_old;
	            % --- 2. 构造 Jacobian A_t ---
	            A_t = eye(15);
	            A_t(1:3, 4:6)   = t_step * eye(3);
	            A_t(1:3, 7:9)   = (t_step^2 / 2) * eye(3);
	            A_t(4:6, 7:9)   = t_step * eye(3);
	            A_t(10:12, 10:12) = obj.so3_exp_safe(-t_step * w_old);
	            A_t(10:12, 13:15) = t_step * obj.so3_right_jacobian_safe(t_step * w_old);
	            % --- 3. 标准卡尔曼协方差传播 (无分裂操作) ---
	            Sigma_curr = obj.safe_inv(obj.I_total);
	            Sigma_pred = A_t * Sigma_curr * A_t' + obj.Q;
	            Sigma_pred = 0.5 * (Sigma_pred + Sigma_pred'); % 保证对称正定
	            obj.I_total = obj.sanitize_matrix(obj.safe_inv(Sigma_pred));
	        end
	        %% 2.2 局部高频 IMU 更新 (IMU Update - 100Hz)
	        function obj = update_imu(obj, raw_acc, raw_gyro, bias_a, bias_w)
	            acc_tilde = raw_acc - bias_a;
	            gyro_tilde = raw_gyro - bias_w;
	            R_IMU = blkdiag(obj.Sigma_a, obj.Sigma_w);
	            R_IMU_inv = obj.safe_inv(R_IMU);
	            s_IMU = zeros(9, 1);
	            a_prior = obj.state.a;
	            R_prior = obj.state.R;
	            w_prior = obj.state.omega;
	            max_iter = 5;
	            for iter = 1:max_iter
	                da = s_IMU(1:3); dphi = s_IMU(4:6); domega = s_IMU(7:9);
	                a_iter = a_prior + da;
	                R_iter = obj.robust_orthonormalize(R_prior * obj.so3_exp_safe(dphi));
	                w_iter = w_prior + domega;
	                r_acc = acc_tilde - R_iter' * (a_iter - obj.g_vec);
	                r_gyro = gyro_tilde - w_iter;
	                r = [r_acc; r_gyro];
	                Jr_phi = obj.so3_right_jacobian_safe(dphi);
	                H = zeros(6, 9);
	                H(1:3, 1:3) = -R_iter';
	                H(1:3, 4:6) = -obj.skew_matrix(R_iter' * (a_iter - obj.g_vec)) * Jr_phi;
	                H(4:6, 7:9) = -eye(3);
	                Hessian = H' * R_IMU_inv * H;
	                grad = H' * R_IMU_inv * r;
	                step = obj.safe_solve(Hessian, grad, obj.mu);
	                max_step = 0.1;
	                n_step = norm(step);
	                if n_step > max_step
	                    step = step * (max_step / n_step);
	                end
	                s_IMU = s_IMU - step;
	                if norm(step) < 1e-4, break; end
	                if any(isnan(s_IMU)) || any(isinf(s_IMU))
	                    s_IMU = zeros(9, 1);
	                    break;
	                end
	            end
	            s_IMU = obj.sanitize_vector(s_IMU);
	            da = s_IMU(1:3); dphi = s_IMU(4:6); domega = s_IMU(7:9);
	            a_converged = a_prior + da;
	            R_converged = obj.robust_orthonormalize(R_prior * obj.so3_exp_safe(dphi));
	            Jr_phi = obj.so3_right_jacobian_safe(dphi);
	            H_final = zeros(6, 9);
	            H_final(1:3, 1:3) = -R_converged';
	            H_final(1:3, 4:6) = -obj.skew_matrix(R_converged' * (a_converged - obj.g_vec)) * Jr_phi;
	            H_final(4:6, 7:9) = -eye(3);
	            Lambda_IMU = H_final' * R_IMU_inv * H_final;
	            lambda_IMU = Lambda_IMU * s_IMU;
	            Lambda_full = zeros(15, 15); lambda_full = zeros(15, 1);
	            idx_15 = 7:15;
	            Lambda_full(idx_15, idx_15) = Lambda_IMU;
	            lambda_full(idx_15) = lambda_IMU;
	            % 标准信息滤波融合 (无SCI分裂)
	            obj.I_total = obj.sanitize_matrix(obj.I_total + Lambda_full);
	            Sigma_post = obj.safe_inv(obj.I_total);
	            delta_theta = Sigma_post * lambda_full;
	            obj.state.p = obj.state.p + delta_theta(1:3);
	            obj.state.v = obj.state.v + delta_theta(4:6);
	            obj.state.a = obj.state.a + delta_theta(7:9);
	            obj.state.R = obj.robust_orthonormalize(obj.state.R * obj.so3_exp_safe(delta_theta(10:12)));
	            obj.state.omega = obj.state.omega + delta_theta(13:15);
	        end
	        %% 3.0 UWB 总信息投影 (边缘化输出接口 - 邻居车辆调用)
	        function [p_est, Sigma_pos] = get_marginalized_position_info(obj)
	            % 邻居通过该方法获取自身 3D 位置的先验估计与总协方差
	            p_est = obj.state.p;
	            Sigma = obj.safe_inv(obj.I_total);
	            Sigma_pos = Sigma(1:3, 1:3);
	            Sigma_pos = 0.5 * (Sigma_pos + Sigma_pos');
	        end
	        %% 【公共接口 1】Primal Update 求解器 (内层 GN 迭代 - 由外部ADMM循环调用)
	        function s_admm = solve_primal_public(obj, s_admm_init, anchor_ranges, anchor_positions, ...
	                                              neighbor_positions, relative_ranges, ...
	                                              sigma_s, sigma_z, rho, neighbor_ids, ...
	                                              dp_neigh_neigh, dp_neigh_self)
	            K = length(anchor_ranges);
	            M = length(neighbor_ids);
	            s_dim = length(s_admm_init);
	            s_admm = s_admm_init;
	            max_inner_gn = 5;
	            for inner_iter = 1:max_inner_gn
	                p_self_est = obj.state.p + s_admm(1:3);
	                r_anc = zeros(K, 1); u_anc = zeros(3, K);
	                for k = 1:K
	                    vec = p_self_est - anchor_positions(k, :)';
	                    dist = max(norm(vec), 1e-6);
	                    u_anc(:, k) = vec / dist;
	                    r_anc(k) = anchor_ranges(k) - dist;
	                end
	                r_int = zeros(M, 1); u_int = zeros(3, M);
	                for i = 1:M
	                    dp_self_neigh = s_admm(3*i + (1:3));
	                    p_neigh_est = neighbor_positions(i, :)' + dp_self_neigh;
	                    vec = p_self_est - p_neigh_est;
	                    dist = max(norm(vec), 1e-6);
	                    u_int(:, i) = vec / dist;
	                    r_int(i) = relative_ranges(i) - dist;
	                end
	                g_L = zeros(s_dim, 1);
	                g_dp11 = zeros(3, 1);
	                for k = 1:K
	                    g_dp11 = g_dp11 - (1 / sigma_s^2) * u_anc(:, k) * r_anc(k);
	                end
	                for i = 1:M
	                    nid = neighbor_ids(i);
	                    if ~isKey(obj.lambda_remote, nid), obj.lambda_remote(nid) = zeros(3, 1); end
	                    if ~isKey(obj.lambda_local, nid), obj.lambda_local(nid) = zeros(3, 1); end
	                    g_dp11 = g_dp11 - (1 / sigma_z^2) * u_int(:, i) * r_int(i) ...
	                             + (-obj.lambda_remote(nid) - rho * (dp_neigh_self(:, i) - s_admm(1:3)));
	                end
	                g_L(1:3) = g_dp11;
	                for i = 1:M
	                    nid = neighbor_ids(i);
	                    if ~isKey(obj.lambda_remote, nid), obj.lambda_remote(nid) = zeros(3, 1); end
	                    if ~isKey(obj.lambda_local, nid), obj.lambda_local(nid) = zeros(3, 1); end
	                    dp_self_neigh = s_admm(3*i + (1:3));
	                    g_dp1j = (1 / sigma_z^2) * u_int(:, i) * r_int(i) ...
	                             + obj.lambda_local(nid) + rho * (dp_self_neigh - dp_neigh_neigh(:, i));
	                    g_L(3*i + (1:3)) = g_dp1j;
	                end
	                H_L = zeros(s_dim, s_dim);
	                for k = 1:K
	                    H_k = zeros(1, s_dim); H_k(1:3) = -u_anc(:, k)';
	                    H_L = H_L + (1 / sigma_s^2) * (H_k' * H_k);
	                end
	                for i = 1:M
	                    H_int = zeros(1, s_dim); H_int(1:3) = -u_int(:, i)';
	                    H_int(3*i + (1:3)) = u_int(:, i)';
	                    H_L = H_L + (1 / sigma_z^2) * (H_int' * H_int);
	                end
	                D_penalty = zeros(s_dim, s_dim);
	                D_penalty(1:3, 1:3) = (M * rho) * eye(3);
	                for i = 1:M
	                    idx = 3*i + (1:3);
	                    D_penalty(idx, idx) = rho * eye(3);
	                end
	                H_L = H_L + D_penalty;
	                step = obj.safe_solve(H_L, g_L, obj.mu);
	                max_step_admm = 0.1;
	                n_step_admm = norm(step);
	                if n_step_admm > max_step_admm
	                    step = step * (max_step_admm / n_step_admm);
	                end
	                s_admm = s_admm - step;
	                if norm(step) < 1e-4, break; end
	                if any(isnan(s_admm)) || any(isinf(s_admm))
	                    s_admm = s_admm_init; break;
	                end
	            end
	            s_admm = obj.sanitize_vector(s_admm);
	        end
	        %% 【公共接口 2】Dual Variable Update (拉格朗日乘子更新)
	        function obj = update_dual(obj, s_admm, neighbor_ids, dp_neigh_neigh, dp_neigh_self, rho)
	            M = length(neighbor_ids);
	            dp_self_est = s_admm(1:3);
	            for i = 1:M
	                nid = neighbor_ids(i);
	                dp_self_neigh = s_admm(3*i + (1:3));
	                if ~isKey(obj.lambda_local, nid)
	                    obj.lambda_local(nid) = zeros(3, 1);
	                    obj.lambda_remote(nid) = zeros(3, 1);
	                end
	                obj.lambda_local(nid) = obj.lambda_local(nid) + rho * (dp_self_neigh - dp_neigh_neigh(:, i));
	                obj.lambda_remote(nid) = obj.lambda_remote(nid) + rho * (dp_neigh_self(:, i) - dp_self_est);
	            end
	        end
	        %% 【核心重构：DMLKF_V2】CI加权舒尔补边缘化与状态回射
	        function obj = apply_uwb_update(obj, s_star, anchor_ranges, anchor_positions, ...
	                                        neighbor_ids, neighbor_positions, ...
	                                        neighbor_Sigma_pos, relative_ranges, sigma_s, sigma_z)
	            % neighbor_Sigma_pos: 1xM cell, 存储邻车发来的 3x3 总位置协方差
	            K = length(anchor_ranges);
	            M = length(neighbor_ids);
	            if M == 0
	                obj = obj.update_uwb_anchor_only(anchor_ranges, anchor_positions, sigma_s);
	                return;
	            end
	            p_self_converged = obj.state.p + s_star(1:3);
	            s_self_mle = s_star(1:3);
	            % --- 1. 构造收敛后的联合测量信息矩阵 Xi (Eq. 59-60) ---
	            Xi_aa = zeros(3, 3);
	            for k = 1:K
	                u_k = (p_self_converged - anchor_positions(k, :)') / ...
	                      max(norm(p_self_converged - anchor_positions(k, :)'), 1e-6);
	                Xi_aa = Xi_aa + (1 / sigma_s^2) * (u_k * u_k');
	            end
	            Xi_ii_11 = zeros(3, 3);
	            Xi_ii_12 = zeros(3, 3 * M);
	            Xi_ii_22 = zeros(3 * M, 3 * M);
	            s_neigh_mle = zeros(3 * M, 1);
	            for i = 1:M
	                dp_self_neigh = s_star(3*i + (1:3));
	                p_neigh_converged = neighbor_positions(i, :)' + dp_self_neigh;
	                u_ij = (p_self_converged - p_neigh_converged) / ...
	                       max(norm(p_self_converged - p_neigh_converged), 1e-6);
	                Xi_block = (1 / sigma_z^2) * (u_ij * u_ij');
	                Xi_ii_11 = Xi_ii_11 + Xi_block;
	                Xi_ii_12(:, 3*(i-1)+(1:3)) = -Xi_block;
	                Xi_ii_22(3*(i-1)+(1:3), 3*(i-1)+(1:3)) = Xi_block;
	                s_neigh_mle(3*(i-1)+(1:3)) = dp_self_neigh;
	            end
	            Xi_11 = Xi_aa + Xi_ii_11;
	            Xi_12 = Xi_ii_12;
	            Xi_21 = Xi_ii_12';
	            Xi_22 = Xi_ii_22;

	            % --- 2. 基于 CI 的舒尔补边缘化 (Eq. 65-68) ---
	            omega_neigh = (1 - obj.omega_self) / M;

	            P22_CI_inv = zeros(3 * M, 3 * M);
	            for i = 1:M
	                idx = 3*(i-1)+(1:3);
	                % 邻车先验信息乘以 CI 权重
	                Lambda_neigh_CI = omega_neigh * obj.safe_inv(neighbor_Sigma_pos{i});
	                P22_CI_inv(idx, idx) = Lambda_neigh_CI;
	            end
	            % 边缘化等效测量信息矩阵
	            temp_inv = obj.safe_inv(Xi_22 + P22_CI_inv);
	            Lambda_t_CI = Xi_11 - Xi_12 * temp_inv * Xi_21;
	            % 边缘化等效信息向量
	            lambda_t_CI = Lambda_t_CI * s_self_mle + Xi_12 * temp_inv * P22_CI_inv * s_neigh_mle;
	            % --- 3. 局部 15维 MLKF 信息 CI 融合 (Eq. 69-72) ---
	            Lambda_full = zeros(15, 15); Lambda_full(1:3, 1:3) = Lambda_t_CI;
	            lambda_full = zeros(15, 1);  lambda_full(1:3) = lambda_t_CI;
	            % 本车先验信息乘以 CI 权重 (对应公式71中的 omega1 * Sigma^{-1})
	            I_post = obj.omega_self * obj.I_total + Lambda_full;
	            I_post = obj.sanitize_matrix(I_post);
	            Sigma_post = obj.safe_inv(I_post);
	            delta_theta = Sigma_post * lambda_full;
	            % --- 4. 流形回射 (Eq. 73) ---
	            obj.state.p = obj.state.p + delta_theta(1:3);
	            obj.state.v = obj.state.v + delta_theta(4:6);
	            obj.state.a = obj.state.a + delta_theta(7:9);
	            obj.state.R = obj.robust_orthonormalize(obj.state.R * obj.so3_exp_safe(delta_theta(10:12)));
	            obj.state.omega = obj.state.omega + delta_theta(13:15);
	            % 更新统一信息矩阵
	            obj.I_total = obj.sanitize_matrix(obj.safe_inv(Sigma_post));
	        end
	        %% 退化的单车 UWB 位置更新 (Anchor Only)
	        function obj = update_uwb_anchor_only(obj, anchor_ranges, anchor_positions, sigma_s)
	            K = length(anchor_ranges);
	            s_pos = zeros(3, 1);
	            p_prior = obj.state.p;
	            for iter = 1:5
	                p_est = p_prior + s_pos;
	                r = zeros(K, 1); H = zeros(K, 3);
	                for k = 1:K
	                    vec = p_est - anchor_positions(k, :)';
	                    dist = max(norm(vec), 1e-6);
	                    H(k, :) = -vec' / dist;
	                    r(k) = anchor_ranges(k) - dist;
	                end
	                Hessian = (1/sigma_s^2) * (H' * H);
	                grad = (1/sigma_s^2) * (H' * r);
	                step = obj.safe_solve(Hessian, grad, obj.mu);
	                s_pos = s_pos - step;
	            end
	            Lambda_anc = (1/sigma_s^2) * (H' * H);
	            lambda_anc = Lambda_anc * s_pos;
	            Lambda_full = zeros(15, 15); Lambda_full(1:3, 1:3) = Lambda_anc;
	            lambda_full = zeros(15, 1);  lambda_full(1:3) = lambda_anc;
	            % 单车时 omega_self 为 1
	            I_post = obj.I_total + Lambda_full;
	            Sigma_post = obj.safe_inv(I_post);
	            delta_theta = Sigma_post * lambda_full;
	            obj.state.p = obj.state.p + delta_theta(1:3);
	            obj.state.v = obj.state.v + delta_theta(4:6);
	            obj.state.a = obj.state.a + delta_theta(7:9);
	            obj.state.R = obj.robust_orthonormalize(obj.state.R * obj.so3_exp_safe(delta_theta(10:12)));
	            obj.I_total = obj.sanitize_matrix(obj.safe_inv(Sigma_post));
	        end
	    end
	    %% 数值安全防护辅助函数 (私有)
	    methods (Access = private)
	        function R_orth = robust_orthonormalize(~, R)
	            if any(isnan(R(:))) || any(isinf(R(:)))
	                R_orth = eye(3); return;
	            end
	            [U, ~, V] = svd(R);
	            R_orth = U * V';
	            if det(R_orth) < 0
	                R_orth = U * diag([1, 1, -1]) * V';
	            end
	        end
	        function R = so3_exp_safe(obj, phi)
	            theta = norm(phi);
	            phi_skew = obj.skew_matrix(phi);
	            if theta < 1e-6
	                R = eye(3) + phi_skew;
	            else
	                R = eye(3) + (sin(theta)/theta) * phi_skew + ((1 - cos(theta))/theta^2) * (phi_skew * phi_skew);
	            end
	        end
	        function Jr = so3_right_jacobian_safe(obj, phi)
	            theta = norm(phi);
	            phi_skew = obj.skew_matrix(phi);
	            if theta < 1e-3
	                Jr = eye(3) - 0.5 * phi_skew + (1/6) * (phi_skew * phi_skew);
	            else
	                Jr = eye(3) - ((1 - cos(theta))/theta^2) * phi_skew + ((theta - sin(theta))/theta^3) * (phi_skew * phi_skew);
	            end
	        end
	        function x = safe_solve(~, A, b, reg)
	            A_reg = A + reg * eye(size(A));
	            x = A_reg \ b;
	            if any(isnan(x)) || any(isinf(x))
	                x = pinv(A) * b;
	            end
	        end
	        function A_inv = safe_inv(~, A)
	            A_reg = A + 1e-11 * eye(size(A));
	            A_inv = inv(A_reg);
	            if any(isnan(A_inv(:))) || any(isinf(A_inv(:)))
	                A_inv = pinv(A);
	            end
	        end
	        function M_out = sanitize_matrix(~, M_in)
	            M_out = M_in;
	            M_out(isnan(M_out)) = 0;
	            M_out(isinf(M_out)) = 0;
	            M_out = 0.5 * (M_out + M_out');
	        end
	        function v_out = sanitize_vector(~, v_in)
	            v_out = v_in;
	            v_out(isnan(v_out)) = 0;
	            v_out(isinf(v_out)) = 0;
	        end
	        function skew_R = skew_matrix(~, v)
	            skew_R = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
	        end
	    end
	end