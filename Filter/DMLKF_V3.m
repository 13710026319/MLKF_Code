classdef DMLKF_V3
    % DMLKF_V3: 15维分布式最大似然卡尔曼滤波器类 (纯CI融合 + 无联合状态对照版)
    % 采用标准CI进行更新缩放，且无ADMM对偶迭代，仅执行3D局部GN并基于CI规则做噪声膨胀
    
    properties
        id              % 节点ID (1, 2, 3, 4...)
        state           % 状态结构体: .p(3x1), .v(3x1), .a(3x1), .R(3x3), .omega(3x1)
        I_total         % 15x15 统一总信息矩阵 (无独立/相关分裂)
        Q               % 15x15 过程噪声协方差矩阵
        Sigma_a         % 3x3 加速度计噪声协方差
        Sigma_w         % 3x3 陀螺仪噪声协方差
        g_vec           % 3x1 重力向量
        tau             % IMU采样周期 (秒)
        mu              % 高斯-牛顿正则化参数 (防御NaN)
        omega_self      % 本车CI权重
    end
    
    methods
        %% 构造函数 (签名与各版本严格一致)
        function obj = DMLKF_V3(id, init_state, init_cov, Q_matrix, Sigma_a, Sigma_w, tau)
            obj.id = id;
            obj.state = init_state;
            obj.state.R = obj.robust_orthonormalize(init_state.R);
            
            init_cov_safe = init_cov + 1e-10 * eye(15);
            obj.I_total = obj.safe_inv(init_cov_safe);
            
            obj.Q = Q_matrix;
            obj.Sigma_a = Sigma_a;
            obj.Sigma_w = Sigma_w;
            obj.tau = tau;
            obj.g_vec = [0; 0; -9.81];
            obj.mu = 1e-5;

            obj.omega_self = 0.93;
        end
        
        %% 【接口兼容哑方法】
        function obj = reset_dual_variables(obj)
        end
        
        %% 1.2 去中心化状态与协方差传播 (Prediction Step - 100Hz)
        function obj = predict(obj)
            p_old = obj.state.p;
            v_old = obj.state.v;
            a_old = obj.state.a;
            R_old = obj.state.R;
            w_old = obj.state.omega;
            t_step = obj.tau;
            
            obj.state.p = p_old + t_step * v_old + (t_step^2 / 2) * a_old;
            obj.state.v = v_old + t_step * a_old;
            obj.state.a = a_old;
            
            exp_w = obj.so3_exp_safe(t_step * w_old);
            obj.state.R = obj.robust_orthonormalize(R_old * exp_w);
            obj.state.omega = w_old;
            
            A_t = eye(15);
            A_t(1:3, 4:6)   = t_step * eye(3);
            A_t(1:3, 7:9)   = (t_step^2 / 2) * eye(3);
            A_t(4:6, 7:9)   = t_step * eye(3);
            A_t(10:12, 10:12) = obj.so3_exp_safe(-t_step * w_old);
            A_t(10:12, 13:15) = t_step * obj.so3_right_jacobian_safe(t_step * w_old);
            
            % 标准卡尔曼协方差预测 (无分裂)
            Sigma_curr = obj.safe_inv(obj.I_total);
            Sigma_pred = A_t * Sigma_curr * A_t' + obj.Q;
            Sigma_pred = 0.5 * (Sigma_pred + Sigma_pred');
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
                    s_IMU = zeros(9, 1); break;
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
            
            % 直接混入总信息矩阵
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
            p_est = obj.state.p;
            Sigma = obj.safe_inv(obj.I_total);
            Sigma_pos = Sigma(1:3, 1:3);
            Sigma_pos = 0.5 * (Sigma_pos + Sigma_pos');
        end
        
        %% 【接口兼容哑方法】
        function s_admm = solve_primal_public(obj, s_admm_init, varargin)
            s_admm = s_admm_init;
        end
        function obj = update_dual(obj, varargin)
        end
        
        %% 【核心重构：DMLKF_V3】纯CI融合下的单车3维位置更新
        function obj = apply_uwb_update(obj, ~, anchor_ranges, anchor_positions, ...
                                        neighbor_ids, neighbor_positions, ...
                                        neighbor_Sigma_pos, relative_ranges, sigma_s, sigma_z)
            K = length(anchor_ranges);
            M = length(neighbor_ids);
            
            if M == 0
                obj = obj.update_uwb_anchor_only(anchor_ranges, anchor_positions, sigma_s);
                return;
            end
            
            p_prior = obj.state.p;
            s_pos = zeros(3, 1);
            
            % --- 1. 基于 CI 比例折损邻车先验精度，得到膨胀邻车协方差 ---
            omega_neigh = (1 - 0.8) / M;
            Sigma_neigh_expanded = cell(M, 1);
            for i = 1:M
                % CI规则下，等效测量误差被膨胀为 1/omega_neigh 倍 [71]
                Sigma_neigh_expanded{i} = (1 / omega_neigh) * neighbor_Sigma_pos{i};
            end
            
            % --- 2. 纯3维 Gauss-Newton 迭代解算 ---
            max_inner_gn = 5;
            for iter = 1:max_inner_gn
                p_est = p_prior + s_pos;
                g_L = zeros(3, 1);
                H_L = zeros(3, 3);
                
                % 基站测距
                for k = 1:K
                    vec = p_est - anchor_positions(k, :)';
                    dist = max(norm(vec), 1e-6);
                    u_k = vec / dist;
                    r_anc = anchor_ranges(k) - dist;
                    
                    g_L = g_L - (1 / sigma_s^2) * u_k * r_anc;
                    H_L = H_L + (1 / sigma_s^2) * (u_k * u_k');
                end
                
                % 协同测距 (动态CI噪声膨胀)
                for i = 1:M
                    p_neigh_est = neighbor_positions(i, :)';
                    vec = p_est - p_neigh_est;
                    dist = max(norm(vec), 1e-6);
                    u_i = vec / dist;
                    r_int = relative_ranges(i) - dist;
                    
                    % CI 噪声膨胀核心公式：基础测距方差 + 膨胀后的邻车不确定性投影
                    sigma2_z_eff = sigma_z^2 + u_i' * Sigma_neigh_expanded{i} * u_i;
                    
                    g_L = g_L - (1 / sigma2_z_eff) * u_i * r_int;
                    H_L = H_L + (1 / sigma2_z_eff) * (u_i * u_i');
                end
                
                step = obj.safe_solve(H_L, g_L, obj.mu);
                max_step_pos = 0.1;
                n_step_pos = norm(step);
                if n_step_pos > max_step_pos
                    step = step * (max_step_pos / n_step_pos);
                end
                s_pos = s_pos - step;
                
                if norm(step) < 1e-4, break; end
                if any(isnan(s_pos)) || any(isinf(s_pos))
                    s_pos = zeros(3, 1); break;
                end
            end
            s_pos = obj.sanitize_vector(s_pos);
            p_converged = p_prior + s_pos;
            
            % --- 3. 构造测量信息矩阵 (CI 噪声膨胀版本) ---
            Lambda_t_total = zeros(3, 3);
            % 基站贡献
            for k = 1:K
                u_k = (p_converged - anchor_positions(k, :)') / ...
                      max(norm(p_converged - anchor_positions(k, :)'), 1e-6);
                Lambda_t_total = Lambda_t_total + (1 / sigma_s^2) * (u_k * u_k');
            end
            % 协同邻车贡献
            for i = 1:M
                p_neigh_est = neighbor_positions(i, :)';
                u_i = (p_converged - p_neigh_est) / ...
                      max(norm(p_converged - p_neigh_est), 1e-6);
                sigma2_z_eff = sigma_z^2 + u_i' * Sigma_neigh_expanded{i} * u_i;
                Lambda_t_total = Lambda_t_total + (1 / sigma2_z_eff) * (u_i * u_i');
            end
            
            lambda_t_total = Lambda_t_total * s_pos;
            
            % --- 4. 映射到15维并在 CI 机制下融合 ---
            Lambda_full = zeros(15, 15); Lambda_full(1:3, 1:3) = Lambda_t_total;
            lambda_full = zeros(15, 1);  lambda_full(1:3) = lambda_t_total;
            
            % 传统 CI 融合核心步骤：本车先验总信息乘以自身的 omega_self (防止信息双重计数)
            I_post = obj.omega_self * obj.I_total + Lambda_full;
            I_post = obj.sanitize_matrix(I_post);
            
            Sigma_post = obj.safe_inv(I_post);
            delta_theta = Sigma_post * lambda_full;
            
            % --- 5. 回射与更新统一总信息矩阵 ---
            obj.state.p = obj.state.p + delta_theta(1:3);
            obj.state.v = obj.state.v + delta_theta(4:6);
            obj.state.a = obj.state.a + delta_theta(7:9);
            obj.state.R = obj.robust_orthonormalize(obj.state.R * obj.so3_exp_safe(delta_theta(10:12)));
            obj.state.omega = obj.state.omega + delta_theta(13:15);
            
            obj.I_total = obj.sanitize_matrix(obj.safe_inv(Sigma_post));
        end
        
        %% 单车退化 UWB 更新
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