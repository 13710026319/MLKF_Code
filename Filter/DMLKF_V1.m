classdef DMLKF_V1
    % DMLKF_V1: 15维去中心化最大似然卡尔曼滤波器基准类 (无联合状态对照组)
    % 采用邻车协方差投影膨胀法处理协同测距，内部直接执行纯3维GN位置解算，无ADMM对偶迭代
    
    properties
        id              % 节点ID (1, 2, 3, 4...)
        state           % 状态结构体: .p(3x1), .v(3x1), .a(3x1), .R(3x3), .omega(3x1)
        I_indep         % 15x15 独立信息矩阵
        I_dep           % 15x15 相关信息矩阵
        Q               % 15x15 过程噪声协方差矩阵
        Sigma_a         % 3x3 加速度计噪声协方差
        Sigma_w         % 3x3 陀螺仪噪声协方差
        g_vec           % 3x1 重力向量 (通常为 [0; 0; -9.81])
        tau             % IMU采样周期 (秒)
        mu              % 高斯-牛顿正则化参数 (防御NaN)
        omega_self      % 本车SCI权重
    end
    
    methods
        %% 构造函数 (签名与原DMLKF严格一致)
        function obj = DMLKF_V1(id, init_state, init_cov, Q_matrix, Sigma_a, Sigma_w, tau, omega_self)
            obj.id = id;
            obj.state = init_state;
            obj.state.R = obj.robust_orthonormalize(init_state.R);
            
            % 保证初始协方差不奇异，防 NaN
            init_cov_safe = init_cov + 1e-10 * eye(15);
            obj.I_indep = obj.safe_inv(init_cov_safe);
            obj.I_dep = zeros(15, 15);
            
            obj.Q = Q_matrix;
            obj.Sigma_a = Sigma_a;
            obj.Sigma_w = Sigma_w;
            obj.tau = tau;
            obj.g_vec = [0; 0; -9.81];
            obj.mu = 1e-5; % 防止GN迭代中Hessian退化
            obj.omega_self = omega_self;
        end
        
        %% 【接口兼容】重置对偶变量哑接口 (V1无对偶变量，直接返回)
        function obj = reset_dual_variables(obj)
            % 仅为与 DMLKF 保持接口调用一致，不执行任何实质重置
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
            
            % --- 1. 名义状态传播 (Eq. 7-11) ---
            obj.state.p = p_old + t_step * v_old + (t_step^2 / 2) * a_old;
            obj.state.v = v_old + t_step * a_old;
            obj.state.a = a_old;
            
            % 姿态传播
            exp_w = obj.so3_exp_safe(t_step * w_old);
            obj.state.R = obj.robust_orthonormalize(R_old * exp_w);
            obj.state.omega = w_old;
            
            % --- 2. 构造 Jacobian A_t (Eq. 15) ---
            A_t = eye(15);
            A_t(1:3, 4:6)   = t_step * eye(3);
            A_t(1:3, 7:9)   = (t_step^2 / 2) * eye(3);
            A_t(4:6, 7:9)   = t_step * eye(3);
            A_t(10:12, 10:12) = obj.so3_exp_safe(-t_step * w_old);
            A_t(10:12, 13:15) = t_step * obj.so3_right_jacobian_safe(t_step * w_old);
            
            % --- 3. 协方差与分裂信息矩阵传播 ---
            Sigma_curr = obj.safe_inv(obj.I_indep + obj.I_dep);
            Sigma_pred = A_t * Sigma_curr * A_t' + obj.Q;
            Sigma_pred = 0.5 * (Sigma_pred + Sigma_pred'); % 保证对称正定
            
            I_total_pred = obj.safe_inv(Sigma_pred);
            W_t = Sigma_pred \ (A_t * Sigma_curr);
            if any(isnan(W_t(:))) || any(isinf(W_t(:)))
                W_t = pinv(Sigma_pred) * (A_t * Sigma_curr);
            end
            
            I_dep_new = W_t * obj.I_dep * W_t';
            I_dep_new = 0.5 * (I_dep_new + I_dep_new');
            
            I_indep_new = I_total_pred - I_dep_new;
            I_indep_new = 0.5 * (I_indep_new + I_indep_new');
            
            obj.I_indep = obj.sanitize_matrix(I_indep_new);
            obj.I_dep = obj.sanitize_matrix(I_dep_new);
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
            
            obj.I_indep = obj.sanitize_matrix(obj.I_indep + Lambda_full);
            
            Sigma_post = obj.safe_inv(obj.I_indep + obj.I_dep);
            delta_theta = Sigma_post * lambda_full;
            
            obj.state.p = obj.state.p + delta_theta(1:3);
            obj.state.v = obj.state.v + delta_theta(4:6);
            obj.state.a = obj.state.a + delta_theta(7:9);
            obj.state.R = obj.robust_orthonormalize(obj.state.R * obj.so3_exp_safe(delta_theta(10:12)));
            obj.state.omega = obj.state.omega + delta_theta(13:15);
        end
        
        %% 3.0 UWB 独立/相关信息投影 (签名与原DMLKF一致，供邻居提取使用)
        function [p_est, I_pos_indep, I_pos_dep] = get_marginalized_position_info(obj)
            p_est = obj.state.p;
            Sigma_j = obj.safe_inv(obj.I_indep + obj.I_dep);
            
            Sigma_pos = Sigma_j(1:3, 1:3);
            I_pos_total = obj.safe_inv(Sigma_pos);
            
            pi_p = [eye(3), zeros(3, 12)];
            W_pos = I_pos_total * pi_p * Sigma_j;
            
            I_pos_dep = W_pos * obj.I_dep * W_pos';
            I_pos_dep = 0.5 * (I_pos_dep + I_pos_dep');
            
            I_pos_indep = I_pos_total - I_pos_dep;
            I_pos_indep = 0.5 * (I_pos_indep + I_pos_indep');
            
            I_pos_indep = obj.sanitize_matrix(I_pos_indep);
            I_pos_dep = obj.sanitize_matrix(I_pos_dep);
        end
        
        %% 【接口兼容哑方法】Primal Update 求解器 (V1跳过ADMM，直接返回零误差状态)
        function s_admm = solve_primal_public(obj, s_admm_init, varargin)
            % 对外伪装输出，实际测试脚本在运行DMLKF_V1路径时将不经过此迭代
            s_admm = s_admm_init;
        end
        
        %% 【接口兼容哑方法】Dual Variable Update (V1无乘子，不更新)
        function obj = update_dual(obj, varargin)
            % 仅用于接口兼容
        end
        
        %% 【核心重构：DMLKF_V1】位置解算、等效信息提取与流形回射
        function obj = apply_uwb_update(obj, ~, anchor_ranges, anchor_positions, ...
                                        neighbor_ids, neighbor_positions, ...
                                        neighbor_I_indep, neighbor_I_dep, ...
                                        relative_ranges, sigma_s, sigma_z)
            % 形参1填入 [] 即可。内部执行3维纯位置GN，并结合邻车SCI先验做测距噪声膨胀
            K = length(anchor_ranges);
            M = length(neighbor_ids);
            
            % 若完全无邻车，退化为纯基站解算
            if M == 0
                obj = obj.update_uwb_anchor_only(anchor_ranges, anchor_positions, sigma_s);
                return;
            end
            
            p_prior = obj.state.p;
            s_pos = zeros(3, 1); % 仅3维位置优化状态 [dp_self]
            
            % --- 1. 预先计算邻车投影后的位置协方差矩阵 (SCI) ---
            omega_neigh = (1 - obj.omega_self) / M;
            Sigma_neigh_SCI = cell(M, 1);
            for i = 1:M
                Lambda_j_SCI = neighbor_I_indep{i} + omega_neigh * neighbor_I_dep{i};
                Sigma_neigh_SCI{i} = obj.safe_inv(Lambda_j_SCI);
            end
            
            % --- 2. 纯3维非线性高斯-牛顿迭代 (局部单车解算) ---
            max_inner_gn = 5;
            for iter = 1:max_inner_gn
                p_est = p_prior + s_pos;
                
                % 初始化 3x1 梯度与 3x3 Hessian
                g_L = zeros(3, 1);
                H_L = zeros(3, 3);
                
                % A. 基站测距约束
                for k = 1:K
                    vec = p_est - anchor_positions(k, :)';
                    dist = max(norm(vec), 1e-6);
                    u_k = vec / dist;
                    r_anc = anchor_ranges(k) - dist;
                    
                    g_L = g_L - (1 / sigma_s^2) * u_k * r_anc;
                    H_L = H_L + (1 / sigma_s^2) * (u_k * u_k');
                end
                
                % B. 协同邻车测距约束 (动态注入膨胀噪声)
                for i = 1:M
                    p_neigh_est = neighbor_positions(i, :)';
                    vec = p_est - p_neigh_est;
                    dist = max(norm(vec), 1e-6);
                    u_i = vec / dist;
                    r_int = relative_ranges(i) - dist;
                    
                    % 噪声膨胀核心公式: 基础传感器噪声 + 邻车不确定性投影
                    sigma2_z_eff = sigma_z^2 + u_i' * Sigma_neigh_SCI{i} * u_i;
                    
                    g_L = g_L - (1 / sigma2_z_eff) * u_i * r_int;
                    H_L = H_L + (1 / sigma2_z_eff) * (u_i * u_i');
                end
                
                % 求解迭代增量并安全限幅
                step = obj.safe_solve(H_L, g_L, obj.mu);
                max_step_pos = 0.1;
                n_step_pos = norm(step);
                if n_step_pos > max_step_pos
                    step = step * (max_step_pos / n_step_pos);
                end
                s_pos = s_pos - step;
                
                if norm(step) < 1e-4, break; end
                if any(isnan(s_pos)) || any(isinf(s_pos))
                    s_pos = zeros(3, 1);
                    break;
                end
            end
            s_pos = obj.sanitize_vector(s_pos);
            p_converged = p_prior + s_pos;
            
            % --- 3. 提取收敛点处的等效等边际测量信息 ---
            % A. 绝对基站测量信息矩阵 (独立项)
            Lambda_t_anc = zeros(3, 3);
            for k = 1:K
                u_k = (p_converged - anchor_positions(k, :)') / ...
                      max(norm(p_converged - anchor_positions(k, :)'), 1e-6);
                Lambda_t_anc = Lambda_t_anc + (1 / sigma_s^2) * (u_k * u_k');
            end
            
            % B. 相对协同测距测量信息矩阵 (相关项，含不确定性膨胀)
            Lambda_t_int = zeros(3, 3);
            for i = 1:M
                p_neigh_est = neighbor_positions(i, :)';
                u_i = (p_converged - p_neigh_est) / ...
                      max(norm(p_converged - p_neigh_est), 1e-6);
                  
                sigma2_z_eff = sigma_z^2 + u_i' * Sigma_neigh_SCI{i} * u_i;
                Lambda_t_int = Lambda_t_int + (1 / sigma2_z_eff) * (u_i * u_i');
            end
            
            % C. 构建等效信息向量
            lambda_t_anc = Lambda_t_anc * s_pos;
            lambda_t_int = Lambda_t_int * s_pos;
            
            % --- 4. 映射到 15 维全局空间并更新信息状态 ---
            Lambda_anc_full = zeros(15, 15); Lambda_anc_full(1:3, 1:3) = Lambda_t_anc;
            lambda_anc_full = zeros(15, 1);  lambda_anc_full(1:3) = lambda_t_anc;
            
            Lambda_int_full = zeros(15, 15); Lambda_int_full(1:3, 1:3) = Lambda_t_int;
            lambda_int_full = zeros(15, 1);  lambda_int_full(1:3) = lambda_t_int;
            
            % 独立项/相关项累加
            obj.I_indep = obj.sanitize_matrix(obj.I_indep + Lambda_anc_full);
            obj.I_dep = obj.sanitize_matrix(obj.omega_self * obj.I_dep + Lambda_int_full);
            
            % --- 5. 状态全量后延解算与流形回射 ---
            Sigma_post = obj.safe_inv(obj.I_indep + obj.I_dep);
            delta_theta = Sigma_post * (lambda_anc_full + lambda_int_full);
            
            obj.state.p = obj.state.p + delta_theta(1:3);
            obj.state.v = obj.state.v + delta_theta(4:6);
            obj.state.a = obj.state.a + delta_theta(7:9);
            obj.state.R = obj.robust_orthonormalize(obj.state.R * obj.so3_exp_safe(delta_theta(10:12)));
            obj.state.omega = obj.state.omega + delta_theta(13:15);
        end
        
        %% 退化的单车 UWB 位置更新 (Anchor Only)
        function obj = update_uwb_anchor_only(obj, anchor_ranges, anchor_positions, sigma_s)
            K = length(anchor_ranges);
            s_pos = zeros(3, 1);
            p_prior = obj.state.p;
            
            for iter = 1:5
                p_est = p_prior + s_pos;
                r = zeros(K, 1);
                H = zeros(K, 3);
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
            
            obj.I_indep = obj.sanitize_matrix(obj.I_indep + Lambda_full);
            Sigma_post = obj.safe_inv(obj.I_indep + obj.I_dep);
            delta_theta = Sigma_post * lambda_full;
            
            obj.state.p = obj.state.p + delta_theta(1:3);
            obj.state.v = obj.state.v + delta_theta(4:6);
            obj.state.a = obj.state.a + delta_theta(7:9);
            obj.state.R = obj.robust_orthonormalize(obj.state.R * obj.so3_exp_safe(delta_theta(10:12)));
        end
    end
    
    %% 数值安全防护辅助函数 (私有)
    methods (Access = private)
        function R_orth = robust_orthonormalize(~, R)
            if any(isnan(R(:))) || any(isinf(R(:)))
                R_orth = eye(3);
                return;
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