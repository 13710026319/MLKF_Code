classdef DIEKF
    % DIEKF: 9维经典误差状态分布式迭代EKF类 (纯CI批量迭代融合版)
    % CI: 先验协方差只放大位置状态
    % 更新步只对整体R当中相对测距部分进行等效噪声放大
    % 状态结构：.p (3x1), .v (3x1), .R (3x3)，IMU作为传播输入，UWB测量进行流形迭代更新
    
    properties
        id              % 节点ID (1, 2, 3, 4...)
        state           % 状态结构体: .p(3x1), .v(3x1), .R(3x3)
        P               % 9x9 本地状态误差协方差矩阵
        Q               % 9x9 离散单步系统过程噪声协方差矩阵 (严格对齐预设)
        g_vec           % 3D 重力加速度常数 (通常为 [0; 0; -9.81])
        tau             % IMU采样预测周期 dt (秒)
        omega_self      % 本车CI权重
        omega_neigh     % 邻居CI权重
        mu              % 高斯-牛顿正则化参数 (防御NaN)
    end
    
    methods
        %% 构造函数 (9维经典结构)
        function obj = DIEKF(id, init_state, init_cov, tau)
            obj.id = id;
            obj.state.p = init_state.p;
            obj.state.v = init_state.v;
            obj.state.R = obj.robust_orthonormalize(init_state.R);
            
            % 严格对齐通道：提取 15D 的位置(1:3)、速度(4:6)以及姿态(10:12)协方差重组为 9D 协方差
            if size(init_cov, 1) == 15
                obj.P = blkdiag(init_cov(1:3, 1:3), init_cov(4:6, 4:6), init_cov(10:12, 10:12));
            else
                obj.P = init_cov;
            end
            obj.P = obj.P + 1e-10 * eye(9); % 保证正定防 NaN
            
            Q_sigmas_9d = [ ...
                0.0005 * ones(1,3), ... % 位置过程噪声标准差 
                0.005 * ones(1,3), ...  % 速度过程噪声标准差 
                0.0005 * ones(1,3) ...  % 姿态过程噪声标准差 
                ];
            obj.Q = diag(Q_sigmas_9d.^2);
            
            obj.tau = tau;
            obj.omega_self = 0.8;
            obj.omega_neigh = 1- obj.omega_self;
            obj.g_vec = [0; 0; -9.81];
            obj.mu = 1e-4;
        end
        
        %% 【接口兼容哑方法】
        function obj = reset_dual_variables(obj)
        end
        function s_admm = solve_primal_public(obj, s_admm_init, varargin)
            s_admm = s_admm_init;
        end
        function obj = update_dual(obj, varargin)
        end
        
        %% 1.2 标称状态与误差协方差时间传播 (Prediction Step - 100Hz)
        function obj = predict(obj, imu_acc, imu_gyro)
            % 提取当前状态
            p_t = obj.state.p;
            v_t = obj.state.v;
            R_t = obj.state.R;
            dt = obj.tau;
            
            % 1. 标称状态时间积分 (IMU作为系统输入驱动)
            acc_nav = R_t * imu_acc + obj.g_vec; % 转换至导航系并叠加重力
            obj.state.p = p_t + dt * v_t + 0.5 * dt^2 * acc_nav;
            obj.state.v = v_t + dt * acc_nav;
            obj.state.R = obj.robust_orthonormalize(R_t * obj.so3_exp_safe(dt * imu_gyro));
            
            % 2. 构造 9 维误差状态转移矩阵 A (对应经典惯导误差方程)
            A = eye(9);
            A(1:3, 4:6) = dt * eye(3);                      % \delta p <- \delta v
            A(4:6, 7:9) = -dt * R_t * obj.skew_matrix(imu_acc); % \delta v <- \delta \phi (比力驱动)
            A(7:9, 7:9) = obj.so3_exp_safe(-dt * imu_gyro);  % \delta \phi <- \delta \phi
            
            % 3. 联合协方差传播 (对齐离散噪声更新)
            obj.P = A * obj.P * A' + obj.Q;
            obj.P = 0.5 * (obj.P + obj.P'); % 数值对称正定保护
        end
        
        %% 3.0 UWB 状态与协方差投影 (边缘化输出接口 - 邻居车辆调用)
        function [p_est, Sigma_pos] = get_marginalized_position_info(obj)
            p_est = obj.state.p;
            Sigma_pos = obj.P(1:3, 1:3);
            Sigma_pos = 0.5 * (Sigma_pos + Sigma_pos');
        end
        
        %% 【核心重构】标准 CI 批量迭代更新步 (IEKF 姿态/位置流形迭代求解)
        function obj = apply_uwb_update(obj, ~, anchor_ranges, anchor_positions, ...
                                        neighbor_ids, neighbor_positions, ...
                                        neighbor_Sigma_pos, relative_ranges, sigma_s, sigma_z)
            K = length(anchor_ranges);
            M = length(neighbor_ids);
            
            if (K + M) == 0, return; end
            
            % --- 1. 本车先验协方差 CI 比例膨胀 ---
            D_factor = 1 / sqrt(obj.omega_self);
            D = diag([D_factor * ones(1, 3), ones(1, 3), ones(1, 3)]); % 构造 9x9 对角缩放矩阵
            P_scaled = D * obj.P * D; % 两侧对称相乘，严格保持对称正定性
            P_scaled = 0.5 * (P_scaled + P_scaled');
            
            % --- 2. 基于标称先验初始化迭代变量 ---
            dx = zeros(9, 1); % 先验流形切空间中的迭代扰动变量
            y_meas = [anchor_ranges; relative_ranges];

            % 迭代上限通常设为 10 次
            max_iekf_iter = 10;
            for iter = 1:max_iekf_iter
                % A. 计算当前迭代工作点标称状态 (流形指数更新)
                p_curr = obj.state.p + dx(1:3);
                
                % B. 计算当前迭代工作点处的测量估值与雅可比
                h_val = zeros(K + M, 1);
                H = zeros(K + M, 9);
                R_list = zeros(K + M, 1);
                
                % 基站测距
                for k = 1:K
                    c_k = anchor_positions(k, :)';
                    dist = norm(p_curr - c_k);
                    if dist < 1e-6, dist = 1e-6; end
                    h_val(k) = dist;
                    u_k = (p_curr - c_k) / dist;
                    H(k, 1:3) = u_k';
                    R_list(k) = sigma_s^2;
                end
                
                % 协同测距 (基于当前更新的位置方向动态计算 CI 噪声膨胀)
                for idx = 1:M
                    p_j = neighbor_positions(idx, :)';
                    dist = norm(p_curr - p_j);
                    if dist < 1e-6, dist = 1e-6; end
                    h_val(K + idx) = dist;
                    u_j = (p_curr - p_j) / dist;
                    H(K + idx, 1:3) = u_j';
                    
                    % 动态噪声膨胀：基础测距方差 + 膨胀后的邻车不确定性投影
                    Sigma_pos_j_CI = (1 / obj.omega_neigh) * neighbor_Sigma_pos{idx};
                    R_list(K + idx) = sigma_z^2 + u_j' * Sigma_pos_j_CI * u_j;
                end
                
                R_cov = diag(R_list);
                y_err = y_meas - h_val;
                
                % C. IEKF 标准迭代滤波更新步骤
                S = H * P_scaled * H' + R_cov;
                S_reg = S + obj.mu * eye(size(S));
                K_gain = P_scaled * H' / S_reg;
                if any(isnan(K_gain(:))) || any(isinf(K_gain(:)))
                    K_gain = P_scaled * H' * pinv(S);
                end
                
                % 在先验约束下计算下一步迭代误差状态量 (对齐 IEKF 扰动形式)
                dx_new = K_gain * (y_err + H * dx);
                
                % 增量极其微小时提前终止
                if norm(dx_new - dx) < 1e-4
                    dx = dx_new;
                    break;
                end
                dx = dx_new;
            end
            
            % --- 3. 最终流形回馈纠正标称状态 ---
            dx = obj.sanitize_vector(dx);
            obj.state.p = obj.state.p + dx(1:3);
            obj.state.v = obj.state.v + dx(4:6);
            obj.state.R = obj.robust_orthonormalize(obj.state.R * obj.so3_exp_safe(dx(7:9)));
            
            % --- 4. 在最终收敛工作点处更新后验误差协方差 (标准 CI 格式) ---
            obj.P = (eye(9) - K_gain * H) * P_scaled;
            obj.P = 0.5 * (obj.P + obj.P');
            obj.P = obj.sanitize_matrix(obj.P);
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
            Lambda_full = zeros(9, 9); Lambda_full(1:3, 1:3) = Lambda_anc;
            lambda_full = zeros(9, 1);  lambda_full(1:3) = lambda_anc;
            
            I_post = obj.safe_inv(obj.P) + Lambda_full;
            Sigma_post = obj.safe_inv(I_post);
            delta_theta = Sigma_post * lambda_full;
            
            obj.state.p = obj.state.p + delta_theta(1:3);
            obj.state.v = obj.state.v + delta_theta(4:6);
            obj.state.R = obj.robust_orthonormalize(obj.state.R * obj.so3_exp_safe(delta_theta(7:9)));
            obj.P = obj.sanitize_matrix(Sigma_post);
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