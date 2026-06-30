classdef DEKF
    % DEKF: 9维经典误差状态EKF类 (纯CI批量融合版)
    % CI: 先验协方差只放大位置状态(速度状态为了数值而放大了1.05倍)
    % 更新步只对整体R当中相对测距部分进行等效噪声放大
    % 状态结构：.p (3x1), .v (3x1), .R (3x3)，IMU作为传播输入，无高频IMU独立更新步
    
    properties
        id              % 节点ID (1, 2, 3, 4...)
        state           % 状态结构体: .p(3x1), .v(3x1), .R(3x3)
        P               % 9x9 本地状态误差协方差矩阵
        Q               % 9x9 离散单步系统过程噪声协方差矩阵 (严格对齐Q_single)
        g_vec           % 3D 重力加速度常数 (通常为 [0; 0; -9.81])
        tau             % IMU采样预测周期 dt (秒)
        omega_self      % 本车CI权重
        omega_neigh     % 邻居CI权重
    end
    
    methods
        %% 构造函数 (9维经典结构)
        function obj = DEKF(id, init_state, init_cov, tau)
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
            
            % 直接接收并使用外部传入的 9D 过程噪声矩阵 Q_9d
            Q_sigmas_9d = [ ...
                0.001 * ones(1,3), ... % 位置过程噪声标准差 
                0.01 * ones(1,3), ...  % 速度过程噪声标准差 
                0.001 * ones(1,3) ...  % 姿态过程噪声标准差 
                ];
            obj.Q = diag(Q_sigmas_9d.^2);
            
            obj.tau = tau;
            obj.omega_self = 0.8;
            obj.omega_neigh = 1 - obj.omega_self;
            obj.g_vec = [0; 0; -9.81];
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
        
        %% 【核心重构】标准 CI 批量更新步 (仅处理 UWB 测距，包含基站与邻车)
        function obj = apply_uwb_update(obj, ~, anchor_ranges, anchor_positions, ...
                                        neighbor_ids, neighbor_positions, ...
                                        neighbor_Sigma_pos, relative_ranges, sigma_s, sigma_z)
            K = length(anchor_ranges);
            M = length(neighbor_ids);
            
            if (K + M) == 0, return; end
            
            % --- 1. 本车先验协方差 子空间 CI 比例膨胀 (仅缩放位置通道，100%保护速度与姿态) ---
            D_factor = 1 / sqrt(obj.omega_self);
            D = diag([D_factor * ones(1, 3), 1.05*ones(1, 3), ones(1, 3)]); % 构造 9x9 对角缩放矩阵
            P_scaled = D * obj.P * D; % 两侧对称相乘，严格保持对称正定性
            P_scaled = 0.5 * (P_scaled + P_scaled');
            
            % --- 2. 构建批量测量向量、等效噪声矩阵与观测雅可比 ---
            y_meas = [anchor_ranges; relative_ranges];
            h_val = zeros(K + M, 1);
            H = zeros(K + M, 9);
            R_list = zeros(K + M, 1);
            
            % A. 绝对基站观测处理
            p_i = obj.state.p;
            for k = 1:K
                c_k = anchor_positions(k, :)';
                dist = norm(p_i - c_k);
                if dist < 1e-6, dist = 1e-6; end
                h_val(k) = dist;
                u_k = (p_i - c_k) / dist;
                H(k, 1:3) = u_k';
                R_list(k) = sigma_s^2;
            end
            

            for idx = 1:M
                p_j = neighbor_positions(idx, :)';
                dist = norm(p_i - p_j);
                if dist < 1e-6, dist = 1e-6; end
                h_val(K + idx) = dist;
                u_j = (p_i - p_j) / dist;
                H(K + idx, 1:3) = u_j';
                
                % 测量噪声膨胀：基础测距方差 + 膨胀后的邻车不确定性投影
                Sigma_pos_j_CI = (1 / obj.omega_neigh) * neighbor_Sigma_pos{idx};
                R_list(K + idx) = sigma_z^2 + u_j' * Sigma_pos_j_CI * u_j;
            end
            
            R_cov = diag(R_list);
            y_err = y_meas - h_val;
            
            % --- 3. 批量经典 EKF 更新解算 ---
            S = H * P_scaled * H' + R_cov;
            
            % 引入安全阻尼防奇异求逆
            S_reg = S + 1e-11 * eye(size(S));
            K_gain = P_scaled * H' / S_reg;
            if any(isnan(K_gain(:))) || any(isinf(K_gain(:)))
                K_gain = P_scaled * H' * pinv(S);
            end
            
            % 计算 9 维误差状态增量
            dx = K_gain * y_err;
            
            % 更新状态误差协方差 P (CI规则)
            obj.P = (eye(9) - K_gain * H) * P_scaled;
            obj.P = 0.5 * (obj.P + obj.P');
            obj.P = obj.sanitize_matrix(obj.P);
            
            % --- 4. 标称状态 9维 流形回馈纠正 ---
            obj.state.p = obj.state.p + dx(1:3);
            obj.state.v = obj.state.v + dx(4:6);
            obj.state.R = obj.robust_orthonormalize(obj.state.R * obj.so3_exp_safe(dx(7:9)));
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