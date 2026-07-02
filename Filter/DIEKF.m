classdef DIEKF
    % DIEKF: 15维分布式迭代扩展卡尔曼滤波器类
    % 采用固定先验切空间 MAP 迭代架构，含严格姿态右雅可比映射与全维等比例步长保护
    
    properties
        id              % 节点ID (1, 2, 3, 4...)
        state           % 状态结构体: .p(3x1), .v(3x1), .a(3x1), .R(3x3), .omega(3x1)
        P               % 15x15 状态误差协方差矩阵
        Q               % 15x15 过程噪声协方差矩阵
        Sigma_a         % 3x3 加速度计噪声协方差
        Sigma_w         % 3x3 陀螺仪噪声协方差
        g_vec           % 3x1 重力向量
        tau             % IMU采样周期 (秒)
        mu              % 岭正则化参数 (防御工作点退化)
        omega_self      % 本车CI权重 (类内部强制预设为 0.8)
    end
    
    methods
        %% 构造函数 (签名与各版本严格一致)
        function obj = DIEKF(id, init_state, init_cov, Q_matrix, Sigma_a, Sigma_w, tau, ~)
            obj.id = id;
            obj.state = init_state;
            obj.state.R = obj.robust_orthonormalize(init_state.R);
            
            % 保证初始协方差不奇异
            init_cov_safe = init_cov + 1e-10 * eye(15);
            obj.P = 0.5 * (init_cov_safe + init_cov_safe');
            
            obj.Q = Q_matrix;
            obj.Sigma_a = Sigma_a;
            obj.Sigma_w = Sigma_w;
            obj.tau = tau;
            obj.g_vec = [0; 0; -9.81];
            obj.mu = 1e-6; 

            obj.omega_self = 0.93;
        end
        
        %% 接口兼容哑方法
        function obj = reset_dual_variables(obj)
        end
        function s_admm = solve_primal_public(obj, s_admm_init, varargin)
            s_admm = s_admm_init;
        end
        function obj = update_dual(obj, varargin)
        end
        
        %% 3.0 位置边缘化数据提取接口 (适配外部 compare.m 脚本)
        function [p_est, Sigma_pos] = get_marginalized_position_info(obj)
            p_est = obj.state.p;
            Sigma_pos = obj.P(1:3, 1:3);
            Sigma_pos = 0.5 * (Sigma_pos + Sigma_pos');
        end
        
        %% 1.2 状态与协方差高频传播 (Prediction Step - 100Hz)
        function obj = predict(obj)
            p_old = obj.state.p;
            v_old = obj.state.v;
            a_old = obj.state.a;
            R_old = obj.state.R;
            w_old = obj.state.omega;
            t_step = obj.tau;
            
            % 名义状态传播 (同DMLKF一致)
            obj.state.p = p_old + t_step * v_old + (t_step^2 / 2) * a_old;
            obj.state.v = v_old + t_step * a_old;
            obj.state.a = a_old;
            
            exp_w = obj.so3_exp_safe(t_step * w_old);
            obj.state.R = obj.robust_orthonormalize(R_old * exp_w);
            obj.state.omega = w_old;
            
            % 构造 Jacobian A_t
            A_t = eye(15);
            A_t(1:3, 4:6)   = t_step * eye(3);
            A_t(1:3, 7:9)   = (t_step^2 / 2) * eye(3);
            A_t(4:6, 7:9)   = t_step * eye(3);
            
            A_t(10:12, 10:12) = obj.so3_exp_safe(-t_step * w_old);
            A_t(10:12, 13:15) = t_step * obj.so3_right_jacobian_safe(t_step * w_old);
            
            % 协方差预测递推
            P_pred = A_t * obj.P * A_t' + obj.Q;
            P_pred = 0.5 * (P_pred + P_pred');
            obj.P = obj.sanitize_matrix(P_pred);
        end
        
        %% 2.2 局部高频 MAP 迭代 IMU 更新 (IMU Update - 100Hz)
        function obj = update_imu(obj, raw_acc, raw_gyro, bias_a, bias_w)
            acc_tilde = raw_acc - bias_a;
            gyro_tilde = raw_gyro - bias_w;
            R_IMU = blkdiag(obj.Sigma_a, obj.Sigma_w);
            
            % 暂存先验状态 (作为固定先验名义原点 check_x)
            check_p = obj.state.p;
            check_v = obj.state.v;
            check_a = obj.state.a;
            check_R = obj.state.R;
            check_omega = obj.state.omega;
            
            % 初始化切空间先验误差状态
            delta_theta_0 = zeros(15, 1);
            
            % 内部非线性迭代求解 MAP 优化
            max_iter = 5;
            for iter = 1:max_iter
                % 2.1 映射获得当前迭代工作点的状态估计
                p_iter = check_p + delta_theta_0(1:3);
                v_iter = check_v + delta_theta_0(4:6);
                a_iter = check_a + delta_theta_0(7:9);
                R_iter = obj.robust_orthonormalize(check_R * obj.so3_exp_safe(delta_theta_0(10:12)));
                w_iter = check_omega + delta_theta_0(13:15);
                
                % 2.2 评估固定先验切空间的转换雅可比 (正向右雅可比 J_r)
                delta_phi_0_curr = delta_theta_0(10:12);
                J_manifold = eye(15);
                J_manifold(10:12, 10:12) = obj.so3_right_jacobian_safe(delta_phi_0_curr);
                
                % 2.3 评估工作点处的局部 EKF 观测雅可比
                H_local = zeros(6, 15);
                H_local(1:3, 7:9)   = R_iter';
                H_local(1:3, 10:12) = obj.skew_matrix(R_iter' * (a_iter - obj.g_vec));
                H_local(4:6, 13:15) = eye(3);
                
                % 2.4 链式相乘获得先验切空间等效观测雅可比
                H_prior = H_local * J_manifold;
                
                % 2.5 评估先验 nominal 状态偏差 (check_x \boxminus x_iter)
                delta_x_prior = zeros(15, 1);
                delta_x_prior(1:3) = check_p - p_iter;
                delta_x_prior(4:6) = check_v - v_iter;
                delta_x_prior(7:9) = check_a - a_iter;
                delta_x_prior(10:12) = obj.so3_log_safe(R_iter' * check_R);
                delta_x_prior(13:15) = check_omega - w_iter;
                
                % 2.6 计算 MAP 等效创新向量
                r_acc = acc_tilde - R_iter' * (a_iter - obj.g_vec);
                r_gyro = gyro_tilde - w_iter;
                r_meas = [r_acc; r_gyro];
                
                r_MAP = r_meas - H_local * delta_x_prior;
                
                % 2.7 滤波更新求解
                S_IMU = H_prior * obj.P * H_prior' + R_IMU;
                S_IMU_reg = S_IMU + obj.mu * eye(6);
                
                K_gain = obj.P * H_prior' / S_IMU_reg;
                delta_theta_0_new = K_gain * r_MAP;
                
                % 检查收敛性并推进迭代
                diff_step = norm(delta_theta_0_new - delta_theta_0);
                delta_theta_0 = delta_theta_0_new;
                
                if diff_step < 1e-5, break; end
            end
            
            % 迭代收敛，状态最终更新
            obj.state.p = check_p + delta_theta_0(1:3);
            obj.state.v = check_v + delta_theta_0(4:6);
            obj.state.a = check_a + delta_theta_0(7:9);
            obj.state.R = obj.robust_orthonormalize(check_R * obj.so3_exp_safe(delta_theta_0(10:12)));
            obj.state.omega = check_omega + delta_theta_0(13:15);
            
            % 一次性 Joseph 形式协方差后验更新
            I_15 = eye(15);
            ImKH = I_15 - K_gain * H_prior;
            obj.P = ImKH * obj.P * ImKH' + K_gain * R_IMU * K_gain';
            
            obj.P = 0.5 * (obj.P + obj.P');
            obj.P = obj.sanitize_matrix(obj.P);
        end
        
        %% 3.4 堆叠协同 UWB MAP 迭代更新与全维 CI 融合 (UWB Update - 10Hz)
        function obj = apply_uwb_update(obj, ~, anchor_ranges, anchor_positions, ...
                                        neighbor_ids, neighbor_positions, ...
                                        neighbor_Sigma_pos, relative_ranges, sigma_s, sigma_z)
            K = length(anchor_ranges);
            M = length(neighbor_ids);
            L = K + M;
            
            if L == 0
                return;
            end
            
            % 1. CI 全维放大本车先验协方差 (未观测成分随之放大，后期通过 MAP 协同约束)
            P_scaled = (1.0 / obj.omega_self) * obj.P;
            
            % 暂存先验状态作为 MAP 固定原点 check_p
            check_p = obj.state.p;
            check_v = obj.state.v;
            check_a = obj.state.a;
            check_R = obj.state.R;
            check_omega = obj.state.omega;
            
            % 初始化先验切空间状态误差
            delta_theta_0 = zeros(15, 1);
            
            % 迭代递推解算
            max_iter = 10;
            for iter = 1:max_iter
                p_iter = check_p + delta_theta_0(1:3);
                
                % 构造堆叠测距残差、雅可比与时变噪声
                r_meas = zeros(L, 1);
                H_local = zeros(L, 15);
                R_joint = zeros(L, L);
                
                % A. 基站相对测距
                for k = 1:K
                    vec = p_iter - anchor_positions(k, :)';
                    dist = max(norm(vec), 1e-6);
                    u_k = vec / dist;
                    
                    r_meas(k) = anchor_ranges(k) - dist;
                    H_local(k, 1:3) = u_k';
                    R_joint(k, k) = sigma_s^2;
                end
                
                % B. 邻车协同测距
                for m = 1:M
                    idx = K + m;
                    p_neigh = neighbor_positions(m, :)';
                    
                    vec = p_iter - p_neigh;
                    dist = max(norm(vec), 1e-6);
                    u_j = vec / dist;
                    
                    rel_val = relative_ranges(m);
                    r_meas(idx) = rel_val - dist;
                    H_local(idx, 1:3) = u_j';
                    
                    % 经典 CI 比例膨胀邻车先验精度
                    P_neigh_pos = neighbor_Sigma_pos{m};
                    denom_protect = max(1.0 - 0.8, 1e-6); % 邻车除零防御
                    P_neigh_scaled = (1.0 / denom_protect) * P_neigh_pos;
                    
                    % 在迭代工作点重新投射等效测量噪声
                    R_joint(idx, idx) = sigma_z^2 + u_j' * P_neigh_scaled * u_j;
                end
                
                % C. 在位置切空间，雅可比链式转换等价于 H_local 本身 (J_manifold 位置块为 I_3)
                H_prior = H_local;
                
                % D. 计算先验位置 nominal 偏差 (check_p - p_iter)
                delta_x_prior = zeros(15, 1);
                delta_x_prior(1:3) = check_p - p_iter;
                
                % E. 计算等效 MAP 残差
                r_MAP = r_meas - H_local * delta_x_prior;
                
                % F. EKF MAP 解算
                S = H_prior * P_scaled * H_prior' + R_joint;
                S_reg = S + obj.mu * eye(L);
                
                K_gain = P_scaled * H_prior' / S_reg;
                % 引入阻尼因子，平抑由于 CI 膨胀带来的迭代震荡
                alpha = 0.8;
                delta_theta_0_new = (1.0 - alpha) * delta_theta_0 + alpha * (K_gain * r_MAP);
                
                % 15维等比例步长阶段，维持复合空间物理一致性
                max_pos_step = 0.15;
                norm_pos = norm(delta_theta_0_new(1:3));
                if norm_pos > max_pos_step
                    delta_theta_0_new = delta_theta_0_new * (max_pos_step / norm_pos);
                end
                
                diff_step = norm(delta_theta_0_new - delta_theta_0);
                delta_theta_0 = delta_theta_0_new;
                
                if diff_step < 1e-5, break; end
            end
            
            % 迭代收敛，Nominal 状态更新
            obj.state.p = check_p + delta_theta_0(1:3);
            obj.state.v = check_v + delta_theta_0(4:6);
            obj.state.a = check_a + delta_theta_0(7:9);
            obj.state.R = obj.robust_orthonormalize(check_R * obj.so3_exp_safe(delta_theta_0(10:12)));
            obj.state.omega = check_omega + delta_theta_0(13:15);
            
            % 协方差 Joseph 形式后延更新
            I_15 = eye(15);
            ImKH = I_15 - K_gain * H_prior;
            obj.P = ImKH * P_scaled * ImKH' + K_gain * R_joint * K_gain';
            
            obj.P = 0.5 * (obj.P + obj.P');
            obj.P = obj.sanitize_matrix(obj.P);
        end
    end
    
    %% 流形数值防护计算辅助函数 (私有)
    methods (Access = private)
        %% SVD 归一化旋转矩阵 (防退化NaN)
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
        
        %% 李群 SO(3) 安全对数映射
        function phi = so3_log_safe(obj, R)
            tr = trace(R);
            val = (tr - 1) / 2;
            val = max(-1, min(1, val));
            theta = acos(val);
            R_diff = R - R';
            if theta < 1e-6
                phi = obj.unskew(R_diff) / 2;
            else
                phi = (theta / (2 * sin(theta))) * obj.unskew(R_diff);
            end
        end
        
        %% 李群 SO(3) 反对称提取
        function phi = unskew(~, phi_skew)
            phi = [phi_skew(3, 2); phi_skew(1, 3); phi_skew(2, 1)];
        end
        
        %% 李群 SO(3) 安全指数映射
        function R = so3_exp_safe(obj, phi)
            theta = norm(phi);
            phi_skew = obj.skew_matrix(phi);
            if theta < 1e-6
                R = eye(3) + phi_skew;
            else
                R = eye(3) + (sin(theta)/theta) * phi_skew + ((1 - cos(theta))/theta^2) * (phi_skew * phi_skew);
            end
        end
        
        %% 李群 SO(3) 安全右雅可比
        function Jr = so3_right_jacobian_safe(obj, phi)
            theta = norm(phi);
            phi_skew = obj.skew_matrix(phi);
            if theta < 1e-3
                Jr = eye(3) - 0.5 * phi_skew + (1/6) * (phi_skew * phi_skew);
            else
                Jr = eye(3) - ((1 - cos(theta))/theta^2) * phi_skew + ((theta - sin(theta))/theta^3) * (phi_skew * phi_skew);
            end
        end
        
        %% 矩阵净化与强制对称
        function M_out = sanitize_matrix(~, M_in)
            M_out = M_in;
            M_out(isnan(M_out)) = 0;
            M_out(isinf(M_out)) = 0;
            M_out = 0.5 * (M_out + M_out');
        end
        
        %% 向量净化
        function v_out = sanitize_vector(~, v_in)
            v_out = v_in;
            v_out(isnan(v_out)) = 0;
            v_out(isinf(v_out)) = 0;
        end
        
        %% 3D 向量反对称矩阵构造
        function skew_R = skew_matrix(~, v)
            skew_R = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
        end
    end
end