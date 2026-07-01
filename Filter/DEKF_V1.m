classdef DEKF_V1
    % DEKF_V1: 15维经典分布式扩展卡尔曼滤波器类 (DEKF + 经典 CI 协同定位基准)
    % 严格对应 15维切空间状态建模，采用经典的全维协方差 CI 放大，不含状态联合与独立/相关拆分。
    
    properties
        id              % 节点ID (1, 2, 3, 4...)
        state           % 状态结构体: .p(3x1), .v(3x1), .a(3x1), .R(3x3), .omega(3x1)
        P               % 15x15 全维系统误差状态协方差矩阵
        Q               % 15x15 过程噪声协方差矩阵
        Sigma_a         % 3x3 加速度计噪声协方差
        Sigma_w         % 3x3 陀螺仪噪声协方差
        g_vec           % 3x1 重力向量 [0; 0; -9.81]
        tau             % IMU采样周期 (秒)
        mu              % 数值阻尼参数 (防御矩阵奇异)
        omega_self      % 本车 CI 权重系数 (内部强制赋值 0.8)
    end
    
    methods
        %% 构造函数
        function obj = DEKF_V1(id, init_state, init_cov, Q_matrix, Sigma_a, Sigma_w, tau, ~)
            obj.id = id;
            obj.state = init_state;
            obj.state.R = obj.robust_orthonormalize(init_state.R);
            
            % 保证初始协方差对称正定，防 NaN
            init_cov_safe = init_cov + 1e-10 * eye(15);
            obj.P = 0.5 * (init_cov_safe + init_cov_safe');
            
            obj.Q = Q_matrix;
            obj.Sigma_a = Sigma_a;
            obj.Sigma_w = Sigma_w;
            obj.tau = tau;
            obj.g_vec = [0; 0; -9.81];
            obj.mu = 1e-6; % 数值阻尼
            
            % 经典 DEKF-CI 基准参数配置 
            obj.omega_self = 0.95; 
        end
        
        %% 对偶变量重置兼容接口 (经典 DEKF 无对偶变量，此接口仅作调用兼容)
        function obj = reset_dual_variables(obj)
            % 保持与 DMLKF 调用链路一致
        end
        
        %% 1.2 状态与协方差高频传播 (Prediction - 100Hz)
        function obj = predict(obj)
            % 提取当前名义状态
            p_old = obj.state.p;
            v_old = obj.state.v;
            a_old = obj.state.a;
            R_old = obj.state.R;
            w_old = obj.state.omega;
            t_step = obj.tau;
            
            % --- 1. 名义状态传播 (同DMLKF运动学建模) ---
            obj.state.p = p_old + t_step * v_old + (t_step^2 / 2) * a_old;
            obj.state.v = v_old + t_step * a_old;
            obj.state.a = a_old;
            
            exp_w = obj.so3_exp_safe(t_step * w_old);
            obj.state.R = obj.robust_orthonormalize(R_old * exp_w);
            obj.state.omega = w_old;
            
            % --- 2. 构造误差传播 Jacobian A_t (DMLKF Eq. 15) ---
            A_t = eye(15);
            A_t(1:3, 4:6)   = t_step * eye(3);
            A_t(1:3, 7:9)   = (t_step^2 / 2) * eye(3);
            A_t(4:6, 7:9)   = t_step * eye(3);
            
            A_t(10:12, 10:12) = obj.so3_exp_safe(-t_step * w_old);
            A_t(10:12, 13:15) = t_step * obj.so3_right_jacobian_safe(t_step * w_old);
            
            % --- 3. 协方差传播 ---
            P_pred = A_t * obj.P * A_t' + obj.Q;
            
            % 数值净化与对称化保护
            P_pred = 0.5 * (P_pred + P_pred');
            obj.P = obj.sanitize_matrix(P_pred);
        end
        
        %% 2.2 高频 15D 局部 IMU 更新 (IMU Update - 100Hz)
        function obj = update_imu(obj, raw_acc, raw_gyro, bias_a, bias_w)
            % 预修正测量值
            acc_tilde = raw_acc - bias_a;
            gyro_tilde = raw_gyro - bias_w;
            
            % 1. 计算测量残差 r
            r_acc = acc_tilde - obj.state.R' * (obj.state.a - obj.g_vec);
            r_gyro = gyro_tilde - obj.state.omega;
            r = [r_acc; r_gyro];
            
            % 2. 构造全维 EKF 观测雅可比 (正号，标准 EKF 测量偏导形式，提供负反馈校正)
            H_IMU = zeros(6, 15);
            H_IMU(1:3, 7:9)   = obj.state.R';
            H_IMU(1:3, 10:12) = obj.skew_matrix(obj.state.R' * (obj.state.a - obj.g_vec));
            H_IMU(4:6, 13:15) = eye(3);
            
            % 3. EKF 经典后验更新
            R_IMU = blkdiag(obj.Sigma_a, obj.Sigma_w);
            S_IMU = H_IMU * obj.P * H_IMU' + R_IMU;
            S_IMU_reg = S_IMU + obj.mu * eye(6); % 阻尼保护
            
            K_gain = obj.P * H_IMU' / S_IMU_reg;
            
            delta_theta = K_gain * r;
            
            % 状态更新回射
            obj.state.p = obj.state.p + delta_theta(1:3);
            obj.state.v = obj.state.v + delta_theta(4:6);
            obj.state.a = obj.state.a + delta_theta(7:9);
            obj.state.R = obj.robust_orthonormalize(obj.state.R * obj.so3_exp_safe(delta_theta(10:12)));
            obj.state.omega = obj.state.omega + delta_theta(13:15);
            
            % Joseph 形式后验协方差更新，提供强数值安全性
            I_15 = eye(15);
            ImKH = I_15 - K_gain * H_IMU;
            obj.P = ImKH * obj.P * ImKH' + K_gain * R_IMU * K_gain';
            
            % 对称化与数值净化
            obj.P = 0.5 * (obj.P + obj.P');
            obj.P = obj.sanitize_matrix(obj.P);
        end
        
        %% 3.0 广播数据边缘化接口 (适配外部 DFilter_compare.m 调用)
        function [p_est, Sigma_pos] = get_marginalized_position_info(obj)
            % 经典 DEKF 无分裂信息矩阵，直接输出 3D 估计位置和 3x3 位置协方差块
            p_est = obj.state.p;
            Sigma_pos = obj.P(1:3, 1:3);
        end
        
        %% 3.4 堆叠式分布式 UWB 更新与经典 CI 融合 (UWB Update - 10Hz)
        function obj = apply_uwb_update(obj, ~, anchor_ranges, anchor_positions, ...
                                        neighbor_ids, neighbor_positions, ...
                                        neighbor_Sigma_pos, relative_ranges, sigma_s, sigma_z)
            % 注：形参一 (s_star) 仅为对齐 DMLKF 接口设计，此处不作处理
            K = length(anchor_ranges);
            M = length(neighbor_ids);
            L = K + M; % 堆叠观测方程维度
            
            if L == 0
                return;
            end
            
            % --- Step 1: 经典 CI 缩放本车全维协方差 (未观测状态协方差也在此被动放大) ---
            P_scaled = (1.0 / obj.omega_self) * obj.P;

            % --- Step 2: 构造堆叠测距观测残差、雅可比与测量等效噪声 ---
            r = zeros(L, 1);
            H = zeros(L, 15);
            R_joint = zeros(L, L);
            
            p_self = obj.state.p;
            
            % A. 基站相对距离部分 (绝对观测，独立不引入交叉相关)
            for k = 1:K
                vec = p_self - anchor_positions(k, :)';
                dist = max(norm(vec), 1e-6);
                u_k = vec / dist;
                
                r(k) = anchor_ranges(k) - dist;
                H(k, 1:3) = u_k'; % 仅作用于 3D 位置维
                R_joint(k, k) = sigma_s^2; 
            end
            
            % B. 邻车协同相对测距部分 (引入等效噪声建模及经典 CI 缩放)
            for m = 1:M
                idx = K + m;
                p_neigh = neighbor_positions(m, :)';
                
                vec = p_self - p_neigh;
                dist = max(norm(vec), 1e-6);
                u_j = vec / dist;
                
                rel_val = relative_ranges(m);
                r(idx) = rel_val - dist;
                H(idx, 1:3) = u_j'; % 仅作用于 3D 位置维
                
                % 经典 CI 放大邻车位置协方差，补偿未知交叉相关性
                P_neigh_pos = neighbor_Sigma_pos{m};
                P_neigh_scaled = (1.0 / (1.0 - 0.8)) * P_neigh_pos;

                % 等效测量噪声 = 白噪声 + 邻车位置不确定度在视线矢量上的投影
                R_joint(idx, idx) = sigma_z^2 + u_j' * P_neigh_scaled * u_j;
            end
            
            % --- Step 3: 经典卡尔曼后验结算 ---
            S = H * P_scaled * H' + R_joint;
            S_reg = S + obj.mu * eye(L); % 数值保护
            
            K_gain = P_scaled * H' / S_reg;
            
            delta_theta = K_gain * r;
            
            % 步长限制数值保护：防止非线性几何突变导致大偏角发散
            max_pos_step = 0.5;
            norm_pos = norm(delta_theta(1:3));
            if norm_pos > max_pos_step
                delta_theta(1:3) = delta_theta(1:3) * (max_pos_step / norm_pos);
            end
            
            % 状态更新回射
            obj.state.p = obj.state.p + delta_theta(1:3);
            obj.state.v = obj.state.v + delta_theta(4:6);
            obj.state.a = obj.state.a + delta_theta(7:9);
            obj.state.R = obj.robust_orthonormalize(obj.state.R * obj.so3_exp_safe(delta_theta(10:12)));
            obj.state.omega = obj.state.omega + delta_theta(13:15);
            
            % Joseph 形式更新后验协方差，防止浮点数精度截断导致矩阵非正定
            I_15 = eye(15);
            ImKH = I_15 - K_gain * H;
            obj.P = ImKH * P_scaled * ImKH' + K_gain * R_joint * K_gain';
            
            % 数值收尾净化
            obj.P = 0.5 * (obj.P + obj.P');
            obj.P = obj.sanitize_matrix(obj.P);
        end
    end
    
    %% 数值安全与李代数辅助计算函数 (封闭保护)
    methods (Access = private)
        %% SVD 姿态矩阵正交归一化 (防退化发散)
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
        
        %% 矩阵数值净化与对称化 (彻底清除 NaN 与 Inf 隐患)
        function M_out = sanitize_matrix(~, M_in)
            M_out = M_in;
            M_out(isnan(M_out)) = 0;
            M_out(isinf(M_out)) = 0;
            M_out = 0.5 * (M_out + M_out');
        end
        
        %% 3D 向量反对称矩阵构造
        function skew_R = skew_matrix(~, v)
            skew_R = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
        end
    end
end