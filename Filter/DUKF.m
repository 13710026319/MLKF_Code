classdef DUKF
    % DUKF: 15维分布式无迹卡尔曼滤波器类 (Lie-Group UKF + 经典 CI 全维膨胀)
    % 严格对应 15维切空间状态建模，采用迭代法解算流形均值，含动态邻车自适应与强鲁棒数值保护
    
    properties
        id              % 节点ID (1, 2, 3, 4...)
        state           % 状态结构体: .p(3x1), .v(3x1), .a(3x1), .R(3x3), .omega(3x1)
        P               % 15x15 全维误差协方差矩阵
        Q               % 15x15 过程噪声协方差矩阵
        Sigma_a         % 3x3 加速度计噪声协方差
        Sigma_w         % 3x3 陀螺仪噪声协方差
        g_vec           % 3x1 重力向量
        tau             % IMU采样周期 (秒)
        mu              % 岭正则化参数 (防御工作点退化)
        omega_self      % 本车CI权重
        
        % UT 无迹变换权重相关超参数
        alpha           % 无迹点分布范围因子 (1e-4 ~ 1)
        beta            % 引入分布先验因子 (高斯分布下默认为 2.0)
        kappa           % 辅助缩放因子 (通常置为 0)
    end
    
    methods
        %% 构造函数 (签名与各版本严格一致)
        function obj = DUKF(id, init_state, init_cov, Q_matrix, Sigma_a, Sigma_w, tau, omega_self)
            obj.id = id;
            obj.state = init_state;
            obj.state.R = obj.robust_orthonormalize(init_state.R);
            
            % 保证初始协方差不奇异，防 NaN
            init_cov_safe = init_cov + 1e-10 * eye(15);
            obj.P = 0.5 * (init_cov_safe + init_cov_safe');
            
            obj.Q = Q_matrix;
            obj.Sigma_a = Sigma_a;
            obj.Sigma_w = Sigma_w;
            obj.tau = tau;
            obj.g_vec = [0; 0; -9.81];
            obj.mu = 1e-6; 
            
            % 提取计算权重时的超参数，在构造函数处赋值
            obj.alpha = 1e-3;
            obj.beta = 2.0;
            obj.kappa = 0;
            
            % CI权重配置 (默认 0.88，支持外部覆盖输入)
            if nargin >= 8 && ~isempty(omega_self)
                obj.omega_self = omega_self;
            else
                obj.omega_self = 0.88;
            end
        end
        
        %% 接口兼容哑方法
        function obj = reset_dual_variables(obj)
        end
        
        %% 1.2 状态与协方差高频无迹递推 (Prediction Step - 100Hz)
        function obj = predict(obj)
            % 1. 获取 15 维 UT 权重与缩放参数
            [Wm, Wc, lambda] = obj.calculate_ut_weights(15);
            
            % 2. 协方差数值保护后进行 Cholesky 分解
            S = obj.robust_chol((15 + lambda) * obj.P);
            
            % 3. 生成 31 个全维无迹点
            X = cell(31, 1);
            X{1} = obj.state;
            for i = 1:15
                X{1+i}  = manifold_add(obj.state, S(:, i));
                X{16+i} = manifold_add(obj.state, -S(:, i));
            end
            
            % 4. 每个无迹点通过非线性运动学方程向前传播
            X_prop = cell(31, 1);
            for i = 1:31
                X_prop{i} = obj.kinematic_propagate(X{i});
            end
            
            % 5. 迭代重投影法解算流形本征均值 (Intrinsic Riemannian Mean)
            x_bar = X_prop{1}; % 初始猜测
            max_iter_mean = 10;
            for iter_mean = 1:max_iter_mean
                Delta_x = zeros(15, 1);
                for i = 1:31
                    e_i = manifold_sub(X_prop{i}, x_bar);
                    Delta_x = Delta_x + Wm(i) * e_i;
                end
                x_bar = manifold_add(x_bar, Delta_x);
                if norm(Delta_x) < 1e-5
                    break;
                end
            end
            obj.state = x_bar;
            
            % 6. 计算先验协方差矩阵并加过程噪声
            P_new = zeros(15, 15);
            for i = 1:31
                dx = manifold_sub(X_prop{i}, obj.state);
                P_new = P_new + Wc(i) * (dx * dx');
            end
            P_new = P_new + obj.Q;
            
            P_pred = 0.5 * (P_new + P_new');
            obj.P = obj.sanitize_matrix(P_pred);
        end
        
        %% 2.2 高频局部 IMU 更新 (IMU Update - 100Hz)
        function obj = update_imu(obj, raw_acc, raw_gyro, bias_a, bias_w)
            acc_tilde = raw_acc - bias_a;
            gyro_tilde = raw_gyro - bias_w;
            R_IMU = blkdiag(obj.Sigma_a, obj.Sigma_w);
            
            % 1. 生成 15 维无迹点
            [Wm, Wc, lambda] = obj.calculate_ut_weights(15);
            S = obj.robust_chol((15 + lambda) * obj.P);
            
            X = cell(31, 1);
            X{1} = obj.state;
            for i = 1:15
                X{1+i}  = manifold_add(obj.state, S(:, i));
                X{16+i} = manifold_add(obj.state, -S(:, i));
            end
            
            % 2. 无迹点通过非线性 IMU 观测方程投射
            Z = zeros(6, 31);
            for i = 1:31
                Z(:, i) = [X{i}.R' * (X{i}.a - obj.g_vec); X{i}.omega];
            end
            
            % 3. 统计预测观测均值
            z_hat = zeros(6, 1);
            for i = 1:31
                z_hat = z_hat + Wm(i) * Z(:, i);
            end
            
            % 4. 统计自协方差 P_zz 与 互协方差 P_xz
            P_zz = zeros(6, 6);
            P_xz = zeros(15, 6);
            for i = 1:31
                dz = Z(:, i) - z_hat;
                dx = manifold_sub(X{i}, obj.state);
                P_zz = P_zz + Wc(i) * (dz * dz');
                P_xz = P_xz + Wc(i) * (dx * dz');
            end
            P_zz = P_zz + R_IMU;
            P_zz = 0.5 * (P_zz + P_zz');
            
            % 5. 安全求解无迹卡尔曼增益
            K_gain = obj.safe_solve(P_zz', P_xz')';
            
            % 6. 后验状态更新回射与协方差更新
            z_meas = [acc_tilde; gyro_tilde];
            delta_theta = K_gain * (z_meas - z_hat);
            
            % 全维等比例步长截断防护
            max_step = 0.5;
            norm_step = norm(delta_theta(1:3));
            if norm_step > max_step
                delta_theta = delta_theta * (max_step / norm_step);
            end
            
            obj.state = manifold_add(obj.state, delta_theta);
            
            % 更新协方差，并附加对称净化
            P_new = obj.P - K_gain * P_zz * K_gain';
            P_new = 0.5 * (P_new + P_new');
            obj.P = obj.sanitize_matrix(P_new);
        end
        
        %% 3.0 位置边缘化数据提取接口 (适配外部 compare.m 脚本)
        function [p_est, Sigma_pos] = get_marginalized_position_info(obj)
            p_est = obj.state.p;
            Sigma_pos = obj.P(1:3, 1:3);
            Sigma_pos = 0.5 * (Sigma_pos + Sigma_pos');
        end
        
        %% 3.4 堆叠协同 UWB 更新与全维 CI 融合 (UWB Update - 10Hz)
        function obj = apply_uwb_update(obj, ~, anchor_ranges, anchor_positions, ...
                                        neighbor_ids, neighbor_positions, ...
                                        neighbor_Sigma_pos, relative_ranges, sigma_s, sigma_z)
            K = length(anchor_ranges);
            M = length(neighbor_ids);
            L = K + M; % 动态总堆叠观测维度
            
            if L == 0
                return;
            end
            
            % --- Step 1: CI 膨胀本车全维先验协方差 ---
            P_scaled = (1.0 / obj.omega_self) * obj.P;
            P_scaled = 0.5 * (P_scaled + P_scaled');
            
            % --- Step 2: 动态解算邻车非线性等效测量噪声 (3D UT 变换) ---
            R_eq = zeros(M, 1);
            if M > 0
                % CI规则下的邻车协方差动态放大系数，分母引入 eps 防御 1.0 置零
                denom = max(1.0 - obj.omega_self, 1e-6);
                P_neigh_factor = M / denom;
                
                % 提取 3 维位置空间 UT 权重
                [Wm3D, Wc3D, lambda_p] = obj.calculate_ut_weights(3);
                
                for m = 1:M
                    P_p_j = neighbor_Sigma_pos{m};
                    P_scaled_j = P_neigh_factor * P_p_j + 1e-10 * eye(3); % 对角加载防奇异
                    P_scaled_j = 0.5 * (P_scaled_j + P_scaled_j');
                    
                    % 局部 3D 协方差进行 Cholesky 抽取
                    S_j = obj.robust_chol((3 + lambda_p) * P_scaled_j);
                    p_hat_j = neighbor_positions(m, :)';
                    
                    % 生成 7 个位置无迹点
                    P_points_j = zeros(3, 7);
                    P_points_j(:, 1) = p_hat_j;
                    for i = 1:3
                        P_points_j(:, 1+i) = p_hat_j + S_j(:, i);
                        P_points_j(:, 4+i) = p_hat_j - S_j(:, i);
                    end
                    
                    % 计算每个位置无迹点投射的一维测距
                    d_j = zeros(7, 1);
                    for i = 1:7
                        d_j(i) = norm(obj.state.p - P_points_j(:, i));
                    end
                    
                    % 计算距离均值与方差
                    d_hat_j = sum(Wm3D .* d_j);
                    P_zz_neigh_j = sum(Wc3D .* (d_j - d_hat_j).^2);
                    
                    % 合并等效噪声：测距噪声 + 邻车空间散布等效噪声
                    R_eq(m) = sigma_z^2 + P_zz_neigh_j;
                end
            end
            
            % --- Step 3: 构建堆叠联合测量及噪声对角阵 ---
            z_stack = [anchor_ranges; relative_ranges];
            R_diag = [sigma_s^2 * ones(K, 1); R_eq];
            R_joint = diag(R_diag);
            
            % --- Step 4: 产生全维先验无迹点 (基于膨胀后的 P_scaled) ---
            [Wm, Wc, lambda] = obj.calculate_ut_weights(15);
            S_scaled = obj.robust_chol((15 + lambda) * P_scaled);
            
            X_scaled = cell(31, 1);
            X_scaled{1} = obj.state;
            for i = 1:15
                X_scaled{1+i}  = manifold_add(obj.state, S_scaled(:, i));
                X_scaled{16+i} = manifold_add(obj.state, -S_scaled(:, i));
            end
            
            % --- Step 5: 无迹点通过堆叠非线性观测投射 ---
            Z_stack = zeros(L, 31);
            for i = 1:31
                % A. 基站距离预测
                for k = 1:K
                    Z_stack(k, i) = norm(X_scaled{i}.p - anchor_positions(k, :)');
                end
                % B. 邻车相对测距预测 (视邻车位置为当前周期的虚拟锚点)
                for m = 1:M
                    Z_stack(K+m, i) = norm(X_scaled{i}.p - neighbor_positions(m, :)');
                end
            end
            
            % --- Step 6: 统计后验均值与协方差结构 ---
            z_stack_pred = Z_stack * Wm;
            
            P_zz_stack = zeros(L, L);
            P_xz_stack = zeros(15, L);
            for i = 1:31
                dz = Z_stack(:, i) - z_stack_pred;
                dx = manifold_sub(X_scaled{i}, obj.state);
                P_zz_stack = P_zz_stack + Wc(i) * (dz * dz');
                P_xz_stack = P_xz_stack + Wc(i) * (dx * dz');
            end
            P_zz_stack = P_zz_stack + R_joint;
            P_zz_stack = 0.5 * (P_zz_stack + P_zz_stack');
            
            % --- Step 7: 解算并执行无迹卡尔曼更新 ---
            K_gain = obj.safe_solve(P_zz_stack', P_xz_stack')';
            delta_theta = K_gain * (z_stack - z_stack_pred);
            
            % 15维等比例步长阶段，防范由于协方差全维膨胀引起的超调发散
            max_pos_step = 0.15;
            norm_pos = norm(delta_theta(1:3));
            if norm_pos > max_pos_step
                delta_theta = delta_theta * (max_pos_step / norm_pos);
            end
            
            obj.state = manifold_add(obj.state, delta_theta);
            
            % 全维协方差更新，附加对称保护机制
            P_new = P_scaled - K_gain * P_zz_stack * K_gain';
            P_new = 0.5 * (P_new + P_new');
            obj.P = obj.sanitize_matrix(P_new);
        end
            
            
    end
    
    %% 内部数值安全与数学辅助函数 (私有)
    methods (Access = private)
        %% UT 无迹变换权重生成函数
        function [Wm, Wc, lambda] = calculate_ut_weights(obj, N)
            lambda = obj.alpha^2 * (N + obj.kappa) - N;
            Wm = zeros(2*N + 1, 1);
            Wc = zeros(2*N + 1, 1);
            
            Wm(1) = lambda / (N + lambda);
            Wc(1) = Wm(1) + (1 - obj.alpha^2 + obj.beta);
            
            for i = 2:(2*N + 1)
                Wm(i) = 1 / (2 * (N + lambda));
                Wc(i) = Wm(i);
            end
        end
        
        %% 高频运动学积分预测器
        function x_next = kinematic_propagate(obj, x)
            x_next = struct();
            t_step = obj.tau;
            x_next.p = x.p + t_step * x.v + (t_step^2 / 2) * x.a;
            x_next.v = x.v + t_step * x.a;
            x_next.a = x.a;
            x_next.R = obj.robust_orthonormalize(x.R * obj.so3_exp_safe(t_step * x.omega));
            x_next.omega = x.omega;
        end
        
        %% 鲁棒 Cholesky 分解 (防御 NaN 崩溃的核心锁)
        function S = robust_chol(~, A)
            A_safe = 0.5 * (A + A');
            % 1. 首先尝试经典 Cholesky 分解
            [S, status] = chol(A_safe, 'lower');
            if status > 0
                % 2. 失败后进行多级对角线载荷补偿
                for k = 1:5
                    A_safe = A_safe + (10^(-10 + k)) * eye(size(A_safe));
                    [S, status] = chol(A_safe, 'lower');
                    if status == 0, break; end
                end
            end
            if status > 0
                % 3. 极端病态退化，使用 SVD 分解物理兜底
                [U, D, ~] = svd(A_safe);
                S = U * sqrt(max(D, 0));
            end
        end
        
        %% 阻尼线性求解器
        function x = safe_solve(~, A, b)
            A_safe = 0.5 * (A + A');
            A_reg = A_safe + 1e-9 * eye(size(A_safe));
            x = A_reg \ b;
            if any(isnan(x(:))) || any(isinf(x(:)))
                x = pinv(A_safe) * b;
            end
        end
        
        %% SVD 姿态归一化
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
        
        %% 矩阵数值净化
        function M_out = sanitize_matrix(~, M_in)
            M_out = M_in;
            M_out(isnan(M_out)) = 0;
            M_out(isinf(M_out)) = 0;
            M_out = 0.5 * (M_out + M_out');
        end
        
        %% 反对称矩阵构造
        function skew_R = skew_matrix(~, v)
            skew_R = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
        end
    end
end