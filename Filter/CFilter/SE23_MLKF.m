classdef SE23_MLKF < handle
    % 基于 SE2(3) 矩阵群建模的紧耦合多车最大似然卡尔曼滤波器 (SE3-MLKF)
    properties
        Vehicle_num         % 车辆数量 N
        states              % 结构体数组：.X (5x5 SE2(3) 状态矩阵，去偏置纯几何形式)
        P                   % 联合协方差矩阵 [9N x 9N] (已剔除偏置维数，对应 PDF Eq 63)
        Q_joint             % 联合过程噪声矩阵 [9N x 9N] (已剔除偏置维数)
        g_vec               % 3D重力加速度矢量
    end
    
    methods
        function obj = SE23_MLKF(init_states_struct, init_P, Q_sigmas)
            % SE23_MLKF 构造函数 (降维至 9D 纯几何状态，接受外部预校正输入)
            obj.Vehicle_num = length(init_states_struct);
            obj.P = init_P;
            obj.g_vec = [0; 0; -9.81]; 
            
            % 初始化状态矩阵 X 
            N = obj.Vehicle_num;
            obj.states = struct('X', {});
            for i = 1:N
                R_i = init_states_struct(i).R;
                v_i = init_states_struct(i).v;
                p_i = init_states_struct(i).p;
                
                obj.states(i).X = [R_i, v_i, p_i;
                                   0, 0, 0, 1, 0;
                                   0, 0, 0, 0, 1];
            end
            
            % 构建 9N x 9N 过程噪声矩阵
            obj.Q_joint = zeros(9*N, 9*N);
            
            sig_g = Q_sigmas.sig_wR;  % 陀螺仪白噪声 (对应 Qg)
            sig_a = Q_sigmas.sig_wa;  % 加速度计白噪声 (对应 Qa)
            
            Qc_single = diag([sig_g^2 * ones(1, 3), sig_a^2 * ones(1, 3)]);
            
            % 9D 几何切空间噪声注入矩阵 Gc (对应 PDF Eq 52 剔除偏置项后)
            Gc = [
                -eye(3),  zeros(3);
                zeros(3), -eye(3);
                zeros(3),  zeros(3)
            ];
            
            Qd_single_cont = Gc * Qc_single * Gc';
            
            for idx = 1:N
                row_idx = (idx-1)*9 + (1:9);
                obj.Q_joint(row_idx, row_idx) = Qd_single_cont; 
            end
        end
        
        
            
            function propagate(obj, imu_acc, imu_gyro, dt)
            % 状态与协方差时间传播 (9D 纯几何状态，输入已由外部减去偏置)
            N = obj.Vehicle_num;
            Phi_joint = zeros(9*N, 9*N);
            Qd_joint_discrete = zeros(9*N, 9*N);
            
            for i = 1:N
                R_k1 = obj.states(i).X(1:3, 1:3);
                v_k1 = obj.states(i).X(1:3, 4);
                p_k1 = obj.states(i).X(1:3, 5);
                
                % 直接接收外部处理好偏置的预校正数据
                omega_corr = imu_gyro(i, :)';
                acc_corr   = imu_acc(i, :)';
                
                % 标称状态递推 (对应 PDF Eq 53 - Eq 55)
                R_k = R_k1 * so3_exp(omega_corr * dt);
                v_k = v_k1 + (R_k1 * acc_corr + obj.g_vec) * dt;
                p_k = p_k1 + v_k1 * dt + 0.5 * (R_k1 * acc_corr + obj.g_vec) * dt^2;
                
                obj.states(i).X = [R_k, v_k, p_k;
                                   0, 0, 0, 1, 0;
                                   0, 0, 0, 0, 1];
                
                % 构建单车 9x9 误差状态转移矩阵 Phi_n (对应 PDF Eq 59 去除偏置项)
                I3 = eye(3);
                omega_skew = skew(omega_corr);
                acc_skew   = skew(acc_corr);
                
                Phi_n = zeros(9, 9);
                Phi_n(1:3, 1:3)   = I3 - omega_skew * dt;
                Phi_n(4:6, 1:3)   = -acc_skew * dt;
                Phi_n(4:6, 4:6)   = I3 - omega_skew * dt;
                Phi_n(7:9, 4:6)   = I3 * dt;
                Phi_n(7:9, 7:9)   = I3 - omega_skew * dt;
                
                row_idx = (i-1)*9 + (1:9);
                Phi_joint(row_idx, row_idx) = Phi_n;
                Qd_joint_discrete(row_idx, row_idx) = obj.Q_joint(row_idx, row_idx) * dt;
            end
            
            % 协方差传播 (对应 PDF Eq 63)
            obj.P = Phi_joint * obj.P * Phi_joint' + Qd_joint_discrete;
            obj.P = 0.5 * (obj.P + obj.P'); 
        end
        
        function update(obj, anchors, uwb_anc, uwb_rel, sig_s, sig_z)
            % 联合 UWB 测量极小化更新步 (对应 PDF 第 4 节)
            N = obj.Vehicle_num;
            
            % --- 1. 整理活跃 UWB 观测链路 ---
            active_anc = [];   
            active_rel = [];   
            for i = 1:N
                for k = 1:size(anchors, 1)
                    if ~isnan(uwb_anc(i, k))
                        active_anc = [active_anc; i, k, uwb_anc(i, k)];
                    end
                end
                for j = (i+1):N
                    if ~isnan(uwb_rel(i, j))
                        active_rel = [active_rel; i, j, uwb_rel(i, j)];
                    end
                end
            end
            
            M_anc = size(active_anc, 1);
            M_rel = size(active_rel, 1);
            M = M_anc + M_rel;
            
            if M == 0, return; end
            
            % 构建测量向量及噪声协方差矩阵 R
            y_meas = [];
            R_list = [];
            for r = 1:M_anc
                y_meas = [y_meas; active_anc(r, 3)];
                R_list = [R_list; sig_s^2];
            end
            for r = 1:M_rel
                y_meas = [y_meas; active_rel(r, 3)];
                R_list = [R_list; sig_z^2];
            end
            inv_R = diag(1 ./ R_list);
            
            states_prior = obj.states;
            
            % --- 2. 迭代高斯-牛顿位置优化 (对应 PDF 4.2 节，在 3N 维位置空间上迭代) ---
            p_prior = zeros(3*N, 1);
            for i = 1:N
                p_prior((i-1)*3 + (1:3)) = states_prior(i).X(1:3, 5);
            end
            
            p_opt = p_prior; % 初始值设为先验值 (对应 Eq 69)
            max_iter = 5;
            tol = 1e-4;
            
            for iter = 1:max_iter
                h_val = zeros(M, 1);
                H_jac = zeros(M, 3*N); % 评估位置空间雅可比 3N 维
                
                row_ptr = 1;
                % A. 基站测距观测及雅可比 (对应 Eq 72)
                for r = 1:M_anc
                    i = active_anc(r, 1); k = active_anc(r, 2);
                    p_i = p_opt((i-1)*3 + (1:3));
                    c_k = anchors(k, :)';
                    dist = norm(p_i - c_k);
                    if dist < 1e-6, dist = 1e-6; end
                    h_val(row_ptr) = dist;
                    H_jac(row_ptr, (i-1)*3 + (1:3)) = (p_i - c_k)' / dist;
                    row_ptr = row_ptr + 1;
                end
                
                % B. 车间测距观测及雅可比 (对应 Eq 73)
                for r = 1:M_rel
                    i = active_rel(r, 1); j = active_rel(r, 2);
                    p_i = p_opt((i-1)*3 + (1:3));
                    p_j = p_opt((j-1)*3 + (1:3));
                    dist = norm(p_i - p_j);
                    if dist < 1e-6, dist = 1e-6; end
                    h_val(row_ptr) = dist;
                    u_dir = (p_i - p_j)' / dist;
                    H_jac(row_ptr, (i-1)*3 + (1:3)) =  u_dir;
                    H_jac(row_ptr, (j-1)*3 + (1:3)) = -u_dir;
                    row_ptr = row_ptr + 1;
                end
                
                residual = y_meas - h_val;
                
                % Gauss-Newton 位置更新 (对应 Eq 78, 79)
                Hessian_pos = H_jac' * inv_R * H_jac + 1e-5 * eye(3*N); 
                Delta_p = Hessian_pos \ (H_jac' * inv_R * residual);
                
                p_opt = p_opt + Delta_p;
                
                if norm(Delta_p) < tol
                    break;
                end
            end
            
            % --- 3. 构造投影矩阵 D_k 与 Hessian 信息阵 \Xi_k (对应 Eq 80, 83) ---
            % 重新评估收敛点处的 H_jac_ML
            H_jac_ML = zeros(M, 3*N);
            row_ptr = 1;
            for r = 1:M_anc
                i = active_anc(r, 1); k = active_anc(r, 2);
                u_dir = (p_opt((i-1)*3 + (1:3)) - anchors(k, :)')' / norm(p_opt((i-1)*3 + (1:3)) - anchors(k, :)');
                H_jac_ML(row_ptr, (i-1)*3 + (1:3)) = u_dir;
                row_ptr = row_ptr + 1;
            end
            for r = 1:M_rel
                i = active_rel(r, 1); j = active_rel(r, 2);
                u_dir = (p_opt((i-1)*3 + (1:3)) - p_opt((j-1)*3 + (1:3)))' / norm(p_opt((i-1)*3 + (1:3)) - p_opt((j-1)*3 + (1:3)));
                H_jac_ML(row_ptr, (i-1)*3 + (1:3)) = u_dir;
                H_jac_ML(row_ptr, (j-1)*3 + (1:3)) = -u_dir;
                row_ptr = row_ptr + 1;
            end
            
            % 计算收敛 Hessian (对应 Eq 80)
            Xi_k = H_jac_ML' * inv_R * H_jac_ML;
            
            % 构造投影矩阵 D_k (3N x 15N)
            D_k = zeros(3*N, 9*N);
            for i = 1:N
                R_i = states_prior(i).X(1:3, 1:3); 
                D_i = zeros(3, 9);
                % 对应结构 [0_3x3, 0_3x3, R_i] (对应 PDF Eq 83 剔除偏置)
                % 误差状态分布：\delta \theta (1:3), \delta v (4:6), \delta p (7:9)
                D_i(1:3, 7:9) = R_i; 
                
                row_idx = (i-1)*3 + (1:3);
                col_idx = (i-1)*9 + (1:9);
                D_k(row_idx, col_idx) = D_i;
            end
            
            % --- 4. 先验融合与协方差稳定更新 (对应 Eq 87) ---
            Sigma_prior = obj.P;  % 此时为 9N x 9N
            Lambda_info = D_k' * Xi_k * D_k; 
            
            % Woodbury 形式协方差更新
            S_eff = eye(9*N) + Lambda_info * Sigma_prior;
            Sigma_post = Sigma_prior - Sigma_prior * (S_eff \ (Lambda_info * Sigma_prior));
            Sigma_post = 0.5 * (Sigma_post + Sigma_post');
            obj.P = Sigma_post;
            
            % --- 5. 误差状态投影与标称流形修正 (对应 Eq 88) ---
            dp_innov = p_opt - p_prior; 
            dx_joint = Sigma_post * (D_k' * (Xi_k * dp_innov)); % 此时为 9N x 1
            
            for i = 1:N
                dx_i = dx_joint((i-1)*9 + (1:9));
                dtheta = dx_i(1:3);
                dv     = dx_i(4:6);
                dp     = dx_i(7:9);
                
                % A. 提取先验 nominal 分量
                R_hat = states_prior(i).X(1:3, 1:3);
                v_hat = states_prior(i).X(1:3, 4);
                p_hat = states_prior(i).X(1:3, 5);
                
                % B. 利用指数映射与左雅可比修正名义值 (对应 PDF Eq 91 - Eq 93)
                Jl = so3_left_jacobian(dtheta);
                
                R_new = R_hat * so3_exp(dtheta);
                v_new = v_hat + R_hat * Jl * dv;
                p_new = p_hat + R_hat * Jl * dp;
                
                % C. 写回名义几何状态
                obj.states(i).X = [R_new, v_new, p_new;
                                   0, 0, 0, 1, 0;
                                   0, 0, 0, 0, 1];
            end
        end
    end
end