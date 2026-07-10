classdef CMLKF < handle
    properties
        Vehicle_num         % 车辆数量 I
        states              % 结构体数组：.p (3x1), .v (3x1), .a (3x1), .R (3x3), .omega (3x1) [1]
        P                   % 联合协方差矩阵 [15I x 15I] [3]
        Q_joint             % 联合过程噪声矩阵 [15I x 15I] [4]
        g_vec               % 3D重力加速度常数 [1]
        use_Jt = 0;
    end
    
    methods
        function obj = CMLKF(init_states_struct, init_P, Q_sigmas)
            % CMLKF_1 构造函数
            obj.Vehicle_num = length(init_states_struct);
            obj.states = init_states_struct;
            obj.P = init_P;
            obj.g_vec = [0; 0; -9.81]; % 3D 重力矢量 [1]
            
            % 构建联合过程噪声协方差矩阵 Q (15I x 15I)
            I = obj.Vehicle_num;
            obj.Q_joint = zeros(15*I, 15*I);
            
            Q_single = diag([ ...
                Q_sigmas.sig_wp^2 * ones(1, 3), ...
                Q_sigmas.sig_wv^2 * ones(1, 3), ...
                Q_sigmas.sig_wa^2 * ones(1, 3), ...
                Q_sigmas.sig_wR^2 * ones(1, 3), ...
                Q_sigmas.sig_womega^2 * ones(1, 3) ...
            ]);
            
            for idx = 1:I
                row_idx = (idx-1)*15 + (1:15);
                obj.Q_joint(row_idx, row_idx) = Q_single;
            end
        end
        
        function propagate(obj, dt)
            % 标称状态与协方差时间传播 (15 维系统动力学前向前向预测) [3]
            I = obj.Vehicle_num;
            A_joint = zeros(15*I, 15*I);
            
            for i = 1:I
                p_t = obj.states(i).p;
                v_t = obj.states(i).v;
                a_t = obj.states(i).a;
                R_t = obj.states(i).R;
                omega_t = obj.states(i).omega;
                
                p_next = p_t + dt * v_t + 0.5 * dt^2 * a_t;
                v_next = v_t + dt * a_t;
                a_next = a_t;
                R_next = R_t * so3_exp(dt * omega_t);
                omega_next = omega_t;
                
                obj.states(i).p = p_next;
                obj.states(i).v = v_next;
                obj.states(i).a = a_next;
                obj.states(i).R = R_next;
                obj.states(i).omega = omega_next;
                
                I3 = eye(3);
                A_i = zeros(15, 15);
                A_i(1:3, 1:3)   = I3;
                A_i(1:3, 4:6)   = dt * I3;
                A_i(1:3, 7:9)   = 0.5 * dt^2 * I3;
                A_i(4:6, 4:6)   = I3;
                A_i(4:6, 7:9)   = dt * I3;
                A_i(7:9, 7:9)   = I3;
                
                A_i(10:12, 10:12) = so3_exp(-dt * omega_t);
                A_i(10:12, 13:15) = dt * so3_right_jacobian(dt * omega_t);
                A_i(13:15, 13:15) = I3;
                
                row_idx = (i-1)*15 + (1:15);
                A_joint(row_idx, row_idx) = A_i;
            end
            
            obj.P = A_joint * obj.P * A_joint' + obj.Q_joint ;
            obj.P = 0.5 * (obj.P + obj.P'); 
        end
        
        function update(obj, imu_acc, imu_gyro, anchors, uwb_anc, uwb_rel, ...
                        sig_acc, sig_gyro, sig_s, sig_z)
            
            I = obj.Vehicle_num;
            
            % --- 1. 活跃观测链路统计 ---
            active_acc = [];   
            active_gyro = [];  
            active_anc = [];   
            active_rel = [];   
            
            for i = 1:I
                if ~any(isnan(imu_acc(i, :)))
                    active_acc = [active_acc; i, imu_acc(i, :)];
                end
                if ~any(isnan(imu_gyro(i, :)))
                    active_gyro = [active_gyro; i, imu_gyro(i, :)];
                end
                for k = 1:size(anchors, 1)
                    if ~isnan(uwb_anc(i, k))
                        active_anc = [active_anc; i, k, uwb_anc(i, k)];
                    end
                end
                for j = (i+1):I
                    if ~isnan(uwb_rel(i, j))
                        active_rel = [active_rel; i, j, uwb_rel(i, j)];
                    end
                end
            end
            
            M_acc  = size(active_acc, 1);
            M_gyro = size(active_gyro, 1);
            M_anc  = size(active_anc, 1);
            M_rel  = size(active_rel, 1);
            M = 3*M_acc + 3*M_gyro + M_anc + M_rel; 
            
            if M == 0, return; end
            
            % 构造测量向量 y_meas
            y_meas = [];
            R_list = [];
            for r = 1:M_acc
                y_meas = [y_meas; active_acc(r, 2:4)'];
                R_list = [R_list; sig_acc^2 * ones(3, 1)];
            end
            for r = 1:M_gyro
                y_meas = [y_meas; active_gyro(r, 2:4)'];
                R_list = [R_list; sig_gyro^2 * ones(3, 1)];
            end
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
            
            % --- 2. 迭代高斯-牛顿优化 (测量流形 \chi 维度 12I) [4, 5] ---
            chi_opt = struct('p', {}, 'a', {}, 'R', {}, 'omega', {});
            for i = 1:I
                chi_opt(i).p     = obj.states(i).p;
                chi_opt(i).a     = obj.states(i).a;
                chi_opt(i).R     = obj.states(i).R;
                chi_opt(i).omega = obj.states(i).omega;
            end
            
            max_iter = 5; % 目前UWB+IMU的需要迭代，而IMU_only的可以置为1
            tol = 1e-4;
            
            for iter = 1:max_iter
                h_val = zeros(M, 1);
                H_jac = zeros(M, 12*I);
                row_ptr = 1;
                
                % A. 加速度计
                for r = 1:M_acc
                    i = active_acc(r, 1); R_i = chi_opt(i).R; a_i = chi_opt(i).a;
                    h_val(row_ptr:row_ptr+2) = R_i' * (a_i - obj.g_vec);
                    H_jac(row_ptr:row_ptr+2, (i-1)*12 + (4:6)) = R_i'; 
                    H_jac(row_ptr:row_ptr+2, (i-1)*12 + (7:9)) = skew(R_i' * (a_i - obj.g_vec)); 
                    row_ptr = row_ptr + 3;
                end
                % B. 陀螺仪
                for r = 1:M_gyro
                    i = active_gyro(r, 1); omega_i = chi_opt(i).omega;
                    h_val(row_ptr:row_ptr+2) = omega_i;
                    H_jac(row_ptr:row_ptr+2, (i-1)*12 + (10:12)) = eye(3); 
                    row_ptr = row_ptr + 3;
                end
                % C. UWB 基站测距
                for r = 1:M_anc
                    i = active_anc(r, 1); k = active_anc(r, 2);
                    p_i = chi_opt(i).p; c_k = anchors(k, :)';
                    dist = norm(p_i - c_k);
                    if dist < 1e-6, dist = 1e-6; end
                    h_val(row_ptr) = dist;
                    H_jac(row_ptr, (i-1)*12 + (1:3)) = (p_i - c_k)' / dist; 
                    row_ptr = row_ptr + 1;
                end
                % D. UWB 车间测距
                for r = 1:M_rel
                    i = active_rel(r, 1); j = active_rel(r, 2);
                    p_i = chi_opt(i).p; p_j = chi_opt(j).p;
                    dist = norm(p_i - p_j);
                    if dist < 1e-6, dist = 1e-6; end
                    h_val(row_ptr) = dist;
                    u_dir = (p_i - p_j)' / dist;
                    H_jac(row_ptr, (i-1)*12 + (1:3)) = u_dir;   
                    H_jac(row_ptr, (j-1)*12 + (1:3)) = -u_dir;  
                    row_ptr = row_ptr + 1;
                end
                
                residual = y_meas - h_val;
                Hessian_pos = H_jac' * inv_R * H_jac + 1e-4 * eye(12*I); 
                Delta_s = Hessian_pos \ (H_jac' * inv_R * residual);     
                
                for i = 1:I
                    ds_i = Delta_s((i-1)*12 + (1:12));
                    chi_opt(i).p     = chi_opt(i).p + ds_i(1:3);
                    chi_opt(i).a     = chi_opt(i).a + ds_i(4:6);
                    chi_opt(i).R     = chi_opt(i).R * so3_exp(ds_i(7:9));
                    chi_opt(i).omega = chi_opt(i).omega + ds_i(10:12);
                end
                
                if norm(Delta_s) < tol
                    break;
                end
            end
            
            % 在收敛点 chi_L 处评估最终 H_jac
            chi_L = chi_opt;
            H_jac_ML = zeros(M, 12*I);
            row_ptr = 1;
            for r = 1:M_acc
                i = active_acc(r, 1);
                H_jac_ML(row_ptr:row_ptr+2, (i-1)*12 + (4:6)) = chi_L(i).R';
                H_jac_ML(row_ptr:row_ptr+2, (i-1)*12 + (7:9)) = skew(chi_L(i).R' * (chi_L(i).a - obj.g_vec));
                row_ptr = row_ptr + 3;
            end
            for r = 1:M_gyro
                i = active_gyro(r, 1);
                H_jac_ML(row_ptr:row_ptr+2, (i-1)*12 + (10:12)) = eye(3);
                row_ptr = row_ptr + 3;
            end
            for r = 1:M_anc
                i = active_anc(r, 1); k = active_anc(r, 2);
                u_dir = (chi_L(i).p - anchors(k, :)')' / norm(chi_L(i).p - anchors(k, :)');
                H_jac_ML(row_ptr, (i-1)*12 + (1:3)) = u_dir;
                row_ptr = row_ptr + 1;
            end
            for r = 1:M_rel
                i = active_rel(r, 1); j = active_rel(r, 2);
                u_dir = (chi_L(i).p - chi_L(j).p)' / norm(chi_L(i).p - chi_L(j).p);
                H_jac_ML(row_ptr, (i-1)*12 + (1:3)) = u_dir;
                H_jac_ML(row_ptr, (j-1)*12 + (1:3)) = -u_dir;
                row_ptr = row_ptr + 1;
            end
            
            % 提取极大似然观测信息 (Eq 44) [6]
             inv_Xi_t = H_jac_ML' * inv_R * H_jac_ML;
            
            
            % --- 构造全局空间变换矩阵 J_joint (12I x 12I) ---
            J_joint = zeros(12*I, 12*I);
            for i = 1:I
                if obj.use_Jt
                    R_hat = states_prior(i).R;
                    R_L = chi_L(i).R;
                    % 计算当前迭代点与先验姿态的旋转偏差向量 (对应文档 Eq 60 下方公式)
                    phi_e = so3_log(R_L' * R_hat);
                    Jr_inv = so3_inv_right_jacobian(phi_e);
                else
                    Jr_inv = eye(3);
                end
                
                J_i = eye(12);
                J_i(7:9, 7:9) = Jr_inv; % \delta \phi 对应误差向量第 7:9 维
                
                row_idx = (i-1)*12 + (1:12);
                J_joint(row_idx, row_idx) = J_i;
            end
            
            % 构造似然偏差 \mu_t (对应文档 Eq 54)
            mu_t = zeros(12*I, 1);
            for i = 1:I
                p_hat = states_prior(i).p;
                a_hat = states_prior(i).a;
                R_hat = states_prior(i).R;
                omega_hat = states_prior(i).omega;
                
                mu_t((i-1)*12 + (1:3))   = chi_L(i).p - p_hat;
                mu_t((i-1)*12 + (4:6))   = chi_L(i).a - a_hat;
                mu_t((i-1)*12 + (7:9))   = so3_log(R_hat' * chi_L(i).R);
                mu_t((i-1)*12 + (10:12)) = chi_L(i).omega - omega_hat;
            end
            
            % 构造切平面选择矩阵 \pi (12I x 15I)
            pi_mat = zeros(12*I, 15*I);
            for i = 1:I
                pi_i = zeros(12, 15);
                pi_i(1:3, 1:3)     = eye(3);  % 选择 \delta p
                pi_i(4:6, 7:9)     = eye(3);  % 选择 \delta a
                pi_i(7:9, 10:12)   = eye(3);  % 选择 \delta \phi
                pi_i(10:12, 13:15) = eye(3);  % 选择 \delta \omega
                
                row_idx = (i-1)*12 + (1:12);
                col_idx = (i-1)*15 + (1:15);
                pi_mat(row_idx, col_idx) = pi_i;
            end
            
            % 结合 J_joint 进行信息投影 (对应文档 Eq 67 与 Eq 68)
            inv_Xi_prior = J_joint' * inv_Xi_t * J_joint;
            Lambda_t = pi_mat' * inv_Xi_prior * pi_mat;
            lambda_t = pi_mat' * (J_joint' * (inv_Xi_t * mu_t));
            
            % 稳定的 Woodbury 形式协方差更新
            Sigma_prior = obj.P;
            S_eff = eye(15*I) + Lambda_t * Sigma_prior;
            Sigma_post = Sigma_prior - Sigma_prior * (S_eff \ (Lambda_t * Sigma_prior));
            Sigma_post = 0.5 * (Sigma_post + Sigma_post'); 
            obj.P = Sigma_post;
            
            % 状态更新 Eq 70
            dx_joint = Sigma_post * lambda_t;
            for i = 1:I
                dx_i = dx_joint((i-1)*15 + (1:15));
                obj.states(i) = manifold_add(obj.states(i), dx_i);
            end
        end
        
        function update_imu_only(obj, imu_acc, imu_gyro, sig_acc, sig_gyro)
            % 【全新重构】：利用 MLKF 非线性投影结构在 9维 IMU测量空间上进行高频优化更新 [4]
            
            I = obj.Vehicle_num;
            M = 6*I; % IMU测量的联合维度 (3D加速度 + 3D角速度) [1]
            
            % 测量数据
            y_meas = zeros(M, 1);
            R_list = zeros(M, 1);
            for i = 1:I
                y_meas((i-1)*6 + (1:3)) = imu_acc(i, :)';
                y_meas((i-1)*6 + (4:6)) = imu_gyro(i, :)';
                R_list((i-1)*6 + (1:3)) = sig_acc^2;
                R_list((i-1)*6 + (4:6)) = sig_gyro^2;
            end
            inv_R = diag(1 ./ R_list);
            
            states_prior = obj.states;
            
            % 1. 迭代高斯-牛顿优化 (仅在 9维 IMU 测量流形 \chi_IMU = [a; R; \omega] 上优化)
            chi_opt = struct('a', {}, 'R', {}, 'omega', {});
            for i = 1:I
                chi_opt(i).a     = obj.states(i).a;
                chi_opt(i).R     = obj.states(i).R;
                chi_opt(i).omega = obj.states(i).omega;
            end
            
            max_iter = 1;
            tol = 1e-4;
            
            for iter = 1:max_iter
                h_val = zeros(M, 1);
                H_jac = zeros(M, 9*I); % [da; dphi; domega] 空间雅可比
                
                for i = 1:I
                    R_i = chi_opt(i).R;
                    a_i = chi_opt(i).a;
                    omega_i = chi_opt(i).omega;
                    
                    % 评估IMU测量方程 (文档 Eq 7, 8) [1]
                    h_val((i-1)*6 + (1:3)) = R_i' * (a_i - obj.g_vec);
                    h_val((i-1)*6 + (4:6)) = omega_i;
                    
                    H_i = zeros(6, 9);
                    H_i(1:3, 1:3) = R_i';                                  % wrt \delta a
                    H_i(1:3, 4:6) = skew(R_i' * (a_i - obj.g_vec));        % wrt \delta \phi
                    H_i(4:6, 7:9) = eye(3);                                % wrt \delta \omega
                    
                    H_jac((i-1)*6 + (1:6), (i-1)*9 + (1:9)) = H_i;
                end
                
                residual = y_meas - h_val;
                Hessian_pos = H_jac' * inv_R * H_jac + 1e-4 * eye(9*I); % LM阻尼保护
                Delta_s = Hessian_pos \ (H_jac' * inv_R * residual);
                
                % IMU 本地流形退回更新 [5]
                for i = 1:I
                    ds_i = Delta_s((i-1)*9 + (1:9));
                    chi_opt(i).a     = chi_opt(i).a + ds_i(1:3);
                    chi_opt(i).R     = chi_opt(i).R * so3_exp(ds_i(4:6));
                    chi_opt(i).omega = chi_opt(i).omega + ds_i(7:9);
                end
                
                if norm(Delta_s) < tol
                    break;
                end
            end
            
            % 收敛点 chi_L 处计算最终特征 Hessian 信息矩阵
            chi_L = chi_opt;
            H_jac_ML = zeros(M, 9*I);
            for i = 1:I
                H_i = zeros(6, 9);
                H_i(1:3, 1:3) = chi_L(i).R';
                H_i(1:3, 4:6) = skew(chi_L(i).R' * (chi_L(i).a - obj.g_vec));
                H_i(4:6, 7:9) = eye(3);
                H_jac_ML((i-1)*6 + (1:6), (i-1)*9 + (1:9)) = H_i;
            end
             inv_Xi_t = H_jac_ML' * inv_R * H_jac_ML;
            
            
            % --- 构造全局空间变换矩阵 J_joint_imu (9I x 9I) ---
            J_joint_imu = zeros(9*I, 9*I);
            for i = 1:I
                if obj.use_Jt
                    R_hat = states_prior(i).R;
                    R_L = chi_L(i).R;
                    phi_e = so3_log(R_L' * R_hat);
                    Jr_inv = so3_inv_right_jacobian(phi_e);
                else
                    Jr_inv = eye(3);
                end
                
                J_i = eye(9);
                J_i(4:6, 4:6) = Jr_inv; % \delta \phi 在 IMU 局部估计中对应第 4:6 维
                
                row_idx = (i-1)*9 + (1:9);
                J_joint_imu(row_idx, row_idx) = J_i;
            end
            
            % 标称状态似然偏差 \mu_t 提取
            mu_t = zeros(9*I, 1);
            for i = 1:I
                a_hat = states_prior(i).a;
                R_hat = states_prior(i).R;
                omega_hat = states_prior(i).omega;
                
                mu_t((i-1)*9 + (1:3)) = chi_L(i).a - a_hat;
                mu_t((i-1)*9 + (4:6)) = so3_log(R_hat' * chi_L(i).R);
                mu_t((i-1)*9 + (7:9)) = chi_L(i).omega - omega_hat;
            end
            
            % 构造 9I 维 IMU 测量误差切空间到 15I 维系统全误差切空间的投影选择矩阵
            pi_IMU = zeros(9*I, 15*I);
            for i = 1:I
                pi_i = zeros(9, 15);
                pi_i(1:3, 7:9)   = eye(3);  % 选择 \delta a
                pi_i(4:6, 10:12) = eye(3);  % 选择 \delta \phi
                pi_i(7:9, 13:15) = eye(3);  % 选择 \delta \omega
                
                row_idx = (i-1)*9 + (1:9);
                col_idx = (i-1)*15 + (1:15);
                pi_IMU(row_idx, col_idx) = pi_i;
            end
            
            % 结合 J_joint_imu 进行信息投影
            inv_Xi_prior = J_joint_imu' * inv_Xi_t * J_joint_imu;
            Lambda_t = pi_IMU' * inv_Xi_prior * pi_IMU;
            lambda_t = pi_IMU' * (J_joint_imu' * (inv_Xi_t * mu_t));
            
            % 稳定的无求逆卡尔曼融合
            Sigma_prior = obj.P;
            S_eff = eye(15*I) + Lambda_t * Sigma_prior;
            Sigma_post = Sigma_prior - Sigma_prior * (S_eff \ (Lambda_t * Sigma_prior));
            Sigma_post = 0.5 * (Sigma_post + Sigma_post');
            obj.P = Sigma_post;
            
            % 流形状态纠正与重置 [7, 13]
            dx_joint = Sigma_post * lambda_t;
            for i = 1:I
                dx_i = dx_joint((i-1)*15 + (1:15));
                obj.states(i) = manifold_add(obj.states(i), dx_i);
            end
        end
    end
end