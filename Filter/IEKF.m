classdef IEKF < handle
    % 离散噪声下的迭代误差状态卡尔曼滤波 (IEKF)
    properties
        Vehicle_num         % 车辆数量 I
        states              % 结构体数组：.p (3x1), .v (3x1), .R (3x3)
        P                   % 联合协方差矩阵 [9I x 9I]
        Q_joint             % 联合过程噪声矩阵 [9I x 9I]
        g_vec               % 3D重力加速度常数 [1]
        use_Jt = false;
    end
    
    methods
        function obj = IEKF(init_states_struct, init_P)
            % IEKF 构造函数
            obj.Vehicle_num = length(init_states_struct);
            obj.states = init_states_struct;
            obj.P = init_P;
            obj.g_vec = [0; 0; -9.81];
            
            % 构建过程噪声矩阵 (9I x 9I)
            I = obj.Vehicle_num;
            obj.Q_joint = zeros(9*I, 9*I);
            
            Q_single = diag([ ...
                1e-10 * ones(1, 3), ...     
                (0.005)^2 * ones(1, 3), ... 
                (0.0005)^2 * ones(1, 3) ... 
            ]);
            
            for idx = 1:I
                row_idx = (idx-1)*9 + (1:9);
                obj.Q_joint(row_idx, row_idx) = Q_single;
            end
        end
        
        function propagate(obj, imu_acc, imu_gyro, dt)
            % 标称状态与协方差时间传播 (与 EKF 相同)
            I = obj.Vehicle_num;
            A_joint = zeros(9*I, 9*I);
            
            for i = 1:I
                p_t = obj.states(i).p;
                v_t = obj.states(i).v;
                R_t = obj.states(i).R;
                
                acc_tilde = imu_acc(i, :)';
                gyro_tilde = imu_gyro(i, :)';
                
                % 标称状态积分
                acc_nav = R_t * acc_tilde + obj.g_vec;
                p_next = p_t + dt * v_t + 0.5 * dt^2 * acc_nav;
                v_next = v_t + dt * acc_nav;
                R_next = R_t * so3_exp(dt * gyro_tilde);
                
                obj.states(i).p = p_next;
                obj.states(i).v = v_next;
                obj.states(i).R = R_next;
                
                % 状态转移矩阵 A_i
                A_i = eye(9);
                A_i(1:3, 4:6) = dt * eye(3);
                A_i(4:6, 7:9) = -dt * R_t * skew(acc_tilde);
                A_i(7:9, 7:9) = so3_exp(-dt * gyro_tilde);
                
                row_idx = (i-1)*9 + (1:9);
                A_joint(row_idx, row_idx) = A_i;
            end
            
            obj.P = A_joint * obj.P * A_joint' + obj.Q_joint;
            obj.P = 0.5 * (obj.P + obj.P');
        end
        
        function update(obj, anchors, uwb_anc, uwb_rel, sig_s, sig_z)
            % 迭代 ESKEF 更新步 (迭代期间 J_t 设为 I)
            I = obj.Vehicle_num;
            
            % 1. 整理活跃 UWB 观测
            active_anc = [];   
            active_rel = [];   
            for i = 1:I
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
            
            M_anc = size(active_anc, 1);
            M_rel = size(active_rel, 1);
            M = M_anc + M_rel;
            
            if M == 0, return; end
            
            % 构造测量向量与测量噪声
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
            R_cov = diag(R_list);
            
            
            states_prior = obj.states;
            chi_opt = states_prior; % 初始时，当前迭代点为先验点
            theta_curr = zeros(9*I, 1); % 迭代误差状态
            
            max_iter = 5;
            tol = 1e-4;
            
            % 2. 迭代高斯-牛顿循环 (计算雅可比并更新流形状态)
            for iter = 1:max_iter
                % --- A. 构造全局空间变换矩阵 J_joint 及其逆矩阵 J_joint_inv ---
                J_joint = zeros(9*I, 9*I);
                J_joint_inv = zeros(9*I, 9*I);
                for i = 1:I
                    if obj.use_Jt
                        % 计算当前迭代姿态相对于先验姿态的偏差向量
                        phi_e = so3_log(states_prior(i).R' * chi_opt(i).R);
                        % 理论推导：J_t 的姿态部分为 Jr_inv，其逆矩阵 J_t^-1 的姿态部分直接为 Jr
                        Jr_inv = so3_inv_right_jacobian(phi_e);
                        Jr     = so3_right_jacobian(phi_e);
                    else
                        Jr_inv = eye(3);
                        Jr     = eye(3);
                    end
                    
                    J_i = eye(9); J_i(7:9, 7:9) = Jr_inv;
                    J_i_inv = eye(9); J_i_inv(7:9, 7:9) = Jr;
                    
                    row_idx = (i-1)*9 + (1:9);
                    J_joint(row_idx, row_idx) = J_i;
                    J_joint_inv(row_idx, row_idx) = J_i_inv;
                end
                
                h_val = zeros(M, 1);
                H_jac = zeros(M, 9*I); 
                
                row_ptr = 1;
                % B. 基站测距雅可比
                for r = 1:M_anc
                    i = active_anc(r, 1); k = active_anc(r, 2);
                    p_i = chi_opt(i).p; c_k = anchors(k, :)';
                    dist = norm(p_i - c_k);
                    if dist < 1e-6, dist = 1e-6; end
                    h_val(row_ptr) = dist;
                    H_jac(row_ptr, (i-1)*9 + (1:3)) = (p_i - c_k)' / dist;
                    row_ptr = row_ptr + 1;
                end
                % C. 车间测距雅可比
                for r = 1:M_rel
                    i = active_rel(r, 1); j = active_rel(r, 2);
                    p_i = chi_opt(i).p; p_j = chi_opt(j).p;
                    dist = norm(p_i - p_j);
                    if dist < 1e-6, dist = 1e-6; end
                    h_val(row_ptr) = dist;
                    u_dir = (p_i - p_j)' / dist;
                    H_jac(row_ptr, (i-1)*9 + (1:3)) =  u_dir;
                    H_jac(row_ptr, (j-1)*9 + (1:3)) = -u_dir;
                    row_ptr = row_ptr + 1;
                end
                
                y_err = y_meas - h_val;
                
                % 计算投影后的有效先验协方差 P_eff 与卡尔曼增益 (J_t=I时 P_eff = obj.P)
                P_eff = J_joint_inv * obj.P * J_joint_inv';
                S = H_jac * P_eff * H_jac' + R_cov;
                K = (P_eff * H_jac') / S;
                
                % IEKF 误差递推核心公式 (带 J_t 流形变换的一致性形式)
                theta_next = J_joint * K * (y_err + H_jac * (J_joint_inv * theta_curr));
                
                % 将更新后的总误差状态 theta_next 作用于先验状态，生成下一次迭代的名义状态
                for i = 1:I
                    th_i = theta_next((i-1)*9 + (1:9));
                    chi_opt(i).p = states_prior(i).p + th_i(1:3);
                    chi_opt(i).v = states_prior(i).v + th_i(4:6);
                    chi_opt(i).R = states_prior(i).R * so3_exp(th_i(7:9));
                end
                
                % 收敛判定
                if norm(theta_next - theta_curr) < tol
                    theta_curr = theta_next;
                    break;
                end
                theta_curr = theta_next;
            end
            
            % 3. 后处理：使用最终迭代点评估的 P_eff 进行协方差更新与状态回馈
            obj.P = (eye(9*I) - K * H_jac) * P_eff;
            obj.P = 0.5 * (obj.P + obj.P');
            obj.states = chi_opt;
        end
    end
end