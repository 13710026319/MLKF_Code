classdef DMLKF < handle
    % 分布式极大似然卡尔曼滤波器 (DMLKF) - 单车实例 (完全修复版)
    properties
        id                  % 车辆自身 ID (1~N)
        neighbors           % 邻车 ID 列表, 例如 [2, 4]
        states              % 结构体：.p (3x1), .v (3x1), .a (3x1), .R (3x3), .omega (3x1)
        P                   % 协方差矩阵 [15 x 15]
        I_indep             % 独立信息矩阵 [15 x 15]
        I_dep               % 依赖信息矩阵 [15 x 15]
        Q                   % 本地过程噪声矩阵 [15 x 15]
        g_vec               % 重力向量 [0; 0; -9.81]
        
        % ADMM 共识状态缓存变量
        s1_curr             % 联合本地位置误差 [9 x 1] ([dp_self; dp_n1; dp_n2])
        lambda_1j           % Lagrange 乘子 1_j [6 x 1] (主车对邻车的惩罚)
        lambda_j1           % Lagrange 乘子 j_1 [6 x 1] (邻车对主车的惩罚)
        dp_neighbor_self    % 邻车对自身误差的估计 \delta p^{j,i} [6 x 1]
        dp_neighbor_own     % 邻车对其自身误差的估计 \delta p^{j,j} [6 x 1]
    end
    
    methods
        function obj = DMLKF(id, neighbors, init_state_struct, init_P, Q_sigmas)
            % DMLKF 构造函数
            obj.id = id;
            obj.neighbors = neighbors;
            obj.g_vec = [0; 0; -9.81];
            obj.states = init_state_struct;
            obj.P = init_P;
            
            % 使用左除初始化信息阵 (对应 PDF Eq 18 及 Bug 3 优化)
            obj.I_indep = obj.P \ eye(15);
            obj.I_dep = zeros(15, 15);
            
            % 解决 Bug 1: 构造函数中直接使用 Q_sigmas 完成过程噪声初始化 (属性统一为 Q)
            obj.Q = diag([ ...
                Q_sigmas.sig_wp^2 * ones(1, 3), ...
                Q_sigmas.sig_wv^2 * ones(1, 3), ...
                Q_sigmas.sig_wa^2 * ones(1, 3), ...
                Q_sigmas.sig_wR^2 * ones(1, 3), ...
                Q_sigmas.sig_womega^2 * ones(1, 3) ...
            ]);
        end
        
        function propagate(obj, dt)
            % 分布式独立传播步与信息分离递推 (对应 PDF 1.2 节)
            p_t = obj.states.p;
            v_t = obj.states.v;
            a_t = obj.states.a;
            R_t = obj.states.R;
            omega_t = obj.states.omega;
            
            % 标称状态预测 (Eq 7 - 11)
            p_next = p_t + dt * v_t + 0.5 * dt^2 * a_t;
            v_next = v_t + dt * a_t;
            a_next = a_t;
            R_next = R_t * so3_exp(dt * omega_t);
            omega_next = omega_t;
            
            obj.states.p = p_next;
            obj.states.v = v_next;
            obj.states.a = a_next;
            obj.states.R = R_next;
            obj.states.omega = omega_next;
            
            % 计算状态转移雅可比矩阵 A_i (Eq 15)
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
            
            % 计算先验总协方差 (Eq 19)
            Sigma_prior_total = (obj.I_indep + obj.I_dep) \ eye(15);
            Sigma_prop = A_i * Sigma_prior_total * A_i' + obj.Q * dt;
            Sigma_prop = 0.5 * (Sigma_prop + Sigma_prop');
            
            % 解决 Bug 3: 全文使用更高效的左除 \ 代替 inv()
            I_total_prop = Sigma_prop \ eye(15);
            
            % 解决 Bug 3: 构造信息分离算子 W_i (Eq 20)
            W_i = Sigma_prop \ (A_i * Sigma_prior_total);
            
            % 拆分信息的分离传播 (Eq 21, 22)
            obj.I_dep = W_i * obj.I_dep * W_i';
            obj.I_indep = I_total_prop - obj.I_dep;
            obj.I_indep = 0.5 * (obj.I_indep + obj.I_indep');
            
            % 解决 Bug 6: 同步更新类的成员协方差属性 P
            obj.P = Sigma_prop;
        end
        
        function update_imu_only(obj, imu_acc, imu_gyro, sig_acc, sig_gyro)
            % 本地高频 IMU-Only 最大似然更新 (对应 PDF 第 2 节)
            y_meas = [imu_acc'; imu_gyro'];
            inv_R = diag([1/sig_acc^2 * ones(1, 3), 1/sig_gyro^2 * ones(1, 3)]);
            
            states_prior = obj.states;
            s_imu = zeros(9, 1); % [da; dphi; domega]
            
            max_iter = 2;
            tol = 1e-4;
            
            % Gauss-Newton 迭代评估
            for iter = 1:max_iter
                da_v     = s_imu(1:3);
                dphi_v   = s_imu(4:6);
                domega_v = s_imu(7:9);
                
                a_v     = states_prior.a + da_v;
                R_v     = states_prior.R * so3_exp(dphi_v);
                omega_v = states_prior.omega + domega_v;
                
                % 评估残差 r_IMU (Eq 31)
                r_val = zeros(6, 1);
                r_val(1:3) = y_meas(1:3) - R_v' * (a_v - obj.g_vec);
                r_val(4:6) = y_meas(4:6) - omega_v;
                
                % 评估残差雅可比 H_IMU_p (Eq 35)
                H_p = zeros(6, 9);
                H_p(1:3, 1:3) = -R_v';                               
                H_p(1:3, 4:6) = -skew(R_v' * (a_v - obj.g_vec)) * so3_right_jacobian(dphi_v); 
                H_p(4:6, 7:9) = -eye(3);                             
                
                g_vec_opt = H_p' * inv_R * r_val;
                H_mat_opt = H_p' * inv_R * H_p;
                
                % 更新迭代步 (Eq 38)
                Delta_s = (H_mat_opt + 1e-4*eye(9)) \ g_vec_opt;
                s_imu = s_imu - Delta_s;
                
                if norm(Delta_s) < tol, break; end
            end
            
            % 提取等效测量信息对并映射回 15 维空间 (Eq 40 - 43)
            Lambda_imu = H_mat_opt;
            lambda_imu = Lambda_imu * s_imu;
            
            % 15 维选择矩阵 (Eq 28)
            pi_IMU = zeros(9, 15);
            pi_IMU(1:3, 7:9)   = eye(3);  % \delta a
            pi_IMU(4:6, 10:12) = eye(3);  % \delta \phi
            pi_IMU(7:9, 13:15) = eye(3);  % \delta \omega
            
            Lambda_full = pi_IMU' * Lambda_imu * pi_IMU;
            lambda_full = pi_IMU' * lambda_imu;
            
            % 仅累加到独立信息中 (Eq 44, 45)
            obj.I_indep = obj.I_indep + Lambda_full;
            
            % 解决 Bug 6: 状态和协方差更新
            obj.P = (obj.I_indep + obj.I_dep) \ eye(15);
            obj.P = 0.5 * (obj.P + obj.P');
            
            dx_joint = obj.P * lambda_full;
            obj.states = obj.retract(states_prior, dx_joint);
            obj.I_indep = 0.5 * (obj.I_indep + obj.I_indep');
        end
        
        function [p_hat, I_indep_3d, I_dep_3d] = get_projected_info(obj)
            % UWB 更新前：向邻车分享投影的 3D 位置信息量 (对应 PDF 第 3 节 Eq 49 - 51)
            Sigma_total = (obj.I_indep + obj.I_dep) \ eye(15);
            
            % 提取位置相关 3D 协方差与信息
            Sigma_pos = Sigma_total(1:3, 1:3);
            I_pos_3d = Sigma_pos \ eye(3); % 解决 Bug 3: 用左除计算，替代 inv
            
            % 投影算子 W_pos (Eq 49)
            W_pos = I_pos_3d * Sigma_total(1:3, :);
            
            % 投影计算独立与依赖 3D 信息分量
            I_dep_3d = W_pos * obj.I_dep * W_pos';
            I_indep_3d = I_pos_3d - I_dep_3d;
            I_indep_3d = 0.5 * (I_indep_3d + I_indep_3d');
            
            p_hat = obj.states.p;
        end
        
        function init_uwb_update(obj)
            % ADMM 观测更新：初始化外层共识变量
            obj.s1_curr = zeros(9, 1);       
            obj.lambda_1j = zeros(6, 1);     
            obj.lambda_j1 = zeros(6, 1);     
            obj.dp_neighbor_self = zeros(6, 1); 
            obj.dp_neighbor_own = zeros(6, 1);
        end
        
        function admm_primal_step(obj, anchors, uwb_anc, uwb_rel, sig_s, sig_z, p_neigh, rho)
            % 解决 Bug 4: 移除了未在 Primal GN 步骤中使用的 I_indep_neigh 与 I_dep_neigh
            
            states_prior = obj.states;
            p1_hat = states_prior.p;
            
            % 评估活跃绝对测距链路
            active_k = find(~isnan(uwb_anc));
            K_num = length(active_k);
            
            max_inner = 5;
            tol_inner = 1e-4;
            
            % 邻车先验位置
            p2_hat = p_neigh(:, 1);
            p4_hat = p_neigh(:, 2);
            
            for inner = 1:max_inner
                dp11 = obj.s1_curr(1:3);
                dp12 = obj.s1_curr(4:6);
                dp14 = obj.s1_curr(7:9);
                
                % 评估视线单位向量与残差
                u_1k = zeros(3, K_num); r_k = zeros(K_num, 1);
                for k_idx = 1:K_num
                    k = active_k(k_idx);
                    c_k = anchors(k, :)';
                    vec = p1_hat + dp11 - c_k;
                    dist = norm(vec);
                    if dist < 1e-6, dist = 1e-6; end
                    r_k(k_idx) = uwb_anc(k) - dist;
                    u_1k(:, k_idx) = vec / dist;
                end
                
                % 评估与邻居 2 的相对残差
                vec_12 = (p1_hat + dp11) - (p2_hat + dp12);
                dist_12 = norm(vec_12);
                if dist_12 < 1e-6, dist_12 = 1e-6; end
                r_z2 = uwb_rel(obj.neighbors(1)) - dist_12;
                u_12 = vec_12 / dist_12;
                
                % 评估与邻居 4 的相对残差
                vec_14 = (p1_hat + dp11) - (p4_hat + dp14);
                dist_14 = norm(vec_14);
                if dist_14 < 1e-6, dist = 1e-6; end
                r_z4 = uwb_rel(obj.neighbors(2)) - dist;
                u_14 = vec_14 / dist;
                
                % 构造观测梯度向量 (Eq 62, 63)
                g_dp11_obs = - (1/sig_s^2) * sum(u_1k .* r_k', 2) ...
                             - (1/sig_z^2) * u_12 * r_z2 ...
                             - (1/sig_z^2) * u_14 * r_z4;
                         
                g_dp12_obs = (1/sig_z^2) * u_12 * r_z2;
                g_dp14_obs = (1/sig_z^2) * u_14 * r_z4;
                
                % 叠加共识罚梯度 (Eq 62, 63)
                g_dp11_pen = (-obj.lambda_j1(1:3) - rho * (obj.dp_neighbor_self(1:3) - dp11)) + ...
                             (-obj.lambda_j1(4:6) - rho * (obj.dp_neighbor_self(4:6) - dp11));
                         
                g_dp12_pen = obj.lambda_1j(1:3) + rho * (dp12 - obj.dp_neighbor_own(1:3));
                g_dp14_pen = obj.lambda_1j(4:6) + rho * (dp14 - obj.dp_neighbor_own(4:6));
                
                g_L = [g_dp11_obs + g_dp11_pen;
                       g_dp12_obs + g_dp12_pen;
                       g_dp14_obs + g_dp14_pen];
                   
                % 构造高斯-牛顿拉格朗日 Hessian 矩阵 H_L
                H_L_obs = zeros(9, 9);
                for k_idx = 1:K_num
                    H_k = [-u_1k(:, k_idx)', zeros(1, 6)];
                    H_L_obs = H_L_obs + (1/sig_s^2) * (H_k' * H_k);
                end
                
                H2_int = [-u_12', u_12', zeros(1, 3)];
                H4_int = [-u_14', zeros(1, 3), u_14'];
                H_L_obs = H_L_obs + (1/sig_z^2) * (H2_int' * H2_int) ...
                                  + (1/sig_z^2) * (H4_int' * H4_int);
                
                % 罚因子 D_penalty (Eq 66)
                D_pen = diag([2*rho*ones(1, 3), rho*ones(1, 3), rho*ones(1, 3)]);
                H_L_total = H_L_obs + D_pen;
                
                Delta_s = H_L_total \ g_L;
                obj.s1_curr = obj.s1_curr - Delta_s;
                
                if norm(Delta_s) < tol_inner, break; end
            end
        end
        
        function exchange_consensus(obj, received_consensus)
            % 外部共识变量写入接口
            obj.dp_neighbor_own  = received_consensus.dp_jj;
            obj.dp_neighbor_self = received_consensus.dp_ji;
        end
        
        function admm_dual_update(obj, rho)
            % ADMM 双边拉格朗日乘子更新 (对应 PDF Eq 56, 57)
            dp11 = obj.s1_curr(1:3);
            dp12 = obj.s1_curr(4:6);
            dp14 = obj.s1_curr(7:9);
            
            % 更新主车对邻车估计偏差的惩罚项 (Eq 56)
            obj.lambda_1j(1:3) = obj.lambda_1j(1:3) + rho * (dp12 - obj.dp_neighbor_own(1:3));
            obj.lambda_1j(4:6) = obj.lambda_1j(4:6) + rho * (dp14 - obj.dp_neighbor_own(4:6));
            
            % 更新邻车对主车估计偏差的惩罚项 (Eq 57)
            obj.lambda_j1(1:3) = obj.lambda_j1(1:3) + rho * (obj.dp_neighbor_self(1:3) - dp11);
            obj.lambda_j1(4:6) = obj.lambda_j1(4:6) + rho * (obj.dp_neighbor_self(4:6) - dp11);
        end
        
        function finalize_uwb_update(obj, anchors, uwb_anc, uwb_rel, sig_s, sig_z, ...
                                     p_neigh, I_indep_neigh, I_dep_neigh)
            % 解决 Bug 5: 重新设计活跃链路评估，使 uwb_anc 和 uwb_rel 在 Hessian 与融合阶段发挥作用
            
            states_prior = obj.states;
            p1_hat = states_prior.p;
            p2_hat = p_neigh(:, 1);
            p4_hat = p_neigh(:, 2);
            
            dp11 = obj.s1_curr(1:3);
            dp12 = obj.s1_curr(4:6);
            dp14 = obj.s1_curr(7:9);
            
            % 1. 评估活跃绝对测距并提取测量 Hessian (Eq 69)
            active_k = find(~isnan(uwb_anc)); 
            K_num = length(active_k);
            
            Xi11_aa = zeros(3, 3); 
            for k_idx = 1:K_num
                k = active_k(k_idx);
                vec = p1_hat + dp11 - anchors(k, :)';
                dist_val = norm(vec);
                if dist_val < 1e-6, dist_val = 1e-6; end
                u_k = vec / dist_val;
                Xi11_aa = Xi11_aa + (1/sig_s^2) * (u_k * u_k');
            end
            
            % 评估活跃相对测距块并计算车间相对测距部分
            Xi_int = zeros(9, 9);
            if ~isnan(uwb_rel(obj.neighbors(1)))
                vec_12 = (p1_hat + dp11) - (p2_hat + dp12);
                dist_12 = norm(vec_12);
                if dist_12 < 1e-6, dist_12 = 1e-6; end
                u_12 = vec_12 / dist_12;
                H2_int = [-u_12', u_12', zeros(1, 3)];
                Xi_int = Xi_int + (1/sig_z^2) * (H2_int' * H2_int);
            end
            
            if ~isnan(uwb_rel(obj.neighbors(2)))
                vec_14 = (p1_hat + dp11) - (p4_hat + dp14);
                dist_14 = norm(vec_14);
                if dist_14 < 1e-6, dist_14 = 1e-6; end
                u_14 = vec_14 / dist_14;
                H4_int = [-u_14', zeros(1, 3), u_14'];
                Xi_int = Xi_int + (1/sig_z^2) * (H4_int' * H4_int);
            end
            
            Xi11_ii = Xi_int(1:3, 1:3);
            Xi12_ii = Xi_int(1:3, 4:9);
            Xi21_ii = Xi_int(4:9, 1:3);
            Xi22_ii = Xi_int(4:9, 4:9);
            
            % 2. 邻居分割依赖度加权 (SCI 算法：自身分配 0.6, 两个邻居分别分配 0.2) (Eq 70)
            Lam_SCI_2 = I_indep_neigh(:, 1:3) + 0.2 * I_dep_neigh(:, 1:3);
            Lam_SCI_4 = I_indep_neigh(:, 4:6) + 0.2 * I_dep_neigh(:, 4:6);
            P22_SCI_inv = blockdiag(Lam_SCI_2, Lam_SCI_4);
            
            % 3. 舒尔补边缘化，压缩至主车自身 3D 等效信息对 (Eq 72 - 77)
            Lam1_anc = Xi11_aa;
            % 解决 Bug 3: 使用左除计算逆
            Lam1_int = Xi11_ii - Xi12_ii * ((Xi22_ii + P22_SCI_inv) \ Xi21_ii);
            
            % 计算等效信息矢量
            s_neigh_MLE = [dp12; dp14];
            lam1_anc = Lam1_anc * dp11;
            lam1_int = Lam1_int * dp11 + Xi12_ii * ((Xi22_ii + P22_SCI_inv) \ (P22_SCI_inv * s_neigh_MLE));
            
            % 4. 逆投影回 15 维系统全空间 (Eq 78, 79)
            Lambda_anc_full = zeros(15, 15); Lambda_anc_full(1:3, 1:3) = Lam1_anc;
            lambda_anc_full = zeros(15, 1);  lambda_anc_full(1:3)     = lam1_anc;
            
            Lambda_int_full = zeros(15, 15); Lambda_int_full(1:3, 1:3) = Lam1_int;
            lambda_int_full = zeros(15, 1);  lambda_int_full(1:3)     = lam1_int;
            
            % 5. 本地拆分信息分类累加更新 (Eq 80, 81)
            obj.I_indep = obj.I_indep + Lambda_anc_full;
            obj.I_dep   = 0.6 * obj.I_dep + Lambda_int_full; % 自我加权 0.6 叠加相对协同项
            
            % 6. 解决 Bug 6: 后验协方差矩阵和状态更新
            obj.P = (obj.I_indep + obj.I_dep) \ eye(15);
            obj.P = 0.5 * (obj.P + obj.P');
            
            dx_joint = obj.P * (lambda_anc_full + lambda_int_full);
            
            % 名义值流形更新
            obj.states = obj.retract(states_prior, dx_joint);
            obj.I_indep = 0.5 * (obj.I_indep + obj.I_indep');
            obj.I_dep = 0.5 * (obj.I_dep + obj.I_dep');
        end
        
        function st_new = retract(~, st, dx)
            % 本地流形 retraction 增量直和映射
            st_new.p     = st.p + dx(1:3);
            st_new.v     = st.v + dx(4:6);
            st_new.a     = st.a + dx(7:9);
            st_new.R     = st.R * so3_exp(dx(10:12));
            st_new.omega = st.omega + dx(13:15);
        end
    end
end