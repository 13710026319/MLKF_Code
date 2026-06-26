classdef DMLKF
    % DMLKF: 去中心化最大似然卡尔曼滤波器类
    % 严格针对预测步与IMU更新步进行了NaN防御与数值优化设计
    
    properties
        id              % 节点ID
        state           % 状态结构体: .p(3x1), .v(3x1), .a(3x1), .R(3x3), .omega(3x1)
        I_indep         % 15x15 独立信息矩阵
        I_dep           % 15x15 相关信息矩阵
        Q               % 15x15 过程噪声协方差矩阵
        Sigma_a         % 3x3 加速度计噪声协方差
        Sigma_w         % 3x3 陀螺仪噪声协方差
        g_vec           % 3x1 重力向量 (通常为 [0; 0; -9.81])
        tau             % IMU采样周期 (秒)
        mu              % 高斯-牛顿正则化参数 (防御NaN)
        
        % ADMM 持久化变量 (用于热启动，加速收敛并防抖动)
        lambda_local    % 容器: 存储该车对邻车的拉格朗日乘子 (对应邻车ID)
        lambda_remote   % 容器: 存储邻车对该车的拉格朗日乘子 (对应邻车ID)
    end
    
    methods
        %% 构造函数
        function obj = DMLKF(id, init_state, init_cov, Q_matrix, Sigma_a, Sigma_w, tau)
            obj.id = id;
            obj.state = init_state;
            obj.state.R = obj.robust_orthonormalize(init_state.R);
            
            % 保证初始协方差不奇异，防NaN
            init_cov_safe = init_cov + 1e-10 * eye(15);
            obj.I_indep = obj.safe_inv(init_cov_safe);
            obj.I_dep = zeros(15, 15);
            
            obj.Q = Q_matrix;
            obj.Sigma_a = Sigma_a;
            obj.Sigma_w = Sigma_w;
            obj.tau = tau;
            obj.g_vec = [0; 0; -9.81];
            obj.mu = 1e-5; % 防止GN迭代中Hessian退化
            
            obj.lambda_local = containers.Map('KeyType', 'double', 'ValueType', 'any');
            obj.lambda_remote = containers.Map('KeyType', 'double', 'ValueType', 'any');
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
            
            % 姿态传播，采用安全指数映射防NaN
            exp_w = obj.so3_exp_safe(t_step * w_old);
            obj.state.R = obj.robust_orthonormalize(R_old * exp_w);
            obj.state.omega = w_old;
            
            % --- 2. 构造 Jacobian A_t (Eq. 15) ---
            A_t = eye(15);
            A_t(1:3, 4:6)   = t_step * eye(3);
            A_t(1:3, 7:9)   = (t_step^2 / 2) * eye(3);
            A_t(4:6, 7:9)   = t_step * eye(3);
            
            % 旋转误差传播项
            A_t(10:12, 10:12) = obj.so3_exp_safe(-t_step * w_old);
            A_t(10:12, 13:15) = t_step * obj.so3_right_jacobian_safe(t_step * w_old);
            
            % --- 3. 协方差与分裂信息矩阵传播 (Eq. 19-22) ---
            Sigma_curr = obj.safe_inv(obj.I_indep + obj.I_dep);
            Sigma_pred = A_t * Sigma_curr * A_t' + obj.Q;
            Sigma_pred = 0.5 * (Sigma_pred + Sigma_pred'); % 保证对称正定
            
            I_total_pred = obj.safe_inv(Sigma_pred);
            
            % 数值优化计算分离算子 W_t以防极小值求逆导致NaN
            % W_t = I_total_pred * A_t * Sigma_curr
            W_t = Sigma_pred \ (A_t * Sigma_curr);
            if any(isnan(W_t(:))) || any(isinf(W_t(:)))
                W_t = pinv(Sigma_pred) * (A_t * Sigma_curr);
            end
            
            % 传播独立/相关成分
            I_dep_new = W_t * obj.I_dep * W_t';
            I_dep_new = 0.5 * (I_dep_new + I_dep_new');
            
            I_indep_new = I_total_pred - I_dep_new;
            I_indep_new = 0.5 * (I_indep_new + I_indep_new');
            
            % 更新并保护属性
            obj.I_indep = obj.sanitize_matrix(I_indep_new);
            obj.I_dep = obj.sanitize_matrix(I_dep_new);
        end
        
        %% 2.2 局部高频 IMU 更新 (IMU Update - 100Hz)
        function obj = update_imu(obj, raw_acc, raw_gyro, bias_a, bias_w)
            % 预修正测量值
            acc_tilde = raw_acc - bias_a;
            gyro_tilde = raw_gyro - bias_w;
            
            % 测量噪声协方差 (6x6)
            R_IMU = blkdiag(obj.Sigma_a, obj.Sigma_w);
            R_IMU_inv = obj.safe_inv(R_IMU);
            
            % 初始化局部扰动 s_IMU (9x1) -> [da; dphi; domega]
            s_IMU = zeros(9, 1);
            
            % 暂存先验状态
            a_prior = obj.state.a;
            R_prior = obj.state.R;
            w_prior = obj.state.omega;
            
            % 局部 Gauss-Newton 迭代求解非线性极值 (防御大偏差及极小值不收敛)
            max_iter = 5;
            for iter = 1:max_iter
                % 根据切空间更新当前名义状态估值 (Eq. 30)
                da = s_IMU(1:3);
                dphi = s_IMU(4:6);
                domega = s_IMU(7:9);
                
                a_iter = a_prior + da;
                R_iter = obj.robust_orthonormalize(R_prior * obj.so3_exp_safe(dphi));
                w_iter = w_prior + domega;
                
                % 计算残差向量 r_IMU (Eq. 31)
                r_acc = acc_tilde - R_iter' * (a_iter - obj.g_vec);
                r_gyro = gyro_tilde - w_iter;
                r = [r_acc; r_gyro];
                
                % 计算右雅可比
                Jr_phi = obj.so3_right_jacobian_safe(dphi);
                
                % 构造解析残差雅可比 H_IMU_p (6x9) (Eq. 35)
                H = zeros(6, 9);
                H(1:3, 1:3) = -R_iter';
                H(1:3, 4:6) = -obj.skew_matrix(R_iter' * (a_iter - obj.g_vec)) * Jr_phi;
                H(4:6, 7:9) = -eye(3);
                
                % 计算 Hessian 与 梯度 (Eq. 36, 37)
                Hessian = H' * R_IMU_inv * H;
                grad = H' * R_IMU_inv * r;
                
                % 求解更新增量并引入正则阻尼阻绝NaN (Eq. 38)
                step = obj.safe_solve(Hessian, grad, obj.mu);
                
                s_IMU = s_IMU - step;
                
                % 如果发生数值奇异，提前终止避免扩散
                if any(isnan(s_IMU)) || any(isinf(s_IMU))
                    s_IMU = zeros(9, 1);
                    break;
                end
            end
            
            % 再次提纯 s_IMU，确保迭代结果安全
            s_IMU = obj.sanitize_vector(s_IMU);
            
            % --- 2.3 提取局部测量等效信息并融合 ---
            % 提取收敛点对应的雅可比并计算 Lambda 和 lambda
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
            
            % 映射回 15 维全局先验切空间 (Eq. 42, 43)
            % 选择矩阵 pi_IMU 的转置转回 15 维
            Lambda_full = zeros(15, 15);
            lambda_full = zeros(15, 1);
            
            % 9D 对应 15D 中的索引: da->7:9, dphi->10:12, domega->13:15
            idx_15 = 7:15;
            Lambda_full(idx_15, idx_15) = Lambda_IMU;
            lambda_full(idx_15) = lambda_IMU;
            
            % 独立部分累加，相关部分不变 (Eq. 44, 45)
            obj.I_indep = obj.sanitize_matrix(obj.I_indep + Lambda_full);
            
            % 计算后验协方差与切空间扰动更新 (Eq. 46-48)
            Sigma_post = obj.safe_inv(obj.I_indep + obj.I_dep);
            delta_theta = Sigma_post * lambda_full;
            
            % 状态流形回射
            obj.state.p = obj.state.p + delta_theta(1:3);
            obj.state.v = obj.state.v + delta_theta(4:6);
            obj.state.a = obj.state.a + delta_theta(7:9);
            obj.state.R = obj.robust_orthonormalize(obj.state.R * obj.so3_exp_safe(delta_theta(10:12)));
            obj.state.omega = obj.state.omega + delta_theta(13:15);
        end
        
        %% 3.0 UWB 独立/相关信息投影 (边缘化输出接口 - 邻居车辆调用)
        function [p_est, I_pos_indep, I_pos_dep] = get_marginalized_position_info(obj)
            % 邻居节点通过此接口提取并广播其3D位置边缘化信息
            p_est = obj.state.p;
            
            Sigma_j = obj.safe_inv(obj.I_indep + obj.I_dep);
            
            % 位置对应前3维，提取对应的位置协方差
            Sigma_pos = Sigma_j(1:3, 1:3);
            I_pos_total = obj.safe_inv(Sigma_pos);
            
            % 构造投影矩阵 W_pos (Eq. 49)
            % W_pos = I_pos_total * pi_p * Sigma
            pi_p = [eye(3), zeros(3, 12)];
            W_pos = I_pos_total * pi_p * Sigma_j;
            
            % 计算投影边际化后的独立/相关信息 (Eq. 50, 51)
            I_pos_dep = W_pos * obj.I_dep * W_pos';
            I_pos_dep = 0.5 * (I_pos_dep + I_pos_dep');
            
            I_pos_indep = I_pos_total - I_pos_dep;
            I_pos_indep = 0.5 * (I_pos_indep + I_pos_indep');
            
            I_pos_indep = obj.sanitize_matrix(I_pos_indep);
            I_pos_dep = obj.sanitize_matrix(I_pos_dep);
        end
        
        % 3.1 - 3.4 去中心化低频 UWB 更新 (UWB Update - 10Hz)
        function obj = update_uwb(obj, anchor_ranges, anchor_positions, ...
                                  neighbor_ids, neighbor_positions, ...
                                  neighbor_I_indep, neighbor_I_dep, ...
                                  relative_ranges, sigma_s, sigma_z)
            % ... (前面的代码保持不变) ...
            
            K = length(anchor_ranges);
            M = length(neighbor_ids);
            
            if M == 0
                obj = obj.update_uwb_anchor_only(anchor_ranges, anchor_positions, sigma_s);
                return;
            end
            
            % --- 初始化 ADMM 迭代状态 ---
            s_dim = 3 * (M + 1);
            s_admm = zeros(s_dim, 1);
            
            for i = 1:M
                nid = neighbor_ids(i);
                if ~isKey(obj.lambda_local, nid)
                    obj.lambda_local(nid) = zeros(3, 1);
                    obj.lambda_remote(nid) = zeros(3, 1);
                end
            end
            
            rho = 1.0;
            max_admm_iter = 10;
            dual_tol = 1e-3;
            
            dp_neigh_neigh = zeros(3, M); 
            dp_neigh_self = zeros(3, M);  
            
            % --- ADMM 外层循环 ---
            for admm_k = 1:max_admm_iter
                s_admm = obj.solve_admm_primal(s_admm, anchor_ranges, anchor_positions, ...
                                               neighbor_positions, relative_ranges, ...
                                               sigma_s, sigma_z, rho, neighbor_ids, ...
                                               dp_neigh_neigh, dp_neigh_self);
                
                % 模拟通信获取邻居估计值
                for i = 1:M
                    dp_neigh_neigh(:, i) = zeros(3, 1); 
                    dp_neigh_self(:, i) = s_admm(1:3);  
                end
                
                dp_self_est = s_admm(1:3);
                for i = 1:M
                    nid = neighbor_ids(i);
                    dp_self_neigh = s_admm(3*i + (1:3)); 
                    obj.lambda_local(nid) = obj.lambda_local(nid) + rho * (dp_self_neigh - dp_neigh_neigh(:, i));
                    obj.lambda_remote(nid) = obj.lambda_remote(nid) + rho * (dp_neigh_self(:, i) - dp_self_est);
                end
                
                dual_residual = 0;
                for i = 1:M
                    nid = neighbor_ids(i);
                    dual_residual = dual_residual + norm(dp_self_est - dp_neigh_self(:, i));
                end
                if dual_residual < dual_tol
                    break;
                end
            end
            
            % --- 3.2 构造收敛后的联合测量信息矩阵 ---
            s_star = s_admm;
            p_self_converged = obj.state.p + s_star(1:3);
            
            Xi_aa = zeros(3, 3);
            for k = 1:K
                u_k = (p_self_converged - anchor_positions(k, :)') / ...
                      max(norm(p_self_converged - anchor_positions(k, :)'), 1e-6);
                Xi_aa = Xi_aa + (1 / sigma_s^2) * (u_k * u_k');
            end
            
            Xi_ii_11 = zeros(3, 3);
            Xi_ii_12 = zeros(3, 3 * M);
            Xi_ii_22 = zeros(3 * M, 3 * M);
            
            for i = 1:M
                dp_self_neigh = s_star(3*i + (1:3)); % 此处索引已修正
                p_neigh_converged = neighbor_positions(i, :)' + dp_self_neigh;
                
                u_ij = (p_self_converged - p_neigh_converged) / ...
                       max(norm(p_self_converged - p_neigh_converged), 1e-6);
                
                Xi_block = (1 / sigma_z^2) * (u_ij * u_ij');
                
                Xi_ii_11 = Xi_ii_11 + Xi_block;
                Xi_ii_12(:, 3*(i-1)+(1:3)) = -Xi_block;
                Xi_ii_22(3*(i-1)+(1:3), 3*(i-1)+(1:3)) = Xi_block;
            end
            
            % --- 3.3 基于 SCI 的舒尔补边缘化 (此处为报错修改点) ---
            omega_self = 0.5;
            omega_neigh = 0.5 / M; 
            
            P22_SCI_inv = zeros(3 * M, 3 * M);
            s_neigh_mle = zeros(3 * M, 1);
            for i = 1:M
                idx = 3*(i-1)+(1:3); % 对应 P22_SCI_inv 块的索引 (1:3, 4:6...)
                
                % SCI加权先验
                Lambda_j_SCI = neighbor_I_indep{i} + omega_neigh * neighbor_I_dep{i};
                P22_SCI_inv(idx, idx) = Lambda_j_SCI;
                
                % 【修正项】：s_neigh_mle 的对应部分来自 s_star 的第 i 个邻居槽位
                % s_star = [dp_self (1:3); dp_neigh1 (4:6); dp_neigh2 (7:9)]
                s_neigh_mle(idx) = s_star(3*i + (1:3)); 
            end
            
            % 舒尔补计算
            P22_temp_inv = obj.safe_inv(Xi_ii_22 + P22_SCI_inv);
            
            % 边缘化信息矩阵
            Lambda_t_anc = Xi_aa;
            Lambda_t_int = Xi_ii_11 - Xi_ii_12 * P22_temp_inv * Xi_ii_12';
            
            % 边缘化信息向量 (Eq. 77)
            s_self_mle = s_star(1:3);
            lambda_t_anc = Lambda_t_anc * s_self_mle;
            lambda_t_int = Lambda_t_int * s_self_mle + Xi_ii_12 * P22_temp_inv * P22_SCI_inv * s_neigh_mle;
            
            % --- 3.4 局部 MLKF 信息融合 ---
            Lambda_anc_full = zeros(15, 15); Lambda_anc_full(1:3, 1:3) = Lambda_t_anc;
            lambda_anc_full = zeros(15, 1);  lambda_anc_full(1:3) = lambda_t_anc;
            
            Lambda_int_full = zeros(15, 15); Lambda_int_full(1:3, 1:3) = Lambda_t_int;
            lambda_int_full = zeros(15, 1);  lambda_int_full(1:3) = lambda_t_int;
            
            obj.I_indep = obj.sanitize_matrix(obj.I_indep + Lambda_anc_full);
            obj.I_dep = obj.sanitize_matrix(omega_self * obj.I_dep + Lambda_int_full);
            
            Sigma_post = obj.safe_inv(obj.I_indep + obj.I_dep);
            delta_theta = Sigma_post * (lambda_anc_full + lambda_int_full);
            
            % 状态更新
            obj.state.p = obj.state.p + delta_theta(1:3);
            obj.state.v = obj.state.v + delta_theta(4:6);
            obj.state.a = obj.state.a + delta_theta(7:9);
            obj.state.R = obj.robust_orthonormalize(obj.state.R * obj.so3_exp_safe(delta_theta(10:12)));
            obj.state.omega = obj.state.omega + delta_theta(13:15);
        end
        
        %% UWB定位外部调试接口
        function [dp_self, dp_neigh] = get_admm_primal_estimates(obj, s_admm, neighbor_ids)
            % 获取当前 ADMM 的一阶解以便外部网络分发
            dp_self = s_admm(1:3);
            M = length(neighbor_ids);
            dp_neigh = zeros(3, M);
            for i = 1:M
                dp_neigh(:, i) = s_admm(3*i + (1:3));
            end
        end
    end
    
    %% 内部辅助/数值安全函数 (NaN防线)
    methods (Access = private)
        %% ADMM Primal Update 求解器 (内层 GN 迭代)
        function s_admm = solve_admm_primal(obj, s_admm_init, anchor_ranges, anchor_positions, ...
                                            neighbor_positions, relative_ranges, ...
                                            sigma_s, sigma_z, rho, neighbor_ids, ...
                                            dp_neigh_neigh, dp_neigh_self)
            K = length(anchor_ranges);
            M = length(neighbor_ids);
            s_dim = length(s_admm_init);
            s_admm = s_admm_init;
            
            max_inner_gn = 5;
            for inner_iter = 1:max_inner_gn
                % 1. 计算线段向量及测量残差
                p_self_est = obj.state.p + s_admm(1:3);
                
                % 基站残差与LOS
                r_anc = zeros(K, 1);
                u_anc = zeros(3, K);
                for k = 1:K
                    vec = p_self_est - anchor_positions(k, :)';
                    dist = max(norm(vec), 1e-6);
                    u_anc(:, k) = vec / dist;
                    r_anc(k) = anchor_ranges(k) - dist;
                end
                
                % 邻居残差与LOS
                r_int = zeros(M, 1);
                u_int = zeros(3, M);
                for i = 1:M
                    dp_self_neigh = s_admm(3*i + (1:3));
                    p_neigh_est = neighbor_positions(i, :)' + dp_self_neigh;
                    vec = p_self_est - p_neigh_est;
                    dist = max(norm(vec), 1e-6);
                    u_int(:, i) = vec / dist;
                    r_int(i) = relative_ranges(i) - dist;
                end
                
                % 2. 构造拉格朗日梯度向量 g_L (Eq. 61-63)
                g_L = zeros(s_dim, 1);
                
                % 本车位置梯度分量 (Eq. 62)
                g_dp11 = zeros(3, 1);
                for k = 1:K
                    g_dp11 = g_dp11 - (1 / sigma_s^2) * u_anc(:, k) * r_anc(k);
                end
                for i = 1:M
                    nid = neighbor_ids(i);
                    g_dp11 = g_dp11 - (1 / sigma_z^2) * u_int(:, i) * r_int(i) ...
                             + (-obj.lambda_remote(nid) - rho * (dp_neigh_self(:, i) - s_admm(1:3)));
                end
                g_L(1:3) = g_dp11;
                
                % 邻车估计梯度分量 (Eq. 63)
                for i = 1:M
                    nid = neighbor_ids(i);
                    dp_self_neigh = s_admm(3*i + (1:3));
                    g_dp1j = (1 / sigma_z^2) * u_int(:, i) * r_int(i) ...
                             + obj.lambda_local(nid) + rho * (dp_self_neigh - dp_neigh_neigh(:, i));
                    g_L(3*i + (1:3)) = g_dp1j;
                end
                
                % 3. 构造近似 Hessian H_L (Eq. 64)
                H_L = zeros(s_dim, s_dim);
                
                % 基站观测雅可比项
                for k = 1:K
                    H_k = zeros(1, s_dim);
                    H_k(1:3) = -u_anc(:, k)';
                    H_L = H_L + (1 / sigma_s^2) * (H_k' * H_k);
                end
                
                % 邻车协同观测雅可比项 (Eq. 65)
                for i = 1:M
                    H_int = zeros(1, s_dim);
                    H_int(1:3) = -u_int(:, i)';
                    H_int(3*i + (1:3)) = u_int(:, i)';
                    H_L = H_L + (1 / sigma_z^2) * (H_int' * H_int);
                end
                
                % ADMM 罚项块 H_penalty (Eq. 66)
                D_penalty = zeros(s_dim, s_dim);
                D_penalty(1:3, 1:3) = (M * rho) * eye(3); % M个惩罚流
                for i = 1:M
                    idx = 3*i + (1:3);
                    D_penalty(idx, idx) = rho * eye(3);
                end
                H_L = H_L + D_penalty;
                
                % 4. 计算更新步并引入极小阻尼 (Eq. 67)
                step = obj.safe_solve(H_L, g_L, obj.mu);
                s_admm = s_admm - step;
                
                if any(isnan(s_admm)) || any(isinf(s_admm))
                    s_admm = s_admm_init;
                    break;
                end
            end
            s_admm = obj.sanitize_vector(s_admm);
        end
        
        %% 无邻车时退化的单车UWB位置更新 (Anchor Only)
        function obj = update_uwb_anchor_only(obj, anchor_ranges, anchor_positions, sigma_s)
            K = length(anchor_ranges);
            s_pos = zeros(3, 1);
            p_prior = obj.state.p;
            
            % 纯基站 Gauss-Newton
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
            
            % 更新
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
        
        %% 数值安全：SVD强制三维正交归一化 (防止由于姿态漂移引入NaN)
        function R_orth = robust_orthonormalize(~, R)
            if any(isnan(R(:))) || any(isinf(R(:)))
                R_orth = eye(3);
                return;
            end
            [U, ~, V] = svd(R);
            R_orth = U * V';
            % 防止镜像旋转
            if det(R_orth) < 0
                R_orth = U * diag([1, 1, -1]) * V';
            end
        end
        
        %% 数值安全：安全的李群指数映射 (防止 w=0 时零除产生NaN)
        function R = so3_exp_safe(obj, phi)
            theta = norm(phi);
            phi_skew = obj.skew_matrix(phi);
            if theta < 1e-6
                R = eye(3) + phi_skew;
            else
                R = eye(3) + (sin(theta)/theta) * phi_skew + ((1 - cos(theta))/theta^2) * (phi_skew * phi_skew);
            end
        end
        
        %% 数值安全：安全的右雅可比，泰勒级数截断保护 (关键防御NaN点)
        function Jr = so3_right_jacobian_safe(obj, phi)
            theta = norm(phi);
            phi_skew = obj.skew_matrix(phi);
            if theta < 1e-3
                % 在 1e-3 以下采用泰勒展开，防止数值下溢除以 theta 造成 NaN
                Jr = eye(3) - 0.5 * phi_skew + (1/6) * (phi_skew * phi_skew);
            else
                Jr = eye(3) - ((1 - cos(theta))/theta^2) * phi_skew + ((theta - sin(theta))/theta^3) * (phi_skew * phi_skew);
            end
        end
        
        %% 数值安全：逆求解器保护
        function x = safe_solve(~, A, b, reg)
            % 叠加小正则化项防止阻尼退化奇异
            A_reg = A + reg * eye(size(A));
            x = A_reg \ b;
            if any(isnan(x)) || any(isinf(x))
                x = pinv(A) * b; % 伪逆二次防御
            end
        end
        
        %% 数值安全：求逆保护
        function A_inv = safe_inv(~, A)
            A_reg = A + 1e-11 * eye(size(A));
            A_inv = inv(A_reg);
            if any(isnan(A_inv(:))) || any(isinf(A_inv(:)))
                A_inv = pinv(A);
            end
        end
        
        %% 数值安全：过滤 NaN/Inf 矩阵，强制收敛边界
        function M_out = sanitize_matrix(~, M_in)
            M_out = M_in;
            M_out(isnan(M_out)) = 0;
            M_out(isinf(M_out)) = 0;
            % 保证对称性
            M_out = 0.5 * (M_out + M_out');
        end
        
        %% 数值安全：过滤 NaN/Inf 向量
        function v_out = sanitize_vector(~, v_in)
            v_out = v_in;
            v_out(isnan(v_out)) = 0;
            v_out(isinf(v_out)) = 0;
        end
        
        %% 伴随与反对称辅助函数
        function skew_R = skew_matrix(~, v)
            skew_R = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
        end
    end
end