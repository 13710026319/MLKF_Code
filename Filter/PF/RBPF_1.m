classdef RBPF_1
    % Rao-Blackwellized Particle Filter for 15D Cooperative Localization

    properties
        id;
        Np;
        dt;
        particles;   % 字段: p(3x1), v(3x1), R(3x3), a(3x1), omega(3x1), P_l(12x12), w
        state;       % .p .v .R .a .omega
        g = [0; 0; -9.81];

        % --- IMU 量测噪声 (用于 update_imu 中 R_IMU 的构造, 式11-17) ---
        sigma_na = 0.045;    % 加速度计量测噪声标准差 (m/s^2)
        sigma_nw = 0.0045;   % 陀螺仪量测噪声标准差 (rad/s)
        Sigma_IMU;           % 6x6 = blkdiag(sigma_na^2*I3, sigma_nw^2*I3)

        % --- 12D 解析线性态 [v;theta;a;omega] 过程噪声 (式9, Ql=blkdiag(Qv,Qtheta,Qa,Qomega)) ---
        sigma_v_proc     = 0.001;
        sigma_theta_proc = 0.0001;
        sigma_a_proc     = 0.00025;
        sigma_omega_proc = 0.00025;
        Ql;    % 12x12
        F;     % 12x12 误差态转移矩阵 (式8, dt恒定故常数矩阵, 构造时预计算)

        % --- 位置粒子采样维 (x_s) 过程噪声 (式4) ---
        
        sigma_p_proc = 0.001;   % 起始经验值(m)，比KF量级大两个数量级，需实测调优
        Qp;    % 3x3

        % --- 抗贫化：仅对位置做roughening (位置是本模型的粒子采样维)，
        %     v/R/a/omega 靠各自解析 P_l 天然保有不确定性，无需额外扰动 ---
        neff_ratio       = 0.4;    % 重采样触发阈值比例 N_eff < neff_ratio*Np (对应式30, Nthresh=0.5Np附近)
        rough_coeff      = 0.4;    % roughening 带宽系数 (乘在"当前粒子散布"上)
        jitter_floor_mult = 1.0;   % roughening 下限相对 Qp 自身标准差的倍数
                                    % (下限与Qp绑定，而不是一个固定小常数，
                                    %  避免粒子塌缩后 std_p→0 导致抖动也跟着塌缩、无法翻身)
    end

    methods
        function obj = RBPF_1(id, init_state, dt_imu, Np)
            obj.id = id;
            obj.dt = dt_imu;
            obj.Np = Np;

            obj.Sigma_IMU = blkdiag(obj.sigma_na^2 * eye(3), obj.sigma_nw^2 * eye(3));

            obj.Ql = blkdiag( ...
                obj.sigma_v_proc^2 * eye(3), ...
                obj.sigma_theta_proc^2 * eye(3), ...
                obj.sigma_a_proc^2 * eye(3), ...
                obj.sigma_omega_proc^2 * eye(3)) + 1e-12 * eye(12);

            obj.Qp = obj.sigma_p_proc^2 * eye(3) + 1e-12 * eye(3);

            % 误差态转移矩阵 (式8): 状态顺序 [v; theta; a; omega]
            %   v_{t+1}     = v_t + dt*a_t
            %   theta_{t+1} = theta_t + dt*omega_t
            %   a, omega    保持 (随机游走)
            obj.F = [ eye(3),  zeros(3), obj.dt*eye(3), zeros(3);
                      zeros(3), eye(3),  zeros(3),       obj.dt*eye(3);
                      zeros(3), zeros(3), eye(3),        zeros(3);
                      zeros(3), zeros(3), zeros(3),      eye(3) ];

            % 初始误差协方差 P_l0 (12x12): [v; theta; a; omega]
            P_l0 = blkdiag( ...
                diag(0.01 * ones(1, 3)), ...              % 速度
                diag((1 * pi / 180)^2 * ones(1, 3)), ...  % 姿态 1deg
                diag(0.005 * ones(1, 3)), ...              % 加速度
                diag(0.005 * ones(1, 3)) ...               % 角速度
                );

            % 初始位置粒子撒开 (位置是粒子采样维，需要初始弥散来体现不确定性)
            P_p0 = diag(0.01 * ones(1, 3));
            L_p0 = obj.robust_chol(P_p0);

            obj.particles = repmat(struct('p', zeros(3,1), 'v', zeros(3,1), 'R', eye(3), ...
                'a', zeros(3,1), 'omega', zeros(3,1), 'P_l', P_l0, 'w', 1/Np), Np, 1);

            for k = 1 : Np
                dp = L_p0 * randn(3, 1);
                obj.particles(k).p     = init_state.p + dp;
                obj.particles(k).v     = init_state.v;
                obj.particles(k).R     = init_state.R;
                obj.particles(k).a     = init_state.a;
                obj.particles(k).omega = init_state.omega;
                obj.particles(k).P_l   = P_l0;
                obj.particles(k).w     = 1 / Np;
            end

            obj = obj.update_mean_state();
        end

        %% 预测步: 位置粒子采样(式4) + [v;R;a;omega]解析KF传播(式5-10)
        %  注意: 无需外部传入IMU，传播完全依赖滤波器自身当前的v/a/R/omega估计
        function obj = predict(obj)
            Lp = obj.robust_chol(obj.Qp);

            for k = 1 : obj.Np
                v_k     = obj.particles(k).v;
                R_k     = obj.particles(k).R;
                a_k     = obj.particles(k).a;
                omega_k = obj.particles(k).omega;

                % A. 位置粒子采样 (式4)
                wp = Lp * randn(3, 1);
                obj.particles(k).p = obj.particles(k).p + v_k * obj.dt + 0.5 * a_k * obj.dt^2 + wp;

                % B. 解析线性态传播 (式5-10)
                v_pred     = v_k + a_k * obj.dt;
                R_pred     = obj.robust_orthonormalize(R_k * obj.so3_exp(omega_k * obj.dt));
                a_pred     = a_k;
                omega_pred = omega_k;

                P_l_pred = obj.F * obj.particles(k).P_l * obj.F' + obj.Ql;
                P_l_pred = 0.5 * (P_l_pred + P_l_pred') + 1e-9 * eye(12);

                obj.particles(k).v     = v_pred;
                obj.particles(k).R     = R_pred;
                obj.particles(k).a     = a_pred;
                obj.particles(k).omega = omega_pred;
                obj.particles(k).P_l   = P_l_pred;
            end

            obj = obj.update_mean_state();
        end

        %% IMU更新步: 对[v;R;a;omega]做EKF式量测更新 (式11-20)
        %  后两个入参为占位符 (与DMLKF/DEKF的update_imu签名保持一致，本类不使用)
        function obj = update_imu(obj, imu_acc, imu_gyro, ~, ~)
            R_IMU  = obj.Sigma_IMU;
            y_meas = [imu_acc; imu_gyro];

            for k = 1 : obj.Np
                R_hat     = obj.particles(k).R;
                a_hat     = obj.particles(k).a;
                omega_hat = obj.particles(k).omega;
                P_l       = obj.particles(k).P_l;

                % 预测量测 (式11)
                y_pred  = [R_hat' * (a_hat - obj.g); omega_hat];
                y_resid = y_meas - y_pred;   % 式12

                % 量测雅可比 H_IMU (式13): 列顺序 [dv, dtheta, da, domega]
                H = zeros(6, 12);
                H(1:3, 4:6) = obj.skew(R_hat' * (a_hat - obj.g));
                H(1:3, 7:9) = R_hat';
                H(4:6, 10:12) = eye(3);

                S = H * P_l * H' + R_IMU;                      % 式14
                S = 0.5 * (S + S') + 1e-9 * eye(6);
                K = P_l * H' / S;                                % 式15

                dx = K * y_resid;                                 % 式16: [dv;dtheta;da;domega]
                P_l_new = (eye(12) - K * H) * P_l;                % 式17
                P_l_new = 0.5 * (P_l_new + P_l_new') + 1e-9 * eye(12);

                % 名义状态回代 (式18-20)
                obj.particles(k).v     = obj.particles(k).v + dx(1:3);
                obj.particles(k).R     = obj.robust_orthonormalize(R_hat * obj.so3_exp(dx(4:6)));
                obj.particles(k).a     = obj.particles(k).a + dx(7:9);
                obj.particles(k).omega = obj.particles(k).omega + dx(10:12);
                obj.particles(k).P_l   = P_l_new;
            end

            obj = obj.update_mean_state();
        end

        %% 位置边缘化信息: 位置是粒子采样维，无解析协方差，
        %  故 Sigma_pos 完全由粒子间加权散布给出 (Rao-Blackwellized 部分的PF分量)
        function [p_est, Sigma_pos] = get_marginalized_position_info(obj)
            p_est = obj.state.p;
            Sigma_pos = zeros(3, 3);
            for k = 1 : obj.Np
                dp = obj.particles(k).p - p_est;
                Sigma_pos = Sigma_pos + obj.particles(k).w * (dp * dp');
            end
            Sigma_pos = 0.5 * (Sigma_pos + Sigma_pos') + 1e-9 * eye(3);
        end

        %% UWB更新: 纯粒子权重更新 (式21-32, bootstrap PF, 不修正状态) + 重采样(式33-37)
        function obj = apply_uwb_update(obj, anchor_ranges, anchor_positions, sigma_s, ...
                neighbor_ids, neighbor_positions, neighbor_Sigma_pos, ...
                relative_ranges, sigma_z, gamma)
            K = length(anchor_ranges);
            M = length(neighbor_ids);
            if K + M == 0
                return;
            end
            if gamma <= 1
                fprintf(2, '[警告] 节点 %d 传入的 gamma=%.3f <= 1，违反文档要求(gamma>1)，会导致相对测距对邻居协方差的膨胀不足，协同网络中的相关误差可能被放大而非保守抑制！\n', obj.id, gamma);
            end

            sigma_s2 = sigma_s^2;
            sigma_z2 = sigma_z^2;

            log_lik = zeros(obj.Np, 1);
            for k = 1 : obj.Np
                p_k = obj.particles(k).p;
                ll = 0;

                % A. 锚点测距似然 (式21-22, 26)
                for a = 1 : K
                    dist = norm(p_k - anchor_positions(a, :)');
                    z_resid = anchor_ranges(a) - dist;
                    S = max(sigma_s2, 1e-9);
                    ll = ll - 0.5 * log(2 * pi * S) - 0.5 * z_resid^2 / S;
                end

                % B. 节点间相对测距似然 (式23-26, 含数据近亲退化的协方差膨胀)
                for m = 1 : M
                    p_neigh = neighbor_positions(m, :)';
                    Sig_neigh = neighbor_Sigma_pos{m};
                    Sig_neigh = 0.5 * (Sig_neigh + Sig_neigh') + 1e-9 * eye(3);

                    d_vec = p_k - p_neigh;
                    dist = max(norm(d_vec), 1e-6);
                    e_ij = d_vec / dist;
                    sig2_equiv = sigma_z2 + gamma * (e_ij' * Sig_neigh * e_ij);

                    z_resid = relative_ranges(m) - dist;
                    S = max(sig2_equiv, 1e-9);
                    ll = ll - 0.5 * log(2 * pi * S) - 0.5 * z_resid^2 / S;
                end

                log_lik(k) = ll;
            end

            max_ll = max(log_lik);
            if isnan(max_ll) || isinf(max_ll) || max_ll < -40
                fprintf(2, '[警告] 节点 %d 全体粒子似然过低 (max_log=%.2f)，疑似跑飞或严重退化！\n', obj.id, max_ll);
            end

            % C. 权重更新与归一化 (式27-28)
            log_w = log([obj.particles.w] + 1e-300) + (log_lik' - max_ll);
            w_new = exp(log_w);
            w_sum = sum(w_new);
            if w_sum > 1e-12 && isfinite(w_sum)
                for k = 1 : obj.Np
                    obj.particles(k).w = w_new(k) / w_sum;
                end
            else
                fprintf(2, '[警告] 节点 %d 权重归一化失败，重置为均匀权重！\n', obj.id);
                for k = 1 : obj.Np
                    obj.particles(k).w = 1 / obj.Np;
                end
            end

            % D. 退化判定与重采样 (式29-33)
            N_eff = 1 / sum([obj.particles.w].^2);
            if N_eff < obj.neff_ratio * obj.Np
                obj = obj.resample();
                obj = obj.roughen();
            end

            obj = obj.update_mean_state();
        end
    end

    methods (Access = private)
        function obj = resample(obj)
            w_arr = [obj.particles.w];
            edges = [0, cumsum(w_arr)];
            edges(end) = 1.0;

            u1 = rand() / obj.Np;
            new_particles = obj.particles;
            idx = 1;
            for m = 1 : obj.Np
                u_m = u1 + (m - 1) / obj.Np;
                while u_m > edges(idx + 1) && idx < obj.Np
                    idx = idx + 1;
                end
                new_particles(m) = obj.particles(idx);
                new_particles(m).w = 1 / obj.Np;
            end
            obj.particles = new_particles;
        end

        % 重采样后位置roughening: 恢复克隆丢失的多样性 (v/R/a/omega 不额外扰动，
        % 因为各粒子的解析P_l已保有自身不确定性；且[v;R;a;omega]在本模型中
        % 由IMU量测确定性驱动，未来多样性完全由下一次predict中位置分支的
        % 随机采样w_p引入)
        function obj = roughen(obj)
            p_all = zeros(obj.Np, 3);
            for k = 1 : obj.Np
                p_all(k, :) = obj.particles(k).p';
            end
            std_p = std(p_all, 0, 1);
            h = obj.rough_coeff * obj.Np^(-1/3);
            floor_jit = obj.jitter_floor_mult * sqrt(diag(obj.Qp))';
            jit = max(h * std_p, floor_jit);

            for k = 1 : obj.Np
                dp = (jit .* randn(1, 3))';
                obj.particles(k).p = obj.particles(k).p + dp;
            end
        end

        function obj = update_mean_state(obj)
            mean_p     = zeros(3, 1);
            mean_v     = zeros(3, 1);
            mean_a     = zeros(3, 1);
            mean_omega = zeros(3, 1);
            for k = 1 : obj.Np
                w = obj.particles(k).w;
                mean_p     = mean_p + w * obj.particles(k).p;
                mean_v     = mean_v + w * obj.particles(k).v;
                mean_a     = mean_a + w * obj.particles(k).a;
                mean_omega = mean_omega + w * obj.particles(k).omega;
            end

            mean_R = obj.particles(1).R;
            for iter = 1 : 5
                dphi_sum = zeros(3, 1);
                for k = 1 : obj.Np
                    dphi_sum = dphi_sum + obj.particles(k).w * obj.so3_log(mean_R' * obj.particles(k).R);
                end
                mean_R = obj.robust_orthonormalize(mean_R * obj.so3_exp(dphi_sum));
                if norm(dphi_sum) < 1e-6
                    break;
                end
            end

            obj.state.p     = mean_p;
            obj.state.v     = mean_v;
            obj.state.R     = mean_R;
            obj.state.a     = mean_a;
            obj.state.omega = mean_omega;

            if any(isnan(mean_p)) || any(isinf(mean_p))
                fprintf(2, '[警告] 节点 %d 状态出现 NaN/Inf！\n', obj.id);
            end
        end
    end

    methods (Static)
        function R = so3_exp(v)
            theta = norm(v);
            if theta < 1e-8
                R = eye(3) + RBPF_1.skew(v);
            else
                axis = v / theta;
                R = eye(3) + sin(theta) * RBPF_1.skew(axis) + (1 - cos(theta)) * RBPF_1.skew(axis)^2;
            end
        end

        function phi = so3_log(R)
            val = max(-1, min(1, (trace(R) - 1) / 2));
            theta = acos(val);
            R_diff = R - R';
            if theta < 1e-6
                phi = 0.5 * RBPF_1.unskew(R_diff);
            else
                phi = (theta / (2 * sin(theta))) * RBPF_1.unskew(R_diff);
            end
        end

        function m = skew(v)
            m = [  0,   -v(3),  v(2);
                 v(3),    0,   -v(1);
                -v(2),   v(1),    0 ];
        end

        function v = unskew(M)
            v = [M(3,2); M(1,3); M(2,1)];
        end

        function R_orth = robust_orthonormalize(R)
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

        function L = robust_chol(A)
            A = 0.5 * (A + A');
            [L, p] = chol(A, 'lower');
            if p > 0
                L = diag(sqrt(max(diag(A), 1e-9)));
            end
        end
    end
end