classdef RBPF_1
    % 15D Rao-Blackwellized Particle Filter (经典版, 与 DMLKF/DEKF 同一套过程噪声假设)
    % 状态划分: 姿态 R 用SO(3)粒子采样, [p;v;a;w] 12维每个粒子用解析KF跟踪(P_l)
   
    properties
        id;
        Np;
        tau;
        particles;  % 字段: R(3x3), x_l(12x1=[p;v;a;w]), P_l(12x12), w
        state;      % .p .v .a .R .omega
        g_vec = [0; 0; -9.81];

        % --- IMU 测量噪声 (与数据生成端真实传感器噪声一致) ---
        Sigma_a;    % 3x3 加速度计噪声协方差
        Sigma_w;    % 3x3 陀螺仪噪声协方差

        % --- 过程噪声 (与 DMLKF 的 Q_15d 完全一致, 保证公平对比) ---
        Q_R;   % 3x3 姿态额外过程噪声(用于Σφ)
        Ql;    % 12x12 = blkdiag(Qp, Qv, Qa, Qw)

        % --- 抗贫化 (仅对姿态roughening，位置/速度/加速度/角速度靠各自解析P_l天然保有不确定性) ---
        neff_ratio = 0.4;
        rough_coeff = 0.4;
        min_att_jitter = 0.001;
    end

    methods
        function obj = RBPF_1(id, init_state, tau, Np)
            obj.id = id;
            obj.tau = tau;
            obj.Np = Np;

            obj.Sigma_a = (0.03)^2 * eye(3);
            obj.Sigma_w = (0.003)^2 * eye(3);

            Q_p = (0.0001)^2 * eye(3);
            Q_v = (0.001)^2 * eye(3);
            Q_a = (0.00025)^2 * eye(3);
            Q_w = (0.00025)^2 * eye(3);
            obj.Q_R = (0.0001)^2 * eye(3);
            obj.Ql = blkdiag(Q_p, Q_v, Q_a, Q_w) + 1e-12 * eye(12);

            % 初始协方差 (与 DMLKF 一致: p/v var=0.01, a/w var=0.005, 姿态1deg)
            P_R0 = (1 * pi / 180)^2 * eye(3);
            P_l0 = blkdiag(0.01*eye(3), 0.01*eye(3), 0.005*eye(3), 0.005*eye(3));

            L_R0 = obj.robust_chol(P_R0);
            obj.particles = repmat(struct('R', eye(3), 'x_l', zeros(12,1), 'P_l', P_l0, 'w', 1/Np), Np, 1);
            for k = 1 : Np
                dphi = L_R0 * randn(3, 1);
                obj.particles(k).R = obj.robust_orthonormalize(init_state.R * obj.so3_exp(dphi));
                obj.particles(k).x_l = [init_state.p; init_state.v; init_state.a; init_state.omega];
                obj.particles(k).P_l = P_l0;
                obj.particles(k).w = 1 / Np;
            end

            obj = obj.update_mean_state();
        end

        %% 预测步: SO3姿态边缘化采样(式4-11) + 12维线性态解析KF传播(式12-15)
        function obj = predict(obj)
            t = obj.tau;
            F = eye(12);
            F(1:3, 4:6) = t * eye(3);
            F(1:3, 7:9) = 0.5 * t^2 * eye(3);
            F(4:6, 7:9) = t * eye(3);

            for k = 1 : obj.Np
                R_k = obj.particles(k).R;
                xl = obj.particles(k).x_l;
                Pl = obj.particles(k).P_l;

                omega_hat = xl(10:12);
                P_omega = Pl(10:12, 10:12);

                theta_nom = t * omega_hat;
                Jr = obj.right_jacobian(theta_nom);
                Sigma_phi = t^2 * (Jr * P_omega * Jr') + obj.Q_R;
                Sigma_phi = 0.5 * (Sigma_phi + Sigma_phi');
                L_phi = obj.robust_chol(Sigma_phi);
                dphi = L_phi * randn(3, 1);

                R_next = obj.robust_orthonormalize(R_k * obj.so3_exp(theta_nom) * obj.so3_exp(dphi));

                xl_pred = F * xl;
                Pl_pred = F * Pl * F' + obj.Ql;
                Pl_pred = 0.5 * (Pl_pred + Pl_pred') + 1e-9 * eye(12);

                obj.particles(k).R = R_next;
                obj.particles(k).x_l = xl_pred;
                obj.particles(k).P_l = Pl_pred;
            end

            obj = obj.update_mean_state();
        end

        %% IMU更新: 加速度分量决定姿态粒子权重(式25-31) + 6维解析KF更新线性态(式16-24)
        function obj = update_imu(obj, raw_acc, raw_gyro, bias_a, bias_w)
            acc_tilde = raw_acc - bias_a;
            gyro_tilde = raw_gyro - bias_w;

            log_w = log([obj.particles.w] + 1e-300);

            for k = 1 : obj.Np
                R_k = obj.particles(k).R;
                xl = obj.particles(k).x_l;
                Pl = obj.particles(k).P_l;

                % --- 姿态粒子权重: 仅用加速度分量, 用更新前(先验)的Pa ---
                a_hat = xl(7:9);
                Pa_prior = Pl(7:9, 7:9);
                y_hat_a = R_k' * (a_hat - obj.g_vec);
                S_a = R_k' * Pa_prior * R_k + obj.Sigma_a;
                S_a = 0.5 * (S_a + S_a') + 1e-9 * eye(3);
                r_a = acc_tilde - y_hat_a;
                log_w(k) = log_w(k) - 0.5 * log((2*pi)^3 * det(S_a)) - 0.5 * r_a' * (S_a \ r_a);

                % --- 6维解析KF更新线性态 ---
                H = zeros(6, 12);
                H(1:3, 7:9) = R_k';
                H(4:6, 10:12) = eye(3);
                u = [-R_k' * obj.g_vec; zeros(3, 1)];
                R_imu = blkdiag(obj.Sigma_a, obj.Sigma_w);

                y_hat = H * xl + u;
                r = [acc_tilde; gyro_tilde] - y_hat;
                S = H * Pl * H' + R_imu;
                S = 0.5 * (S + S') + 1e-9 * eye(6);
                Kg = Pl * H' / S;

                xl = xl + Kg * r;
                Pl = (eye(12) - Kg * H) * Pl;
                Pl = 0.5 * (Pl + Pl') + 1e-9 * eye(12);

                obj.particles(k).x_l = xl;
                obj.particles(k).P_l = Pl;
            end

            obj = obj.normalize_and_resample(log_w);
            obj = obj.update_mean_state();
        end

        %% 位置边缘化信息: 混合高斯 (粒子内解析协方差 + 粒子间散布), 式32-33
        function [p_est, Sigma_pos] = get_marginalized_position_info(obj)
            p_est = obj.state.p;
            Sigma_pos = zeros(3, 3);
            for k = 1 : obj.Np
                dp = obj.particles(k).x_l(1:3) - p_est;
                Sigma_pos = Sigma_pos + obj.particles(k).w * (obj.particles(k).P_l(1:3,1:3) + dp * dp');
            end
            Sigma_pos = 0.5 * (Sigma_pos + Sigma_pos') + 1e-9 * eye(3);
        end

        %% UWB更新: 逐条序贯解析KF更新(式34-52) + 权重归一化 + 重采样(式53-54)
        function obj = apply_uwb_update(obj, anchor_ranges, anchor_positions, sigma_s, ...
                neighbor_ids, neighbor_positions, neighbor_Sigma_pos, ...
                relative_ranges, sigma_z, gamma)
            K = length(anchor_ranges);
            M = length(neighbor_ids);
            if K + M == 0
                return;
            end

            sigma_s2 = sigma_s^2;
            sigma_z2 = sigma_z^2;

            log_lik = zeros(obj.Np, 1);
            for k = 1 : obj.Np
                xl = obj.particles(k).x_l;
                Pl = obj.particles(k).P_l;
                ll = 0;

                for a = 1 : K
                    d_vec = xl(1:3) - anchor_positions(a, :)';
                    dist = max(norm(d_vec), 1e-6);
                    u = d_vec / dist;
                    H = zeros(1, 12);
                    H(1:3) = u';

                    z_resid = anchor_ranges(a) - dist;
                    S = H * Pl * H' + sigma_s2;
                    S = max(S, 1e-9);
                    Kg = Pl * H' / S;

                    xl = xl + Kg * z_resid;
                    Pl = (eye(12) - Kg * H) * Pl;
                    Pl = 0.5 * (Pl + Pl') + 1e-9 * eye(12);

                    ll = ll - 0.5 * log(2 * pi * S) - 0.5 * z_resid^2 / S;
                end

                for m = 1 : M
                    p_neigh = neighbor_positions(m, :)';
                    Sig_neigh = neighbor_Sigma_pos{m};
                    Sig_neigh = 0.5 * (Sig_neigh + Sig_neigh') + 1e-9 * eye(3);

                    d_vec = xl(1:3) - p_neigh;
                    dist = max(norm(d_vec), 1e-6);
                    e_ij = d_vec / dist;
                    sig2_equiv = sigma_z2 + gamma * (e_ij' * Sig_neigh * e_ij);
                    H = zeros(1, 12);
                    H(1:3) = e_ij';

                    z_resid = relative_ranges(m) - dist;
                    S = H * Pl * H' + sig2_equiv;
                    S = max(S, 1e-9);
                    Kg = Pl * H' / S;

                    xl = xl + Kg * z_resid;
                    Pl = (eye(12) - Kg * H) * Pl;
                    Pl = 0.5 * (Pl + Pl') + 1e-9 * eye(12);

                    ll = ll - 0.5 * log(2 * pi * S) - 0.5 * z_resid^2 / S;
                end

                obj.particles(k).x_l = xl;
                obj.particles(k).P_l = Pl;
                log_lik(k) = ll;
            end

            max_ll = max(log_lik);
            if isnan(max_ll) || isinf(max_ll) || max_ll < -40
                fprintf(2, '[警告] 节点 %d 全体粒子似然过低 (max_log=%.2f)，疑似跑飞或严重退化！\n', obj.id, max_ll);
            end

            log_w = log([obj.particles.w] + 1e-300) + log_lik';
            obj = obj.normalize_and_resample(log_w);
            obj = obj.update_mean_state();
        end
    end

    methods (Access = private)
        function obj = normalize_and_resample(obj, log_w)
            max_log_w = max(log_w);
            w_shifted = exp(log_w - max_log_w) + 1e-300;
            w_sum = sum(w_shifted);
            if w_sum > 0 && isfinite(w_sum)
                for k = 1 : obj.Np
                    obj.particles(k).w = w_shifted(k) / w_sum;
                end
            else
                fprintf(2, '[警告] 节点 %d 权重归一化失败，重置为均匀权重！\n', obj.id);
                for k = 1 : obj.Np
                    obj.particles(k).w = 1 / obj.Np;
                end
            end

            N_eff = 1 / sum([obj.particles.w].^2);
            if N_eff < obj.neff_ratio * obj.Np
                obj = obj.resample();
                obj = obj.roughen();
            end
        end

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

        % 重采样后姿态roughening: 位置/速度/加速度/角速度不额外扰动，
        % 因为各粒子解析P_l已保有自身不确定性，未来多样性由下一次predict中
        % 不同的姿态采样、以及IMU更新中不同R_k对a/w的观测传导过去
        function obj = roughen(obj)
            phi_all = zeros(obj.Np, 3);
            mean_R = obj.state.R;
            for k = 1 : obj.Np
                phi_all(k, :) = obj.so3_log(mean_R' * obj.particles(k).R)';
            end
            std_phi = std(phi_all, 0, 1);
            h = obj.rough_coeff * obj.Np^(-1/3);
            jit = max(h * std_phi, obj.min_att_jitter * ones(1, 3));

            for k = 1 : obj.Np
                dphi = (jit .* randn(1, 3))';
                obj.particles(k).R = obj.robust_orthonormalize(obj.particles(k).R * obj.so3_exp(dphi));
            end
        end

        function obj = update_mean_state(obj)
            mean_xl = zeros(12, 1);
            for k = 1 : obj.Np
                mean_xl = mean_xl + obj.particles(k).w * obj.particles(k).x_l;
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

            obj.state.R = mean_R;
            obj.state.p = mean_xl(1:3);
            obj.state.v = mean_xl(4:6);
            obj.state.a = mean_xl(7:9);
            obj.state.omega = mean_xl(10:12);

            if any(isnan(mean_xl)) || any(isinf(mean_xl))
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

        function J = right_jacobian(theta)
            th = norm(theta);
            sk = RBPF_1.skew(theta);
            if th < 1e-6
                J = eye(3) - 0.5 * sk + (1/6) * (sk * sk);
            else
                J = eye(3) - ((1 - cos(th)) / th^2) * sk + ((th - sin(th)) / th^3) * (sk * sk);
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