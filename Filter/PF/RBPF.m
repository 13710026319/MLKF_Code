classdef RBPF
    % Rao-Blackwellized Particle Filter for 9D Cooperative Localization on SO(3)
  
    properties
        id;
        Np;
        dt;
        particles;   % 字段: R(3x3), x_l(6x1=[p;v]), P_l(6x6), w
        state;       % .R .p .v
        g = [0; 0; -9.81];

        % --- 过程噪声(由物理传感器噪声标准差推导，与 PF 保持同一套噪声假设) ---
        sigma_a = 0.045;   % 加速度计噪声标准差 (m/s^2)
        sigma_w = 0.0045;  % 陀螺仪噪声标准差 (rad/s)
        Q_R;   % 3x3 姿态过程噪声 (= sigma_w^2 * I)
        Ql;    % 6x6 = blkdiag(Q_p, Q_v)，由 sigma_a 通过运动学映射推得

        % --- 抗贫化(仅对姿态做roughening，位置/速度靠各自解析P_l天然保有不确定性) ---
        neff_ratio = 0.4;        % 重采样触发阈值比例 N_eff < neff_ratio*Np
        rough_coeff = 0.4;       % roughening 带宽系数
        min_att_jitter = 0.001;  % roughening 绝对下限 (rad)
    end

    methods
        function obj = RBPF(id, init_state, dt_imu, Np)
            obj.id = id;
            obj.dt = dt_imu;
            obj.Np = Np;

            obj.Q_R = obj.sigma_w^2 * eye(3);
            Q_v = (obj.sigma_a * obj.dt)^2 * eye(3);
            Q_p = (0.5 * obj.sigma_a * obj.dt^2)^2 * eye(3);
            obj.Ql = blkdiag(Q_p, Q_v) + 1e-12 * eye(6);

            % 初始协方差: 姿态1deg, 位置/速度std=0.1 (与 PF 一致)
            init_cov = blkdiag( ...
                diag((1 * pi / 180)^2 * ones(1, 3)), ...
                diag(0.01 * ones(1, 3)), ...
                diag(0.01 * ones(1, 3)) ...
                );
            P_R0 = init_cov(1:3, 1:3);
            P_l0 = init_cov(4:9, 4:9);

            L_R0 = obj.robust_chol(P_R0);
            obj.particles = repmat(struct('R', eye(3), 'x_l', zeros(6,1), 'P_l', P_l0, 'w', 1/Np), Np, 1);
            for k = 1 : Np
                dphi = L_R0 * randn(3, 1);
                obj.particles(k).R = obj.robust_orthonormalize(init_state.R * obj.so3_exp(dphi));
                obj.particles(k).x_l = [init_state.p; init_state.v];
                obj.particles(k).P_l = P_l0;
                obj.particles(k).w = 1 / Np;
            end

            obj = obj.update_mean_state();
        end

        %% 预测步: SO3流形姿态采样 + 位置/速度解析KF传播 (式4-14)
        function obj = predict(obj, imu_acc, imu_gyro)
            theta_nom = obj.dt * imu_gyro;
            Jr = obj.right_jacobian(theta_nom);
            Sigma_phi = obj.dt^2 * (Jr * obj.Q_R * Jr');
            Sigma_phi = 0.5 * (Sigma_phi + Sigma_phi');
            L_phi = obj.robust_chol(Sigma_phi);
            exp_theta_nom = obj.so3_exp(theta_nom);

            F = [eye(3), obj.dt * eye(3); zeros(3), eye(3)];

            for k = 1 : obj.Np
                R_k = obj.particles(k).R;

                dphi = L_phi * randn(3, 1);
                R_next = obj.robust_orthonormalize(R_k * exp_theta_nom * obj.so3_exp(dphi));

                a_world = R_k * imu_acc + obj.g;
                B_t = [0.5 * obj.dt^2 * a_world; obj.dt * a_world];

                x_l_pred = F * obj.particles(k).x_l + B_t;
                P_l_pred = F * obj.particles(k).P_l * F' + obj.Ql;
                P_l_pred = 0.5 * (P_l_pred + P_l_pred') + 1e-9 * eye(6);

                obj.particles(k).R = R_next;
                obj.particles(k).x_l = x_l_pred;
                obj.particles(k).P_l = P_l_pred;
            end

            obj = obj.update_mean_state();
        end

        %% 位置边缘化信息: 混合高斯 (粒子内解析协方差 + 粒子间散布)
        function [p_est, Sigma_pos] = get_marginalized_position_info(obj)
            p_est = obj.state.p;
            Sigma_pos = zeros(3, 3);
            for k = 1 : obj.Np
                dp = obj.particles(k).x_l(1:3) - p_est;
                Sigma_pos = Sigma_pos + obj.particles(k).w * (obj.particles(k).P_l(1:3,1:3) + dp * dp');
            end
            Sigma_pos = 0.5 * (Sigma_pos + Sigma_pos') + 1e-9 * eye(3);
        end

        %% UWB更新: 逐条序贯KF更新线性态(式15-30) + 权重更新(式31-32) + 重采样(式33-37)
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
                    H = [u', zeros(1, 3)];

                    z_resid = anchor_ranges(a) - dist;
                    S = H * Pl * H' + sigma_s2;
                    S = max(S, 1e-9);
                    Kg = Pl * H' / S;

                    xl = xl + Kg * z_resid;
                    Pl = (eye(6) - Kg * H) * Pl;
                    Pl = 0.5 * (Pl + Pl') + 1e-9 * eye(6);

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
                    H = [e_ij', zeros(1, 3)];

                    z_resid = relative_ranges(m) - dist;
                    S = H * Pl * H' + sig2_equiv;
                    S = max(S, 1e-9);
                    Kg = Pl * H' / S;

                    xl = xl + Kg * z_resid;
                    Pl = (eye(6) - Kg * H) * Pl;
                    Pl = 0.5 * (Pl + Pl') + 1e-9 * eye(6);

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

        % 重采样后姿态roughening: 恢复克隆丢失的多样性 (位置/速度不额外扰动，
        % 因为各粒子的解析P_l已保有自身不确定性，未来多样性由下一次predict中
        % 不同的姿态采样通过 a_world = R*acc 自然传导到线性态)
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
            mean_xl = zeros(6, 1);
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

            if any(isnan(mean_xl)) || any(isinf(mean_xl))
                fprintf(2, '[警告] 节点 %d 状态出现 NaN/Inf！\n', obj.id);
            end
        end
    end

    methods (Static)
        function R = so3_exp(v)
            theta = norm(v);
            if theta < 1e-8
                R = eye(3) + RBPF.skew(v);
            else
                axis = v / theta;
                R = eye(3) + sin(theta) * RBPF.skew(axis) + (1 - cos(theta)) * RBPF.skew(axis)^2;
            end
        end

        function phi = so3_log(R)
            val = max(-1, min(1, (trace(R) - 1) / 2));
            theta = acos(val);
            R_diff = R - R';
            if theta < 1e-6
                phi = 0.5 * RBPF.unskew(R_diff);
            else
                phi = (theta / (2 * sin(theta))) * RBPF.unskew(R_diff);
            end
        end

        function J = right_jacobian(theta)
            th = norm(theta);
            sk = RBPF.skew(theta);
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
