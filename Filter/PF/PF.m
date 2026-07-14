classdef PF
    % Particle Filter for 9D Cooperative Localization on SO(3)
    % 对外接口保持不变：PF(id, init_state, dt_imu, Np)
    %   predict(imu_acc, imu_gyro)
    %   get_marginalized_position_info()
    %   apply_uwb_update(anchor_ranges_raw, anchor_positions_veh, sigma_s,
    %                     active_neighbors, neigh_positions, neigh_Sigma_pos,
    %                     relative_ranges, sigma_z, gamma)

    properties
        id;
        Np;
        dt;
        particles;
        state;
        g = [0; 0; 9.81];

        % 过程噪声标准差：略高于实际传感器白噪声，用于吸收未建模误差
        sigma_w = 0.0045;   % 陀螺仪 (rad/s)
        sigma_a = 0.045;    % 加速度计 (m/s^2)

        % --- 重采样后抗贫化相关参数，越小精度上升，抗贫化下降 ---
        roughen_h = 0.5;          % Liu-West 型带宽系数
        min_pos_jitter = 0.006;   % 位置抖动下限 (m)
        min_vel_jitter = 0.006;   % 速度抖动下限 (m/s)
        min_att_jitter = 0.0008;  % 姿态抖动下限 (rad)

        neff_ratio = 0.4;         % 重采样触发阈值比例
        loglik_warn_thresh = -30; % 退化/跑飞诊断阈值（对数似然峰值）
    end

    methods
        function obj = PF(id, init_state, dt_imu, Np)
            obj.id = id;
            obj.dt = dt_imu;
            obj.Np = Np;

            obj.state = struct('R', init_state.R, 'p', init_state.p, 'v', init_state.v);
            init_cov_pf = blkdiag( ...
                diag((1 * pi / 180)^2 * ones(1, 3)), ...
                diag(0.01 * ones(1, 3)), ...
                diag(0.01 * ones(1, 3)) ...
                );
            init_cov_pf = 0.5 * (init_cov_pf + init_cov_pf');
            [L, p_val] = chol(init_cov_pf, 'lower');
            if p_val > 0
                L = diag(sqrt(max(diag(init_cov_pf), 1e-9)));
            end

            obj.particles = repmat(struct('R', eye(3), 'p', zeros(3,1), 'v', zeros(3,1), 'w', 1/Np), Np, 1);
            for k = 1:Np
                dx = L * randn(9, 1);
                dphi = dx(1:3);
                dp = dx(4:6);
                dv = dx(7:9);
                obj.particles(k).R = init_state.R * PF.exp_so3(dphi);
                obj.particles(k).p = init_state.p + dp;
                obj.particles(k).v = init_state.v + dv;
                obj.particles(k).w = 1 / Np;
            end
            obj = obj.update_mean_state();
        end

        function obj = predict(obj, imu_acc, imu_gyro)
            for k = 1:obj.Np
                w_omega = obj.sigma_w * randn(3, 1);
                w_acc = obj.sigma_a * randn(3, 1);

                theta = (imu_gyro - w_omega) * obj.dt;
                obj.particles(k).R = obj.particles(k).R * PF.exp_so3(theta);

                a_body = imu_acc - w_acc;
                a_world = obj.particles(k).R * a_body - obj.g;

                p_prev = obj.particles(k).p;
                v_prev = obj.particles(k).v;

                obj.particles(k).v = v_prev + a_world * obj.dt;
                obj.particles(k).p = p_prev + v_prev * obj.dt + 0.5 * a_world * obj.dt^2;
            end
            obj = obj.update_mean_state();
        end

        function [p_est, Sigma_pos] = get_marginalized_position_info(obj)
            p_est = obj.state.p;
            Sigma_pos = zeros(3, 3);
            for k = 1:obj.Np
                dp = obj.particles(k).p - p_est;
                Sigma_pos = Sigma_pos + obj.particles(k).w * (dp * dp');
            end
            Sigma_pos = Sigma_pos + 1e-9 * eye(3);
        end

        function obj = apply_uwb_update(obj, anchor_ranges_raw, anchor_positions_veh, sigma_s, ...
                active_neighbors, neigh_positions, neigh_Sigma_pos, ...
                relative_ranges, sigma_z, gamma)

            valid_anc_idx = find(~isnan(anchor_ranges_raw) & ~isinf(anchor_ranges_raw));
            has_anchors = ~isempty(valid_anc_idx);

            valid_neigh_idx = [];
            if ~isempty(active_neighbors)
                for idx = 1:length(active_neighbors)
                    z = relative_ranges(idx);
                    if ~isnan(z) && ~isinf(z)
                        valid_neigh_idx = [valid_neigh_idx, idx];
                    end
                end
            end
            has_neighbors = ~isempty(valid_neigh_idx);

            if ~has_anchors && ~has_neighbors
                return;
            end

            % 对数似然累加（避免直接连乘下溢）
            log_lik = zeros(obj.Np, 1);
            for k = 1:obj.Np
                p_k = obj.particles(k).p;

                if has_anchors
                    for idx = 1:length(valid_anc_idx)
                        m = valid_anc_idx(idx);
                        y = anchor_ranges_raw(m);
                        c = anchor_positions_veh(m, :)';
                        y_hat = norm(p_k - c);
                        log_lik(k) = log_lik(k) - 0.5 * (y - y_hat)^2 / sigma_s^2 - 0.5 * log(2 * pi * sigma_s^2);
                    end
                end

                if has_neighbors
                    for idx = 1:length(valid_neigh_idx)
                        orig_idx = valid_neigh_idx(idx);
                        z = relative_ranges(orig_idx);
                        p_neigh = neigh_positions(orig_idx, :)';
                        Sigma_neigh = neigh_Sigma_pos{orig_idx};

                        dp = p_k - p_neigh;
                        dist = norm(dp);
                        if dist < 1e-5
                            e_ij = [1; 0; 0];
                        else
                            e_ij = dp / dist;
                        end

                        sigma_equiv_sq = sigma_z^2 + gamma * (e_ij' * Sigma_neigh * e_ij);
                        sigma_equiv_sq = max(sigma_equiv_sq, 1e-6); % 数值安全下限

                        log_lik(k) = log_lik(k) - 0.5 * (z - dist)^2 / sigma_equiv_sq - 0.5 * log(2 * pi * sigma_equiv_sq);
                    end
                end
            end

            % --- 真正有意义的退化/跑飞诊断：看最优粒子的原始对数似然是否过低 ---
            max_log = max(log_lik);
            if isnan(max_log) || isinf(max_log) || max_log < obj.loglik_warn_thresh
                fprintf(2, '[警告] 车辆 %d 全体粒子似然过低 (max_log=%.2f)，疑似跑飞或严重退化！\n', obj.id, max_log);
            end

            lik = exp(log_lik - max_log);
            for k = 1:obj.Np
                obj.particles(k).w = obj.particles(k).w * lik(k);
            end

            w_sum = sum([obj.particles.w]);
            if w_sum > 1e-12 && isfinite(w_sum)
                for k = 1:obj.Np
                    obj.particles(k).w = obj.particles(k).w / w_sum;
                end
            else
                fprintf(2, '[警告] 车辆 %d 权重归一化失败 (w_sum=%.3e)，重置为均匀权重！\n', obj.id, w_sum);
                for k = 1:obj.Np
                    obj.particles(k).w = 1 / obj.Np;
                end
            end

            % --- 有效粒子数监测 + 重采样 + roughening 抗贫化 ---
            N_eff = 1 / sum([obj.particles.w].^2);
            if N_eff < obj.neff_ratio * obj.Np
                edges = cumsum([obj.particles.w]);
                edges = [0, edges / edges(end)];

                u = rand() / obj.Np + (0:(obj.Np-1)) / obj.Np;
                [~, indices] = histc(u, edges);

                % 重采样前粒子群的位置/速度分布，用于确定 roughening 幅度
                p_all = reshape([obj.particles.p], 3, obj.Np)';
                v_all = reshape([obj.particles.v], 3, obj.Np)';
                std_p = std(p_all, 0, 1);
                std_v = std(v_all, 0, 1);

                new_particles = obj.particles(indices);
                for k = 1:obj.Np
                    new_particles(k).w = 1 / obj.Np;
                end
                obj.particles = new_particles;

                obj = obj.roughen(std_p, std_v);
            end

            obj = obj.update_mean_state();

            if any(isnan(obj.state.p)) || any(isinf(obj.state.p))
                fprintf(2, '[警告] 车辆 %d 更新后位置出现 NaN/Inf！\n', obj.id);
            end
        end
    end

    methods (Access = private)
        function obj = roughen(obj, std_p, std_v)
            % 重采样后抖动，防止粒子完全重复导致贫化
            % 幅度正比于重采样前的经验分布宽度，并设最小下限
            h = obj.roughen_h * obj.Np^(-1/9);
            jit_p = max(h * std_p, obj.min_pos_jitter * ones(1, 3));
            jit_v = max(h * std_v, obj.min_vel_jitter * ones(1, 3));
            jit_att = max(h * obj.sigma_w * obj.dt * 10, obj.min_att_jitter);

            for k = 1:obj.Np
                obj.particles(k).p = obj.particles(k).p + (jit_p .* randn(1, 3))';
                obj.particles(k).v = obj.particles(k).v + (jit_v .* randn(1, 3))';
                dphi = jit_att * randn(3, 1);
                obj.particles(k).R = obj.particles(k).R * PF.exp_so3(dphi);
            end
        end

        function obj = update_mean_state(obj)
            state_p = zeros(3, 1);
            state_v = zeros(3, 1);
            R_sum = zeros(3, 3);
            for k = 1:obj.Np
                w_k = obj.particles(k).w;
                state_p = state_p + w_k * obj.particles(k).p;
                state_v = state_v + w_k * obj.particles(k).v;
                R_sum = R_sum + w_k * obj.particles(k).R;
            end
            obj.state.p = state_p;
            obj.state.v = state_v;

            [U, ~, V] = svd(R_sum);
            R_est = U * V';
            if det(R_est) < 0
                R_est = U * diag([1, 1, -1]) * V';
            end
            obj.state.R = R_est;
        end
    end

    methods (Static)
        function R = exp_so3(v)
            theta = norm(v);
            if theta < 1e-8
                R = eye(3) + PF.skew(v);
            else
                axis = v / theta;
                R = eye(3) + sin(theta) * PF.skew(axis) + (1 - cos(theta)) * PF.skew(axis)^2;
            end
        end

        function m = skew(v)
            m = [  0,   -v(3),  v(2);
                 v(3),    0,   -v(1);
                -v(2),   v(1),    0 ];
        end
    end
end