function [Pareto_fitness, Pareto_solution, Convergence_curve] = NMCMO(N, Max_iteration, lb, ub, dim, fobj, num_obj, problem_name)
% ========================================================================
% NMCMO - 改进版（针对 DTLZ1/3/4 优化）
% 改进点：
%   1. 均匀 Das-Dennis 权重向量（代替重复格点法）
%   2. 混合变异策略（降低 current-to-best 频率）
%   3. 非支配排序 + 拥挤距离修剪存档（替代聚类）
%   4. 动态邻域更新概率（早期探索更多）
%   5. 加强局部搜索（更多个体、更多迭代）
%   6. 放宽停滞重置条件（更灵敏）
%   7. 重置时注入随机个体
% ========================================================================

    if nargin < 7, num_obj = 2; end
    if nargin < 8, problem_name = ''; end

    if max(size(lb)==1)
        lb = lb.*ones(1, dim);
        ub = ub.*ones(1, dim);
    end

    % ========== 权重向量（均匀 Das-Dennis） ==========
    if num_obj == 2
        weights = zeros(N,2);
        for i = 1:N
            weights(i,1) = (i-1)/(N-1);
            weights(i,2) = 1 - weights(i,1);
        end
    elseif num_obj == 3
        weights = generate_uniform_weights(N, 3);
    else
        error('NMCMO 仅支持 2 或 3 目标');
    end
    % 归一化
    weights = weights ./ vecnorm(weights, 2, 2);

    % ========== 参数设置 ==========
    T = ceil(N/10);
    delta = 0.7;                % 初始邻域概率，后续动态更新
    F_min = 0.3; F_max = 0.9;
    CR = 0.9;
    pm_base = 1/dim;

    % PSO参数（保留但概率降低）
    w_min = 0.2; w_max = 0.9;
    c1 = 1.5; c2 = 1.5;
    pso_prob = 0.15;            % PSO概率降低

    % 局部搜索参数（加强）
    ls_freq = 15;               % 更频繁
    ls_iter = 4;               % 增加迭代次数
    ls_step_init = 0.02;

    % 存档大小
    archive_size = N;

    % 停滞与反弹参数（放宽）
    hv_improve_threshold = 1e-5;
    stagnation_window = 10;     % 缩短窗口
    last_reset_iter = -50;      % 冷却期减至50
    reset_ratio = 0.02;         % 重置比例
    rebound_threshold = 0.005;  % 降低触发阈值
    best_hv = -inf;
    best_archive = [];
    best_archive_fit = [];
    rebound_counter = 0;
    stagnation_counter = 0;

    % ========== 初始化 ==========
    population = lhsdesign(N, dim) .* (ub - lb) + lb;
    fitness = zeros(N, num_obj);
    for i = 1:N
        fitness(i,:) = fobj(population(i,:));
    end
    velocity = zeros(N, dim);

    z = min(fitness, [], 1);
    nadir = max(fitness, [], 1);

    archive = population;
    archive_fit = fitness;

    dist_w = pdist2(weights, weights);
    B = zeros(N, T);
    for i = 1:N
        [~, idx] = sort(dist_w(i,:));
        B(i,:) = idx(2:T+1);
    end

    ref_points = generate_reference_points(N, num_obj);
    Convergence_curve = zeros(Max_iteration, 1);
    best_hv_history = [];

    % ========== 主循环 ==========
    for t = 1:Max_iteration
        progress = t / Max_iteration;

        % ---- 自适应参数 ----
        if size(fitness,1) > 1
            pairwise_dist = pdist2(fitness, fitness);
            avg_dist = mean(pairwise_dist(pairwise_dist > 0));
            norm_avg_dist = min(1, avg_dist / sqrt(num_obj));
        else
            norm_avg_dist = 0;
        end
        diversity_factor = max(0.5, min(2, 1 + (1 - norm_avg_dist)));
        mutation_rate = min(0.5, max(0.01, 0.25 * (1 - progress) + 0.01 * progress) * diversity_factor);
        crossover_rate = min(0.95, max(0.5, 0.8 * progress + 0.9 * (1 - progress)) / diversity_factor);
        F = F_max - (F_max - F_min) * progress;
        w = w_max - (w_max - w_min) * progress;
        % 动态邻域概率：早期探索多
        delta = 0.7 + 0.2 * progress;  % 从0.7升至0.9

        % ---- 计算存档中的最佳个体 ----
        if ~isempty(archive)
            [~, ranks_arc] = fast_non_dominated_sort(archive_fit);
            nd_idx = find(ranks_arc == 1);
            if ~isempty(nd_idx)
                conv_arc = sum(archive_fit(nd_idx, :), 2);
                [~, best_idx] = min(conv_arc);
                elite_base = archive(nd_idx(best_idx), :);
            else
                elite_base = archive(randi(size(archive,1)), :);
            end
        else
            elite_base = population(randi(N), :);
        end

        % ---- 从存档中选择稀疏区域个体作为PSO全局最优 ----
        if size(archive,1) > 0
            if size(archive,1) == 1
                gbest = archive(1,:);
            else
                dist_crowd = calculate_crowding_distance(archive_fit);
                [~, idx_max] = max(dist_crowd);
                gbest = archive(idx_max, :);
            end
        else
            gbest = population(randi(N), :);
        end

        % ---- 子代生成（DE为主，PSO为辅） ----
        offspring_pop = zeros(N, dim);
        offspring_fit = zeros(N, num_obj);

        for i = 1:N
            if rand < delta
                P = B(i,:);
            else
                P = randperm(N, T);
            end
            idx_perm = randperm(length(P), 3);
            k1 = P(idx_perm(1)); k2 = P(idx_perm(2)); k3 = P(idx_perm(3));

            if rand < pso_prob
                % PSO更新
                r1 = rand(1, dim); r2 = rand(1, dim);
                velocity(i,:) = w * velocity(i,:) + ...
                                c1 * r1 .* (population(i,:) - population(i,:)) + ...
                                c2 * r2 .* (gbest - population(i,:));
                max_vel = (ub - lb) * 0.2;
                velocity(i,:) = max(-max_vel, min(max_vel, velocity(i,:)));
                child = population(i,:) + velocity(i,:);
                child = max(lb, min(ub, child));
            else
                % ===== 改进的变异策略 =====
                if rand < 0.4
                    % rand/1：全局探索
                    mutant = population(k1,:) + F * (population(k2,:) - population(k3,:));
                elseif rand < 0.7
                    % current-to-best/1：收敛引导（频率降低至30%）
                    mutant = population(i,:) + F * (elite_base - population(i,:)) + F * (population(k1,:) - population(k2,:));
                else
                    % rand/2：更强探索
                    rp = randperm(N, 5);
                    mutant = population(rp(1),:) + F*(population(rp(2),:)-population(rp(3),:)) + F*(population(rp(4),:)-population(rp(5),:));
                end
                mutant = max(lb, min(ub, mutant));
                child = population(k1, :);
                j_rand = randi(dim);
                for j = 1:dim
                    if rand < CR || j == j_rand
                        child(j) = mutant(j);
                    end
                end
                % 多项式变异
                for j = 1:dim
                    if rand < mutation_rate
                        eta_m = 20 + 15 * (1 - progress);
                        y = child(j); yl = lb(j); yu = ub(j);
                        delta1 = (y - yl) / (yu - yl);
                        delta2 = (yu - y) / (yu - yl);
                        r = rand; mut_pow = 1.0 / (eta_m + 1.0);
                        if r <= 0.5
                            xy = 1.0 - delta1;
                            val = 2.0 * r + (1.0 - 2.0 * r) * (xy^(eta_m + 1.0));
                            deltaq = val^mut_pow - 1.0;
                        else
                            xy = 1.0 - delta2;
                            val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy^(eta_m + 1.0));
                            deltaq = 1.0 - val^mut_pow;
                        end
                        y = y + deltaq * (yu - yl);
                        y = max(yl, min(yu, y));
                        child(j) = y;
                    end
                end
                child = max(lb, min(ub, child));
            end

            child_fit = fobj(child);
            offspring_pop(i,:) = child;
            offspring_fit(i,:) = child_fit;

            % 更新理想点和最差点
            z = min(z, child_fit);
            nadir = max(nadir, child_fit);

            % 邻域更新
            for j = 1:length(P)
                idx_j = P(j);
                f_old = max(abs(fitness(idx_j,:) - z) ./ (weights(idx_j,:) + 1e-10));
                f_new = max(abs(child_fit - z) ./ (weights(idx_j,:) + 1e-10));
                if f_new < f_old * (1 - 1e-6)
                    population(idx_j,:) = child;
                    fitness(idx_j,:) = child_fit;
                end
            end
        end

        % ---- 环境选择（混合拥挤距离） ----
        merged_pop = [population; offspring_pop];
        merged_fit = [fitness; offspring_fit];
        [fronts, ~] = fast_non_dominated_sort(merged_fit);

        new_pop = []; new_fit = []; count = 0;
        for fi = 1:length(fronts)
            if count + length(fronts{fi}) <= N
                new_pop = [new_pop; merged_pop(fronts{fi}, :)];
                new_fit = [new_fit; merged_fit(fronts{fi}, :)];
                count = count + length(fronts{fi});
            else
                front_idx = fronts{fi};
                front_pop = merged_pop(front_idx, :);
                front_fit = merged_fit(front_idx, :);
                distances = calculate_crowding_distance_hybrid(front_fit, front_pop, 0.7, 0.3);
                [~, sorted_idx] = sort(distances, 'descend');
                remain = N - count;
                selected = sorted_idx(1:remain);
                new_pop = [new_pop; front_pop(selected, :)];
                new_fit = [new_fit; front_fit(selected, :)];
                break;
            end
        end
        population = new_pop;
        fitness = new_fit;

        % ---- 更新存档（使用非支配排序 + 拥挤距离修剪） ----
        archive = [archive; population];
        archive_fit = [archive_fit; fitness];
        % 先进行非支配排序
        [fronts_arc, ~] = fast_non_dominated_sort(archive_fit);
        new_archive = []; new_arc_fit = [];
        count_arc = 0;
        for fi = 1:length(fronts_arc)
            front_idx = fronts_arc{fi};
            front_pop = archive(front_idx, :);
            front_fit = archive_fit(front_idx, :);
            if count_arc + length(front_idx) <= archive_size
                new_archive = [new_archive; front_pop];
                new_arc_fit = [new_arc_fit; front_fit];
                count_arc = count_arc + length(front_idx);
            else
                % 计算拥挤距离
                dist_crowd = calculate_crowding_distance(front_fit);
                [~, sorted] = sort(dist_crowd, 'descend');
                remain = archive_size - count_arc;
                new_archive = [new_archive; front_pop(sorted(1:remain), :)];
                new_arc_fit = [new_arc_fit; front_fit(sorted(1:remain), :)];
                break;
            end
        end
        archive = new_archive;
        archive_fit = new_arc_fit;

        % ---- 反弹检测与恢复 ----
        if size(archive_fit,1) > 0
            hv_now = hypervolume_exact(archive_fit);
            best_hv_history = [best_hv_history, hv_now];
            if hv_now > best_hv
                best_hv = hv_now;
                best_archive = archive;
                best_archive_fit = archive_fit;
                rebound_counter = 0;
            end
            % 反弹检测（连续3代HV下降超过阈值）
            if length(best_hv_history) >= 3
                recent_hv = best_hv_history(end-2:end);
                if all(diff(recent_hv) < 0) && (recent_hv(1) - recent_hv(end)) / (recent_hv(1) + 1e-10) > rebound_threshold
                    rebound_counter = rebound_counter + 1;
                else
                    rebound_counter = 0;
                end
            end
            % 恢复：用历史最优解替换种群最差个体
            if rebound_counter >= 2 && ~isempty(best_archive)
                obj_sum_pop = sum(fitness, 2);
                [~, worst_idx] = sort(obj_sum_pop, 'descend');
                num_replace = max(1, round(0.1 * N));
                replace_idx = worst_idx(1:min(num_replace, N));
                [fronts_best, ~] = fast_non_dominated_sort(best_archive_fit);
                best_pf_fit = best_archive_fit(fronts_best{1}, :);
                best_pf_pop = best_archive(fronts_best{1}, :);
                conv_best = sum(best_pf_fit, 2);
                [~, sort_conv] = sort(conv_best, 'ascend');
                for ri = 1:length(replace_idx)
                    src_idx = sort_conv(mod(ri-1, length(sort_conv)) + 1);
                    population(replace_idx(ri), :) = best_pf_pop(src_idx, :);
                    fitness(replace_idx(ri), :) = best_pf_fit(src_idx, :);
                end
                rebound_counter = 0;
            end
        end

        % ---- 停滞重置（冷却期50代，条件放宽） ----
        if length(best_hv_history) >= stagnation_window
            recent_improve = (best_hv_history(end) - best_hv_history(end-stagnation_window+1)) / (best_hv_history(end-stagnation_window+1) + 1e-10);
            % 同时检查多样性
            if recent_improve < hv_improve_threshold && norm_avg_dist < 0.2
                stagnation_counter = stagnation_counter + 1;
            else
                stagnation_counter = 0;
            end
            if stagnation_counter >= 2 && (t - last_reset_iter) >= 50  % 冷却期50
                num_reset = max(1, round(reset_ratio * N));
                obj_sum_pop = sum(fitness, 2);
                [~, worst_idx] = sort(obj_sum_pop, 'descend');
                worst_idx = worst_idx(1:min(num_reset, N));
                % 从存档中选择最优个体，加扰动；并注入20%随机个体
                [fronts_arc, ~] = fast_non_dominated_sort(archive_fit);
                if ~isempty(fronts_arc)
                    pf_idx = fronts_arc{1};
                    pf_fit = archive_fit(pf_idx, :);
                    pf_pop = archive(pf_idx, :);
                    conv_pf = sum(pf_fit, 2);
                    [~, best_conv] = sort(conv_pf, 'ascend');
                    num_rand = round(0.2 * num_reset);  % 随机注入比例
                    for ri = 1:length(worst_idx)
                        if ri <= num_rand
                            % 完全随机
                            new_x = lb + (ub - lb) .* rand(1, dim);
                        else
                            src = best_conv(mod(ri-1, length(best_conv)) + 1);
                            new_x = pf_pop(src, :) + 0.005 * (ub - lb) .* randn(1, dim);
                            new_x = max(lb, min(ub, new_x));
                            % 微小局部修复
                            for rep_iter = 1:3
                                step = 0.005 * (ub - lb);
                                for j = 1:dim
                                    for sgn = [-1, 1]
                                        x_try = new_x;
                                        x_try(j) = x_try(j) + sgn * step(j);
                                        x_try = max(lb, min(ub, x_try));
                                        f_try = fobj(x_try);
                                        if dominates(f_try, fobj(new_x))
                                            new_x = x_try;
                                        end
                                    end
                                end
                            end
                        end
                        new_f = fobj(new_x);
                        population(worst_idx(ri), :) = new_x;
                        fitness(worst_idx(ri), :) = new_f;
                    end
                end
                stagnation_counter = 0;
                last_reset_iter = t;
            end
        end

        % ---- 局部搜索（增强） ----
        if mod(t, ls_freq) == 0 && size(archive,1) > 0
            current_step = max(1e-4, 0.01 * (1 - progress));
            obj_sum_arc = sum(archive_fit, 2);
            [~, idx_sum] = sort(obj_sum_arc, 'ascend');
            arch_dist = calculate_crowding_distance(archive_fit);
            [~, idx_dist] = sort(arch_dist, 'descend');
            top_sum = idx_sum(1:max(1, round(0.3*size(archive,1))));
            top_dist = idx_dist(1:max(1, round(0.3*size(archive,1))));
            search_idx = intersect(top_sum, top_dist);
            if isempty(search_idx)
                search_idx = top_sum(1:min(3, length(top_sum)));
            end
            % 扩大搜索个体数
            num_search = min(length(search_idx), max(5, round(0.1*size(archive,1))));
            search_idx = search_idx(1:num_search);
            for si = 1:length(search_idx)
                i = search_idx(si);
                x = archive(i, :);
                f_x = archive_fit(i, :);
                step = current_step;
                for iter = 1:ls_iter
                    improved = false;
                    for j = 1:dim
                        for sgn = [-1, 1]
                            x_new = x;
                            x_new(j) = x_new(j) + sgn * step;
                            x_new = max(lb, min(ub, x_new));
                            f_new = fobj(x_new);
                            if dominates(f_new, f_x)
                                x = x_new;
                                f_x = f_new;
                                improved = true;
                            end
                        end
                    end
                    if ~improved
                        step = step * 0.6;  % 衰减慢一些
                    else
                        break;
                    end
                end
                archive(i, :) = x;
                archive_fit(i, :) = f_x;
            end
            % 重新修剪存档
            [fronts_arc, ~] = fast_non_dominated_sort(archive_fit);
            new_archive = []; new_arc_fit = [];
            count_arc = 0;
            for fi = 1:length(fronts_arc)
                front_idx = fronts_arc{fi};
                front_pop = archive(front_idx, :);
                front_fit = archive_fit(front_idx, :);
                if count_arc + length(front_idx) <= archive_size
                    new_archive = [new_archive; front_pop];
                    new_arc_fit = [new_arc_fit; front_fit];
                    count_arc = count_arc + length(front_idx);
                else
                    dist_crowd = calculate_crowding_distance(front_fit);
                    [~, sorted] = sort(dist_crowd, 'descend');
                    remain = archive_size - count_arc;
                    new_archive = [new_archive; front_pop(sorted(1:remain), :)];
                    new_arc_fit = [new_arc_fit; front_fit(sorted(1:remain), :)];
                    break;
                end
            end
            archive = new_archive;
            archive_fit = new_arc_fit;
        end

        % ---- 收敛曲线 ----
        if ~isempty(archive_fit)
            avg_fit = mean(archive_fit(:));
        else
            avg_fit = mean(fitness(:));
        end
        Convergence_curve(t) = avg_fit;

        if mod(t, 50) == 0 || t == Max_iteration
            fprintf('迭代: %d/%d, 存档解数量: %d\n', t, Max_iteration, size(archive,1));
        end
    end

    % ---- 最终输出 ----
    [fronts, ~] = fast_non_dominated_sort(archive_fit);
    if ~isempty(fronts)
        nd_idx = fronts{1};
        Pareto_fitness = archive_fit(nd_idx, :);
        Pareto_solution = archive(nd_idx, :);
    else
        Pareto_fitness = archive_fit;
        Pareto_solution = archive;
    end
    if size(Pareto_fitness,1) > N
        idx = randperm(size(Pareto_fitness,1), N);
        Pareto_fitness = Pareto_fitness(idx, :);
        Pareto_solution = Pareto_solution(idx, :);
    end
end

% ===================== 辅助函数 =====================

% ----- 均匀权重向量生成（Das-Dennis）-----
function weights = generate_uniform_weights(N, M)
    H = 1;
    while nchoosek(H+M-1, M-1) < N
        H = H + 1;
    end
    weights = get_weights(H, M);
    if size(weights,1) > N
        idx = randperm(size(weights,1), N);
        weights = weights(idx, :);
    else
        while size(weights,1) < N
            w = rand(1,M); w = w / sum(w);
            weights = [weights; w];
        end
    end
    weights = weights ./ vecnorm(weights, 2, 2);
end

function w = get_weights(H, M)
    % 递归生成所有组合，和为 H
    if M == 1
        w = H;
    else
        w = [];
        for i = 0:H
            sub = get_weights(H-i, M-1);
            w = [w; i*ones(size(sub,1),1), sub];
        end
    end
end

% ----- 非支配排序 -----
function [F, ranks] = fast_non_dominated_sort(fitness)
    [N, M] = size(fitness);
    S = cell(N, 1);
    n = zeros(N, 1);
    F = {};
    ranks = zeros(N, 1);
    for i = 1:N
        S{i} = [];
        n(i) = 0;
        for j = 1:N
            if i == j, continue; end
            if dominates(fitness(i,:), fitness(j,:))
                S{i} = [S{i}, j];
            elseif dominates(fitness(j,:), fitness(i,:))
                n(i) = n(i) + 1;
            end
        end
        if n(i) == 0
            ranks(i) = 1;
            if isempty(F)
                F{1} = i;
            else
                F{1} = [F{1}, i];
            end
        end
    end
    i = 1;
    while i <= length(F) && ~isempty(F{i})
        Q = [];
        for j = 1:length(F{i})
            p = F{i}(j);
            for k = 1:length(S{p})
                q = S{p}(k);
                n(q) = n(q) - 1;
                if n(q) == 0
                    ranks(q) = i + 1;
                    Q = [Q, q];
                end
            end
        end
        i = i + 1;
        if ~isempty(Q)
            F{i} = Q;
        else
            break;
        end
    end
end

function flag = dominates(obj1, obj2)
    flag = all(obj1 <= obj2) && any(obj1 < obj2);
end

% ----- 拥挤距离（目标空间）-----
function distances = calculate_crowding_distance(fitness)
    [N, M] = size(fitness);
    distances = zeros(N, 1);
    if N <= 2
        distances(:) = inf;
        return;
    end
    for m = 1:M
        [sorted_vals, sort_idx] = sort(fitness(:, m));
        distances(sort_idx(1)) = inf;
        distances(sort_idx(end)) = inf;
        f_max = sorted_vals(end);
        f_min = sorted_vals(1);
        if f_max ~= f_min
            for i = 2:(N-1)
                distances(sort_idx(i)) = distances(sort_idx(i)) + ...
                    (sorted_vals(i+1) - sorted_vals(i-1)) / (f_max - f_min);
            end
        end
    end
end

% ----- 混合拥挤距离（目标+决策）-----
function distances = calculate_crowding_distance_hybrid(fitness, pop, obj_weight, dec_weight)
    [N, M] = size(fitness);
    dim = size(pop, 2);
    distances = zeros(N, 1);
    if N <= 2
        distances(:) = inf;
        return;
    end
    % 目标空间距离
    obj_dist = zeros(N, 1);
    for m = 1:M
        [sorted_vals, sort_idx] = sort(fitness(:, m));
        obj_dist(sort_idx(1)) = inf;
        obj_dist(sort_idx(end)) = inf;
        f_max = sorted_vals(end);
        f_min = sorted_vals(1);
        if f_max ~= f_min
            for i = 2:(N-1)
                obj_dist(sort_idx(i)) = obj_dist(sort_idx(i)) + ...
                    (sorted_vals(i+1) - sorted_vals(i-1)) / (f_max - f_min);
            end
        end
    end
    % 决策空间距离（最近邻）
    dec_dist = zeros(N, 1);
    pop_norm = (pop - min(pop)) ./ (max(pop) - min(pop) + eps);
    for i = 1:N
        diff = pop_norm - pop_norm(i, :);
        dist_i = sqrt(sum(diff.^2, 2));
        dist_i(i) = inf;
        dec_dist(i) = min(dist_i);
    end
    dec_dist = dec_dist / (max(dec_dist) + eps);
    distances = obj_weight * obj_dist + dec_weight * dec_dist;
end

% ----- HV 计算（2D/3D精确，高维蒙特卡洛）-----
function hv = hypervolume_exact(fitness)
    if size(fitness,1) == 0
        hv = 0;
        return;
    end
    M = size(fitness,2);
    if M == 2
        hv = hypervolume_2d(fitness);
    elseif M == 3
        hv = hypervolume_3d(fitness);
    else
        hv = hypervolume_estimate(fitness);
    end
end

function hv = hypervolume_2d(points)
    if isempty(points)
        hv = 0;
        return;
    end
    points = sortrows(points, 1);
    ref = max(points, [], 1) + 0.1;
    hv = 0;
    prev_x = 0;
    for i = 1:size(points,1)
        hv = hv + (points(i,1) - prev_x) * (ref(2) - points(i,2));
        prev_x = points(i,1);
    end
end

function hv = hypervolume_3d(points)
    if isempty(points)
        hv = 0;
        return;
    end
    ref = max(points, [], 1) + 0.1;
    hv = hypervolume_recursive(points, ref, 1);
end

function hv = hypervolume_recursive(points, ref, dim)
    if dim > size(ref,2)
        hv = 0;
        return;
    end
    if size(points,1) == 0
        hv = 0;
        return;
    end
    points = sortrows(points, dim);
    hv = 0;
    prev = 0;
    for i = 1:size(points,1)
        current = points(i, dim);
        if current > prev
            sub_points = points(i:end, :);
            sub_points(:, dim) = sub_points(:, dim) - current;
            sub_ref = ref;
            sub_ref(dim) = sub_ref(dim) - current;
            hv = hv + (current - prev) * hypervolume_recursive(sub_points, sub_ref, dim+1);
            prev = current;
        end
    end
    if prev < ref(dim)
        hv = hv + (ref(dim) - prev) * hypervolume_recursive([], ref, dim+1);
    end
end

function hv = hypervolume_estimate(fitness)
    if size(fitness,1) < 2
        hv = 0;
        return;
    end
    M = size(fitness,2);
    ref_point = max(fitness, [], 1) + 0.1;
    N_samples = 500;
    volume = prod(ref_point - min(fitness, [], 1));
    samples = rand(N_samples, M);
    for i = 1:M
        samples(:,i) = samples(:,i) * ref_point(i);
    end
    dominated = false(N_samples,1);
    for s = 1:N_samples
        for j = 1:size(fitness,1)
            if all(fitness(j,:) <= samples(s,:)) && any(fitness(j,:) < samples(s,:))
                dominated(s) = true;
                break;
            end
        end
    end
    hv = (sum(dominated) / N_samples) * volume;
end

% ----- 生成参考点（备用）-----
function ref_points = generate_reference_points(N, M)
    p = round((N/2)^(1/3));
    ref_points = [];
    for i = 0:p
        for j = 0:p-i
            k = p - i - j;
            w = [i, j, k] / p;
            ref_points = [ref_points; w];
        end
    end
    if size(ref_points,1) > N
        idx = randperm(size(ref_points,1), N);
        ref_points = ref_points(idx, :);
    elseif size(ref_points,1) < N
        rep = ceil(N / size(ref_points,1));
        ref_points = repmat(ref_points, rep, 1);
        ref_points = ref_points(1:N, :);
    end
    for i = 1:size(ref_points,1)
        ref_points(i,:) = ref_points(i,:) / norm(ref_points(i,:));
    end
end