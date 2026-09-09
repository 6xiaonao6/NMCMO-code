function [Pareto_fitness, Pareto_solution, Convergence_curve] = RVEA(N, Max_iter, lb, ub, dim, fobj, num_obj)
% Reference Vector Guided Evolutionary Algorithm (RVEA)
% 标准实现，无局部搜索，无重置，固定参数
    if num_obj ~= 3
        error('当前 RVEA 实现仅支持三目标问题');
    end

    % 生成参考向量
    ref_vectors = generate_reference_vectors(N, num_obj);
    alpha = 2;  % 角度惩罚参数

    % 随机初始化
    population = lb + rand(N, dim) .* (ub - lb);
    fitness = zeros(N, num_obj);
    for i = 1:N
        fitness(i,:) = fobj(population(i,:));
    end

    % 理想点与最差点
    ideal = min(fitness, [], 1);
    nadir = max(fitness, [], 1);

    Convergence_curve = zeros(1, Max_iter);
    mutation_rate = 1 / dim;

    for t = 1:Max_iter
        % 生成子代（SBX + 多项式变异）
        offspring_pop = zeros(N, dim);
        offspring_fit = zeros(N, num_obj);
        for i = 1:N/2
            p1 = tournament_selection(fitness, 2);
            p2 = tournament_selection(fitness, 2);
            [c1, c2] = sbx_crossover(population(p1,:), population(p2,:), lb, ub);
            c1 = polynomial_mutation(c1, mutation_rate, lb, ub);
            c2 = polynomial_mutation(c2, mutation_rate, lb, ub);
            offspring_pop(2*i-1,:) = c1;
            offspring_pop(2*i,:)   = c2;
            offspring_fit(2*i-1,:) = fobj(c1);
            offspring_fit(2*i,:)   = fobj(c2);
        end

        merged_pop = [population; offspring_pop];
        merged_fit = [fitness; offspring_fit];

        % 更新理想点和最差点
        ideal = min([ideal; merged_fit], [], 1);
        nadir = max([nadir; merged_fit], [], 1);

        % 归一化
        range = nadir - ideal;
        range(range == 0) = 1;
        norm_fit = (merged_fit - ideal) ./ range;

        % 关联到参考向量
        [assoc, dist] = associate_to_vectors(norm_fit, ref_vectors);

        % 环境选择（基于角度惩罚距离）
        [selected_pop, selected_fit] = environmental_selection_rvea(merged_pop, merged_fit, norm_fit, ref_vectors, assoc, dist, N, alpha, t, Max_iter);

        population = selected_pop;
        fitness = selected_fit;

        Convergence_curve(t) = mean(fitness(:));
    end

    % 输出非支配解
    fronts = fast_non_dominated_sort(fitness);
    nd_idx = fronts{1};
    Pareto_fitness = fitness(nd_idx, :);
    Pareto_solution = population(nd_idx, :);
end

% ==================== RVEA 特有辅助函数 ====================
function ref_vectors = generate_reference_vectors(N, M)
    p = round((N/2)^(1/3));
    ref_vectors = [];
    for i = 0:p
        for j = 0:p-i
            k = p - i - j;
            w = [i, j, k] / p;
            ref_vectors = [ref_vectors; w];
        end
    end
    if size(ref_vectors,1) > N
        idx = randperm(size(ref_vectors,1), N);
        ref_vectors = ref_vectors(idx, :);
    elseif size(ref_vectors,1) < N
        rep = ceil(N / size(ref_vectors,1));
        ref_vectors = repmat(ref_vectors, rep, 1);
        ref_vectors = ref_vectors(1:N, :);
    end
    % 归一化
    for i = 1:size(ref_vectors,1)
        ref_vectors(i,:) = ref_vectors(i,:) / norm(ref_vectors(i,:));
    end
end

function [assoc, dist] = associate_to_vectors(norm_fit, ref_vectors)
    num_ind = size(norm_fit, 1);
    num_ref = size(ref_vectors, 1);
    assoc = zeros(num_ind, 1);
    dist = zeros(num_ind, 1);
    for i = 1:num_ind
        best = 1;
        best_dist = inf;
        for r = 1:num_ref
            cos_theta = dot(norm_fit(i,:), ref_vectors(r,:)) / (norm(norm_fit(i,:)) * norm(ref_vectors(r,:)) + eps);
            theta = acos(min(1, max(-1, cos_theta)));
            d = norm(norm_fit(i,:)) * theta;
            if d < best_dist
                best_dist = d;
                best = r;
            end
        end
        assoc(i) = best;
        dist(i) = best_dist;
    end
end

function [selected_pop, selected_fitness] = environmental_selection_rvea(pop, fitness, norm_fit, ref_vectors, assoc, dist, N, alpha, t, Max_iter)
    % 非支配排序（可选，但RVEA通常直接按角度惩罚距离选择）
    % 这里简化：直接按每个参考向量保留距离最小的个体
    num_ref = size(ref_vectors, 1);
    selected = [];
    % 对每个参考向量，选择距离最小的个体
    for r = 1:num_ref
        idx = find(assoc == r);
        if ~isempty(idx)
            [~, best] = min(dist(idx));
            selected = [selected, idx(best)];
        end
    end
    % 如果不足N个，从剩余个体中按距离补充
    if length(selected) < N
        remaining = setdiff(1:size(pop,1), selected);
        [~, sort_idx] = sort(dist(remaining));
        add = remaining(sort_idx(1:min(N-length(selected), length(remaining))));
        selected = [selected, add];
    elseif length(selected) > N
        % 如果超过N，按距离排序，保留前N个
        [~, sort_idx] = sort(dist(selected));
        selected = selected(sort_idx(1:N));
    end
    selected_pop = pop(selected, :);
    selected_fitness = fitness(selected, :);
end

% ==================== 通用辅助函数（与NSGA-II相同） ====================
function idx = tournament_selection(fitness, tour_size)
    N = size(fitness, 1);
    tour_size = min(tour_size, N);
    candidates = randperm(N, tour_size);
    [~, ranks] = fast_non_dominated_sort(fitness(candidates, :));
    [~, min_rank_idx] = min(ranks);
    idx = candidates(min_rank_idx);
end

function [child1, child2] = sbx_crossover(p1, p2, lb, ub)
    eta_c = 20;
    dim = length(p1);
    child1 = p1; child2 = p2;
    for i = 1:dim
        if rand < 0.9
            if abs(p1(i)-p2(i)) > 1e-10
                if p1(i) < p2(i)
                    y1 = p1(i); y2 = p2(i);
                else
                    y1 = p2(i); y2 = p1(i);
                end
                yl = lb(i); yu = ub(i);
                beta = 1 + 2*(y1-yl)/(y2-y1);
                alpha = 2 - beta^(-(eta_c+1));
                r = rand;
                if r <= 1/alpha
                    betaq = (r*alpha)^(1/(eta_c+1));
                else
                    betaq = (1/(2 - r*alpha))^(1/(eta_c+1));
                end
                c1 = 0.5*((y1+y2) - betaq*(y2-y1));
                c2 = 0.5*((y1+y2) + betaq*(y2-y1));
                c1 = max(yl, min(yu, c1));
                c2 = max(yl, min(yu, c2));
                child1(i) = c1; child2(i) = c2;
            end
        end
    end
end

function mutated = polynomial_mutation(individual, mutation_rate, lb, ub)
    dim = length(individual);
    mutated = individual;
    for i = 1:dim
        if rand < mutation_rate
            eta_m = 20;
            y = individual(i);
            yl = lb(i);
            yu = ub(i);
            delta1 = (y - yl) / (yu - yl);
            delta2 = (yu - y) / (yu - yl);
            r = rand;
            mut_pow = 1.0 / (eta_m + 1.0);
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
            mutated(i) = y;
        end
    end
end

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
            if all(fitness(i,:) <= fitness(j,:)) && any(fitness(i,:) < fitness(j,:))
                S{i} = [S{i}, j];
            elseif all(fitness(j,:) <= fitness(i,:)) && any(fitness(j,:) < fitness(i,:))
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