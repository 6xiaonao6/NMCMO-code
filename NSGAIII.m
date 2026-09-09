function [Pareto_fitness, Pareto_solution, Convergence_curve] = NSGAIII(N, Max_iter, lb, ub, dim, fobj, num_obj)
% 标准 NSGA-III（无局部搜索、无重置、固定变异率）
    if num_obj ~= 3
        error('当前 NSGA-III 实现仅支持三目标问题');
    end

    % 随机初始化
    population = lb + rand(N, dim) .* (ub - lb);
    fitness = zeros(N, num_obj);
    for i = 1:N
        fitness(i,:) = fobj(population(i,:));
    end

    ref_points = generate_reference_points(N, num_obj);
    Convergence_curve = zeros(1, Max_iter);
    mutation_rate = 1 / dim;   % 标准建议值

    for t = 1:Max_iter
        % 产生子代（SBX + 多项式变异）
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

        % 环境选择（NSGA-III 参考点方法）
        [selected_pop, selected_fit] = environmental_selection_nsga3(merged_pop, merged_fit, N, ref_points);

        population = selected_pop;
        fitness = selected_fit;

        Convergence_curve(t) = mean(fitness(:));
    end

    % 最终帕累托前沿
    fronts = fast_non_dominated_sort(fitness);
    nd_idx = fronts{1};
    Pareto_fitness = fitness(nd_idx, :);
    Pareto_solution = population(nd_idx, :);
end

% ==================== NSGA-III 特有辅助函数 ====================
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
end

function [assoc_ref, dist2ref] = associate_to_reference(norm_fit, ref_points)
    num_ind = size(norm_fit, 1);
    num_ref = size(ref_points, 1);
    assoc_ref = zeros(num_ind, 1);
    dist2ref = zeros(num_ind, 1);
    for i = 1:num_ind
        best_dist = inf;
        best_ref = 1;
        norm_i = norm(norm_fit(i,:));
        for r = 1:num_ref
            cos_theta = dot(norm_fit(i,:), ref_points(r,:)) / (norm_i * norm(ref_points(r,:)) + eps);
            theta = acos(min(1, max(-1, cos_theta)));
            dist = norm_i * theta;
            if dist < best_dist
                best_dist = dist;
                best_ref = r;
            end
        end
        assoc_ref(i) = best_ref;
        dist2ref(i) = best_dist;
    end
end

function [selected_pop, selected_fitness] = environmental_selection_nsga3(pop, fitness, N, ref_points)
    [fronts, ~] = fast_non_dominated_sort(fitness);
    new_pop = [];
    new_fit = [];
    last_front_idx = 0;
    for i = 1:length(fronts)
        if size(new_pop,1) + length(fronts{i}) <= N
            new_pop = [new_pop; pop(fronts{i}, :)];
            new_fit = [new_fit; fitness(fronts{i}, :)];
        else
            last_front_idx = i;
            break;
        end
    end

    if last_front_idx > 0 && size(new_pop,1) < N
        last_front_pop = pop(fronts{last_front_idx}, :);
        last_front_fit = fitness(fronts{last_front_idx}, :);
        remain = N - size(new_pop,1);

        % 归一化
        if ~isempty(new_fit)
            f_min = min(new_fit, [], 1);
            f_max = max(new_fit, [], 1);
        else
            f_min = min(last_front_fit, [], 1);
            f_max = max(last_front_fit, [], 1);
        end
        f_range = f_max - f_min;
        f_range(f_range == 0) = 1;
        if ~isempty(new_fit)
            norm_fit = (new_fit - f_min) ./ f_range;
        else
            norm_fit = [];
        end
        norm_last = (last_front_fit - f_min) ./ f_range;

        [assoc_ref, dist2ref] = associate_to_reference(norm_last, ref_points);
        if ~isempty(norm_fit)
            [assoc_selected, ~] = associate_to_reference(norm_fit, ref_points);
            niche_count = accumarray(assoc_selected, 1, [size(ref_points,1), 1]);
        else
            niche_count = zeros(size(ref_points,1), 1);
        end

        selected_idx = [];
        for r = 1:remain
            [min_count, min_ref] = min(niche_count);
            cand = find(assoc_ref == min_ref);
            if ~isempty(cand)
                [~, best] = min(dist2ref(cand));
                idx = cand(best);
            else
                [~, best] = min(dist2ref);
                idx = best;
            end
            selected_idx = [selected_idx; idx];
            niche_count(assoc_ref(idx)) = niche_count(assoc_ref(idx)) + 1;
            assoc_ref(idx) = [];
            dist2ref(idx) = [];
        end
        new_pop = [new_pop; last_front_pop(selected_idx, :)];
        new_fit = [new_fit; last_front_fit(selected_idx, :)];
    end
    selected_pop = new_pop;
    selected_fitness = new_fit;
end

% ==================== 通用辅助函数（与NSGA-II共用） ====================
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