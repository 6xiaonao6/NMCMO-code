function [Pareto_fitness, Pareto_solution, Convergence_curve] = NSGAII(N, Max_iter, lb, ub, dim, fobj, num_obj)
% 标准 NSGA-II（无局部搜索，无重置，固定变异率）

    % 随机初始化
    population = lb + rand(N, dim) .* (ub - lb);
    fitness = zeros(N, num_obj);
    for i = 1:N
        fitness(i,:) = fobj(population(i,:));
    end

    Convergence_curve = zeros(Max_iter, 1);
    mutation_rate = 1 / dim;   % 标准建议

    for t = 1:Max_iter
        % 产生子代（SBX + 多项式变异，固定参数）
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

        % 环境选择（非支配排序 + 拥挤距离）
        [selected_pop, selected_fit] = environmental_selection(merged_pop, merged_fit, N);

        population = selected_pop;
        fitness = selected_fit;

        % 记录收敛曲线（种群目标平均值）
        Convergence_curve(t) = mean(fitness(:));
    end

    % 最终帕累托前沿
    fronts = fast_non_dominated_sort(fitness);
    nd_idx = fronts{1};
    Pareto_fitness = fitness(nd_idx, :);
    Pareto_solution = population(nd_idx, :);
end

% -------------------- 辅助函数（标准实现） --------------------
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

function idx = tournament_selection(fitness, tour_size)
    N = size(fitness, 1);
    tour_size = min(tour_size, N);
    candidates = randperm(N, tour_size);
    [~, ranks] = fast_non_dominated_sort(fitness(candidates, :));
    [~, min_rank_idx] = min(ranks);
    idx = candidates(min_rank_idx);
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

function [selected_pop, selected_fitness] = environmental_selection(pop, fitness, N)
    [fronts, ~] = fast_non_dominated_sort(fitness);
    selected_pop = [];
    selected_fitness = [];
    count = 0;
    for i = 1:length(fronts)
        front_pop = pop(fronts{i}, :);
        front_fitness = fitness(fronts{i}, :);
        if count + size(front_pop,1) <= N
            selected_pop = [selected_pop; front_pop];
            selected_fitness = [selected_fitness; front_fitness];
            count = count + size(front_pop,1);
        else
            distances = crowding_distance(front_fitness);
            [~, sorted_idx] = sort(distances, 'descend');
            remain = N - count;
            selected_indices = sorted_idx(1:remain);
            selected_pop = [selected_pop; front_pop(selected_indices, :)];
            selected_fitness = [selected_fitness; front_fitness(selected_indices, :)];
            break;
        end
    end
end

function distances = crowding_distance(fitness)
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

% 以下共用函数（与原来相同，保留）
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