function [Pareto_fitness, Pareto_solution, Convergence_curve] = SPEA2(N, Max_iter, lb, ub, dim, fobj, num_obj)
% 标准 SPEA2（无局部搜索、无重置，无 dominates 函数依赖）
    archive_size = N;

    % 随机初始化
    population = lb + rand(N, dim) .* (ub - lb);
    fitness = zeros(N, num_obj);
    for i = 1:N
        fitness(i,:) = fobj(population(i,:));
    end
    archive = [];
    archive_fit = [];

    Convergence_curve = zeros(1, Max_iter);
    mutation_rate = 1 / dim;

    for t = 1:Max_iter
        combined_pop = [population; archive];
        combined_fit = [fitness; archive_fit];
        M = size(combined_pop, 1);

        % 强度值 S(i) —— 使用内联支配比较
        S = zeros(M, 1);
        for i = 1:M
            for j = 1:M
                if i ~= j && all(combined_fit(j,:) <= combined_fit(i,:)) && any(combined_fit(j,:) < combined_fit(i,:))
                    S(i) = S(i) + 1;
                end
            end
        end

        % 原始适应度 R(i)
        R = zeros(M, 1);
        for i = 1:M
            for j = 1:M
                if all(combined_fit(j,:) <= combined_fit(i,:)) && any(combined_fit(j,:) < combined_fit(i,:))
                    R(i) = R(i) + S(j);
                end
            end
        end

        % 密度 D(i)
        k = max(1, floor(sqrt(M)));
        if k > M - 1, k = M - 1; end
        D = zeros(M, 1);
        if M > 1
            dist_mat = pdist2(combined_fit, combined_fit);
            for i = 1:M
                dist_i = dist_mat(i, :);
                dist_i(i) = inf;
                sorted_dist = sort(dist_i);
                sigma = sorted_dist(min(k, length(sorted_dist)));
                D(i) = 1 / (sigma + 2);
            end
        else
            D = ones(M, 1);
        end
        F = R + D;

        % 环境选择：选取适应度最小的 archive_size 个个体
        [~, idx] = sort(F);
        num_keep = min(archive_size, M);
        archive = combined_pop(idx(1:num_keep), :);
        archive_fit = combined_fit(idx(1:num_keep), :);
        archive_F = F(idx(1:size(archive,1)));

        % 产生子代
        new_pop = zeros(N, dim);
        for i = 1:N/2
            idx1 = tournament_selection_SPEA2(archive, archive_F, 2);
            idx2 = tournament_selection_SPEA2(archive, archive_F, 2);
            p1 = archive(idx1, :);
            p2 = archive(idx2, :);
            [c1, c2] = sbx_crossover(p1, p2, lb, ub);
            c1 = polynomial_mutation(c1, mutation_rate, lb, ub);
            c2 = polynomial_mutation(c2, mutation_rate, lb, ub);
            new_pop(2*i-1,:) = c1;
            new_pop(2*i,:)   = c2;
        end
        population = new_pop;
        for i = 1:N
            fitness(i,:) = fobj(population(i,:));
        end

        Convergence_curve(t) = mean(fitness(:));
    end

    % 最终帕累托前沿
    fronts = fast_non_dominated_sort(archive_fit);
    nd_idx = fronts{1};
    Pareto_fitness = archive_fit(nd_idx, :);
    Pareto_solution = archive(nd_idx, :);
end

% ==================== SPEA2 特有辅助函数 ====================
function idx = tournament_selection_SPEA2(pop, F, k)
    N = size(pop, 1);
    if N == 0, idx = 1; return; end
    candidates = randi(N, 1, k);
    [~, best] = min(F(candidates));
    idx = candidates(best);
end

% ==================== 通用辅助函数（内联支配比较） ====================
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