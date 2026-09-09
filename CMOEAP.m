function [Pareto_fitness, Pareto_solution, Convergence_curve] = CMOEAP(N, Max_iter, lb, ub, dim, fobj, num_obj)
% CMOEAP: 一种基于分解和Pareto支配的混合算法（简化版）
% 实际可参考 "CMOEA" 或 "MOEA/D with dominance"
% 这里采用类似 MOEA/DD 但更强调收敛性的策略
    T = ceil(N/10);
    delta = 0.9;
    weights = generate_weights_uniform(N, num_obj);
    dist = pdist2(weights, weights);
    B = zeros(N, T);
    for i = 1:N
        [~, idx] = sort(dist(i,:));
        B(i,:) = idx(2:T+1);
    end
    
    population = lb + rand(N, dim) .* (ub - lb);
    fitness = zeros(N, num_obj);
    for i = 1:N
        fitness(i,:) = fobj(population(i,:));
    end
    z = min(fitness, [], 1);
    
    Convergence_curve = zeros(1, Max_iter);
    F = 0.5; CR = 0.9;
    mutation_rate = 1/dim;
    
    for t = 1:Max_iter
        for i = 1:N
            if rand < delta
                P = B(i,:);
            else
                P = randperm(N, T);
            end
            idx = randperm(length(P), 3);
            k1 = P(idx(1)); k2 = P(idx(2)); k3 = P(idx(3));
            child = population(k1,:) + F*(population(k2,:) - population(k3,:));
            child = bound_handling(child, lb, ub);
            child = polynomial_mutation(child, mutation_rate, lb, ub);
            child_fit = fobj(child);
            z = min(z, child_fit);
            % 更新：若新解支配旧解或聚合函数值更优
            for j = 1:length(P)
                idx_j = P(j);
                if all(child_fit <= fitness(idx_j,:)) && any(child_fit < fitness(idx_j,:))
                    population(idx_j,:) = child;
                    fitness(idx_j,:) = child_fit;
                else
                    f_old = max(abs(fitness(idx_j,:) - z) ./ (weights(idx_j,:) + 1e-10));
                    f_new = max(abs(child_fit - z) ./ (weights(idx_j,:) + 1e-10));
                    if f_new < f_old
                        population(idx_j,:) = child;
                        fitness(idx_j,:) = child_fit;
                    end
                end
            end
        end
        Convergence_curve(t) = mean(fitness(:));
    end
    
    fronts = fast_non_dominated_sort(fitness);
    nd_idx = fronts{1};
    Pareto_fitness = fitness(nd_idx, :);
    Pareto_solution = population(nd_idx, :);
end

% 辅助函数与 MOEADD 相同（可共用，为独立文件可重复定义）
function weights = generate_weights_uniform(N, M)
    if M == 2
        weights = zeros(N,2);
        for i = 1:N
            weights(i,1) = (i-1)/(N-1);
            weights(i,2) = 1 - weights(i,1);
        end
    elseif M == 3
        p = round((N/2)^(1/3));
        weights = [];
        for i = 0:p
            for j = 0:p-i
                k = p - i - j;
                w = [i, j, k] / p;
                weights = [weights; w];
            end
        end
        if size(weights,1) > N
            idx = randperm(size(weights,1), N);
            weights = weights(idx,:);
        elseif size(weights,1) < N
            rep = ceil(N / size(weights,1));
            weights = repmat(weights, rep, 1);
            weights = weights(1:N,:);
        end
    else
        error('仅支持 2 或 3 目标');
    end
end

function individual = bound_handling(individual, lb, ub)
    individual = max(lb, min(ub, individual));
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