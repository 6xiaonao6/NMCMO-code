function [Pareto_fitness, Pareto_solution, Convergence_curve] = MOPSOCD(N, Max_iter, lb, ub, dim, fobj, num_obj)
% MOPSO with Crowding Distance (CD) archive pruning
    w = 0.4; c1 = 1.5; c2 = 1.5;
    archive_size = N;
    
    % 初始化
    positions = lb + rand(N, dim) .* (ub - lb);
    velocities = zeros(N, dim);
    fitness = zeros(N, num_obj);
    for i = 1:N
        fitness(i,:) = fobj(positions(i,:));
    end
    pbest_pos = positions;
    pbest_fit = fitness;
    
    archive = positions;
    archive_fit = fitness;
    Convergence_curve = zeros(Max_iter, 1);
    
    for t = 1:Max_iter
        % ---- 选择全局最优 ----
        if size(archive,1) > 0
            if size(archive,1) == 1
                gbest_idx = 1;
            else
                fronts = fast_non_dominated_sort(archive_fit);
                if ~isempty(fronts) && ~isempty(fronts{1})
                    nd_idx = fronts{1};
                else
                    nd_idx = 1:size(archive,1);
                end
                if length(nd_idx) > 1
                    dist = crowding_distance(archive_fit(nd_idx, :));
                    [~, best] = max(dist);
                    gbest_idx = nd_idx(best);
                else
                    gbest_idx = nd_idx(1);
                end
            end
            gbest_pos = archive(gbest_idx, :);
        else
            gbest_pos = pbest_pos(randi(N), :);
        end
        
        % ---- 更新粒子 ----
        for i = 1:N
            r1 = rand(1,dim); r2 = rand(1,dim);
            velocities(i,:) = w * velocities(i,:) + ...
                c1 * r1 .* (pbest_pos(i,:) - positions(i,:)) + ...
                c2 * r2 .* (gbest_pos - positions(i,:));
            velocities(i,:) = max(-(ub-lb)*0.5, min((ub-lb)*0.5, velocities(i,:)));
            positions(i,:) = positions(i,:) + velocities(i,:);
            positions(i,:) = max(lb, min(ub, positions(i,:)));
            new_fit = fobj(positions(i,:));
            fitness(i,:) = new_fit;
            % 更新个人最优
            if all(new_fit <= pbest_fit(i,:)) && any(new_fit < pbest_fit(i,:))
                pbest_pos(i,:) = positions(i,:);
                pbest_fit(i,:) = new_fit;
            elseif ~(all(pbest_fit(i,:) <= new_fit) && any(pbest_fit(i,:) < new_fit)) && rand < 0.5
                pbest_pos(i,:) = positions(i,:);
                pbest_fit(i,:) = new_fit;
            end
        end
        
        % ---- 更新存档 ----
        archive = [archive; positions];
        archive_fit = [archive_fit; fitness];
        [fronts, ~] = fast_non_dominated_sort(archive_fit);
        if ~isempty(fronts) && ~isempty(fronts{1})
            nd_idx = fronts{1};
            if max(nd_idx) <= size(archive,1)
                archive = archive(nd_idx, :);
                archive_fit = archive_fit(nd_idx, :);
            else
                nd_idx = 1:size(archive,1);
                archive = archive(nd_idx, :);
                archive_fit = archive_fit(nd_idx, :);
            end
        else
            if size(archive,1) > archive_size
                idx = randperm(size(archive,1), archive_size);
                archive = archive(idx, :);
                archive_fit = archive_fit(idx, :);
            end
        end
        % 用拥挤距离修剪（同时修剪 archive 和 archive_fit）
        if size(archive,1) > archive_size
            [archive, archive_fit] = prune_by_crowding(archive, archive_fit, archive_size);
        end
        
        Convergence_curve(t) = mean(fitness(:));
    end
    
    % ---- 输出最终帕累托前沿（确保 archive 和 archive_fit 行数一致） ----
    if size(archive,1) ~= size(archive_fit,1)
        % 若行数不一致，则取较小行数进行截断（通常不会发生）
        minRows = min(size(archive,1), size(archive_fit,1));
        archive = archive(1:minRows, :);
        archive_fit = archive_fit(1:minRows, :);
    end
    
    fronts = fast_non_dominated_sort(archive_fit);
    if ~isempty(fronts) && ~isempty(fronts{1})
        nd_idx = fronts{1};
        % 确保 nd_idx 中的索引不超过 archive 的行数
        if max(nd_idx) <= size(archive,1)
            Pareto_fitness = archive_fit(nd_idx, :);
            Pareto_solution = archive(nd_idx, :);
        else
            % 若索引超界，则返回所有存档个体（作为后备）
            Pareto_fitness = archive_fit;
            Pareto_solution = archive;
        end
    else
        Pareto_fitness = archive_fit;
        Pareto_solution = archive;
    end
end

% ==================== 辅助函数 ====================
function dist = crowding_distance(fitness)
    [N, M] = size(fitness);
    dist = zeros(N, 1);
    if N <= 2
        dist(:) = inf;
        return;
    end
    for m = 1:M
        [sorted_vals, sort_idx] = sort(fitness(:, m));
        dist(sort_idx(1)) = inf;
        dist(sort_idx(end)) = inf;
        f_max = sorted_vals(end);
        f_min = sorted_vals(1);
        if f_max ~= f_min
            for i = 2:(N-1)
                dist(sort_idx(i)) = dist(sort_idx(i)) + ...
                    (sorted_vals(i+1) - sorted_vals(i-1)) / (f_max - f_min);
            end
        end
    end
end

function [archive, archive_fit] = prune_by_crowding(archive, archive_fit, max_size)
    % 同时修剪 archive 和 archive_fit
    if size(archive,1) <= max_size
        return;
    end
    dist = crowding_distance(archive_fit);
    [~, idx] = sort(dist, 'descend');
    archive = archive(idx(1:max_size), :);
    archive_fit = archive_fit(idx(1:max_size), :);
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