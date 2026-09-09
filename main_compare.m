% ==================== 主对比脚本（9种算法，含绘图、归一化指标、保存） ====================
clear; clc; close all;

% ---- 保存目录 ----
save_dir = 'comparison_results';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% ---- 算法配置（9种） ----
algorithms = {'NMCMO', 'NSGAIII', 'MOEADD', 'CMOEAP', 'RVEA', 'NSGAII', 'ThetaDEA', 'SPEA2', 'MOPSOCD'};
algo_names = {'NMCMO', 'NSGA-III', 'MOEA/DD', 'CMOEAP', 'RVEA', 'NSGA-II', 'Θ-DEA', 'SPEA2', 'MOPSO-CD'};
num_algos = length(algorithms);

% 算法颜色（9种）
algo_colors = [
    0    0.4470    0.7410;  % NMCMOs
    0.8500    0.3250    0.0980;
    0.9290    0.6940    0.1250;
    0.4940    0.1840    0.5560;
    0.4660    0.6740    0.1880;
    0.3010    0.7450    0.9330;
    0.6350    0.0780    0.1840;
    0.2     0.6     0.8;
    0.9290    0.6940    0.1250  % 最后一个稍作调整
];
algo_colors(9,:) = [0.8 0.2 0.6];  % 区分最后一个

N = 200;          % 种群大小（正式实验建议200~300）
Max_iter = 200;   % 迭代次数（正式实验建议≥100）
dim = 12;
lb = zeros(1, dim);
ub = ones(1, dim);
num_obj = 3;

problems = {'DTLZ1','DTLZ2','DTLZ3','DTLZ4','DTLZ5','DTLZ6','DTLZ7'};

results = struct();

for p = 1:length(problems)
    prob = problems{p};
    fprintf('\n========== 问题：%s ==========\n', prob);
    fobj = @(x) DTLZ(x, prob);

    true_file = [prob '.txt'];
    true_front = [];
    if exist(true_file, 'file')
        data = importdata(true_file);
        if size(data,2) >= 3
            true_front = data(:,1:3);
        end
    else
        warning('理想前沿文件 %s 不存在，将跳过 IGD/HV 计算', true_file);
    end

    perf = struct();
    perf.IGD = []; perf.SP = []; perf.HV = [];
    conv_curves = [];          % 存储所有算法的收敛曲线（行向量）
    all_fronts = cell(num_algos, 1);

    for a = 1:num_algos
        algo = algorithms{a};
        fprintf('  运行算法：%s ...\n', algo_names{a});
        
        % 检查函数是否存在
        if ~exist(algo, 'file')
            error('算法函数 %s.m 不存在，请将文件放入当前目录。', algo);
        end
        
        % 调用算法（注意 NMCMO 需要额外传递 prob）
        if strcmp(algo, 'NMCMO')
            [Pareto_fitness, ~, conv] = feval(algo, N, Max_iter, lb, ub, dim, fobj, num_obj, prob);
        else
            [Pareto_fitness, ~, conv] = feval(algo, N, Max_iter, lb, ub, dim, fobj, num_obj);
        end
        
        all_fronts{a} = Pareto_fitness;

        % ---- 处理收敛曲线（统一为长度为 Max_iter 的行向量） ----
        if isvector(conv)
            if size(conv, 1) > 1
                conv = conv';   % 转为行向量
            end
        else
            % 如果是矩阵（例如多目标），取各目标的平均值作为性能指标
            conv = mean(conv, 2)';
        end
        if length(conv) < Max_iter
            conv = [conv, zeros(1, Max_iter - length(conv))];
        elseif length(conv) > Max_iter
            conv = conv(1:Max_iter);
        end
        conv_curves = [conv_curves; conv];

        % ---- 计算归一化指标 ----
        [igd, sp, hv] = compute_metrics(Pareto_fitness, true_front);
        perf.IGD = [perf.IGD; igd];
        perf.SP  = [perf.SP; sp];
        perf.HV  = [perf.HV; hv];
    end

    results.(prob).algorithms = algo_names;
    results.(prob).IGD = perf.IGD;
    results.(prob).SP  = perf.SP;
    results.(prob).HV  = perf.HV;

    % ---- 控制台输出性能表格 ----
    fprintf('\n%s 各算法性能指标（单次运行）：\n', prob);
    fprintf('%-12s %-12s %-12s %-12s\n', 'Algorithm', 'IGD', 'Spacing', 'HV');
    for a = 1:num_algos
        fprintf('%-12s %-12.6f %-12.6f %-12.6f\n', ...
            algo_names{a}, perf.IGD(a), perf.SP(a), perf.HV(a));
    end

    % ========== 图1：帕累托前沿对比（3行×3列） ==========
    fig_front = figure('Name', [prob ' 帕累托前沿对比'], 'Color', 'w', 'Position', [100, 100, 1400, 1000]);
    for a = 1:num_algos
        subplot(3, 3, a);
        front = all_fronts{a};
        if ~isempty(front)
            scatter3(front(:,1), front(:,2), front(:,3), 25, 'filled', ...
                'MarkerFaceColor', algo_colors(a,:), 'MarkerEdgeColor', 'k', ...
                'DisplayName', algo_names{a});
        end
        hold on;
        if ~isempty(true_front)
            scatter3(true_front(:,1), true_front(:,2), true_front(:,3), 15, ...
                'filled', 'MarkerFaceColor', [0.6 0.8 1.0], 'MarkerEdgeColor', 'none', ...
                'DisplayName', 'True PF');
        end
        xlabel('f1'); ylabel('f2'); zlabel('f3');
        title(algo_names{a});
        grid on; view(135,30);
        legend('Location','best', 'FontSize', 8);
        hold off;
    end
    sgtitle([prob ' 帕累托前沿对比'], 'FontSize', 14);
    drawnow;

    % ========== 图2：收敛曲线对比（单独一张图） ==========
    fig_conv = figure('Name', [prob ' 收敛曲线对比'], 'Color', 'w', 'Position', [200, 200, 800, 600]);
    for a = 1:num_algos
        plot(1:Max_iter, conv_curves(a,:), 'Color', algo_colors(a,:), 'LineWidth', 1.5);
        hold on;
    end
    xlabel('迭代次数'); 
    ylabel('平均目标值（归一化? 由算法定义）');
    title([prob ' 算法收敛曲线对比']);
    legend(algo_names, 'Location','southeast', 'FontSize', 9);
    grid on;
    hold off;
    drawnow;

    % ---- 保存图像 ----
    base_name = fullfile(save_dir, prob);
    % 保存前沿图
    fig_file_fig = [base_name '_front.fig'];
    fig_file_png = [base_name '_front.png'];
    savefig(fig_front, fig_file_fig);
    exportgraphics(fig_front, fig_file_png, 'Resolution', 600);
    % 保存收敛曲线
    fig_file_fig = [base_name '_conv.fig'];
    fig_file_png = [base_name '_conv.png'];
    savefig(fig_conv, fig_file_fig);
    exportgraphics(fig_conv, fig_file_png, 'Resolution', 600);
end

% ===================== 辅助函数（指标计算，含归一化） =====================
function [IGD, SP, HV] = compute_metrics(approx_front, true_front)
    % 计算 IGD、Spacing、HV（归一化后）
    if isempty(approx_front)
        IGD = inf; SP = inf; HV = 0;
        return;
    end
    % 非支配筛选
    [fronts, ~] = fast_non_dominated_sort(approx_front);
    if ~isempty(fronts) && ~isempty(fronts{1})
        nd_idx = fronts{1};
    else
        nd_idx = 1:size(approx_front,1);
    end
    approx_front = approx_front(nd_idx, :);
    if isempty(approx_front)
        IGD = inf; SP = inf; HV = 0;
        return;
    end
    
    % 归一化：基于真实前沿的范围
    if ~isempty(true_front)
        min_vals = min(true_front, [], 1);
        max_vals = max(true_front, [], 1);
        range = max_vals - min_vals;
        range(range == 0) = 1;
        true_front_norm = (true_front - min_vals) ./ range;
        approx_norm = (approx_front - min_vals) ./ range;
    else
        % 无真实前沿时，用近似前沿自身范围
        min_vals = min(approx_front, [], 1);
        max_vals = max(approx_front, [], 1);
        range = max_vals - min_vals;
        range(range == 0) = 1;
        true_front_norm = [];
        approx_norm = (approx_front - min_vals) ./ range;
    end
    
    % ---- IGD ----
    if ~isempty(true_front_norm)
        D = pdist2(true_front_norm, approx_norm);
        minD = min(D, [], 2);
        IGD = sqrt(mean(minD.^2));
    else
        IGD = inf;
    end
    
    % ---- Spacing ----
    K = size(approx_norm, 1);
    if K <= 1
        SP = 0;
    else
        D2 = pdist2(approx_norm, approx_norm);
        D2(logical(eye(K))) = inf;
        minD2 = min(D2, [], 2);
        d_mean = mean(minD2);
        SP = sqrt(mean((minD2 - d_mean).^2));
    end
    
    % ---- HV (参考点 1.1) ----
    if ~isempty(true_front_norm) && ~isempty(approx_norm)
        ref_point = 1.1 * ones(1, size(approx_norm, 2));
        HV = hypervolume_mc(approx_norm, ref_point, 10000);
    else
        HV = 0;
    end
end

function HV = hypervolume_mc(front, ref_point, N_samples)
    if nargin < 3, N_samples = 10000; end
    if isempty(front), HV = 0; return; end
    M = size(front, 2);
    min_vals = zeros(1, M);
    volume = prod(ref_point - min_vals);
    samples = rand(N_samples, M);
    for i = 1:M
        samples(:,i) = samples(:,i) * ref_point(i);
    end
    dominated = false(N_samples,1);
    for s = 1:N_samples
        for j = 1:size(front,1)
            if all(front(j,:) <= samples(s,:)) && any(front(j,:) < samples(s,:))
                dominated(s) = true;
                break;
            end
        end
    end
    HV = (sum(dominated) / N_samples) * volume;
end

% 非支配排序（内联）
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