% ========================================================================
% run_NMCMO_experiment.m - 完整性能验证实验（含贝叶斯因子）
% 功能：
%   1. 对 DTLZ1~DTLZ7，在 dim = [12,30,50] 上独立运行 30 次
%   2. 计算 IGD、HV
%   3. 输出均值±标准差表格（Excel），包含 Wilcoxon p、Holm校正、效应量、贝叶斯因子
%   4. 执行 Friedman 检验并绘制 Critical Difference 图
%   5. 计算贝叶斯因子（独立样本 JZS t 检验）
% ========================================================================
clear; clc; close all;

%% ==================== 参数设置 ====================
runs = 30;               % 正式实验设为 30
N = 200;                % 种群大小
Max_iter = 200;          % 迭代次数（演示用，实际会基于 MaxFEs 重新计算）
dim_list = [12, 30];    % 正式实验加入 50
num_obj = 3;
lb = zeros(1, dim_list(1));  % 临时
ub = ones(1, dim_list(1));

% 算法列表（共9种）
algorithms = {'NMCMO', 'NSGAII', 'NSGAIII', 'MOEADD', 'CMOEAP', 'RVEA', 'ThetaDEA', 'SPEA2', 'MOPSOCD'};
algo_names = {'NMCMO', 'NSGA-II', 'NSGA-III', 'MOEA/DD', 'CMOEAP', 'RVEA', 'Θ-DEA', 'SPEA2', 'MOPSO-CD'};
num_algos = length(algorithms);

% 问题列表
problems = {'DTLZ1','DTLZ2','DTLZ3','DTLZ4','DTLZ5','DTLZ6','DTLZ7'};

% 结果存储结构
results = struct();

%% ==================== 主循环 ====================
for p = 1:length(problems)
    prob = problems{p};
    fprintf('\n========== 问题：%s ==========\n', prob);
    fobj = @(x) DTLZ(x, prob);
    
    true_file = [prob '.txt'];
    if exist(true_file, 'file')
        true_data = importdata(true_file);
        if size(true_data,2) >= 3
            true_front = true_data(:,1:3);
        else
            error('真实前沿文件格式错误');
        end
    else
        error('请提供真实前沿文件 %s', true_file);
    end
    
    for d = 1:length(dim_list)
        dim = dim_list(d);
        fprintf('  维度 D = %d\n', dim);
        lb = zeros(1, dim);
        ub = ones(1, dim);
        MaxFEs = dim * 1000;   % 正式实验：dim*1000
        Max_iter = ceil(MaxFEs / N);
        
        % 存储当前维度下各算法的运行结果
        alg_igd = zeros(runs, num_algos);
        alg_hv  = zeros(runs, num_algos);
        
        for a = 1:num_algos
            algo = algorithms{a};
            fprintf('    运行算法：%s ...\n', algo_names{a});
            
            if ~exist(algo, 'file')
                error('算法函数 %s.m 不存在', algo);
            end
            
            igd_vals = zeros(runs, 1);
            hv_vals  = zeros(runs, 1);
            
            for r = 1:runs
                if strcmp(algo, 'NMCMO')
                    [Pareto_fitness, ~, ~] = feval(algo, N, Max_iter, lb, ub, dim, fobj, num_obj, prob);
                else
                    [Pareto_fitness, ~, ~] = feval(algo, N, Max_iter, lb, ub, dim, fobj, num_obj);
                end
                [igd, hv] = compute_IGD_HV(Pareto_fitness, true_front);
                igd_vals(r) = igd;
                hv_vals(r) = hv;
            end
            
            alg_igd(:, a) = igd_vals;
            alg_hv(:, a) = hv_vals;
        end
        
        results.(prob).(['D' num2str(dim)]).IGD = alg_igd;
        results.(prob).(['D' num2str(dim)]).HV = alg_hv;
        results.(prob).(['D' num2str(dim)]).algorithms = algo_names;
    end
end

%% ==================== 统计分析 ====================
fprintf('\n========== 开始统计分析 ==========\n');

% ---- 1. Friedman 检验 + CD 图 ----
metric_names = {'IGD', 'HV'};
for mi = 1:length(metric_names)
    metric = metric_names{mi};
    scenario_count = length(problems) * length(dim_list);
    data_matrix = zeros(scenario_count, num_algos);
    row = 1;
    for p = 1:length(problems)
        prob = problems{p};
        for d = 1:length(dim_list)
            dim_key = ['D' num2str(dim_list(d))];
            tmp = results.(prob).(dim_key).(metric);
            mean_vals = mean(tmp, 1);
            data_matrix(row, :) = mean_vals;
            row = row + 1;
        end
    end
    
    [p_friedman, ~, stats] = friedman(data_matrix, 1, 'off');
    fprintf('\n指标 %s 的 Friedman p 值 = %.4e\n', metric, p_friedman);
    
    avg_ranks = stats.meanranks;
    if strcmp(metric, 'HV')
        avg_ranks = num_algos + 1 - avg_ranks;
    end
    fprintf('各算法平均排名（越小越好）：\n');
    for i = 1:num_algos
        fprintf('  %-10s : %.3f\n', algo_names{i}, avg_ranks(i));
    end
    
    k = num_algos; n = scenario_count;
    q_alpha = 2.85;
    CD = q_alpha * sqrt((k*(k+1))/(6*n));
    fprintf('临界差异 CD = %.4f\n', CD);
    
    figure('Name', ['CD图 - ' metric], 'Color', 'w', 'Position', [100,100,1000,350]);
    draw_cd_diagram(avg_ranks, algo_names, CD, metric);
    saveas(gcf, ['CD_diagram_' metric '.png']);
end

% ---- 2. Wilcoxon + Holm + 贝叶斯因子（以 NMCMO 为对照） ----
fprintf('\n=== 详细对比（NMCMO vs 其他算法）===\n');
nmcmo_idx = find(strcmp(algorithms, 'NMCMO'));

% 用于保存贝叶斯因子结果（可写入 Excel）
bayes_results = struct();

for d = 1:length(dim_list)
    dim = dim_list(d);
    fprintf('\n维度 D=%d\n', dim);
    
    % 收集该维度下所有问题的数据（合并）
    all_igd = []; all_hv = [];
    for p = 1:length(problems)
        prob = problems{p};
        dim_key = ['D' num2str(dim)];
        igd_mat = results.(prob).(dim_key).IGD;
        hv_mat  = results.(prob).(dim_key).HV;
        all_igd = [all_igd; igd_mat];
        all_hv  = [all_hv; hv_mat];
    end
    
    nmcmo_igd = all_igd(:, nmcmo_idx);
    nmcmo_hv  = all_hv(:, nmcmo_idx);
    
    p_vals_igd = zeros(1, num_algos);
    p_vals_hv  = zeros(1, num_algos);
    cliff_igd = zeros(1, num_algos);
    cliff_hv  = zeros(1, num_algos);
    bf10_igd  = zeros(1, num_algos);   % 存储贝叶斯因子
    bf10_hv   = zeros(1, num_algos);
    
    for a = 1:num_algos
        if a == nmcmo_idx, continue; end
        % IGD
        p = ranksum(nmcmo_igd, all_igd(:, a));
        p_vals_igd(a) = p;
        cliff_igd(a) = compute_cliff_delta(nmcmo_igd, all_igd(:, a));
        bf10_igd(a) = bayes_factor_indep(nmcmo_igd, all_igd(:, a));   % 新增
        
        % HV
        p = ranksum(nmcmo_hv, all_hv(:, a));
        p_vals_hv(a) = p;
        cliff_hv(a) = compute_cliff_delta(nmcmo_hv, all_hv(:, a));
        bf10_hv(a) = bayes_factor_indep(nmcmo_hv, all_hv(:, a));
    end
    
    % Holm 校正
    p_adj_igd = holm_adjust(p_vals_igd);
    p_adj_hv  = holm_adjust(p_vals_hv);
    
    % 显示结果（含贝叶斯因子）
    fprintf('  IGD:\n');
    fprintf('    %-10s %-12s %-12s %-12s %-12s\n', 'Algorithm', 'p_raw', 'p_holm', 'Cliff', 'BF10');
    for a = 1:num_algos
        if a == nmcmo_idx, continue; end
        fprintf('    %-10s %-12.2e %-12.2e %-12.3f %-12.2f\n', ...
            algo_names{a}, p_vals_igd(a), p_adj_igd(a), cliff_igd(a), bf10_igd(a));
    end
    fprintf('  HV:\n');
    fprintf('    %-10s %-12s %-12s %-12s %-12s\n', 'Algorithm', 'p_raw', 'p_holm', 'Cliff', 'BF10');
    for a = 1:num_algos
        if a == nmcmo_idx, continue; end
        fprintf('    %-10s %-12.2e %-12.2e %-12.3f %-12.2f\n', ...
            algo_names{a}, p_vals_hv(a), p_adj_hv(a), cliff_hv(a), bf10_hv(a));
    end
    
    % 保存贝叶斯结果到结构（后续可写入 Excel）
    bayes_results.(['D' num2str(dim)]).IGD.p = p_vals_igd;
    bayes_results.(['D' num2str(dim)]).IGD.p_holm = p_adj_igd;
    bayes_results.(['D' num2str(dim)]).IGD.cliff = cliff_igd;
    bayes_results.(['D' num2str(dim)]).IGD.BF10 = bf10_igd;
    bayes_results.(['D' num2str(dim)]).HV.p = p_vals_hv;
    bayes_results.(['D' num2str(dim)]).HV.p_holm = p_adj_hv;
    bayes_results.(['D' num2str(dim)]).HV.cliff = cliff_hv;
    bayes_results.(['D' num2str(dim)]).HV.BF10 = bf10_hv;
end

% ---- 可选：将贝叶斯结果写入 Excel ----
try
    for d = 1:length(dim_list)
        dim = dim_list(d);
        sheet_name = ['D' num2str(dim)];
        % 构建表格
        T = table(algo_names', 'VariableNames', {'Algorithm'});
        T.p_IGD = bayes_results.(['D' num2str(dim)]).IGD.p';
        T.p_holm_IGD = bayes_results.(['D' num2str(dim)]).IGD.p_holm';
        T.cliff_IGD = bayes_results.(['D' num2str(dim)]).IGD.cliff';
        T.BF10_IGD = bayes_results.(['D' num2str(dim)]).IGD.BF10';
        T.p_HV = bayes_results.(['D' num2str(dim)]).HV.p';
        T.p_holm_HV = bayes_results.(['D' num2str(dim)]).HV.p_holm';
        T.cliff_HV = bayes_results.(['D' num2str(dim)]).HV.cliff';
        T.BF10_HV = bayes_results.(['D' num2str(dim)]).HV.BF10';
        writetable(T, 'Bayes_results.xlsx', 'Sheet', sheet_name, 'WriteMode', 'overwritesheet');
    end
    fprintf('\n贝叶斯因子结果已写入 Bayes_results.xlsx\n');
catch ME
    % ========== 修正处：添加消息标识符 ==========
    warning('MATLAB:experiment:ioError', '无法写入 Excel 文件: %s', ME.message);
    % ===========================================
end

fprintf('\n所有实验完成！\n');

%% ===================== 辅助函数 =====================

% ----- 计算 IGD 和 HV （同原代码） -----
function [IGD, HV] = compute_IGD_HV(approx_front, true_front)
    if isempty(approx_front)
        IGD = inf; HV = 0; return;
    end
    [fronts, ~] = fast_non_dominated_sort(approx_front);
    if ~isempty(fronts) && ~isempty(fronts{1})
        nd_idx = fronts{1};
    else
        nd_idx = 1:size(approx_front,1);
    end
    approx_front = approx_front(nd_idx, :);
    if isempty(approx_front)
        IGD = inf; HV = 0; return;
    end
    min_vals = min(true_front, [], 1);
    max_vals = max(true_front, [], 1);
    range = max_vals - min_vals;
    range(range == 0) = 1;
    true_norm = (true_front - min_vals) ./ range;
    approx_norm = (approx_front - min_vals) ./ range;
    D = pdist2(true_norm, approx_norm);
    minD = min(D, [], 2);
    IGD = sqrt(mean(minD.^2));
    ref_point = 1.1 * ones(1, size(approx_norm,2));
    HV = hypervolume_mc(approx_norm, ref_point, 100000);
end

function HV = hypervolume_mc(front, ref_point, N_samples)
    if isempty(front), HV = 0; return; end
    M = size(front,2);
    min_vals = zeros(1,M);
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

% ----- 非支配排序（内联）-----
function [F, ranks] = fast_non_dominated_sort(fitness)
    [N, M] = size(fitness);
    S = cell(N, 1); n = zeros(N,1);
    F = {}; ranks = zeros(N,1);
    for i = 1:N
        S{i} = []; n(i) = 0;
        for j = 1:N
            if i==j, continue; end
            if all(fitness(i,:) <= fitness(j,:)) && any(fitness(i,:) < fitness(j,:))
                S{i} = [S{i}, j];
            elseif all(fitness(j,:) <= fitness(i,:)) && any(fitness(j,:) < fitness(i,:))
                n(i) = n(i) + 1;
            end
        end
        if n(i) == 0
            ranks(i) = 1;
            if isempty(F), F{1} = i; else F{1} = [F{1}, i]; end
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
                    ranks(q) = i+1;
                    Q = [Q, q];
                end
            end
        end
        i = i+1;
        if ~isempty(Q), F{i} = Q; else break; end
    end
end

% ----- Cliff's delta (同原代码) -----
function cliff_d = compute_cliff_delta(x, y)
    n1 = length(x); n2 = length(y);
    if n1==0 || n2==0, cliff_d=0; return; end
    combined = [x; y];
    ranks = tiedrank(combined);
    rank_sum_x = sum(ranks(1:n1));
    U = n1*n2 + n1*(n1+1)/2 - rank_sum_x;
    U = max(0, min(n1*n2, U));
    cliff_d = (U/(n1*n2) - 0.5) * 2;
end

% ----- Holm 校正 (同原代码) -----
function p_adj = holm_adjust(p_vals)
    m = length(p_vals);
    [sorted_p, sort_idx] = sort(p_vals);
    p_adj = zeros(1,m);
    for i = 1:m
        p_adj(sort_idx(i)) = min(1, sorted_p(i) * (m - i + 1));
    end
    [~, rev] = sort(sorted_p, 'descend');
    for i = 2:m
        if p_adj(sort_idx(rev(i))) < p_adj(sort_idx(rev(i-1)))
            p_adj(sort_idx(rev(i))) = p_adj(sort_idx(rev(i-1)));
        end
    end
end

% ----- 绘制 Critical Difference 图 (同原代码) -----
function draw_cd_diagram(avg_ranks, algo_names, CD, title_str)
    k = length(avg_ranks);
    [sorted_ranks, idx] = sort(avg_ranks);
    sorted_names = algo_names(idx);
    figure(gcf); clf; hold on;
    set(gcf, 'Position', [200,200,1000,350]);
    x_min = 0.5; x_max = k + 0.5;
    plot([x_min, x_max], [0,0], 'k-', 'LineWidth',1.5);
    y_offset = 0.12;
    for i = 1:k
        x_pos = i;
        plot(x_pos, 0, 'ko', 'MarkerSize',8, 'LineWidth',2, 'MarkerFaceColor','w');
        line([x_pos, x_pos], [0, sign(i-(k+1)/2)*y_offset*1.5], 'Color','k','LineWidth',1);
        if mod(i,2)==1
            y_text = y_offset; va = 'bottom';
        else
            y_text = -y_offset; va = 'top';
        end
        text(x_pos, y_text, sprintf('%s\n(%.2f)', sorted_names{i}, sorted_ranks(i)), ...
            'HorizontalAlignment','center','VerticalAlignment',va,'FontSize',11,'FontWeight','bold');
    end
    for i = 1:k
        for j = i+1:k
            if abs(sorted_ranks(i)-sorted_ranks(j)) < CD
                line([i,j], [0,0], 'Color',[0.6 0.6 0.6], 'LineWidth',3);
            end
        end
    end
    y_scale = -0.2;
    plot([1, 1+CD], [y_scale, y_scale], 'k-','LineWidth',2);
    text(1+CD/2, y_scale-0.03, sprintf('CD=%.3f',CD), 'HorizontalAlignment','center','FontSize',10);
    line([1,1], [y_scale-0.03,y_scale+0.03], 'Color','k','LineWidth',1.5);
    line([1+CD,1+CD], [y_scale-0.03,y_scale+0.03], 'Color','k','LineWidth',1.5);
    xlim([x_min, x_max]); ylim([-0.35,0.35]);
    axis off;
    title(title_str, 'FontSize',14,'FontWeight','bold');
    text(x_min, -0.3, 'Gray lines: no significant difference (rank difference < CD)', ...
        'HorizontalAlignment','left','FontSize',9,'FontAngle','italic');
    hold off;
end

% ========== 新增：贝叶斯因子（独立样本 JZS t 检验）==========
function BF10 = bayes_factor_indep(x, y)
    % 计算独立样本贝叶斯因子 BF10（支持 H1 相对于 H0）
    % 使用 Jeffreys-Zellner-Siow (JZS) 先验，基于 Rouder 等 (2009)
    % 输入：x, y 为两个独立样本的数值向量
    % 输出：BF10，>1 表示支持 H1（有差异），<1 支持 H0
    
    n1 = length(x); n2 = length(y);
    if n1 < 2 || n2 < 2
        BF10 = NaN;
        return;
    end
    
    % 合并样本，计算 t 统计量（Welch 校正）
    mean_x = mean(x); mean_y = mean(y);
    var_x = var(x); var_y = var(y);
    se = sqrt(var_x/n1 + var_y/n2);
    t_stat = (mean_x - mean_y) / se;
    df = (var_x/n1 + var_y/n2)^2 / ( (var_x/n1)^2/(n1-1) + (var_y/n2)^2/(n2-1) );
    
    % JZS prior: 使用数值积分计算 BF10
    N_eff = 1 / (1/n1 + 1/n2);
    r = linspace(0.001, 0.999, 500);
    g_r = r ./ (1 - r);
    dg_dr = 1 ./ (1 - r).^2;
    term1 = (1 + N_eff .* g_r).^(-0.5);
    term2 = (1 + t_stat^2 ./ ((1 + N_eff .* g_r) .* df)).^(- (df+1)/2);
    integrand = term1 .* term2 .* dg_dr;
    integral = trapz(r, integrand);
    BF10 = integral;
    
    % 处理极小/极大值
    if BF10 > 1e6, BF10 = 1e6; end
    if BF10 < 1e-6, BF10 = 1e-6; end
end