% ==================== complexity_theoretical_plot.m ====================
% 基于大O理论复杂度绘制各算法随 M, N, D 变化的曲线
% 输出：3张理论复杂度对比图（9种算法）
% ========================================================================
clear; clc; close all;

% 定义颜色和线型（9种）
colors = lines(9);
algo_names = {'NMCMO', 'NSGA-II', 'NSGA-III', 'MOEA/DD', 'CMOEAP', 'RVEA', 'Θ-DEA', 'SPEA2', 'MOPSO-CD'};

%% ========== 图1：复杂度随目标数 M 的变化 ==========
% 固定 N=200, D=12
N_fixed = 200;
D_fixed = 12;
M_vals = 2:10;

% 复杂度公式（近似大O）
C_NMCMO   = M_vals .* N_fixed^2 + N_fixed * D_fixed;   % O(M*N^2 + N*D)
C_NSGA2   = M_vals .* N_fixed^2;                        % O(M*N^2)
C_NSGA3   = M_vals .* N_fixed^2;                        % O(M*N^2)
C_MOEADD  = M_vals .* N_fixed;                          % O(M*N)（T为常数）
C_CMOEAP  = M_vals .* N_fixed;                          % O(M*N)
C_RVEA    = M_vals .* N_fixed^2;                        % O(M*N^2)
C_ThetaDEA= M_vals .* N_fixed^2;                        % O(M*N^2)
C_SPEA2   = M_vals .* N_fixed^2;                        % O(M*N^2)
C_MOPSOCD = N_fixed * D_fixed * ones(size(M_vals));     % O(N*D)（不随M变化）

figure('Name', '复杂度 vs M', 'Color', 'w', 'Position', [100, 100, 800, 600]);
hold on;
plot(M_vals, C_NMCMO,   'o-', 'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(M_vals, C_NSGA2,   's-', 'Color', colors(2,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(M_vals, C_NSGA3,   'd-', 'Color', colors(3,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(M_vals, C_MOEADD,  '^-', 'Color', colors(4,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(M_vals, C_CMOEAP,  'v-', 'Color', colors(5,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(M_vals, C_RVEA,    '*-', 'Color', colors(6,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(M_vals, C_ThetaDEA,'x-', 'Color', colors(7,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(M_vals, C_SPEA2,   '+-', 'Color', colors(8,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(M_vals, C_MOPSOCD, '.-', 'Color', colors(9,:), 'LineWidth', 2, 'MarkerSize', 8);
hold off;

set(gca, 'YScale', 'log');
xlabel('目标数 M');
ylabel('理论计算量 (log scale)');
title(sprintf('复杂度随 M 增长 (N=%d, D=%d)', N_fixed, D_fixed));
legend(algo_names, 'Location', 'northwest');
grid on;

%% ========== 图2：复杂度随种群规模 N 的变化 ==========
% 固定 M=3, D=12
M_fixed = 3;
D_fixed = 12;
N_vals = 50:50:500;

C_NMCMO   = M_fixed .* N_vals.^2 + N_vals * D_fixed;
C_NSGA2   = M_fixed .* N_vals.^2;
C_NSGA3   = M_fixed .* N_vals.^2;
C_MOEADD  = M_fixed .* N_vals;
C_CMOEAP  = M_fixed .* N_vals;
C_RVEA    = M_fixed .* N_vals.^2;
C_ThetaDEA= M_fixed .* N_vals.^2;
C_SPEA2   = M_fixed .* N_vals.^2;
C_MOPSOCD = N_vals * D_fixed;

figure('Name', '复杂度 vs N', 'Color', 'w', 'Position', [100, 100, 800, 600]);
hold on;
plot(N_vals, C_NMCMO,   'o-', 'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(N_vals, C_NSGA2,   's-', 'Color', colors(2,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(N_vals, C_NSGA3,   'd-', 'Color', colors(3,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(N_vals, C_MOEADD,  '^-', 'Color', colors(4,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(N_vals, C_CMOEAP,  'v-', 'Color', colors(5,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(N_vals, C_RVEA,    '*-', 'Color', colors(6,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(N_vals, C_ThetaDEA,'x-', 'Color', colors(7,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(N_vals, C_SPEA2,   '+-', 'Color', colors(8,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(N_vals, C_MOPSOCD, '.-', 'Color', colors(9,:), 'LineWidth', 2, 'MarkerSize', 8);
hold off;

set(gca, 'YScale', 'log');
xlabel('种群规模 N');
ylabel('理论计算量 (log scale)');
title(sprintf('复杂度随 N 增长 (M=%d, D=%d)', M_fixed, D_fixed));
legend(algo_names, 'Location', 'northwest');
grid on;

%% ========== 图3：复杂度随决策变量维度 D 的变化 ==========
% 固定 M=3, N=200
M_fixed = 3;
N_fixed = 200;
D_vals = 10:20:200;

C_NMCMO   = M_fixed .* N_fixed^2 + N_fixed * D_vals;
C_NSGA2   = M_fixed .* N_fixed^2 * ones(size(D_vals));
C_NSGA3   = M_fixed .* N_fixed^2 * ones(size(D_vals));
C_MOEADD  = M_fixed .* N_fixed * ones(size(D_vals));
C_CMOEAP  = M_fixed .* N_fixed * ones(size(D_vals));
C_RVEA    = M_fixed .* N_fixed^2 * ones(size(D_vals));
C_ThetaDEA= M_fixed .* N_fixed^2 * ones(size(D_vals));
C_SPEA2   = M_fixed .* N_fixed^2 * ones(size(D_vals));
C_MOPSOCD = N_fixed * D_vals;

figure('Name', '复杂度 vs D', 'Color', 'w', 'Position', [100, 100, 800, 600]);
hold on;
plot(D_vals, C_NMCMO,   'o-', 'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(D_vals, C_NSGA2,   's-', 'Color', colors(2,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(D_vals, C_NSGA3,   'd-', 'Color', colors(3,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(D_vals, C_MOEADD,  '^-', 'Color', colors(4,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(D_vals, C_CMOEAP,  'v-', 'Color', colors(5,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(D_vals, C_RVEA,    '*-', 'Color', colors(6,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(D_vals, C_ThetaDEA,'x-', 'Color', colors(7,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(D_vals, C_SPEA2,   '+-', 'Color', colors(8,:), 'LineWidth', 2, 'MarkerSize', 8);
plot(D_vals, C_MOPSOCD, '.-', 'Color', colors(9,:), 'LineWidth', 2, 'MarkerSize', 8);
hold off;

set(gca, 'YScale', 'log');
xlabel('决策变量维度 D');
ylabel('理论计算量 (log scale)');
title(sprintf('复杂度随 D 增长 (M=%d, N=%d)', M_fixed, N_fixed));
legend(algo_names, 'Location', 'northwest');
grid on;

% 保存图片（可选）
% saveas(gcf, 'complexity_theoretical.png');