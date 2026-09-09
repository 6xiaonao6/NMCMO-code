function y = DTLZ(x, problem)
% 标准 DTLZ 系列测试函数（三目标）
% 输入：x - 1×dim 决策变量，dim >= 3
%      problem - 'DTLZ1'~'DTLZ7'
%       x向量：包含所有决策变量，长度为问题维度dim
% 输出：y - 1×3 目标值

    dim = length(x);
    x1 = x(1);
    x2 = x(2);
    % 从向量x中截取第3个元素到最后一个元素，把这一段数据存到新变量 xM 里
    xM = x(3:end);  % 决策变量的后k个变量表示为xM
    k = dim - 2;   % k=n-M+1    （n=dim）

    switch lower(problem)
        case 'dtlz1'
            g = 100 * (k + sum((xM - 0.5).^2 - cos(20*pi*(xM-0.5))));
            y(1) = (1 + g) * 0.5 * x1 * x2;
            y(2) = (1 + g) * 0.5 * x1 * (1 - x2);
            y(3) = (1 + g) * 0.5 * (1 - x1);
            % 非负约束保留（合理）
            y = max(y, 0);
            % 对DTLZ1进行归一化处理，使其范围在[0, 0.5]左右
        case 'dtlz2'
            g = sum((xM - 0.5).^2);
            y(1) =  (1 + g) *cos(x1*pi/2) * cos(x2*pi/2);
            y(2) =  (1 + g) *sin(x1*pi/2) * cos(x2*pi/2);
            y(3) =  (1 + g) *sin(x2*pi/2);
        case 'dtlz3'
            g = 100 * (k + sum((xM - 0.5).^2 - cos(20*pi*(xM-0.5))));
            y(1) = (1 + g) * cos(x1 * pi/2) * cos(x2 * pi/2);
            y(2) = (1 + g) * sin(x1 * pi/2) * cos(x2 * pi/2);
            y(3) = (1 + g) * sin(x2 * pi/2);
        case 'dtlz4'
            alpha = 100;
            x1a = x1^alpha;
            x2a = x2^alpha;
            g = sum((xM - 0.5).^2);
            y(1) = (1+g) * cos(x1a*pi/2) * cos(x2a*pi/2);
            y(2) = (1+g) * cos(x1a*pi/2) * sin(x2a*pi/2);
            y(3) = (1+g) * sin(x1a*pi/2);
        case 'dtlz5'
            g = sum((xM - 0.5).^2);
            theta1 = x1 * pi/2;
            theta2 = pi * (1 + 2*g*x2) / (4*(1+g));
            y(1) = (1+g) * cos(theta1) * cos(theta2);
            y(2) = (1+g) * cos(theta1) * sin(theta2);
            y(3) = (1+g) * sin(theta1);
        case 'dtlz6'
            g = sum(xM .^ 2);
            theta1 = x1 * pi/2;
            theta2 = pi * (1 + 2*g*x2) / (4*(1+g));
            y(1) = (1+g) * cos(theta1) * cos(theta2);
            y(2) = (1+g) * cos(theta1) * sin(theta2);
            y(3) = (1+g) * sin(theta1);        
        case 'dtlz7'
            g = 1 + 9 * sum(xM) / k;
            f1 = x1;
            f2 = x2;
            h = 3 - (f1/(1+g)).*(1+sin(3*pi*f1)) - (f2/(1+g)).*(1+sin(3*pi*f2));
            y = [f1, f2, (1+g)*h];
        otherwise
            error('不支持的问题：%s', problem);
    end
end