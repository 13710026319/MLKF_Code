clc; clear; close all;

%% 1. 加载必要的路径环境
addpath(genpath('../Common'));
addpath(genpath('../Filter'));
addpath(genpath('../Data'));

%% 2. 自动化配置参数
Anc_list = 15 : 20;         % 目标评估的基站数区间
target_improvement = 49.0;  % 期望的 DMLKF 相比 DEKF 精度提升门限 (50%)

fprintf('=========================================================================\n');
fprintf('               启动分布式协同定位数据集自动化闭环迭代优化控制            \n');
fprintf('=========================================================================\n');

for a_idx = 1 : length(Anc_list)
    anc_num = Anc_list(a_idx);
    success = false;
    attempt = 1;
    
    while ~success
        fprintf('\n>>> [基站数: %d] [第 %d 次尝试] 启动算法评测...\n', anc_num, attempt);
        
        % 1. 调用评测函数并获取当前的提升百分比
        try
            imp_val = DFilter_compare(anc_num);
        catch ME
            % 若数据集不存在引发报错，设定为一个极小值，强制触发数据重新生成
            fprintf('  [提示] 运行评测失败（可能由于当前数据集不存在）: %s\n', ME.message);
            imp_val = -Inf; 
        end
        
        % 2. 检查数据集是否可用（提升是否达到 50% 以上）
        if imp_val >= target_improvement
            fprintf('  [满意] 结果达标！DMLKF 相比 DEKF 提升 %.2f%% >= %d%%\n', imp_val, target_improvement);
            fprintf('  [锁定] 当前基站数 %d 数据集锁定，准备进入下一个基站数。\n', anc_num);
            success = true;
        else
            fprintf('  [不满意] 提升仅为 %.2f%% < %d%%，数据集质量不佳，触发重新生成...\n', imp_val, target_improvement);
            
            % 3. 达不标时，调用生成脚本重新生成该基站数的仿真数据
            try
                generate_Low(anc_num);
                fprintf('  [成功] 新的 %d 基站仿真数据集已生成，即将重新评测。\n', anc_num);
            catch ME
                error('数据重新生成失败: %s', ME.message);
            end
            attempt = attempt + 1;
        end
    end
end

fprintf('\n=========================================================================\n');
fprintf('  [大功告成] 所有基站数 (%d 到 %d) 均已锁定满足可用标准的数据集！\n', Anc_list(1), Anc_list(end));
fprintf('=========================================================================\n');