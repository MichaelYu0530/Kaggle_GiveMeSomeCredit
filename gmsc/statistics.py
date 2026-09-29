"""Statistics functions from the original research pipeline."""

from __future__ import annotations

from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
import random
from .schema import age, credit, dep, income, mortgage, target



def format_p_value(p: float) -> str:
    if p < 0.001:
        return "<0.001***"
    elif p < 0.01:
        return f"{p:.4f}**"
    elif p < 0.05:
        return f"{p:.4f}*"
    else:
        return f"{p:.4f}"


def format_stat(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.2e}"
    else:
        return f"{value:.2f}"


def smart_round(x: float, decimals: int=3) -> int | float | str:
    if x > 10000:
        return f"{x:.2e}"
    
    if x == int(x):
        return int(x)
    
    str_x = str(x)
    if '.' in str_x: # 检查是否为小数
        decimal_part = str_x.split('.')[1].rstrip('0') # 去掉末尾的0后的小数位数
        if len(decimal_part) < decimals:
            return x
        else:
            return round(x, decimals)


def format_interval(interval: pd.Interval) -> str:
    left, right, closed = interval.left, interval.right, interval.closed
    
    # 对异常值或缺失值的特殊处理
    if left == -1:
        return '不可靠组1'
    if left == -2:
        return '不可靠组2'
    if left == -3:
        return '不可靠组3'
    if left == -4:
        return '不可靠组4'
    
    # 科学计数法或者四舍五入
    left_str = smart_round(left)
    right_str = smart_round(right)
    # 根据闭合方式返回对应的括号
    if closed == 'both':
        return f"[{left_str}, {right_str}]"
    elif closed == 'neither':
        return f"({left_str}, {right_str})"
    elif closed == 'left':
        return f"[{left_str}, {right_str})"
    else:  # 'right'
        return f"({left_str}, {right_str}]"


def conduct_independent_t(groups: list[list[float]]) -> tuple[float, float, float]:
    """
    对两组独立样本执行Welch t检验，返回t统计量、p值和Cohen's d效应量
    
    参数
    ----------
    groups : list or tuple
        包含两个数组的列表或元组，每个数组为一组独立样本的观测值
    
    返回
    ----------
    t_stat : float
        Welch t检验的t统计量
    p_value : float
        双尾检验的p值，若任一组样本量为0则返回1
    cohens_d : float
        Cohen's d效应量（使用两组的平均标准差），衡量两组均值差异的大小
    """
    
    if len(groups) != 2:
        raise ValueError("In function conduct_independent_t: independent t test requires 2 groups.")
    
    group1, group2 = np.asarray(groups[0]), np.asarray(groups[1])
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan, 1, 0
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # 执行Welch t检验
    t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
    
    if std1 == 0 and std2 == 0:
        cohens_d = 0.0
    else:
        sd_avg = np.sqrt((std1**2 + std2**2) / 2)
        cohens_d = (mean1 - mean2) / sd_avg
    
    return t_stat, p_value, cohens_d


def judge_cohens_d(d: float, small_threshold: float=0.2, medium_threshold: float=0.5) -> str:
    """
    判断Cohen's d效应量的业务意义
    
    参数
    ----------
    d : float
        Cohen's d效应量值（将自动取绝对值进行判断）
    small_threshold : float, default=0.2
        小效应的上限阈值，小于此值为小效应
    medium_threshold : float, default=0.5
        中效应的上限阈值，大于等于small_threshold且小于此值为中效应
    
    返回
    ----------
    str
        效应量大小的业务解释：
        - '小效应'：|d| < small_threshold
        - '中效应'：small_threshold ≤ |d| < medium_threshold
        - '大效应'：|d| ≥ medium_threshold
    """
    
    d = abs(d)
    if d < small_threshold:
        return '小效应'
    elif d < medium_threshold:
        return '中效应'
    else:
        return '大效应'


def conduct_one_way_anova(groups: list[list[float]]) -> tuple[float, float, float]:
    """
    对多组独立样本执行单因素方差分析，返回F统计量、p值和Cohen's f效应量
    
    参数
    ----------
    groups : list
        包含多个数组的列表，每个数组为一组独立样本的观测值，至少需要三组
    
    返回
    ----------
    f_stat : float
        单因素方差分析的F统计量
    p_value : float
        F检验对应的p值
    cohens_f : float
        Cohen's f效应量，衡量组间差异的大小
    """
    
    if len(groups) < 3:
        raise ValueError("In function conduct_one_way_anova: one-way ANOVA requires at least 3 groups.")
    
    # 拼接所有数据
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    # 计算组间平方和 (SS_between)
    ss_between = 0
    for group in groups:
        n_group = len(group)
        group_mean = np.mean(group)
        ss_between += n_group * (group_mean - grand_mean) ** 2
    # 计算组内平方和 (SS_within)
    ss_within = 0
    for group in groups:
        ss_within += np.sum((group - np.mean(group)) ** 2)
    
    # 计算 Cohen's f效应量
    if ss_within == 0:
        cohens_f = 0
    else:
        cohens_f = np.sqrt(ss_between / ss_within)
    
    # 计算F检验统计量和p值
    f_stat, p_value = f_oneway(*groups)
    
    return f_stat, p_value, cohens_f


def judge_cohens_f(
    f: float, 
    ignore_threshold: float=0.1, small_threshold: float=0.25, medium_threshold: float=0.4
) -> str:
    """
    判断Cohen's f效应量的业务意义
    
    参数
    ----------
    f : float
        Cohen's f效应量值
    ignore_threshold : float, default=0.1
        可忽略效应的上限阈值，小于此值为可忽略
    small_threshold : float, default=0.25
        小效应的上限阈值，大于等于ignore_threshold且小于此值为小效应
    medium_threshold : float, default=0.4
        中效应的上限阈值，大于等于small_threshold且小于此值为中效应
    
    返回
    ----------
    str
        效应量大小的业务解释：
        - '可忽略'：f < ignore_threshold
        - '小效应'：ignore_threshold ≤ f < small_threshold
        - '中效应'：small_threshold ≤ f < medium_threshold
        - '大效应'：f ≥ medium_threshold
    """
    
    if f < ignore_threshold:
        return '可忽略'
    elif f < small_threshold:
        return '小效应'
    elif f < medium_threshold:
        return '中效应'
    else:
        return '大效应'


def conduct_chi2(groups: list[list[float]]) -> tuple[float, float, float]:
    """
    对多组二分类变量执行卡方独立性检验，返回卡方统计量、p值及相应的效应量
    
    参数
    ----------
    groups : list
        包含多个数组的列表，每个数组为一组样本的二分类观测值（0或1）
    
    返回
    ----------
    chi2 : float
        卡方独立性检验的卡方统计量
    p_value : float
        卡方检验对应的p值
    effect_size : float
        两组比较时返回比值比（Odds Ratio），三组及以上时返回Cramer's V效应量
    """
    
    if len(groups) < 2:
        raise ValueError("In function conduct_chi2: Chi-square test requires at least 2 groups.")
    
    # 构建列联表
    contingency_rows = []
    for g in groups:
        g_arr = np.array(g)
        n_1 = np.sum(g_arr == 1)
        n_0 = np.sum(g_arr == 0)
        contingency_rows.append([n_1, n_0])
    contingency_table = np.array(contingency_rows)
    
    # 执行卡方检验
    chi2, p_value, _1, _2 = chi2_contingency(contingency_table)
    
    if len(groups) == 2:
        a, b = contingency_table[0]
        c, d = contingency_table[1]
        odds_ratio = (a / b) / (c / d)
        
        return chi2, p_value, odds_ratio
    
    else: # 不低于三组的比较odds ratio无意义
        # 计算 Cramer's V
        n = contingency_table.sum().sum()
        r, c = contingency_table.shape
        k = min(r - 1, c - 1)
        
        if k == 0:
            cramers_v = 0
        else:
            cramers_v = np.sqrt(chi2 / (n * k))
        
        return chi2, p_value, cramers_v


def judge_cramers_v(
    v: float, ignore_threshold: float=0.1, small_threshold: float=0.3, medium_threshold: float=0.5
) -> str:
    """
    判断Cramer's V效应量的业务意义
    
    参数
    ----------
    v : float
        Cramer's V效应量值
    ignore_threshold : float, default=0.1
        可忽略效应的上限阈值，小于此值为可忽略
    small_threshold : float, default=0.3
        小效应的上限阈值，大于等于ignore_threshold且小于此值为小效应
    medium_threshold : float, default=0.5
        中效应的上限阈值，大于等于small_threshold且小于此值为中效应
    
    返回
    ----------
    str
        效应量大小的业务解释：
        - '可忽略'：v < ignore_threshold
        - '小效应'：ignore_threshold ≤ v < small_threshold
        - '中效应'：small_threshold ≤ v < medium_threshold
        - '大效应'：v ≥ medium_threshold
    """
    
    if v < ignore_threshold:
        return '可忽略'
    elif v < small_threshold:
        return '小效应'
    elif v < medium_threshold:
        return '中效应'
    else:
        return '大效应'


def conduct_mannwhitney_u(groups: list[list[float]]) -> tuple[float, float, float]:
    """
    对两组独立样本执行Mann-Whitney U检验，返回U统计量、p值和Cliff's delta效应量
    
    参数
    ----------
    groups : list or tuple
        包含两个数组的列表，每个数组为一组独立样本的观测值
    
    返回
    ----------
    u_stat : float
        Mann-Whitney U检验的U统计量
    p_value : float
        双尾检验对应的p值
    cliff_delta : float
        Cliff's delta效应量，衡量两组之间差异的大小，取值范围[-1, 1]
    """
    
    if len(groups) != 2:
        raise ValueError("In function conduct_mannwhitney_u: Mann-Whitney U test requires 2 groups.")
    
    group1, group2 = groups
    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return u_stat, p_value, 0.0
    
    x = np.asarray(group1)
    y = np.asarray(group2)
    y_sorted = np.sort(y)
    
    count_less_y = np.searchsorted(y_sorted, x, side='left')
    count_leq_y = np.searchsorted(y_sorted, x, side='right')
    greater = np.sum(count_less_y)
    less = np.sum(n2 - count_leq_y)
    
    cliff_delta = (greater - less) / (n1 * n2)
    
    return u_stat, p_value, cliff_delta


def judge_cliffs_delta(
    delta: float, ignore_threshold: float=0.147, small_threshold: float=0.33, medium_threshold: float=0.474
) -> str:
    """
    判断Cliff's delta效应量的业务意义
    
    参数
    ----------
    delta : float
        Cliff's delta效应量值（将自动取绝对值进行判断）
    ignore_threshold : float, default=0.147
        可忽略效应的上限阈值，小于此值为可忽略
    small_threshold : float, default=0.33
        小效应的上限阈值，大于等于ignore_threshold且小于此值为小效应
    medium_threshold : float, default=0.474
        中效应的上限阈值，大于等于small_threshold且小于此值为中效应
    
    返回
    ----------
    str
        效应量大小的业务解释：
        - '可忽略'：|delta| < ignore_threshold
        - '小效应'：ignore_threshold ≤ |delta| < small_threshold
        - '中效应'：small_threshold ≤ |delta| < medium_threshold
        - '大效应'：|delta| ≥ medium_threshold
    """
    
    delta = abs(delta)
    if delta < ignore_threshold:
        return '可忽略'
    elif delta < small_threshold:
        return '小效应'
    elif delta < medium_threshold:
        return '中效应'
    else:
        return '大效应'


def conduct_kruskalwallis_h(groups: list[list[float]]) -> tuple[float, float, float]:
    """
    对多组独立样本执行Kruskal-Wallis H检验，返回H统计量、p值和Epsilon2效应量
    
    参数
    ----------
    groups : list
        包含多个数组的列表，每个数组为一组独立样本的观测值，至少需要三组
    
    返回
    ----------
    h_stat : float
        Kruskal-Wallis H检验的H统计量
    p_value : float
        H检验对应的p值
    epsilon2 : float
        Epsilon2效应量，衡量组间差异的大小，取值范围[0, 1]
    """
    
    if len(groups) < 3:
        raise ValueError("In function conduct_kruskalwallis_h: Kruskal-Wallis H test requires at least 3 groups.")
    
    # 执行Kruskal-Wallis H检验  
    h_stat, p_value = kruskal(*groups)  
    
    # 计算Epsilon2效应量  
    n_total = sum(len(g) for g in groups)  
    k_groups = len(groups)
    
    if n_total <= k_groups:  
        epsilon2 = 0  
    else:  
        epsilon2 = (h_stat - k_groups + 1) / (n_total - k_groups)  
        epsilon2 = max(0, epsilon2)  # 防止因H过小导致负值  

    return h_stat, p_value, epsilon2


def judge_epsilon2(
    epsilon2: float, ignore_threshold: float=0.01, low_threshold: float=0.08, mid_threshold: float=0.26
) -> str:
    """
    判断Epsilon2(ε²)效应量的业务意义
    
    参数
    ----------
    epsilon_sq : float
        Epsilon2效应量值
    ignore_threshold : float, default=0.01
        可忽略效应的上限阈值，小于此值为可忽略
    small_threshold : float, default=0.08
        小效应的上限阈值，大于等于ignore_threshold且小于此值为小效应
    medium_threshold : float, default=0.26
        中效应的上限阈值，大于等于small_threshold且小于此值为中效应
    
    返回
    ----------
    str
        效应量大小的业务解释：
        - '可忽略'：epsilon_sq < ignore_threshold
        - '小效应'：ignore_threshold ≤ epsilon_sq < small_threshold
        - '中效应'：small_threshold ≤ epsilon_sq < medium_threshold
        - '大效应'：epsilon_sq ≥ medium_threshold
    """
    
    if epsilon2 < ignore_threshold:
        return '可忽略'
    elif epsilon2 < low_threshold:
        return '小效应'
    elif epsilon2 < mid_threshold:
        return '中效应'
    else:
        return '大效应'


def judge_odds_ratio(
    o_r: float, 
    ignore_threshold: float=1.5, low_threshold: float=2, mid_threshold: float=3, high_threshold: float=5
) -> str:
    """
    判断比值比（Odds Ratio）的业务信号强度
    
    参数
    ----------
    o_r : float
        比值比值（将自动取大于1的方向进行判断）
    ignore_threshold : float, default=1.5
        可忽略效应的上限阈值，小于此值为可忽略
    small_threshold : float, default=2
        小效应的上限阈值，大于等于ignore_threshold且小于此值为小效应
    medium_threshold : float, default=3
        中效应的上限阈值，大于等于small_threshold且小于此值为中效应
    large_threshold : float, default=5
        大效应的上限阈值，大于等于medium_threshold且小于此值为大效应
    
    返回
    ----------
    str
        业务信号强度的解释：
        - '可忽略'：|OR| < ignore_threshold
        - '小效应'：ignore_threshold ≤ |OR| < small_threshold
        - '中效应'：small_threshold ≤ |OR| < medium_threshold
        - '大效应'：medium_threshold ≤ |OR| < large_threshold
        - '极大效应'：|OR| ≥ large_threshold
    """
    
    o_r = max(o_r, 1 / o_r)
    if o_r < ignore_threshold:
        return '可忽略'
    elif o_r < low_threshold:
        return '小效应'
    elif o_r < mid_threshold:
        return '中效应'
    elif o_r < high_threshold:
        return '大效应'
    else:
        return '极大效应'


def conduct_descriptive_stats(
    data: pd.DataFrame, masks: list[bool], group_names: list[str], colnames: list[str]=None
) -> pd.DataFrame:
    """
    计算各组别的描述性指标，返回结果表
    
    参数
    ----------
    data : DataFrame
        原始数据集
    masks : list[bool]
        布尔筛选器列表，每个元素对应一个组别
    group_names : list[str]
        组别名称列表，与masks一一对应
    colnames : list[str], default=None
        需要计算的列名列表，默认包含违约率、年龄、信贷数、房贷数
    
    返回
    ----------
    DataFrame
        包含组别、人数、平均年龄、平均信贷数、平均房贷数、平均违约率的描述性统计表
    """
    
    data = data.copy()
    
    if colnames is None:
        colnames = [target, age, credit, mortgage]
    
    if len(masks) != len(group_names):
        raise ValueError("In function conduct_descriptive_stats: masks and group_names must be the same length.")
    
    desc_data = []
    for name, mask in zip(group_names, masks):
        group = data[mask]
        n = len(group)
        if n == 0:
            # 跳过空组或填 NaN，这里选择跳过
            continue
        
        row = {
            '组别': name,
            '人数': n,
            '平均年龄': round(group[age].mean(), 2),
            '平均信贷数': round(group[credit].mean(), 2),
            '平均房贷数': round(group[mortgage].mean(), 2),
            '平均违约率(%)': round(group[target].mean() * 100, 2)
        }
        desc_data.append(row)
    
    return pd.DataFrame(desc_data)


def conduct_stat_analysis(data: pd.DataFrame, masks: list[bool], colnames: list[str]=None) -> pd.DataFrame:
    """
    执行显著性检验和效应量分析的总函数，返回结果表

    根据组数（2组或3组及以上）和变量类型自动选择合适的检验方法：
    - 违约情况：卡方检验 + Odds Ratio (2组) 或 Cramer's V (3组及以上)
    - 年龄：Welch's t检验 + Cohen's d (2组) 或 单因素方差分析 + Cohen's f (3组及以上)
    - 信贷数/房贷数：Mann-Whitney U检验 + Cliff's delta (2组) 或 Kruskal-Wallis H检验 + Epsilon2 (3组及以上)

    参数
    ----------
    data : DataFrame
        原始数据集
    masks : list
        布尔筛选器列表，每个元素对应一个组别（长度必须为2或3）
    colnames : list, default=None
        需要分析的列名列表，默认包含违约率、年龄、信贷数、房贷数，除非特别要求，否则设为默认值即可

    返回
    ----------
    DataFrame
        包含变量、检验方法、统计量、p值、效应值名称、效应量、效应解释的结果表
    """
    
    data = data.copy()
    
    if colnames is None:
        colnames = [target, age, credit, mortgage]
    # 中文变量名映射
    colname_zh = {
        target: '违约情况',
        age: '年龄',
        credit: '信贷数',
        mortgage: '房贷数'
    }

    if len(masks) not in (2, 3):
        raise ValueError("In function conduct_stat_analysis: masks must have length 2 or 3.")
    
    n_groups = len(masks)
    groups_list = {col: [] for col in colnames}
    for mask in masks:
        for col in colnames:
            group_data = data.loc[mask, col].values
            groups_list[col].append(group_data)
    
    results = []
    for col in colnames:
        groups = groups_list[col]
        if col == target:
            if n_groups == 2:
                chi2, p_val, odds_ratio = conduct_chi2(groups)
                test_name = "卡方检验"
                effect_size = odds_ratio
                effect_name = "Odds Ratio"
                interpretation = judge_odds_ratio(odds_ratio)
                results.append([
                    colname_zh[col], test_name,
                    format_stat(chi2), format_p_value(p_val),
                    effect_name, f"{effect_size:.4f}", interpretation
                ])
            else:
                chi2, p_val, cramers_v = conduct_chi2(groups)
                test_name = "卡方检验"
                effect_size = cramers_v
                effect_name = "Cramer's V"
                interpretation = judge_cramers_v(cramers_v)
                results.append([
                    colname_zh[col], test_name,
                    format_stat(chi2), format_p_value(p_val),
                    effect_name, f"{effect_size:.4f}", interpretation
                ])
        
        elif col == age:
            if n_groups == 2:
                t_stat, p_val, cohens_d = conduct_independent_t(groups)
                test_name = "Welch's t 检验"
                effect_size = cohens_d
                effect_name = "Cohen's d"
                interpretation = judge_cohens_d(cohens_d)
                results.append([
                    colname_zh[col], test_name,
                    format_stat(t_stat), format_p_value(p_val),
                    effect_name, f"{effect_size:.4f}", interpretation
                ])
            else:
                f_stat, p_val, cohens_f = conduct_one_way_anova(groups)
                test_name = "单因素方差分析"
                effect_size = cohens_f
                effect_name = "Cohen's f"
                interpretation = judge_cohens_f(cohens_f)
                results.append([
                    colname_zh[col], test_name,
                    format_stat(f_stat), format_p_value(p_val),
                    effect_name, f"{effect_size:.4f}", interpretation
                ])
                
        elif col in [credit, mortgage]:
            if n_groups == 2:
                u_stat, p_val, cliff_delta = conduct_mannwhitney_u(groups)
                test_name = "Mann-Whitney U 检验"
                effect_size = cliff_delta
                effect_name = "Cliff's delta"
                interpretation = judge_cliffs_delta(cliff_delta)
                results.append([
                    colname_zh[col], test_name,
                    format_stat(u_stat), format_p_value(p_val),
                    effect_name, f"{effect_size:.4f}", interpretation
                ])
            else:
                h_stat, p_val, epsilon2 = conduct_kruskalwallis_h(groups)
                test_name = "Kruskal-Wallis H 检验"
                effect_size = epsilon2
                effect_name = "Epsilon2"
                interpretation = judge_epsilon2(epsilon2)
                results.append([
                    colname_zh[col], test_name,
                    format_stat(h_stat), format_p_value(p_val),
                    effect_name, f"{effect_size:.4f}", interpretation
                ])
        else:
            continue
    
    test_df = pd.DataFrame(
        results,
        columns=['变量', '检验方法', '统计量', 'p 值', '效应值名称', '效应量', '效应解释']
    )
    
    return test_df


def compare_corr(
    data: pd.DataFrame, 
    colnames: list[str]=None, target_colname: str=None, prior_corr: str='Spearman r'
) -> tuple[list[str], pd.DataFrame]:
    """
    计算多个变量与目标变量的三种相关系数（Pearson、Spearman、Kendall），
    并找出两个最强相关的变量（按照Spearman为优先参考）。
    
    参数
    ----------
    data : DataFrame
        原始数据集
    colnames : list, default=None
        需要计算相关系数的列名列表，默认是信贷数、房贷数、家属数
    target_colname : str, default=None
        目标列名称，默认是月收入
    prior_corr : str, default='Pearson r'
        最高优先级的相关系数名称
    
    返回
    ----------
    best_two_colnames : list[str]
        相关性最强的变量名
    corr_compare_df : DataFrame
        包含列名及三种相关系数的统计表：['变量名', 'Pearson r', 'Spearman r', 'Kendall tau']
    """
    
    data = data.copy()
    if colnames is None:
        colnames = [
            age,
            credit, 
            mortgage, 
            dep
        ]
    if target_colname is None:
        target_colname = income
    
    corr_results = []
    for colname in colnames:
        if colname not in data.columns:
            raise ValueError(f"In function compare_corr: column {colname} is not in data.")
        
        # 临时 DataFrame,处理缺失值
        temp_df = data[[colname, target_colname]].copy()
        temp_df = temp_df.dropna()  # 相关系数计算需剔除 NA
        
        x = temp_df[colname].values
        y = temp_df[target_colname].values
        
        # Pearson
        try:
            pearson_corr = pearsonr(x, y)[0]
        except:
            pearson_corr = np.nan
        # Spearman
        try:
            spearman_corr = spearmanr(x, y)[0]
        except:
            spearman_corr = np.nan
        # Kendall
        try:
            kendall_corr = kendalltau(x, y)[0]
        except:
            kendall_corr = np.nan
        
        corr_results.append({
            '变量名': colname,
            'Pearson r': pearson_corr,
            'Spearman r': spearman_corr,
            'Kendall tau': kendall_corr
        })
    
    corr_compare_df = pd.DataFrame(corr_results)
    
    # 排序，按最高优先级的相关系数的绝对值降序
    corr_compare_df = corr_compare_df.assign(prior_corr_abs=corr_compare_df[prior_corr].abs())
    corr_compare_df = corr_compare_df.sort_values(by='prior_corr_abs', ascending=False).drop(columns='prior_corr_abs').reset_index(drop=True)
    
    best_two_colnames = list(corr_compare_df.loc[0: 1, '变量名']) # 取排名前二的指标为最佳指标
    
    return best_two_colnames, corr_compare_df


def analyze_residual(y_true: list[float], y_pred: list[float]) -> tuple[float, float, float, float]:
    """
    计算回归预测的多种误差指标
    
    参数
    ----------
    y_true : list[float]
        真实值序列
    y_pred : list[float]
        预测值序列
    
    返回
    ----------
    mse : float
        均方误差
    mae : float
        平均绝对误差
    mape : float
        平均绝对百分比误差（%）
    smape : float
        对称平均绝对百分比误差（%）
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 避免除零
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # sMAPE 分母用(|真实|+|预测|)/2，避免除零
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator > 0
    smape = 100 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])
    
    return mse, mae, mape, smape


def calculate_vif(X: pd.DataFrame, use_small_size: bool=True, small_size: int=5000) -> pd.DataFrame:
    """
    计算方差膨胀因子（Variance Inflation Factor, VIF），用于检测多重共线性
    
    VIF公式
    ----------
    VIF = 1 / (1 - R²)，其中R²是当前特征对其他特征回归的拟合优度
    
    参数
    ----------
    X : pd.DataFrame
        特征矩阵
    use_small_size : bool, default=True
        是否使用抽样计算（当样本量很大时，可显著提升计算速度）
    small_size : int, default=5000
        若use_small_size=True，随机抽样的样本量
    
    返回
    ----------
    pd.DataFrame
        包含 colname 和 VIF 两列的结果表，按VIF值降序排列
    """

    from statsmodels.stats.outliers_influence import variance_inflation_factor
	
    if use_small_size:
        small_size_idx_list = random.sample(range(len(X)), small_size)
        X = X.iloc[small_size_idx_list]
    
    vif_df = pd.DataFrame()
    vif_df["colname"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    vif_df = vif_df.sort_values(by='VIF', ascending=False, ignore_index=True)
    
    return vif_df
