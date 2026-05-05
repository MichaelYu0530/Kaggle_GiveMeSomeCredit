from analysis import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os

""" 全局变量 """
# region colnames_str_variables
target = 'SeriousDlqin2yrs'
age = 'Age'
credit = 'NumberOfOpenCreditLinesAndLoans'
mortgage = 'NumberRealEstateLoansOrLines'
dep = 'NumberOfDependents'
income = 'MonthlyIncome'
debt = 'DebtRatio'
util = 'RevolvingUtilizationOfUnsecuredLines'
late30 = 'NumberOfTime30-59DaysPastDueNotWorse'
late60 = 'NumberOfTime60-89DaysPastDueNotWorse'
late90 = 'NumberOfTimes90DaysLate'
# endregion

colnames_abbr_map = {
        'Age': 'age',
        'NumberOfOpenCreditLinesAndLoans': 'credit',
        'NumberRealEstateLoansOrLines': 'mortgage',
        'NumberOfDependents': 'dep',
        'MonthlyIncome': 'income',
        'DebtRatio': 'debt',
        'RevolvingUtilizationOfUnsecuredLines': 'util',
        'NumberOfTime30-59DaysPastDueNotWorse': '30-59late',
        'NumberOfTime60-89DaysPastDueNotWorse': '60-89late',
        'NumberOfTimes90DaysLate': '90+late'
    }

colnames_cn_map= {
    # 原始变量
    'Age': '年龄',
    'DebtRatio': '负债率',
    'MonthlyIncome': '月收入',
    'NumberOfDependents': '家属数',
    'NumberOfOpenCreditLinesAndLoans': '信贷数',
    'NumberOfTime30-59DaysPastDueNotWorse': '30-59天逾期次数',
    'NumberOfTime60-89DaysPastDueNotWorse': '60-89天逾期次数',
    'NumberOfTimes90DaysLate': '90+天逾期次数',
    'NumberRealEstateLoansOrLines': '房贷数',
    'RevolvingUtilizationOfUnsecuredLines': '信用额度使用率',
    # 衍生变量
    'short_late': '短期逾期次数',
    'late_severity_score': '逾期严重程度评分',
    'mortgage_ratio': '房贷占比',
    'credit_late_density': '逾期密度',
    'income_per_dep': '人均月收入',
    'monthly_debt': '月债务',
    'free_cashflow_income': '自由现金流收入',
    'credit_pressure_index' : '信用压力指数'
}

""" 绘图函数 """
# 展示基础可视化图，地位是主绘图函数，构建整体布局
def show_basic_visualization(data: pd.DataFrame, is_saving: bool=True, save_folder: str='figures')\
    -> Figure:
    """
    展示原始数据集的基础可视化图，构建整体布局
    
    参数
    ----------
    data : pd.DataFrame
        原始数据集
    is_saving : bool, default=True
        是否保存文件
    save_folder : str, default='figures'
        保存的文件夹名
    
    返回
    ----------
    Figure
        matplotlib图形对象，包含所有子图
    
    说明
    ----------
    该函数为顶层绘图函数，内部调用各指标的专用绘图函数，
    生成3行4列共12个子图，删除最后一个空白子图后保存为PNG文件
    """
	
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建画布
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), gridspec_kw={'hspace': 0.4})  
    fig.suptitle('原始数据集简单可视化图', fontsize=16)
    
    # 列名到绘图函数的映射
    plot_functions = {
        'SeriousDlqin2yrs': plot_target_summary,
        'NumberOfTime30-59DaysPastDueNotWorse': plot_late30_summary,
        'NumberOfTime60-89DaysPastDueNotWorse': plot_late60_summary,
        'NumberOfTimes90DaysLate': plot_late90_summary,
        'MonthlyIncome': plot_income_summary,
        'NumberOfDependents': plot_dep_summary,
        'RevolvingUtilizationOfUnsecuredLines': plot_util_summary,
        'DebtRatio': plot_debt_summary,
        'Age': plot_age_summary,
        'NumberOfOpenCreditLinesAndLoans': plot_credit_summary,
        'NumberRealEstateLoansOrLines': plot_mortgage_summary,
    }
    
    # 对各指标（列名）作图
    for i, plot_func in enumerate(plot_functions.values()):
        ax = axes.flat[i]
        plot_func(data, ax)
    fig.delaxes(axes.flat[11]) # 删除多余子图
    
    plt.tight_layout()
    # 保存文件
    if is_saving:
        os.makedirs(save_folder, exist_ok=True)
        filename = f'{save_folder}/origin_data_basic_visualization.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

# 目标变量子绘图函数
def plot_target_summary(data: pd.DataFrame, ax: plt.Axes) -> None:
    counts = data[target].value_counts().sort_index()
    total = len(data)
    normal_count = counts[0]
    default_count = counts[1]
    normal_pct = normal_count / total
    default_pct = default_count / total
    
    bars = ax.bar(['未违约', '违约'], [normal_count, default_count], color=['steelblue', 'crimson'], alpha = 0.7)
    ax.set_ylim(0, 150000)
    ax.set_title('SeriousDlqin2yrs\n两年内是否90天以上违约，目标变量', fontweight='bold')
    
    height_normal = bars[0].get_height()
    ax.text(bars[0].get_x() + bars[0].get_width()/2, height_normal * 0.9,
            f'{normal_count:,}人\n({normal_pct:.1%})',
            ha='center', va='center', fontsize=9, fontweight='bold')
    height_default = bars[1].get_height()
    ax.text(bars[1].get_x() + bars[1].get_width()/2, height_default + 500,
            f'{default_count:,}人\n({default_pct:.1%})',
            ha='center', va='bottom', fontsize=9)

# 年龄子绘图函数
def plot_age_summary(data: pd.DataFrame, ax: plt.Axes) -> None:
    bins = [0, 21, 31, 41, 51, 61, 71, 81, float('inf')]
    labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']
    data_copy = data[age].copy()
    data_cut = pd.cut(data_copy, bins=bins, labels=labels, right=False)
    freq = data_cut.value_counts().sort_index()
    
    bars = ax.bar(range(len(freq)), freq.values, tick_label=labels, color='steelblue', alpha=0.7)
    ax.tick_params(axis='x', labelsize=8) 
    ax.set_ylim(0, 40000)
    ax.set_title('age\n年龄', fontweight='bold')
    
    for bar, count in zip(bars, freq.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                f'{count:,}',
                ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

# 信贷数子绘图函数
def plot_credit_summary(data: pd.DataFrame, ax: plt.Axes) -> None:
    bins = [0, 1, 3, 6, 11, 21, float('inf')]
    labels = ['0', '1-2', '3-5', '6-10', '11-20', '20+']
    data_copy = data[credit].copy()
    data_cut = pd.cut(data_copy, bins=bins, labels=labels, right=False, include_lowest=True)
    freq = data_cut.value_counts().sort_index()
    
    bars = ax.bar(range(len(freq)), freq.values, tick_label=labels, color='steelblue', alpha=0.7)
    ax.set_yscale('log')
    ax.set_yticks([10**3, 10**4, 10**5])
    ax.set_yticklabels(['$10^3$', '$10^4$', '$10^5$'])
    ax.set_title('NumberOfOpenCreditLinesAndLoans\n信贷数量', fontweight='bold')
    
    for bar, count in zip(bars, freq.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                f'{count:,}',
                ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

# 房贷数子绘图函数
def plot_mortgage_summary(data: pd.DataFrame, ax: plt.Axes) -> None:
    bins = [0, 1, 2, 4, 6, 11, float('inf')]
    labels = ['0', '1', '2-3', '4-5', '6-10', '10+']
    data_copy = data[mortgage].copy()
    data_cut = pd.cut(data_copy, bins=bins, labels=labels, right=False, include_lowest=True)
    freq = data_cut.value_counts().sort_index()
    
    bars = ax.bar(range(len(freq)), freq.values, tick_label=labels, color='steelblue', alpha=0.7)
    ax.set_yscale('log')
    ax.set_yticks([10**3, 10**4, 10**5])
    ax.set_yticklabels(['$10^3$', '$10^4$', '$10^5$'])
    ax.set_title('NumberRealEstateLoansOrLines\n房贷数量', fontweight='bold')
    
    for bar, count in zip(bars, freq.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                f'{count:,}',
                ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

# 家属数子绘图函数
def plot_dep_summary(data: pd.DataFrame, ax: plt.Axes) -> None:
    bins = [0, 1, 2, 3, 6, float('inf')]
    labels = ['0', '1', '2', '3-5', '5+']
    data_copy = data[dep].copy()
    data_cut = pd.cut(data_copy, bins=bins, labels=labels, right=False, include_lowest=True)
    data_copy_na = data_copy.isna()
    data_cut = data_cut.cat.add_categories(['NA'])
    data_cut[data_copy_na] = 'NA'
    freq = data_cut.value_counts().sort_index()
    
    bars = ax.bar(range(len(freq)), freq.values, tick_label=freq.index, color='steelblue', alpha=0.7)
    ax.set_yscale('log')
    ax.set_yticks([10**3, 10**4, 10**5])
    ax.set_yticklabels(['$10^3$', '$10^4$', '$10^5$'])
    ax.set_title('NumberOfDependents\n家属数量', fontweight='bold')
    
    for bar, count in zip(bars, freq.values):
        height = bar.get_height()
        if count > 75000:
            ax.text(bar.get_x() + bar.get_width()/2, height * 0.95,
                    f'{count:,}',
                    ha='center', va='top', fontsize=8, color='black', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                    f'{count:,}',
                    ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

# 月收入子绘图函数
def plot_income_summary(data: pd.DataFrame, ax: plt.Axes) -> None:
    bins = [0, 1, 2, 100, 1000, 10000, float('inf')]
    labels = ['0', '1', '2-100', '100-1k', '1k-10k', '10k+']
    data_copy = data[income].copy()
    data_copy_na = data_copy.isna()
    data_cut = pd.cut(data_copy, bins=bins, labels=labels, right=False, include_lowest=True)
    
    data_cut = data_cut.cat.add_categories(['NA'])
    data_cut[data_copy_na] = 'NA'
    freq = data_cut.value_counts().sort_index()
    
    bars = ax.bar(range(len(freq)), freq.values, tick_label=freq.index, color='steelblue', alpha=0.7)
    ax.tick_params(axis='x', labelsize=8) 
    ax.set_yscale('log')
    ax.set_yticks([10**3, 10**4, 10**5])
    ax.set_yticklabels(['$10^3$', '$10^4$', '$10^5$'])
    ax.set_title('MonthlyIncome\n月收入', fontweight='bold')
    
    for bar, count in zip(bars, freq.values):
        height = bar.get_height()
        if count > 75000:
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.95,
                    f'{count:,}',
                    ha='center', va='top', fontsize=8, color='black', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                    f'{count:,}',
                    ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

# 负债率子绘图函数
def plot_debt_summary(data: pd.DataFrame, ax: plt.Axes) -> None:
    bins = [0, 0.5, 1, 2, 5, 10, 100, float('inf')]
    labels = ['0-0.5', '0.5-1', '1-2', '2-5', '5-10', '10-100', '100+']
    data_copy = data[debt].copy()
    data_cut = pd.cut(data_copy, bins=bins, labels=labels, right=False, include_lowest=True)
    freq = data_cut.value_counts().sort_index()
    
    bars = ax.bar(range(len(freq)), freq.values, tick_label=labels, color='steelblue', alpha=0.7)
    ax.tick_params(axis='x', labelsize=8) 
    ax.set_yscale('log')
    ax.set_yticks([10**3, 10**4, 10**5])
    ax.set_yticklabels(['$10^3$', '$10^4$', '$10^5$'])
    ax.set_title('DebtRatio\n负债率', fontweight='bold')
    
    for bar, count in zip(bars, freq.values):
        height = bar.get_height()
        if count > 75000:
            ax.text(bar.get_x() + bar.get_width()/2, height * 0.95,
                    f'{count:,}',
                    ha='center', va='top', fontsize=8, color='black', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                    f'{count:,}',
                    ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

# 信用额度使用率子绘图函数
def plot_util_summary(data: pd.DataFrame, ax: plt.Axes) -> None:
    bins = [0, 0.5, 1, 2, 5, 10, 100, float('inf')]
    labels = ['0-0.5', '0.5-1', '1-2', '2-5', '5-10', '10-100', '100+']
    data_copy = data[util].copy()
    data_cut = pd.cut(data_copy, bins=bins, labels=labels, right=False, include_lowest=True)
    freq = data_cut.value_counts().sort_index()
    
    bars = ax.bar(range(len(freq)), freq.values, tick_label=labels, color='steelblue', alpha=0.7)
    ax.tick_params(axis='x', labelsize=8) 
    ax.set_yscale('log')
    ax.set_yticks([10**3, 10**4, 10**5])
    ax.set_yticklabels(['$10^3$', '$10^4$', '$10^5$'])
    ax.set_title(f'{util}\n信用额度使用率', fontweight='bold')
    
    for bar, count in zip(bars, freq.values):
        height = bar.get_height()
        if count > 75000:
            ax.text(bar.get_x() + bar.get_width()/2, height * 0.95,
                    f'{count:,}',
                    ha='center', va='top', fontsize=8, color='black', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                    f'{count:,}',
                    ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

# 30-59天逾期次数（近两年）子绘图函数
def plot_late30_summary(data: pd.DataFrame, ax: plt.Axes) -> None:
    bins = [-1, 0, 1, 2, 3, 6, float('inf')]
    labels = ['96/98', '0', '1', '2', '3-5', '5+']
    data_copy = data[late30].copy()
    data_copy.loc[data_copy.isin([96, 98])] = -1
    data_cut = pd.cut(data_copy, bins=bins, labels=labels, right=False, include_lowest=True)
    freq = data_cut.value_counts().sort_index()
    
    bars = ax.bar(range(len(freq)), freq.values, tick_label=labels, color='steelblue', alpha=0.7)
    ax.set_yscale('log')
    ax.set_yticks([10**3, 10**4, 10**5])
    ax.set_yticklabels(['$10^3$', '$10^4$', '$10^5$'])
    ax.set_title('NumberOfTime30-59DaysPastDueNotWorse\n30-59天逾期次数（近两年）', fontweight='bold')
    
    for bar, count in zip(bars, freq.values):
        height = bar.get_height()
        if count > 75000:
            ax.text(bar.get_x() + bar.get_width()/2, height * 0.95,
                    f'{count:,}',
                    ha='center', va='top', fontsize=8, color='black', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                    f'{count:,}',
                    ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

# 60-89天逾期次数（近两年）子绘图函数
def plot_late60_summary(data: pd.DataFrame, ax: plt.Axes) -> None:
    bins = [-1, 0, 1, 2, 3, 6, float('inf')]
    labels = ['96/98', '0', '1', '2', '3-5', '5+']
    data_copy = data[late60].copy()
    data_copy.loc[data_copy.isin([96, 98])] = -1
    data_cut = pd.cut(data_copy, bins=bins, labels=labels, right=False, include_lowest=True)
    freq = data_cut.value_counts().sort_index()
    
    bars = ax.bar(range(len(freq)), freq.values, tick_label=labels, color='steelblue', alpha=0.7)
    ax.set_yscale('log')
    ax.set_yticks([10**3, 10**4, 10**5])
    ax.set_yticklabels(['$10^3$', '$10^4$', '$10^5$'])
    ax.set_title('NumberOfTime60-89DaysPastDueNotWorse\n60-89天逾期次数（近两年）', fontweight='bold')
    
    for bar, count in zip(bars, freq.values):
        height = bar.get_height()
        if count > 75000:
            ax.text(bar.get_x() + bar.get_width()/2, height * 0.95,
                    f'{count:,}',
                    ha='center', va='top', fontsize=8, color='black', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                    f'{count:,}',
                    ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

# 90天以上逾期次数（历史）子绘图函数
def plot_late90_summary(data: pd.DataFrame, ax: plt.Axes) -> None:
    bins = [-1, 0, 1, 2, 3, 6, float('inf')]
    labels = ['96/98', '0', '1', '2', '3-5', '5+']
    data_copy = data[late90].copy()
    data_copy.loc[data_copy.isin([96, 98])] = -1
    data_cut = pd.cut(data_copy, bins=bins, labels=labels, right=False, include_lowest=True)
    freq = data_cut.value_counts().sort_index()
    
    bars = ax.bar(range(len(freq)), freq.values, tick_label=labels, color='steelblue', alpha=0.7)
    ax.set_yscale('log')
    ax.set_yticks([10**3, 10**4, 10**5])
    ax.set_yticklabels(['$10^3$', '$10^4$', '$10^5$'])
    ax.set_title(f'{late90}\n90天以上逾期次数（历史）', fontweight='bold')
    
    for bar, count in zip(bars, freq.values):
        height = bar.get_height()
        if count > 75000:
            ax.text(bar.get_x() + bar.get_width()/2, height * 0.95,
                    f'{count:,}',
                    ha='center', va='top', fontsize=8, color='black', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                    f'{count:,}',
                    ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

# 画出二维分箱的热力图
def plot_heatmap(
    data: pd.DataFrame, colname_pair: list[str],  is_saving: bool=True, save_folder: str='figures'
) -> Figure:
    """
    画出二维分箱的热力图，展示不同分箱组合下的违约率分布
    
    处理流程
    ----------
    1. 根据二维分箱列提取行标签和列标签（退化的分箱用特殊区间表示）
    2. 计算按行/列分类的违约率透视表
    3. 处理退化的分箱：将一维分箱的值填充到对应的行或列
    4. 对区间排序，使热力图美观
    5. 格式化区间显示，绘制热力图
    
    参数
    ----------
    data : pd.DataFrame
        包含目标变量和二维分箱列的数据集
    colname_pair : list[str]
        两个指标列名组成的列表
    is_saving : bool, default=True
        是否保存文件
    save_folder : str, default='figures'
        保存的文件夹名
    
    返回
    ----------
    Figure
        热力图的matplotlib图形对象
    """
	
    if len(colname_pair) != 2:
        raise ValueError("In function draw_heatmap: colname_pair must be at length 2.")
    
    colname1, colname2 = colname_pair
    binbase1 = colnames_abbr_map.get(colname1, colname1)
    binbase2 = colnames_abbr_map.get(colname2, colname2)
    binname_2D = f'{binbase1}_x_{binbase2}_bin'
    
    selected_data = data[[target, binname_2D]].copy()
    
    # 分两列存储二维分箱列中的信息，提前用不可靠区间标记“退化的”一维区间
    unreliable_interval = pd.Interval(-10, -9, 'left')
    lambda_key_row = lambda x: x[0] if x[0] is not None else unreliable_interval
    lambda_key_col = lambda x: x[1] if x[1] is not None else unreliable_interval
    selected_data['row_label'] = selected_data[binname_2D].apply(lambda_key_row)
    selected_data['col_label'] = selected_data[binname_2D].apply(lambda_key_col)
    
    # 计算按两指标的一维分箱分类的违约率表
    default_rate_table = pd.pivot_table(
        selected_data, 
        index='row_label', 
        columns='col_label', 
        values=target, 
        aggfunc='mean', 
        fill_value=np.nan
    )
    
    # 剔除不可靠区间，正确赋值“退化的”一维区间
    if unreliable_interval in list(default_rate_table.columns):
        unreliable_col = default_rate_table[unreliable_interval]
        values_row = unreliable_col.dropna()
        bins_stay_1D_row = values_row.index.tolist()
        
        default_rate_table = default_rate_table.drop(columns=unreliable_interval)
        for value, bin in zip(values_row, bins_stay_1D_row):
            default_rate_table.loc[bin] = value
    
    if unreliable_interval in list(default_rate_table.index):
        unreliable_row = default_rate_table.loc[unreliable_interval]
        values_col = unreliable_row.dropna()
        bins_stay_1D_col = values_col.index.tolist()
        
        default_rate_table = default_rate_table.drop(index=unreliable_interval)
        for value, bin in zip(values_col, bins_stay_1D_col):
            default_rate_table[bin] = value
    
    # 改为百分数形式
    default_rate_table = default_rate_table.applymap(lambda x: round(x * 100, 2))
    
    # 重新对区间排序，使后续热力图更美观
    from pipeline import sort_bins
    sorted_rows = sort_bins(default_rate_table.index.tolist())
    sorted_cols = sort_bins(default_rate_table.columns.tolist())
    default_rate_table = default_rate_table.reindex(index=sorted_rows, columns=sorted_cols)
    
    # 让区间显示得更美观
    formatted_cols = []
    for col_interval in default_rate_table.columns:
        formatted_cols.append(format_interval(col_interval))
    formatted_rows = []
    for row_interval in default_rate_table.index:
        formatted_rows.append(format_interval(row_interval))
    default_rate_table.columns = formatted_cols
    default_rate_table.index = formatted_rows
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    # 绘制热力图
    sns.heatmap(
        default_rate_table,
        annot=True,
        fmt=".3f",
        cmap='RdYlGn_r',
        linewidths=.5,
        linecolor='gray',
        cbar_kws={'label': f'违约率(%)'},
        ax=ax
    )
    # 大标题和坐标轴标题
    ax.set_title(f'二维分箱热力图: {colnames_cn_map[colname1]}X{colnames_cn_map[colname2]}\n(红色=高违约率)')
    ax.set_xlabel(colnames_cn_map[colname2])
    ax.set_ylabel(colnames_cn_map[colname1])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    # 保存图片
    if is_saving:
        os.makedirs(save_folder, exist_ok=True)
        filename = f'{save_folder}/{binbase1}_x_{binbase2}_heatmap.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

# 横向比较各模型的拟合效果，可视化各模型评价指标Recall、Precision等
from sklearn.linear_model import LogisticRegression; import xgboost as xgb
def visualize_fit_goodness(
    y_test: pd.Series,
    X_test_raw_lr: pd.DataFrame, X_test_woe_lr: pd.DataFrame, X_test_xgb: pd.DataFrame,
    raw_lr_model: LogisticRegression, sc_model, xgb_model: xgb.XGBClassifier,
    is_saving: bool=True, save_folder: str='figures'
) -> tuple[Figure, Figure, Figure]:
    # 计算测试集上的预测概率
    y_prob_test_raw_lr = raw_lr_model.predict_proba(X_test_raw_lr)[: , 1] # raw LR
    y_prob_test_woe_lr = sc_model.predict_proba(X_test_woe_lr)[: , 1] # WOE LR
    y_prob_test_xgb = xgb_model.predict_proba(X_test_xgb)[: , 1] # XGBoost
    
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    model_y_prob_map = {
        'Raw_LR': y_prob_test_raw_lr,
        'ScoreCard': y_prob_test_woe_lr,
        'XGBoost': y_prob_test_xgb
    }
    fig_list = []
    for model_name, y_prob_test in model_y_prob_map.items():
        # 创建画布
        fig, axes = plt.subplots(2, 3, figsize=(16, 12), gridspec_kw={'hspace': 0.4})  
        fig.suptitle(f'{model_name} Fit Goodness Visualization', fontsize=16)
        
        # 列名到绘图函数的映射
        plot_functions = {
            'P-R Curve': plot_pr_curve,
            'F2-Score Curve': plot_f2_curve,
            'Gain Curve': plot_gain_curve,
            'Lift Curve': plot_lift_curve,
            'KS Chart': plot_ks_chart,
        }
        
        # 对各指标（列名）作图
        for i, plot_func in enumerate(plot_functions.values()):
            ax = axes.flat[i]
            plot_func(y_test, y_prob_test, model_name, ax)
        if model_name == 'Raw_LR':
            fig.delaxes(axes[1, 2])
        if model_name == 'ScoreCard':
            ax = axes.flat[i + 1]
            plot_score_bins_bar(sc_model, X_test_woe_lr, y_test, ax)
        if model_name == 'XGBoost':
            ax = axes.flat[i + 1]
            plot_shap_bar(X_test_xgb, xgb_model, ax)
        
        fig_list.append(fig)
        
        plt.tight_layout()
        # 保存文件
        if is_saving:
            os.makedirs(save_folder, exist_ok=True)
            filename = f'{save_folder}/{model_name}_fit_goodness_visualization.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig_list

# Precision-Recall曲线绘图子函数
def plot_pr_curve(y_true: pd.Series, y_prob: pd.Series, model_name: str, ax: plt.Axes) -> None:
    # 构造阈值序列
    threshold_list = np.linspace(0, 1, 200)
    
    recall_list = []
    precision_list = []
    f2_score_list = []
    for threshold in threshold_list:
        # 计算Recall、Precision、F2-Score
        recall, precision = calculate_recall_precision(y_true, y_prob, threshold)
        recall_list.append(recall)
        precision_list.append(precision)
        if recall + precision == 0:
            f2_score = 0
        else:
            beta = 2
            f2_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        f2_score_list.append(f2_score)
    
    # 转换为array形式
    recall_arr = np.array(recall_list)
    precision_arr = np.array(precision_list)
    f2_score_arr = np.array(f2_score_list)
    
    # 根据Recall排序
    sorted_idx = np.argsort(recall_arr)
    recall_arr = recall_arr[sorted_idx]
    precision_arr = precision_arr[sorted_idx]
    f2_score_arr = f2_score_arr[sorted_idx]
    threshold_list = threshold_list[sorted_idx]
    
    # 找出F2-Score最大值点
    best_idx = np.argmax(f2_score_arr)
    best_recall = recall_arr[best_idx]
    best_precision = precision_arr[best_idx]
    
    # 绘图
    ax.plot(recall_arr, precision_arr)
    # 注明F2-Score最大值点
    ax.scatter(
        recall_arr[best_idx],
        precision_arr[best_idx],
        marker='o',
        label='Best F2-Score Point'
    )
    # 由标注点画两条到坐标轴的垂线
    ax.plot([best_recall, best_recall], [0, best_precision], linestyle='--', alpha=0.6, color='red')
    ax.plot([0, best_recall], [best_precision, best_precision], linestyle='--', alpha=0.6, color='green')
    # 在标注点附近添加文本框标注
    ax.annotate(
        f"Precision = {best_precision:.3f}\nRecall = {best_recall:.3f}",
        xy=(best_recall, best_precision),
        xytext=(best_recall - 0.32, best_precision - 0.08),
        fontsize=9,
        bbox=dict(
            boxstyle="round, pad=0.3",
            fc="white",
            ec="gray",
            alpha=0.9
        )
    )
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'PR Curve ({model_name})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(True)

# F2-Score-阈值曲线绘图子函数
def plot_f2_curve(y_true: pd.Series, y_prob: pd.Series, model_name: str, ax: plt.Axes) -> None:
    # 构造阈值序列
    thresholds = np.linspace(0, 1, 200)
    
    beta = 2
    f2_score_list = []
    for threshold in thresholds:
        # 计算Recall、Precision、F2-Score
        recall, precision = calculate_recall_precision(y_true, y_prob, threshold)
        if recall + precision == 0:
            f2_score = 0
        else:
            f2_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        f2_score_list.append(f2_score)
    
    # 找出F2-Score最大值点
    f2_score_arr = np.array(f2_score_list)
    best_index = np.argmax(f2_score_arr)
    best_threshold = thresholds[best_index]
    max_f2_score = f2_score_arr[best_index]
    
    # 绘图
    ax.plot(thresholds, f2_score_arr)
    # 注明F2-Score最大值点
    ax.scatter(best_threshold, f2_score_arr[best_index], label=f'Best F2-Score Point')
    # 由标注点画两条到坐标轴的垂线
    ax.plot([best_threshold, best_threshold], [0, max_f2_score], linestyle='--', alpha=0.6, color='red')
    ax.plot([0, best_threshold], [max_f2_score, max_f2_score], linestyle='--', alpha=0.6, color='green')
    # 在标注点附近添加文本框标注
    ax.annotate(
        f"F2-Score = {max_f2_score:.3f}\nThreshold = {best_threshold:.3f}",
        xy=(best_threshold, max_f2_score),
        xytext=(best_threshold + 0.03, max_f2_score + 0.02),
        fontsize=9,
        bbox=dict(
            boxstyle="round, pad=0.3",
            fc="white",
            ec="gray",
            alpha=0.9
        )
    )
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    ax.set_title(f'F{beta}-Score-Threshold Curve ({model_name})')
    ax.set_xlabel('Threshold')
    ax.set_ylabel(f'F2-Score')
    ax.legend()
    ax.grid(True)

# Gain-累计样本量占比曲线绘图子函数
def plot_gain_curve(y_true: pd.Series, y_prob: pd.Series, model_name: str, ax: plt.Axes) -> None:
    # 直接得出Gain等评价指标的数值表
    gain_df = calculate_gain_lift(y_true, y_prob)[['population', 'gain', 'ks']]
    # 找出KS点及对应的Gain、Population
    ks_idx = gain_df['ks'].idxmax()
    ks_value = gain_df.loc[ks_idx, "ks"]
    best_gain = gain_df.loc[ks_idx, 'gain']
    best_population = gain_df.loc[ks_idx, 'population']
    
    # 绘图
    ax.plot(gain_df['population'], gain_df['gain'], label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', label='Random')
    # 标记KS点
    ax.scatter(best_population, best_gain, label=f'KS={ks_value:.3f}')
    # 由标注点画两条到坐标轴的垂线
    ax.plot([best_population, best_population], [0, best_gain], linestyle='--', alpha=0.6, color='red')
    ax.plot([0, best_population], [best_gain, best_gain], linestyle='--', alpha=0.6, color='green')
    # 在标注点附近添加文本框标注
    ax.annotate(
        f"Gain = {best_gain:.3f}\nPopulation = {best_population:.3f}",
        xy=(best_population, best_gain),
        xytext=(best_population + 0.03, best_gain - 0.08),
        fontsize=9,
        bbox=dict(
            boxstyle="round, pad=0.3",
            fc="white",
            ec="gray",
            alpha=0.9
        )
    )
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Gain Curve ({model_name})')
    ax.set_xlabel('Population')
    ax.set_ylabel('Gain')
    ax.legend()
    ax.grid(True)

# Lift-累计样本量占比曲线绘图子函数
def plot_lift_curve(y_true: pd.Series, y_prob: pd.Series, model_name: str, ax: plt.Axes) -> None:
    # 直接求出Lift
    lift_df = calculate_gain_lift(y_true, y_prob)[['population', 'lift']]
    
    # 绘图
    ax.plot(lift_df['population'], lift_df['lift'])
    ax.axhline(1, linestyle='--', color='crimson', alpha=0.7)  # 基准线
    
    # 标注的百分位点
    highlight_points = [0.10, 0.25]
    for pt in highlight_points:
        # 找到最接近的实际人口比例的索引
        idx = (lift_df['population'] - pt).abs().idxmin()
        population_value = lift_df.loc[idx, 'population']
        lift_value = lift_df.loc[idx, 'lift']
        
        # 标记前10%与前25%点
        ax.scatter(population_value, lift_value, s=60, color='red', zorder=5)
        # 添加文本框标注
        ax.annotate(
            f"Population={pt:.0%}\nLift={lift_value:.2f}",
            xy=(population_value, lift_value),
            xytext=(population_value + 0.04, lift_value + 0.06),
            fontsize=9,
            bbox=dict(boxstyle="round, pad=0.3", fc="white", ec="gray", alpha=0.9)
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, lift_df['lift'].max() + 1)
    ax.set_title(f'Lift Curve ({model_name})')
    ax.set_xlabel('Population')
    ax.set_ylabel('Lift')
    ax.grid(True)

# KS图绘图子函数
def plot_ks_chart(y_true: pd.Series, y_prob: pd.Series, model_name: str, ax: plt.Axes) -> None:
    # 建立真实值与预测概率的对照表
    df = pd.DataFrame({'y': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    
    # 计算KS及好/坏样本的cdf
    total_bad = df['y'].sum()
    total_good = len(df) - total_bad
    df['bad_cdf'] = df['y'].cumsum() / total_bad
    df['good_cdf'] = (1 - df['y']).cumsum() / total_good
    df['ks'] = df['bad_cdf'] - df['good_cdf']
    ks_idx = df['ks'].idxmax()
    best_bad_cdf = df.loc[ks_idx, 'bad_cdf']
    best_good_cdf = df.loc[ks_idx, 'good_cdf']
    best_population = ks_idx / len(df)
    
    # 绘图
    ax.plot(df.index / len(df), df['bad_cdf'], label='Bad CDF')
    ax.plot(df.index / len(df), df['good_cdf'], label='Good CDF')
    # 画出取到KS值的水平线
    ax.vlines(
        x=best_population, ymin=best_good_cdf, ymax=best_bad_cdf,
        linestyles='--', label=f'KS={df.loc[ks_idx,"ks"]:.3f}'
    )
    # 标记取到KS值的好坏CDF曲线上的点
    ax.scatter(best_population, best_good_cdf, s=60, color='red', zorder=5)
    ax.scatter(best_population, best_bad_cdf, s=60, color='red', zorder=5)
    # 添加文本框标注
    ax.annotate(
        f"Population={best_population:.2%}\nGood CDF={best_good_cdf:.2f}",
        xy=(best_population, best_good_cdf),
        xytext=(best_population + 0.03, best_good_cdf - 0.08),
        fontsize=9,
        bbox=dict(boxstyle="round, pad=0.3", fc="white", ec="gray", alpha=0.9)
    )
    ax.annotate(
        f"Population={best_population:.2%}\nBad CDF={best_bad_cdf:.2f}",
        xy=(best_population, best_bad_cdf),
        xytext=(best_population + 0.03, best_bad_cdf - 0.08),
        fontsize=9,
        bbox=dict(boxstyle="round, pad=0.3", fc="white", ec="gray", alpha=0.9)
    )
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'KS Curve ({model_name})')
    ax.set_xlabel('Population')
    ax.set_ylabel('CDF')
    ax.legend()
    ax.grid(True)

# 评分卡各分箱违约率柱状图绘图子函数
def plot_score_bins_bar(sc_model, X: pd.DataFrame, y: pd.Series, ax: plt.Axes, n_bins: int=10) -> None:
    # 直接得出各分箱违约率对照表
    score_bins_df = sc_model.show_score_bins(X, y, n_bins=n_bins)
    bin_labels = score_bins_df['score_bin']
    bin_labels_short = [f"{bin.left:.1f}-{bin.right:.1f}" for bin in bin_labels]
    default_rate_series = score_bins_df['bad_rate'] * 100
    
    # 绘图
    bars = ax.bar(bin_labels_short, default_rate_series)
    
    ax.set_ylim(0, 40)
    ax.set_xticklabels(bin_labels_short, rotation=45, ha='right')
    for bar, default_rate in zip(bars, default_rate_series):
        ax.text(
            x=bar.get_x() + bar.get_width() / 2,
            y=default_rate,
            s=f'{default_rate:.2f}%',
            ha='center',
            va='bottom',
            fontsize=9
        )
    ax.set_title(f'Default Rate Among Score Bins')
    ax.set_xlabel('Score bins')
    ax.set_ylabel('Default Rate')
    
    ax.grid(True)

# SHAP柱状图绘图子函数
def plot_shap_bar(X_test: pd.DataFrame, xgb_model: xgb.XGBClassifier, ax: plt.Axes, top_n: int=10) -> None:
    # 增设部分衍生指标的缩写
    colnames_abbr_map_here =  {
        age: 'Age',
        util: 'Util',
        debt: 'DR',
        credit: 'Credit',
        late90: 'late90',
        'late_severity_score': 'LSS',
        'short_late': 'SL',
        'free_cashflow_income': 'FCI',
        'credit_late_density': 'CLD',
        'credit_pressure_index': 'CPI',
        'mortgage_ratio': 'MR'
    }
    
    # 直接得出指标与SHAP importance的对照表
    shap_df = calculate_shap_importance(xgb_model, X_test)
    shap_df = shap_df.sort_values('shap_importance', ascending=False).head(top_n)
    feature_series = shap_df['feature'].iloc[::-1]
    feature_abbr_series = feature_series.apply(lambda x: colnames_abbr_map_here[x])
    shap_importance_series = shap_df['shap_importance'].iloc[::-1]
    
    # 绘图
    bars = ax.barh(feature_abbr_series, shap_importance_series)
    # 在每个横柱右侧展示数值
    for bar, shap_importance in zip(bars, shap_importance_series):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{shap_importance:.3f}",
            va='center',
            ha='left',
            fontsize=9
        )
    
    ax.set_xlim(0, 0.7)
    ax.set_title(f'SHAP Importance (XGBoost)')
    ax.set_xlabel('Mean |SHAP|')
    ax.set_ylabel('Features')
    ax.grid(True)

# 可视化模型的预测效果，绘制散点图、误差分布图、核密度估计图
def show_pred_effect_visualization(
    y_true: np.ndarray, target_name: str, name_to_y_pred: dict, save_folder: str='figures'
) -> None:
    """
    可视化模型的预测效果，绘制散点图、误差分布图、核密度估计图[已弃用]
    
    绘制内容
    ----------
    1. 真实值vs预测值散点图（每个模型/方法一个子图）
    2. 残差分布直方图（每个模型/方法一个子图）
    3. 真实值与预测值的核密度估计对比图
    
    参数
    ----------
    y_true : array-like
        目标变量的真实值
    target_name : str
        目标变量名称，用于图表标签
    name_to_y_pred : dict
        字典，键为模型/方法名称，值为对应的预测值数组
    save_folder : str, default='figures'
        保存的文件夹名
    
    返回
    ----------
    None
        直接保存图形文件，无返回值
    """
    
    # 确保存储图片的文件夹存在
    os.makedirs(save_folder, exist_ok=True)
    
    # 绘制真实vs预测散点图
    plt.figure(figsize=(12, 5))
    for i, (name, y_pred) in enumerate(name_to_y_pred.items()):
        plt.subplot(1, 2, i + 1)
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.plot([0, y_true.max()], [0, y_true.max()], 'r--')
        
        plt.title(name)
        plt.xlabel(f"{target_name}真实值")
        plt.ylabel(f"{name}预测值")
        
        true_max_lim, pred_max_lim = np.quantile(y_true, 0.99), np.quantile(y_pred, 0.995) # 避免异常大值干扰
        max_lim =max(true_max_lim, pred_max_lim)
        plt.xlim(0, max_lim) 
        plt.ylim(0, max_lim) # x, y轴范围应一致，保证直线y=x不变形
    
    plt.tight_layout()
    filename1 = f'{save_folder}/2Dbin_vs_RF_scatter.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制残差分布图
    plt.figure(figsize=(12, 5))
    for i, (name, y_pred) in enumerate(name_to_y_pred.items()):
        plt.subplot(1, 2, i + 1)
        
        residual = y_true - y_pred
        q1, q99 = np.quantile(residual, 0.01), np.quantile(residual, 0.99)
        mask = (residual >= q1) & (residual <= q99) # 除去异常大值的筛选器
        
        sns.histplot(residual[mask], bins=50, kde=True)
        plt.title(f"{name}误差分布")
        plt.xlim(q1, q99)
    
    plt.tight_layout()
    filename2 = f'{save_folder}/2Dbin_vs_RF_residual.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制核密度估计图
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_true, label='真实值', linewidth=2)
    for name, y_pred in name_to_y_pred.items():
        sns.kdeplot(y_pred, label=name)
    
    plt.legend()
    plt.title(f"{target_name}分布对比")
    plt.xlabel(target_name)
    plt.ylabel("分布密度")
    plt.xlim(0, np.quantile(y_true, 0.99))
    
    plt.tight_layout()
    filename3 = f'{save_folder}/2Dbin_vs_RF_KDE.png'
    plt.savefig(filename3, dpi=300, bbox_inches='tight')
    plt.close()

""" 建表函数 """
# 展示特别发现
def show_special_discovery(data: pd.DataFrame) -> pd.DataFrame:
    """
    展示数据中的特别发现，包括缺失值关系、特殊编码、字段间包含关系
    
    发现内容
    ----------
    1. 缺失值检查：判断是否只有月收入和家属数存在缺失值
    2. 缺失值关系：验证家属数缺失时月收入是否必然缺失
    3. 特殊编码：验证三个逾期列中96/98编码是否对应同一批人
    4. 字段包含关系：验证房贷数是否始终不超过信贷数
    
    参数
    ----------
    data : pd.DataFrame
        原始数据集
    
    返回
    ----------
    pd.DataFrame
        包含相关列、重要发现、由此推出的结论三列的结果表
    """
	
    results = []
    
    # 发现1：缺失值
    # 检查哪些列有缺失值
    missing_cols = data.columns[data.isna().any()].tolist()
    if not set(missing_cols) - set([income, dep]):
        results.append({
            '相关列': '月收入与家属数',
            '重要发现': f'只有这两列存在缺失值，其他列无缺失',
            '由此推出的结论': '后续只需处理这两列的缺失值'
        })
    else:
        results.append({
            '相关列名': '月收入与家属数',
            '重要发现': f'除了这两列外仍有其他列存在缺失值',
            '由此推出的结论': '无'
        })
    
    # 验证家庭成员缺失的人，月收入也一定缺失
    dependents_missing = data[data[dep].isna()]
    income_when_dependents_missing = dependents_missing[income].isna().all()
    results.append({
        '相关列': '家属数→月收入',
        '重要发现': f'家庭成员缺失共{len(dependents_missing)}人，其中月收入也缺失{len(dependents_missing)}人（{income_when_dependents_missing}）',
        '由此推出的结论': '家庭成员缺失的人，月收入必然缺失'
    })
    
    # 反向验证
    income_missing = data[data[income].isna()]
    dependents_when_income_missing = income_missing[dep].isna().all()
    results.append({
        '相关列': '月收入→家属数',
        '重要发现': f'月收入缺失共{len(income_missing)}人，其中家庭成员也缺失{income_missing["NumberOfDependents"].isna().sum()}人',
        '由此推出的结论': '月收入缺失时，家庭成员不一定缺失（存在仅月收入缺失的样本）'
    })
    
    # 发现2：特殊编码
    late_cols = [late30, late60, late90]
    # 分别找出各列有特殊编码的人
    special_30 = data[data[late_cols[0]].isin([96, 98])].index
    special_60 = data[data[late_cols[1]].isin([96, 98])].index
    special_90 = data[data[late_cols[2]].isin([96, 98])].index
    
    # 验证三者是否完全相同
    all_same = (set(special_30) == set(special_60) == set(special_90))
    if all_same:
        total_special = len(special_30)
        results.append({
            '相关列': '三个逾期次数指标',
            '重要发现': f'96/98编码完全对应同一批人，共{total_special}人',
            '由此推出的结论': '96/98不具有业务意义，是银行赋予的特殊标记'
        })
    else:
        results.append({
            '相关列': '三个逾期次数指标',
            '重要发现': f'96/98编码不完全对应',
            '由此推出的结论': '无'
        })
    
    # 发现3：房贷数在信贷数之内
    is_included = (data[credit] >= data[mortgage]).all()
    if is_included:
        results.append({
            '相关列': '房贷数与信贷数',
            '重要发现': f'同一个样本的信贷数不会低于房贷数',
            '由此推出的结论': '房贷是信贷的其中一种'
        })
    else:
        results.append({
            '相关列': '房贷数与信贷数',
            '重要发现': f'存在样本的信贷数低于房贷数',
            '由此推出的结论': '无'
        })
    
    return pd.DataFrame(results)

# 展示一个分箱列的统计数据
def summary_bins(data: pd.DataFrame, colname: str) -> pd.DataFrame:
    """
    展示一个分箱列的统计数据
    
    参数
    ----------
    data : pd.DataFrame
        包含分箱列和目标变量的数据集
    colname : str
        分箱列名或原始指标列名（会自动转换为分箱列名）
    
    返回
    ----------
    pd.DataFrame
        包含区间、样本数、占比、未违约、违约、违约率六列的分箱统计表
    """
	
    if colname.endswith('_bin'):
        binname = colname
    else:
        base = colnames_abbr_map.get(colname, colname)
        binname = f'{base}_bin'
    
    # 分组统计
    bin_df = data.groupby(binname, observed=True).\
        agg(样本数=(target, 'count'), 违约=(target, 'sum')).reset_index()
    
    # 计算衍生指标
    bin_df['占比'] = bin_df['样本数'] / len(data)
    bin_df['未违约'] = bin_df['样本数'] - bin_df['违约']
    bin_df['违约率'] = bin_df['违约'] / bin_df['样本数']
    
    bin_df.rename(columns={binname: '区间'}, inplace=True) # 更改列名
    order = ['区间', '样本数', '占比', '未违约', '违约', '违约率']
    bin_df = bin_df[order]
    
    return bin_df

# 按类别列出完成统一数据清洗后当前数据集中的全部指标名称
def display_current_colnames_general(origin_data: pd.DataFrame, current_data: pd.DataFrame) -> pd.DataFrame:
    """
    按类别列出完成统一数据清洗后当前数据集中的全部指标名称
    
    分类规则
    ----------
    1. 原始指标：在原始数据集中存在且在当前数据集中保留的列
    2. 数据质量标记：以 _flag 结尾的列
    3. 用户行为标记：以 is_ 或 has_ 开头的列
    4. 交互标记：以 _signal 结尾的列
    5. 衍生指标：剩余的列
    
    参数
    ----------
    origin_data : pd.DataFrame
        原始数据集，用于识别保留下来的原始变量
    current_data : pd.DataFrame
        当前处理后的数据集，包含所有待分类的列
    
    返回
    ----------
    pd.DataFrame
        包含分好类的指标名称的表格，每列内是同类型的指标名称列表
        列顺序：原始指标 | 数据质量标记 | 衍生指标 | 用户行为标记 | 交互标记
    """
    
    current_columns = set(current_data.columns)
    origin_columns = set(origin_data.columns)
    remaining_columns = current_columns
    
    # 数据质量标记 (以'_flag'结尾)
    flag_list = [col for col in remaining_columns if col.endswith('_flag')]
    remaining_columns -= set(flag_list)
    # 原始指标（包括目标变量）
    original_list = list(origin_columns.intersection(current_columns))
    remaining_columns -= set(original_list)
    # 用户行为标记（以'is_'/'has_'开头）
    behavior_list = [col for col in remaining_columns if col.startswith('is_') or col.startswith('has_')]
    remaining_columns -= set(behavior_list)
    # 交互标记（以'_signal'结尾）
    intersection_list = [col for col in remaining_columns if col.endswith('_signal')]
    remaining_columns -= set(intersection_list)
    # 指标的分箱或者分箱排名序号（内含'_bin'）
    bins_list = [col for col in remaining_columns if '_bin' in col]
    remaining_columns -= set(bins_list)
    # 衍生指标 (剩余的所有列)
    derived_list = list(remaining_columns)
    
    # 为了美观，做一次排序
    if target in original_list:
        original_list.remove(target)
        original_list = [f'{target}（目标变量）'] + sorted(original_list)
    else:
        original_list.sort()
    flag_list.sort()
    derived_list.sort()
    behavior_list.sort()
    intersection_list.sort()
    
    # 建表
    colname_general_df = pd.DataFrame({
        '原始指标': pd.Series(original_list),
        '数据质量标记': pd.Series(flag_list),
        '衍生指标': pd.Series(derived_list),
        '用户行为标记': pd.Series(behavior_list),
        '交互标记': pd.Series(intersection_list)
    })
    colname_general_df = colname_general_df.dropna(axis=1, how='all')
    colname_general_df = colname_general_df.fillna('')
    
    return colname_general_df

# 展示raw LR特化表达构建完毕时与统一数据清洗后有变化的指标名称
def display_current_colnames_raw_lr(
    origin_data: pd.DataFrame, general_data: pd.DataFrame, raw_lr_data: pd.DataFrame
) -> pd.DataFrame:
    """
    展示raw LR特化表达构建完毕时与统一数据清洗后有变化的指标名称
    
    对比内容
    ----------
    1. raw LR中新增指标：在raw LR数据集中存在但在通用数据集中不存在的列
    2. raw LR中删去的原始指标：在通用数据集中存在但在raw LR数据集中不存在的原始指标
    3. raw LR中删去的衍生指标：在通用数据集中存在但在raw LR数据集中不存在的衍生指标
    
    参数
    ----------
    origin_data : pd.DataFrame
        原始数据集，用于识别原始指标
    general_data : pd.DataFrame
        统一数据清洗后的数据集
    raw_lr_data : pd.DataFrame
        raw LR特化表达构建完毕后的数据集
    
    返回
    ----------
    pd.DataFrame
        包含三列的结果表：raw LR中新增指标、raw LR中删去原始指标、raw LR中删去衍生指标
    """
	
    origin_colnames, general_colnames, current_colnames = \
        set(origin_data.columns), set(general_data.columns), set(raw_lr_data.columns)
    added_colnames = current_colnames - general_colnames
    removed_colnames = general_colnames - current_colnames
    removed_origin_colnames = {colname for colname in removed_colnames if colname in origin_colnames}
    removed_derived_colnames = removed_colnames - removed_origin_colnames
    
    colname_raw_lr_df = pd.DataFrame({
        'raw LR中新增指标': pd.Series(list(added_colnames)),
        'raw LR中删去原始指标': pd.Series(list(removed_origin_colnames)),
        'raw LR中删去衍生指标': pd.Series(list(removed_derived_colnames))
    })
    colname_raw_lr_df = colname_raw_lr_df.dropna(axis=1, how='all')
    colname_raw_lr_df = colname_raw_lr_df.fillna('')
    
    return colname_raw_lr_df

# 展示woe LR特化表达构建完毕时与统一数据清洗后有变化的指标名称
def display_current_colnames_woe_lr(
    general_data: pd.DataFrame, woe_lr_data: pd.DataFrame, save_folder: str=None
) -> pd.DataFrame:
    """
    展示woe LR特化表达构建完毕时与统一数据清洗后有变化的指标名称
    
    对比内容
    ----------
    1. WOE LR中未构建WOE的标记：在通用数据集中存在但未转换为WOE列的标记变量
    2. WOE LR中未构建WOE的非标记指标：在通用数据集中存在但未转换为WOE列的连续变量
    3. WOE LR中构建了WOE的二维分箱：已转换为WOE列的二维分箱
    
    参数
    ----------
    general_data : pd.DataFrame
        统一数据清洗后的数据集
    woe_lr_data : pd.DataFrame
        WOE LR特化表达构建完毕后的数据集（除目标变量外均为WOE列）
    
    返回
    ----------
    pd.DataFrame
        包含三列的结果表：WOE LR中未构建WOE的标记、WOE LR中未构建WOE的非标记指标、WOE LR中构建了WOE的二维分箱
    
    异常
    ----------
    ValueError
        如果woe_lr_data中除目标变量外存在不以_woe结尾的列
    """
	
    woe_lr_colnames = woe_lr_data.columns.tolist()
    woe_lr_colnames.remove(target)
    if not pd.Series(woe_lr_colnames).str.endswith('_woe').all(): # 检查除了目标变量外其余列是否全部为WOE列
        raise ValueError("In function display_current_colnames_woe_lr: woe_lr_data hss not-woe columns.")
    
    woe_lr_colnames = pd.Series(woe_lr_colnames).str.removesuffix('_woe') # 还原为构建WOE列前的列名
    woe_lr_colnames_binary = [colname for colname in woe_lr_colnames \
        if colname.endswith(('_flag', '_signal')) or colname.startswith(('has_', 'is_'))]
    woe_lr_colnames_not_binary = [colname.removesuffix('_bin') for colname in woe_lr_colnames \
        if colname not in woe_lr_colnames_binary and '_x_' not in colname]
    woe_lr_binnames_2D = [colname for colname in woe_lr_colnames \
        if colname not in woe_lr_colnames_binary and '_x_' in colname]
    
    from pipeline import get_colnames_whether_binary
    general_colnames_binary = get_colnames_whether_binary(general_data, is_binary=True)
    general_colnames_not_binary = get_colnames_whether_binary(general_data, is_binary=False)
    
    colnames_binary_without_woe = \
        [colname for colname in general_colnames_binary if colname not in woe_lr_colnames_binary]
    if colnames_binary_without_woe == []:
        colnames_binary_without_woe = ['全部标记均构建了WOE']
    colnames_not_binary_without_woe = [colname for colname in general_colnames_not_binary \
        if colnames_abbr_map.get(colname, colname) not in woe_lr_colnames_not_binary]
    
    colname_woe_lr_df = pd.DataFrame({
        'WOE LR中未构建WOE的标记': pd.Series(colnames_binary_without_woe),
        'WOE LR中未构建WOE的非标记指标': pd.Series(colnames_not_binary_without_woe),
        'WOE LR中构建了WOE的二维分箱': pd.Series(woe_lr_binnames_2D)
    })
    colname_woe_lr_df = colname_woe_lr_df.dropna(axis=1, how='all')
    colname_woe_lr_df = colname_woe_lr_df.fillna('')
    
    return colname_woe_lr_df

















