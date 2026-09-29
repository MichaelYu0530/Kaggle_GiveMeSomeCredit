"""Plotting eda functions from the original research pipeline."""

from __future__ import annotations

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from .binning import sort_bins

from .schema import age, colnames_abbr_map, colnames_cn_map, credit, debt, dep, income, late30, late60, late90, mortgage, target, util

from .statistics import format_interval



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
