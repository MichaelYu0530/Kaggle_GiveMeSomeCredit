"""Cleaning functions from the original research pipeline."""

from __future__ import annotations

import pandas as pd
from .schema import age, debt, dep, income, late30, late60, late90, target, util

from .statistics import conduct_descriptive_stats, conduct_stat_analysis



def add_missing_flag(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    添加月收入和家属数的缺失标记，并进行描述性统计和效应量分析
    
    标记说明
    ----------
    single_missing_flag : 1表示仅有月收入缺失
    both_missing_flag : 1表示月收入和家属数均缺失
    
    参数
    ----------
    data : pd.DataFrame
        包含月收入和家属数列的数据集
    
    返回
    ----------
    tuple
        - desc_df : pd.DataFrame
            三组（完整组、单缺组、全缺组）的描述性统计
        - test_df : pd.DataFrame
            三组间差异的显著性检验和效应量分析结果
    
    副作用
    ----------
    直接在原数据框上添加single_missing_flag和both_missing_flag列
    将缺失的月收入赋值为-1，将缺失的家属数赋值为-1
    """
    
    # 创建缺失标记
    data['single_missing_flag'] = \
        ((data[income].isnull()) & (data[dep].notnull())).astype(int)
    data['both_missing_flag'] = \
        ((data[income].isnull()) & (data[dep].isnull())).astype(int)

    # 创建筛选器
    mask_complete = (data['single_missing_flag'] == 0) & (data['both_missing_flag'] == 0) # 完整组
    mask_single_missing = data['single_missing_flag'] == 1 # 单缺组
    mask_both_missing = data['both_missing_flag'] == 1 # 全缺组
    masks = [mask_complete, mask_single_missing, mask_both_missing]
    
    # 描述性统计
    group_names = ['完整组', '单缺组', '全缺组']
    desc_df = conduct_descriptive_stats(data, masks, group_names)
    
    # 对完整组、单缺组、全缺组三组之间，进行差异显著性检验与效应量分析
    test_df = conduct_stat_analysis(data, masks)
    
    # 对缺失值赋值为-1
    data.loc[mask_single_missing | mask_both_missing, income] = -1
    data.loc[mask_both_missing, dep] = -1
    
    return desc_df, test_df


def add_blacklist_flag(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于90天逾期次数的96/98编码添加黑名单标记，并进行描述性统计和效应量分析
    
    标记说明
    ----------
    blacklist_flag : 1表示有特殊编码(96或98)，0表示无
    
    参数
    ----------
    data : pd.DataFrame
        包含90+天逾期次数列的数据集
    
    返回
    ----------
    tuple
        - desc_df : pd.DataFrame
            有编码组和无编码组的描述性统计
        - test_df : pd.DataFrame
            两组间差异的显著性检验和效应量分析结果
    
    副作用
    ----------
    直接在原数据框上添加blacklist_flag列
    将黑名单样本的三个逾期次数列赋值为-1
    """
    
    # 创建特征
    data['blacklist_flag'] = (data[late90].isin([98, 96])).astype(int)
    
    # 创建筛选器
    mask_blacklist = (data['blacklist_flag'] == 1)
    mask_normal = (data['blacklist_flag'] == 0)
    masks = [mask_blacklist, mask_normal]
    
    # 描述性统计
    group_names = ['有编码组', '无编码组']
    desc_df = conduct_descriptive_stats(data, masks, group_names)
    # 对有编码组、无编码组两组之间，进行差异显著性检验与效应量分析
    test_df = conduct_stat_analysis(data, masks)
    
    # 赋值为-1
    colnames = [late30, late60, late90]
    data.loc[mask_blacklist, colnames] = -1
    
    return desc_df, test_df


def add_income_anomaly_flag(data: pd.DataFrame) \
    -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    根据月收入是否为0-10添加异常标记，并进行两组间的统计学分析
    
    标记说明
    ----------
    income_anomaly_flag : 1表示月收入在[0,10]区间内
    
    参数
    ----------
    data : pd.DataFrame
        包含月收入列的数据集
    
    返回
    ----------
    tuple
        - zero_vs_low_desc_df : pd.DataFrame
            月收入0组与月收入1-10组的描述性统计对比
        - zero_vs_low_test_df : pd.DataFrame
            月收入0组与月收入1-10组的显著性检验与效应量分析结果
        - abnormal_vs_normal_desc_df : pd.DataFrame
            月收入异常组[0,10]与正常组(>10且非NA)的描述性统计对比
        - abnormal_vs_normal_test_df : pd.DataFrame
            月收入异常组与正常组的显著性检验与效应量分析结果
    
    副作用
    ----------
    直接在原数据框上添加income_anomaly_flag列
    将月收入异常样本的月收入赋值为-2
    """
    
    # 月收入0组与月收入1-10组的比较，证明两组可合并为月收入异常组
    # 创建筛选器
    mask_zero = data[income] == 0
    mask_one_ten = (data[income] >= 1) & (data[income] <= 10)
    masks = [mask_zero, mask_one_ten]
    
    # 描述性统计
    group_names = ['月收入0组', '月收入1-10组']
    zero_vs_low_desc_df = conduct_descriptive_stats(data, masks, group_names)
    
    # 对月收入0组与月收入1-10组两组之间，进行差异显著性检验与效应量分析
    zero_vs_low_test_df = conduct_stat_analysis(data, masks)
    
    # 月收入极低组与其余月收入组（剔除NA）的比较
    # 创建标签
    data['income_anomaly_flag'] = 0
    data.loc[(data[income] >= 0) & (data[income] <= 10), 'income_anomaly_flag'] = 1
    
    # 创建筛选器
    mask_abnormal = data['income_anomaly_flag'] == 1
    mask_normal = data[income] > 10
    masks = [mask_abnormal, mask_normal]
    
    # 描述性统计
    group_names = ['月收入异常[0, 10]组', '其余月收入（无NA）组']
    abnormal_vs_normal_desc_df = conduct_descriptive_stats(data, masks, group_names)
    # 对月收入异常组与其余月收入组两组之间，进行差异显著性检验与效应量分析
    abnormal_vs_normal_test_df = conduct_stat_analysis(data, masks)
    
    # 赋值为-2
    data.loc[mask_abnormal, income] = -2
    
    return zero_vs_low_desc_df, zero_vs_low_test_df, abnormal_vs_normal_desc_df, abnormal_vs_normal_test_df


def cap_dep_num(data: pd.DataFrame, cap_val: int=10, silently: bool=True) -> None:
    """
    将家属数大于阈值的样本删除
    
    参数
    ----------
    data : pd.DataFrame
        包含家属数列的数据集
    cap_val : int, default=10
        截断阈值，大于此值的样本将被删除
    silently : bool, default=True
        是否静默执行（不打印信息）
    
    返回
    ----------
    None
        无返回值，直接修改原数据框
    """
    
    mask = data[dep] > cap_val
    count = mask.sum()
    
    if count == 0:
        if not silently:
            print(f"没有{dep}大于{cap_val}的样本")
    else:
        data.drop(data[mask].index, inplace=True)
        if not silently:
            print(f"{dep}大于{cap_val}的样本数为{count}，已删除")


def delete_small_age(data: pd.DataFrame, min_age: int=18, silently: bool=True) -> None:
    """
    将年龄小于最小值的样本删除
    
    参数
    ----------
    data : pd.DataFrame
        包含年龄列的数据集
    min_age : int, default=18
        最小年龄阈值，小于此值的样本将被删除
    silently : bool, default=True
        是否静默执行（不打印信息）
    
    返回
    ----------
    None
        无返回值，直接修改原数据框
    """
	
    mask = data[age] < min_age
    count = mask.sum()
    
    
    if count == 0:
        if not silently:
            print(f"没有{age}大于{min_age}的样本")
    else:
        data.drop(data[mask].index, inplace=True)
        if not silently:
            print(f"{age}小于{min_age}的样本数为{count}，已删除")


def add_util_flags(
    data: pd.DataFrame, silently: bool=True,
    high_threshold: float=0.75, overlimit_threshold: float=1, anomaly_threshold: float=5
) -> None:
    """
    为信用额度使用率添加过高、超限、异常三类标记，并对异常值进行特殊赋值
    
    标记说明
    ----------
    is_util_high : [high_threshold, overlimit_threshold) 过高
    is_util_overlimit : [overlimit_threshold, anomaly_threshold) 超限
    util_anomaly_flag : [anomaly_threshold, ∞) 异常
    
    参数
    ----------
    data : pd.DataFrame
        包含信用额度使用率列的数据集
    silently : bool, default=True
        是否静默执行（不打印统计信息）
    high_threshold : float, default=0.75
        使用率过高的判定阈值
    overlimit_threshold : float, default=1
        使用率超限的判定阈值
    anomaly_threshold : float, default=5
        使用率异常的判定阈值
    
    返回
    ----------
    None
        无返回值，直接修改原数据框
    
    副作用
    ----------
    在原数据框上添加is_util_high、is_util_overlimit、util_anomaly_flag列
    将异常使用率赋值为-1
    """
	
    
    if anomaly_threshold <= 1:
        raise ValueError("In function add_util_flags: anomaly_threshold requires a number over 1.")
    
    # 添加异常标记
    values = data[util].values
    data['is_util_high'] = \
        ((values >= high_threshold) & (values < overlimit_threshold)).astype(int)
    data['is_util_overlimit'] = \
        ((values >= overlimit_threshold) & (values < anomaly_threshold)).astype(int)
    data['util_anomaly_flag'] = (values >= anomaly_threshold).astype(int)
    
    # 计算相关数据
    high_mask = data['is_util_high'] == 1
    high_count = high_mask.sum()
    high_ratio = round(high_count / len(high_mask) * 100, 2)
    high_late_rate = round(data.loc[high_mask, target].mean() * 100, 2)
    overlimit_mask = data['is_util_overlimit'] == 1
    overlimit_count = overlimit_mask.sum()
    overlimit_ratio = round(overlimit_count / len(overlimit_mask) * 100, 2)
    overlimit_late_rate = round(data.loc[overlimit_mask, target].mean() * 100, 2)
    anomaly_mask = data['util_anomaly_flag'] == 1
    anomaly_count = anomaly_mask.sum()
    anomaly_ratio = round(anomaly_count / len(anomaly_mask) * 100, 2)
    anomaly_late_rate = round(data.loc[anomaly_mask, target].mean() * 100, 2)
    
    data.loc[data['util_anomaly_flag'] == 1, util] = -1 # 异常值赋值为-1
    
    if not silently:
        print(f"{util}处于{high_threshold}-{overlimit_threshold}的样本数有{high_count}，占比{high_ratio}%，违约率{high_late_rate}%")
        print(f"{util}处于{overlimit_threshold}-{anomaly_threshold}的样本数有{overlimit_count}，占比{overlimit_ratio}%，违约率{overlimit_late_rate}%")
        print(f"{util}处于{anomaly_threshold}以上的样本数有{anomaly_count}，占比{anomaly_ratio}%，违约率{anomaly_late_rate}%")


def add_debt_flags(
    data: pd.DataFrame, silently: bool=True, 
    high_threshold: float=0.75, overlimit_threshold: float=1, anomaly_threshold: float=5
) -> None:
    """
    为月收入可靠且DebtRatio处于阈值范围内的样本添加标记，并对异常值进行特殊赋值
    
    标记说明
    ----------
    is_debt_high : [high_threshold, overlimit_threshold) 过高
    is_debt_overlimit : [overlimit_threshold, anomaly_threshold) 超限
    debt_anomaly_flag : [anomaly_threshold, ∞) 异常
    
    参数
    ----------
    data : pd.DataFrame
        包含月收入和负债率列的数据集
    silently : bool, default=True
        是否静默执行（不打印统计信息）
    high_threshold : float, default=0.75
        负债率过高的判定阈值
    overlimit_threshold : float, default=1
        负债率超限的判定阈值
    anomaly_threshold : float, default=5
        负债率异常的判定阈值
    
    返回
    ----------
    None
        无返回值，直接修改原数据框
    
    副作用
    ----------
    在原数据框上添加is_debt_high、is_debt_overlimit、debt_anomaly_flag列
    异常负债率赋值为-3，月收入缺失/异常对应的负债率赋值为-1/-2
    """
	
    if anomaly_threshold <= 1:
        raise ValueError("In function add_debt_flags: anomaly_threshold requires a number over 1.")
    
    # 月收入可靠样本的筛选器
    reliable_mask = data[income] > 10
    # reliable_mask = (data[income] > 10) & (data[income].notna())
    
    # 添加异常标记
    values = data[debt].values
    data['is_debt_high'] = \
        ((values >= high_threshold) & (values < overlimit_threshold) & reliable_mask).astype(int)
    data['is_debt_overlimit'] = \
        ((values >= overlimit_threshold) & (values < anomaly_threshold) & reliable_mask).astype(int)
    data['debt_anomaly_flag'] = ((values >= anomaly_threshold) & reliable_mask).astype(int)
    
    # 计算相关数据
    reliable_data = data[reliable_mask]
    reliable_count = reliable_mask.sum()
    reliable_ratio = round(reliable_count / len(data) * 100, 2)
    high_mask = reliable_data['is_debt_high'] == 1
    high_count = high_mask.sum()
    high_ratio = round(high_count / reliable_count * 100, 2)
    high_late_rate = round(reliable_data.loc[high_mask, target].mean() * 100, 2)
    overlimit_mask = reliable_data['is_debt_overlimit'] == 1
    overlimit_count = overlimit_mask.sum()
    overlimit_ratio = round(overlimit_count / reliable_count * 100, 2)
    overlimit_late_rate = round(reliable_data.loc[overlimit_mask, target].mean() * 100, 2)
    anomaly_mask = reliable_data['debt_anomaly_flag'] == 1
    anomaly_count = anomaly_mask.sum()
    anomaly_ratio = round(anomaly_count / reliable_count * 100, 2)
    anomaly_late_rate = round(reliable_data.loc[anomaly_mask, target].mean() * 100, 2)
    
    data.loc[data['debt_anomaly_flag'] == 1, debt] = -3 # 负债率异常则赋值为-3
    data.loc[data[income] == -1, debt] = -1 # 因月收入缺失导致负债率不可靠者，赋值为-1
    data.loc[data[income] == -2, debt] = -2 # 因月收入异常导致负债率不可靠者，赋值为-2
    
    if not silently:
        print(f'月收入可靠的样本数为{reliable_count}，占比{reliable_ratio}%')
        print(f"月收入可靠的样本中，{debt}处于{high_threshold}-{overlimit_threshold}的样本数有{high_count}，占比{high_ratio}%，违约率{high_late_rate}%")
        print(f"月收入可靠的样本中，{debt}处于{overlimit_threshold}-{anomaly_threshold}的样本数有{overlimit_count}，占比{overlimit_ratio}%，违约率{overlimit_late_rate}%")
        print(f"月收入可靠的样本中，{debt}处于{anomaly_threshold}以上有{anomaly_count}，占比{anomaly_ratio}%，违约率{anomaly_late_rate}%")
