"""Transforms functions from the original research pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from .schema import age, colnames_abbr_map, debt, util



def fit_centering_means(data: pd.DataFrame, fixed_colnames: list[str]=None) -> dict[str, float]:
    """
    在训练数据上拟合raw LR中心化所需的均值。

    参数
    ----------
    data : pd.DataFrame
        训练数据
    fixed_colnames : list[str], default=None
        需要中心化的列名列表，为None时默认对年龄、负债率、使用率做中心化

    返回
    ----------
    dict[str, float]
        每个待中心化指标对应的训练集均值
    """

    if fixed_colnames is None:
        colnames = [age, debt, util]
    else:
        colnames = fixed_colnames

    centering_means = {}
    for colname in colnames:
        if colname == age:
            mask_reliable = data[colname] >= 18
        elif colname == debt:
            mask_reliable = (data['single_missing_flag'] == 0) & (data['both_missing_flag'] == 0) & \
                (data['debt_anomaly_flag'] == 0)
        elif colname == util:
            mask_reliable = data['util_anomaly_flag'] == 0
        else:
            raise ValueError(f"In function fit_centering_means: unsupported column {colname}.")

        centering_means[colname] = data.loc[mask_reliable, colname].mean()

    return centering_means


def apply_centered_features(data: pd.DataFrame, centering_means: dict[str, float]) -> list[str]:
    """
    将训练集拟合好的中心化均值应用到任意数据集。
    
    中心化公式
    ----------
    col_centered = col - mean_train(col)  （训练集可靠样本的均值）
    不可靠样本的中心化列赋值为0
    
    参数
    ----------
    data : pd.DataFrame
        待处理数据集
    centering_means : dict[str, float]
        训练集拟合得到的中心化均值
    
    返回
    ----------
    list[str]
        实际参与中心化的列名列表
    """

    colnames = list(centering_means.keys())

    for colname in colnames:
        binbase = colnames_abbr_map.get(colname, colname)
        colname_centered = f'{binbase}_centered'
        
        if colname == age:
            mask_reliable = data[colname] >= 18
        elif colname == debt:
            mask_reliable = (data['single_missing_flag'] == 0) & (data['both_missing_flag'] == 0) & \
                (data['debt_anomaly_flag'] == 0)
        elif colname == util:
            mask_reliable = data['util_anomaly_flag'] == 0
        else:
            raise ValueError(f"In function apply_centered_features: unsupported column {colname}.")
        reliable_mean = centering_means[colname]
        
        # 验证集/测试集只复用训练集均值，不重新fit中心化统计量
        data[colname_centered] = (data[colname] - reliable_mean).where(mask_reliable, 0)
        if colname != age:
            colname_multiplied = f'age_centered_*_{binbase}_centered'
            data[colname_multiplied] = data['age_centered'] * data[colname_centered]
    
    return colnames


def add_centered_features(data: pd.DataFrame, fixed_colnames: list[str]=None) -> list[str]:
    """
    兼容旧调用的中心化入口：在当前数据上拟合并应用中心化。
    仅适合训练数据；验证集/测试集请使用 fit_centering_means + apply_centered_features。
    """

    centering_means = fit_centering_means(data, fixed_colnames=fixed_colnames)
    return apply_centered_features(data, centering_means)


def fit_log_feature_colnames(data: pd.DataFrame, fixed_colnames: list[str]=None) -> list[str]:
    """
    在训练数据上确定需要取对数的列。
    """

    if fixed_colnames is None:
        colnames = data.columns.tolist()
    else:
        colnames = fixed_colnames

    colnames_to_log = []
    for colname in colnames:
        if pd.api.types.is_numeric_dtype(data[colname]) and data[colname].max() > 200:
            colnames_to_log.append(colname)

    return colnames_to_log


def add_log_features(data: pd.DataFrame, fixed_colnames: list[str]=None) -> list[str]:
    """
    对有长尾分布的数值指标取对数（log1p）
    
    参数
    ----------
    data : pd.DataFrame
        包含待处理指标的数据集
    fixed_colnames : list[str], default=None
        需要取对数的列名列表，为None时自动筛选最大值>200的数值列
    
    返回
    ----------
    list[str]
        实际添加了对数变换列的列名列表
    
    副作用
    ----------
    直接在原数据框上添加 {base}_log 列
    """
	
    if fixed_colnames is None:
        colnames_to_log = fit_log_feature_colnames(data)
    else:
        colnames_to_log = fixed_colnames

    for colname in colnames_to_log:
        if not pd.api.types.is_numeric_dtype(data[colname]):
            raise ValueError(f"In function add_log_features: {colname} must be numeric.")

        binbase = colnames_abbr_map.get(colname, colname)
        colname_logged = f'{binbase}_log' # 对数变换指标的列名
        # 验证集/测试集无条件复用训练集已确定的log特征列表
        data[colname_logged] = np.log1p(data[colname])

    return colnames_to_log


def select_colnames_raw_lr(current_data: pd.DataFrame) -> pd.DataFrame:
    """
    筛选用于raw LR的指标，如果某个原始指标存在变换版本（_centered或_log），则排除原始指标，保留变换版本
    
    参数
    ----------
    current_data : pd.DataFrame
        包含原始指标和变换指标的数据集
    
    返回
    ----------
    list[str]
        筛选后用于训练的列名列表
    """
	
    to_remove = set()
    current_colnames_set = set(current_data.columns)
    
    for colname in current_colnames_set:
        # 构建变换名集合
        binbase = colnames_abbr_map.get(colname, colname)
        transformed_colnames = {f'{binbase}_centered', f'{binbase}_log'}
        
        if transformed_colnames & current_colnames_set:
            to_remove.add(colname)
    
    train_colnames = list(current_colnames_set - to_remove)
    
    return train_colnames
