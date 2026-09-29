"""Woe functions from the original research pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from .schema import colnames_abbr_map, target



def _validate_woe_source_column(colname: str) -> None:
    if not colname.endswith(('_flag', '_signal', '_bin')) and not colname.startswith(('has_', 'is_')):
        raise ValueError(f"In function add_woe_column: {colname} is neither a bin column nor a flag column.")


def fit_woe_mapping(train_data: pd.DataFrame, colname: str) -> dict[pd.Interval | float, float]:
    """
    在训练数据上拟合单列WOE映射。
    
    参数
    ----------
    train_data : pd.DataFrame
        训练数据，必须包含目标变量与待编码列
    colname : str
        分箱列名或标记列名
    
    返回
    ----------
    dict
        训练集拟合得到的WOE映射
    """

    _validate_woe_source_column(colname)
    woe_map, _ = calculate_woe_iv(train_data, colname)
    return woe_map


def apply_woe_mapping(
    data: pd.DataFrame, colname: str, woe_map: dict[pd.Interval | float, float], default_woe: float=0.0
) -> None:
    """
    将既有WOE映射应用到任意数据集，不读取目标变量。
    
    参数
    ----------
    data : pd.DataFrame
        待编码数据集
    colname : str
        分箱列名或标记列名
    woe_map : dict
        由训练集拟合得到的WOE映射
    default_woe : float, default=0.0
        未知分箱或缺失映射时采用的默认WOE值
    """

    _validate_woe_source_column(colname)

    woe_colname = f'{colname}_woe'
    source_values = data[colname].astype('object')
    woe_values = source_values.map(woe_map).fillna(default_woe).astype('float64')
    data[woe_colname] = woe_values


def add_woe_column(data: pd.DataFrame, colname: str) -> None:
    """
    添加分箱列对应的WOE列。
    
    注意
    ----------
    本函数会基于传入数据集当前标签现算WOE，只适合训练数据或旧实验代码。
    对于验证集 / 测试集，请改用 fit_woe_mapping + apply_woe_mapping，
    避免使用验证/测试标签重新计算WOE造成数据泄漏。
    """

    woe_map = fit_woe_mapping(data, colname)
    apply_woe_mapping(data, colname, woe_map)


def calculate_woe_iv(data: pd.DataFrame, colname: str) -> tuple[dict[pd.Interval | float], float]:
    """
    计算单变量的证据权重（WOE）和信息价值（IV）
    
    参数
    ----------
    data : pd.DataFrame
        包含目标变量和待计算变量的数据集
    colname : str
        待计算WOE/IV的列名，可以是分箱列名（以_bin结尾）或原始变量名
    
    返回
    ----------
    woe_map : dict
        从分箱区间到WOE值的映射字典
    iv_value : float
        该变量的信息价值（IV）值
    """
    
    if colname.endswith('_bin'): # 输入的是分箱列名
        binname = colname
    elif set(data.loc[data[colname] >= 0, colname]).issubset({0, 1}): # 输入的是二元指标（标记）列名
        binname = colname
    else: # 输入的是非二元指标列名
        base = colnames_abbr_map.get(colname, colname)
        binname = f'{base}_bin'
    
    # 基础统计
    grouped = data.groupby(binname)[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']
    
    epsilon = 1e-6 # 平滑项，防止除零报错，同时惩罚全好或全坏的分箱
    grouped['bad_safe'] = grouped['bad'].apply(lambda x: epsilon if x == 0 else x)
    grouped['good_safe'] = grouped['good'].apply(lambda x: epsilon if x == 0 else x)
    total_bad = grouped['bad_safe'].sum()
    total_good = grouped['good_safe'].sum()
    
    # 计算 WOE
    grouped['good_dist'] = grouped['good_safe'] / total_good
    grouped['bad_dist'] = grouped['bad_safe'] / total_bad
    grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
    woe_map = grouped['woe'].to_dict() # 构建映射
    
    # 计算 IV
    grouped['iv_component'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']
    iv = grouped['iv_component'].sum()
    
    return woe_map, iv


def calculate_psi(train_data: pd.DataFrame, valid_data: pd.DataFrame, colname: str) -> float:
    """
    计算群体稳定性指数（Population Stability Index, PSI）
    
    PSI公式
    ----------
    PSI = Σ (train_ratio - valid_ratio) * ln(train_ratio / valid_ratio)
    
    参数
    ----------
    train_data : pd.DataFrame
        训练集数据
    valid_data : pd.DataFrame
        验证集数据
    colname : str
        列名，可以是分箱列、二元标记列或原始指标列
        函数会自动识别并转换为对应的分箱列名
    
    返回
    ----------
    float
        计算得到的PSI值
    """
	
    if colname.endswith('_bin'): # 输入的是分箱列名
        binname = colname
    elif set(train_data.loc[train_data[colname] >= 0, colname]).issubset({0, 1}): # 输入的是二元指标（标记）列名
        binname = colname
    else: # 输入的是非二元指标列名
        base = colnames_abbr_map.get(colname, colname)
        binname = f'{base}_bin'
    
    n_train, n_valid = train_data.shape[0], valid_data.shape[0]
    train_counts = train_data[binname].value_counts()
    bins = train_counts.index.tolist()
    n_bins = len(bins)
    valid_counts = valid_data[binname].value_counts().reindex(bins, fill_value=0)
    
    epsilon = 1e-4 # 微小扰动项，防止除0错误
    train_ratio = (train_counts + epsilon) / (n_train + n_bins * epsilon)
    valid_ratio = (valid_counts + epsilon) / (n_valid + n_bins * epsilon)
    psi_series = (train_ratio - valid_ratio) * np.log(train_ratio / valid_ratio)
    
    psi = psi_series.sum()
    return psi
