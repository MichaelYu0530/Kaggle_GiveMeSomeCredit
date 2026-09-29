"""Data functions from the original research pipeline."""

from __future__ import annotations

from sklearn.model_selection import StratifiedKFold
from typing import Literal
import numpy as np
import pandas as pd
import random
from .schema import target



def import_data(filetype: Literal['train', 'test', 'others'], filename: str=None) -> pd.DataFrame:
    """
    导入原始CSV文件
    
    参数
    ----------
    filetype : {'train', 'test', 'others'}
        数据类型，train导入cs-training.csv，test导入cs-test.csv
    filename : str, optional
        自定义文件名，当filetype='others'时使用
    
    返回
    ----------
    pd.DataFrame
        导入并预处理后的数据框（目标变量置首列，年龄列重命名）
    """
    
    if filetype == 'train':
        filename = 'cs-training.csv'
    elif filetype == 'test':
        filename = 'cs-test.csv'
    
    data = pd.read_csv(filename) # 读取数据
    if filename == 'cs-training.csv' or filename == 'cs-test.csv':
        data = data.drop('Unnamed: 0', axis=1) # 删去无意义的第一列
        data.rename(columns={'age': 'Age'}, inplace=True) # 修改为大写列名，指示这是原始指标
        
        colnames = data.columns.tolist()
        colnames.remove(target)
        data = data[[target] + sorted(colnames)]
    
    return data


def export_data(data: pd.DataFrame, filename: str=None, n_rows: int=10000, silently: bool=False):
    """
    导出数据框为CSV文件，按变量类型排序列顺序
    
    参数
    ----------
    data : pd.DataFrame
        要导出的数据框
    filename : str, default=None
        导出文件名前缀，若不设置则自动命名为“随机数+data.csv”
    n_rows : int, default=10000
        导出行数，<=0时导出全部
    silently : bool, default=False
        是否静默导出（不打印信息）
    
    返回
    ----------
    None
        无返回值，直接导出文件
    """
    
    remaining_columns = set(data.columns)
    # 原始指标
    original_list = [col for col in remaining_columns if any(char.isupper() for char in col)]
    remaining_columns -= set(original_list)
    # 变换指标
    transformed_list = [col for col in remaining_columns if col.endswith('_log') or col.endswith('_centered')]
    remaining_columns -= set(transformed_list)
    # 数据质量标记
    flag_list = [col for col in remaining_columns if col.endswith('_flag')]
    remaining_columns -= set(flag_list)
    # 用户行为标记
    behavior_list = [col for col in remaining_columns if col.startswith(('is_', 'has_'))]
    remaining_columns -= set(behavior_list)
    # 交互标记
    intersection_list = [col for col in remaining_columns if col.endswith('_signal')]
    remaining_columns -= set(intersection_list)
    # 分箱列
    bin_list = [col for col in remaining_columns if col.endswith('_bin')]
    bin_1D_list = [col for col in bin_list if '_x_' not in col]
    bin_2D_list = [col for col in bin_list if '_x_' in col]
    remaining_columns -= set(bin_list)
    # WOE列
    woe_list = [col for col in remaining_columns if col.endswith('_woe')]
    woe_1D_list = [col for col in woe_list if '_x_' not in col]
    woe_2D_list = [col for col in woe_list if '_x_' in col]
    remaining_columns -= set(woe_list)
    # 衍生指标
    derived_list = list(remaining_columns)
    
    original_list.remove(target)
    feature_list = [target] + sorted(original_list + transformed_list + derived_list, key=str.lower)
    flag_list.sort()
    behavior_list.sort()
    intersection_list.sort()
    oneD_list = sorted(bin_1D_list + woe_1D_list)
    twoD_list = sorted(bin_2D_list + woe_2D_list)
    
    export_columns = feature_list + flag_list + behavior_list + intersection_list + oneD_list + twoD_list
    data = data[export_columns]
    
    random_num = random.randint(0, 10000)
    if filename is None:
        filename = f'{random_num}data.csv'
    if n_rows > 0:
        data = data[0: n_rows]
    data.to_csv(filename, index=False)
    
    if not silently:
        print(f'已导出文件{filename}')


def create_kfold(data: pd.DataFrame, seed: int, n_splits: int=5) -> list[tuple[list[int]]]:
    """
    生成分层K折交叉验证的索引
    
    参数
    ----------
    data : DataFrame
        包含目标变量的数据集
    seed : int
        随机种子，确保结果可复现
    n_splits : int, default=5
        折数
    
    返回
    ----------
    folds : list[tuple]
        每折的训练集索引和验证集索引
    """
    
    y = data[target]
    n_samples = len(y)
    
    # 初始化分层 K 折
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )
    
    folds = []
    for train_idx, valid_idx in skf.split(np.zeros(n_samples), y):
        folds.append((train_idx, valid_idx))
    return folds
