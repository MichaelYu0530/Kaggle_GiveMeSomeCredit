"""Binning functions from the original research pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from .schema import colnames_abbr_map, colnames_cluster_near_zero, colnames_income_relevant, target



def get_colnames_whether_binary(data: pd.DataFrame, is_binary=False) -> list:
    """
    获取二元变量或者非二元变量的列名
    
    参数
    ----------
    data : pd.DataFrame
        待检查的数据框
    is_binary : bool, default=False
        True时返回二元变量列名，False时返回非二元连续变量列名
    
    返回
    ----------
    list
        符合条件的列名列表（排除目标变量）
    """
	
    colnames_wanted = []
    for colname in data.columns:
        if pd.api.types.is_numeric_dtype(data[colname]):
            if colname == target: # 排除目标变量
                continue
            values_reliable = data.loc[data[colname] >= 0, colname] # 负整数对应不可靠
            if not is_binary and not set(values_reliable).issubset({0, 1}):
                colnames_wanted.append(colname)
            if is_binary and set(values_reliable).issubset({0, 1}):
                colnames_wanted.append(colname)
        else: # 非数值列（如分箱列）直接跳过
            continue
    
    return colnames_wanted


def add_bins_1D(
    data: pd.DataFrame, colname: str, n_bins: int=6, fixed_div_pts: list=None, 
    constraint: list[str]=None,
) -> tuple[str, list[pd.Interval]] | None:
    """
    对指定特征进行自适应分箱处理，添加一维分箱列
    
    分箱规则
    ----------
    1. 不可靠值（<= -1）单独成箱，保留原负值作为区间边界
    2. 对在0附近聚集的指标（如逾期次数），强制在0/1/2处分箱
    3. 剩余数据使用等频分箱（pd.qcut）
    4. 最后一个区间右边界设为正无穷
    
    参数
    ----------
    data : pd.DataFrame
        包含目标变量和待分箱特征的数据集
    colname : str
        待分箱的特征列名
    n_bins : int, default=6
        期望的分箱数量（实际可能因固定分割点而增加）
    fixed_div_pts : list, default=None
        固定分割点列表，若提供则直接使用固定分箱
    constraint : list[str], default=None
        允许添加的一维分箱列名白名单，若提供且当前列不在名单中则跳过
    
    返回
    ----------
    tuple[str, list[pd.Interval]] | None
        - 若fixed_div_pts为None且constraint通过，返回 (binname_1D, div_pts)
        - 若fixed_div_pts不为None，返回None
        - 若constraint不通过，返回 (None, None)
    
    副作用
    ----------
    直接在原数据框上添加分箱列（列名格式：{base}_bin）
    """
    
    if not colname in list(data.columns):
        raise ValueError("In function add_bins_1D: colname is not a variable in data.")
    if set(data[colname].dropna()).issubset({0, 1}):
        raise ValueError("In function add_bins_1D: current colname is binary, cannot create bins.")
    
    binbase = colnames_abbr_map.get(colname, colname)
    binname_1D = binbase + '_bin' # 分箱列名
    
    if constraint is not None and binname_1D not in constraint:
        return None, None # 若添加的一维分箱存在限制且当前将要添加的不在限制内，则直接中断函数运行
    
    if fixed_div_pts is not None: # 提供固定分割点，依据固定分箱添加分箱列即可
        data[binname_1D] = pd.cut(data[colname], fixed_div_pts, right=False, include_lowest=True)
        return # 提供固定分割点时只在data中新增分箱列
    
    data_copy = data.copy()
    values = data_copy[colname]
    unreliable_counts = values[values <= -1].value_counts()
    unreliable_counts = unreliable_counts[unreliable_counts > 0]
    unreliable_values = sorted(unreliable_counts.index.tolist())
    
    normal_values = values[values >= 0]
    fixed_div_pts = [min(normal_values)]
    
    if colname in colnames_cluster_near_zero: # 一些指标在0附近严重聚集，故设定这样的固定分割点
        fixed_div_pts = fixed_div_pts + [1, 2, 3] # 强制这些指标在0/1/2处单独成一个分箱
    if len(unreliable_values) > 0:
        fixed_div_pts = unreliable_values + fixed_div_pts # 强制每个不可靠组单独成一个分箱
    fixed_div_pts = np.array(fixed_div_pts).astype('float64')
    
    remaining_n_bins = n_bins - (len(fixed_div_pts) - 1) # 除固定分割点对应区间外的区间数量
    if len(unreliable_values) > 0:
        remaining_n_bins += len(unreliable_values) - 1 # 根据不可靠分组数补偿区间数量
    
    min_val = fixed_div_pts[-1] # 固定分割点列表的最后一个元素是后续分箱的最小起始点，它本身也在后续分箱中
    series_to_cut = values[values >= min_val].copy() # 用于计算剩余区间的序列
    # credit_late_density与late_severity_score是在0处大量聚集的浮点型（连续）指标，需特别处理
    if colname == 'credit_late_density' or colname == 'late_severity_score':
        series_to_cut = values[values > min_val].copy()
    _, unfixed_div_pts = pd.qcut(series_to_cut, q=remaining_n_bins, retbins=True, duplicates='drop')
    unfixed_div_pts[-1] = np.inf # 最大的区间右端点应改为正无穷，避免迁移至验证集时不能包含验证集最大值
    
    div_pts = sorted(list(set(list(fixed_div_pts) + list(unfixed_div_pts)))) # 完整的分割点
    # 与指标列数据一一对应的分箱列数据
    data[binname_1D] = pd.cut(data_copy[colname], bins=div_pts, right=False, include_lowest=True)
    
    return binname_1D, div_pts


def convert_bins_value_to_rank(data: pd.DataFrame) -> None:
    """
    将一维分箱的具体数值转换为排名序号
    
    转换规则
    ----------
    1. 不可靠区间（left < 0）：保留原负值作为排名
    2. 可靠区间（left >= 0）：按左边界排序，从1开始编号
    
    参数
    ----------
    data : pd.DataFrame
        包含一维分箱列的数据集
    
    返回
    ----------
    None
        无返回值，直接修改原数据框，添加排名列（列名格式：{binname}rank）
    
    注意
    ----------
    仅处理一维分箱列（_bin结尾且不含_x_）
    """
	
    for colname in data.columns:
        if colname.endswith('_bin') and '_x_' not in colname: # 是一维分箱列
            bins_1D_map = {}
            
            bins_1D = set(data[colname])
            bins_1D_unreliable = {bin_1D for bin_1D in bins_1D if bin_1D.left < 0}
            bins_1D_reliable = sorted(list(bins_1D - bins_1D_unreliable))
            
            for bin_1D in bins_1D_unreliable:
                bins_1D_map[bin_1D] = int(bin_1D.left)
            bins_1D_reliable_map = dict(zip(bins_1D_reliable, range(1, len(bins_1D_reliable) + 1)))
            bins_1D_map.update(bins_1D_reliable_map)
            
            binrank_colname = f'{colname}rank'
            data[binrank_colname] = data[colname].apply(lambda x: bins_1D_map[x])


def add_bins_2D(
    data: pd.DataFrame, colname_pair: list, 
    fixed_bins_2D: list[tuple[pd.Interval, pd.Interval]]=None, constraint: list[str]=None,
    min_1D_samples: int=500, n_need_check_1D_samples: int=3000, min_2D_samples: int=50
) -> tuple[str, list[tuple[pd.Interval, pd.Interval]]]:
    """
    对已经建立一维分箱的变量组合，建立二维分箱
    
    分箱规则
    ----------
    1. 根据一维分箱的样本量，将区间分为三类：
        - 不可交互区间（样本量 < min_1D_samples）：退化为 (bin, None) 或 (None, bin)
        - 待检查区间（样本量在[min_1D_samples, n_need_check_1D_samples)）：检查二维交叉格子样本量
        - 可交互区间（样本量 >= n_need_check_1D_samples）：直接做二维交互
    2. 对于月收入相关指标，不可靠区间（left <= -1）不允许与其他区间交互
    
    参数
    ----------
    data : pd.DataFrame
        包含一维分箱列的数据集
    colname_pair : list
        两个原始变量名组成的列表，如 [age, income]
    fixed_bins_2D : list, default=None
        固定的二维分箱列表，若提供则直接使用
    constraint : list[str], default=None
        允许添加的二维分箱列名白名单
    min_1D_samples : int, default=500
        一维区间的最小样本量阈值
    n_need_check_1D_samples : int, default=3000
        一维区间需要进一步检查的样本量阈值
    min_2D_samples : int, default=50
        二维交叉格子的最小样本量阈值
    
    返回
    ----------
    tuple[str, list] | None
        - 若fixed_bins_2D为None且constraint通过，返回 (binname_2D, bins_2D)
        - 若fixed_bins_2D不为None，返回None
        - 若constraint不通过，返回 (None, None)
    
    副作用
    ----------
    直接在原数据框上添加二维分箱列（列名格式：{base1}_x_{base2}_bin）
    """
	
    if len(colname_pair) != 2:
        raise ValueError("In function add_2D_bins: colname_pair must be at length 2.")
    
    colname1, colname2 = colname_pair
    # 还原分箱列名
    binbase1 = colnames_abbr_map.get(colname1, colname1)
    binname1_1D = binbase1 + '_bin'
    binbase2 = colnames_abbr_map.get(colname2, colname2)
    binname2_1D = binbase2 + '_bin'
    binname_2D = f'{binbase1}_x_{binbase2}_bin' # 二维分箱列名
    binname_2D_another = f'{binbase2}_x_{binbase1}_bin' # 本质上相同的二维分箱列名
    
    if constraint is not None:
        is_in_constraint = binname_2D in constraint or binname_2D_another in constraint
        if not is_in_constraint:
            return None, None # 若添加的二维分箱存在限制且当前将要添加的不在限制内，则直接中断函数运行
        elif binname_2D_another in constraint:
            binname_2D = binname_2D_another # 依据限定二维分箱列上面的名称来命名
    
    if fixed_bins_2D is not None: # 传入了固定的二维分箱的分箱方式
        data[binname_2D] = [(None, None)] * data.shape[0] # 初始化赋值
        data[binname_2D] = data[binname_2D].astype(object) # 更改序列数据类型使之后续能接收元组
        
        go_2D_bins1, go_2D_bins2 = [], []
        stay_1D_bins1, stay_1D_bins2 = [], []
        for bin_2D in fixed_bins_2D:
            bin1, bin2 = bin_2D
            if bin1 is not None and bin2 is not None:
                go_2D_bins1.append(bin1); go_2D_bins2.append(bin2)
            elif bin1 is not None and bin2 is None:
                stay_1D_bins1.append(bin1)
            elif bin1 is None and bin2 is not None:
                stay_1D_bins2.append(bin2)
        
    else:
        # 一维分箱各区间内样本数
        bin_1D_counts1= data[binname1_1D].value_counts().sort_index()
        bin_1D_counts2= data[binname2_1D].value_counts().sort_index()
        # 不可做二维分箱的区间
        stay_1D_bins1 = set(bin_1D_counts1[bin_1D_counts1 < min_1D_samples].index)
        stay_1D_bins2 = set(bin_1D_counts2[bin_1D_counts2 < min_1D_samples].index)
        # 待检查是否能做二维分箱的区间
        check_mask1 = (bin_1D_counts1 >= min_1D_samples) & (bin_1D_counts1 < n_need_check_1D_samples)
        check_mask2 = (bin_1D_counts2 >= min_1D_samples) & (bin_1D_counts2 < n_need_check_1D_samples)
        check_bins1 = set(bin_1D_counts1[check_mask1].index)
        check_bins2 = set(bin_1D_counts2[check_mask2].index)
        # 可做二维分箱的区间
        go_2D_bins1 = set(bin_1D_counts1[bin_1D_counts1 >= n_need_check_1D_samples].index)
        go_2D_bins2 = set(bin_1D_counts2[bin_1D_counts2 >= n_need_check_1D_samples].index)
        
        # 对于月收入以及计算过程涉及月收入的指标，特殊处理
        if set(colname_pair).issubset(colnames_income_relevant):
            unreliable_bins1 = set([bin for bin in go_2D_bins1 if bin.left <= -1])
            unreliable_bins2 = set([bin for bin in go_2D_bins2 if bin.left <= -1])
            # 不允许月收入缺失值以及它导致的异常值所在区间与其他区间生成二维分箱
            go_2D_bins1 = go_2D_bins1 - unreliable_bins1; go_2D_bins2 = go_2D_bins2 - unreliable_bins2
            check_bins1 = check_bins1 - unreliable_bins1; check_bins2 = check_bins2 - unreliable_bins2
            stay_1D_bins1 = stay_1D_bins1.union(unreliable_bins1)
            stay_1D_bins2 = stay_1D_bins2.union(unreliable_bins2)
        
        go_2D_bins1, go_2D_bins2 = list(go_2D_bins1), list(go_2D_bins2)
        check_bins1, check_bins2 = list(check_bins1), list(check_bins2)
        stay_1D_bins1, stay_1D_bins2 = list(stay_1D_bins1), list(stay_1D_bins2)
        
        # 初始化二维分箱列，最终的二维分箱信息储存形式为由两个区间构成的元组
        data[binname_2D] = [(None, None)] * data.shape[0] # 初始化赋值
        data[binname_2D] = data[binname_2D].astype(object) # 更改序列数据类型使之后续能接收元组
        
        # 对样本数中等（待检查是否能做二维分箱）的一维区间
        def get_bins_size(bins, binname, group):
            bins_size = []
            for bin in bins:
                bins_size.append(((data[binname] == bin).sum(), group))
            return bins_size
        if len(check_bins1) + len(check_bins2) > 0:
            bins_size1 = get_bins_size(check_bins1, binname1_1D, group=1)
            bins_size2 = get_bins_size(check_bins2, binname2_1D, group=2)
            # 得到合并的，以区间内样本数和组别为索引的待检查区间列表
            check_bins = pd.concat\
                ([pd.Series(check_bins1, index=bins_size1), pd.Series(check_bins2, index=bins_size2)])\
                .sort_index(key=lambda idx_array: [idx[0] for idx in idx_array])
            
            for i, bin in enumerate(check_bins):
                _, group = check_bins.index[i]
                if group == 1:
                    selected_data = data[data[binname1_1D] == bin]
                    counts = selected_data[binname2_1D].value_counts().\
                        reindex(check_bins2 + go_2D_bins2, fill_value=0)
                    if len(counts) == 0 or min(counts) < min_2D_samples:
                        stay_1D_bins1.append(bin) # 存在过小的二维分箱，检查失败
                    else:
                        go_2D_bins1.append(bin) # 每个二维分箱都足够大，检查成功
                    check_bins1.remove(bin)
                if group == 2:
                    selected_data = data[data[binname2_1D] == bin]
                    counts = selected_data[binname1_1D].value_counts().\
                        reindex(check_bins1 + go_2D_bins1, fill_value=0)
                    if len(counts) == 0 or min(counts) < min_2D_samples:
                        stay_1D_bins2.append(bin) # 存在过小的二维分箱，检查失败
                    else:
                        go_2D_bins2.append(bin) # 每个二维分箱都足够大，检查成功
                    check_bins2.remove(bin)
    
    # 对样本数太少的一维区间，注意覆盖关系
    if len(stay_1D_bins2) > 0:
        mask_stay2 = (data[binname2_1D].isin(stay_1D_bins2))
        values = [(None, bin) for bin in data.loc[mask_stay2, binname2_1D].values]
        # 对齐索引值，否则赋值失败
        data.loc[mask_stay2, binname_2D] = pd.Series(values, index=data.index[mask_stay2])
    if len(stay_1D_bins1) > 0:
        mask_stay1 = data[binname1_1D].isin(stay_1D_bins1)
        values = [(bin, None) for bin in data.loc[mask_stay1, binname1_1D].values]
        # 对齐索引值，否则赋值失败
        data.loc[mask_stay1, binname_2D] = pd.Series(values, index=data.index[mask_stay1])
    
    # 对可以做交互的一维区间
    if len(go_2D_bins1) > 0 and len(go_2D_bins2) > 0:
        mask_go_2D = data[binname1_1D].isin(go_2D_bins1) & data[binname2_1D].isin(go_2D_bins2)
        values = list(zip(data.loc[mask_go_2D, binname1_1D], data.loc[mask_go_2D, binname2_1D]))
        data.loc[mask_go_2D, binname_2D] = pd.Series(values, index=data.index[mask_go_2D])
    
    if fixed_bins_2D is not None:
        return # 提供了固定二维分箱，则无返回值
    
    bins_2D = list(set(data[binname_2D]))
    return binname_2D, bins_2D


def sort_bins(bins: list[pd.Interval], is_reversed: bool=False) -> list:
    """
    对分箱区间列表排序，将正常区间（left >= 0）放在前面，异常/缺失区间放在后面
    
    参数
    ----------
    bins : list[pd.Interval]
        待排序的分箱区间列表
    is_reversed : bool, default=False
        是否反转排序结果
    
    返回
    ----------
    list
        排序后的区间列表
    
    异常
    ----------
    ValueError
        如果bins中包含非pd.Interval类型的元素
    """
	
    if not all(isinstance(bin, pd.Interval) for bin in bins):
        raise ValueError("In function sort_bins: all elements in bins must be pd.Interval.")
    
    bins = set(bins)
    reliable_bins = {bin for bin in bins if bin.left >= 0}
    unreliable_bins = bins - reliable_bins
    sorted_bins = sorted(list(reliable_bins)) + sorted(list(unreliable_bins), reverse=True)
    
    if is_reversed:
        return list(reversed(sorted_bins))
    else:
        return sorted_bins
