"""Summaries functions from the original research pipeline."""

from __future__ import annotations

import pandas as pd
from .binning import get_colnames_whether_binary

from .schema import colnames_abbr_map, credit, dep, income, late30, late60, late90, mortgage, target



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
