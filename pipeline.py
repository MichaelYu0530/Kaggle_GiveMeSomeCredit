# 导入需要的库
from analysis import *
from visualization import *
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap
from typing import Literal
import warnings
warnings.filterwarnings('ignore')

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

colnames_cluster_near_zero = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfDependents',
    'short_late'
]

colnames_income_relevant = {
    'MonthlyIncome',
    'DebtRatio',
    'income_per_dep',
    'monthly_debt',
    'free_cashflow_income',
    'credit_pressure_index'
}

""" 早期数据集准备函数 """
# 导入原始的csv文件
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

# 导出数据框为csv文件
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

# 生成交叉验证各折的索引
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

""" 统一数据清洗函数 """
# 根据MonthlyIncome与NumberOfDependents是否为缺失值，添加missing flag并做后续统计学分析，随后赋值为-1
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

# 根据NumberOfTimes90DaysLate是否为96或98，添加黑名单标记 (blacklist_flag)并后续统计学分析，随后赋值为-1
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

# 根据MonthlyIncome是否为0-10，添加income_anomaly_flag并做后续统计学分析，随后赋值为-2
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

# 将NumberOfDependents大于10的值截断为10
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

# 将Age小于18的样本删除
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

# 为RevolvingUtilizationOfUnsecuredLines
# 处于0.75-1的样本添加过高标记，处于1-5的样本添加超限标记，大于5的样本添加异常标记，随后对异常标记者赋值为-1
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

# 为月收入可靠且DebtRatio
# 处于0.75-1的样本添加过高标记，处于1-5的样本添加超限标记，大于5的样本添加异常标记，随后对异常标记者赋值为-3
# 再对月收入不可靠样本的负债率赋值为与月收入相同的-1/-2，区分是月收入缺失还是极低导致的负债率不可靠
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

# 添加衍生指标late_severity_score（逾期严重程度评分，实际上就是三种逾期次数的加权和），权重由逻辑回归训练
def add_late_severity_score(data: pd.DataFrame, fixed_coefs: list[float]=None) -> tuple[float] | None:
    """
    添加逾期严重程度评分，即三种逾期次数的加权和
    
    评分公式
    ----------
    late_severity_score = beta30 * late30 + beta60 * late60 + beta90 * late90
    
    参数
    ----------
    data : pd.DataFrame
        包含30-59天、60-89天、90+天逾期次数列的数据集
    fixed_coefs : list[float], default=None
        手动指定的三个权重系数 [coef_30, coef_60, coef_90]
        若为None则基于黑名单标记为0的样本训练逻辑回归自动获取权重
    
    返回
    ----------
    tuple[float] | None
        - 若fixed_coefs为None，返回 (beta30, beta60, beta90)
        - 若fixed_coefs不为None，返回None
    
    副作用
    ----------
    直接在原数据框上添加late_severity_score列
    """
	
    
    if fixed_coefs is None: # 没有输入固定beta系数，需要自行训练求出
        selected_data = data.loc[data['blacklist_flag'] == 0, [target, late30, late60, late90]].copy()
        X_train = selected_data[[late30, late60, late90]]
        y_train = selected_data[target] # 划分自变量与因变量
        
        # 训练LR模型并提取beta系数为权重
        lr_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        lr_model.fit(X_train, y_train)
        beta30, beta60, beta90 = lr_model.coef_[0]
        data['late_severity_score'] = \
            beta30 * data[late30] + beta60 * data[late60] + beta90 * data[late90]
        
        return beta30, beta60, beta90
    
    else:
        if len(fixed_coefs) != 3:
            raise ValueError("In function add_late_severity_score: fixed_coefs must be at length 3.")
        
        beta30, beta60, beta90 = fixed_coefs
        data['late_severity_score'] = \
            beta30 * data[late30] + beta60 * data[late60] + beta90 * data[late90]

# 添加一些衍生指标，计算相关指标缺失或异常时赋值为NaN
def add_derived_features(data: pd.DataFrame, silently: bool=True) -> None:
    """
    添加衍生指标，计算相关指标缺失或异常时赋值为特殊值
    
    衍生指标说明
    ----------
    short_late : int
        30-59天与60-89天逾期次数之和，黑名单样本赋值为-1
    has_short_late : int
        是否曾短期逾期（30-89天），黑名单样本赋值为-1
    has_serious_late : int
        是否曾严重逾期（90+天），黑名单样本赋值为-1
    income_per_dep : float
        人均月收入 = 月收入 / (家属数 + 1)，月收入缺失赋-1，异常赋-2
    mortgage_ratio : float
        房贷占比 = 房贷数 / 信贷数，信贷数为0时设为0
    has_no_credit : int
        是否没有信贷
    credit_late_density : float
        短期逾期密度 = short_late / 信贷数
        无短期违约且非黑名单赋0，黑名单赋-1，有短期违约但无信贷赋-2
    has_short_late_but_no_credit : int
        是否有短期逾期记录但无信贷
    monthly_debt : float
        月债务额 = 月收入 * 负债率，月收入缺失赋-1，异常赋-2，负债率异常赋-3
    free_cashflow_income : float
        自由现金流收入 = max(0, 月收入 - 月债务)，同上
    credit_pressure_index : float
        信用压力指数 = 月债务 * 信用额度使用率，月收入缺失赋-1，异常赋-2，使用率异常赋-3
    
    参数
    ----------
    data : pd.DataFrame
        包含原始指标和标记的数据集
    silently : bool, default=True
        是否静默执行（不打印统计信息）
    
    返回
    ----------
    None
        无返回值，直接修改原数据框
    """
    
    # 黑名单筛选器
    mask_blacklist = data['blacklist_flag'] == 1
    mask_not_blacklist = ~mask_blacklist
    # 月收入可靠性筛选器
    mask_income_missing = (data['both_missing_flag'] == 1) | (data['single_missing_flag'] == 1)
    mask_income_anomaly = data['income_anomaly_flag'] == 1
    mask_income_unreliable = mask_income_missing | mask_income_anomaly
    mask_income_reliable = ~mask_income_unreliable
    # 负债率可靠性筛选器
    mask_debt_anomaly = data['debt_anomaly_flag'] == 1
    mask_debt_unreliable = mask_income_unreliable | mask_debt_anomaly
    mask_debt_reliable = ~mask_debt_unreliable
    # 信用额度使用率可靠性筛选器
    mask_util_reliable = data['util_anomaly_flag'] == 0
    
    # short_late: 30-59天与60-89天逾期次数之和
    data['short_late'] = data[late30] + data[late60]
    data.loc[mask_blacklist, 'short_late'] = -1 # 不可靠时赋值为-1
    # has_short_late: 是否曾（两年内）短期逾期（30-89天）
    data['has_short_late'] = ((data['short_late']) > 0).astype(int)
    data.loc[mask_blacklist, 'has_short_late'] = -1 # 不可靠时赋值为-1
    # has_serious_late: 是否曾（历史）严重逾期（90+天）
    data['has_serious_late'] = (data[late90] > 0).astype(int)
    data.loc[mask_blacklist, 'has_serious_late'] = -1 # 不可靠时赋值为-1
    # 逾期次数相关的衍生指标的不可靠标记与黑名单标记重复，不必生成
    
    # income_per_dep = 月收入 / (家属数 + 1)
    data['income_per_dep'] = -1 # 不可靠赋值-1/-2与月收入保持一致
    data.loc[data['income_anomaly_flag'] == 1, 'income_per_dep'] = -2
    data.loc[mask_income_reliable, 'income_per_dep'] = \
        data.loc[mask_income_reliable, income] / (data.loc[mask_income_reliable, dep] + 1)
    
    mask_has_credit = data[credit] > 0 # 信贷数非0筛选器
    mask_no_credit = ~mask_has_credit 
    # mortgage_ratio = 房贷数 / 信贷数
    data['mortgage_ratio'] = 0 # 信贷数为0时房贷数必为0，直接将比率设为0
    data.loc[mask_has_credit, 'mortgage_ratio'] = \
        data.loc[mask_has_credit, mortgage] / data.loc[mask_has_credit, credit]
    # has_no_credit: 是否没有信贷
    data['has_no_credit'] = mask_no_credit.astype(int)
    
    # credit_late_density = short_late / 信贷数
    data['credit_late_density'] = -1 # 因为逾期次数有黑名单标记的，设为-1
    mask_can_compute_here = mask_has_credit & mask_not_blacklist
    mask_has_short_late = data['has_short_late'] == 1
    data.loc[(~mask_has_short_late) & mask_not_blacklist, 'credit_late_density'] = 0 # 非黑名单无短期违约记录的，设为0
    data.loc[mask_can_compute_here, 'credit_late_density'] = \
        data.loc[mask_can_compute_here, 'short_late'] / data.loc[mask_can_compute_here, credit]
    # has_short_late_but_no_credit: 是否有短期逾期记录但无信贷
    data['has_short_late_but_no_credit'] = (mask_no_credit & mask_has_short_late).astype(int)
    data.loc[data['has_short_late_but_no_credit'] == 1, 'credit_late_density'] = -2 # 特别地，设为-2
    
    count = data['has_short_late_but_no_credit'].sum()
    ratio = round(count / len(data['has_short_late_but_no_credit']) * 100, 2)
    default_rate = round((data.loc[data['has_short_late_but_no_credit'] == 1, target]).mean() * 100, 2)
    if not silently:
        print(f'有短期逾期记录但无信贷的样本数为{count}，占比{ratio}%，的违约率为{default_rate}%')
    
    # monthly_debt = 月收入 * 负债率
    data['monthly_debt'] = data[income] * data[debt]
    # 依据导致monthly_debt不可靠的原因而赋值-1/-2/-3
    data.loc[mask_income_missing, 'monthly_debt'] = -1
    data.loc[mask_income_anomaly, 'monthly_debt'] = -2
    data.loc[mask_debt_anomaly, 'monthly_debt'] = -3
    
    # free_cashflow_income = 月债务 * debt_buffer_ratio = max(0, 月收入 - 月债务)
    values = (data[income] - data['monthly_debt']).clip(lower=0)
    data['free_cashflow_income'] = values.where(mask_debt_reliable, -1)
    data.loc[mask_income_anomaly, 'free_cashflow_income'] = -2
    data.loc[mask_debt_anomaly, 'free_cashflow_income'] = -3
    
    # credit_pressure_index = 月债务 * 信用额度使用率
    mask_can_compute_here = mask_income_reliable & mask_util_reliable
    values = data[debt] * data[util]
    data['credit_pressure_index'] = values.where(mask_can_compute_here, -1)
    data.loc[mask_income_anomaly, 'credit_pressure_index'] = -2
    data.loc[mask_debt_anomaly, 'credit_pressure_index'] = -3
    data.loc[~mask_util_reliable, 'credit_pressure_index'] = -4

# 在统一数据清洗阶段即将结束时重排指标顺序，使数据框更美观
def reorder_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    重排数据框列顺序，使输出更美观
    
    排序规则
    ----------
    1. 目标变量（第一列）
    2. 原始/衍生指标（按字母顺序）
    3. 数据质量标记（_flag结尾，按字母顺序）
    4. 用户行为标记（is_/has_开头，按字母顺序）
    5. 交互标记（_signal结尾，按字母顺序）
    
    参数
    ----------
    data : pd.DataFrame
        待重排的数据框
    
    返回
    ----------
    pd.DataFrame
        重排列顺序后的数据框
    """
	
    current_columns = data.columns.tolist()
    current_columns.remove(target)
    remaining_columns = set(current_columns)
    
    # 数据质量标记 (以'_flag'结尾)
    flag_list = [col for col in remaining_columns if '_flag' in col]
    remaining_columns -= set(flag_list)
    # 用户行为标记（以'is_'/'has_'开头）
    behavior_list = [col for col in remaining_columns if col.startswith(('is_', 'has_'))]
    remaining_columns -= set(behavior_list)
    # 交互标记（以'_signal'结尾）
    intersection_list = [col for col in remaining_columns if '_signal' in col]
    remaining_columns -= set(intersection_list)
    # 原始、衍生指标 (剩余的所有列)
    remaining_list = list(remaining_columns)
    
    reordered_columns = \
        [target] + sorted(remaining_list, key=lambda s: s.lower()) + \
        sorted(flag_list) + sorted(behavior_list) + sorted(intersection_list)
    
    return data[reordered_columns]

# 找出值得添加交互标记的二维分箱
def find_valuable_intersections(
    data: pd.DataFrame, colname_pairs: list[list[str]], 
    min_lift: float=1.75, min_corrected_lift: float=1.2,
    min_asr: float=2.58, min_inter_z_score: float=2.58
) -> pd.DataFrame:
    """
    找出值得添加交互标记的二维分箱
    
    计算指标
    ----------
    - LIFT = 实际违约率 / 整体违约率
    - corrected LIFT = 实际违约率 / 期望违约率（考虑边际分布）
    - ASR = (实际违约数 - 期望违约数) / 标准差
    - intersection Z-score = (实际违约数 - 期望违约数) / sqrt(方差)
    
    筛选条件
    ----------
    - max(LIFT, 1/LIFT) >= min_lift
    - max(corrected LIFT, 1/corrected LIFT) >= min_corrected_lift
    - |ASR| >= min_asr
    - |intersection Z-score| >= min_inter_z_score
    - (LIFT-1) / (corrected LIFT-1) > 0（方向一致）
    - ASR / intersection Z-score > 0（方向一致）
    
    参数
    ----------
    data : pd.DataFrame
        包含目标变量、一维分箱排名列、二维分箱列的数据集
    colname_pairs : list[list[str]]
        需要检查的原始变量名对列表
    min_lift : float, default=1.75
        LIFT最小值阈值
    min_corrected_lift : float, default=1.2
        corrected LIFT最小值阈值
    min_asr : float, default=2.58
        ASR绝对值最小值阈值
    min_inter_z_score : float, default=2.58
        intersection Z-score绝对值最小值阈值
    
    返回
    ----------
    pd.DataFrame
        通过筛选的二维分箱信息表，包含 binname_2D, bin1_1D, bin2_1D,
        LIFT, corrected LIFT, ASR, intersection Z-score
    """
	
    # 初始化存储数据框列名
    valuable_intersections_df= pd.DataFrame(columns=[
        'binname_2D', 'bin1_1D', 'bin2_1D', 
        'LIFT', 'corrected LIFT', 'ASR', 'intersection Z-score'
    ])
    
    for colname_pair in colname_pairs:
        colname1, colname2 = colname_pair
        binbase1, binbase2 = colnames_abbr_map.get(colname1, colname1), colnames_abbr_map.get(colname2, colname2)
        binname1_1D, binname2_1D = f'{binbase1}_binrank', f'{binbase2}_binrank' # 还原一维分箱序号列名
        binname_2D = f'{binbase1}_x_{binbase2}_bin' # 还原二维分箱列名
        
        bins_2D = data[binname_2D].unique()
        bins_2D_real = [bin_2D for bin_2D in bins_2D if bin_2D[0] and bin_2D[1]]
        mask_real = data[binname_2D].isin(bins_2D_real) # 筛选真正参与了二维分箱的样本
        needed_colnames = [target, binname1_1D, binname2_1D, binname_2D]
        selected_data = data.loc[mask_real, needed_colnames].copy() # 截取要用到的行与列
        
        # 求出交叉矩阵
        default_count_mat = selected_data.pivot_table(
            index=binname1_1D, # 行名是第一个指标的一维分箱
            columns=binname2_1D, # 列名是第二个指标的一维分箱
            values=target,
            aggfunc='sum',
            fill_value=0
        )
        total_count_mat = selected_data.pivot_table(
            index=binname1_1D, # 行名是第一个指标的一维分箱
            columns=binname2_1D, # 列名是第二个指标的一维分箱
            values=target,
            aggfunc='count',
            fill_value=0
        )
        default_rate_mat = default_count_mat / total_count_mat
        
        # 总体统计信息
        B = default_count_mat.sum().sum() # 总违约人数
        N = total_count_mat.sum().sum() # 总人数
        R = B / N # 总违约率
        
        # 计算LIFT矩阵
        lift_mat = default_rate_mat / R
        
        # 计算corrected LIFT矩阵
        default_count_col_sums = default_count_mat.sum(axis=0) # 列求和
        default_count_row_sums = default_count_mat.sum(axis=1) # 行求和
        total_count_col_sums = total_count_mat.sum(axis=0)
        total_count_row_sums = total_count_mat.sum(axis=1)
        default_rate_col_series = default_count_col_sums / total_count_col_sums
        default_rate_row_series = default_count_row_sums / total_count_row_sums
        
        default_rate_row_temp_df = default_rate_row_series.to_frame()
        default_rate_col_temp_df = default_rate_col_series.to_frame().T
        expected_default_rate_mat = \
            ((default_rate_row_temp_df @ default_rate_col_temp_df) / R) # 求出期望违约率矩阵
        expected_default_rate_mat = expected_default_rate_mat.clip(lower=1e-4) # 防止除零
        corrected_lift_mat = default_rate_mat / expected_default_rate_mat # 求出corrected LIFT矩阵
        
        # 计算ASR矩阵
        expected_default_count_mat = total_count_mat * R
        col_proportion_series = total_count_col_sums / N
        row_proportion_series = total_count_row_sums / N
        
        row_proportion_temp_df = row_proportion_series.to_frame()
        col_proportion_temp_df = col_proportion_series.to_frame().T
        denominator_mat = np.sqrt(
            (expected_default_count_mat * ((1 - row_proportion_temp_df) @ (1 - col_proportion_temp_df)))
        )
        denominator_mat = denominator_mat.clip(lower=1e-4) # 防止除零
        asr_mat = (default_count_mat - expected_default_count_mat) / denominator_mat # 求出ASR矩阵
        
        # 计算intersection Z-score矩阵
        expected_default_count_inter_mat = total_count_mat * expected_default_rate_mat
        variance_mat = (1 - expected_default_rate_mat) * expected_default_count_inter_mat
        inter_z_score_mat = \
            (default_count_mat - expected_default_count_inter_mat) / np.sqrt(variance_mat)
        
        # 各指标数值足够可观的筛选器
        mask_lift_passed = np.maximum(lift_mat, 1 / lift_mat) >= min_lift
        mask_corrected_lift_passed = \
            np.maximum(corrected_lift_mat, 1 / corrected_lift_mat) >= min_corrected_lift
        mask_asr_passed = abs(asr_mat) >= min_asr
        mask_inter_z_score_passed = abs(inter_z_score_mat) >= min_inter_z_score
        # 绝对数值和相对数值方向相同的筛选器
        mask_lift_same_dir = (lift_mat - 1) / (corrected_lift_mat - 1) > 0
        mask_asr_same_dir = asr_mat / inter_z_score_mat > 0
        # 总筛选器
        mask_passed = \
            mask_lift_passed & mask_corrected_lift_passed & mask_asr_passed & mask_inter_z_score_passed & \
            mask_lift_same_dir & mask_asr_same_dir
        
        if mask_passed.any().any(): # 存在通过筛选的二维分箱
            # 挑出对应的LIFT
            lift_passed_df = lift_mat[mask_passed].stack().reset_index()
            lift_passed_df.columns = ['bin1_1D', 'bin2_1D', 'LIFT']
            lift_passed_df = lift_passed_df.set_index(['bin1_1D', 'bin2_1D'])
            # 挑出对应的correcte LIFT
            corrected_lift_passed_df = corrected_lift_mat[mask_passed].stack().reset_index()
            corrected_lift_passed_df.columns = ['bin1_1D', 'bin2_1D', 'corrected LIFT']
            corrected_lift_passed_df = corrected_lift_passed_df.set_index(['bin1_1D', 'bin2_1D'])
            # 挑出对应的ASR
            asr_passed_df = asr_mat[mask_passed].stack().reset_index()
            asr_passed_df.columns = ['bin1_1D', 'bin2_1D', 'ASR']
            asr_passed_df = asr_passed_df.set_index(['bin1_1D', 'bin2_1D'])
            # 挑出对应的intersection Z-score
            inter_z_score_passed_df = inter_z_score_mat[mask_passed].stack().reset_index()
            inter_z_score_passed_df.columns = ['bin1_1D', 'bin2_1D', 'intersection Z-score']
            inter_z_score_passed_df = inter_z_score_passed_df.set_index(['bin1_1D', 'bin2_1D'])
            # 整合通过筛选的二维分箱的信息
            passed_intersections_df = pd.concat(
                [lift_passed_df, corrected_lift_passed_df, asr_passed_df, inter_z_score_passed_df], axis=1
            ).reset_index()
            passed_intersections_df['binname_2D'] = binname_2D
            valuable_intersections_df = pd.concat(
                [valuable_intersections_df, passed_intersections_df], axis=0, ignore_index=True
            )
    
    return valuable_intersections_df

# 为二维分箱的交互效应强度排序，筛选出最适合添加交互标记的二维分箱
def filter_intersections(
    intersections_df_list: list[pd.DataFrame], positive_count: int=50, negetive_count: int=10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    为二维分箱的交互效应强度排序，筛选出最适合添加交互标记的二维分箱
    
    处理流程
    ----------
    1. 找出在5折中都出现的二维分箱（取交集）
    2. 计算各指标的平均值
    3. 计算综合得分 = log(LIFT) * log(corrected LIFT) * log(|ASR|) * log(|Z-score|)
    4. 按ASR正负分组，分别按得分排序
    
    参数
    ----------
    intersections_df_list : list[pd.DataFrame]
        五折交叉验证得到的五个结果表
    positive_count : int, default=50
        正向效应（ASR>0）保留数量
    negetive_count : int, default=10
        负向效应（ASR<0）保留数量
    
    返回
    ----------
    tuple[pd.DataFrame, pd.DataFrame]
        - top_positive_intersections_df : 正向效应Top N结果表
        - top_negative_intersections_df : 负向效应Top N结果表
    """
	
    ids_list = []
    for df in intersections_df_list:
        id_series = df['binname_2D'] + '_' + df['bin1_1D'].astype(str) + '_' + df['bin2_1D'].astype(str)
        df['bin_id'] = id_series
        ids_list.append(set(id_series))
    # 求出在全部表中都出现的二维分箱（即每次循环中都通过筛选的交互项）
    stable_ids = set.intersection(*ids_list)
    if not stable_ids:
        raise ValueError("In function select_best_intersections: no proper 2D bin is found.")
    
    # 提取出每次循环中都通过筛选的交互项
    stable_df_list = []
    for df in intersections_df_list:
        stable_df = df.loc[df['bin_id'].isin(stable_ids), list(set(df.columns) - {'bin_id'})]
        stable_df_list.append(stable_df)
    # 合并为一个大数据框方便分组统计
    combined_df = pd.concat(stable_df_list, ignore_index=True)
    
    # 取四个评判指标的平均值
    passed_intersections_df = combined_df.groupby(['binname_2D', 'bin1_1D', 'bin2_1D']).agg(
        LIFT_mean=('LIFT', 'mean'),
        corrected_LIFT_mean=('corrected LIFT', 'mean'),
        ASR_mean=('ASR', 'mean'),
        intersection_Z_score_mean=('intersection Z-score', 'mean'),
    ).reset_index()
    
    # 计算得分
    passed_intersections_df['mark'] = \
        np.log(passed_intersections_df['LIFT_mean']).abs() * \
        np.log(passed_intersections_df['corrected_LIFT_mean']).abs() * \
        np.log(passed_intersections_df['ASR_mean'].abs()) * \
        np.log(passed_intersections_df['intersection_Z_score_mean'].abs())
    
    # 区分交互效应为增强与削弱的交互项
    passed_positive_df = passed_intersections_df.loc[passed_intersections_df['ASR_mean'] > 0]
    passed_negative_df = passed_intersections_df.loc[passed_intersections_df['ASR_mean'] < 0]
    
    # 根据得分值排序结果表
    top_positive_intersections_df = passed_positive_df.sort_values(by='mark', ascending=False)\
        .iloc[0: positive_count].reset_index(drop=True)
    top_negative_intersections_df = passed_negative_df.sort_values(by='mark', ascending=False)\
        .iloc[0: negetive_count].reset_index(drop=True)
    
    return top_positive_intersections_df, top_negative_intersections_df

# 添加交互标记
def add_intersection_signals(
    data: pd.DataFrame, positive_interactions_df: pd.DataFrame, negative_interactions_df: pd.DataFrame
) -> None:
    """
    在数据中添加交互效应标记列
    
    参数
    ----------
    data : pd.DataFrame
        包含一维分箱排名列的数据集
    positive_interactions_df : pd.DataFrame
        增强效应交互项配置表，需包含 signal_name, binrank_colname1, binrank_colname2, binrank1, binrank2
    negative_interactions_df : pd.DataFrame
        削弱效应交互项配置表，格式同上
    
    返回
    ----------
    None
        无返回值，直接在原数据框上添加标记列
    """
	
    intersection_df = pd.concat([positive_interactions_df, negative_interactions_df], ignore_index=True)
    # 获取两个一维分箱的排名序号跨度
    split_df1 = intersection_df['binrank1'].str.split('_', expand=True)
    intersection_df['binrank1_start'] = split_df1[0].astype(int)
    intersection_df['binrank1_end'] = split_df1[1].astype(int)
    split_df2 = intersection_df['binrank2'].str.split('_', expand=True)
    intersection_df['binrank2_start'] = split_df2[0].astype(int)
    intersection_df['binrank2_end'] = split_df2[1].astype(int)
    
    for idx in range(len(intersection_df)):
        row_series = intersection_df.loc[idx]
        # 提取行序列内的信息
        signal_name = row_series['signal_name']
        binrank_colname1, binrank_colname2 = row_series[['binrank_colname1', 'binrank_colname2']]
        binrank1_start, binrank1_end = row_series[['binrank1_start', 'binrank1_end']]
        binrank2_start, binrank2_end = row_series[['binrank2_start', 'binrank2_end']]
        
        # 筛选恰在交互项中的样本，直接修改输入的数据框
        mask_match = \
            (data[binrank_colname1] >= binrank1_start) & (data[binrank_colname1] <= binrank1_end) & \
            (data[binrank_colname2] >= binrank2_start) & (data[binrank_colname2] <= binrank2_end)
        data[signal_name] = mask_match.astype(int)

""" 分箱相关函数 """
# 获取所有非二元的列名
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

# 计算指定指标的分箱分割点，为该指标添加分箱一列
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

# 将所有的一维分箱的具体数值转换为排名序号
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

# 对已经建立一维分箱的装备，建立二维分箱
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

# 个性化区间排序，将异常/缺失区间排在正常区间之后
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

""" raw LR特化表达函数 """
# 新增若干列的年龄、负债率、信用额度使用率对应中心化列
def add_centered_features(data: pd.DataFrame, fixed_colnames: list[str]=None) -> None:
    """
    添加中心化特征列
    
    中心化公式
    ----------
    col_centered = col - mean(col)  （可靠样本的均值）
    不可靠样本的中心化列赋值为0
    
    参数
    ----------
    data : pd.DataFrame
        包含原始指标的数据集
    fixed_colnames : list[str], default=None
        需要中心化的列名列表，为None时默认对年龄、负债率、使用率做中心化
    
    返回
    ----------
    None
        无返回值，直接修改原数据框，添加 {base}_centered 列
        若对年龄中心化，还会添加 age_centered_*_{base}_centered 交互项
    """
	
    if fixed_colnames is None:
        colnames = [age, debt, util]
    else:
        colnames = fixed_colnames
    
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
        reliable_mean = data.loc[mask_reliable, colname].mean() # 可靠样本中的均值
        
        # 将不可靠样本的中心化列中的数据赋值为0
        data[colname_centered] = (data[colname] - reliable_mean).where(mask_reliable, 0)
        if colname != age:
            colname_multiplied = f'age_centered_*_{binbase}_centered'
            data[colname_multiplied] = data['age_centered'] * data[colname_centered]
    
    colnames_to_center = colnames
    return colnames_to_center

# 将有长尾的指标取对数（实际上是log1p）
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
        colnames = data.columns.tolist()
    else:
        colnames = fixed_colnames
    
    colnames_to_log = []
    for colname in colnames:
        if pd.api.types.is_numeric_dtype(data[colname]) and data[colname].max() > 200:
            colnames_to_log.append(colname)
            
            binbase = colnames_abbr_map.get(colname, colname)
            colname_logged = f'{binbase}_log' # 对数变换指标的列名
            data[colname_logged] = np.log1p(data[colname])
    
    return colnames_to_log

# 筛选投入训练的列名，用变换后的指标替代原指标
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

""" WOE LR特化表达函数 """
# 构建所有标记的风险比率、IV的结果表
def test_binary_columns(data: pd.DataFrame, colnames: list[str]) -> pd.DataFrame:
    """
    计算二元标记的风险比率和IV值
    
    参数
    ----------
    data : pd.DataFrame
        包含目标变量和二元标记列的数据集
    colnames : list[str]
        需要测试的二元标记列名列表
    
    返回
    ----------
    pd.DataFrame
        包含 colname, count_marked, count_normal, RR, IV 列的结果表
    """
	
    binary_columns_test_df = pd.DataFrame(
        columns=['colname', 'count_marked', 'count_normal', 'RR', 'IV']
    )
    for colname in colnames:
        group_df = data.groupby(colname, observed=True).agg(
            count=(target, 'count'), # 标记变量各整数值的样本数
            default_count=(target, 'sum'), # 违约人数
            default_rate=(target, 'mean') # 违约率
        ).reset_index()
        mask_marked = group_df[colname] == 1
        mask_normal = group_df[colname] == 0
        count_marked = group_df.loc[mask_marked, 'count'].values[0]
        count_normal = group_df.loc[mask_normal, 'count'].values[0]
        default_rate_normal = group_df.loc[mask_normal, 'default_rate'].values[0]
        default_rate_marked = group_df.loc[mask_marked, 'default_rate'].values[0]
        risk_ratio = default_rate_marked / default_rate_normal # 计算风险比率
        _, iv_binary = calculate_woe_iv(data, colname) # 计算IV
        
        row_info_series = pd.Series(
            [colname, count_marked, count_normal, risk_ratio, iv_binary],
            index=['colname', 'count_marked', 'count_normal', 'RR', 'IV']
        )
        binary_columns_test_df.loc[len(binary_columns_test_df)] = row_info_series
    
    return binary_columns_test_df

# 构建所有一维分箱的IV、PSI的结果表
def test_bins_1D(
    train_data: pd.DataFrame, valid_data: pd.DataFrame, colnames: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    计算一维分箱的IV、PSI、卡方p值
    
    参数
    ----------
    train_data : pd.DataFrame
        训练集数据
    valid_data : pd.DataFrame
        验证集数据
    colnames : list[str]
        原始指标列名列表（会自动转换为分箱列名）
    
    返回
    ----------
    tuple
        - bins_1D_test_df : pd.DataFrame
            包含 binname_1D, IV, PSI, p-value 列的结果表
        - iv_1D_dict : dict[str, float]
            原始指标名到IV值的映射字典
    """
	
    iv_1D_dict = {}
    bins_1D_test_df = pd.DataFrame(columns=['binname_1D', 'IV', 'PSI', 'p-value'])
    for colname in colnames:
        binbase = colnames_abbr_map.get(colname, colname)
        binname_1D = f'{binbase}_bin'
        
        _, iv_1D = calculate_woe_iv(train_data, binname_1D) # 计算WOE和IV
        psi_1D = calculate_psi(train_data, valid_data, binname_1D) # 计算PSI
        iv_1D_dict[colname] = iv_1D
        
        bins_1D_df = summary_bins(train_data, binname_1D)
        contingency_table = bins_1D_df[['未违约', '违约']].values # 获取列联表
        _1, p_value_1D, _2, _3 = chi2_contingency(contingency_table) # 计算p值
        
        row_info_series = pd.Series(
            [binname_1D, iv_1D, psi_1D, p_value_1D], index=['binname_1D', 'IV', 'PSI', 'p-value']
        )
        bins_1D_test_df.loc[len(bins_1D_test_df)] = row_info_series
    
    return bins_1D_test_df, iv_1D_dict

# 构建所有二维分箱的IV、IV增量、IS、PSI、卡方检验p值、风险跨度、风险方差的结果表
def test_bins_2D(
    train_data: pd.DataFrame, valid_data: pd.DataFrame, colname_pairs: list[list[str]], iv_1D_dict: dict,
) -> pd.DataFrame:
    """
    计算二维分箱的IV、IV增量、IS、PSI、卡方p值、风险跨度、风险方差
    
    参数
    ----------
    train_data : pd.DataFrame
        训练集数据
    valid_data : pd.DataFrame
        验证集数据
    colname_pairs : list[list[str]]
        原始变量名对列表
    iv_1D_dict : dict
        一维分箱的IV值字典
    
    返回
    ----------
    pd.DataFrame
        包含 binname_2D, IV, delta IV, IS, PSI, p-value, risk range, risk Var 列的结果表
    """
	
    bins_2D_test_df = pd.DataFrame(
        columns=['binname_2D', 'IV', 'delta IV', 'IS', 'PSI', 'p-value', 'risk range', 'risk Var']
    )
    for colname_pair in colname_pairs:
        colname1, colname2 = colname_pair
        binbase1 = colnames_abbr_map.get(colname1, colname1)
        binbase2 = colnames_abbr_map.get(colname2, colname2)
        binname_2D = f'{binbase1}_x_{binbase2}_bin' # 还原二维分箱列名
        
        _, iv_2D = calculate_woe_iv(train_data, binname_2D) # 计算WOE和IV
        delta_iv = iv_2D - max(iv_1D_dict[colname1], iv_1D_dict[colname2]) # 计算IV增量
        i_s = iv_2D / (iv_1D_dict[colname1] + iv_1D_dict[colname2]) # 计算IS
        psi_2D = calculate_psi(train_data, valid_data, binname_2D) # 计算PSI
        
        bins_2D_df = summary_bins(train_data, binname_2D)
        contingency_table = bins_2D_df[['未违约', '违约']].values # 获取列联表
        _1, p_value_2D, _2, _3 = chi2_contingency(contingency_table) # 计算p值
        risk_range_2D = bins_2D_df['违约率'].max() - bins_2D_df['违约率'].min() # 计算风险跨度
        default_rate_total = bins_2D_df['违约'].sum() / bins_2D_df['样本数'].sum() # 计算整体违约率
        risk_var_2D = \
            (bins_2D_df['占比'] * (bins_2D_df['违约率'] - default_rate_total) ** 2).sum() #计算风险方差
        
        row_info_series = pd.Series(
            [binname_2D, iv_2D, delta_iv, i_s, psi_2D, p_value_2D, risk_range_2D, risk_var_2D],
            index=['binname_2D', 'IV', 'delta IV', 'IS', 'PSI', 'p-value', 'risk range', 'risk Var']
        )
        bins_2D_test_df.loc[len(bins_2D_test_df)] = row_info_series
    
    return bins_2D_test_df

# 筛选可以保留并投入训练的二元标记
def filter_binary_columns(binary_columns_test_df_list: list[pd.DataFrame], min_rr: float=1.7) \
    -> pd.DataFrame:
    """
    筛选可保留的二元标记
    
    筛选规则
    ----------
    风险比率 RR 不低于阈值（取 RR 和 1/RR 中的最大值）
    
    参数
    ----------
    binary_columns_test_df_list : list[pd.DataFrame]
        五折交叉验证得到的五个结果表
    min_rr : float, default=1.7
        风险比率阈值
    
    返回
    ----------
    pd.DataFrame
        包含 colname, count_marked_mean, count_normal_mean, RR_mean, IV_mean, is_passed 列的汇总表
    """
	
    combined_df = pd.concat(binary_columns_test_df_list, ignore_index=True) # 把全部数据表合并为一个大表
    # 计算评判指标的平均值
    summary_df = combined_df.groupby('colname').agg(
        count_marked_mean = ('count_marked', 'mean'),
        count_normal_mean = ('count_normal', 'mean'),
        RR_mean = ('RR', 'mean'),
        IV_mean = ('IV', 'mean')
    )
    # 为标记列名排序使之美观
    colnames_group1 = [colname for colname in summary_df.index if colname.endswith('_flag')]
    colnames_group2 = [colname for colname in summary_df.index if colname.startswith(('is_', 'has_'))]
    colnames_group3 = [colname for colname in summary_df.index if colname.endswith('_signal')]
    sorted_colnames = sorted(colnames_group1) + sorted(colnames_group2) + sorted(colnames_group3)
    summary_df = summary_df.reindex(index=sorted_colnames).reset_index()
    
    rr_effect_series = summary_df['RR_mean'].apply(lambda x: max(x, 1 / x))
    # 标记可以保留的条件：风险比率与其倒数的最大值不低于阈值
    mask_passed = rr_effect_series >= min_rr
    summary_df['is_passed'] = mask_passed
    
    return summary_df

# 筛选可以保留并投入训练的一维分箱
def filter_bins_1D(
    bins_1D_test_df_list: list[pd.DataFrame],
    min_iv: float=0.05, max_psi: float=0.05, max_p_value: float=0.01
) -> pd.DataFrame:
    """
    筛选可保留的一维分箱
    
    筛选规则
    ----------
    IV >= min_iv 且 PSI <= max_psi 且 p值 <= max_p_value
    
    参数
    ----------
    bins_1D_test_df_list : list[pd.DataFrame]
        五折交叉验证得到的五个结果表
    min_iv : float, default=0.05
        最小IV阈值
    max_psi : float, default=0.05
        最大PSI阈值
    max_p_value : float, default=0.01
        最大p值阈值
    
    返回
    ----------
    pd.DataFrame
        包含 binname_1D, IV_mean, PSI_upper, p_value_upper, is_passed 列的汇总表
    """
	
    combined_df = pd.concat(bins_1D_test_df_list, ignore_index=True) # 把全部数据表合并为一个大表
    summary_df = combined_df.groupby('binname_1D').agg(
        IV_mean = ('IV', 'mean'),
        PSI_upper = ('PSI', 'max'),
        p_value_upper = ('p-value', 'max')
    ).reset_index()
    
    # 标记可以保留的条件：所有评判指标均满足筛选标准
    mask_passed = (summary_df['IV_mean'] >= min_iv) & \
        (summary_df['PSI_upper'] <= max_psi) & (summary_df['p_value_upper'] <= max_p_value)
    summary_df['is_passed'] = mask_passed
    
    return summary_df

# 筛选可以保留并投入训练的二维分箱
def filter_bins_2D(
    bins_2D_test_df_list: list[pd.DataFrame],
    min_iv: float=0.05, min_delta_iv: float=0.02, min_is: float=1.2, max_psi: float=0.1,
    max_p_value: float=0.01, min_rr: float=0.05, min_rv: float=0.0005
) -> pd.DataFrame:
    """
    筛选可保留的二维分箱
    
    筛选规则
    ----------
    IV >= min_iv 且 delta IV >= min_delta_iv 且 IS >= min_is 且
    PSI <= max_psi 且 p值 <= max_p_value 且 risk range >= min_rr 且 risk Var >= min_rv
    
    参数
    ----------
    bins_2D_test_df_list : list[pd.DataFrame]
        五折交叉验证得到的五个结果表
    min_iv : float, default=0.05
        最小IV阈值
    min_delta_iv : float, default=0.02
        最小IV增量阈值
    min_is : float, default=1.2
        最小IS阈值
    max_psi : float, default=0.1
        最大PSI阈值
    max_p_value : float, default=0.01
        最大p值阈值
    min_rr : float, default=0.05
        最小风险跨度阈值
    min_rv : float, default=0.0005
        最小风险方差阈值
    
    返回
    ----------
    pd.DataFrame
        通过筛选的二维分箱结果表
    """
	
    combined_df = pd.concat(bins_2D_test_df_list, ignore_index=True) # 把全部数据表合并为一个大表
    summary_df = combined_df.groupby('binname_2D').agg(
        IV_mean = ('IV', 'mean'),
        delta_IV_mean = ('delta IV', 'mean'),
        IS_mean = ('IS', 'mean'),
        PSI_upper = ('PSI', 'max'),
        p_value_upper = ('p-value', 'max'),
        RR_mean = ('risk range', 'mean'),
        RV_mean = ('risk Var', 'mean')
    ).reset_index()
    # 标记可以保留的条件：所有评判指标均满足筛选标准
    mask_passed = \
        (summary_df['IV_mean'] >= min_iv) & (summary_df['delta_IV_mean'] >= min_delta_iv) & \
        (summary_df['PSI_upper'] <= max_psi) & (summary_df['p_value_upper'] <= max_p_value) & \
        (summary_df['RR_mean'] >= min_rr) & (summary_df['RV_mean'] >= min_rv) & \
        (summary_df['IS_mean'] >= min_is)
    
    passed_bins_2D_df = summary_df.loc[mask_passed]
    
    return passed_bins_2D_df

# 创建分箱列对应的WOE列
def add_woe_column(data: pd.DataFrame, colname: str) -> None:
    """
    添加分箱列对应的WOE列
    
    参数
    ----------
    data : pd.DataFrame
        包含分箱列或标记列的数据集
    colname : str
        分箱列名或标记列名（必须以 _flag, _signal, _bin 结尾或以 has_, is_ 开头）
    
    返回
    ----------
    None
        无返回值，直接修改原数据框，添加 {colname}_woe 列
    
    异常
    ----------
    ValueError
        如果colname不符合命名规范，或WOE映射不完整
    """
	
    if not colname.endswith(('_flag', '_signal', '_bin')) and not colname.startswith(('has_', 'is_')):
        raise ValueError(f"In function add_woe_column: {colname} is neither a bin column nor a flag column.")
    
    woe_colname = f'{colname}_woe'
    woe_map, _ = calculate_woe_iv(data, colname)
    woe_values = data[colname].map(woe_map)
    if woe_values.isna().any():
        raise ValueError("In function add_woe_col: woe_map does not match current bin column.")
    else:
        data[woe_colname] = woe_values

# 自定义的评分卡模型类
class ScoreCard:
    """
    自定义评分卡模型类
    
    评分卡原理
    ----------
    评分 = A - B * log(odds)
    其中 odds = p / (1-p)，p为违约概率
    
    A和B通过基准点确定：
    - 当 odds = base_odds 时，评分为 base_score
    - 当 odds = 2 * base_odds 时，评分为 base_score + PDO
    
    参数
    ----------
    PDO : float, default=20
        Points to Double Odds，odds翻倍时分数增加量
    base_score : float, default=600
        基准分数（对应base_odds时的分数）
    base_odds : float, default=50
        基准odds值
    
    属性
    ----------
    model : LogisticRegression
        训练好的逻辑回归模型
    coef_ : ndarray
        模型系数
    intercept_ : float
        模型截距
    feature_names : list
        特征列名列表
    base_points : float
        基础分（不含变量贡献的分数）
    scorecard_df : pd.DataFrame
        变量级评分卡，包含 feature, coef, score_factor 三列
    """
    
    # 评分卡初始化函数
    def __init__(self, PDO: float=20, base_score: float=600, base_odds: float=50) -> None:
        """
        初始化评分卡参数
        
        参数
        ----------
        PDO : float, default=20
            Points to Double Odds，odds翻倍时分数增加量
        base_score : float, default=600
            基准分数
        base_odds : float, default=50
            基准odds值
        """
        
        # 基础参数
        self.PDO = PDO
        self.base_score = base_score
        self.base_odds = base_odds
        # scaling参数
        self.A: float = None
        self.B: float = None
        # 模型参数
        self.model: LogisticRegression = None
        self.coef_: np.ndarray = None
        self.intercept_: float = None
        self.feature_names: list[str] = None
        # 评分卡结构
        self.base_points: float = None
        self.scorecard_df: pd.DataFrame = None
    
    # 拟合评分卡函数
    def fit(self, woe_lr_model: LogisticRegression, X: pd.DataFrame) -> None:
        """
        拟合评分卡，计算缩放参数和评分卡表
        
        处理流程
        ----------
        1. 保存模型参数（系数、截距、特征名）
        2. 计算缩放参数 B = PDO / ln(2)
        3. 计算缩放参数 A = base_score + B * ln(base_odds)
        4. 计算基础分 base_points = A - B * intercept_
        5. 构建评分卡表，计算各特征的分数因子 = -B * coef
        
        参数
        ----------
        woe_lr_model : LogisticRegression
            已训练好的WOE逻辑回归模型
        X : pd.DataFrame
            训练集特征（WOE编码后）
        """
        
        self.model = woe_lr_model
        self.coef_ = woe_lr_model.coef_[0]
        self.intercept_ = woe_lr_model.intercept_[0]
        self.feature_names = X.columns.tolist()
        # scaling
        self.B = self.PDO / np.log(2)
        self.A = self.base_score + self.B * np.log(self.base_odds)
        # 基础分
        self.base_points = self.A - self.B * self.intercept_
        # 构建评分卡表
        self.scorecard_df = pd.DataFrame({
            'feature': self.feature_names,
            'coef': self.coef_,
        })
        self.scorecard_df['score_factor'] = -self.B * self.scorecard_df['coef']
    
    # 计算分数函数
    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        """
        计算样本的预测分数
        
        分数公式
        ----------
        score = A - B * (X·coef + intercept)
        
        参数
        ----------
        X : pd.DataFrame
            待预测的特征数据（WOE编码后）
        
        返回
        ----------
        pd.Series
            每个样本的预测分数（分数越高，违约风险越低）
        """
        
        linear_part = np.dot(X, self.coef_) + self.intercept_
        score = self.A - self.B * linear_part
        return score
    
    # 概率预测函数
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)
    
    # 计算AUC
    def get_auc(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        计算模型在指定数据集上的AUC
        
        参数
        ----------
        X : pd.DataFrame
            特征数据（WOE编码后）
        y : pd.Series
            真实标签
        
        返回
        ----------
        float
            AUC值
        """
        
        y_prob = self.model.predict_proba(X)[:, 1]
        auc, _ = calculate_auc_ks(y, y_prob)
        return auc
    
    # 构建评分分箱与违约率的关系表
    def show_score_bins(self, X: pd.DataFrame, y: pd.Series, n_bins: int=10) -> pd.DataFrame:
        """
        构建评分分箱与违约率的关系表，用于评估评分卡区分能力
        
        处理流程
        ----------
        1. 计算每个样本的预测分数
        2. 对分数进行等频分箱
        3. 统计各分箱的样本数、坏账数、平均分数
        4. 计算坏账率、累计坏账率、累计好账率
        5. 计算各分箱的KS值
        
        参数
        ----------
        X : pd.DataFrame
            特征数据（WOE编码后）
        y : pd.Series
            真实标签
        n_bins : int, default=10
            评分分箱数量
        
        返回
        ----------
        pd.DataFrame
            包含 score_bin, total_count, bad_count, score_mean, bad_rate, KS列的结果表
        """
        
        score = self.predict_score(X) # 求出评分序列
        score_df = pd.DataFrame({
            'score': score,
            'y': y
        }) # 构建评分与违约情况的对照表
        score_df['score_bin'] = pd.qcut(score_df['score'], n_bins, duplicates='drop') # 对评分分箱
        
        score_bins_df = score_df.groupby('score_bin').agg(
            total_count=('y', 'count'),
            bad_count=('y', 'sum'),
            score_mean=('score', 'mean')
        )
        
        score_bins_df['good_count'] = score_bins_df['total_count'] - score_bins_df['bad_count']
        score_bins_df['bad_rate'] = score_bins_df['bad_count'] / score_bins_df['total_count']
        # 重新排序，高分（低违约率）者在上
        score_bins_df = score_bins_df.sort_values(by='score_bin', ascending=False).reset_index()
        
        # 违约者/未违约者的累计计数
        score_bins_df['cum_total_count'] = score_bins_df['total_count'].cumsum()
        score_bins_df['cum_bad_count'] = score_bins_df['bad_count'].cumsum()
        score_bins_df['cum_good_count'] = score_bins_df['good_count'].cumsum()
        # 违约者/未违约者的总数
        total_bad_count = score_bins_df['bad_count'].sum()
        total_good_count = score_bins_df['good_count'].sum()
        # 违约者/未违约者的占比
        score_bins_df['cum_bad_rate'] = score_bins_df['cum_bad_count'] / total_bad_count
        score_bins_df['cum_good_rate'] = score_bins_df['cum_good_count'] / total_good_count
        
        # KS曲线基础
        score_bins_df['KS'] = abs(score_bins_df['cum_bad_rate'] - score_bins_df['cum_good_rate'])
        
        # 简化输出的表格
        colnames_to_drop = [
            'good_count', 'cum_total_count', 'cum_bad_count', 'cum_good_count',
            'cum_bad_rate', 'cum_good_rate'
        ]
        score_bins_df.drop(colnames_to_drop, axis=1, inplace=True)
        
        return score_bins_df
    
    # 计算基于评分的KS
    def get_ks_score_based(self, X: pd.DataFrame, y: pd.Series, n_bins: int=10) -> float:
        """
        计算基于评分的KS值
        
        KS值 = max(|累计好账率 - 累计坏账率|)
        
        参数
        ----------
        X : pd.DataFrame
            特征数据（WOE编码后）
        y : pd.Series
            真实标签
        n_bins : int, default=10
            评分分箱数量
        
        返回
        ----------
        float
            KS值
        """
        
        score_bins_df = self.show_score_bins(X, y, n_bins=n_bins)
        ks_score_based = score_bins_df['KS'].max()
        return ks_score_based
    
    # 打印摘要函数
    def summary(self) -> None:
        """
        打印评分卡摘要信息
        
        输出内容
        ----------
        - Base Score：基础分
        - PDO：Points to Double Odds
        - A, B：缩放参数
        """
        
        print("==== ScoreCard Summary ====")
        print(f"Base Score: {self.base_points:.2f}")
        print(f"PDO: {self.PDO}")
        print(f"A: {self.A:.2f}, B: {self.B:.2f}")

""" 模型调参函数 """
# 对LR做网格搜索，记录不同拟合指标和模型参数情况下的模型拟合效果
def grid_search_lr(
    X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series,
    colnames_to_fit_list: list[list[str]], c_list: list[float]
) -> pd.DataFrame:
    lr_fit_goodness_df = pd.DataFrame(columns=[
        'fit columns index', 'C value', 
        'train AUC', 'valid AUC', 'AUC gap', 'train KS', 'valid KS', 'KS gap'
    ] + X_train.columns.tolist()) # 初始化结果储存表
    X_train_copy = X_train.copy(); X_valid_copy = X_valid.copy()
    
    for idx, colnames_to_fit in enumerate(colnames_to_fit_list):
        X_train = X_train_copy; X_valid = X_valid_copy
        X_train = X_train[colnames_to_fit]; X_valid = X_valid[colnames_to_fit]
        
        for c in c_list:
            lr_model = LogisticRegression(
                penalty='l2',
                C=c,
                solver='lbfgs',
                max_iter=1000,
                fit_intercept=True
            ) # 设置LR参数
            lr_model.fit(X_train, y_train) # 拟合模型
            y_prob_train = lr_model.predict_proba(X_train)[: , 1] # 训练集预测概率
            y_prob_valid = lr_model.predict_proba(X_valid)[: , 1] # 验证集预测概率
            
            # 求出AUC与KS
            auc_train, ks_train = calculate_auc_ks(y_train, y_prob_train)
            auc_valid, ks_valid = calculate_auc_ks(y_valid, y_prob_valid)
            # 求出系数符号稳定性
            coef = lr_model.coef_.flatten()
            coef_dict = dict(zip(X_train.columns, coef)) # 每个指标对应一个系数
            
            row_dict = {
                'fit columns index':int(idx), 'C value': c,
                'train AUC': auc_train, 'valid AUC': auc_valid, 'AUC gap': auc_train - auc_valid,
                'train KS': ks_train, 'valid KS': ks_valid, 'KS gap': ks_train - ks_valid
            }
            row_dict.update(coef_dict)
            row_series = pd.Series(row_dict)
            lr_fit_goodness_df.loc[len(lr_fit_goodness_df)] = row_series
    
    return lr_fit_goodness_df

# 整理五折交叉验证后的LR的网格搜索拟合效果
def test_fit_goodness_lr(
    lr_fit_goodness_df_list: list[pd.DataFrame], unstable_threshold: float=0.6, max_std: float=0.05
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    整理五折交叉验证后的LR的网格搜索拟合效果，评估系数稳定性
    
    系数稳定性判断
    ----------
    若符号稳定性 < unstable_threshold，或符号稳定性 == unstable_threshold 且标准差 > max_std，则视为不稳定
    
    参数
    ----------
    lr_fit_goodness_df_list : list[pd.DataFrame]
        五折交叉验证得到的五个网格搜索结果表
    unstable_threshold : float, default=0.6
        符号稳定性阈值（低于此值视为不稳定）
    max_std : float, default=0.05
        系数标准差阈值（当符号稳定性等于阈值时使用）
    
    返回
    ----------
    tuple[pd.DataFrame, pd.DataFrame]
        - auc_ks_df : AUC/KS汇总表
        - coef_sign_stability_df : 系数稳定性结果表
    """
	
    # 将五折交叉验证得到的五个结果表合并
    combined_df = pd.concat(lr_fit_goodness_df_list, ignore_index=True)
    combined_df = combined_df.dropna(axis=1, how='all')
    
    # 汇总AUC、KS的均值
    auc_ks_df = combined_df.groupby(['fit columns index', 'C value']).agg(
        valid_AUC_mean = ('valid AUC', 'mean'),
        AUC_gap_mean = ('AUC gap', 'mean'),
        valid_KS_mean = ('valid KS', 'mean'),
        KS_gap_mean = ('KS gap', 'mean'),
    ).reset_index()
    
    # 初始化系数稳定性的结果表
    colnames = list(set(combined_df.columns) - {
        'fit columns index', 'C value',
        'train AUC', 'valid AUC', 'AUC gap', 'train KS', 'valid KS', 'KS gap'
    })
    coef_sign_stability_df = pd.DataFrame(
        columns=['fit columns index', 'C value', 'colname', 'sign stabilty', 'std']
    )
    for idx in auc_ks_df['fit columns index'].unique():
        mask_idx_match = combined_df['fit columns index'] == idx
        for c in auc_ks_df['C value'].unique():
            mask_c_match = combined_df['C value'] == c
            mask_match = mask_idx_match & mask_c_match # 筛选器
            
            coef_df = combined_df.loc[mask_match, colnames].dropna(axis=1)
            coef_std_series = coef_df.std(axis=0, ddof=0)
            coef_sign_df = np.sign(coef_df)
            coef_sign_stability_series = coef_sign_df.mean(axis=0)
            
            # 指标系数不稳定的筛选器
            mask_unstable = \
                (coef_sign_stability_series.abs() < unstable_threshold) | (
                (coef_sign_stability_series.abs() == unstable_threshold) & \
                (coef_std_series > max_std)
            ) # 若一个指标符号系数稳定性太低，或者稳定性一般但系数标准差过高，则视为指标系数不稳定
            # 若有系数符号不稳定者，则记录于结果表中
            if mask_unstable.any():
                colnames_unstable = coef_sign_stability_series[mask_unstable].index.tolist()
                for colname in colnames_unstable:
                    coef_std = coef_std_series[colname]
                    coef_sign_stability = coef_sign_stability_series[colname]
                    # 将系数不稳定者的相关信息填入结果表
                    row_info_series = pd.Series(
                        [idx, c, colname, coef_sign_stability, coef_std],
                        index=['fit columns index', 'C value', 'colname', 'sign stabilty', 'std']
                    )
                    coef_sign_stability_df.loc[len(coef_sign_stability_df)] = row_info_series
    
    # 为结果表排序
    auc_ks_df = auc_ks_df.sort_values(
        ['valid_AUC_mean', 'valid_KS_mean'], ascending=False
    ).reset_index(drop=True)
    coef_sign_stability_df = coef_sign_stability_df.sort_values(['fit columns index', 'C value'])
    
    return auc_ks_df, coef_sign_stability_df

# 检测舍弃不同指标后的VIF
def test_vif_after_drop(
    X: pd.DataFrame, colnames_to_drop_list: list[list[str]], 
    severe_vif_value: float=20, high_vif_value: float=10
) -> pd.DataFrame:
    """
    检测舍弃不同指标后的VIF
    
    参数
    ----------
    X : pd.DataFrame
        特征矩阵
    colnames_to_drop_list : list[list[str]]
        需要删除的列名组合列表
    severe_vif_value : float, default=20
        严重高VIF阈值
    high_vif_value : float, default=10
        较高VIF阈值
    
    返回
    ----------
    pd.DataFrame
        包含 drop colnames index, VIF>severe_vif_value count, 
        high_vif_value<VIF<=severe_vif_value count, max VIF, max VIF colname 列的结果表
    """
	
    X_copy = X.copy()
    vif_test_df = pd.DataFrame(columns=[
        'drop colnames index',
        f'VIF>{severe_vif_value} count', f'{high_vif_value}<VIF<={severe_vif_value} count',
        'max VIF', 'max VIF colname'
    ])
    for idx, colnames_to_drop in enumerate(colnames_to_drop_list):
        X = X_copy.drop(colnames_to_drop, axis=1, inplace=False)
        
        # 构建指标对应的VIF结果表，求出需要的统计信息
        vif_df = calculate_vif(X)
        
        severe_vif_count = (vif_df['VIF'] > severe_vif_value).sum() # VIF极端高的指标数
        high_vif_count = \
            ((vif_df['VIF'] <= severe_vif_value) & (vif_df['VIF'] > high_vif_value)).sum()
        max_vif = vif_df['VIF'].max()
        max_vif_colname = vif_df.loc[vif_df['VIF'].idxmax(), 'colname']
        
        # 整合统计信息，加至汇总表
        row_info_series = pd.Series(
            [idx, severe_vif_count, high_vif_count, max_vif, max_vif_colname],
            index=vif_test_df.columns
        )
        vif_test_df.loc[len(vif_test_df)] = row_info_series
    
    return vif_test_df

# 检测选定最佳的拟合指标与参数C后，再舍去指定指标的前后LR拟合效果对比
def test_fit_goodness_lr_after_drop(
    X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series,
    colnames_to_fit: list[str], c: float, colnames_to_drop: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    检测选定最佳的拟合指标与参数C后，再舍去指定指标的前后LR拟合效果对比
    
    参数
    ----------
    X_train : pd.DataFrame
        训练集特征
    y_train : pd.Series
        训练集标签
    X_valid : pd.DataFrame
        验证集特征
    y_valid : pd.Series
        验证集标签
    colnames_to_fit : list[str]
        保留的特征列名
    c : float
        正则化参数C
    colnames_to_drop : list[str]
        需要删除的列名
    
    返回
    ----------
    tuple[pd.DataFrame, pd.DataFrame]
        - lr_fit_goodness_after_drop_test_df : 保留与删除后的拟合效果对比表
        - 第二个返回值为 None（保留接口一致性）
    """
	
    X_train_retained = X_train[colnames_to_fit]; X_valid_retained = X_valid[colnames_to_fit]
    X_train_dropped = X_train.drop(colnames_to_drop, axis=1, inplace=False)
    X_valid_dropped = X_valid.drop(colnames_to_drop, axis=1, inplace=False)
    
    lr_retained = LogisticRegression(
        penalty='l2',
        C=c,
        solver='lbfgs',
        max_iter=1000,
        fit_intercept=True
    ) # 设置LR参数
    lr_retained.fit(X_train_retained, y_train) # 拟合模型
    y_prob_train_retained = lr_retained.predict_proba(X_train_retained)[: , 1] # 训练集预测概率
    y_prob_valid_retained = lr_retained.predict_proba(X_valid_retained)[: , 1] # 验证集预测概率
    train_auc_retained, train_ks_retained = calculate_auc_ks(y_train, y_prob_train_retained) # 求出AUC与KS
    valid_auc_retained, valid_ks_retained = calculate_auc_ks(y_valid, y_prob_valid_retained) # 求出AUC与KS
    auc_gap_retained = train_auc_retained - valid_auc_retained
    ks_gap_retained = train_ks_retained - valid_ks_retained
    
    lr_dropped = LogisticRegression(
        penalty='l2',
        C=c,
        solver='lbfgs',
        max_iter=1000,
        fit_intercept=True
    ) # 设置LR参数
    lr_dropped.fit(X_train_dropped, y_train) # 拟合模型
    y_prob_train_dropped = lr_dropped.predict_proba(X_train_dropped)[: , 1] # 训练集预测概率
    y_prob_valid_dropped = lr_dropped.predict_proba(X_valid_dropped)[: , 1] # 验证集预测概率
    train_auc_dropped, train_ks_dropped = calculate_auc_ks(y_train, y_prob_train_dropped) # 求出AUC与KS
    valid_auc_dropped, valid_ks_dropped = calculate_auc_ks(y_valid, y_prob_valid_dropped) # 求出AUC与KS
    auc_gap_dropped = train_auc_dropped - valid_auc_dropped
    ks_gap_dropped = train_ks_dropped - valid_ks_dropped
    
    lr_fit_goodness_after_drop_test_df = pd.DataFrame({
        'status': ['retained', 'dropped', 'difference'], 
        'valid AUC': [valid_auc_retained, valid_auc_dropped, valid_auc_retained - valid_auc_dropped],
        'AUC gap': [auc_gap_retained, auc_gap_dropped, auc_gap_retained - auc_gap_dropped],
        'valid KS': [valid_ks_retained, valid_ks_dropped, valid_ks_retained - valid_ks_dropped],
        'KS gap': [ks_gap_retained, ks_gap_dropped, ks_gap_retained - ks_gap_dropped],
    })
    return lr_fit_goodness_after_drop_test_df

# 筛选XGBoost的拟合指标，记录不同拟合指标条件下模型的AUC、KS
def grid_search_colnames_xgb(
    X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series,
    colnames_to_fit_list: list[list[str]]
) -> pd.DataFrame:
    # 初始的参数字典
    para_dict = dict(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=1,
    )
    
    xgb_fit_goodness_df = pd.DataFrame(columns=[
        'fit columns index', 
        'train AUC', 'valid AUC', 'AUC gap', 'train KS', 'valid KS', 'KS gap', 'best iteration'
    ] + X_train.columns.tolist()) # 初始化结果储存表
    X_train_copy = X_train.copy(); X_valid_copy = X_valid.copy()
    
    for idx, colnames_to_fit in enumerate(colnames_to_fit_list):
        X_train = X_train_copy; X_valid = X_valid_copy
        X_train = X_train[colnames_to_fit]; X_valid = X_valid[colnames_to_fit]
        
        xgb_model = xgb.XGBClassifier(
            **para_dict,
            random_state=600,
            eval_metric='auc',
            early_stopping_rounds=20,
            use_label_encoder=False
        ) # 初始化模型
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        ) # 拟合模型
        y_prob_train = xgb_model.predict_proba(X_train)[: , 1] # 训练集预测概率
        y_prob_valid = xgb_model.predict_proba(X_valid)[: , 1] # 验证集预测概率
        
        # 求出AUC与KS
        auc_train, ks_train = calculate_auc_ks(y_train, y_prob_train)
        auc_valid, ks_valid = calculate_auc_ks(y_valid, y_prob_valid)
        
        row_dict = {
            'fit columns index':int(idx),
            'train AUC': auc_train, 'valid AUC': auc_valid, 'AUC gap': auc_train - auc_valid,
            'train KS': ks_train, 'valid KS': ks_valid, 'KS gap': ks_train - ks_valid,
            'best iteration': xgb_model.best_iteration
        }
        row_series = pd.Series(row_dict)
        xgb_fit_goodness_df.loc[len(xgb_fit_goodness_df)] = row_series
    
    return xgb_fit_goodness_df

# 进阶筛选XGBoost的拟合指标，记录不同拟合指标条件下模型的top 5% Recall等评价指标
def grid_search_colnames_upgraded_xgb(
    X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series,
    base_colnames: list[str], colnames_to_try: list[str],
    min_shap_importance_rank: int=10, top_recall_proportions: list[float]=[0.05, 0.1, 0.2]
) -> pd.DataFrame:
    # 初始的参数字典
    para_dict = dict(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=1,
    )
    
    # 构建包含所有指标的XGBoost
    xgb_model_all = xgb.XGBClassifier(
        **para_dict,
        random_state=700,
        eval_metric='auc',
        early_stopping_rounds=20,
        use_label_encoder=False
    ) # 初始化模型
    colnames_to_fit_all = base_colnames + colnames_to_try # 合并需要拟合的所有指标
    xgb_model_all.fit(
        X_train[colnames_to_fit_all], y_train,
        eval_set=[(X_valid[colnames_to_fit_all], y_valid)],
        verbose=False
    ) # 拟合模型
    
    # 只考虑保留SHAP importance排名高的指标
    shap_df = calculate_shap_importance(xgb_model_all, X_valid[colnames_to_fit_all])
    top_colnames = shap_df.loc[0: min_shap_importance_rank, 'feature']
    colnames_to_consider = [colname for colname in top_colnames if colname in colnames_to_try]
    
    # 构建只有基础固定指标参与拟合的XGBoost，后称基础模型
    xgb_model_base = xgb.XGBClassifier(
        **para_dict,
        random_state=700,
        eval_metric='auc',
        early_stopping_rounds=20,
        use_label_encoder=False
    ) # 初始化模型
    xgb_model_base.fit(
        X_train[base_colnames], y_train,
        eval_set=[(X_valid[base_colnames], y_valid)],
        verbose=False
    ) # 拟合模型
    # 计算当前情形下前5%、10%、20%的Recall
    y_prob_valid_base = xgb_model_base.predict_proba(X_valid[base_colnames])[: , 1]
    top_recall_base_list = []
    for prop in top_recall_proportions:
        top_recall_base = calculate_top_recall(y_valid, y_prob_valid_base, top_proportion=prop)
        top_recall_base_list.append(top_recall_base)
    
    top_recall_names = []; top_recall_gap_names = []
    for prop in top_recall_proportions:
        top_recall_names.append(f'top {prop:.0%} recall')
        top_recall_gap_names.append(f'top {prop:.0%} recall gap')
    top_recall_df = pd.DataFrame(
        columns=['added feature', 'appear time'] + top_recall_names + top_recall_gap_names
    ) # 初始化结果表
    for colname in colnames_to_consider:
        colnames_to_fit = base_colnames + [colname]
        xgb_model = xgb.XGBClassifier(
            **para_dict,
            random_state=700,
            eval_metric='auc',
            early_stopping_rounds=20,
            use_label_encoder=False
        ) # 初始化模型
        xgb_model.fit(
            X_train[colnames_to_fit], y_train,
            eval_set=[(X_valid[colnames_to_fit], y_valid)],
            verbose=False
        ) # 拟合模型
        
        # 计算前5%、10%、20%的Recall
        y_prob_valid = xgb_model.predict_proba(X_valid[colnames_to_fit])[: , 1]
        top_recall_list = []; top_recall_gap_list = []
        for i, prop in enumerate(top_recall_proportions):
            top_recall = calculate_top_recall(y_valid, y_prob_valid, top_proportion=prop)
            top_recall_gap = top_recall - top_recall_base_list[i]
            top_recall_list.append(top_recall); top_recall_gap_list.append(top_recall_gap)
        
        row_info_series = pd.Series(
            [colname, 1] + top_recall_list + top_recall_gap_list,
            index=top_recall_df.columns
        )
        top_recall_df.loc[len(top_recall_df)] = row_info_series
    
    return top_recall_df

# 


# 对XGBoost的参数做网格搜索，记录不同模型参数条件下模型的AUC、KS
def grid_search_paras_xgb(
    X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series,
    grid_search_paras: list[str], para_value_list1: list[float], para_value_list2: list[float],
    best_para_dict: dict[str, float | int]=None
) -> pd.DataFrame:
    """
    对XGB做一轮网格搜索，记录不同参数下的模型拟合效果
    
    参数
    ----------
    X_train : pd.DataFrame
        训练集特征
    y_train : pd.Series
        训练集标签
    X_valid : pd.DataFrame
        验证集特征
    y_valid : pd.Series
        验证集标签
    grid_search_paras : list[str]
        需要网格搜索的两个参数名，如 ['max_depth', 'min_child_weight']
    para_value_list1 : list[float]
        第一个参数的候选值列表
    para_value_list2 : list[float]
        第二个参数的候选值列表
    best_para_dict : dict, default=None
        已经确定的最佳参数（除网格搜索的两个参数外），如 {'learning_rate': 0.05}
    
    返回
    ----------
    pd.DataFrame
        网格搜索结果表，包含两个参数、train AUC、valid AUC、AUC gap、train KS、valid KS、KS gap、best iteration
    """
	
    # 提取网格搜索的一对参数名
    para_name1, para_name2 = grid_search_paras
    # 初始的参数字典
    para_dict = dict(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=1,
    )
    
    # 防止输入的模型参数名不匹配
    if not all([para in para_dict.keys() for para in grid_search_paras]):
        raise ValueError("In function grid_search_paras_xgb: \
            a parameter from grid_search_paras are not a parameter of XGBoost.")
    if best_para_dict and not all([para in para_dict.keys() for para in best_para_dict.keys()]):
        raise ValueError("In function grid_search_paras_xgb: \
            a parameter from best_para_dict are not a parameter of XGBoost.")
    
    # 若有已经固定下来的最佳参数传入，则使用它们
    if best_para_dict is not None:
        para_dict.update(best_para_dict)
    
    # 初始化结果表
    eval_metrics = ['train AUC', 'valid AUC', 'AUC gap', 'train KS', 'valid KS', 'KS gap', 'best iteration']
    xgb_fit_goodness_df = pd.DataFrame(columns=grid_search_paras + eval_metrics)
    
    for para_value1 in para_value_list1:
        para_dict[para_name1] = para_value1
        for para_value2 in para_value_list2:
            para_dict[para_name2] = para_value2
            
            # 初始化XGB，固定除将要网格搜索的两个参数外的其余参数
            xgb_model = xgb.XGBClassifier(
                **para_dict,
                random_state=500,
                eval_metric='auc',
                early_stopping_rounds=20,
                use_label_encoder=False
            )
            # 拟合XGB
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False
            )
            
            # 预测概率
            y_prob_train = xgb_model.predict_proba(X_train)[:, 1]
            y_prob_valid = xgb_model.predict_proba(X_valid)[:, 1]
            # 计算指标
            auc_train, ks_train = calculate_auc_ks(y_train, y_prob_train)
            auc_valid, ks_valid = calculate_auc_ks(y_valid, y_prob_valid)
            # 记录结果
            row_dict = {
                para_name1: para_value1,
                para_name2: para_value2,
                'train AUC': auc_train,
                'valid AUC': auc_valid,
                'AUC gap': auc_train - auc_valid,
                'train KS': ks_train,
                'valid KS': ks_valid,
                'KS gap': ks_train - ks_valid,
                'best iteration': xgb_model.best_iteration
            }
            
            row_series = pd.Series(row_dict)
            xgb_fit_goodness_df.loc[len(xgb_fit_goodness_df)] = row_series
    
    return xgb_fit_goodness_df

# 整理五折交叉验证后XGBoost的一轮网格搜索拟合效果
def test_fit_goodness_xgb(xgb_fit_goodness_df_list: list[pd.DataFrame]) -> pd.DataFrame:
    
    
    # 将五折交叉验证得到的五个结果表合并
    combined_df = pd.concat(xgb_fit_goodness_df_list, ignore_index=True)
    # 提取本轮做网格搜索的两个参数名
    if 'fit columns index' in combined_df.columns:
        groupby_colnames = 'fit columns index'
    else:
        groupby_colnames = combined_df.columns.tolist()[0: 2]
    
    auc_ks_df = combined_df.groupby(groupby_colnames).agg(
        valid_AUC_mean=('valid AUC', 'mean'),
        AUC_gap_mean=('AUC gap', 'mean'),
        valid_KS_mean=('valid KS', 'mean'),
        KS_gap_mean=('KS gap', 'mean'),
        best_iteration_mean=('best iteration', 'mean')
    ).reset_index()
    
    # 按验证集表现排序
    auc_ks_df = auc_ks_df.sort_values(
        ['valid_AUC_mean', 'valid_KS_mean'], ascending=False
    ).reset_index(drop=True)
    return auc_ks_df

# 整理五折交叉验证后不同拟合指标条件下的拟合效果，精确筛选可以保留的指标
def test_fit_goodness_upgraded_xgb(
    xgb_fit_goodness_df_list: list[pd.DataFrame]
):
    combined_df = pd.concat(xgb_fit_goodness_df_list, ignore_index=True) # 合并五折交叉验证得到的五个结果表
    top_recall_aggregate_df = combined_df.groupby('added feature').agg(
        appear_count = ('appear time', 'sum'),
        top5_recall_mean = (f'top 5% recall', 'mean'),
        top10_recall_mean = (f'top 10% recall', 'mean'),
        top20_recall_mean = (f'top 20% recall', 'mean'),
        top5_recall_gap_mean = (f'top 5% recall gap', 'mean'),
        top10_recall_gap_mean = (f'top 10% recall gap', 'mean'),
        top20_recall_gap_mean = (f'top 20% recall gap', 'mean'),
    ) # 整理数据，得到指标出现次数与模型评价指标的均值
    
    # 只保留在每一折均出现的指标
    mask_passed = top_recall_aggregate_df['appear_count'] == 5
    top_recall_aggregate_df = top_recall_aggregate_df.loc[mask_passed].drop('appear_count', axis=1)
    
    # 
    return top_recall_aggregate_df

# 横向比较各模型的拟合效果，计算各模型评价指标AUC、KS
def quantify_model_comparison(
    y_train: pd.Series, y_test: pd.Series,
    X_train_raw_lr: pd.DataFrame, X_test_raw_lr: pd.DataFrame,
    X_train_woe_lr: pd.DataFrame, X_test_woe_lr: pd.DataFrame,
    X_train_xgb: pd.DataFrame, X_test_xgb: pd.DataFrame,
    raw_lr_model: LogisticRegression,
    sc_model: ScoreCard,
    xgb_model: xgb.XGBClassifier
) -> pd.DataFrame:
    """
    在三种模型确定了参数后横向比较各模型的拟合效果
    
    参数
    ----------
    y_train : pd.Series
        训练集标签
    y_test : pd.Series
        测试集标签
    X_train_raw_lr : pd.DataFrame
        原始LR训练集特征
    X_test_raw_lr : pd.DataFrame
        原始LR测试集特征
    X_train_woe_lr : pd.DataFrame
        WOE LR训练集特征
    X_test_woe_lr : pd.DataFrame
        WOE LR测试集特征
    X_train_xgb : pd.DataFrame
        XGBoost训练集特征
    X_test_xgb : pd.DataFrame
        XGBoost测试集特征
    raw_lr_model : LogisticRegression
        训练好的原始LR模型
    sc_model : ScoreCard
        训练好的评分卡模型
    xgb_model : xgb.XGBClassifier
        训练好的XGBoost模型
    
    返回
    ----------
    pd.DataFrame
        模型对比结果表，包含 model, test AUC, AUC gap, test KS, KS gap
    """
	
    # raw LR的AUC与KS
    y_prob_train_raw_lr = raw_lr_model.predict_proba(X_train_raw_lr)[: , 1] # 训练集预测概率
    y_prob_test_raw_lr = raw_lr_model.predict_proba(X_test_raw_lr)[: , 1] # 测试集预测概率
    auc_train_raw_lr, ks_train_raw_lr = calculate_auc_ks(y_train, y_prob_train_raw_lr)
    auc_test_raw_lr, ks_test_raw_lr = calculate_auc_ks(y_test, y_prob_test_raw_lr)
    auc_gap_raw_lr = auc_train_raw_lr - auc_test_raw_lr
    ks_gap_raw_lr = ks_train_raw_lr - ks_test_raw_lr
    # WOE LR的AUC与KS（KS为基于评分的KS）
    auc_train_woe_lr = sc_model.get_auc(X_train_woe_lr, y_train)
    ks_train_score_based = sc_model.get_ks_score_based(X_train_woe_lr, y_train)
    auc_test_woe_lr = sc_model.get_auc(X_test_woe_lr, y_test)
    ks_test_score_based = sc_model.get_ks_score_based(X_test_woe_lr, y_test)
    auc_gap_woe_lr = auc_train_woe_lr - auc_test_woe_lr
    ks_gap_score_based = ks_train_score_based - ks_test_score_based
    # XGBoost的AUC与KS
    y_prob_train_xgb = xgb_model.predict_proba(X_train_xgb)[: , 1] # 训练集预测概率
    y_prob_test_xgb = xgb_model.predict_proba(X_test_xgb)[: , 1] # 测试集预测概率
    auc_train_xgb, ks_train_xgb = calculate_auc_ks(y_train, y_prob_train_xgb)
    auc_test_xgb, ks_test_xgb = calculate_auc_ks(y_test, y_prob_test_xgb)
    auc_gap_xgb = auc_train_xgb - auc_test_xgb
    ks_gap_xgb = ks_train_xgb - ks_test_xgb
    
    model_comparison_df = pd.DataFrame({
        'model': ['raw LR', 'WOE LR', 'XGBoost'],
        'test AUC': [auc_test_raw_lr, auc_test_woe_lr, auc_test_xgb],
        'AUC gap': [auc_gap_raw_lr, auc_gap_woe_lr, auc_gap_xgb],
        'test KS': [ks_test_raw_lr, ks_test_score_based, ks_test_xgb],
        'KS gap': [ks_gap_raw_lr, ks_gap_score_based, ks_gap_xgb]
    })
    
    return model_comparison_df





""" 填充月收入的相关函数[已弃用] """
# 用二维分箱的方法填充不可靠的MonthlyIncome
def fill_income_bin(
    data: pd.DataFrame, colnames: list, min_count: int=20, is_return_value: bool=False
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.Series:
    """
    用二维分箱的方法填充不可靠的MonthlyIncome
    
    三级填充策略
    ----------
    二维分箱中位数 → 一维分箱中位数 → 全局中位数
    
    参数
    ----------
    data : pd.DataFrame
        数据集（训练集或验证集），需已包含分箱列和目标变量
    colnames : list
        两个指标列名，如 [age, 'credit']
    min_count : int, default=20
        每组最小样本量阈值，低于此值的组的中位数记为NaN，不用于填充
    is_return_value : bool, default=False
        为True时返回填充值序列而非填充后的数据集
    
    返回
    ----------
    tuple | pd.Series
        - 若is_return_value=False，返回 (data_bin, median_2bin_df)
        - 若is_return_value=True，返回 fill_values
    """
	
    if len(colnames) != 2:
        raise ValueError("In function fill_income_bin: colnames must be at length 2.")
    
    data_bin = data.copy()
    target = income
    
    colname1, colname2 = colnames
    # 获取分箱列名
    binbase1 = colnames_abbr_map.get(colname1, colname1)
    binname1 = binbase1 + '_bin'
    binbase2 = colnames_abbr_map.get(colname2, colname2)
    binname2 = binbase2 + '_bin'
    
    mask_reliable = (data_bin[target] > 10) & (~data_bin[target].isna()) # 可靠样本筛选器
    mask_unreliable = ~mask_reliable # 不可靠样本筛选器
    
    # 计算三级分箱的中位数信息
    reliable_data_selected = data_bin.loc[mask_reliable, [target, binname1, binname2]]
    reliable_median_2bin_df, reliable_median_1bin_df, reliable_median_whole = \
        calculate_lv3_medians(reliable_data_selected, min_count)
        
    # 三级填充不可靠样本的月收入
    unreliable_data_selected = data_bin.loc[mask_unreliable, [target, binname1, binname2]]
    fill_values = predict_income_bin(
        unreliable_data_selected, 
        reliable_median_2bin_df, reliable_median_1bin_df, reliable_median_whole
    )
    
    if not is_return_value:
        data_bin.loc[mask_unreliable, target] = fill_values
        return data_bin, reliable_median_2bin_df
    else:
        fill_values = data_bin.loc[mask_unreliable, target].copy()
        return fill_values

# 由精简且准确分割的训练集输出二维分箱中位数DataFrame，一维分箱中位数Series和全局中位数
def calculate_lv3_medians(selected_data: pd.DataFrame, min_count: int)\
    -> tuple[pd.DataFrame, pd.Series, float]:
    """
    由精简且准确分割的训练集输出二维分箱中位数DataFrame，一维分箱中位数Series和全局中位数
    
    参数
    ----------
    selected_data : pd.DataFrame
        只包含三列的数据集：[目标列, 第一分箱列, 第二分箱列]
    min_count : int
        每组最小样本量阈值，低于此值的组的中位数记为NaN
    
    返回
    ----------
    tuple
        - reliable_median_2bin_df : pd.DataFrame
            二维分箱中位数表，行=第一分箱，列=第二分箱
        - reliable_median_1bin_df : pd.Series
            一维分箱中位数序列，索引=第一分箱
        - reliable_median_whole : float
            全局中位数
    """
	
    colnames = selected_data.columns.tolist()
    if len(colnames) != 3:
        raise ValueError("In function build_lv3_medians: selected data requires DataFrame with 3 columns.")
    
    target, binname1, binname2 = colnames
    bin1, bin2 = sorted(list(set(selected_data[binname1]))), sorted(list(set(selected_data[binname2]))) # 从小到大的分箱种类
    
    # 构造二维分箱的可靠样本中位数DataFrame
    reliable_median_2bin_df = pd.DataFrame(index=bin1, columns=bin2)
    for itvl1 in bin1:
        for itvl2 in bin2:
            mask1, mask2 = selected_data[binname1] == itvl1, selected_data[binname2] == itvl2
            mask = mask1 & mask2
            if mask.sum() >= min_count:
                reliable_median_2bin_df.loc[itvl1, itvl2] = selected_data.loc[mask, target].median()
            else:
                reliable_median_2bin_df.loc[itvl1, itvl2] = np.nan
    # 构建一维分箱的可靠样本中位数Series
    reliable_median_1bin_df = pd.Series(index=bin1)
    for itvl1 in bin1:
        mask1= selected_data[binname1] == itvl1
        mask = mask1
        if mask.sum() >= min_count:
            reliable_median_1bin_df[itvl1] = selected_data.loc[mask, target].median()
        else:
            reliable_median_1bin_df[itvl1] = np.nan
    # 计算全局可靠样本中位数
    reliable_median_whole = selected_data[target].median()
    
    return reliable_median_2bin_df, reliable_median_1bin_df, reliable_median_whole

# 由计算完毕的三级中位数信息预测精简且准确分割的验证/测试集的目标变量
def predict_income_bin(
    selected_data: pd.DataFrame, 
    median_2bin_df: pd.DataFrame, median_1bin_df: pd.Series, median_whole: float
) -> pd.Series:
    """
    由计算完毕的三级中位数信息预测精简且准确分割的验证/测试集的目标变量
    
    参数
    ----------
    selected_data : pd.DataFrame
        只包含三列的数据集：[目标列, 第一分箱列, 第二分箱列]（目标列值可为空或原始值）
    median_2bin_df : pd.DataFrame
        二维分箱中位数表
    median_1bin_df : pd.Series
        一维分箱中位数序列
    median_whole : float
        全局中位数
    
    返回
    ----------
    pd.Series
        填充后的目标变量值序列，索引与selected_data一致
    """
	
    
    colnames = selected_data.columns.tolist()
    if len(colnames) != 3:
        raise ValueError("In function predict_by_bin: selected data requires DataFrame with 3 columns.")
    
    selected_data = selected_data.copy()
    target, bin_name1, bin_name2 = colnames
    bin1, bin2 = sorted(list(set(selected_data[bin_name1]))), sorted(list(set(selected_data[bin_name2]))) # 从小到大的分箱种类
    
    for itvl1 in bin1:
        mask1 = selected_data[bin_name1] == itvl1
        median_1bin = median_1bin_df[itvl1]
        
        for itvl2 in bin2:
            mask2 = selected_data[bin_name2] == itvl2
            median_2bin = median_2bin_df.loc[itvl1, itvl2]
            
            mask_to_fill = mask1 & mask2
            if pd.notna(median_2bin):
                selected_data.loc[mask_to_fill, target] = median_2bin
            elif pd.notna(median_1bin):
                selected_data.loc[mask_to_fill, target] = median_1bin
            else:
                selected_data.loc[mask_to_fill, target] = median_whole
    
    fill_values = selected_data[target].copy()
    return fill_values

# 用随机森林的方法填充不可靠的MonthlyIncome
def fill_income_rf(
    data: pd.DataFrame, X_colnames: list=None, 
    n_estimators: int=100, max_depth: int=10, seed: int=1, 
    is_return_regressor_only: bool=False, is_return_value: bool=False
) -> tuple[pd.DataFrame, pd.DataFrame] | RandomForestRegressor | np.ndarray[float]:
    """
    用随机森林的方法填充不可靠的MonthlyIncome
    
    参数
    ----------
    data : pd.DataFrame
        数据集（训练集或验证集），需包含月收入列和指定的特征列
    X_colnames : list, default=None
        用于预测月收入的特征列名列表
    n_estimators : int, default=100
        随机森林的树数量
    max_depth : int, default=10
        随机森林的最大深度
    seed : int, default=1
        随机种子，确保结果可复现
    is_return_regressor_only : bool, default=False
        为True时不做填充，在随机森林模型训练完毕后立刻输出它
    is_return_value : bool, default=False
        为True时返回填充值序列而非填充后的数据集
    
    返回
    ----------
    tuple | RandomForestRegressor | np.ndarray
        - 若is_return_regressor_only=True，返回 rf_model
        - 若is_return_value=False，返回 (data_rf_copy, rf_model)
        - 若is_return_value=True，返回 y_pred
    """
    
    target = income
    if X_colnames is None:
        X_colnames = [
            age, 
            'credit',
            'mortgage',
            dep,
            'both_missing_flag',
            late30,
            late60,
            late90,
            'blacklist_flag'
        ]
    
    data_rf = data.copy()
    
    # 1. 家属数缺失值赋值为0
    data_rf.loc[data_rf['both_missing_flag'] == 1, dep] = 0
    # 2. 逾期次数96/98赋值为0
    late_colnames = [
        late30, 
        late60,
        late90
    ]
    for colname in late_colnames:
        data_rf.loc[data_rf['blacklist_flag'] == 1, colname] = 0
    
    # 可靠/不可靠样本筛选器
    mask_reliable = (data_rf[target] > 10) & (~data_rf[target].isna())
    mask_unreliable = ~mask_reliable
    
    # 对月收入取对数，处理长尾分布
    data_rf['MonthlyIncome_log'] = np.log1p(data_rf[target])
    
    # 训练随机森林
    X_train = data_rf.loc[mask_reliable, X_colnames]
    y_train_log = data_rf.loc[mask_reliable, 'MonthlyIncome_log']
    
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed
    )
    rf_model.fit(X_train, y_train_log)
    
    if is_return_regressor_only: # 提前输出随机森林模型
        return rf_model
    
    # 做出预测（预测的是对数后的值）
    X_pred = data_rf.loc[mask_unreliable, X_colnames]
    y_pred_log = rf_model.predict(X_pred)
    y_pred = np.expm1(y_pred_log) # 还原为原始收入尺度
    
    if not is_return_value:
        data_rf_copy = data.copy()  # 从原始data复制，保持所有列不变
        data_rf_copy.loc[mask_unreliable, target] = y_pred # 填充不可靠月收入
        return data_rf_copy, rf_model
    else:
        return y_pred

# 在可靠样本中随机屏蔽20%样本的月收入，分别用二维分箱和随机森林由剩余的80%做出预测，计算MSE等误差指标
def compare_fill_effect(
    data: pd.DataFrame, bin_colnames: list[str], rf_X_colnames: list[str]=None, 
    mask_ratio: float=0.2, split_seed: int=1, 
    min_count: int=20, 
    n_estimators: int=100, max_depth: int=10, rf_seed: int=2, 
    is_show_visualization: bool=True
) -> pd.DataFrame:
    """
    在可靠样本中随机屏蔽20%样本的月收入，分别用二维分箱和随机森林由剩余的80%做出预测，计算MSE等误差指标
    
    参数
    ----------
    data : pd.DataFrame
        包含月收入列和分箱列的原始数据集
    bin_colnames : list
        二维分箱用的两个特征列名，如 [age, 'credit']
    rf_X_colnames : list, default=None
        随机森林用的特征列名列表，为None时使用默认特征集
    mask_ratio : float, default=0.2
        测试集占比，即被掩盖的健康样本比例
    split_seed : int, default=1
        训练/测试集划分的随机种子
    min_count : int, default=20
        二维分箱每组最小样本量阈值
    n_estimators : int, default=100
        随机森林的树数量
    max_depth : int, default=10
        随机森林的最大深度
    rf_seed : int, default=2
        随机森林模型的随机种子
    is_show_visualization : bool, default=True
        为True时，展示预测效果可视化图
    
    返回
    ----------
    pd.DataFrame
        包含两种填充方法的MSE、MAE、MAPE、sMAPE指标的结果表
    """
	
    if len(bin_colnames) != 2:
        raise ValueError("In function compare_fill_effect: bin_colnames must be at length 2.")
    
    if rf_X_colnames is None:
        rf_X_colnames = [
            age, credit, mortgage,
            dep, 'both_missing_flag',
            late30, late60, late90, 'blacklist_flag'
        ]
    colname1, colname2 = bin_colnames
    # 还原分箱列名
    binbase1 = colnames_abbr_map.get(colname1, colname1)
    binname1 = binbase1 + '_bin'
    binbase2 = colnames_abbr_map.get(colname2, colname2)
    binname2 = binbase2 + '_bin'
    
    # 筛选健康样本（月收入>10且非缺失）
    mask_reliable = (data[income] > 10) & (~data[income].isna())
    reliable_data = data[mask_reliable].copy()
    
    # 将健康样本分为训练集（80%）和测试集（20%）
    train_reliable, test_reliable = train_test_split(
        reliable_data, 
        test_size=mask_ratio, 
        random_state=split_seed
    )
    
    selected_train_reliable = train_reliable[[income, binname1, binname2]] # 精简分割需要的训练集
    reliable_median_2bin_df, reliable_median_1bin_df, reliable_median_whole = \
        calculate_lv3_medians(selected_train_reliable, min_count=min_count) # 得到三级分箱的中位数数据
    selected_test_reliable = test_reliable[[income, binname1, binname2]] # 精简分割需要的测试集
    bin_pred_vals = predict_income_bin(
        selected_test_reliable, 
        reliable_median_2bin_df, reliable_median_1bin_df, reliable_median_whole
    ) # 得到二维分箱方法的预测值
    
    rf_model = fill_income_rf(
        train_reliable, rf_X_colnames, is_return_regressor_only=True,
        n_estimators=n_estimators, max_depth=max_depth, seed=rf_seed
    ) # 得到由训练集训练好的随机森林模型
    # 做出预测（预测的是对数后的值）
    X_pred = test_reliable[rf_X_colnames].copy()
    y_pred_log = rf_model.predict(X_pred)
    rf_pred_vals = np.expm1(y_pred_log) # 还原为原始收入尺度
    
    # 计算误差
    real_vals = test_reliable[income].values
    bin_mse, bin_mae, bin_mape, bin_smape = analyze_residual(real_vals, bin_pred_vals)
    rf_mse, rf_mae, rf_mape, rf_smape = analyze_residual(real_vals, rf_pred_vals)
    
    # 返回结果
    residual_df = pd.DataFrame({
        '填充方法': ['二维分箱', '随机森林'],
        'MSE': [bin_mse, rf_mse],
        'MAE': [bin_mae, rf_mae],
        "MAPE": [bin_mape, rf_mape],
        'sMAPE': [bin_smape, rf_smape]
    })
    
    if is_show_visualization:
        name_to_y_pred = {
            '二维分箱': bin_pred_vals,
            '随机森林': rf_pred_vals
        }
        show_pred_effect_visualization(real_vals, '月收入', name_to_y_pred)
    
    return residual_df
