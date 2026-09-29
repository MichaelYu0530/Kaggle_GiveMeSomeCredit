"""Legacy income functions from the original research pipeline."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from .plotting_evaluation import show_pred_effect_visualization

from .schema import age, colnames_abbr_map, credit, dep, income, late30, late60, late90, mortgage, target

from .statistics import analyze_residual



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
