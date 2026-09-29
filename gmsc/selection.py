"""Selection functions from the original research pipeline."""

from __future__ import annotations

from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import xgboost as xgb
from .interpretability import calculate_shap_importance

from .metrics import calculate_auc_ks, calculate_top_recall

from .schema import colnames_abbr_map, target

from .scorecard import ScoreCard

from .statistics import calculate_vif

from .summaries import summary_bins

from .woe import calculate_psi, calculate_woe_iv



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
    top_colnames = shap_df.iloc[:min_shap_importance_rank]['feature']
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
    # WOE LR的AUC与KS：主比较表统一使用连续预测概率的KS口径
    y_prob_train_woe_lr = sc_model.predict_proba(X_train_woe_lr)[:, 1]
    y_prob_test_woe_lr = sc_model.predict_proba(X_test_woe_lr)[:, 1]
    auc_train_woe_lr, ks_train_woe_lr = calculate_auc_ks(y_train, y_prob_train_woe_lr)
    auc_test_woe_lr, ks_test_woe_lr = calculate_auc_ks(y_test, y_prob_test_woe_lr)
    auc_gap_woe_lr = auc_train_woe_lr - auc_test_woe_lr
    ks_gap_woe_lr = ks_train_woe_lr - ks_test_woe_lr
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
        'test KS': [ks_test_raw_lr, ks_test_woe_lr, ks_test_xgb],
        'KS gap': [ks_gap_raw_lr, ks_gap_woe_lr, ks_gap_xgb]
    })
    
    return model_comparison_df
