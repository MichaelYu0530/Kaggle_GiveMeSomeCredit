"""Features functions from the original research pipeline."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from .schema import colnames_abbr_map, credit, debt, dep, income, late30, late60, late90, mortgage, target, util



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
