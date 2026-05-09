""" 导入库 """
from pipeline import *
from visualization import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display
import warnings

warnings.filterwarnings('ignore') # 忽略警告信息
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False

""" 早期准备 """ 
# region
# 加载原始数据
origin_data = import_data(filetype='train')
# 基础可视化
basic_visualization_fig = show_basic_visualization(origin_data)
# 补充的重要发现
special_discovery_df = show_special_discovery(origin_data)
# endregion
""" 初步数据清洗，针对客观事实，允许在全局数据集进行 """
# region
# 缺失值相关标记
completeness_desc_df, completeness_test_df = add_missing_flag(origin_data)
# 逾期次数96/98相关标记
blacklist_desc_df, blacklist_test_df = add_blacklist_flag(origin_data)
# 月收入异常（极低，0-10）相关标记
zero_vs_low_desc_df, zero_vs_low_test_df, abnormal_vs_normal_desc_df, abnormal_vs_normal_test_df = \
    add_income_anomaly_flag(origin_data)
# endregion
""" 分离测试集 """
# region
random_seed = 50
train_valid_data, test_data = train_test_split(
    origin_data,
    test_size=0.2, 
    random_state=random_seed,
    stratify=origin_data[target]
)
train_valid_data_copy = train_valid_data.copy(); test_data_copy = test_data.copy() # 备份
# endregion

""" 筛选值得添加交互标记的交互项 """
is_processing_here1 = False
if is_processing_here1:
    # region
    valuable_intersections_df_list = []
    # 划分训练集、验证集，准备K=5折交叉验证
    random_seed = 100
    folds = create_kfold(train_valid_data, random_seed, n_splits=5)
    for fold_idx in range(len(folds)):
        # 训练集、验证集拆分
        train_idx_list, _ = folds[fold_idx]
        train_data = train_valid_data.iloc[train_idx_list].copy()
        # endregion
        
        """ 统一数据清洗阶段 """
        # region
        silently = False if fold_idx == 0 else True
        # 截断家属数异常大值
        cap_dep_num(train_data, silently=silently)
        # 删除年龄异常小值
        delete_small_age(train_data, silently=silently)
        # 添加信用额度使用率的若干标记
        add_util_flags(train_data, silently=silently)
        # 添加负债率的若干标记
        add_debt_flags(train_data, silently=silently)
        # 添加三种逾期次数的加权求和，权重由逻辑回归模型训练
        beta30, beta60, beta90 = add_late_severity_score(train_data)
        # 添加衍生指标
        add_derived_features(train_data)
        # 整理指标顺序
        train_data = reorder_columns(train_data)
        # endregion
        
        """ 构建全部的一维分箱、二维分箱，计算每个交互项的评判指标 """
        # region
        # 筛选出所有非二元指标
        colnames_not_binary = get_colnames_whether_binary(train_data, is_binary=False)
        # 对所有非布尔值类型的指标建立一维分箱
        for colname in colnames_not_binary:
            _, div_pts = add_bins_1D(train_data, colname)
        # 把一维分箱的具体数值转换为排名序号
        convert_bins_value_to_rank(train_data)
        # 列出通过筛选的一维分箱对应指标的所有无序对
        colname_pairs_not_binary = []
        for i in range(len(colnames_not_binary) - 1):
                for j in range(i + 1, len(colnames_not_binary)):
                    colname_pairs_not_binary.append([colnames_not_binary[i], colnames_not_binary[j]])
        # 对两两配对的每对指标建立二维分箱
        for colname_pair in colname_pairs_not_binary:
            _, bins_2D = add_bins_2D(train_data, colname_pair)
        # 找出有价值添加交互标记的二维分箱
        valuable_intersections_df = find_valuable_intersections(train_data, colname_pairs_not_binary)
        valuable_intersections_df_list.append(valuable_intersections_df)
        # 筛选交互效应为增强的排名前30的二维分箱与交互效应为削弱的排名前10的二维分箱，再做人工复查：删除或合并
        if fold_idx == len(folds) - 1: # 只在最后一折时做
            top_positive_intersections_df, top_negative_intersections_df = \
                filter_intersections(valuable_intersections_df_list)
        # endregion

""" 列出值得添加的交互标记 """
# region
# 增强效应交互项
positive_interactions_df = pd.DataFrame({
    # 交互标记名称
    'signal_name': [
        # 参与交互的两个一维分箱都是只与逾期次数相关的，后续记为第一组
        'late_severity_mid_x_short_late_low_signal',
        'late_severity_high_x_short_late_low_signal',
        '30-59late_low_x_short_late_mid_high_signal',
        '90+late_mid_high_x_short_late_low_signal',
        # 参与交互的两个一维分箱一个只与逾期次数相关，另一个也涉及逾期次数，后续记为第二组
        '30-59late_low_x_credit_late_density_mid_high_signal',
        'age_high_x_credit_late_density_high_signal',
        'util_low_x_credit_late_density_high_signal',
        'mortgage_ratio_mid_x_credit_late_density_high_signal',
        'mortgage_mid_x_credit_late_density_high_signal',
        # 参与交互的两个一维分箱有且只有一个与逾期次数相关，后续记为第三组
        'dep_missing_x_60-89late_mid_signal',
        'age_high_x_late_severity_high_signal',
        'age_high_x_90+late_mid_signal',
        'util_low_mid_x_late_severity_high_signal',
        'util_mid_x_90+late_mid_signal',
        'age_high_x_60-89late_mid_signal',
        '30-59late_mid_x_credit_mid_signal',
        # 参与交互的两个一维分箱都不与逾期次数相关，后续记为第四组
        'debt_low_x_credit_pressure_mid_signal',
        'debt_mid_x_credit_pressure_high_signal',
        'util_high_x_credit_pressure_low_signal',
        'monthly_debt_low_x_credit_pressure_high_signal',
        'late_severity_high_x_credit_pressure_low_signal'
    ],
    # 参与交互的第一个一维分箱排名序号列名
    'binrank_colname1': [
        # 第一组
        'late_severity_score_binrank', 'late_severity_score_binrank', '30-59late_binrank',
        '90+late_binrank',
        # 第二组
        '30-59late_binrank', 'age_binrank', 'util_binrank',
        'mortgage_ratio_binrank', 'mortgage_binrank',
        # 第三组
        'dep_binrank', 'age_binrank', 'age_binrank', 
        'util_binrank', 'util_binrank', 'age_binrank', '30-59late_binrank',
        # 第四组
        'debt_binrank', 'debt_binrank',
        'util_binrank', 'monthly_debt_binrank',
        'late_severity_score_binrank'
    ],
    # 参与交互的第二个一维分箱排名序号列名
    'binrank_colname2': [
        # 第一组
        'short_late_binrank', 'short_late_binrank', 'short_late_binrank',
        'short_late_binrank',
        # 第二组
        'credit_late_density_binrank', 'credit_late_density_binrank', 'credit_late_density_binrank',
        'credit_late_density_binrank', 'credit_late_density_binrank',
        # 第三组
        '60-89late_binrank', 'late_severity_score_binrank', '90+late_binrank', 
        'late_severity_score_binrank','90+late_binrank', '60-89late_binrank', 'credit_binrank',
        # 第四组
        'credit_pressure_index_binrank', 'credit_pressure_index_binrank', 
        'credit_pressure_index_binrank', 'credit_pressure_index_binrank', 
        'credit_pressure_index_binrank'
    ],
    # 参与交互的第一个一维分箱的排名序号跨度
    'binrank1': [
        # 第一组
        '2_2', '4_4', '1_1',
        '3_4',
        # 第二组
        '1_1', '5_6', '1_1',
        '2_2', '2_2',
        # 第三组
        '-1_-1', '6_6', '6_6', 
        '1_2', '2_2', '6_6', '3_3',
        # 第四组
        '1_1', '2_2', 
        '5_5', '1_1',
        '4_4'
    ],
    # 参与交互的第二个一维分箱的排名序号跨度
    'binrank2': [
        # 第一组
        '1_1', '1_1', '2_4',
        '1_1',
        # 第二组
        '4_6', '6_6', '6_6',
        '6_6', '6_6',
        # 第三组
        '2_2', '4_4', '2_3', 
        '4_4', '2_2', '2_2', '2_2',
        # 第四组
        '3_4', '4_5',
        '1_2', '3_4',
        '1_2'
    ]
})# 削弱效应交互项
# 削弱效应交互项
negative_interactions_df = pd.DataFrame({
    # 交互标记名称
    'signal_name': [
        'credit_low_x_util_mid_signal',
        'monthly_debt_low_x_mortgage_ratio_mid_signal',
        'credit_pressure_mid_x_debt_mid_signal',
        'monthly_debt_high_x_debt_mid_signal',
        'credit_pressure_mid_x_mortgage_mid_signal',
        'monthly_debt_low_x_mortgage_mid_signal',
        'monthly_debt_low_x_age_high_signal',
        'monthly_debt_high_x_credit_pressure_mid_high_signal',
        'monthly_debt_mid_x_credit_pressure_mid_signal',
        'age_high_x_debt_low_signal'
    ],
    # 参与交互的第一个一维分箱排名序号列名
    'binrank_colname1': [
        'credit_binrank', 'monthly_debt_binrank', 'credit_pressure_index_binrank',
        'monthly_debt_binrank', 'credit_pressure_index_binrank', 'monthly_debt_binrank',
        'monthly_debt_binrank', 'monthly_debt_binrank', 'monthly_debt_binrank', 'age_binrank'
    ],
    # 参与交互的第二个一维分箱排名序号列名
    'binrank_colname2': [
        'util_binrank', 'mortgage_ratio_binrank', 'debt_binrank',
        'debt_binrank', 'mortgage_binrank', 'mortgage_binrank',
        'age_binrank', 'credit_pressure_index_binrank', 'credit_pressure_index_binrank', 'debt_binrank'
    ],
    # 参与交互的第一个一维分箱的排名序号跨度
    'binrank1': [
        '1_1', '1_1', '2_2',
        '4_4', '2_2', '1_1',
        '1_1', '4_4', '3_3', '6_6'
    ],
    # 参与交互的第二个一维分箱的排名序号跨度
    'binrank2': [
        '2_2', '2_2', '2_2',
        '2_2', '2_2', '2_2',
        '6_6', '3_3', '2_2', '1_1'
    ]
})
# 释放内存
if 'train_data' in dir():
    del train_data
# endregion

""" 筛选值得保留的标记、一维分箱与二维分箱 """
is_processing_here2 = False
if is_processing_here2:
# region
    binary_columns_test_df_list = []; bins_1D_test_df_list = []; bins_2D_test_df_list = []
    # 划分训练集、验证集，准备K=5折交叉验证
    random_seed = 1000
    folds = create_kfold(train_valid_data, random_seed, n_splits=5)
    for fold_idx in range(len(folds)):
        # 训练集、验证集拆分
        train_idx_list, valid_idx_list = folds[fold_idx]
        train_data = train_valid_data.iloc[train_idx_list].copy()
        valid_data = train_valid_data.iloc[valid_idx_list].copy()
        # endregion
        
        """ 统一数据清洗阶段 """
        # region
        # 截断家属数异常大值
        cap_dep_num(train_data); cap_dep_num(valid_data)
        # 删除年龄异常小值
        delete_small_age(train_data); delete_small_age(valid_data)
        # 添加信用额度使用率的若干标记
        add_util_flags(train_data); add_util_flags(valid_data)
        # 添加负债率的若干标记
        add_debt_flags(train_data); add_debt_flags(valid_data)
        # 添加三种逾期次数的加权求和，权重由逻辑回归模型训练
        beta30, beta60, beta90 = add_late_severity_score(train_data)
        add_late_severity_score(valid_data, [beta30, beta60, beta90])
        # 添加衍生指标
        add_derived_features(train_data); add_derived_features(valid_data)
        # 筛选出所有非二元指标
        colnames_not_binary = get_colnames_whether_binary(train_data, is_binary=False)
        # 对所有非布尔值类型的指标建立一维分箱
        for colname in colnames_not_binary:
            _, div_pts = add_bins_1D(train_data, colname)
            add_bins_1D(valid_data, colname, fixed_div_pts=div_pts)
        # 把一维分箱的具体数值转换为排名序号
        convert_bins_value_to_rank(train_data); convert_bins_value_to_rank(valid_data)
        # 添加交互标记
        add_intersection_signals(train_data, positive_interactions_df, negative_interactions_df)
        add_intersection_signals(valid_data, positive_interactions_df, negative_interactions_df)
        # 整理指标顺序
        train_data = reorder_columns(train_data); valid_data = reorder_columns(valid_data)
        # 展示当前指标名称
        if fold_idx == 0:
            colname_general_df = display_current_colnames_general(train_valid_data, train_data)
        # endregion
        
        """ 整理标记、一维分箱、二维分箱的评判指标 """
        # region
        train_data_woe_lr = train_data.copy()
        # 找出所有二元列
        colnames_binary = get_colnames_whether_binary(train_data_woe_lr, is_binary=True)
        # 计算二元列的风险比率、IV
        binary_columns_test_df = test_binary_columns(train_data_woe_lr, colnames_binary)
        binary_columns_test_df_list.append(binary_columns_test_df)
        # 计算一维分箱的WOE、IV和PSI，做第一轮筛选
        bins_1D_test_df, iv_1D_dict = test_bins_1D(train_data_woe_lr, valid_data, colnames_not_binary)
        bins_1D_test_df_list.append(bins_1D_test_df)
        # 列出所有一维分箱对应指标的所有无序对
        colname_pairs_not_binary = []
        for i in range(len(colnames_not_binary) - 1):
                for j in range(i + 1, len(colnames_not_binary)):
                    colname_pairs_not_binary.append([colnames_not_binary[i], colnames_not_binary[j]])
        # 对两两配对的每对指标建立二维分箱
        # colname_pairs_not_binary = colname_pairs_not_binary[0: 20]
        for colname_pair in colname_pairs_not_binary:
            _, bins_2D = add_bins_2D(train_data_woe_lr, colname_pair)
            add_bins_2D(valid_data, colname_pair, fixed_bins_2D=bins_2D)
        # 计算二维分箱的WOE、IV、IV增量、IS、PSI，做第二轮筛选
        bins_2D_test_df = test_bins_2D(train_data_woe_lr, valid_data, colname_pairs_not_binary, iv_1D_dict)
        bins_2D_test_df_list.append(bins_2D_test_df)
        # 筛选出可以投入训练的标记、一维分箱、二维分箱
        if fold_idx == len(folds) - 1: # 只在最后一折时做
            binary_columns_filter_df = filter_binary_columns(binary_columns_test_df_list)
            bins_1D_filter_df = filter_bins_1D(bins_1D_test_df_list)
            passed_bins_2D_df = filter_bins_2D(bins_2D_test_df_list)
        # endregion

""" 列出通过筛选的标记、一维分箱、二维分箱 """
# region
# 交互标记的相关数据表
interaction_signals_test_df = pd.DataFrame({
    'signal colname': [
        '30-59late_low_x_credit_late_density_mid_high_signal',
        '30-59late_low_x_short_late_mid_high_signal',
        '30-59late_mid_x_credit_mid_signal',
        '90+late_mid_high_x_short_late_low_signal',
        'age_high_x_60-89late_mid_signal',
        'age_high_x_90+late_mid_signal',
        'age_high_x_credit_late_density_high_signal',
        'age_high_x_debt_low_signal',
        'age_high_x_late_severity_high_signal',
        'credit_low_x_util_mid_signal',
        'credit_pressure_mid_x_debt_mid_signal',
        'credit_pressure_mid_x_mortgage_mid_signal',
        'debt_low_x_credit_pressure_mid_signal',
        'debt_mid_x_credit_pressure_high_signal',
        'dep_missing_x_60-89late_mid_signal',
        'late_severity_high_x_credit_pressure_low_signal',
        'late_severity_high_x_short_late_low_signal',
        'late_severity_mid_x_short_late_low_signal',
        'monthly_debt_high_x_credit_pressure_mid_high_signal',
        'monthly_debt_high_x_debt_mid_signal',
        'monthly_debt_low_x_age_high_signal',
        'monthly_debt_low_x_credit_pressure_high_signal',
        'monthly_debt_low_x_mortgage_mid_signal',
        'monthly_debt_low_x_mortgage_ratio_mid_signal',
        'monthly_debt_mid_x_credit_pressure_mid_signal',
        'mortgage_mid_x_credit_late_density_high_signal',
        'mortgage_ratio_mid_x_credit_late_density_high_signal',
        'util_high_x_credit_pressure_low_signal',
        'util_low_mid_x_late_severity_high_signal',
        'util_low_x_credit_late_density_high_signal',
        'util_mid_x_90+late_mid_signal'
    ],
    'count marked': [
        1594.4, 2559.4, 549.6, 670.4, 356.0, 366.4, 714.4, 5405.8, 278.4,
        2404.8, 5945.2, 7693.0, 3049.0, 5038.8, 67.2, 518.6, 670.4, 1865.6,
        4325.2, 2079.8, 5407.8, 3844.4, 699.6, 204.4, 4502.8, 1290.4, 332.0,
        1636.2, 408.0, 287.4, 232.0
    ],
    'count normal': [
        118403.2, 117438.2, 119448.0, 119327.2, 119641.6, 119631.2, 119283.2, 114591.8, 119719.2,
        117592.8, 114052.4, 112304.6, 116948.6, 114958.8, 119930.4, 119479.0, 119327.2, 118132.0,
        115672.4, 117917.8, 114589.8, 116153.2, 119298.0, 119793.2, 115494.8, 118707.2, 119665.6,
        118361.4, 119589.6, 119710.2, 119765.6
    ],
    'RR': [
        5.097823, 4.286049, 5.877717, 6.705545, 3.454014, 4.489173, 5.541378, 0.17482, 6.485335,
        0.120923, 0.186938, 0.177958, 2.58384, 2.624934, 6.257462, 5.650464, 6.705545, 3.675369,
        0.316487, 0.314162, 0.191971, 2.507337, 0.255186, 0.117621, 0.202404, 6.714629, 6.963397,
        3.080146, 5.38985, 4.051518, 3.303112
    ],
    'IV': [
        0.106663, 0.116951, 0.051023, 0.079228, 0.01106, 0.020132, 0.058689, 0.071377, 0.031728,
        0.041168, 0.074248, 0.099769, 0.043014, 0.070817, 0.007285, 0.044717, 0.079228, 0.063386,
        0.031333, 0.015315, 0.066196, 0.049119, 0.006663, 0.003679, 0.05276, 0.146641, 0.043129,
        0.03737, 0.03227, 0.012746, 0.006525
    ]
})
# 通过筛选的标记
passed_binary_colnames = [
    # 数据质量标记
    'blacklist_flag',
    # 用户行为标记
    'has_no_credit',
    'has_serious_late',
    'has_short_late',
    'has_short_late_but_no_credit',
    'is_debt_high',
    'is_debt_overlimit',
    'is_util_high',
    'is_util_overlimit',
    # 交互标记
    '30-59late_low_x_credit_late_density_mid_high_signal',
    '30-59late_low_x_short_late_mid_high_signal',
    '30-59late_mid_x_credit_mid_signal',
    '90+late_mid_high_x_short_late_low_signal',
    'age_high_x_60-89late_mid_signal',
    'age_high_x_90+late_mid_signal',
    'age_high_x_credit_late_density_high_signal',
    'age_high_x_debt_low_signal',
    'age_high_x_late_severity_high_signal',
    'credit_low_x_util_mid_signal',
    'credit_pressure_mid_x_debt_mid_signal',
    'credit_pressure_mid_x_mortgage_mid_signal',
    'debt_low_x_credit_pressure_mid_signal',
    'debt_mid_x_credit_pressure_high_signal',
    'dep_missing_x_60-89late_mid_signal',
    'late_severity_high_x_credit_pressure_low_signal',
    'late_severity_high_x_short_late_low_signal',
    'late_severity_mid_x_short_late_low_signal',
    'monthly_debt_high_x_credit_pressure_mid_high_signal',
    'monthly_debt_high_x_debt_mid_signal',
    'monthly_debt_low_x_age_high_signal',
    'monthly_debt_low_x_credit_pressure_high_signal',
    'monthly_debt_low_x_mortgage_mid_signal',
    'monthly_debt_low_x_mortgage_ratio_mid_signal',
    'monthly_debt_mid_x_credit_pressure_mid_signal',
    'mortgage_mid_x_credit_late_density_high_signal',
    'mortgage_ratio_mid_x_credit_late_density_high_signal',
    'util_high_x_credit_pressure_low_signal',
    'util_low_mid_x_late_severity_high_signal',
    'util_low_x_credit_late_density_high_signal',
    'util_mid_x_90+late_mid_signal'
]
# 通过筛选的一维分箱
passed_binnames_1D = [
    '30-59late_bin',
    '60-89late_bin',
    '90+late_bin',
    'age_bin',
    'credit_bin',
    'credit_late_density_bin',
    'credit_pressure_index_bin',
    'debt_bin',
    'free_cashflow_income_bin',
    'income_bin',
    'income_per_dep_bin',
    'late_severity_score_bin',
    'mortgage_bin',
    'short_late_bin',
    'util_bin'
]
# 因IV低而未通过筛选的一维分箱
low_iv_binnames_1D = [
    'dep_bin', 'monthly_debt_bin', 'mortgage_ratio_bin'
]
# 通过筛选的二维分箱
passed_binnames_2D = [
    'credit_x_credit_pressure_index_bin',
    'credit_x_dep_bin',
    'credit_x_monthly_debt_bin',
    'debt_x_credit_bin',
    'debt_x_credit_pressure_index_bin',
    'debt_x_monthly_debt_bin',
    'debt_x_mortgage_bin',
    'debt_x_mortgage_ratio_bin',
    'income_x_monthly_debt_bin',
    'monthly_debt_x_credit_pressure_index_bin',
    'mortgage_x_credit_pressure_index_bin',
    'mortgage_x_dep_bin',
    'mortgage_x_monthly_debt_bin'
]
# 释放内存
if 'train_data' in dir() and 'valid_data' in dir():
    del train_data, valid_data
# endregion

""" 对raw LR模型的调试 """
is_processing_here3 = True
if is_processing_here3:
    # region
    raw_lr_fit_goodness_df_list = []
    # 划分训练集、验证集，准备K=5折交叉验证
    random_seed = 200
    folds = create_kfold(train_valid_data, random_seed, n_splits=5)
    for fold_idx in range(len(folds)):
        # 训练集、验证集拆分
        train_idx_list, valid_idx_list = folds[fold_idx]
        train_data = train_valid_data.iloc[train_idx_list].copy()
        valid_data = train_valid_data.iloc[valid_idx_list].copy()
        train_data_copy = train_data.copy(); valid_data_copy = valid_data.copy() # 备份
        # endregion
        
        """ 统一数据清洗阶段 """
        # region
        # 截断家属数异常大值
        cap_dep_num(train_data); cap_dep_num(valid_data)
        # 删除年龄异常小值
        delete_small_age(train_data); delete_small_age(valid_data)
        # 添加信用额度使用率的若干标记
        add_util_flags(train_data); add_util_flags(valid_data)
        # 添加负债率的若干标记
        add_debt_flags(train_data); add_debt_flags(valid_data)
        # 添加三种逾期次数的加权求和，权重由逻辑回归模型训练
        beta30, beta60, beta90 = add_late_severity_score(train_data)
        add_late_severity_score(valid_data, [beta30, beta60, beta90])
        # 添加衍生指标
        add_derived_features(train_data); add_derived_features(valid_data)
        # 筛选出所有非二元指标
        colnames_not_binary = get_colnames_whether_binary(train_data, is_binary=False)
        # 对所有非布尔值类型的指标建立一维分箱
        for colname in colnames_not_binary:
            _, div_pts = add_bins_1D(train_data, colname)
            add_bins_1D(valid_data, colname, fixed_div_pts=div_pts)
        # 把一维分箱的具体数值转换为排名序号
        convert_bins_value_to_rank(train_data); convert_bins_value_to_rank(valid_data)
        # 添加交互标记
        add_intersection_signals(train_data, positive_interactions_df, negative_interactions_df)
        add_intersection_signals(valid_data, positive_interactions_df, negative_interactions_df)
        # 暂时删除一维分箱列以及其对应的排名序号列
        colnames_without_bins = [colname for colname in train_data.columns if '_bin' not in colname]
        train_data = train_data[colnames_without_bins]; valid_data = valid_data[colnames_without_bins]
        # 整理指标顺序
        train_data = reorder_columns(train_data); valid_data = reorder_columns(valid_data)
        # endregion
        
        """ 对raw LR建立特化表达 """
        # region
        train_data_raw_lr = train_data.copy(); valid_data_raw_lr = valid_data.copy()
        # 缺失值、异常值全部赋值为0
        train_data_raw_lr = train_data_raw_lr.clip(lower=0)
        valid_data_raw_lr = valid_data_raw_lr.clip(lower=0)
        # 添加年龄、负债率、信用额度使用率的中心化三列以及中心化的年龄*负债率、年龄*信用额度使用率两列
        colnames_to_center = add_centered_features(train_data_raw_lr)
        add_centered_features(valid_data_raw_lr, fixed_colnames=colnames_to_center)
        # 对有长尾的指标取对数
        colnames_to_log = add_log_features(train_data_raw_lr)
        add_log_features(valid_data_raw_lr, fixed_colnames=colnames_to_log)
        # 筛选将要投入训练的列名
        selected_colnames = select_colnames_raw_lr(train_data_raw_lr)
        train_data_raw_lr = train_data_raw_lr[selected_colnames]
        valid_data_raw_lr = valid_data_raw_lr[selected_colnames]
        # 展示对raw LR建立特化表达完毕后的即将投入训练的列名
        if fold_idx == 0:
            colname_raw_lr_df = \
                display_current_colnames_raw_lr(train_valid_data, train_data, train_data_raw_lr)
        # endregion
        
        """ 训练raw LR """
        # region
        # 拆分目标变量和自变量
        y_train = train_data[target]; y_valid = valid_data[target]
        X_train_raw_lr = train_data_raw_lr.drop(target, axis=1, inplace=False)
        X_valid_raw_lr = valid_data_raw_lr.drop(target, axis=1, inplace=False)
        # 以下是可能尝试参与拟合的多种指标序列
        # 原始指标
        original_features = [
            'age_centered', 'util_centered', 'debt_centered',
            'income_log', dep, credit, mortgage,
            late30, late60, late90
        ]
        # 数据质量标记
        quality_flags = [
            'single_missing_flag', 'both_missing_flag', 'blacklist_flag',
            'income_anomaly_flag', 'debt_anomaly_flag', 'util_anomaly_flag'
        ]
        # 衍生指标
        derived_features = [
            'free_cashflow_income_log', 'income_per_dep_log', 'monthly_debt_log',
            'credit_pressure_index', 'mortgage_ratio', 'credit_late_density',
            'short_late', 'late_severity_score'
        ]
        # 用户行为标记
        risk_signals = [
            'has_no_credit', 'has_short_late_but_no_credit', 'has_serious_late', 'has_short_late',
            'is_debt_high', 'is_debt_overlimit', 'is_util_high', 'is_util_overlimit'
        ]
        # 削弱效应第1-5名
        negative_1_5 = [
            'monthly_debt_low_x_mortgage_ratio_mid_signal',
            'credit_low_x_util_mid_signal',
            'credit_pressure_mid_x_mortgage_mid_signal',
            'credit_pressure_mid_x_debt_mid_signal',
            'monthly_debt_low_x_age_high_signal'
        ]
        # 增强效应第1-5名
        positive_1_5 = [
            'mortgage_ratio_mid_x_credit_late_density_high_signal',
            'mortgage_mid_x_credit_late_density_high_signal',
            '90+late_mid_high_x_short_late_low_signal',
            'late_severity_high_x_short_late_low_signal',
            'age_high_x_late_severity_high_signal'
        ]
        # 增强效应第6-10名
        positive_6_10 = [
            'dep_missing_x_60-89late_mid_signal',
            '30-59late_mid_x_credit_mid_signal',
            'late_severity_high_x_credit_pressure_low_signal',
            'age_high_x_credit_late_density_high_signal',
            'util_low_mid_x_late_severity_high_signal'
        ]
        # 增强效应第11-15名
        positive_11_15 = [
            '30-59late_low_x_credit_late_density_mid_high_signal',
            '30-59late_low_x_short_late_mid_high_signal',
            'age_high_x_90+late_mid_signal',
            'util_low_x_credit_late_density_high_signal',
            'late_severity_mid_x_short_late_low_signal'
        ]
        # 以下是不同的进行拟合的指标组合
        colnames_to_fit0 = original_features + quality_flags
        colnames_to_fit1 = colnames_to_fit0 + derived_features + risk_signals
        colnames_to_fit2 = colnames_to_fit1 + positive_1_5
        colnames_to_fit3 = colnames_to_fit1 + negative_1_5
        colnames_to_fit4 = colnames_to_fit2 + positive_6_10
        colnames_to_fit5 = colnames_to_fit4 + positive_11_15
        colnames_to_fit_list = [
            colnames_to_fit0, colnames_to_fit1, colnames_to_fit2,
            colnames_to_fit3, colnames_to_fit4, colnames_to_fit5
        ]
        # 正则化强度的倒数（参数C）的尝试列表
        c_list = [0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2]
        # 构建raw LR拟合效果的数据表
        raw_lr_fit_goodness_df = grid_search_lr\
            (X_train_raw_lr, y_train, X_valid_raw_lr, y_valid, colnames_to_fit_list, c_list)
        raw_lr_fit_goodness_df_list.append(raw_lr_fit_goodness_df)
        # 网格搜索找出最合适的拟合指标方案和正则化强度的倒数C
        if fold_idx == len(folds) - 1: # 在交叉验证的最后一折时再做
            raw_lr_auc_ks_df, raw_lr_coef_sign_stability_df = \
                test_fit_goodness_lr(raw_lr_fit_goodness_df_list)
        # 确定最佳的拟合指标序号与参数C
        colnames_to_fit = colnames_to_fit1
        X_train_raw_lr = X_train_raw_lr[colnames_to_fit]; X_valid_raw_lr = X_valid_raw_lr[colnames_to_fit]
        c = 1
        # 以下是尝试舍弃以改善VIF的指标序列
        colnames_to_drop0 = []
        colnames_to_drop1 = [late30, late60, 'income_log', 'free_cashflow_income_log']
        colnames_to_drop2 = colnames_to_drop1 + [late90]
        colnames_to_drop3 = colnames_to_drop1 + ['monthly_debt_log']
        colnames_to_drop4 = colnames_to_drop1 + [late90, 'monthly_debt_log']
        colnames_to_drop_list = [
            colnames_to_drop0, colnames_to_drop1, colnames_to_drop2, colnames_to_drop3, colnames_to_drop4
        ]
        if fold_idx == len(folds) - 1: # 在交叉验证的最后一折时再做
            # 构建不同舍弃指标方案的VIF测试结果表
            vif_test_df = test_vif_after_drop(X_train_raw_lr, colnames_to_drop_list)
        # endregion

""" 确定对于raw LR的最佳拟合指标与参数C """
# region
colnames_to_fit_raw_lr = [
        # 原始指标
        'age_centered', 'util_centered', 'debt_centered',
        'income_log', dep, credit, mortgage,
        late30, late60, late90,
        # 数据质量标记
        'single_missing_flag', 'both_missing_flag', 'blacklist_flag',
        'income_anomaly_flag', 'debt_anomaly_flag', 'util_anomaly_flag',
        # 衍生指标
        'free_cashflow_income_log', 'income_per_dep_log', 'monthly_debt_log',
        'credit_pressure_index', 'mortgage_ratio', 'credit_late_density',
        'short_late', 'late_severity_score',
        # 用户行为标记
        'has_no_credit', 'has_short_late_but_no_credit', 'has_serious_late', 'has_short_late',
        'is_debt_high', 'is_debt_overlimit', 'is_util_high', 'is_util_overlimit'
]
c_raw_lr = 1
colnames_to_drop_raw_lr = [
    late30, late60, late90, 
    'income_log', 'free_cashflow_income_log', 'monthly_debt_log'
]
# 检测固定最佳的删除指标后模型拟合效果的变化情况
raw_lr_fit_goodness_after_drop_test_df = test_fit_goodness_lr_after_drop(
    X_train_raw_lr, y_train, X_valid_raw_lr, y_valid, 
    colnames_to_fit_raw_lr, c_raw_lr, colnames_to_drop_raw_lr
) if is_processing_here3 else None
# 简化变量
colnames_to_fit_raw_lr = \
    [colname for colname in colnames_to_fit_raw_lr if colname not in colnames_to_drop_raw_lr]
# 释放内存
if 'train_data_raw_lr' in dir() and 'valid_data_raw_lr' in dir():
    del train_data, valid_data, train_data_raw_lr, valid_data_raw_lr
# endregion

""" 对WOE LR模型的调试 """
is_processing_here4 = True
if is_processing_here4:
    # region
    woe_lr_fit_goodness_df_list = []
    # 划分训练集、验证集，准备K=5折交叉验证
    random_seed = 300
    folds = create_kfold(train_valid_data, random_seed, n_splits=5)
    for fold_idx in range(len(folds)):
        # 训练集、验证集拆分
        train_idx_list, valid_idx_list = folds[fold_idx]
        train_data = train_valid_data.iloc[train_idx_list].copy()
        valid_data = train_valid_data.iloc[valid_idx_list].copy()
        train_data_copy = train_data.copy(); valid_data_copy = valid_data.copy() # 备份
    # endregion
        
        """ 统一数据清洗阶段 """
        # region
        # 截断家属数异常大值
        cap_dep_num(train_data); cap_dep_num(valid_data)
        # 删除年龄异常小值
        delete_small_age(train_data); delete_small_age(valid_data)
        # 添加信用额度使用率的若干标记
        add_util_flags(train_data); add_util_flags(valid_data)
        # 添加负债率的若干标记
        add_debt_flags(train_data); add_debt_flags(valid_data)
        # 添加三种逾期次数的加权求和，权重由逻辑回归模型训练
        beta30, beta60, beta90 = add_late_severity_score(train_data)
        add_late_severity_score(valid_data, [beta30, beta60, beta90])
        # 添加衍生指标
        add_derived_features(train_data); add_derived_features(valid_data)
        # 筛选出所有非二元指标
        colnames_not_binary = get_colnames_whether_binary(train_data, is_binary=False)
        # 对所有非布尔值类型的指标建立一维分箱
        for colname in colnames_not_binary:
            _, div_pts = add_bins_1D(train_data, colname)
            add_bins_1D(valid_data, colname, fixed_div_pts=div_pts)
        # 把一维分箱的具体数值转换为排名序号
        convert_bins_value_to_rank(train_data); convert_bins_value_to_rank(valid_data)
        # 添加交互标记
        add_intersection_signals(train_data, positive_interactions_df, negative_interactions_df)
        add_intersection_signals(valid_data, positive_interactions_df, negative_interactions_df)
        # 暂时删除一维分箱列以及其对应的排名序号列
        colnames_without_bins = [colname for colname in train_data.columns if '_bin' not in colname]
        train_data = train_data[colnames_without_bins]; valid_data = valid_data[colnames_without_bins]
        # 整理指标顺序
        train_data = reorder_columns(train_data); valid_data = reorder_columns(valid_data)
        # endregion
        
        """ 对WOE LR建立特化表达 """
        # region
        train_data_woe_lr = train_data.copy(); valid_data_woe_lr = valid_data.copy()
        # 找出所有的非二元列
        colnames_not_binary = get_colnames_whether_binary(train_data_woe_lr, is_binary=False)
        # 对所有非布尔值类型的指标建立一维分箱（虽然构建了理应被淘汰的一维分箱，但后续控制其进入模型训练即可）
        for colname in colnames_not_binary:
            _, div_pts = add_bins_1D(train_data_woe_lr, colname)
            add_bins_1D(valid_data_woe_lr, colname, fixed_div_pts=div_pts)
        # 列出所有一维分箱对应指标的所有无序对
        colname_pairs_not_binary = []
        for i in range(len(colnames_not_binary) - 1):
                for j in range(i + 1, len(colnames_not_binary)):
                    colname_pairs_not_binary.append([colnames_not_binary[i], colnames_not_binary[j]])
        # 只建立通过筛选的二维分箱
        for colname_pair in colname_pairs_not_binary:
            _, bins_2D = add_bins_2D(train_data_woe_lr, colname_pair, constraint=passed_binnames_2D)
            add_bins_2D\
                (valid_data_woe_lr, colname_pair, fixed_bins_2D=bins_2D, constraint=passed_binnames_2D)
        # 把所有标记列、一维分箱列、二维分箱列转换为WOE列
        binnames_map_to_woe = \
            passed_binary_colnames + passed_binnames_1D + low_iv_binnames_1D + passed_binnames_2D
        for colname in binnames_map_to_woe:
            woe_map = fit_woe_mapping(train_data_woe_lr, colname)
            apply_woe_mapping(train_data_woe_lr, colname, woe_map)
            apply_woe_mapping(valid_data_woe_lr, colname, woe_map)
        # 在数据集中只留下投入训练的WOE列
        woe_colnames = [colname for colname in train_data_woe_lr.columns if colname.endswith('_woe')]
        train_data_woe_lr = train_data_woe_lr[[target] + woe_colnames]
        valid_data_woe_lr = valid_data_woe_lr[[target] + woe_colnames]
        # 重排指标列名
        train_data_woe_lr = reorder_columns(train_data_woe_lr)
        valid_data_woe_lr = reorder_columns(valid_data_woe_lr)
        # 展示对WOE LR建立特化表达完毕后的即将投入训练的列名
        if fold_idx == len(folds) - 1: # 在交叉验证的最后一折时再做
            colname_woe_lr_df = display_current_colnames_woe_lr(train_data, train_data_woe_lr)
        # endregion
        
        """ 训练WOE LR """
        # region
        # 拆分目标变量和自变量
        y_train = train_data[target]; y_valid = valid_data[target]
        X_train_woe_lr = train_data_woe_lr.drop(target, axis=1, inplace=False)
        X_valid_woe_lr = valid_data_woe_lr.drop(target, axis=1, inplace=False)
        # 以下是可能尝试参与拟合的多种指标序列（已补充"_woe"后缀）
        # 原始指标
        original_features = [
            'age_bin_woe', 'util_bin_woe', 'debt_bin_woe',
            'income_bin_woe', 'credit_bin_woe', 'mortgage_bin_woe',
            '30-59late_bin_woe', '60-89late_bin_woe', '90+late_bin_woe'
        ]
        # 衍生指标
        derived_features = [
            'free_cashflow_income_bin_woe', 'income_per_dep_bin_woe',
            'credit_pressure_index_bin_woe', 'credit_late_density_bin_woe',
            'short_late_bin_woe', 'late_severity_score_bin_woe'
        ]
        # 含有较强风险信息的标记
        risk_signals = [
            'has_no_credit_woe', 'has_short_late_but_no_credit_woe', 
            'has_serious_late_woe', 'has_short_late_woe', 'blacklist_flag_woe',
            'is_debt_high_woe', 'is_debt_overlimit_woe', 
            'is_util_high_woe', 'is_util_overlimit_woe'
        ]
        # 二维分箱
        bins_2D = [
            'credit_x_credit_pressure_index_bin_woe',
            'credit_x_dep_bin_woe',
            'credit_x_monthly_debt_bin_woe',
            'debt_x_credit_bin_woe',
            'debt_x_credit_pressure_index_bin_woe',
            'debt_x_monthly_debt_bin_woe',
            'debt_x_mortgage_bin_woe',
            'debt_x_mortgage_ratio_bin_woe',
            'income_x_monthly_debt_bin_woe',
            'monthly_debt_x_credit_pressure_index_bin_woe',
            'mortgage_x_credit_pressure_index_bin_woe',
            'mortgage_x_dep_bin_woe',
            'mortgage_x_monthly_debt_bin_woe'
        ]
        # 低信息价值指标
        low_iv_features = [
            'dep_bin_woe', 'monthly_debt_bin_woe', 'mortgage_ratio_bin_woe'
        ]
        # 增强效应第1-5名
        positive_1_5 = [
            'mortgage_ratio_mid_x_credit_late_density_high_signal_woe',
            'mortgage_mid_x_credit_late_density_high_signal_woe',
            '90+late_mid_high_x_short_late_low_signal_woe',
            'late_severity_high_x_short_late_low_signal_woe',
            'age_high_x_late_severity_high_signal_woe'
        ]
        # 以下是不同的进行拟合的指标组合
        colnames_to_fit0 = original_features
        colnames_to_fit1 = colnames_to_fit0 + derived_features
        colnames_to_fit2 = colnames_to_fit1 + risk_signals
        colnames_to_fit3 = colnames_to_fit2 + bins_2D
        colnames_to_fit4 = colnames_to_fit2 + low_iv_features
        colnames_to_fit5 = colnames_to_fit2 + positive_1_5
        colnames_to_fit_list = [
            colnames_to_fit0, colnames_to_fit1, colnames_to_fit2, 
            colnames_to_fit3, colnames_to_fit4, colnames_to_fit5
        ]
        # 正则化强度的倒数（参数C）的尝试列表
        c_list = [0.01, 0.025, 0.1, 0.2, 0.5, 1, 2]
        # 构建WOE LR拟合效果的数据表
        woe_lr_fit_goodness_df = grid_search_lr\
            (X_train_woe_lr, y_train, X_valid_woe_lr, y_valid, colnames_to_fit_list, c_list)
        woe_lr_fit_goodness_df_list.append(woe_lr_fit_goodness_df)
        # 网格搜索找出最合适的拟合指标方案和正则化强度的倒数C
        if fold_idx == len(folds) - 1: # 在交叉验证的最后一折时再做
            woe_lr_auc_ks_df, woe_lr_coef_sign_stability_df = \
                test_fit_goodness_lr(woe_lr_fit_goodness_df_list)
        # 用最佳的拟合指标去精简数据集
        colnames_to_fit = colnames_to_fit3
        X_train_woe_lr = X_train_woe_lr[colnames_to_fit3]; X_valid_woe_lr = X_valid_woe_lr[colnames_to_fit3]
        c = 0.025
        # 以下是尝试舍弃以改善VIF的指标序列
        colnames_to_drop0 = []
        colnames_to_drop1 = [
            '30-59late_bin_woe', '60-89late_bin_woe', '90+late_bin_woe',
            'income_bin_woe', 'free_cashflow_income_bin_woe'
        ]
        colnames_to_drop2 = colnames_to_drop1 + [
            'short_late_bin_woe',
            'credit_x_credit_pressure_index_bin_woe', 'mortgage_x_credit_pressure_index_bin_woe',
            'monthly_debt_x_credit_pressure_index_bin_woe'
        ]
        colnames_to_drop3 = colnames_to_drop2 + ['has_short_late_woe', 'credit_late_density_bin_woe']
        colnames_to_drop_list = [
            colnames_to_drop0, colnames_to_drop1, colnames_to_drop2, colnames_to_drop3
        ]
        if fold_idx == 0: # 只在交叉验证的第一折时做，避免重复
            # 构建不同舍弃指标方案的VIF测试结果表
            vif_test_df = test_vif_after_drop(X_train_woe_lr, colnames_to_drop_list)
        # endregion

""" 确定对于WOE LR的最佳拟合指标与参数C """
# region
colnames_to_fit_woe_lr = [
    # 原始指标
    'age_bin_woe', 'util_bin_woe', 'debt_bin_woe',
    'income_bin_woe', 'credit_bin_woe', 'mortgage_bin_woe',
    '30-59late_bin_woe', '60-89late_bin_woe', '90+late_bin_woe',
    # 衍生指标
    'free_cashflow_income_bin_woe', 'income_per_dep_bin_woe',
    'credit_pressure_index_bin_woe', 'credit_late_density_bin_woe',
    'short_late_bin_woe', 'late_severity_score_bin_woe',
    # 含有较强风险信息的标记
    'has_no_credit_woe', 'has_short_late_but_no_credit_woe', 
    'has_serious_late_woe', 'has_short_late_woe', 'blacklist_flag_woe',
    'is_debt_high_woe', 'is_debt_overlimit_woe', 
    'is_util_high_woe', 'is_util_overlimit_woe',
    # 二维分箱
    'credit_x_credit_pressure_index_bin_woe',
    'credit_x_dep_bin_woe',
    'credit_x_monthly_debt_bin_woe',
    'debt_x_credit_bin_woe',
    'debt_x_credit_pressure_index_bin_woe',
    'debt_x_monthly_debt_bin_woe',
    'debt_x_mortgage_bin_woe',
    'debt_x_mortgage_ratio_bin_woe',
    'income_x_monthly_debt_bin_woe',
    'monthly_debt_x_credit_pressure_index_bin_woe',
    'mortgage_x_credit_pressure_index_bin_woe',
    'mortgage_x_dep_bin_woe',
    'mortgage_x_monthly_debt_bin_woe'
]
c_woe_lr = 0.025
colnames_to_drop_woe_lr = [
    '30-59late_bin_woe', '60-89late_bin_woe', '90+late_bin_woe',
    'short_late_bin_woe', 'has_short_late_woe',
    'credit_late_density_bin_woe', 'income_bin_woe', 'free_cashflow_income_bin_woe',
    'credit_x_credit_pressure_index_bin_woe',
    'mortgage_x_credit_pressure_index_bin_woe',
    'monthly_debt_x_credit_pressure_index_bin_woe'
]
# 检测固定最佳的删除指标后模型拟合效果的变化情况
woe_lr_fit_goodness_after_drop_test_df = test_fit_goodness_lr_after_drop(
    X_train_woe_lr, y_train, X_valid_woe_lr, y_valid, 
    colnames_to_fit_woe_lr, c_woe_lr, colnames_to_drop_woe_lr
) if is_processing_here4 else None
# 简化变量
colnames_to_fit_woe_lr = \
    [colname for colname in colnames_to_fit_woe_lr if colname not in colnames_to_drop_woe_lr]
# 释放内存
if 'train_data_woe_lr' in dir() and 'valid_data_woe_lr' in dir():
    del train_data, valid_data, train_data_woe_lr, valid_data_woe_lr
# endregion

""" 对XGBoost模型的调试 """
is_processing_here5 = True
if is_processing_here5:
    # region
    xgb_auc_ks_round1_df_list = []
    xgb_top_recall_df_list = []
    xgb_auc_ks_round2_df_list = []
    xgb_auc_ks_round3_df_list = []
    xgb_auc_ks_round4_df_list = []
    xgb_auc_ks_round5_df_list = []
    # 划分训练集、验证集，准备K=5折交叉验证
    random_seed = 400
    folds = create_kfold(train_valid_data, random_seed, n_splits=5)
    for fold_idx in range(len(folds)):
        # 训练集、验证集拆分
        train_idx_list, valid_idx_list = folds[fold_idx]
        train_data = train_valid_data.iloc[train_idx_list].copy()
        valid_data = train_valid_data.iloc[valid_idx_list].copy()
        train_data_copy = train_data.copy(); valid_data_copy = valid_data.copy() # 备份
        # endregion
        
        """ 统一数据清洗阶段 """
        # region
        # 截断家属数异常大值
        cap_dep_num(train_data); cap_dep_num(valid_data)
        # 删除年龄异常小值
        delete_small_age(train_data); delete_small_age(valid_data)
        # 添加信用额度使用率的若干标记
        add_util_flags(train_data); add_util_flags(valid_data)
        # 添加负债率的若干标记
        add_debt_flags(train_data); add_debt_flags(valid_data)
        # 添加三种逾期次数的加权求和，权重由逻辑回归模型训练
        beta30, beta60, beta90 = add_late_severity_score(train_data)
        add_late_severity_score(valid_data, [beta30, beta60, beta90])
        # 添加衍生指标
        add_derived_features(train_data); add_derived_features(valid_data)
        # 筛选出所有非二元指标
        colnames_not_binary = get_colnames_whether_binary(train_data, is_binary=False)
        # 对所有非布尔值类型的指标建立一维分箱
        for colname in colnames_not_binary:
            _, div_pts = add_bins_1D(train_data, colname)
            add_bins_1D(valid_data, colname, fixed_div_pts=div_pts)
        # 把一维分箱的具体数值转换为排名序号
        convert_bins_value_to_rank(train_data); convert_bins_value_to_rank(valid_data)
        # 添加交互标记
        add_intersection_signals(train_data, positive_interactions_df, negative_interactions_df)
        add_intersection_signals(valid_data, positive_interactions_df, negative_interactions_df)
        # 暂时删除一维分箱列以及其对应的排名序号列
        colnames_without_bins = [colname for colname in train_data.columns if '_bin' not in colname]
        train_data = train_data[colnames_without_bins]; valid_data = valid_data[colnames_without_bins]
        # endregion
        
        """ 对XGBoost建立特化表达 """
        # region
        train_data_xgb = train_data.copy(); valid_data_xgb = valid_data.copy()
        # 将所有表示数据不可靠的负整数转换为NA
        train_data_xgb = train_data_xgb.mask(train_data_xgb < 0, np.nan)
        valid_data_xgb = valid_data_xgb.mask(valid_data_xgb < 0, np.nan)
        # endregion
        
        """ 训练XGBoost模型 """
        # region
        # 拆分目标变量和自变量
        y_train = train_data[target]; y_valid = valid_data[target]
        X_train_xgb = train_data_xgb.drop(target, axis=1, inplace=False)
        X_valid_xgb = valid_data_xgb.drop(target, axis=1, inplace=False)
        # 第一轮网格搜索（对参与拟合的指标）
        # 原始指标
        original_features = [
            age, util, debt,
            income, dep, credit, mortgage,
            late30, late60, late90
        ]
        # 数据质量标记
        quality_flags = [
            'single_missing_flag', 'both_missing_flag', 'blacklist_flag',
            'income_anomaly_flag', 'debt_anomaly_flag', 'util_anomaly_flag'
        ]
        # 衍生指标
        derived_features = [
            'free_cashflow_income', 'income_per_dep', 'monthly_debt',
            'credit_pressure_index', 'mortgage_ratio', 'credit_late_density',
            'short_late', 'late_severity_score'
        ]
        # 用户行为标记
        risk_signals = [
            'has_no_credit', 'has_short_late_but_no_credit', 'has_serious_late', 'has_short_late',
            'is_debt_high', 'is_debt_overlimit', 'is_util_high', 'is_util_overlimit'
        ]
        # 削弱效应第1-5名
        negative_1_5 = [
            'monthly_debt_low_x_mortgage_ratio_mid_signal',
            'credit_low_x_util_mid_signal',
            'credit_pressure_mid_x_mortgage_mid_signal',
            'credit_pressure_mid_x_debt_mid_signal',
            'monthly_debt_low_x_age_high_signal'
        ]
        # 增强效应第1-5名
        positive_1_5 = [
            'mortgage_ratio_mid_x_credit_late_density_high_signal',
            'mortgage_mid_x_credit_late_density_high_signal',
            '90+late_mid_high_x_short_late_low_signal',
            'late_severity_high_x_short_late_low_signal',
            'age_high_x_late_severity_high_signal'
        ]
        # 增强效应第6-10名
        positive_6_10 = [
            'dep_missing_x_60-89late_mid_signal',
            '30-59late_mid_x_credit_mid_signal',
            'late_severity_high_x_credit_pressure_low_signal',
            'age_high_x_credit_late_density_high_signal',
            'util_low_mid_x_late_severity_high_signal'
        ]
        # 即将尝试的拟合指标组合
        colnames_to_fit0 = original_features
        colnames_to_fit1 = colnames_to_fit0 + quality_flags
        colnames_to_fit2 = colnames_to_fit1 + derived_features
        colnames_to_fit3 = colnames_to_fit2 + risk_signals
        colnames_to_fit4 = colnames_to_fit2 + negative_1_5
        colnames_to_fit5 = colnames_to_fit2 + positive_1_5
        colnames_to_fit_list_round1 = [
            colnames_to_fit0, colnames_to_fit1, colnames_to_fit2, 
            colnames_to_fit3, colnames_to_fit4, colnames_to_fit5, 
        ]
        xgb_auc_ks_round1_df = grid_search_colnames_xgb(
            X_train_xgb, y_train, X_valid_xgb, y_valid, 
            colnames_to_fit_list_round1
        )
        xgb_auc_ks_round1_df_list.append(xgb_auc_ks_round1_df)
        if fold_idx == len(folds) - 1:
            xgb_auc_ks_round1_df = test_fit_goodness_xgb(xgb_auc_ks_round1_df_list)
        # 在第一轮网格搜索的基础上，探索哪些衍生指标与用户行为标记值得被保留
        xgb_top_recall_df = grid_search_colnames_upgraded_xgb(
            X_train_xgb, y_train, X_valid_xgb, y_valid,
            base_colnames=original_features + quality_flags,
            colnames_to_try=derived_features + risk_signals
        )
        xgb_top_recall_df_list.append(xgb_top_recall_df)
        if fold_idx == len(folds) - 1:
            xgb_top_recall_aggregate_df = test_fit_goodness_upgraded_xgb(xgb_top_recall_df_list)
        # 确定保留下来的衍生指标
        derived_features_retained = [
            'free_cashflow_income', 'credit_late_density', 'credit_pressure_index',
            'mortgage_ratio', 'late_severity_score'
        ]
        colnames_to_fit = original_features + derived_features_retained
        X_train_xgb = X_train_xgb[colnames_to_fit]; X_valid_xgb = X_valid_xgb[colnames_to_fit]
        # 确定下来的最佳参数的储存字典
        best_para_dict = {}
        # 第二轮网格搜索（对模型参数）
        paras_round2 = ['max_depth', 'min_child_weight']
        max_depth_list = [2, 3, 4, 5, 6, 7]
        min_child_weight_list = [1, 3, 5, 7, 10]
        xgb_auc_ks_round2_df = grid_search_paras_xgb(
            X_train_xgb, y_train, X_valid_xgb, y_valid, 
            paras_round2, max_depth_list, min_child_weight_list
        )
        xgb_auc_ks_round2_df_list.append(xgb_auc_ks_round2_df)
        if fold_idx == len(folds) - 1:
            xgb_auc_ks_round2_df = test_fit_goodness_xgb(xgb_auc_ks_round2_df_list)
        # 确定最佳的max_depth和min_child_weight
        best_para_dict['max_depth'] = 3
        best_para_dict['min_child_weight'] = 7
        # 第三轮网格搜索（对模型参数）
        paras_round3 = ['learning_rate', 'n_estimators']
        learning_rate_list = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
        n_estimators_list = [300, 500, 800, 1000, 1500]
        xgb_auc_ks_round3_df = grid_search_paras_xgb(
            X_train_xgb, y_train, X_valid_xgb, y_valid, 
            paras_round3, learning_rate_list, n_estimators_list, best_para_dict
        )
        xgb_auc_ks_round3_df_list.append(xgb_auc_ks_round3_df)
        if fold_idx == len(folds) - 1:
            xgb_auc_ks_round3_df = test_fit_goodness_xgb(xgb_auc_ks_round3_df_list)
        # 确定最佳的learning_rate和n_estimators
        best_para_dict['learning_rate'] = 0.03
        best_para_dict['n_estimators'] = 800
        # 第四轮网格搜索（对模型参数）
        paras_round4 = ['subsample', 'colsample_bytree']
        subsample_list = [0.6, 0.7, 0.8, 0.9]
        colsample_bytree_list = [0.6, 0.7, 0.8, 0.9]
        xgb_auc_ks_round4_df = grid_search_paras_xgb(
            X_train_xgb, y_train, X_valid_xgb, y_valid, 
            paras_round4, subsample_list, colsample_bytree_list, best_para_dict
        )
        xgb_auc_ks_round4_df_list.append(xgb_auc_ks_round4_df)
        if fold_idx == len(folds) - 1:
            xgb_auc_ks_round4_df = test_fit_goodness_xgb(xgb_auc_ks_round4_df_list)
        # 确定最佳的subsample和colsample_bytree
        best_para_dict['subsample'] = 0.7
        best_para_dict['colsample_bytree'] = 0.6
        # # 第五轮网格搜索（对模型参数）
        paras_round5 = ['reg_alpha', 'reg_lambda']
        reg_alpha_list = [0, 0.1, 0.25, 0.5, 1]
        reg_lambda_list = [1, 2, 4, 7, 10]
        xgb_auc_ks_round5_df = grid_search_paras_xgb(
            X_train_xgb, y_train, X_valid_xgb, y_valid, 
            paras_round5, reg_alpha_list, reg_lambda_list, best_para_dict
        )
        xgb_auc_ks_round5_df_list.append(xgb_auc_ks_round5_df)
        if fold_idx == len(folds) - 1:
            xgb_auc_ks_round5_df = test_fit_goodness_xgb(xgb_auc_ks_round5_df_list)
        # 确定最佳的reg_alpha和reg_lambda
        best_para_dict['reg_alpha'] = 1
        best_para_dict['reg_lambda'] = 2
        # endregion

""" 确定对于XGBoost的最佳拟合指标与参数组合 """
# region
# 最佳拟合指标
colnames_to_fit_xgb = [
    # 原始指标
    age, util, debt,
    income, dep, credit, mortgage,
    late30, late60, late90,
    # 衍生指标
    'free_cashflow_income', 'credit_pressure_index', 'credit_late_density',
    'mortgage_ratio', 'late_severity_score'
]
# 最佳模型can
best_para_xgb_dict = dict(
    max_depth = 3,
    min_child_weight = 7,
    learning_rate = 0.03,
    n_estimators = 800,
    subsample = 0.7,
    colsample_bytree = 0.6,
    reg_alpha = 1,
    reg_lambda = 2
)
# 释放内存
if 'train_data_xgb' in dir() and 'valid_data_xgb' in dir():
    del train_data, valid_data, train_data_xgb, valid_data_xgb
# endregion

""" 使用各模型的最佳参数组合，以原训练集与验证集的并集做拟合，观察在测试集中的拟合效果 """
is_processing_here6 = True
train_data = train_valid_data

if is_processing_here6:
    """ 统一数据清洗阶段 """
    # region
    # 截断家属数异常大值
    cap_dep_num(train_data); cap_dep_num(test_data)
    # 删除年龄异常小值
    delete_small_age(train_data); delete_small_age(test_data)
    # 添加信用额度使用率的若干标记
    add_util_flags(train_data); add_util_flags(test_data)
    # 添加负债率的若干标记
    add_debt_flags(train_data); add_debt_flags(test_data)
    # 添加三种逾期次数的加权求和，权重由逻辑回归模型训练
    beta30, beta60, beta90 = add_late_severity_score(train_data)
    add_late_severity_score(test_data, [beta30, beta60, beta90])
    # 添加衍生指标
    add_derived_features(train_data); add_derived_features(test_data)
    # 筛选出所有非二元指标
    colnames_not_binary = get_colnames_whether_binary(train_data, is_binary=False)
    # 对所有非布尔值类型的指标建立一维分箱
    for colname in colnames_not_binary:
        _, div_pts = add_bins_1D(train_data, colname)
        add_bins_1D(test_data, colname, fixed_div_pts=div_pts)
    # 把一维分箱的具体数值转换为排名序号
    convert_bins_value_to_rank(train_data); convert_bins_value_to_rank(test_data)
    # 添加交互标记
    add_intersection_signals(train_data, positive_interactions_df, negative_interactions_df)
    add_intersection_signals(test_data, positive_interactions_df, negative_interactions_df)
    # 暂时删除一维分箱列以及其对应的排名序号列
    colnames_without_bins = [colname for colname in train_data.columns if '_bin' not in colname]
    train_data = train_data[colnames_without_bins]; test_data = test_data[colnames_without_bins]
    # 整理指标顺序
    train_data = reorder_columns(train_data); test_data = reorder_columns(test_data)
    # endregion
    
    """ 对raw LR建立特化表达 """
    # region
    train_data_raw_lr = train_data.copy(); test_data_raw_lr = test_data.copy()
    # 缺失值、异常值全部赋值为0
    train_data_raw_lr = train_data_raw_lr.clip(lower=0)
    test_data_raw_lr = test_data_raw_lr.clip(lower=0)
    # 添加年龄、负债率、信用额度使用率的中心化三列以及中心化的年龄*负债率、年龄*信用额度使用率两列
    colnames_to_center = add_centered_features(train_data_raw_lr)
    add_centered_features(test_data_raw_lr, fixed_colnames=colnames_to_center)
    # 对有长尾的指标取对数
    colnames_to_log = add_log_features(train_data_raw_lr)
    add_log_features(test_data_raw_lr, fixed_colnames=colnames_to_log)
    # 筛选将要投入训练的列名
    selected_colnames = select_colnames_raw_lr(train_data_raw_lr)
    train_data_raw_lr = train_data_raw_lr[selected_colnames]
    test_data_raw_lr = test_data_raw_lr[selected_colnames]
    # endregion
    
    """ 对WOE LR建立特化表达 """
    # region
    train_data_woe_lr = train_data.copy(); test_data_woe_lr = test_data.copy()
    # 找出所有的非二元列
    colnames_not_binary = get_colnames_whether_binary(train_data_woe_lr, is_binary=False)
    # 对所有非布尔值类型的指标建立一维分箱（虽然构建了理应被淘汰的一维分箱，但后续控制其进入模型训练即可）
    for colname in colnames_not_binary:
        _, div_pts = add_bins_1D(train_data_woe_lr, colname)
        add_bins_1D(test_data_woe_lr, colname, fixed_div_pts=div_pts)
    # 列出所有一维分箱对应指标的所有无序对
    colname_pairs_not_binary = []
    for i in range(len(colnames_not_binary) - 1):
            for j in range(i + 1, len(colnames_not_binary)):
                colname_pairs_not_binary.append([colnames_not_binary[i], colnames_not_binary[j]])
    # 只建立通过筛选的二维分箱
    for colname_pair in colname_pairs_not_binary:
        _, bins_2D = add_bins_2D(train_data_woe_lr, colname_pair, constraint=passed_binnames_2D)
        add_bins_2D(test_data_woe_lr, colname_pair, fixed_bins_2D=bins_2D, constraint=passed_binnames_2D)
    # 把所有标记列、一维分箱列、二维分箱列转换为WOE列
    binnames_map_to_woe = \
        passed_binary_colnames + passed_binnames_1D + low_iv_binnames_1D + passed_binnames_2D
    for colname in binnames_map_to_woe:
        woe_map = fit_woe_mapping(train_data_woe_lr, colname)
        apply_woe_mapping(train_data_woe_lr, colname, woe_map)
        apply_woe_mapping(test_data_woe_lr, colname, woe_map)
    # 在数据集中只留下投入训练的WOE列
    woe_colnames = [colname for colname in train_data_woe_lr.columns if colname.endswith('_woe')]
    train_data_woe_lr = train_data_woe_lr[[target] + woe_colnames]
    test_data_woe_lr = test_data_woe_lr[[target] + woe_colnames]
    # 重排指标列名
    train_data_woe_lr = reorder_columns(train_data_woe_lr)
    test_data_woe_lr = reorder_columns(test_data_woe_lr)
    # endregion
    
    """ 对XGBoost建立特化表达 """
    # region
    train_data_xgb = train_data.copy(); test_data_xgb = test_data.copy()
    # 将所有表示数据不可靠的负整数转换为NA
    train_data_xgb = train_data_xgb.mask(train_data_xgb < 0, np.nan)
    test_data_xgb = test_data_xgb.mask(test_data_xgb < 0, np.nan)
    # endregion
    
    """ 按确定好的最佳参数分别构建三种模型 """
    # region
    # 拆分目标变量和自变量
    y_train = train_data[target]; y_test = test_data[target]
    X_train_raw_lr = train_data_raw_lr.drop(target, axis=1, inplace=False)
    X_test_raw_lr = test_data_raw_lr.drop(target, axis=1, inplace=False)
    X_train_woe_lr = train_data_woe_lr.drop(target, axis=1, inplace=False)
    X_test_woe_lr = test_data_woe_lr.drop(target, axis=1, inplace=False)
    X_train_xgb = train_data_xgb.drop(target, axis=1, inplace=False)
    X_test_xgb = test_data_xgb.drop(target, axis=1, inplace=False)
    # 释放内存
    if 'train_data' in dir() and 'test_data' in dir():
        del train_data; del test_data
    
    # 构建raw LR
    X_train_raw_lr = X_train_raw_lr[colnames_to_fit_raw_lr]
    X_test_raw_lr = X_test_raw_lr[colnames_to_fit_raw_lr]
    raw_lr_model = LogisticRegression(
        penalty='l2',
        C=c_raw_lr,
        solver='lbfgs',
        max_iter=1000,
        fit_intercept=True
    ) # 设置LR参数
    raw_lr_model.fit(X_train_raw_lr, y_train) # 拟合模型
    
    # 构建WOE LR和评分卡模型
    X_train_woe_lr = X_train_woe_lr[colnames_to_fit_woe_lr]
    X_test_woe_lr = X_test_woe_lr[colnames_to_fit_woe_lr]
    woe_lr_model = LogisticRegression(
        penalty='l2',
        C=c_woe_lr,
        solver='lbfgs',
        max_iter=1000,
        fit_intercept=True
    ) # 设置LR参数
    woe_lr_model.fit(X_train_woe_lr, y_train) # 拟合模型
    sc_model = ScoreCard() # 初始化评分卡类
    sc_model.fit(woe_lr_model, X_train_woe_lr) # 拟合评分卡模型
    score_bins_train_df = sc_model.show_score_bins(X_train_woe_lr, y_train)
    score_bins_test_df = sc_model.show_score_bins(X_test_woe_lr, y_test)
    
    # 构建XGBoost
    X_train_xgb = X_train_xgb[colnames_to_fit_xgb]
    X_test_xgb = X_test_xgb[colnames_to_fit_xgb]
    xgb_model = xgb.XGBClassifier(
        **best_para_xgb_dict,
        random_state=500,
        eval_metric='auc',
        use_label_encoder=False
    ) # 设置XGBoost参数
    xgb_model.fit(
        X_train_xgb, y_train,
        verbose=False
    ) # 拟合模型
    # endregion
    
    """ 横向对比三种模型的拟合效果 """
    # region
    # 对比三种模型的拟合效果
    model_comparison_df = quantify_model_comparison(
        y_train, y_test,
        X_train_raw_lr, X_test_raw_lr, X_train_woe_lr, X_test_woe_lr, X_train_xgb, X_test_xgb,
        raw_lr_model, sc_model, xgb_model
    )
    fig_list = visualize_fit_goodness(
        y_test, 
        X_test_raw_lr, X_test_woe_lr, X_test_xgb,
        raw_lr_model, sc_model, xgb_model
    )
    # endregion
