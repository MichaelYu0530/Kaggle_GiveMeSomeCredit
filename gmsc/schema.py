"""Shared field names and feature groups."""

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

colnames_cn_map= {
    # 原始变量
    'Age': '年龄',
    'DebtRatio': '负债率',
    'MonthlyIncome': '月收入',
    'NumberOfDependents': '家属数',
    'NumberOfOpenCreditLinesAndLoans': '信贷数',
    'NumberOfTime30-59DaysPastDueNotWorse': '30-59天逾期次数',
    'NumberOfTime60-89DaysPastDueNotWorse': '60-89天逾期次数',
    'NumberOfTimes90DaysLate': '90+天逾期次数',
    'NumberRealEstateLoansOrLines': '房贷数',
    'RevolvingUtilizationOfUnsecuredLines': '信用额度使用率',
    # 衍生变量
    'short_late': '短期逾期次数',
    'late_severity_score': '逾期严重程度评分',
    'mortgage_ratio': '房贷占比',
    'credit_late_density': '逾期密度',
    'income_per_dep': '人均月收入',
    'monthly_debt': '月债务',
    'free_cashflow_income': '自由现金流收入',
    'credit_pressure_index' : '信用压力指数'
}

