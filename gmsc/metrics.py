"""Metrics functions from the original research pipeline."""

from __future__ import annotations

from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd



def calculate_auc_ks(y_true: pd.Series, y_prob: pd.Series) -> tuple[float, float]:
    """
    计算AUC（Area Under ROC Curve）和KS（Kolmogorov-Smirnov）统计量
    
    AUC公式
    ----------
    AUC = ROC曲线下的面积，衡量模型区分正负样本的能力
    值域：[0.5, 1.0]，越接近1表示区分能力越强
    
    KS公式
    ----------
    KS = max(TPR - FPR)，衡量模型最大区分度
    值域：[0, 1.0]，通常大于0.3表示模型有较好区分能力
    
    参数
    ----------
    y_true : pd.Series
        真实标签（0/1）
    y_prob : pd.Series
        预测概率（0~1之间的浮点数）
    
    返回
    ----------
    tuple[float, float]
        AUC, KS构成的元组
    """
	
    # 计算AUC
    auc = roc_auc_score(y_true, y_prob)
    # 计算KS
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks =  np.max(tpr - fpr)
    
    return auc, ks


def calculate_top_recall(y_true: pd.Series, y_prob: pd.Series, top_proportion: float) -> float:
    df = pd.DataFrame({
        'y_true': y_true,
        'y_prob': y_prob
    }).sort_values('y_prob', ascending=False)
    
    div_idx = int(len(df) * top_proportion)
    df_top = df.iloc[0: div_idx] # 截取排名前列的样本
    
    total_bad = df['y_true'].sum()
    bad_top = df_top['y_true'].sum()
    recall_top = bad_top / total_bad
    
    return recall_top


def calculate_recall_precision(y_true: pd.Series, y_prob: pd.Series, threshold: float) \
    -> tuple[float, float]:
    """
    计算给定阈值下的召回率（Recall）和精确率（Precision）
    
    指标公式
    ----------
    Recall = TP / (TP + FN)  （真正例率，召回率）
    Precision = TP / (TP + FP)  （正预测值，精确率）
    
    参数
    ----------
    y_true : pd.Series
        真实标签（0/1）
    y_prob : pd.Series
        预测概率（0~1之间的浮点数）
    threshold : float
        分类阈值，概率大于该值预测为正类（1），否则为负类（0）
    
    返回
    ----------
    tuple[float, float]
        recall, precision构成的元组
    """
    
    y_pred = (y_prob > threshold).astype(int)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0) # 定义分母为0时precision为0
    return recall, precision


def calculate_gain_lift(y_true: pd.Series, y_prob: pd.Series) -> pd.DataFrame:
    """
    计算Gain和Lift曲线数据，用于评估模型排序能力
    
    Gain公式
    ----------
    Gain = 累计坏样本数 / 总坏样本数
    表示在给定比例的样本中，捕获了多少比例的坏样本
    
    Lift公式
    ----------
    Lift = Gain / 样本比例
    表示模型相对于随机选择的提升倍数
    
    参数
    ----------
    y_true : pd.Series
        真实标签（0/1）
    y_prob : pd.Series
        预测概率（0~1之间的浮点数）
    
    返回
    ----------
    pd.DataFrame
        包含 gain, lift, population 三列的结果表
        - gain: 累计坏样本比例
        - lift: 提升倍数
        - population: 累计样本比例
    """
    
	# 构建真实值与预测概率对照表
    metric_df = pd.DataFrame({
        'y_true': y_true,
        'y_prob': y_prob
    })
    
    # 排序（高风险在前）
    metric_df = metric_df.sort_values('y_prob', ascending=False).reset_index(drop=True)
    
    # 计算累计样本量
    metric_df['cum_bad'] = metric_df['y_true'].cumsum()
    metric_df['cum_good'] = (1 - metric_df['y_true']).cumsum()
    total_bad = metric_df['y_true'].sum()
    total_good = len(metric_df) - total_bad
    
    # 计算一系列模型评判指标
    metric_df['population'] = np.arange(1, len(metric_df) + 1) / len(metric_df) # 计算样本量累计占比
    metric_df['gain'] = metric_df['cum_bad'] / total_bad # 计算Gain
    metric_df['lift'] = metric_df['gain'] / metric_df['population'] # 计算Lift
    metric_df['cum_bad_rate'] = metric_df['cum_bad'] / total_bad
    metric_df['cum_good_rate'] = metric_df['cum_good'] / total_good
    metric_df['ks'] = metric_df['cum_bad_rate'] - metric_df['cum_good_rate'] # 计算KS（cdf差值）
    
    return metric_df
