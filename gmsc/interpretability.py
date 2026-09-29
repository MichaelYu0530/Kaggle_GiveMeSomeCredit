"""Interpretability functions from the original research pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb



def calculate_shap_importance(xgb_model: xgb.XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """
    计算SHAP特征重要性（平均绝对SHAP值）
    
    SHAP原理
    ----------
    SHAP值基于博弈论中的Shapley值，衡量每个特征对预测结果的贡献
    特征重要性 = mean(|SHAP值|)，反映特征对预测的影响强度
    
    参数
    ----------
    xgb_model : xgb.XGBClassifier
        已训练好的XGBoost模型
    X : pd.DataFrame
        特征数据（用于计算SHAP值的样本集）
    
    返回
    ----------
    pd.DataFrame
        包含 c 和 shap_importance 两列的结果表，按重要性降序排列
    """

    import shap
	
    explainer = shap.TreeExplainer(xgb_model) # 构造解释器
    shap_values = explainer.shap_values(X) # 计算SHAP值
    shap_importance = np.abs(shap_values).mean(axis=0) # 计算Mean |SHAP|
    
    # 构建指标与其SHAP importance的对照表
    shap_df = pd.DataFrame({
        'feature': X.columns,
        'shap_importance': shap_importance
    })
    # 按SHAP importance降序排序
    shap_df = shap_df.sort_values(by='shap_importance', ascending=False).reset_index(drop=True)
    
    return shap_df
