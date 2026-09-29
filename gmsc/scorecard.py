"""Scorecard functions from the original research pipeline."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from .metrics import calculate_auc_ks



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
