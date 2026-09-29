"""Plotting evaluation functions from the original research pipeline."""

from __future__ import annotations

from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import xgboost as xgb
from .interpretability import calculate_shap_importance

from .metrics import calculate_gain_lift, calculate_recall_precision

from .schema import age, credit, debt, late90, util



def visualize_fit_goodness(
    y_test: pd.Series,
    X_test_raw_lr: pd.DataFrame, X_test_woe_lr: pd.DataFrame, X_test_xgb: pd.DataFrame,
    raw_lr_model: LogisticRegression, sc_model, xgb_model: xgb.XGBClassifier,
    is_saving: bool=True, save_folder: str='figures'
) -> tuple[Figure, Figure, Figure]:
    # 计算测试集上的预测概率
    y_prob_test_raw_lr = raw_lr_model.predict_proba(X_test_raw_lr)[: , 1] # raw LR
    y_prob_test_woe_lr = sc_model.predict_proba(X_test_woe_lr)[: , 1] # WOE LR
    y_prob_test_xgb = xgb_model.predict_proba(X_test_xgb)[: , 1] # XGBoost
    
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    model_y_prob_map = {
        'Raw_LR': y_prob_test_raw_lr,
        'ScoreCard': y_prob_test_woe_lr,
        'XGBoost': y_prob_test_xgb
    }
    fig_list = []
    for model_name, y_prob_test in model_y_prob_map.items():
        # 创建画布
        fig, axes = plt.subplots(2, 3, figsize=(16, 12), gridspec_kw={'hspace': 0.4})  
        fig.suptitle(f'{model_name} Fit Goodness Visualization', fontsize=16)
        
        # 列名到绘图函数的映射
        plot_functions = {
            'P-R Curve': plot_pr_curve,
            'F2-Score Curve': plot_f2_curve,
            'Gain Curve': plot_gain_curve,
            'Lift Curve': plot_lift_curve,
            'KS Chart': plot_ks_chart,
        }
        
        # 对各指标（列名）作图
        for i, plot_func in enumerate(plot_functions.values()):
            ax = axes.flat[i]
            plot_func(y_test, y_prob_test, model_name, ax)
        if model_name == 'Raw_LR':
            fig.delaxes(axes[1, 2])
        if model_name == 'ScoreCard':
            ax = axes.flat[i + 1]
            plot_score_bins_bar(sc_model, X_test_woe_lr, y_test, ax)
        if model_name == 'XGBoost':
            ax = axes.flat[i + 1]
            plot_shap_bar(X_test_xgb, xgb_model, ax)
        
        fig_list.append(fig)
        
        plt.tight_layout()
        # 保存文件
        if is_saving:
            os.makedirs(save_folder, exist_ok=True)
            filename = f'{save_folder}/{model_name}_fit_goodness_visualization.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig_list


def plot_pr_curve(y_true: pd.Series, y_prob: pd.Series, model_name: str, ax: plt.Axes) -> None:
    # 构造阈值序列
    threshold_list = np.linspace(0, 1, 200)
    
    recall_list = []
    precision_list = []
    f2_score_list = []
    for threshold in threshold_list:
        # 计算Recall、Precision、F2-Score
        recall, precision = calculate_recall_precision(y_true, y_prob, threshold)
        recall_list.append(recall)
        precision_list.append(precision)
        if recall + precision == 0:
            f2_score = 0
        else:
            beta = 2
            f2_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        f2_score_list.append(f2_score)
    
    # 转换为array形式
    recall_arr = np.array(recall_list)
    precision_arr = np.array(precision_list)
    f2_score_arr = np.array(f2_score_list)
    
    # 根据Recall排序
    sorted_idx = np.argsort(recall_arr)
    recall_arr = recall_arr[sorted_idx]
    precision_arr = precision_arr[sorted_idx]
    f2_score_arr = f2_score_arr[sorted_idx]
    threshold_list = threshold_list[sorted_idx]
    
    # 找出F2-Score最大值点
    best_idx = np.argmax(f2_score_arr)
    best_recall = recall_arr[best_idx]
    best_precision = precision_arr[best_idx]
    
    # 绘图
    ax.plot(recall_arr, precision_arr)
    # 注明F2-Score最大值点
    ax.scatter(
        recall_arr[best_idx],
        precision_arr[best_idx],
        marker='o',
        label='Best F2-Score Point'
    )
    # 由标注点画两条到坐标轴的垂线
    ax.plot([best_recall, best_recall], [0, best_precision], linestyle='--', alpha=0.6, color='red')
    ax.plot([0, best_recall], [best_precision, best_precision], linestyle='--', alpha=0.6, color='green')
    # 在标注点附近添加文本框标注
    ax.annotate(
        f"Precision = {best_precision:.3f}\nRecall = {best_recall:.3f}",
        xy=(best_recall, best_precision),
        xytext=(best_recall - 0.32, best_precision - 0.08),
        fontsize=9,
        bbox=dict(
            boxstyle="round, pad=0.3",
            fc="white",
            ec="gray",
            alpha=0.9
        )
    )
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'PR Curve ({model_name})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(True)


def plot_f2_curve(y_true: pd.Series, y_prob: pd.Series, model_name: str, ax: plt.Axes) -> None:
    # 构造阈值序列
    thresholds = np.linspace(0, 1, 200)
    
    beta = 2
    f2_score_list = []
    for threshold in thresholds:
        # 计算Recall、Precision、F2-Score
        recall, precision = calculate_recall_precision(y_true, y_prob, threshold)
        if recall + precision == 0:
            f2_score = 0
        else:
            f2_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        f2_score_list.append(f2_score)
    
    # 找出F2-Score最大值点
    f2_score_arr = np.array(f2_score_list)
    best_index = np.argmax(f2_score_arr)
    best_threshold = thresholds[best_index]
    max_f2_score = f2_score_arr[best_index]
    
    # 绘图
    ax.plot(thresholds, f2_score_arr)
    # 注明F2-Score最大值点
    ax.scatter(best_threshold, f2_score_arr[best_index], label=f'Best F2-Score Point')
    # 由标注点画两条到坐标轴的垂线
    ax.plot([best_threshold, best_threshold], [0, max_f2_score], linestyle='--', alpha=0.6, color='red')
    ax.plot([0, best_threshold], [max_f2_score, max_f2_score], linestyle='--', alpha=0.6, color='green')
    # 在标注点附近添加文本框标注
    ax.annotate(
        f"F2-Score = {max_f2_score:.3f}\nThreshold = {best_threshold:.3f}",
        xy=(best_threshold, max_f2_score),
        xytext=(best_threshold + 0.03, max_f2_score + 0.02),
        fontsize=9,
        bbox=dict(
            boxstyle="round, pad=0.3",
            fc="white",
            ec="gray",
            alpha=0.9
        )
    )
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    ax.set_title(f'F{beta}-Score-Threshold Curve ({model_name})')
    ax.set_xlabel('Threshold')
    ax.set_ylabel(f'F2-Score')
    ax.legend()
    ax.grid(True)


def plot_gain_curve(y_true: pd.Series, y_prob: pd.Series, model_name: str, ax: plt.Axes) -> None:
    # 直接得出Gain等评价指标的数值表
    gain_df = calculate_gain_lift(y_true, y_prob)[['population', 'gain', 'ks']]
    # 找出KS点及对应的Gain、Population
    ks_idx = gain_df['ks'].idxmax()
    ks_value = gain_df.loc[ks_idx, "ks"]
    best_gain = gain_df.loc[ks_idx, 'gain']
    best_population = gain_df.loc[ks_idx, 'population']
    
    # 绘图
    ax.plot(gain_df['population'], gain_df['gain'], label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', label='Random')
    # 标记KS点
    ax.scatter(best_population, best_gain, label=f'KS={ks_value:.3f}')
    # 由标注点画两条到坐标轴的垂线
    ax.plot([best_population, best_population], [0, best_gain], linestyle='--', alpha=0.6, color='red')
    ax.plot([0, best_population], [best_gain, best_gain], linestyle='--', alpha=0.6, color='green')
    # 在标注点附近添加文本框标注
    ax.annotate(
        f"Gain = {best_gain:.3f}\nPopulation = {best_population:.3f}",
        xy=(best_population, best_gain),
        xytext=(best_population + 0.03, best_gain - 0.08),
        fontsize=9,
        bbox=dict(
            boxstyle="round, pad=0.3",
            fc="white",
            ec="gray",
            alpha=0.9
        )
    )
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Gain Curve ({model_name})')
    ax.set_xlabel('Population')
    ax.set_ylabel('Gain')
    ax.legend()
    ax.grid(True)


def plot_lift_curve(y_true: pd.Series, y_prob: pd.Series, model_name: str, ax: plt.Axes) -> None:
    # 直接求出Lift
    lift_df = calculate_gain_lift(y_true, y_prob)[['population', 'lift']]
    
    # 绘图
    ax.plot(lift_df['population'], lift_df['lift'])
    ax.axhline(1, linestyle='--', color='crimson', alpha=0.7)  # 基准线
    
    # 标注的百分位点
    highlight_points = [0.10, 0.25]
    for pt in highlight_points:
        # 找到最接近的实际人口比例的索引
        idx = (lift_df['population'] - pt).abs().idxmin()
        population_value = lift_df.loc[idx, 'population']
        lift_value = lift_df.loc[idx, 'lift']
        
        # 标记前10%与前25%点
        ax.scatter(population_value, lift_value, s=60, color='red', zorder=5)
        # 添加文本框标注
        ax.annotate(
            f"Population={pt:.0%}\nLift={lift_value:.2f}",
            xy=(population_value, lift_value),
            xytext=(population_value + 0.04, lift_value + 0.06),
            fontsize=9,
            bbox=dict(boxstyle="round, pad=0.3", fc="white", ec="gray", alpha=0.9)
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, lift_df['lift'].max() + 1)
    ax.set_title(f'Lift Curve ({model_name})')
    ax.set_xlabel('Population')
    ax.set_ylabel('Lift')
    ax.grid(True)


def plot_ks_chart(y_true: pd.Series, y_prob: pd.Series, model_name: str, ax: plt.Axes) -> None:
    # 建立真实值与预测概率的对照表
    df = pd.DataFrame({'y': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    
    # 计算KS及好/坏样本的cdf
    total_bad = df['y'].sum()
    total_good = len(df) - total_bad
    df['bad_cdf'] = df['y'].cumsum() / total_bad
    df['good_cdf'] = (1 - df['y']).cumsum() / total_good
    df['ks'] = df['bad_cdf'] - df['good_cdf']
    ks_idx = df['ks'].idxmax()
    best_bad_cdf = df.loc[ks_idx, 'bad_cdf']
    best_good_cdf = df.loc[ks_idx, 'good_cdf']
    best_population = ks_idx / len(df)
    
    # 绘图
    ax.plot(df.index / len(df), df['bad_cdf'], label='Bad CDF')
    ax.plot(df.index / len(df), df['good_cdf'], label='Good CDF')
    # 画出取到KS值的水平线
    ax.vlines(
        x=best_population, ymin=best_good_cdf, ymax=best_bad_cdf,
        linestyles='--', label=f'KS={df.loc[ks_idx,"ks"]:.3f}'
    )
    # 标记取到KS值的好坏CDF曲线上的点
    ax.scatter(best_population, best_good_cdf, s=60, color='red', zorder=5)
    ax.scatter(best_population, best_bad_cdf, s=60, color='red', zorder=5)
    # 添加文本框标注
    ax.annotate(
        f"Population={best_population:.2%}\nGood CDF={best_good_cdf:.2f}",
        xy=(best_population, best_good_cdf),
        xytext=(best_population + 0.03, best_good_cdf - 0.08),
        fontsize=9,
        bbox=dict(boxstyle="round, pad=0.3", fc="white", ec="gray", alpha=0.9)
    )
    ax.annotate(
        f"Population={best_population:.2%}\nBad CDF={best_bad_cdf:.2f}",
        xy=(best_population, best_bad_cdf),
        xytext=(best_population + 0.03, best_bad_cdf - 0.08),
        fontsize=9,
        bbox=dict(boxstyle="round, pad=0.3", fc="white", ec="gray", alpha=0.9)
    )
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'KS Curve ({model_name})')
    ax.set_xlabel('Population')
    ax.set_ylabel('CDF')
    ax.legend()
    ax.grid(True)


def plot_score_bins_bar(sc_model, X: pd.DataFrame, y: pd.Series, ax: plt.Axes, n_bins: int=10) -> None:
    # 直接得出各分箱违约率对照表
    score_bins_df = sc_model.show_score_bins(X, y, n_bins=n_bins)
    bin_labels = score_bins_df['score_bin']
    bin_labels_short = [f"{bin.left:.1f}-{bin.right:.1f}" for bin in bin_labels]
    default_rate_series = score_bins_df['bad_rate'] * 100
    
    # 绘图
    bars = ax.bar(bin_labels_short, default_rate_series)
    
    ax.set_ylim(0, 40)
    ax.set_xticklabels(bin_labels_short, rotation=45, ha='right')
    for bar, default_rate in zip(bars, default_rate_series):
        ax.text(
            x=bar.get_x() + bar.get_width() / 2,
            y=default_rate,
            s=f'{default_rate:.2f}%',
            ha='center',
            va='bottom',
            fontsize=9
        )
    ax.set_title(f'Default Rate Among Score Bins')
    ax.set_xlabel('Score bins')
    ax.set_ylabel('Default Rate')
    
    ax.grid(True)


def plot_shap_bar(X_test: pd.DataFrame, xgb_model: xgb.XGBClassifier, ax: plt.Axes, top_n: int=10) -> None:
    # 增设部分衍生指标的缩写
    colnames_abbr_map_here =  {
        age: 'Age',
        util: 'Util',
        debt: 'DR',
        credit: 'Credit',
        late90: 'late90',
        'late_severity_score': 'LSS',
        'short_late': 'SL',
        'free_cashflow_income': 'FCI',
        'credit_late_density': 'CLD',
        'credit_pressure_index': 'CPI',
        'mortgage_ratio': 'MR'
    }
    
    # 直接得出指标与SHAP importance的对照表
    shap_df = calculate_shap_importance(xgb_model, X_test)
    shap_df = shap_df.sort_values('shap_importance', ascending=False).head(top_n)
    feature_series = shap_df['feature'].iloc[::-1]
    feature_abbr_series = feature_series.apply(lambda x: colnames_abbr_map_here[x])
    shap_importance_series = shap_df['shap_importance'].iloc[::-1]
    
    # 绘图
    bars = ax.barh(feature_abbr_series, shap_importance_series)
    # 在每个横柱右侧展示数值
    for bar, shap_importance in zip(bars, shap_importance_series):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{shap_importance:.3f}",
            va='center',
            ha='left',
            fontsize=9
        )
    
    ax.set_xlim(0, 0.7)
    ax.set_title(f'SHAP Importance (XGBoost)')
    ax.set_xlabel('Mean |SHAP|')
    ax.set_ylabel('Features')
    ax.grid(True)


def show_pred_effect_visualization(
    y_true: np.ndarray, target_name: str, name_to_y_pred: dict, save_folder: str='figures'
) -> None:
    """
    可视化模型的预测效果，绘制散点图、误差分布图、核密度估计图[已弃用]
    
    绘制内容
    ----------
    1. 真实值vs预测值散点图（每个模型/方法一个子图）
    2. 残差分布直方图（每个模型/方法一个子图）
    3. 真实值与预测值的核密度估计对比图
    
    参数
    ----------
    y_true : array-like
        目标变量的真实值
    target_name : str
        目标变量名称，用于图表标签
    name_to_y_pred : dict
        字典，键为模型/方法名称，值为对应的预测值数组
    save_folder : str, default='figures'
        保存的文件夹名
    
    返回
    ----------
    None
        直接保存图形文件，无返回值
    """
    
    # 确保存储图片的文件夹存在
    os.makedirs(save_folder, exist_ok=True)
    
    # 绘制真实vs预测散点图
    plt.figure(figsize=(12, 5))
    for i, (name, y_pred) in enumerate(name_to_y_pred.items()):
        plt.subplot(1, 2, i + 1)
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.plot([0, y_true.max()], [0, y_true.max()], 'r--')
        
        plt.title(name)
        plt.xlabel(f"{target_name}真实值")
        plt.ylabel(f"{name}预测值")
        
        true_max_lim, pred_max_lim = np.quantile(y_true, 0.99), np.quantile(y_pred, 0.995) # 避免异常大值干扰
        max_lim =max(true_max_lim, pred_max_lim)
        plt.xlim(0, max_lim) 
        plt.ylim(0, max_lim) # x, y轴范围应一致，保证直线y=x不变形
    
    plt.tight_layout()
    filename1 = f'{save_folder}/2Dbin_vs_RF_scatter.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制残差分布图
    plt.figure(figsize=(12, 5))
    for i, (name, y_pred) in enumerate(name_to_y_pred.items()):
        plt.subplot(1, 2, i + 1)
        
        residual = y_true - y_pred
        q1, q99 = np.quantile(residual, 0.01), np.quantile(residual, 0.99)
        mask = (residual >= q1) & (residual <= q99) # 除去异常大值的筛选器
        
        sns.histplot(residual[mask], bins=50, kde=True)
        plt.title(f"{name}误差分布")
        plt.xlim(q1, q99)
    
    plt.tight_layout()
    filename2 = f'{save_folder}/2Dbin_vs_RF_residual.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制核密度估计图
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_true, label='真实值', linewidth=2)
    for name, y_pred in name_to_y_pred.items():
        sns.kdeplot(y_pred, label=name)
    
    plt.legend()
    plt.title(f"{target_name}分布对比")
    plt.xlabel(target_name)
    plt.ylabel("分布密度")
    plt.xlim(0, np.quantile(y_true, 0.99))
    
    plt.tight_layout()
    filename3 = f'{save_folder}/2Dbin_vs_RF_KDE.png'
    plt.savefig(filename3, dpi=300, bbox_inches='tight')
    plt.close()
