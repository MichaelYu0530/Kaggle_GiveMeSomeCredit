# final report 运行摘要

- 运行入口：`run_final_report.py`
- Python 路径：`/home/father_yuyue/projects/Kaggle_GiveMeSomeCredits/.venv/bin/python`
- Python 版本：`3.13.13`
- pandas：`2.3.3`
- numpy：`2.2.6`
- scikit-learn：`1.8.0`
- xgboost：`3.2.0`
- shap：`0.50.0`
- statsmodels：`0.14.6`
- 耗时（秒）：`638.83`
- 说明：该脚本跳过所有 grid search / 特征搜索 / 参数搜索，仅复用已定稿常量完成最终模型训练与结果导出。
- 说明：最终 XGBoost 拟合未使用 `eval_set=[(X_test_xgb, y_test)]`，也未使用 test-set early stopping。

## 输出文件

- `reports/final_model_comparison.csv`
- `reports/final_model_comparison.md`
- `reports/final_f2_threshold_summary.csv`
- `reports/final_f2_threshold_summary.md`
- `reports/final_scorecard_bins.csv`
- `reports/final_scorecard_bins.md`
- `reports/final_report_run_summary.md`

## 最终模型比较表

| model   |   test AUC |     AUC gap |   test KS |      KS gap |
|:--------|-----------:|------------:|----------:|------------:|
| raw LR  |   0.858176 |  0.00125024 |  0.565149 | -0.00446311 |
| WOE LR  |   0.861245 | -0.00132357 |  0.564128 | -0.00557842 |
| XGBoost |   0.86616  |  0.00912228 |  0.58237  |  0.00957148 |

## F2 / Precision / Recall 阈值摘要

| model              |   best_threshold |   best_f2 |   precision |   recall |   gain_at_10pct |   gain_at_25pct |   lift_at_10pct |   lift_at_25pct |
|:-------------------|-----------------:|----------:|------------:|---------:|----------------:|----------------:|----------------:|----------------:|
| Raw LR             |        0.0954774 |  0.511921 |    0.284008 | 0.640399 |        0.527182 |        0.772569 |         5.27164 |         3.09017 |
| WOE LR / ScoreCard |        0.0753769 |  0.514374 |    0.242031 | 0.715711 |        0.54015  |        0.782045 |         5.40132 |         3.12808 |
| XGBoost            |        0.0954774 |  0.523936 |    0.268288 | 0.687781 |        0.546633 |        0.790524 |         5.46615 |         3.16199 |

## 评分卡分箱结果说明

- 当前共导出 `20` 行评分卡分箱结果（含 train / test 两部分）

## 绘图状态

- 绘图状态：`skipped`
- 说明：为缩短 final-only 复现耗时，本脚本当前只导出 CSV / Markdown 结果，不生成新的模型效果图。
