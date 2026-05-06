# operation.py 运行摘要

- 运行入口：`operation.py`
- Python 路径：`/home/father_yuyue/projects/Kaggle_GiveMeSomeCredits/.venv_py313/bin/python`
- Python 版本：`3.13.13`
- pandas：`2.3.3`
- numpy：`2.2.6`
- scikit-learn：`1.8.0`
- xgboost：`3.2.0`
- shap：`0.50.0`
- statsmodels：`0.14.6`

## 输出文件

- `reports/model_comparison_from_operation_py.csv`
- `reports/model_comparison_from_operation_py.md`
- `reports/f2_threshold_summary_from_operation_py.csv`
- `reports/f2_threshold_summary_from_operation_py.md`
- `reports/scorecard_bins_from_operation_py.csv`
- `reports/scorecard_bins_from_operation_py.md`
- `reports/operation_py_run_summary.md`

## 最终模型比较表

| model   |   test AUC |     AUC gap |   test KS |      KS gap |
|:--------|-----------:|------------:|----------:|------------:|
| raw LR  |   0.858176 |  0.00125024 |  0.565149 | -0.00446311 |
| WOE LR  |   0.861245 | -0.00132357 |  0.564128 | -0.00557842 |
| XGBoost |   0.86587  |  0.00673496 |  0.582983 |  0.00498007 |

## F2 / Precision / Recall 阈值摘要

| model              |   best_threshold |   best_f2 |   precision |   recall |   gain_at_10pct |   gain_at_25pct |   lift_at_10pct |   lift_at_25pct |
|:-------------------|-----------------:|----------:|------------:|---------:|----------------:|----------------:|----------------:|----------------:|
| Raw LR             |        0.0954774 |  0.511921 |    0.284008 | 0.640399 |        0.527182 |        0.772569 |         5.27164 |         3.09017 |
| WOE LR / ScoreCard |        0.0753769 |  0.514374 |    0.242031 | 0.715711 |        0.54015  |        0.782045 |         5.40132 |         3.12808 |
| XGBoost            |        0.0954774 |  0.524811 |    0.270053 | 0.686783 |        0.545636 |        0.793017 |         5.45618 |         3.17196 |

## 评分卡分箱结果说明

- 当前共导出 `20` 行评分卡分箱结果（含 train / test 两部分）

## SHAP 状态

- SHAP 是否成功运行：`success`
