# F2 Threshold Summary From operation.py

- 结果来源：operation.py 测试集最佳 F2 / Precision / Recall 阈值摘要
- 运行入口：`operation.py`
- Python：`3.13.13`
- 关键库版本：`pandas 2.3.3` / `numpy 2.2.6` / `scikit-learn 1.8.0` / `xgboost 3.2.0` / `shap 0.50.0` / `statsmodels 0.14.6`

| model              |   best_threshold |   best_f2 |   precision |   recall |   gain_at_10pct |   gain_at_25pct |   lift_at_10pct |   lift_at_25pct |
|:-------------------|-----------------:|----------:|------------:|---------:|----------------:|----------------:|----------------:|----------------:|
| Raw LR             |        0.0954774 |  0.511921 |    0.284008 | 0.640399 |        0.527182 |        0.772569 |         5.27164 |         3.09017 |
| WOE LR / ScoreCard |        0.0753769 |  0.514374 |    0.242031 | 0.715711 |        0.54015  |        0.782045 |         5.40132 |         3.12808 |
| XGBoost            |        0.0954774 |  0.524811 |    0.270053 | 0.686783 |        0.545636 |        0.793017 |         5.45618 |         3.17196 |
