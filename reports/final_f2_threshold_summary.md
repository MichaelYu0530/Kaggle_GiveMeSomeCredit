# Final F2 Threshold Summary

- 结果来源：run_final_report.py 测试集最佳 F2 / Precision / Recall 阈值摘要
- 运行入口：`run_final_report.py`
- Python：`3.13.13`
- 关键库版本：`pandas 2.3.3` / `numpy 2.2.6` / `scikit-learn 1.8.0` / `xgboost 3.2.0` / `shap 0.50.0` / `statsmodels 0.14.6`
- 该结果不覆盖 operation.py 历史导出文件。

| model              |   best_threshold |   best_f2 |   precision |   recall |   gain_at_10pct |   gain_at_25pct |   lift_at_10pct |   lift_at_25pct |
|:-------------------|-----------------:|----------:|------------:|---------:|----------------:|----------------:|----------------:|----------------:|
| Raw LR             |        0.0954774 |  0.511921 |    0.284008 | 0.640399 |        0.527182 |        0.772569 |         5.27164 |         3.09017 |
| WOE LR / ScoreCard |        0.0753769 |  0.514288 |    0.243317 | 0.712718 |        0.534663 |        0.776559 |         5.34646 |         3.10613 |
| XGBoost            |        0.0954774 |  0.523936 |    0.268288 | 0.687781 |        0.546633 |        0.790524 |         5.46615 |         3.16199 |
