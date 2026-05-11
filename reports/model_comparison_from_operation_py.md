# Model Comparison From operation.py

- 结果来源：operation.py 最终三模型比较结果
- 运行入口：`operation.py`
- 定位：`historical / research snapshot`
- 说明：该文件保留 `operation.py` 研究流程脚本路径下的结构化导出结果，用于研究归档与人工审阅。
- 说明：当前对外 `headline metrics` 请以 `run_final_report.py` 导出的 `reports/final_model_comparison.md` 为权威来源。
- 说明：若本文件中的局部指标与 `reports/final_*` 不完全一致，应以 `reports/final_model_comparison.md` 为最终对外模型对比口径。
- Python：`3.13.13`
- 关键库版本：`pandas 2.3.3` / `numpy 2.2.6` / `scikit-learn 1.8.0` / `xgboost 3.2.0` / `shap 0.50.0` / `statsmodels 0.14.6`

| model   |   test AUC |     AUC gap |   test KS |      KS gap |
|:--------|-----------:|------------:|----------:|------------:|
| raw LR  |   0.858176 |  0.00125024 |  0.565149 | -0.00446311 |
| WOE LR  |   0.861245 | -0.00132357 |  0.564128 | -0.00557842 |
| XGBoost |   0.86587  |  0.00673496 |  0.582983 |  0.00498007 |
