# Final Model Comparison

- 结果来源：run_final_report.py 最终三模型比较结果
- 运行入口：`run_final_report.py`
- Python：`3.13.13`
- 关键库版本：`pandas 2.3.3` / `numpy 2.2.6` / `scikit-learn 1.8.0` / `xgboost 3.2.0` / `shap 0.50.0` / `statsmodels 0.14.6`
- 该脚本跳过 grid search / 特征搜索 / 参数搜索，直接复用 operation.py 中已定稿的最终特征组合与参数字典。
- 最终 XGBoost 拟合未使用 test-set early stopping。

| model   |   test AUC |     AUC gap |   test KS |      KS gap |
|:--------|-----------:|------------:|----------:|------------:|
| raw LR  |   0.858176 |  0.00125024 |  0.565149 | -0.00446311 |
| WOE LR  |   0.861245 | -0.00132357 |  0.564128 | -0.00557842 |
| XGBoost |   0.86616  |  0.00912228 |  0.58237  |  0.00957148 |
