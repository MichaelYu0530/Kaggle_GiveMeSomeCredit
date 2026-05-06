# Model Comparison From operation.py

- 结果来源：operation.py 最终三模型比较结果
- 运行入口：`operation.py`
- Python：`3.13.13`
- 关键库版本：`pandas 2.3.3` / `numpy 2.2.6` / `scikit-learn 1.8.0` / `xgboost 3.2.0` / `shap 0.50.0` / `statsmodels 0.14.6`

| model   |   test AUC |     AUC gap |   test KS |      KS gap |
|:--------|-----------:|------------:|----------:|------------:|
| raw LR  |   0.858176 |  0.00125024 |  0.565149 | -0.00446311 |
| WOE LR  |   0.861245 | -0.00132357 |  0.564128 | -0.00557842 |
| XGBoost |   0.86587  |  0.00673496 |  0.582983 |  0.00498007 |
