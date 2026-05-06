# 环境说明

## 当前推荐环境

- 推荐 Python 版本：`3.13`
- 标准虚拟环境目录：`.venv`
- 推荐依赖文件：`requirements.txt`
- 完整锁定依赖文件：`requirements_lock.txt`
- 简洁依赖说明：`requirements_minimal.txt`

## 当前项目主要第三方依赖

以下库是当前项目中明确使用到的主要依赖：

- `numpy`
  - 数值计算

- `pandas`
  - 表格数据处理、统计汇总、分箱结果整理

- `scipy`
  - 显著性检验、统计分析

- `scikit-learn`
  - 逻辑回归、随机森林、交叉验证、常见评估函数

- `statsmodels`
  - `VIF` 等统计建模辅助工具

- `matplotlib`
  - 结果图绘制

- `seaborn`
  - 辅助可视化

- `xgboost`
  - 树模型训练

- `shap`
  - 树模型可解释性分析

- `ipython`
  - notebook / 交互环境显示支持

- `xlrd`
  - 读取 `Data Dictionary.xls`

- `jupyter`
  - notebook 运行与阅读

## 环境与兼容性说明

### 1. SHAP + XGBoost

当前项目的核心结果已经在以下两类环境输出中得到保留：

- `operation_full_output.ipynb`
- `reports/operation_full_output.md`
- `reports/operation_full_output.html`
- `reports/model_comparison_from_operation_py.md`
- `reports/f2_threshold_summary_from_operation_py.md`
- `reports/scorecard_bins_from_operation_py.md`
- `reports/operation_py_run_summary.md`

早期在部分 Linux / WSL 环境中：

- `xgboost` 可以完成模型训练
- 但 `shap.TreeExplainer(xgb_model)` 在项目现有代码路径下可能报错

这说明：

- SHAP 兼容性问题主要影响个别环境下的可解释性重算
- 不影响当前项目的 AUC、KS、F2、Gain/Lift、评分分箱等核心结果
- 当前推荐 Python 3.13 环境已能支撑核心结果展示
- 如需进一步追求跨环境一致性，可继续单独跟踪 SHAP 与依赖组合

### 2. Matplotlib 缓存目录

在部分 Linux / WSL 环境中，`matplotlib` 可能提示默认缓存目录不可写，并临时回退到 `/tmp`。这通常不影响运行，但会产生提示信息。

## requirements 文件说明

- `requirements.txt`
  - 当前项目推荐依赖文件
  - 适合作为标准安装入口

- `requirements_lock.txt`
  - 完整锁定依赖文件
  - 适合保留更完整的环境版本信息

- `requirements_minimal.txt`
  - 项目运行所需的简洁依赖清单
  - 更适合作为 GitHub 项目的轻量安装说明

## 当前环境口径说明

- 当前项目的完整实验结果由 Windows Python 3.13 notebook 导出文件保留
- 当前标准 Python 3.13 `.venv` 环境下的 `operation.py` 结果可作为结构化结果摘要引用
- 两套可信来源整体一致，轻微差异不影响主结论
- Linux / WSL 环境仍可用于辅助阅读、脚本开发和部分本地复现

## 建议的环境管理方式

- 使用独立 `.venv`
- 单独保留 `requirements.txt`、`requirements_lock.txt` 与 `requirements_minimal.txt`
- 在正式上传 GitHub 前补充轻量环境检查脚本
- 将“主模型结果复现”和“SHAP 可解释性复现”视为两个可分离步骤
- 后续可继续细化 Python、pandas、scikit-learn、xgboost、shap 等版本说明，以降低跨环境结果差异
