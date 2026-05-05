# 环境说明

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

## 当前已知兼容性问题

### 1. SHAP + XGBoost

当前项目的最终结果已经在 Windows Python 3.13 环境中完成完整重跑，并导出为：

- `operation_full_output.ipynb`
- `reports/operation_full_output.md`
- `reports/operation_full_output.html`

在部分 Linux / WSL 环境中：

- `xgboost` 可以完成模型训练
- 但 `shap.TreeExplainer(xgb_model)` 在项目现有代码路径下可能报错

这说明：

- SHAP 兼容性问题主要影响部分 Linux / WSL 环境下的 SHAP 重算
- 不影响 Windows notebook 已保存结果和核心模型评估指标
- 目前仍未确认应固定到哪一组版本组合，因此文档不编造“某版本必定兼容”的结论

### 2. Matplotlib 缓存目录

在部分 Linux / WSL 环境中，`matplotlib` 可能提示默认缓存目录不可写，并临时回退到 `/tmp`。这通常不影响运行，但会产生提示信息。

## 如何生成 requirements.txt

如果你希望从当前激活环境导出完整依赖快照，可使用：

```bash
pip freeze > requirements.txt
```

这会生成包含所有安装包及版本号的完整文件，更适合记录“当前环境状态”。

## requirements.txt 与 requirements_minimal.txt 的区别

- `requirements.txt`
  - 当前环境完整快照
  - 包含 notebook、Jupyter、widgets、底层依赖等大量包
  - 更适合环境归档，不一定适合展示

- `requirements_minimal.txt`
  - 项目运行所需的核心依赖清单
  - 更短、更清晰，适合作为 GitHub 项目最小安装说明

## 当前环境口径说明

- 当前项目的最终模型结果统一以 Windows Python 3.13 环境完整重跑产物为准
- Linux / WSL 环境仍可用于辅助阅读、脚本开发和部分本地复现
- 但若 Linux / WSL 环境下的复算结果与 Windows notebook 输出不一致，应优先采用 Windows `operation_full_output` 导出结果

## 建议的环境管理方式

- 使用独立 `.venv`
- 单独保留 `requirements.txt` 与 `requirements_minimal.txt`
- 在正式上传 GitHub 前补充轻量环境检查脚本
- 在修复 SHAP 问题前，先将“主模型结果复现”和“SHAP 可解释性复现”视为两个独立步骤
- 后续可进一步固定 Python、pandas、scikit-learn、xgboost、shap 等版本，以降低跨环境结果差异
