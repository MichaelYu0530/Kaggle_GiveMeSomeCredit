# 代码结构与运行边界

## 入口

- `scripts/run_final_report.py`：使用冻结的特征与参数训练最终三模型，输出 `reports/final_*`。
- `scripts/run_research.py`：运行历史研究脚本 `operation.py`，包含启用的五折搜索阶段，耗时较长。
- 仓库根目录的 `run_final_report.py` 和 `operation.py` 保留原有运行方式。
- `operation.ipynb`、`operation_full_output.ipynb` 保留交互式研究和历史结果；新建代码应直接调用 `gmsc` 模块。

## 功能模块

| 模块 | 职责 |
|---|---|
| `gmsc/schema.py`、`data.py` | 字段名、读取、导出、交叉验证划分 |
| `cleaning.py`、`features.py` | 数据质量标记、异常处理、衍生特征、交互信号 |
| `binning.py`、`woe.py` | 一维/二维分箱、区间排名、WOE/IV、PSI |
| `transforms.py` | Raw LR 的中心化、对数与列选择 |
| `scorecard.py` | 评分卡模型与评分分布 |
| `statistics.py`、`metrics.py` | 检验、VIF、AUC、KS、Recall、Gain、Lift |
| `selection.py` | 变量筛选、五折结果汇总与参数搜索 |
| `plotting_eda.py`、`plotting_evaluation.py`、`summaries.py` | 图形与分析表格 |
| `interpretability.py` | SHAP 重要性；只在实际调用时载入 SHAP |
| `config/final_models.py` | 唯一的 final-only 冻结特征、交互项与模型参数配置 |
| `final_report.py`、`reports.py` | 最终模型流程与结构化报告导出 |
| `legacy_income.py` | 保留已弃用的月收入填充实验，不参与最终流程 |

原 `pipeline.py`、`analysis.py`、`visualization.py` 是兼容导入层，供历史脚本与 notebook 使用。新代码使用 `from gmsc.<module> import <name>`，避免依赖星号导入。

## 训练与评估边界

固定种子 `50` 从有标签的 `cs-training.csv` 留出 20% 内部测试集。完整研究只在其余 80% 内做特征筛选和交叉验证；最终入口使用冻结配置，在 80% 数据拟合逾期权重、分箱、WOE、中心化均值及三模型，再在 20% 内部测试集评估。这里的内部测试集不是 Kaggle 的 `cs-test.csv`。

这次迁移保留现有函数计算体及最终报告主流程，不改变 `reports/final_*` 指标口径。以下问题需在单独的结果口径变更中处理：验证/测试数据的分箱排名由各自出现的区间重新编号；F2 最优阈值在内部测试集上选择；评分卡训练/测试分数分箱分别通过 `qcut` 拟合。修正这些行为后，应重新生成并标注新版本指标。

## 校验

```bash
python -m unittest discover -s tests -v
python -m compileall -q gmsc scripts
python scripts/run_final_report.py
```

最后一条需要自行从 Kaggle 获取 `cs-training.csv` 并放在仓库根目录。建议在原数据与推荐依赖环境中比较迁移前后的三模型预测及 `reports/final_*`，确认数值等价后再将新报告作为权威产物。
