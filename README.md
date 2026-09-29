# Give Me Some Credit 风控建模项目

基于 Kaggle **Give Me Some Credit** 数据集，预测借款人未来两年内是否发生严重逾期（`SeriousDlqin2yrs`）。项目展示缺失与异常处理、风险特征、一维/二维分箱、WOE 评分卡、模型筛选及评估，并比较 Raw Logistic Regression、WOE Logistic Regression/ScoreCard 和 XGBoost。

## 结果速览

下表摘自仓库已有的 [最终模型比较报告](reports/final_model_comparison.md)。它是模块化迁移前的运行结果；本次代码拆分尚未在原始数据上重跑确认数值等价。

| 模型 | 内部测试集 AUC | AUC gap | 内部测试集 KS | KS gap |
|---|---:|---:|---:|---:|
| Raw LR | 0.8582 | 0.0013 | 0.5651 | -0.0045 |
| WOE LR / ScoreCard | 0.8596 | 0.0003 | 0.5698 | -0.0041 |
| XGBoost | 0.8662 | 0.0091 | 0.5824 | 0.0096 |

这里的**内部测试集**是从有标签的 `cs-training.csv` 分层留出的 20%，不是 Kaggle 的 `cs-test.csv`。其他现有产物包括 [F2 阈值摘要](reports/final_f2_threshold_summary.md)、[评分卡分箱](reports/final_scorecard_bins.md)和[运行摘要](reports/final_report_run_summary.md)。F2 表中的最佳阈值是在内部测试集上事后选出的，使用时请留意这一口径。

## 如何运行

推荐 Python 3.13。仓库不包含原始数据：自行下载 `cs-training.csv` 并放到项目根目录，随后安装依赖并运行定稿模型入口。

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_final_report.py
```

Windows 下将 `.venv/bin/python` 换成 `.venv\Scripts\python.exe`。原命令 `.venv/bin/python run_final_report.py` 仍可使用。定稿入口跳过特征和参数搜索，复用冻结配置，在 80% 训练验证数据上重新拟合三模型并写出 `reports/final_*`；最终 XGBoost 拟合不使用内部测试集做 early stopping。完整环境与运行说明见 [RUN_GUIDE.md](RUN_GUIDE.md) 和 [ENVIRONMENT.md](ENVIRONMENT.md)。

研究入口 `python scripts/run_research.py` 调用历史 `operation.py`，运行时间明显更长。`operation.ipynb` 用于交互式探索，`operation_full_output.ipynb` 及其 [Markdown 归档](reports/operation_full_output.md) 保留历史研究结果。两条研究入口的阶段开关不同，详见 [架构说明](docs/architecture.md)。

## 代码结构

| 路径 | 职责 |
|---|---|
| [`gmsc/`](gmsc/) | 清洗、特征、分箱、WOE、评分卡、统计评估、绘图、最终报告及定稿配置 |
| [`scripts/`](scripts/) | 最终报告与历史研究流程的可执行入口 |
| `pipeline.py`、`analysis.py`、`visualization.py` | 供旧 `operation.py` 和 notebook 使用的兼容导入层 |
| [`tests/`](tests/) | 小样本模块契约测试 |
| [`reports/`](reports/) 与 [`figures/`](figures/) | 已生成的结果表、历史 notebook 导出与展示图片 |

从数据到三模型的处理顺序、训练侧拟合边界见 [架构说明](docs/architecture.md)；逐文件函数与职责见 [功能模块说明](docs/modules.md)。

## 方法概览

1. 读取 `cs-training.csv`，处理收入/家属数缺失、逾期次数 `96/98` 特殊编码，以及收入、负债率、授信使用率异常值，并保留质量标记。
2. 构造逾期严重程度、逾期密度、人均收入、月债务、信用压力等衍生特征，建立一维/二维分箱和交互信号。
3. 在训练侧拟合分箱、WOE、Raw LR 中心化均值等参数，并对验证或内部测试数据应用已拟合结果。
4. 研究路径使用五折分层交叉验证进行变量与参数搜索；定稿路径直接读取 `gmsc/config/final_models.py` 中冻结的特征和参数。
5. 比较 AUC、KS、F2、Precision/Recall、Gain/Lift，并输出评分卡分数分箱；研究归档还包含图形和 SHAP 分析。

## 当前边界

- 模块化迁移已通过编译、导入和三项小样本测试；仓库缺少原始 CSV，尚未完成新旧入口的全数据预测与结果对照。
- `operation.py` 仍是历史研究编排脚本，notebook 也保留交互式实验代码。根目录兼容导入层仅服务于这两类旧入口。
- 现有分箱排名、测试集最佳 F2 阈值及评分卡训练/测试分箱仍有待单独修正的结果口径问题，详见 [架构说明](docs/architecture.md)。
- Kaggle 数据适合方法展示；实际风控使用仍需时间外验证、持续监控与更严格的业务评估。

项目方法及历史结论的展开版本见 [项目报告草稿](PROJECT_REPORT_DRAFT.md)，后续事项见 [TODO.md](TODO.md)。
