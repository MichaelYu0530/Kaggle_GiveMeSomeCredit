# 架构与数据流

## 两条运行路径

| 路径 | 入口 | 用途 |
|---|---|---|
| 定稿模型复现 | `scripts/run_final_report.py`（根目录 `run_final_report.py` 兼容） | 用冻结的特征、交互规则和参数训练三模型，写出 `reports/final_*` |
| 研究流程 | `scripts/run_research.py`（调用 `operation.py`） | 保留特征筛选、五折交叉验证与参数搜索的历史研究编排，耗时较长 |

`operation.ipynb` 是交互式研究版本；`operation_full_output.ipynb` 及 `reports/operation_full_output.*` 是历史输出归档。Notebook 与 `operation.py` 的阶段开关并不相同：当前 notebook 六段均启用，`operation.py` 中前两段关闭、后四段启用。`scripts/run_research.py` 只负责启动 `operation.py`，不会改变这些开关。

## 定稿复现的数据流

1. `gmsc.data.import_data` 读取有标签的 `cs-training.csv`；清洗函数先添加缺失、特殊逾期编码及极低收入标记。
2. `train_test_split` 使用 `random_state=50`、`stratify=SeriousDlqin2yrs`，从该文件留出 20% **内部测试集**。Kaggle 的 `cs-test.csv` 不参与现有最终报告。
3. 在其余 80% 上拟合 `late_severity_score` 的系数及分箱切点，再把这些值应用到内部测试集。衍生特征和交互标记由 `gmsc.features` 生成。
4. Raw LR 分支在训练侧确定中心化均值和对数列；WOE LR 分支在训练侧建立一维/二维分箱与 WOE 映射；XGBoost 分支把表示不可靠数值的负编码转为缺失值。
5. `gmsc.final_report` 依照 `gmsc.config.final_models` 的冻结特征和参数拟合 Raw LR、WOE LR/ScoreCard、XGBoost，由 `gmsc.reports` 导出模型对比、F2 阈值摘要、评分分箱及运行摘要。

研究流程在内部测试集之外的训练验证数据上使用五折划分做特征和参数探索。定稿入口不会重新执行搜索。完整函数与文件映射见 [模块说明](modules.md)。

## 模块边界

- `gmsc/` 存放可导入的计算与报告代码；`scripts/` 只提供运行入口。
- 根目录的 `pipeline.py`、`analysis.py`、`visualization.py` 是兼容导入层，让历史 `operation.py` 和 notebook 继续引用旧名称；新代码直接从对应 `gmsc` 模块显式导入。
- `gmsc/config/final_models.py` 是 **final-only 入口**使用的冻结配置。研究脚本和 notebook 仍保留各自历史实验常量，不能把三个入口视为完全相同的实验。
- `gmsc/legacy_income.py` 保存已弃用的月收入填充实验，不参与定稿流程。
- 当前多数数据处理函数会原地修改 DataFrame；部分清洗函数同时计算依赖标签的统计表。现有接口尚不是无需标签即可完整执行的独立推理流水线。

## 结果口径与待校验项

仓库中的 `reports/final_*` 来自模块化迁移前，在推荐 Python 3.13 环境中运行的定稿脚本。迁移保留了原函数计算体及最终报告 `main()` 主体，但仓库不包含原始 CSV，本次改动尚未在原数据上重跑并验证数值等价；不要将这次代码迁移描述为一次新的模型评估。

现有流程仍有三处需要单独修正和重新生成指标：

1. `convert_bins_value_to_rank` 按传入数据实际出现的区间独立编号；若内部测试集缺少某区间，交互信号的区间编号可能与训练侧不同。
2. `summarize_best_f2` 在内部测试集上寻找最大 F2 的阈值；表中的“最佳 F2”是测试集上的事后选择结果，不是事先锁定阈值的泛化估计。
3. `ScoreCard.show_score_bins` 对训练集和内部测试集分别运行 `qcut`；两部分表格的十分位区间不能当成相同固定分数区间比较。

这些口径改动应与结构迁移分开提交，并明确标注新旧报告版本。原始数据就位后的复现命令及环境说明见根目录 [RUN_GUIDE.md](../RUN_GUIDE.md)。
