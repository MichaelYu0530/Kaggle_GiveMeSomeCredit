# Give Me Some Credit 风控建模项目

## 项目背景

本项目基于 Kaggle 经典信贷风控数据集 **Give Me Some Credit**，围绕个人借款人未来两年内是否会发生严重违约的问题，搭建了一套较完整的风控建模实验流程。项目重点不在于追求单一最优算法，而在于展示从数据理解、异常处理、特征工程、分箱建模、评分卡到机器学习模型比较的全过程。

该项目适合作为中文求职简历和 GitHub 作品集中的风控建模、金融科技、数据科学、机器学习方向项目。

## 数据集说明

- 数据来源：Kaggle `Give Me Some Credit`
- 训练集文件：`cs-training.csv`
- 测试集文件：`cs-test.csv`
- 数据字典：`Data Dictionary.xls`
- 说明：原始 CSV 和 Excel 文件体积较大，且属于数据集原始材料，不建议直接上传到公开 GitHub 仓库

## 业务目标

目标是基于借款人的授信使用、负债、收入、家属情况、逾期历史、信贷数量等信息，预测其未来两年内是否会出现 **90 天以上严重逾期/违约**，用于风险识别、客户排序和分层管理。

## 目标变量

- 字段名：`SeriousDlqin2yrs`
- 业务含义：借款人未来两年内是否发生过 `90+` 天严重逾期
- 任务类型：不平衡二分类

## 项目技术路线

项目整体流程与 `operation.py` / `operation.ipynb` 基本一致。当前项目同时保留：

- `operation.py` / `operation.ipynb` 对应的完整研究与调参流程
- `operation_full_output.ipynb` 及其导出文件，对应历史完整 notebook 归档
- `run_final_report.py` 对应的 final-only 复现入口，用于快速复现最终三模型结果并导出 `reports/final_*`

1. 导入原始数据并完成基础 EDA
2. 识别缺失值、特殊编码和极端值
3. 构造风险标记、衍生特征和交互特征
4. 构建一维分箱、二维分箱、WOE 编码
5. 使用 5 折分层交叉验证筛选变量与参数
6. 比较 `Raw Logistic Regression`、`WOE Logistic Regression / ScoreCard`、`XGBoost`
7. 输出 AUC / KS / F2 / Precision / Recall / Gain / Lift / 评分分箱等结果与可视化

## 项目结构

项目当前以单目录脚本形式组织，核心文件包括：

- `pipeline.py`：主数据流水线、特征工程、分箱、评分卡、模型训练与评估函数
- `analysis.py`：统计分析、AUC/KS/PSI/IV/SHAP 等指标函数
- `visualization.py`：EDA 与模型效果可视化
- `operation.py`：完整研究/调参主脚本，包含较长耗时的筛选、交叉验证和参数搜索流程
- `run_final_report.py`：final-only 快速复现入口，直接复用已定稿特征列表和参数，导出 `reports/final_*`
- `operation.ipynb`：实验 notebook 版本
- `figures/`：已保存的图像结果

更详细目录说明见 [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)。

## 核心功能模块

### 1. 数据读取与基础处理

- 导入训练集和测试集
- 删除无意义索引列
- 统一列名格式

### 2. 数据清洗与异常处理

- `MonthlyIncome` 缺失与极低值异常处理
- `NumberOfDependents` 缺失结构识别
- 三类逾期变量 `96/98` 特殊编码识别
- `RevolvingUtilizationOfUnsecuredLines` 极端值分层
- `DebtRatio` 极端值分层
- `Age` 异常样本删除

### 3. 风险特征工程

- `late_severity_score`
- `credit_late_density`
- `income_per_dep`
- `monthly_debt`
- `free_cashflow_income`
- `credit_pressure_index`
- 风险标记变量、行为标记变量、交互信号变量

### 4. 风控建模方法

- 一维分箱
- 二维分箱
- WOE 编码
- IV / PSI
- 评分卡 ScoreCard
- KS / Lift / Gain / PR / F2

## 数据清洗与特征工程

项目的清洗思路偏风控建模风格，不是简单删除或均值填补，而是强调：

- 将缺失与异常本身转化为风险信号
- 将不可靠原值与正常样本分开建模
- 尽量保留业务可解释性

例如：

- 用 `single_missing_flag`、`both_missing_flag` 标记收入缺失结构
- 用 `blacklist_flag` 标记逾期变量中的特殊编码人群
- 用 `income_anomaly_flag`、`util_anomaly_flag`、`debt_anomaly_flag` 标记明显异常值
- 用负编码将不可靠取值单独隔离，便于后续分箱和 WOE 处理

## 风控建模方法

项目体现了较完整的传统风控建模能力：

- 基于样本分布和业务语义进行一维分箱
- 基于风险差异和显著性筛选二维分箱交互项
- 对标记变量和分箱变量进行 WOE 编码
- 使用 IV、PSI、KS、风险跨度等指标评估变量质量
- 实现评分卡类并输出分数分箱违约率

## 模型训练与比较

项目当前主要比较三类模型：

- `Raw Logistic Regression`
- `WOE Logistic Regression / ScoreCard`
- `XGBoost`

代码中包含：

- 5 折分层交叉验证
- 多组特征组合比较
- 正则化与树模型参数搜索
- 训练集 / 验证集 gap 对比

说明：

- `operation.py` 主要承担完整研究与调参流程，不作为日常快速复现入口
- `run_final_report.py` 是当前推荐复现入口，会直接导出最终模型比较、F2 阈值摘要、评分卡分箱和运行摘要
- `operation_full_output.ipynb` 及其 `.md` / `.html` 导出文件继续保留完整历史实验上下文
- final-only 脚本中的最终 XGBoost 拟合不使用 `eval_set=[(X_test_xgb, y_test)]`，也不使用 `early_stopping_rounds`
- 不同可信来源之间只存在轻微数值差异，不影响主结论

## 模型评估指标

项目使用的主要评估指标包括：

- `AUC`
- `KS`
- `Precision / Recall`
- `F2`
- `Gain`
- `Lift`
- `PR Curve`
- `KS Curve`
- 评分分箱违约率
- `SHAP` 全局重要性

当前项目最终模型比较结果如下：

| model | test AUC | AUC gap | test KS | KS gap |
|---|---:|---:|---:|---:|
| Raw LR | 0.8582 | 0.0013 | 0.5651 | -0.0045 |
| WOE LR / ScoreCard | 0.8612 | -0.0013 | 0.5641 | -0.0056 |
| XGBoost | 0.8662 | 0.0091 | 0.5824 | 0.0096 |

如需引用结果，请优先以：

- `reports/final_model_comparison.md`
- `reports/final_f2_threshold_summary.md`
- `reports/final_scorecard_bins.md`
- `reports/final_report_run_summary.md`

其中 `reports/final_*` 适合 README、简历和 GitHub 展示时直接引用；`operation_full_output.ipynb` / `.md` / `.html` 与 `operation.py` 导出的旧 `reports/` 更适合作为研究流程与历史结果归档；`figures/` 主要作为辅助展示材料。

## 项目亮点

- 使用风控建模思路而非仅做通用二分类建模
- 同时覆盖传统评分卡与机器学习模型
- 将缺失、异常值和特殊编码转化为结构化风险信号
- 实现一维/二维分箱、WOE、IV、PSI、ScoreCard 等风控关键方法
- 保留了较完整的可视化产物，便于展示风险分层和模型效果

## 当前项目状态

当前项目已经完成 **端到端建模 + 初步工程化整理**，具备 GitHub 作品集展示基础：

- `operation_full_output.ipynb` / `.md` / `.html` 保留了完整实验输出归档
- `run_final_report.py` 已支持快速复现最终模型结果并导出 `reports/final_*`
- `reports/` 已形成“历史完整输出 + final-only 结果摘要”的双层归档
- `figures/` 当前仅保留主线展示图片

当前仍有进一步优化空间，例如 README 展示优化、脚本职责继续拆分和轻量测试补充。

详细待办见 [TODO.md](./TODO.md)。

## 如何运行

推荐先阅读：

1. `reports/final_model_comparison.md`
2. `PROJECT_REPORT_DRAFT.md`
3. `RUN_GUIDE.md`

如需实际运行：

```bash
.venv/bin/python run_final_report.py
```

注意：

- `run_final_report.py` 是当前推荐复现入口，适合快速得到最终模型结果
- `operation.py` 会触发较完整的研究与调参流程，耗时明显更长
- 如只需查看完整实验上下文，可优先阅读 `operation_full_output.ipynb` 和 `reports/operation_full_output.md`

## 环境要求

- 推荐 Python 版本：`3.13`
- 建议使用虚拟环境
- 依赖文件说明：
- `requirements.txt`：推荐依赖文件
- `requirements_lock.txt`：完整锁定依赖文件
- `requirements_minimal.txt`：简洁依赖说明

环境说明见 [ENVIRONMENT.md](./ENVIRONMENT.md)。

## 数据获取说明

- 原始数据来自 Kaggle `Give Me Some Credit`
- 如公开发布 GitHub 仓库，建议只保留代码、文档和小型结果图，不直接上传原始 CSV / Excel 数据文件
- 可在 README 中说明用户需自行从 Kaggle 下载并放置到项目根目录

## 结果说明

- 当前可信结果来源包括：
- `run_final_report.py` 导出的 `reports/final_*` 文件
- Windows Python 3.13 环境完整重跑后的 `operation_full_output.ipynb` 及其导出文件
- 当前标准 Python 3.13 `.venv` 环境下的 final-only 结果建议作为 README、简历和 GitHub 展示的主引用口径
- `Raw LR`、`WOE LR / ScoreCard`、`XGBoost` 三类模型均已完成比较，其中 `XGBoost` 的测试集表现约为 `AUC=0.8662`、`KS=0.5824`，排序区分能力最佳；`WOE LR / ScoreCard` 的测试集表现约为 `AUC=0.8612`、`KS=0.5641`，兼具较强效果和风控解释性
- 早期 Linux / WSL 环境中曾出现 SHAP 或依赖兼容性问题，但这类问题不影响当前项目的 AUC、KS、F2、Gain/Lift 等核心结果引用

更详细说明见 [PROJECT_REPORT_DRAFT.md](./PROJECT_REPORT_DRAFT.md)。

## 后续改进方向

- 进一步精简 README 和 GitHub 展示结构
- 继续优化“完整研究入口”和“final-only 复现入口”的职责边界
- 进一步区分 notebook 与脚本的职责
- 可选地继续优化 SHAP 与环境说明
