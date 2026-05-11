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

项目当前同时保留三个不同定位的入口：

- `run_final_report.py`：当前 final model reproduction 的权威入口，用于快速复现最终三模型结果并导出 `reports/final_*`
- `operation_full_output.ipynb` 及其 `.md` / `.html` 导出文件：完整研究流程归档，保留探索性分析、特征筛选、调参与 notebook 路径下的完整输出
- `operation.py` / `operation.ipynb`：研究流程脚本与实验 notebook，对应完整研究与调参过程，不作为最终对外指标的唯一权威入口

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

- `run_final_report.py` 是当前推荐复现入口，也是对外引用最终指标时的标准入口
- `operation_full_output.ipynb` 及其 `.md` / `.html` 导出文件继续保留完整研究归档，适合人工审阅特征筛选、调参与 notebook 路径输出
- `operation.py` 主要承担完整研究与调参流程，不作为日常快速复现入口，也不应被视为比 `run_final_report.py` 更权威的 final report 入口
- final-only 脚本中的最终 XGBoost 拟合不使用 `eval_set=[(X_test_xgb, y_test)]`，也不使用 `early_stopping_rounds`
- `reports/final_*` 用于对外引用最终指标；`operation_full_output.*` 用于展示完整研究过程，局部指标若与 `final_*` 不完全相同，应理解为不同定位的归档产物而不是同一张最终结果表

## 数据泄漏防控

当前项目的最终主流程已明确区分训练侧拟合与验证/测试侧应用：

- `Raw LR` 中，中心化均值在训练侧拟合，验证/测试侧只复用同一组均值；对数特征列也由训练侧决定，验证/测试侧只复用该列名集合
- `WOE LR` 中，`WOE`、一维分箱、二维分箱以及 `late_severity_score` 等从数据中学习的统计量，都在训练集或交叉验证训练折上拟合
- 验证集与测试集只应用训练侧得到的映射、切分点和权重，不参与任何 `fit` 或统计量学习
- `XGBoost` 使用训练侧特征工程结果，不使用测试标签进行特征筛选或参数选择

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
| WOE LR / ScoreCard | 0.8596 | 0.0003 | 0.5698 | -0.0041 |
| XGBoost | 0.8662 | 0.0091 | 0.5824 | 0.0096 |

如需引用结果，请优先以：

- `reports/final_model_comparison.md`
- `reports/final_f2_threshold_summary.md`
- `reports/final_scorecard_bins.md`
- `reports/final_report_run_summary.md`

其中 `reports/final_*` 适合 README、简历和 GitHub 展示时直接引用；`operation_full_output.ipynb` / `.md` / `.html` 更适合作为完整研究流程归档；`operation.py` 导出的历史 `reports/` 更适合作为脚本路径下的辅助归档；`figures/` 主要作为辅助展示材料。

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

## 模型结论

- `XGBoost` 在当前最终结果中取得了最高的 `AUC` 和 `KS`，整体排序能力稍优，但 `train-valid/test gap` 也略高，可视为非线性强模型和性能上限参考
- `WOE LR / ScoreCard` 的 `AUC` / `KS` 略低于 `XGBoost`，但解释性、稳定性和评分卡表达更适合风控场景，适合作为信用评分卡 / 风控解释性模型
- `Raw LR` 作为原始数值与人工特征工程的线性基线，表现稳定，但非线性表达能力弱于 `XGBoost`，整体结果也略低于 `WOE LR / ScoreCard`

## 模型定稿说明

- `Raw LR` 与 `WOE LR` 的最终方案并不机械追求单一验证指标第一名；当更复杂的特征组合只带来很小的边际收益时，最终方案会综合考虑 `AUC / KS`、`gap`、`VIF`、系数符号稳定性和业务解释性
- `XGBoost` 参数搜索采用多轮顺序网格搜索。由于前一轮定稿参数会影响后一轮搜索空间，且领先参数组合之间的 `AUC / KS` 差距较小，最终参数不是机械采用每轮表格第一名，而是综合考虑验证集表现、`train-valid gap`、模型复杂度、正则化强度和稳定性后的人工定稿配置

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
- `Raw LR`、`WOE LR / ScoreCard`、`XGBoost` 三类模型均已完成比较，其中 `XGBoost` 的测试集表现约为 `AUC=0.8662`、`KS=0.5824`，整体排序能力稍优；`WOE LR / ScoreCard` 的测试集表现约为 `AUC=0.8596`、`KS=0.5698`，在性能略低于 `XGBoost` 的同时兼具较好的解释性与稳定性
- 早期部分 Linux / WSL 环境中曾出现过 SHAP 或依赖兼容性问题，但这类问题主要影响可解释性辅助模块，不影响当前 `AUC`、`KS`、`F2`、`Gain/Lift` 等核心结果引用；当前标准 Python 3.13 `.venv` 环境下 `run_final_report.py` 已成功运行并稳定导出 `reports/final_*`

## 已知限制

- 当前项目没有完整的 `pytest` 测试入口
- `operation.py` 是研究脚本，不是完全自动化 pipeline
- `XGBoost` 最终参数为多轮顺序搜索后的人工定稿，并不保证全局最优
- Kaggle 公开结构化数据集适合作为方法展示；真实业务落地仍需要更严格的时间外验证、监控和合规评估

更详细说明见 [PROJECT_REPORT_DRAFT.md](./PROJECT_REPORT_DRAFT.md)。

## 后续改进方向

- 进一步精简 README 和 GitHub 展示结构
- 继续优化“完整研究入口”和“final-only 复现入口”的职责边界
- 进一步区分 notebook 与脚本的职责
- 可选地继续优化 SHAP 与环境说明
