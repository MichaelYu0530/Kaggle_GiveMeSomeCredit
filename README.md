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

项目整体流程与 `operation.py` / `operation.ipynb` 基本一致：

1. 导入原始数据并完成基础 EDA
2. 识别缺失值、特殊编码和极端值
3. 构造风险标记、衍生特征和交互特征
4. 构建一维分箱、二维分箱、WOE 编码
5. 使用 5 折分层交叉验证筛选变量与参数
6. 比较 `Raw Logistic Regression`、`WOE Logistic Regression / ScoreCard`、`XGBoost`
7. 输出 PR / KS / Lift / Gain / 评分分箱等可视化结果

## 项目结构

项目当前以单目录脚本形式组织，核心文件包括：

- `pipeline.py`：主数据流水线、特征工程、分箱、评分卡、模型训练与评估函数
- `analysis.py`：统计分析、AUC/KS/PSI/IV/SHAP 等指标函数
- `visualization.py`：EDA 与模型效果可视化
- `operation.py`：实验主脚本
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

- 当前仓库内 `Raw LR` 与 `XGBoost` 的复算结果较稳定
- `ScoreCard` 相关图像与当前环境复算结果存在版本不同步现象，使用时建议结合 notebook 记录和项目报告交叉核对

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

若部分指标在当前环境下无法完全复算，请以：

- 已保存图像 `figures/`
- `operation.ipynb` 已保存输出
- 项目报告草稿

为准，不建议自行补写未经确认的数值。

## 项目亮点

- 使用风控建模思路而非仅做通用二分类建模
- 同时覆盖传统评分卡与机器学习模型
- 将缺失、异常值和特殊编码转化为结构化风险信号
- 实现一维/二维分箱、WOE、IV、PSI、ScoreCard 等风控关键方法
- 保留了较完整的可视化产物，便于展示风险分层和模型效果

## 当前项目状态

当前项目属于 **可运行、可阅读、可复算部分关键结果** 的实验型项目，适合作为作品集展示，但仍有一些工程化工作待补充：

- 尚未沉淀统一的最终结果表
- 尚未拆分出轻量复现脚本
- SHAP 与当前 XGBoost 版本存在兼容性问题
- 部分中间结果只保存在 notebook 中

详细待办见 [TODO.md](./TODO.md)。

## 如何运行

推荐先阅读：

1. `operation.ipynb`
2. `PROJECT_REPORT_DRAFT.md`
3. `RUN_GUIDE.md`

如需实际运行：

```bash
python operation.py
```

注意：

- `operation.py` 会触发较完整的实验流程，耗时可能较长
- 不建议在未了解流程前直接运行完整训练
- 优先建议先查看 notebook 已保存输出和 `figures/` 中已有图像

## 环境要求

- 推荐 Python 版本：`3.10`
- 建议使用虚拟环境
- 主要依赖见：
- `requirements_minimal.txt`：项目核心依赖
- `requirements.txt`：当前环境完整依赖快照

环境说明见 [ENVIRONMENT.md](./ENVIRONMENT.md)。

## 数据获取说明

- 原始数据来自 Kaggle `Give Me Some Credit`
- 如公开发布 GitHub 仓库，建议只保留代码、文档和小型结果图，不直接上传原始 CSV / Excel 数据文件
- 可在 README 中说明用户需自行从 Kaggle 下载并放置到项目根目录

## 结果说明

- 部分关键指标可以在当前环境下重新复算
- 部分结果以 `figures/` 图像和 notebook 已保存输出为准
- 如果同一模型在“已保存图像”和“当前代码复算”之间出现差异，应优先标注为版本未完全同步，而不是强行选择某一组数字

更详细说明见 [PROJECT_REPORT_DRAFT.md](./PROJECT_REPORT_DRAFT.md)。

## 后续改进方向

- 将 notebook 中已保存结果抽取为独立报告
- 增加轻量环境检查与结果复现脚本
- 固定更稳定的依赖环境
- 解决 SHAP 与 XGBoost 的兼容性问题
- 将长耗时训练流程与结果读取流程拆开
- 补充更清晰的最终结果表与 benchmark 对比

