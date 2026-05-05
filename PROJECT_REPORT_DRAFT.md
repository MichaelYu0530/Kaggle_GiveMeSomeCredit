# 项目报告草稿

## 项目总览

本项目基于 Kaggle `Give Me Some Credit` 数据集，围绕“借款人未来两年内是否发生严重违约”构建完整风控建模流程。项目兼顾传统风控评分卡方法与机器学习方法，目标是展示数据理解、风险特征工程、风控建模、模型比较和结果解释能力。

## 数据理解

### 数据来源

- Kaggle `Give Me Some Credit`
- 原始文件：
  - `cs-training.csv`
  - `cs-test.csv`
  - `Data Dictionary.xls`

### 数据规模

基于当前项目读取与核对：

- 训练集样本数：`150,000`
- 测试集样本数：`101,503`
- 目标变量违约率约：`6.684%`

### 目标变量

- `SeriousDlqin2yrs`
- 含义：未来两年内是否发生过 `90+` 天严重逾期

### 字段理解

核心字段包括：

- `RevolvingUtilizationOfUnsecuredLines`
- `DebtRatio`
- `MonthlyIncome`
- `NumberOfDependents`
- `NumberOfOpenCreditLinesAndLoans`
- `NumberRealEstateLoansOrLines`
- 三类逾期次数变量

字段含义已在 `Data Dictionary.xls` 中确认。

## 数据清洗

项目采用“保留风险信号 + 隔离异常值”的风控式清洗思路。

### 缺失值处理

- `MonthlyIncome` 是最主要缺失字段之一
- `NumberOfDependents` 也存在缺失
- 项目用 `single_missing_flag`、`both_missing_flag` 区分缺失结构

### 特殊编码处理

- 三个逾期变量中的 `96/98` 被识别为同一批特殊编码样本
- 通过 `blacklist_flag` 显式保留该人群标签

### 异常值处理

- `MonthlyIncome` 的 `0-10` 被视为异常低值
- `RevolvingUtilizationOfUnsecuredLines` 的极端大值被单独标记
- `DebtRatio` 的极端大值被单独标记
- `Age < 18` 的异常样本被删除

## 特征工程

### 衍生特征

项目构造了多类风险衍生特征，包括：

- `late_severity_score`
- `short_late`
- `credit_late_density`
- `income_per_dep`
- `monthly_debt`
- `free_cashflow_income`
- `credit_pressure_index`
- `mortgage_ratio`

### 风险标记

- `blacklist_flag`
- `income_anomaly_flag`
- `util_anomaly_flag`
- `debt_anomaly_flag`
- `is_util_high`
- `is_util_overlimit`
- `is_debt_high`
- `is_debt_overlimit`

### 交互项

项目还筛选并保留了一组交互信号变量，用于增强模型对组合风险的识别能力。

## 一维分箱 / 二维分箱 / WOE / IV / PSI

### 一维分箱

- 对连续变量做分箱处理
- 对 0 附近高度聚集的变量做特殊处理
- 对异常值和缺失值单独成箱

### 二维分箱

- 基于一维分箱结果进一步构造二维风险格子
- 结合样本量门槛避免产生过小分箱

### WOE / IV / PSI

- 使用 WOE 将分箱与标记变量转化为适合 LR/评分卡的输入
- 使用 IV 评估信息量
- 使用 PSI 评估训练/验证分布稳定性

### 说明

这些内容主要来源于：

- `pipeline.py`
- `analysis.py`
- `operation.py`

## Raw LR、ScoreCard、XGBoost 模型比较

### Raw LR

- 使用中心化特征和对数变换特征
- 保留了较强的统计模型可解释性

### ScoreCard / WOE LR

- 使用 WOE 编码后的特征进入 LR
- 再由自定义 `ScoreCard` 类生成评分卡表达

### XGBoost

- 在清洗和特征工程后的数值特征上训练
- 用于补充非线性表达能力

### 当前结果说明

#### 当前最终结果来源

当前项目的最终模型结果统一以 Windows Python 3.13 环境完整重跑后的：

- `operation_full_output.ipynb`
- `reports/operation_full_output.md`
- `reports/operation_full_output.html`

为准。

#### 最终模型比较结果

| model | test AUC | AUC gap | test KS | KS gap |
|---|---:|---:|---:|---:|
| Raw LR | 0.8581 | 0.0012 | 0.5671 | -0.0066 |
| WOE LR / ScoreCard | 0.8612 | -0.0013 | 0.5641 | -0.0056 |
| XGBoost | 0.8659 | 0.0074 | 0.5825 | 0.0067 |

#### 结果解释

- `XGBoost` 在三类模型中测试集 `AUC` 与 `KS` 最优，排序区分能力最佳
- `WOE LR / ScoreCard` 的测试集 `AUC=0.8612`、`KS=0.5641`，在保持较强效果的同时具备较好的风控解释性
- `Raw LR` 也保持了较强基线能力，适合作为传统统计建模对照模型

#### 关于历史辅助复算结果

- 此前 Linux / WSL 环境下存在过辅助复算结果
- 这些结果与 Windows notebook 输出存在差异，推测与 Python、pandas、scikit-learn、xgboost、shap 等环境版本或 notebook 运行状态差异有关
- 因此历史 Linux / WSL 复算结果不作为当前项目最终结果依据

## 评估指标

项目使用了以下评估指标：

- `AUC`
- `KS`
- `Precision / Recall`
- `F2`
- `Gain`
- `Lift`
- 评分分箱违约率
- `PR Curve`
- `KS Curve`

说明：

- 核心模型指标统一来自 Windows Python 3.13 环境完整重跑后的 `operation_full_output` 导出结果
- `figures/` 中的图像主要作为辅助展示材料

## 可解释性分析

### 已实现内容

- 评分卡分箱违约率分析
- XGBoost 的 SHAP 重要性分析接口

### 当前问题

- SHAP 兼容性问题主要影响部分 Linux 环境下的 SHAP 重算
- 该问题不影响 Windows notebook 已保存的结果图与核心模型评估指标
- 当前项目展示时可直接引用已导出的 notebook 结果和已保存图像中的 SHAP 面板

来源：

- SHAP 图：`figures/XGBoost_fit_goodness_visualization.png`

## 数据泄漏控制

项目体现了一定的数据泄漏控制意识：

- 将 `train_valid` 与 `test` 分开
- 使用 `5` 折分层交叉验证
- 分箱、WOE、交互项筛选主要在训练折中拟合，再应用到验证折

但也需要谨慎：

- 部分早期规则标记是在较前阶段统一生成的
- 简历中应写“具备数据泄漏防控意识”，不宜夸大为完全工业级流程

## 已知问题

### 1. SHAP 与 XGBoost 兼容性问题

- 当前环境下可训练 XGBoost
- 但 SHAP 解释器初始化报错

### 2. 结果沉淀仍可进一步完善

- 当前已导出 `operation_full_output.md` / `.html`
- 但仍可进一步增加面向 README 和简历展示的轻量结果摘要页

## 简历可提炼亮点

- 具备风控建模全流程能力：数据清洗、特征工程、分箱、WOE、评分卡、模型比较
- 同时掌握传统可解释建模和树模型方法
- 对缺失、异常值、特殊编码采用风险标签化处理
- 使用交叉验证和稳定性指标进行变量筛选
- 输出多维度模型评估和可视化结果

## 结果来源说明

- 最终模型结果主来源：Windows Python 3.13 环境完整重跑后的 `operation_full_output.ipynb`
- 导出结果文件：`reports/operation_full_output.md`、`reports/operation_full_output.html`
- 辅助展示材料：`figures/`

若某项结果未在最终导出结果文件中明确给出，应标记为：

- `待确认`
