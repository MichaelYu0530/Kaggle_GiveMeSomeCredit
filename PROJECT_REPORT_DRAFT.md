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

当前项目的可信结果主要来自两类来源：

- `operation_full_output.ipynb`
- `reports/operation_full_output.md`
- `reports/operation_full_output.html`
- `reports/model_comparison_from_operation_py.md`
- `reports/f2_threshold_summary_from_operation_py.md`
- `reports/scorecard_bins_from_operation_py.md`
- `reports/operation_py_run_summary.md`

其中：

- Windows Python 3.13 环境完整重跑后的 notebook 导出文件保留了完整实验上下文；
- 当前标准 Python 3.13 `.venv` 环境下的 `operation.py` 导出结果适合做结构化摘要引用；
- 最终用于横向比较的 XGBoost 采用前序调参确定的固定最佳参数组合重新拟合，测试集仅用于最终评估；
- 两套来源整体一致，轻微差异不影响项目主结论。

#### 最终模型比较结果

| model | test AUC | AUC gap | test KS | KS gap |
|---|---:|---:|---:|---:|
| Raw LR | 0.8582 | 0.0013 | 0.5651 | -0.0045 |
| WOE LR / ScoreCard | 0.8612 | -0.0013 | 0.5641 | -0.0056 |
| XGBoost | 0.8659 | 0.0067 | 0.5830 | 0.0050 |

#### 结果解释

- `XGBoost` 在三类模型中测试集 `AUC` 与 `KS` 最优，排序区分能力最佳
- `WOE LR / ScoreCard` 的测试集 `AUC=0.8612`、`KS=0.5641`，在保持较强效果的同时具备较好的风控解释性
- `Raw LR` 也保持了较强基线能力，适合作为传统统计建模对照模型

#### 关于历史辅助复算结果

- 此前 Linux / WSL 环境下存在过辅助复算结果
- 当前标准 Python 3.13 `.venv` 环境下，`operation.py` 已可稳定导出核心结果，且与 Windows notebook 输出整体一致
- 因此当前项目对外展示时，可同时引用完整 notebook 导出结果与当前脚本落盘结果

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

- 核心模型比较结果可以引用 Windows notebook 导出结果，也可以引用当前 `operation.py` 落盘结果
- `operation.py` 当前已支持输出模型比较、F2 / Precision / Recall 阈值摘要、评分卡分箱和运行摘要
- `figures/` 中的图像主要作为辅助展示材料

## 可解释性分析

### 已实现内容

- 评分卡分箱违约率分析
- XGBoost 的 SHAP 重要性分析接口

### 当前说明

- 早期 Linux 环境下曾出现过 SHAP 或依赖兼容性问题
- 当前推荐 Python 3.13 环境已完成核心结果复现，且 `operation.py` 运行摘要显示 SHAP 已成功执行
- SHAP 属于可解释性辅助模块，不影响 AUC、KS、F2、Gain/Lift 等核心指标引用

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

## 当前状态与后续优化

### 当前已完成

- 已完成端到端建模流程与初步工程化整理
- 已保留完整 notebook 导出归档，并补充脚本级结果落盘
- `reports/` 已包含模型比较、F2 / Precision / Recall、评分卡分箱和运行摘要
- `figures/` 当前仅保留主线展示图片

### 后续仍可优化

- 增加更轻量的环境检查与结果复现入口
- 进一步精简 README 与 GitHub 展示内容
- 继续区分 notebook 与脚本的职责
- 可选地继续跟踪 SHAP 与跨环境兼容性说明

## 简历可提炼亮点

- 具备风控建模全流程能力：数据清洗、特征工程、分箱、WOE、评分卡、模型比较
- 同时掌握传统可解释建模和树模型方法
- 对缺失、异常值、特殊编码采用风险标签化处理
- 使用交叉验证和稳定性指标进行变量筛选
- 输出多维度模型评估和可视化结果

## 结果来源说明

- 最终结果主来源：Windows Python 3.13 环境完整重跑后的 `operation_full_output.ipynb` 及其导出文件
- 当前脚本结果来源：标准 Python 3.13 `.venv` 环境下 `operation.py` 导出的 `reports/` 文件
- 辅助展示材料：`figures/`

若某项结果未在最终导出结果文件中明确给出，应标记为：

- `待确认`
