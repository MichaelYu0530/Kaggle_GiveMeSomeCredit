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

- `reports/final_model_comparison.md`
- `reports/final_f2_threshold_summary.md`
- `reports/final_scorecard_bins.md`
- `reports/final_report_run_summary.md`
- `operation_full_output.ipynb`
- `reports/operation_full_output.md`
- `reports/operation_full_output.html`

其中：

- 当前标准 Python 3.13 `.venv` 环境下，`run_final_report.py` 是当前对外引用最终指标时的权威复现入口；
- Windows Python 3.13 环境完整重跑后的 notebook 导出文件保留了完整实验上下文；
- `operation.py` 保留为完整研究与调参流程，不作为日常快速复现入口，也不作为对外最终指标的唯一权威入口；
- final-only 脚本中的最终 XGBoost 采用固定 `best_para_xgb_dict` 重新拟合，不使用 test-set early stopping；
- `reports/final_*` 与 `operation_full_output.*` 定位不同：前者用于最终结果复现与对外引用，后者用于完整研究归档与人工审阅。

#### 最终模型比较结果

| model | test AUC | AUC gap | test KS | KS gap |
|---|---:|---:|---:|---:|
| Raw LR | 0.8582 | 0.0013 | 0.5651 | -0.0045 |
| WOE LR / ScoreCard | 0.8596 | 0.0003 | 0.5698 | -0.0041 |
| XGBoost | 0.8662 | 0.0091 | 0.5824 | 0.0096 |

#### 结果解释

- `XGBoost` 在三类模型中测试集 `AUC` 与 `KS` 最优，整体排序能力稍优，但 `gap` 也略高，可视为非线性强模型与性能上限参考
- `WOE LR / ScoreCard` 的测试集 `AUC=0.8596`、`KS=0.5698`，略低于 `XGBoost`，但兼具较好的风控解释性与稳定性，适合作为评分卡 / 解释性模型
- `Raw LR` 作为原始数值与人工特征工程的线性基线，表现稳定，但非线性表达能力弱于 `XGBoost`，整体结果也略低于 `WOE LR / ScoreCard`

#### 关于历史辅助复算结果

- 此前 Linux / WSL 环境下存在过辅助复算结果
- 当前标准 Python 3.13 `.venv` 环境下，`run_final_report.py` 已可快速复现最终模型结果并稳定导出 `reports/final_*`
- 因此当前项目对外展示时，建议优先引用 `reports/final_*`，并将完整 notebook 导出文件作为历史实验上下文补充

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

- 核心模型比较结果建议优先引用 `reports/final_*`
- `run_final_report.py` 当前已支持输出模型比较、F2 / Precision / Recall 阈值摘要、评分卡分箱和运行摘要
- `figures/` 中的图像主要作为辅助展示材料

## 可解释性分析

### 已实现内容

- 评分卡分箱违约率分析
- XGBoost 的 SHAP 重要性分析接口

### 当前说明

- 早期部分 Linux / WSL 环境下曾出现过 SHAP 或依赖兼容性问题
- 当前推荐 Python 3.13 环境已完成核心结果复现；SHAP 仍属于可解释性辅助模块
- SHAP 属于可解释性辅助模块，不影响 AUC、KS、F2、Gain/Lift 等核心指标引用

来源：

- SHAP 图：`figures/XGBoost_fit_goodness_visualization.png`

## 数据泄漏控制

当前主流程中的数据泄漏防控逻辑应表述为：

- `Raw LR` 中，中心化均值在训练侧拟合，验证/测试侧复用；对数特征列也由训练侧决定，验证/测试侧只复用该列名集合
- `WOE LR` 中，`WOE`、一维分箱、二维分箱以及 `late_severity_score` 等从数据中学习的统计量，都在训练集或交叉验证训练折中拟合
- 验证集与测试集只应用训练侧得到的映射、切分点和权重，不参与任何 `fit` 或统计量学习
- `XGBoost` 使用训练侧特征工程结果，不使用测试标签进行特征筛选或参数选择

这已经体现出明确的数据泄漏防控意识，但仍应诚实地表述为“研究型项目中的严谨训练-应用分离”，不宜夸大为完全工业级流程。

## 当前状态与后续优化

### 当前已完成

- 已完成端到端建模流程与初步工程化整理
- 已保留完整 notebook 导出归档，并新增 final-only 复现入口 `run_final_report.py`
- `reports/` 已同时包含 `final_*` 主结果文件与历史研究流程输出
- `figures/` 当前仅保留主线展示图片

### 后续仍可优化

- 进一步精简 README 与 GitHub 展示内容
- 继续区分 notebook 与脚本的职责
- 可选地继续跟踪 SHAP 与跨环境兼容性说明
- 补充轻量自动化测试，尤其是分箱、WOE、fit/apply 与最终结果导出相关逻辑

## 模型定稿说明

- `Raw LR` 与 `WOE LR` 的最终方案并不机械追求单一验证指标第一名；当更复杂的特征组合只带来很小的边际收益时，最终方案会综合考虑 `AUC / KS`、`gap`、`VIF`、系数符号稳定性和业务解释性
- `XGBoost` 参数搜索采用多轮顺序网格搜索。由于前一轮定稿参数会影响后一轮搜索空间，且领先参数组合之间的 `AUC / KS` 差距较小，最终参数不是机械采用每轮表格第一名，而是综合考虑验证集表现、`train-valid gap`、模型复杂度、正则化强度和稳定性后的人工定稿配置

## 已知限制

- 当前项目没有完整的 `pytest` 测试入口
- `operation.py` 是研究脚本，不是完全自动化 pipeline
- `XGBoost` 最终参数为多轮顺序搜索后的人工定稿，并不保证全局最优
- Kaggle 公开结构化数据集适合作为方法展示；真实业务落地仍需要更严格的时间外验证、监控和合规评估

## 简历可提炼亮点

- 具备风控建模全流程能力：数据清洗、特征工程、分箱、WOE、评分卡、模型比较
- 同时掌握传统可解释建模和树模型方法
- 对缺失、异常值、特殊编码采用风险标签化处理
- 使用交叉验证和稳定性指标进行变量筛选
- 输出多维度模型评估和可视化结果

## 结果来源说明

- 最终展示主来源：标准 Python 3.13 `.venv` 环境下 `run_final_report.py` 导出的 `reports/final_*`
- 历史完整归档来源：Windows Python 3.13 环境完整重跑后的 `operation_full_output.ipynb` 及其导出文件
- 历史脚本结果来源：标准 Python 3.13 `.venv` 环境下 `operation.py` 导出的 `reports/` 文件
- 辅助展示材料：`figures/`
