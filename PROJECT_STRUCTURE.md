# 项目结构说明

## 当前目录树

```text
Kaggle_GiveMeSomeCredits/
├── .gitignore
├── .venv/
├── Data Dictionary.xls
├── analysis.py
├── backup/
│   ├── cs-test.csv
│   └── cs-training.csv
├── cs-test.csv
├── cs-training.csv
├── figures/
│   ├── 2Dbin_vs_RF_KDE.png
│   ├── 2Dbin_vs_RF_residual.png
│   ├── 2Dbin_vs_RF_scatter.png
│   ├── Raw_LR_fit_goodness_visualization.png
│   ├── ScoreCard_fit_goodness_visualization.png
│   ├── XGBoost_fit_goodness_visualization.png
│   └── origin_data_basic_visualization.png
├── operation.ipynb
├── operation.py
├── pipeline.py
├── requirements.txt
└── visualization.py
```

## 主要文件说明

### 核心代码文件

- `pipeline.py`
  - 项目主流水线文件
  - 包含数据导入、清洗、特征工程、分箱、WOE、评分卡、模型调参与评估等核心逻辑
  - 是当前项目最重要的核心代码文件

- `analysis.py`
  - 统计分析与指标函数
  - 包含显著性检验、IV、PSI、AUC、KS、Lift/Gain、SHAP 等计算逻辑
  - 属于核心代码文件

- `visualization.py`
  - 可视化模块
  - 包含原始数据 EDA 图、模型效果图、评分分箱图、SHAP 图等
  - 属于核心代码文件

- `operation.py`
  - 实验主脚本
  - 按顺序组织完整项目流程
  - 包括数据探索、变量筛选、模型训练和最终比较
  - 属于核心代码文件

### Notebook 文件

- `operation.ipynb`
  - `operation.py` 的 notebook 版本
  - 适合阅读实验过程、查看已保存输出和中间结果
  - 建议优先作为展示材料阅读

### 数据文件

- `cs-training.csv`
  - 原始训练数据

- `cs-test.csv`
  - 原始测试数据

- `Data Dictionary.xls`
  - 数据字段说明表

- `backup/cs-training.csv`
- `backup/cs-test.csv`
  - 备份数据文件
  - 当前用途可理解为原始数据备份

### 输出图片目录

- `figures/`
  - 已保存的项目可视化结果
  - 包括基础 EDA 图、Raw LR/ScoreCard/XGBoost 效果图，以及收入填充实验图

## 可能的临时实验文件或待整理文件

- `operation.ipynb`
  - 是主要展示材料之一，但同时也保留了较强的实验 notebook 属性
  - 未来可进一步拆分为“正式展示版”和“实验记录版”

- `requirements.txt`
  - 当前更像完整环境快照，而不是项目最小依赖文件
  - 建议保留，同时增加更简洁的 `requirements_minimal.txt`

- `figures/2Dbin_vs_RF_*.png`
  - 与收入填充对比实验相关
  - 当前在主项目叙事中的优先级低于违约预测主线
  - 可视为待进一步整理的实验产物

## 用途待确认

- `backup/`
  - 从命名看像数据备份目录
  - 除备份外是否参与正式流程，当前未看到核心代码显式依赖，故标记为“用途待确认”

