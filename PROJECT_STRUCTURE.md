# 项目结构说明

## 当前目录树

```text
Kaggle_GiveMeSomeCredits/
├── .gitignore
├── .venv/
├── Data Dictionary.xls
├── ENVIRONMENT.md
├── PROJECT_REPORT_DRAFT.md
├── PROJECT_STRUCTURE.md
├── README.md
├── RUN_GUIDE.md
├── TODO.md
├── analysis.py
├── backup/
│   ├── cs-test.csv
│   └── cs-training.csv
├── cs-test.csv
├── cs-training.csv
├── figures/
│   ├── Raw_LR_fit_goodness_visualization.png
│   ├── ScoreCard_fit_goodness_visualization.png
│   ├── XGBoost_fit_goodness_visualization.png
│   └── origin_data_basic_visualization.png
├── operation.ipynb
├── operation.py
├── operation_full_output.ipynb
├── pipeline.py
├── reports/
│   ├── f2_threshold_summary_from_operation_py.md
│   ├── model_comparison_from_operation_py.md
│   ├── operation_full_output.html
│   ├── operation_full_output.md
│   ├── operation_full_output_files/
│   │   └── reports/
│   │       ├── operation_full_output_2_0.png
│   │       ├── operation_full_output_46_0.png
│   │       ├── operation_full_output_46_1.png
│   │       └── operation_full_output_46_2.png
│   ├── operation_py_run_summary.md
│   ├── RESULTS_SUMMARY.md
│   └── scorecard_bins_from_operation_py.md
├── requirements.txt
├── requirements_lock.txt
├── requirements_minimal.txt
└── visualization.py
```

## 主要文件说明

### 核心代码文件

- `pipeline.py`
  - 项目主流水线文件
  - 包含数据导入、清洗、特征工程、分箱、WOE、评分卡、模型调参与评估等核心逻辑
  - 是当前项目最重要的核心代码文件

- `analysis.py`
  - 统计分析与模型评估函数
  - 包含显著性检验、IV、PSI、AUC、KS、Lift/Gain、SHAP 相关计算逻辑
  - 属于核心代码文件

- `visualization.py`
  - 可视化模块
  - 包含基础 EDA 图、模型效果图、评分分箱图、SHAP 图等
  - 属于核心代码文件

- `operation.py`
  - 实验主脚本
  - 按顺序组织完整项目流程，包括数据探索、变量筛选、模型训练、最终比较和结果导出
  - 属于核心代码文件

### Notebook 文件

- `operation.ipynb`
  - 项目的实验型 notebook
  - 适合阅读主流程、查看中间分析思路和交互式输出

- `operation_full_output.ipynb`
  - 完整重跑后的结果版 notebook
  - 当前项目最终模型结果以该 notebook 及其导出报告为准
  - 是最适合对外展示和引用结果的 notebook 文件

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
  - 当前未见核心流程显式依赖，可视为原始数据备份

### 报告与说明文档

- `README.md`
  - GitHub 首页展示说明
  - 用于介绍项目背景、方法、结果、运行方式和当前状态

- `PROJECT_REPORT_DRAFT.md`
  - 项目报告草稿
  - 用于系统整理项目方法、结果、问题和简历可提炼亮点

- `RUN_GUIDE.md`
  - 运行与阅读指南
  - 说明如何查看 notebook、结果导出文件和避免误跑长流程

- `ENVIRONMENT.md`
  - 环境与依赖说明
  - 记录主要第三方库用途、兼容性问题和环境建议

- `TODO.md`
  - 后续工程化整理清单

### 结果文件与输出目录

- `reports/operation_full_output.md`
  - `operation_full_output.ipynb` 导出的 Markdown 结果文件
  - 当前最终模型结果的重要文本依据

- `reports/operation_full_output.html`
  - `operation_full_output.ipynb` 导出的 HTML 结果文件
  - 适合直接浏览完整 notebook 输出

- `reports/operation_full_output_files/`
  - notebook 导出时生成的配套资源目录
  - 当前包含导出图片文件，供 Markdown/HTML 正常显示使用

- `reports/RESULTS_SUMMARY.md`
  - 结果摘要说明
  - 用于快速查看最终结果来源、关键指标和引用口径

- `reports/model_comparison_from_operation_py.md`
- `reports/f2_threshold_summary_from_operation_py.md`
- `reports/scorecard_bins_from_operation_py.md`
- `reports/operation_py_run_summary.md`
  - `operation.py` 在标准 Python 3.13 `.venv` 环境下导出的结构化结果
  - 便于直接引用模型比较、阈值摘要、评分卡分箱和运行环境信息

- `figures/`
  - 已保存的项目可视化结果
  - 当前包括基础 EDA 图，以及 Raw LR / ScoreCard / XGBoost 三类模型效果图
  - 旧的月收入填充实验图已从当前展示集移除

### 依赖文件

- `requirements_minimal.txt`
  - 项目最小依赖列表
  - 适合快速创建可读可跑环境

- `requirements.txt`
  - 推荐依赖文件
  - 适合作为当前项目的标准安装入口

- `requirements_lock.txt`
  - 完整锁定依赖文件
  - 适合记录更完整的环境版本信息

## 当前说明

- `operation.ipynb`
  - 仍保留较强的实验 notebook 属性
  - 后续可视需要进一步区分“实验记录版”和“对外展示版”

- `backup/`
  - 从目录命名看属于数据备份
  - 除备份外的正式用途待进一步确认

## 当前建议的阅读顺序

1. `README.md`
2. `reports/RESULTS_SUMMARY.md`
3. `operation_full_output.ipynb`
4. `reports/operation_full_output.md`
5. `pipeline.py`、`analysis.py`、`visualization.py`、`operation.py`
