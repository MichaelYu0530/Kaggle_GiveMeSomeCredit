# 功能模块说明

以下映射以当前 `gmsc/` 源码为准。`scripts/` 是运行入口；`gmsc/` 是可导入代码；`operation.py` 和 notebook 保留研究过程。

| 模块 | 主要函数或对象 | 输入、输出及职责 |
|---|---|---|
| `schema.py` | `target`、原始字段名、`colnames_abbr_map` | 集中定义列名和命名映射 |
| `data.py` | `import_data`、`export_data`、`create_kfold` | 读取 CSV、导出数据及生成分层交叉验证索引 |
| `cleaning.py` | `add_missing_flag`、`add_blacklist_flag`、`add_income_anomaly_flag`、`cap_dep_num`、`delete_small_age`、`add_util_flags`、`add_debt_flags` | 在 DataFrame 原地添加质量标记、隔离异常值或删除异常样本；前三个函数还返回统计分析表 |
| `features.py` | `add_late_severity_score`、`add_derived_features`、`find_valuable_intersections`、`filter_intersections`、`add_intersection_signals` | 逾期权重、业务衍生特征及交互信号；逾期权重在训练侧拟合后传给其他数据集 |
| `binning.py` | `add_bins_1D`、`add_bins_2D`、`convert_bins_value_to_rank` | 构建分箱、应用训练侧切点/二维格子，并生成交互信号所需排名 |
| `woe.py` | `calculate_woe_iv`、`calculate_psi`、`fit_woe_mapping`、`apply_woe_mapping` | 训练侧拟合 WOE，验证/测试侧应用映射；未知取值默认 WOE 为 `0.0` |
| `transforms.py` | `fit_centering_means`、`apply_centered_features`、`fit_log_feature_colnames`、`add_log_features`、`select_colnames_raw_lr` | Raw LR 的训练侧均值、对数列决定与数据变换 |
| `scorecard.py` | `ScoreCard` | 包装拟合后的 WOE LR，计算分数、概率及分数分箱 |
| `statistics.py` | `conduct_stat_analysis`、`compare_corr`、`calculate_vif` 等 | 显著性检验、效应量、相关性、共线性等研究分析 |
| `metrics.py` | `calculate_auc_ks`、`calculate_recall_precision`、`calculate_gain_lift` 等 | 分类评估指标；不负责训练模型 |
| `selection.py` | `test_bins_1D/2D`、`filter_*`、`grid_search_*`、`quantify_model_comparison` | 研究阶段的变量筛选、参数搜索、跨模型结果汇总 |
| `summaries.py` | `summary_bins`、`show_special_discovery`、`display_current_colnames_*` | 供研究流程查看的分析表格 |
| `plotting_eda.py`、`plotting_evaluation.py` | `show_basic_visualization`、`visualize_fit_goodness` 等 | EDA、模型效果、评分卡和 SHAP 图形 |
| `interpretability.py` | `calculate_shap_importance` | 仅在计算 SHAP 时载入 SHAP 库 |
| `config/final_models.py` | 交互规则、通过筛选的列名、三模型特征表、XGBoost 参数 | 定稿复现专用的冻结配置；不是搜索算法 |
| `final_report.py`、`reports.py` | `main`、`summarize_best_f2`、`save_dataframe_outputs` | 组织最终训练评估和 CSV/Markdown 报告导出 |
| `legacy_income.py` | `fill_income_bin`、`fill_income_rf` 等 | 历史月收入填充实验；定稿入口不调用 |

## 入口与导入关系

- 日常复现：`scripts/run_final_report.py` → `gmsc.final_report.main` → 数据、变换、模型和报告模块。根目录 `run_final_report.py` 是旧命令的兼容入口。
- 研究：`scripts/run_research.py` → `operation.py` → 根目录兼容导入层 → `gmsc`。历史 notebook 也继续使用兼容导入层。
- 新增功能时从具体模块导入，例如 `from gmsc.woe import fit_woe_mapping, apply_woe_mapping`。兼容层中的星号导入只为旧研究代码保留。

运行顺序、训练与内部测试集边界、已知指标口径见 [架构说明](architecture.md)。
