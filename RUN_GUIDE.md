# 运行指南

## 推荐环境

- 推荐 Python 版本：`3.13`
- 标准虚拟环境目录：`.venv`
- 推荐日常复现入口：`run_final_report.py`
- `operation.py` 是完整研究/调参流程，运行耗时较长，不作为日常快速复现入口
- 如仅用于阅读结果和准备作品集，优先查看 `reports/final_*` 与历史 notebook 导出文件

## 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 安装依赖

推荐安装项目标准依赖：

```bash
pip install -r requirements.txt
```

如果只想安装项目核心依赖，或用于更简洁的展示环境，可使用：

```bash
pip install -r requirements_minimal.txt
```

如果你需要保留更完整的锁定版本信息，可参考：

```bash
pip install -r requirements_lock.txt
```

说明：

- `requirements.txt` 是当前项目推荐依赖文件
- `requirements_lock.txt` 是完整锁定依赖文件
- `requirements_minimal.txt` 是简洁依赖说明

## 如何阅读 notebook

推荐优先阅读：

```text
reports/final_model_comparison.md
reports/final_f2_threshold_summary.md
reports/final_scorecard_bins.md
reports/final_report_run_summary.md
```

原因：

- `reports/final_*` 是当前 final-only 复现脚本生成的主结果文件
- 它们适合直接查看最终模型比较、F2 阈值摘要和评分卡分箱
- `operation_full_output.ipynb` / `.md` / `.html` 继续保留完整历史实验上下文
- 比直接运行完整研究脚本更适合当前作品集展示和日常结果复核

如果已注册 kernel，例如 `Python (gmsc)`，可直接在 Jupyter Notebook 或 VS Code 中打开。

## 如何运行核心脚本

推荐快速复现入口：

```bash
.venv/bin/python run_final_report.py
```

注意：

- 该脚本会跳过 `operation.py` 中耗时的 grid search / 特征搜索 / 参数搜索
- 直接复用已定稿特征列表和参数字典，训练最终三模型并导出 `reports/final_*`
- 最终 XGBoost 拟合不使用 test-set early stopping

完整研究入口：

```bash
python operation.py
```

注意：

- `operation.py` 会执行更完整的研究与调参流程
- 包含分箱、交互项构造、交叉验证、变量筛选、参数搜索和结果导出
- 耗时明显长于 `run_final_report.py`

## 哪些流程可能耗时较长

以下部分可能明显耗时：

- 5 折交叉验证
- 大量一维/二维分箱构造
- XGBoost 多轮参数搜索
- SHAP 重要性计算

在当前项目状态下，完整流程可能明显长于“只读查看 notebook 和已有图像”的方式。

## 建议优先查看哪些结果

建议优先查看：

1. `operation_full_output.ipynb`
2. `reports/final_model_comparison.md`
3. `reports/final_f2_threshold_summary.md`
4. `reports/final_scorecard_bins.md`
5. `reports/final_report_run_summary.md`
6. `reports/operation_full_output.md`
7. `reports/operation_full_output.html`
8. `figures/origin_data_basic_visualization.png`
9. `figures/Raw_LR_fit_goodness_visualization.png`
10. `figures/ScoreCard_fit_goodness_visualization.png`
11. `figures/XGBoost_fit_goodness_visualization.png`

这些内容通常足以帮助你快速理解项目主线和最终结果，而无需立即重跑完整训练。

## 关于 SHAP 与环境兼容性说明

早期在部分 Linux / WSL 环境下，项目中的 SHAP 计算曾出现过依赖兼容性问题。

这意味着：

- 当前推荐 Python 3.13 环境已经完成核心结果复现
- SHAP 主要属于可解释性辅助模块
- 不影响 AUC、KS、F2、Gain/Lift、评分分箱等核心结果引用

因此在展示项目时，建议：

- 优先使用 `run_final_report.py` 导出的 `reports/final_*` 和历史 notebook 导出结果
- 将 SHAP 兼容性说明视为可选环境说明，而不是当前项目主缺陷

## 如何避免误运行完整长流程

推荐顺序：

1. 先读 `README.md`
2. 再读 `PROJECT_STRUCTURE.md`
3. 再看 `reports/final_*`、`operation_full_output.ipynb`、`reports/operation_full_output.md` 和 `figures/`
4. 确认理解项目后，再决定是运行 `run_final_report.py` 还是 `operation.py`

如果只是为了写简历或浏览作品集，不建议第一步就执行完整训练。
