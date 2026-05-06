# 运行指南

## 推荐环境

- 推荐 Python 版本：`3.13`
- 标准虚拟环境目录：`.venv`
- 当前项目的最终结果以 Windows Python 3.13 环境完整重跑产物为准
- 当前标准 Python 3.13 `.venv` 环境下，`operation.py` 也已支持导出结构化结果
- 如仅用于阅读结果和准备作品集，优先查看已导出的 notebook / Markdown / HTML 结果文件与 `operation.py` 生成的 `reports/` 文件

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
operation_full_output.ipynb
reports/operation_full_output.md
reports/operation_full_output.html
```

原因：

- `operation_full_output.ipynb` 是 Windows Python 3.13 环境下完整重跑后的最终结果 notebook
- `reports/operation_full_output.md` / `.html` 适合直接阅读最终结果表和输出图
- `reports/model_comparison_from_operation_py.md`、`reports/f2_threshold_summary_from_operation_py.md`、`reports/scorecard_bins_from_operation_py.md` 适合快速查看当前脚本落盘结果
- 比直接运行完整脚本更安全，也更符合当前作品集展示用途

如果已注册 kernel，例如 `Python (gmsc)`，可直接在 Jupyter Notebook 或 VS Code 中打开。

## 如何运行核心脚本

完整实验主脚本：

```bash
python operation.py
```

注意：

- 该脚本包含较完整的训练流程
- 会执行分箱、交互项构造、交叉验证、模型训练、结果图生成和核心结果落盘
- 不建议在不了解逻辑前直接运行

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
2. `reports/operation_full_output.md`
3. `reports/operation_full_output.html`
4. `reports/model_comparison_from_operation_py.md`
5. `reports/f2_threshold_summary_from_operation_py.md`
6. `reports/scorecard_bins_from_operation_py.md`
7. `reports/operation_py_run_summary.md`
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

- 优先使用已导出的 notebook 结果和 `operation.py` 落盘结果
- 将 SHAP 兼容性说明视为可选环境说明，而不是当前项目主缺陷

## 如何避免误运行完整长流程

推荐顺序：

1. 先读 `README.md`
2. 再读 `PROJECT_STRUCTURE.md`
3. 再看 `operation_full_output.ipynb`、`reports/operation_full_output.md` 和 `figures/`
4. 确认理解项目后，再决定是否运行 `operation.py`

如果只是为了写简历或浏览作品集，不建议第一步就执行完整训练。
