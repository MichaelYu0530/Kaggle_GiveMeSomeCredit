# 运行指南

## 推荐环境

- 推荐 Python 版本：`3.10+`
- 当前项目的最终结果以 Windows Python 3.13 环境完整重跑产物为准
- 如仅用于阅读结果和准备作品集，优先查看已导出的 notebook / Markdown / HTML 结果文件

## 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 安装依赖

如果只想安装项目核心依赖，推荐：

```bash
pip install -r requirements_minimal.txt
```

如果你需要尽量接近当前 notebook / kernel 所使用的完整环境，可参考：

```bash
pip install -r requirements.txt
```

说明：

- `requirements_minimal.txt` 更适合项目复现和 GitHub 展示
- `requirements.txt` 更接近当前实际环境快照，包更多、体积更大

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
- 比直接运行完整脚本更安全，也更符合当前作品集展示用途

如果已注册 kernel，例如 `Python (gmsc)`，可直接在 Jupyter Notebook 或 VS Code 中打开。

## 如何运行核心脚本

完整实验主脚本：

```bash
python operation.py
```

注意：

- 该脚本包含较完整的训练流程
- 会执行分箱、交互项构造、交叉验证、模型训练和结果图生成
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
4. `figures/origin_data_basic_visualization.png`
5. `figures/Raw_LR_fit_goodness_visualization.png`
6. `figures/ScoreCard_fit_goodness_visualization.png`
7. `figures/XGBoost_fit_goodness_visualization.png`

这些内容通常足以帮助你快速理解项目主线和最终结果，而无需立即重跑完整训练。

## 关于 SHAP 与 XGBoost 兼容报错

在部分 Linux / WSL 环境下，项目中的 SHAP 计算在 `calculate_shap_importance` 处可能报错，属于 **SHAP 与当前 XGBoost 版本组合下的兼容性问题**。

这意味着：

- 主模型训练不一定失败
- 该问题主要影响 SHAP 的本地重算
- 不影响 Windows notebook 已保存结果和核心模型评估指标

因此在展示项目时，建议：

- 优先使用 Windows notebook 已导出的结果文件和已保存 SHAP 图
- 不要在未验证兼容性前强行修改模型逻辑

## 如何避免误运行完整长流程

推荐顺序：

1. 先读 `README.md`
2. 再读 `PROJECT_STRUCTURE.md`
3. 再看 `operation_full_output.ipynb`、`reports/operation_full_output.md` 和 `figures/`
4. 确认理解项目后，再决定是否运行 `operation.py`

如果只是为了写简历或浏览作品集，不建议第一步就执行完整训练。
