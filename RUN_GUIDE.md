# 运行指南

## 推荐环境

- 推荐 Python 版本：`3.10`
- 当前 WSL 虚拟环境实测版本：`Python 3.10.12`
- 建议在 Linux / WSL 环境中运行，而不是直接在 Windows 原生路径下运行

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
operation.ipynb
```

原因：

- notebook 前半部分保留了基础 EDA 和部分统计输出
- 更适合作为项目展示材料
- 比直接运行完整脚本更安全

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

1. `operation.ipynb` 已保存输出
2. `figures/origin_data_basic_visualization.png`
3. `figures/Raw_LR_fit_goodness_visualization.png`
4. `figures/ScoreCard_fit_goodness_visualization.png`
5. `figures/XGBoost_fit_goodness_visualization.png`

这些内容通常足以帮助你快速理解项目主线，而无需立即重跑完整训练。

## 关于 SHAP 与 XGBoost 兼容报错

当前环境下，项目中的 SHAP 计算在 `calculate_shap_importance` 处可能报错，属于 **SHAP 与当前 XGBoost 版本组合下的兼容性问题**。

这意味着：

- 主模型训练不一定失败
- AUC / KS / F2 / 评分分箱等结果通常仍可复算
- 但 SHAP 数值可能无法直接从当前环境中重新计算

因此在展示项目时，建议：

- 先使用 `figures/XGBoost_fit_goodness_visualization.png` 中已保存的 SHAP 图
- 不要在未验证兼容性前强行修改模型逻辑

## 如何避免误运行完整长流程

推荐顺序：

1. 先读 `README.md`
2. 再读 `PROJECT_STRUCTURE.md`
3. 再看 `operation.ipynb` 和 `figures/`
4. 确认理解项目后，再决定是否运行 `operation.py`

如果只是为了写简历或浏览作品集，不建议第一步就执行完整训练。

