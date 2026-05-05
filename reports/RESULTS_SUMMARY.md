# Results Summary

## 结果来源

当前项目的最终模型结果统一以 Windows Python 3.13 环境完整重跑后的以下文件为准：

- `operation_full_output.ipynb`
- `reports/operation_full_output.md`
- `reports/operation_full_output.html`

其中，`reports/operation_full_output.md` 和 `.html` 是 `operation_full_output.ipynb` 的导出结果文件，适合快速查阅和对外引用。

## 最终模型比较结果

| model | test AUC | AUC gap | test KS | KS gap |
|---|---:|---:|---:|---:|
| Raw LR | 0.8581 | 0.0012 | 0.5671 | -0.0066 |
| WOE LR / ScoreCard | 0.8612 | -0.0013 | 0.5641 | -0.0056 |
| XGBoost | 0.8659 | 0.0074 | 0.5825 | 0.0067 |

## 结果解读

- `XGBoost` 的测试集 `AUC=0.8659`、`KS=0.5825`，三类模型中排序区分能力最佳。
- `WOE LR / ScoreCard` 的测试集 `AUC=0.8612`、`KS=0.5641`，在保持较强效果的同时具备较好的风控解释性。
- `Raw LR` 的测试集 `AUC=0.8581`、`KS=0.5671`，可作为稳健、可解释的传统基线模型。

## 与 figures/ 的关系

- `figures/` 目录中的图片可作为辅助展示材料，适合在 README、报告或面试讲解中配合使用。
- 若 `figures/` 中个别旧图的数字与最终结果表存在轻微差异，应以 `operation_full_output.ipynb` 和 `reports/operation_full_output.md` 中的最终结果为准。

## 关于跨环境差异

- 此前在 Linux / WSL 环境下进行过辅助复算。
- 这些辅助复算结果与 Windows notebook 输出存在差异，推测与 Python、pandas、scikit-learn、xgboost、shap 等环境版本或 notebook 运行状态差异有关。
- 因此，Linux / WSL 辅助复算结果不作为当前项目最终结果依据。

## 关于 SHAP

- SHAP 兼容性问题主要影响部分 Linux / WSL 环境下的 SHAP 重算。
- 该问题不影响 Windows notebook 已保存结果，也不影响本项目当前引用的核心模型评估指标。

## 对外引用建议

- README、项目报告、简历项目描述中的模型结果，统一引用本文件中的结果表即可。
- 如果需要给出更完整上下文，建议同时附上 `operation_full_output.ipynb` 或 `reports/operation_full_output.md` 作为结果出处。
