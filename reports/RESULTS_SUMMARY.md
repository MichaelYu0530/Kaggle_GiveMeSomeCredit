# Results Summary

## 结果来源

当前项目的可信结果主要来自以下两类文件：

- `operation_full_output.ipynb`
- `reports/operation_full_output.md`
- `reports/operation_full_output.html`
- `reports/model_comparison_from_operation_py.md`
- `reports/f2_threshold_summary_from_operation_py.md`
- `reports/scorecard_bins_from_operation_py.md`
- `reports/operation_py_run_summary.md`

其中：

- `operation_full_output.ipynb` 及其 `.md` / `.html` 导出文件保留了完整 notebook 输出；
- 当前标准 Python 3.13 `.venv` 环境下 `operation.py` 导出的 `reports/` 结果，适合做结构化摘要引用；
- 两套来源整体一致，轻微差异不影响主结论。

## 最终模型比较结果

| model | test AUC | AUC gap | test KS | KS gap |
|---|---:|---:|---:|---:|
| Raw LR | 0.8582 | 0.0013 | 0.5651 | -0.0045 |
| WOE LR / ScoreCard | 0.8612 | -0.0013 | 0.5641 | -0.0056 |
| XGBoost | 0.8659 | 0.0067 | 0.5830 | 0.0050 |

## 结果解读

- `XGBoost` 的测试集 `AUC≈0.8659`、`KS≈0.5830`，三类模型中排序区分能力最佳。
- `WOE LR / ScoreCard` 的测试集 `AUC=0.8612`、`KS=0.5641`，在保持较强效果的同时具备较好的风控解释性。
- `Raw LR` 的测试集 `AUC≈0.8582`、`KS≈0.5651`，可作为稳健、可解释的传统基线模型。

## 与 figures/ 的关系

- `figures/` 目录中的图片可作为辅助展示材料，适合在 README、报告或面试讲解中配合使用。
- 当前 `figures/` 仅保留基础 EDA 图与三类模型效果图；旧的月收入填充实验图已移除。
- 若不同结果来源存在轻微差异，可优先引用四舍五入后的统一口径，不影响项目主结论。

## 关于跨环境差异

- 此前在 Linux / WSL 环境下进行过辅助复算。
- 当前标准 Python 3.13 `.venv` 环境下的 `operation.py` 已能稳定导出核心结果，并与 Windows notebook 输出整体一致。
- 因此，README、项目报告与简历中可以引用当前汇总口径。

## 关于 SHAP

- 早期兼容性问题主要影响部分 Linux / WSL 环境下的 SHAP 重算。
- SHAP 属于可解释性辅助模块，不影响本项目当前引用的 AUC、KS、F2、Gain/Lift 等核心指标。

## 对外引用建议

- README、项目报告、简历项目描述中的模型结果，统一引用本文件中的结果表即可。
- 如果需要给出更完整上下文，建议同时附上 `operation_full_output.ipynb` 或 `reports/operation_full_output.md` 作为完整实验出处；若需要展示当前脚本落盘能力，可同时附上 `reports/operation_py_run_summary.md`。
