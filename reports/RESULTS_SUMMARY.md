# Results Summary

## 结果来源

当前项目的可信结果主要来自以下两类文件：

- `reports/final_model_comparison.md`
- `reports/final_f2_threshold_summary.md`
- `reports/final_scorecard_bins.md`
- `reports/final_report_run_summary.md`
- `operation_full_output.ipynb`
- `reports/operation_full_output.md`
- `reports/operation_full_output.html`

其中：

- `reports/final_*` 是当前 final-only 复现脚本 `run_final_report.py` 导出的主结果文件；
- `operation_full_output.ipynb` 及其 `.md` / `.html` 导出文件保留了完整 notebook 输出；
- `operation.py` 导出的 `reports/` 文件保留为完整研究流程的历史结构化输出；
- `reports/final_*` 是当前对外引用最终指标时的标准口径；`operation_full_output.*` 用于完整研究归档与人工审阅。

## 最终模型比较结果

| model | test AUC | AUC gap | test KS | KS gap |
|---|---:|---:|---:|---:|
| Raw LR | 0.8582 | 0.0013 | 0.5651 | -0.0045 |
| WOE LR / ScoreCard | 0.8596 | 0.0003 | 0.5698 | -0.0041 |
| XGBoost | 0.8662 | 0.0091 | 0.5824 | 0.0096 |

## 结果解读

- `XGBoost` 的测试集 `AUC≈0.8662`、`KS≈0.5824`，整体排序能力稍优，但 `gap` 也略高。
- `WOE LR / ScoreCard` 的测试集 `AUC≈0.8596`、`KS≈0.5698`，略低于 `XGBoost`，但兼具较好的风控解释性与稳定性。
- `Raw LR` 的测试集 `AUC≈0.8582`、`KS≈0.5651`，可作为稳健、可解释的传统线性基线模型。

## 与 figures/ 的关系

- `figures/` 目录中的图片可作为辅助展示材料，适合在 README、报告或面试讲解中配合使用。
- 当前 `figures/` 仅保留基础 EDA 图与三类模型效果图；旧的月收入填充实验图已移除。
- 若不同结果来源存在轻微差异，可优先引用四舍五入后的统一口径，不影响项目主结论。

## 关于跨环境差异

- 此前在 Linux / WSL 环境下进行过辅助复算。
- 当前标准 Python 3.13 `.venv` 环境下的 `run_final_report.py` 已能快速复现最终结果，并稳定导出 `reports/final_*`。
- 因此，README、项目报告与简历中建议优先引用本文件中的 final-only 汇总口径，而不是直接引用 `operation_full_output.*` 中的局部 notebook 路径指标。

## 关于 SHAP

- 早期兼容性问题主要影响部分 Linux / WSL 环境下的 SHAP 重算。
- SHAP 属于可解释性辅助模块，不影响本项目当前引用的 AUC、KS、F2、Gain/Lift 等核心指标。

## 对外引用建议

- README、项目报告、简历项目描述中的模型结果，建议统一引用本文件中的结果表即可。
- 如果需要给出更完整上下文，建议同时附上 `operation_full_output.ipynb` 或 `reports/operation_full_output.md` 作为完整实验出处；若需要展示当前 final-only 复现能力，可同时附上 `reports/final_report_run_summary.md`。
