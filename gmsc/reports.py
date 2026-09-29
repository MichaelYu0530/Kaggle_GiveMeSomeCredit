"""Structured report exports and runtime metadata."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import xgboost

from .metrics import calculate_gain_lift, calculate_recall_precision

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)
GENERATED_REPORT_FILES: list[str] = []


def get_runtime_info() -> dict:
    import shap
    import statsmodels
    return {
        "python_path": sys.executable,
        "python_version": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "statsmodels": statsmodels.__version__,
    }


def print_runtime_info() -> None:
    runtime_info = get_runtime_info()
    print("=== Runtime Info ===")
    print(f"python_path: {runtime_info['python_path']}")
    print(f"python_version: {runtime_info['python_version']}")
    print(f"pandas: {runtime_info['pandas']}")
    print(f"numpy: {runtime_info['numpy']}")
    print(f"scikit-learn: {runtime_info['scikit-learn']}")
    print(f"xgboost: {runtime_info['xgboost']}")
    print(f"shap: {runtime_info['shap']}")
    print(f"statsmodels: {runtime_info['statsmodels']}")


def save_dataframe_outputs(
    df: pd.DataFrame,
    csv_path: str,
    md_path: str,
    title: str,
    source_desc: str,
    extra_lines: list[str] | None = None,
) -> None:
    runtime_info = get_runtime_info()
    csv_file = REPORTS_DIR / csv_path
    md_file = REPORTS_DIR / md_path
    df.to_csv(csv_file, index=False)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"- 结果来源：{source_desc}\n")
        f.write("- 运行入口：`run_final_report.py`\n")
        f.write(f"- Python：`{runtime_info['python_version']}`\n")
        f.write(
            "- 关键库版本："
            f"`pandas {runtime_info['pandas']}` / "
            f"`numpy {runtime_info['numpy']}` / "
            f"`scikit-learn {runtime_info['scikit-learn']}` / "
            f"`xgboost {runtime_info['xgboost']}` / "
            f"`shap {runtime_info['shap']}` / "
            f"`statsmodels {runtime_info['statsmodels']}`\n"
        )
        if extra_lines:
            for line in extra_lines:
                f.write(f"- {line}\n")
        f.write("\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    GENERATED_REPORT_FILES.extend([str(csv_file), str(md_file)])


def summarize_best_f2(y_true: pd.Series, y_prob: np.ndarray, model_name: str) -> pd.DataFrame:
    thresholds = np.linspace(0, 1, 200)
    recall_list = []
    precision_list = []
    f2_score_list = []
    for threshold in thresholds:
        recall, precision = calculate_recall_precision(y_true, y_prob, threshold)
        recall_list.append(recall)
        precision_list.append(precision)
        if recall + precision == 0:
            f2_score = 0
        else:
            beta = 2
            f2_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        f2_score_list.append(f2_score)
    best_index = int(np.argmax(f2_score_list))
    gain_lift_df = calculate_gain_lift(y_true, y_prob)

    def get_metric_near_population(population: float, colname: str) -> float:
        idx = (gain_lift_df["population"] - population).abs().idxmin()
        return float(gain_lift_df.loc[idx, colname])

    return pd.DataFrame(
        [{
            "model": model_name,
            "best_threshold": float(thresholds[best_index]),
            "best_f2": float(f2_score_list[best_index]),
            "precision": float(precision_list[best_index]),
            "recall": float(recall_list[best_index]),
            "gain_at_10pct": get_metric_near_population(0.10, "gain"),
            "gain_at_25pct": get_metric_near_population(0.25, "gain"),
            "lift_at_10pct": get_metric_near_population(0.10, "lift"),
            "lift_at_25pct": get_metric_near_population(0.25, "lift"),
        }]
    )


def write_run_summary(
    model_comparison_df: pd.DataFrame,
    f2_threshold_summary_df: pd.DataFrame,
    score_bins_export_df: pd.DataFrame,
    plot_status: str,
    plot_note: str,
) -> None:
    runtime_info = get_runtime_info()
    summary_file = REPORTS_DIR / "final_report_run_summary.md"
    output_files = GENERATED_REPORT_FILES + [str(summary_file)]
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("# final report 运行摘要\n\n")
        f.write("- 运行入口：`run_final_report.py`\n")
        f.write("- Python executable：`local virtual environment`\n")
        f.write("- Environment：`.venv`\n")
        f.write("- Project root：`repository root`\n")
        f.write(f"- Python 版本：`{runtime_info['python_version']}`\n")
        f.write(f"- pandas：`{runtime_info['pandas']}`\n")
        f.write(f"- numpy：`{runtime_info['numpy']}`\n")
        f.write(f"- scikit-learn：`{runtime_info['scikit-learn']}`\n")
        f.write(f"- xgboost：`{runtime_info['xgboost']}`\n")
        f.write(f"- shap：`{runtime_info['shap']}`\n")
        f.write(f"- statsmodels：`{runtime_info['statsmodels']}`\n")
        f.write("- 说明：该脚本跳过所有 grid search / 特征搜索 / 参数搜索，仅复用已定稿常量完成最终模型训练与结果导出。\n")
        f.write("- 说明：该文件有意省略本机绝对路径与逐次变化的秒级耗时，避免产生与结果无关的展示差异。\n")
        f.write("- 说明：最终 XGBoost 拟合未使用 `eval_set=[(X_test_xgb, y_test)]`，也未使用 test-set early stopping。\n")
        f.write("\n## 输出文件\n\n")
        for file_path in output_files:
            f.write(f"- `{file_path}`\n")
        f.write("\n## 最终模型比较表\n\n")
        f.write(model_comparison_df.to_markdown(index=False))
        f.write("\n\n## F2 / Precision / Recall 阈值摘要\n\n")
        f.write(f2_threshold_summary_df.to_markdown(index=False))
        f.write("\n\n## 评分卡分箱结果说明\n\n")
        f.write(f"- 当前共导出 `{len(score_bins_export_df)}` 行评分卡分箱结果（含 train / test 两部分）\n")
        f.write("\n## 绘图状态\n\n")
        f.write(f"- 绘图状态：`{plot_status}`\n")
        f.write(f"- 说明：{plot_note}\n")
    GENERATED_REPORT_FILES.append(str(summary_file))
