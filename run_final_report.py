"""快速复现最终模型结果并导出 reports。"""

from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-final-report")

import sys
import time
import warnings
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import shap
import statsmodels
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pipeline import *

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)
GENERATED_REPORT_FILES: list[str] = []


def get_runtime_info() -> dict:
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


# 交互标记常量
positive_interactions_df = pd.DataFrame({
    "signal_name": [
        "late_severity_mid_x_short_late_low_signal",
        "late_severity_high_x_short_late_low_signal",
        "30-59late_low_x_short_late_mid_high_signal",
        "90+late_mid_high_x_short_late_low_signal",
        "30-59late_low_x_credit_late_density_mid_high_signal",
        "age_high_x_credit_late_density_high_signal",
        "util_low_x_credit_late_density_high_signal",
        "mortgage_ratio_mid_x_credit_late_density_high_signal",
        "mortgage_mid_x_credit_late_density_high_signal",
        "dep_missing_x_60-89late_mid_signal",
        "age_high_x_late_severity_high_signal",
        "age_high_x_90+late_mid_signal",
        "util_low_mid_x_late_severity_high_signal",
        "util_mid_x_90+late_mid_signal",
        "age_high_x_60-89late_mid_signal",
        "30-59late_mid_x_credit_mid_signal",
        "debt_low_x_credit_pressure_mid_signal",
        "debt_mid_x_credit_pressure_high_signal",
        "util_high_x_credit_pressure_low_signal",
        "monthly_debt_low_x_credit_pressure_high_signal",
        "late_severity_high_x_credit_pressure_low_signal",
    ],
    "binrank_colname1": [
        "late_severity_score_binrank", "late_severity_score_binrank", "30-59late_binrank",
        "90+late_binrank", "30-59late_binrank", "age_binrank", "util_binrank",
        "mortgage_ratio_binrank", "mortgage_binrank", "dep_binrank", "age_binrank",
        "age_binrank", "util_binrank", "util_binrank", "age_binrank",
        "30-59late_binrank", "debt_binrank", "debt_binrank", "util_binrank",
        "monthly_debt_binrank", "late_severity_score_binrank",
    ],
    "binrank_colname2": [
        "short_late_binrank", "short_late_binrank", "short_late_binrank",
        "short_late_binrank", "credit_late_density_binrank", "credit_late_density_binrank",
        "credit_late_density_binrank", "credit_late_density_binrank", "credit_late_density_binrank",
        "60-89late_binrank", "late_severity_score_binrank", "90+late_binrank",
        "late_severity_score_binrank", "90+late_binrank", "60-89late_binrank",
        "credit_binrank", "credit_pressure_index_binrank", "credit_pressure_index_binrank",
        "credit_pressure_index_binrank", "credit_pressure_index_binrank", "credit_pressure_index_binrank",
    ],
    "binrank1": [
        "2_2", "4_4", "1_1", "3_4", "1_1", "5_6", "1_1", "2_2", "2_2",
        "-1_-1", "6_6", "6_6", "1_2", "2_2", "6_6", "3_3", "1_1", "2_2",
        "5_5", "1_1", "4_4",
    ],
    "binrank2": [
        "1_1", "1_1", "2_4", "1_1", "4_6", "6_6", "6_6", "6_6", "6_6",
        "2_2", "4_4", "2_3", "4_4", "2_2", "2_2", "2_2", "3_4", "4_5",
        "1_2", "3_4", "1_2",
    ],
})

negative_interactions_df = pd.DataFrame({
    "signal_name": [
        "credit_low_x_util_mid_signal",
        "monthly_debt_low_x_mortgage_ratio_mid_signal",
        "credit_pressure_mid_x_debt_mid_signal",
        "monthly_debt_high_x_debt_mid_signal",
        "credit_pressure_mid_x_mortgage_mid_signal",
        "monthly_debt_low_x_mortgage_mid_signal",
        "monthly_debt_low_x_age_high_signal",
        "monthly_debt_high_x_credit_pressure_mid_high_signal",
        "monthly_debt_mid_x_credit_pressure_mid_signal",
        "age_high_x_debt_low_signal",
    ],
    "binrank_colname1": [
        "credit_binrank", "monthly_debt_binrank", "credit_pressure_index_binrank",
        "monthly_debt_binrank", "credit_pressure_index_binrank", "monthly_debt_binrank",
        "monthly_debt_binrank", "monthly_debt_binrank", "monthly_debt_binrank", "age_binrank",
    ],
    "binrank_colname2": [
        "util_binrank", "mortgage_ratio_binrank", "debt_binrank", "debt_binrank",
        "mortgage_binrank", "mortgage_binrank", "age_binrank",
        "credit_pressure_index_binrank", "credit_pressure_index_binrank", "debt_binrank",
    ],
    "binrank1": [
        "1_1", "1_1", "2_2", "4_4", "2_2", "1_1", "1_1", "4_4", "3_3", "6_6",
    ],
    "binrank2": [
        "2_2", "2_2", "2_2", "2_2", "2_2", "2_2", "6_6", "3_3", "2_2", "1_1",
    ],
})


passed_binary_colnames = [
    "blacklist_flag",
    "income_anomaly_flag",
    "has_no_credit",
    "has_serious_late",
    "has_short_late",
    "has_short_late_but_no_credit",
    "is_debt_high",
    "is_debt_overlimit",
    "is_util_high",
    "is_util_overlimit",
    "30-59late_low_x_credit_late_density_mid_high_signal",
    "30-59late_low_x_short_late_mid_high_signal",
    "30-59late_mid_x_credit_mid_signal",
    "90+late_mid_high_x_short_late_low_signal",
    "age_high_x_60-89late_mid_signal",
    "age_high_x_90+late_mid_signal",
    "age_high_x_credit_late_density_high_signal",
    "age_high_x_debt_low_signal",
    "age_high_x_late_severity_high_signal",
    "credit_low_x_util_mid_signal",
    "credit_pressure_mid_x_debt_mid_signal",
    "credit_pressure_mid_x_mortgage_mid_signal",
    "debt_low_x_credit_pressure_mid_signal",
    "debt_mid_x_credit_pressure_high_signal",
    "dep_missing_x_60-89late_mid_signal",
    "late_severity_high_x_credit_pressure_low_signal",
    "late_severity_high_x_short_late_low_signal",
    "late_severity_mid_x_short_late_low_signal",
    "monthly_debt_high_x_credit_pressure_mid_high_signal",
    "monthly_debt_high_x_debt_mid_signal",
    "monthly_debt_low_x_age_high_signal",
    "monthly_debt_low_x_credit_pressure_high_signal",
    "monthly_debt_low_x_mortgage_mid_signal",
    "monthly_debt_low_x_mortgage_ratio_mid_signal",
    "monthly_debt_mid_x_credit_pressure_mid_signal",
    "mortgage_mid_x_credit_late_density_high_signal",
    "mortgage_ratio_mid_x_credit_late_density_high_signal",
    "util_high_x_credit_pressure_low_signal",
    "util_low_mid_x_late_severity_high_signal",
    "util_low_x_credit_late_density_high_signal",
    "util_mid_x_90+late_mid_signal",
]

passed_binnames_1D = [
    "30-59late_bin",
    "60-89late_bin",
    "90+late_bin",
    "age_bin",
    "credit_bin",
    "credit_late_density_bin",
    "credit_pressure_index_bin",
    "debt_bin",
    "free_cashflow_income_bin",
    "income_bin",
    "income_per_dep_bin",
    "late_severity_score_bin",
    "mortgage_bin",
    "short_late_bin",
    "util_bin",
]

low_iv_binnames_1D = [
    "dep_bin",
    "monthly_debt_bin",
    "mortgage_ratio_bin",
]

passed_binnames_2D = [
    "credit_x_credit_pressure_index_bin",
    "dep_x_credit_bin",
    "credit_x_monthly_debt_bin",
    "debt_x_credit_bin",
    "debt_x_credit_pressure_index_bin",
    "debt_x_monthly_debt_bin",
    "debt_x_mortgage_bin",
    "debt_x_mortgage_ratio_bin",
    "income_x_monthly_debt_bin",
    "monthly_debt_x_credit_pressure_index_bin",
    "mortgage_x_credit_pressure_index_bin",
    "dep_x_mortgage_bin",
    "mortgage_x_monthly_debt_bin",
]

colnames_to_fit_raw_lr = [
    "age_centered",
    "util_centered",
    "debt_centered",
    "NumberOfDependents",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "single_missing_flag",
    "both_missing_flag",
    "blacklist_flag",
    "income_anomaly_flag",
    "debt_anomaly_flag",
    "util_anomaly_flag",
    "income_per_dep_log",
    "credit_pressure_index",
    "mortgage_ratio",
    "credit_late_density",
    "short_late",
    "late_severity_score",
    "has_no_credit",
    "has_short_late_but_no_credit",
    "has_serious_late",
    "has_short_late",
    "is_debt_high",
    "is_debt_overlimit",
    "is_util_high",
    "is_util_overlimit",
]
c_raw_lr = 1

colnames_to_fit_woe_lr = [
    "age_bin_woe",
    "util_bin_woe",
    "debt_bin_woe",
    "credit_bin_woe",
    "mortgage_bin_woe",
    "income_per_dep_bin_woe",
    "credit_pressure_index_bin_woe",
    "late_severity_score_bin_woe",
    "has_no_credit_woe",
    "has_short_late_but_no_credit_woe",
    "has_serious_late_woe",
    "blacklist_flag_woe",
    "is_debt_high_woe",
    "is_debt_overlimit_woe",
    "is_util_high_woe",
    "is_util_overlimit_woe",
    "dep_x_credit_bin_woe",
    "credit_x_monthly_debt_bin_woe",
    "debt_x_credit_bin_woe",
    "debt_x_credit_pressure_index_bin_woe",
    "debt_x_monthly_debt_bin_woe",
    "debt_x_mortgage_bin_woe",
    "debt_x_mortgage_ratio_bin_woe",
    "income_x_monthly_debt_bin_woe",
    "dep_x_mortgage_bin_woe",
    "mortgage_x_monthly_debt_bin_woe",
]
c_woe_lr = 0.025

colnames_to_fit_xgb = [
    "Age",
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfDependents",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "free_cashflow_income",
    "credit_pressure_index",
    "credit_late_density",
    "mortgage_ratio",
    "late_severity_score",
]

best_para_xgb_dict = dict(
    max_depth=3,
    min_child_weight=7,
    learning_rate=0.03,
    n_estimators=800,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_alpha=1,
    reg_lambda=2,
)


def main() -> None:
    start_time = time.perf_counter()
    print_runtime_info()

    origin_data = import_data(filetype="train")
    add_missing_flag(origin_data)
    add_blacklist_flag(origin_data)
    add_income_anomaly_flag(origin_data)

    train_valid_data, test_data = train_test_split(
        origin_data,
        test_size=0.2,
        random_state=50,
        stratify=origin_data[target],
    )

    train_data = train_valid_data.copy()
    test_data = test_data.copy()

    # 统一数据清洗阶段
    cap_dep_num(train_data)
    cap_dep_num(test_data)
    delete_small_age(train_data)
    delete_small_age(test_data)
    add_util_flags(train_data)
    add_util_flags(test_data)
    add_debt_flags(train_data)
    add_debt_flags(test_data)
    beta30, beta60, beta90 = add_late_severity_score(train_data)
    add_late_severity_score(test_data, [beta30, beta60, beta90])
    add_derived_features(train_data)
    add_derived_features(test_data)
    colnames_not_binary = get_colnames_whether_binary(train_data, is_binary=False)
    for colname in colnames_not_binary:
        _, div_pts = add_bins_1D(train_data, colname)
        add_bins_1D(test_data, colname, fixed_div_pts=div_pts)
    convert_bins_value_to_rank(train_data)
    convert_bins_value_to_rank(test_data)
    add_intersection_signals(train_data, positive_interactions_df, negative_interactions_df)
    add_intersection_signals(test_data, positive_interactions_df, negative_interactions_df)
    colnames_without_bins = [colname for colname in train_data.columns if "_bin" not in colname]
    train_data = train_data[colnames_without_bins]
    test_data = test_data[colnames_without_bins]
    train_data = reorder_columns(train_data)
    test_data = reorder_columns(test_data)

    # Raw LR 特化表达
    train_data_raw_lr = train_data.copy()
    test_data_raw_lr = test_data.copy()
    train_data_raw_lr = train_data_raw_lr.clip(lower=0)
    test_data_raw_lr = test_data_raw_lr.clip(lower=0)
    centering_means = fit_centering_means(train_data_raw_lr)
    apply_centered_features(train_data_raw_lr, centering_means)
    apply_centered_features(test_data_raw_lr, centering_means)
    colnames_to_log = fit_log_feature_colnames(train_data_raw_lr)
    add_log_features(train_data_raw_lr, fixed_colnames=colnames_to_log)
    add_log_features(test_data_raw_lr, fixed_colnames=colnames_to_log)
    selected_colnames = select_colnames_raw_lr(train_data_raw_lr)
    train_data_raw_lr = train_data_raw_lr[selected_colnames]
    test_data_raw_lr = test_data_raw_lr[selected_colnames]

    # WOE LR 特化表达
    train_data_woe_lr = train_data.copy()
    test_data_woe_lr = test_data.copy()
    colnames_not_binary = get_colnames_whether_binary(train_data_woe_lr, is_binary=False)
    for colname in colnames_not_binary:
        _, div_pts = add_bins_1D(train_data_woe_lr, colname)
        add_bins_1D(test_data_woe_lr, colname, fixed_div_pts=div_pts)
    colname_pairs_not_binary = []
    for i in range(len(colnames_not_binary) - 1):
        for j in range(i + 1, len(colnames_not_binary)):
            colname_pairs_not_binary.append([colnames_not_binary[i], colnames_not_binary[j]])
    for colname_pair in colname_pairs_not_binary:
        _, bins_2D = add_bins_2D(train_data_woe_lr, colname_pair, constraint=passed_binnames_2D)
        add_bins_2D(
            test_data_woe_lr,
            colname_pair,
            fixed_bins_2D=bins_2D,
            constraint=passed_binnames_2D,
        )
    binnames_map_to_woe = passed_binary_colnames + passed_binnames_1D + low_iv_binnames_1D + passed_binnames_2D
    for colname in binnames_map_to_woe:
        woe_map = fit_woe_mapping(train_data_woe_lr, colname)
        apply_woe_mapping(train_data_woe_lr, colname, woe_map)
        apply_woe_mapping(test_data_woe_lr, colname, woe_map)
    woe_colnames = [colname for colname in train_data_woe_lr.columns if colname.endswith("_woe")]
    train_data_woe_lr = train_data_woe_lr[[target] + woe_colnames]
    test_data_woe_lr = test_data_woe_lr[[target] + woe_colnames]
    train_data_woe_lr = reorder_columns(train_data_woe_lr)
    test_data_woe_lr = reorder_columns(test_data_woe_lr)

    # XGBoost 特化表达
    train_data_xgb = train_data.copy()
    test_data_xgb = test_data.copy()
    train_data_xgb = train_data_xgb.mask(train_data_xgb < 0, np.nan)
    test_data_xgb = test_data_xgb.mask(test_data_xgb < 0, np.nan)

    # 拆分特征
    y_train = train_data[target]
    y_test = test_data[target]
    X_train_raw_lr = train_data_raw_lr.drop(target, axis=1, inplace=False)
    X_test_raw_lr = test_data_raw_lr.drop(target, axis=1, inplace=False)
    X_train_woe_lr = train_data_woe_lr.drop(target, axis=1, inplace=False)
    X_test_woe_lr = test_data_woe_lr.drop(target, axis=1, inplace=False)
    X_train_xgb = train_data_xgb.drop(target, axis=1, inplace=False)
    X_test_xgb = test_data_xgb.drop(target, axis=1, inplace=False)

    # Raw LR
    X_train_raw_lr = X_train_raw_lr[colnames_to_fit_raw_lr]
    X_test_raw_lr = X_test_raw_lr[colnames_to_fit_raw_lr]
    raw_lr_model = LogisticRegression(
        penalty="l2",
        C=c_raw_lr,
        solver="lbfgs",
        max_iter=1000,
        fit_intercept=True,
    )
    raw_lr_model.fit(X_train_raw_lr, y_train)

    # WOE LR / ScoreCard
    X_train_woe_lr = X_train_woe_lr[colnames_to_fit_woe_lr]
    X_test_woe_lr = X_test_woe_lr[colnames_to_fit_woe_lr]
    woe_lr_model = LogisticRegression(
        penalty="l2",
        C=c_woe_lr,
        solver="lbfgs",
        max_iter=1000,
        fit_intercept=True,
    )
    woe_lr_model.fit(X_train_woe_lr, y_train)
    sc_model = ScoreCard()
    sc_model.fit(woe_lr_model, X_train_woe_lr)
    score_bins_train_df = sc_model.show_score_bins(X_train_woe_lr, y_train)
    score_bins_test_df = sc_model.show_score_bins(X_test_woe_lr, y_test)

    # XGBoost
    X_train_xgb = X_train_xgb[colnames_to_fit_xgb]
    X_test_xgb = X_test_xgb[colnames_to_fit_xgb]
    xgb_model = xgboost.XGBClassifier(
        **best_para_xgb_dict,
        random_state=500,
        eval_metric="auc",
        use_label_encoder=False,
    )
    xgb_model.fit(X_train_xgb, y_train, verbose=False)

    # 导出结果
    model_comparison_df = quantify_model_comparison(
        y_train,
        y_test,
        X_train_raw_lr,
        X_test_raw_lr,
        X_train_woe_lr,
        X_test_woe_lr,
        X_train_xgb,
        X_test_xgb,
        raw_lr_model,
        sc_model,
        xgb_model,
    )
    print("\n=== Final Model Comparison ===")
    print(model_comparison_df.to_string(index=False))
    save_dataframe_outputs(
        model_comparison_df,
        csv_path="final_model_comparison.csv",
        md_path="final_model_comparison.md",
        title="Final Model Comparison",
        source_desc="run_final_report.py 最终三模型比较结果",
        extra_lines=[
            "该脚本跳过 grid search / 特征搜索 / 参数搜索，直接复用 operation.py 中已定稿的最终特征组合与参数字典。",
            "最终 XGBoost 拟合未使用 test-set early stopping。",
        ],
    )

    y_prob_test_raw_lr = raw_lr_model.predict_proba(X_test_raw_lr)[:, 1]
    y_prob_test_woe_lr = sc_model.predict_proba(X_test_woe_lr)[:, 1]
    y_prob_test_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]
    f2_threshold_summary_df = pd.concat(
        [
            summarize_best_f2(y_test, y_prob_test_raw_lr, "Raw LR"),
            summarize_best_f2(y_test, y_prob_test_woe_lr, "WOE LR / ScoreCard"),
            summarize_best_f2(y_test, y_prob_test_xgb, "XGBoost"),
        ],
        ignore_index=True,
    )
    print("\n=== F2 / Precision / Recall Threshold Summary ===")
    print(f2_threshold_summary_df.to_string(index=False))
    save_dataframe_outputs(
        f2_threshold_summary_df,
        csv_path="final_f2_threshold_summary.csv",
        md_path="final_f2_threshold_summary.md",
        title="Final F2 Threshold Summary",
        source_desc="run_final_report.py 测试集最佳 F2 / Precision / Recall 阈值摘要",
        extra_lines=["该结果不覆盖 operation.py 历史导出文件。"],
    )

    score_bins_export_df = pd.concat(
        [
            score_bins_train_df.assign(dataset="train"),
            score_bins_test_df.assign(dataset="test"),
        ],
        ignore_index=True,
    )
    score_bins_export_df = score_bins_export_df[
        ["dataset"] + [col for col in score_bins_export_df.columns if col != "dataset"]
    ]
    print("\n=== ScoreCard Bins (train + test) ===")
    print(score_bins_export_df.to_string(index=False))
    save_dataframe_outputs(
        score_bins_export_df,
        csv_path="final_scorecard_bins.csv",
        md_path="final_scorecard_bins.md",
        title="Final ScoreCard Bins",
        source_desc="run_final_report.py 评分卡分箱结果（train / test）",
    )

    write_run_summary(
        model_comparison_df=model_comparison_df,
        f2_threshold_summary_df=f2_threshold_summary_df,
        score_bins_export_df=score_bins_export_df,
        plot_status="skipped",
        plot_note="为缩短 final-only 复现耗时，本脚本当前只导出 CSV / Markdown 结果，不生成新的模型效果图。",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n=== run_final_report.py Error ===")
        print(traceback.format_exc())
        raise
