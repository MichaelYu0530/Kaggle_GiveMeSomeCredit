"""Reproduce the frozen three-model comparison."""

from __future__ import annotations

import time
import traceback
import numpy as np
import pandas as pd
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .binning import add_bins_1D, add_bins_2D, convert_bins_value_to_rank, get_colnames_whether_binary
from .cleaning import (add_blacklist_flag, add_debt_flags, add_income_anomaly_flag,
                       add_missing_flag, add_util_flags, cap_dep_num, delete_small_age)
from .config.final_models import (best_para_xgb_dict, c_raw_lr, c_woe_lr,
                                  colnames_to_fit_raw_lr, colnames_to_fit_woe_lr,
                                  colnames_to_fit_xgb, low_iv_binnames_1D,
                                  negative_interactions_df, passed_binary_colnames,
                                  passed_binnames_1D, passed_binnames_2D,
                                  positive_interactions_df)
from .data import import_data
from .features import (add_derived_features, add_intersection_signals,
                       add_late_severity_score, reorder_columns)
from .metrics import calculate_gain_lift, calculate_recall_precision
from .reports import (print_runtime_info, save_dataframe_outputs,
                      summarize_best_f2, write_run_summary)
from .schema import target
from .scorecard import ScoreCard
from .selection import quantify_model_comparison
from .transforms import (add_log_features, apply_centered_features,
                         fit_centering_means, fit_log_feature_colnames,
                         select_colnames_raw_lr)
from .woe import apply_woe_mapping, fit_woe_mapping

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


