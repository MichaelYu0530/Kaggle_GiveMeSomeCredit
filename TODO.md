# 后续 TODO

## 已完成

- 统一当前项目在 README、项目报告、运行指南中的核心结果口径
- 扶正 Python 3.13 作为标准 `.venv` 环境
- 整理 `requirements.txt`、`requirements_lock.txt`、`requirements_minimal.txt`
- 为 `operation.py` 增加模型比较、F2 阈值摘要、评分卡分箱和运行摘要导出
- 新增 `run_final_report.py` 作为 final-only 快速复现入口
- 导出 `reports/final_*` 作为当前主展示结果文件
- 在 `reports/` 下保留完整 notebook 导出结果与脚本级结果摘要
- 清理旧的无用月收入填充实验图片，当前 `figures/` 仅保留主线展示图片
- 将 `pipeline.py`、`analysis.py`、`visualization.py` 中的核心函数按职责迁入 `gmsc/`，保留旧路径兼容层
- 新增 `scripts/` 入口、三项小样本测试与 `docs/` 架构/模块文档

## 模块化迁移验收

- 在标准 Python 3.13 环境放入原始 `cs-training.csv`，将当前定稿入口的预测和 `reports/final_*` 与迁移前基线对照
- 扩充针对真实数据边界的测试：分箱、WOE、列顺序、各折 fit/apply 及报告导出
- 将 `operation.py` 的研究阶段开关改为显式运行参数，继续减少 notebook 与脚本的重复配置
- 单独修正分箱排名复用、F2 阈值选择和评分卡固定分箱，并重新标注结果口径

## 可选优化

- 增加轻量环境检查脚本，例如 `check_env.py`
- 进一步拆分“长耗时训练入口”和“读取已保存结果入口”
- 进一步区分 notebook 与脚本职责
- 可选地继续优化 EDA 图和 SHAP 相关展示说明
