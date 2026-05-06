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

## 上传 GitHub 前建议完成

- 再做一次公开仓库检查，确认原始数据、备份数据、虚拟环境和缓存文件未被上传
- 继续精简 README 首页内容，突出项目背景、技术路线、结果表和展示图
- 再核对一次文档中的环境口径、结果口径和文件路径
- 决定是否将 `reports/final_*` 的部分表格直接嵌入 README 首页

## 可选优化

- 增加轻量环境检查脚本，例如 `check_env.py`
- 进一步拆分“长耗时训练入口”和“读取已保存结果入口”
- 进一步区分 notebook 与脚本职责
- 可选地继续优化 EDA 图和 SHAP 相关展示说明
- 中期再评估是否拆分 `pipeline.py` / `operation.py`
