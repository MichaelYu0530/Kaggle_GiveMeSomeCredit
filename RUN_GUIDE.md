# 运行与结果阅读指南

## 准备环境和数据

推荐 Python 3.13，并在仓库根目录执行命令。仓库没有原始数据，需自行取得 Kaggle Give Me Some Credit 的 `cs-training.csv` 并放在根目录。现有最终报告在该有标签文件上分层留出 20% 内部测试集，**不读取 Kaggle 的 `cs-test.csv`**。

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows 下使用 `.venv\Scripts\python.exe` 代替 `.venv/bin/python`。`requirements.txt` 是推荐安装入口，`requirements_lock.txt` 记录更完整的锁定版本，`requirements_minimal.txt` 是不锁版本的简洁列表。详情见 [ENVIRONMENT.md](ENVIRONMENT.md)。

## 快速复现最终模型

```bash
.venv/bin/python scripts/run_final_report.py
```

根目录的旧命令 `.venv/bin/python run_final_report.py` 仍可使用。脚本在训练侧重新拟合定稿的 Raw LR、WOE LR/ScoreCard、XGBoost，跳过特征与参数搜索，然后写出：

- `reports/final_model_comparison.{csv,md}`
- `reports/final_f2_threshold_summary.{csv,md}`
- `reports/final_scorecard_bins.{csv,md}`
- `reports/final_report_run_summary.md`

现有 `reports/final_*` 是模块化迁移前的结果，仓库缺少原始 CSV，迁移后的代码尚未完成全数据数值对照。请先备份已有报告，再运行脚本；与迁移前的结果对齐后再更新对外引用。F2 表在内部测试集上事后选择最佳阈值，具体解释见 [架构说明](docs/architecture.md)。

## 研究流程和历史输出

```bash
.venv/bin/python scripts/run_research.py
```

该入口调用 `operation.py`，运行当前**启用**的五折研究阶段，可能包含较耗时的 XGBoost 参数搜索。`operation.py` 的前两段研究开关当前关闭，后四段启用；`operation.ipynb` 六段开关均启用。若只是查看实验，优先阅读 `operation_full_output.ipynb`、[Markdown 归档](reports/operation_full_output.md) 和已有图像，无需重跑。历史 notebook 产物与 `reports/final_*` 的局部指标可能不同，不能混作同一次最终运行。

## 轻量检查

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q gmsc scripts operation.py run_final_report.py
```

这两项不代替原始数据上的三模型数值复现。模块职责见 [功能模块说明](docs/modules.md)，研究和定稿入口的数据流见 [架构说明](docs/architecture.md)。
