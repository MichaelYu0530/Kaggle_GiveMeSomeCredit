# Operation Full Output Notebook

> **说明：本 notebook 主要用于保存和展示一次完整运行后的输出结果。**
>
> 该文件是结果归档文件，不建议直接重新运行全部单元格，否则可能覆盖当前保存的输出结果。
>
> 如需复现实验流程，请优先运行 `run_final_report.py`，并查看 `reports/final_*` 结果文件。
>
> 如需查看完整研究/调参流程，请参考 `operation.py` 或 `operation.ipynb`。



```python
%load_ext autoreload
%autoreload 2
```


```python
""" 导入库 """
from pipeline import *
from visualization import *
import matplotlib.pyplot as plt
from IPython.display import display
import warnings
import gc

""" 数据处理以外的准备工作 """
warnings.filterwarnings('ignore') # 忽略警告信息
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False

import gc

def cleanup_vars(varnames, namespace=None, close_plots=True, verbose=False):
    """Delete temporary variables from a namespace and trigger garbage collection."""
    if namespace is None:
        namespace = globals()
    deleted = []
    for name in varnames:
        if name in namespace:
            del namespace[name]
            deleted.append(name)
    if close_plots:
        try:
            plt.close('all')
        except NameError:
            pass
    gc.collect()
    if verbose and deleted:
        print(f'Cleaned {len(deleted)} temporary variables:', deleted)

# 更好地打印数据框的辅助函数
def print_df(df: pd.DataFrame, precision: int=4, is_index_hidden: bool=True):
    df_print = df.copy()
    
    def smart_float_format(x: float, epsilon: float=1e-10) -> str:
        for n in range(precision):
            if abs(x - round(x, n)) < epsilon:
                return str(round(x, n))
        return str(round(x, precision))
    
    def smart_p_format(x: float) -> str:
        if x < 0.001:
            return f'<0.001***'
        elif x < 0.01:
            return f'{x}**'
        elif x < 0.05:
            return f'{x}*'
        else:
            return f'{x}'
    
    # 对p值一列添加显著性提示符号
    p_value_cols = [col for col in df_print.columns if 'p_value' in col or 'p value' in col]
    for col in p_value_cols:
        df_print[col] = df_print[col].map(smart_p_format)
    
    # 找出所有可以安全转换为数值类型的列
    float_cols = []
    for col in df_print.columns:
        # 尝试转换为数值，如果成功且是浮点类型，则加入列表
        try:
            converted = pd.to_numeric(df_print[col], errors='raise')
            if pd.api.types.is_float_dtype(converted):
                float_cols.append(col)
                df_print[col] = converted
        except (ValueError, TypeError):
            continue
    
    # 只对浮点数列进行格式化
    float_cols = df_print.select_dtypes(include=['float']).columns
    for col in float_cols:
        df_print[col] = df_print[col].map(smart_float_format)
    if is_index_hidden:
        display(df_print.style.hide(axis='index').format_index(escape="html", axis=1))
    else:
        display(df_print.style.format_index(escape="html", axis=1))

```


```python
""" 初步探索数据 """
# 加载原始数据
origin_data = import_data(filetype='train')
# 基础可视化
fig = show_basic_visualization(origin_data)
```


    
![png](reports/operation_full_output_files/reports/operation_full_output_3_0.png)
    



```python
plt.close(fig)
```


```python
# 补充的重要发现
special_discovery_df = show_special_discovery(origin_data)
print_df(special_discovery_df)
```


<style type="text/css">
</style>
<table id="T_1000e">
  <thead>
    <tr>
      <th id="T_1000e_level0_col0" class="col_heading level0 col0" >相关列</th>
      <th id="T_1000e_level0_col1" class="col_heading level0 col1" >重要发现</th>
      <th id="T_1000e_level0_col2" class="col_heading level0 col2" >由此推出的结论</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_1000e_row0_col0" class="data row0 col0" >月收入与家属数</td>
      <td id="T_1000e_row0_col1" class="data row0 col1" >只有这两列存在缺失值，其他列无缺失</td>
      <td id="T_1000e_row0_col2" class="data row0 col2" >后续只需处理这两列的缺失值</td>
    </tr>
    <tr>
      <td id="T_1000e_row1_col0" class="data row1 col0" >家属数→月收入</td>
      <td id="T_1000e_row1_col1" class="data row1 col1" >家庭成员缺失共3924人，其中月收入也缺失3924人（True）</td>
      <td id="T_1000e_row1_col2" class="data row1 col2" >家庭成员缺失的人，月收入必然缺失</td>
    </tr>
    <tr>
      <td id="T_1000e_row2_col0" class="data row2 col0" >月收入→家属数</td>
      <td id="T_1000e_row2_col1" class="data row2 col1" >月收入缺失共29731人，其中家庭成员也缺失3924人</td>
      <td id="T_1000e_row2_col2" class="data row2 col2" >月收入缺失时，家庭成员不一定缺失（存在仅月收入缺失的样本）</td>
    </tr>
    <tr>
      <td id="T_1000e_row3_col0" class="data row3 col0" >三个逾期次数指标</td>
      <td id="T_1000e_row3_col1" class="data row3 col1" >96/98编码完全对应同一批人，共269人</td>
      <td id="T_1000e_row3_col2" class="data row3 col2" >96/98不具有业务意义，是银行赋予的特殊标记</td>
    </tr>
    <tr>
      <td id="T_1000e_row4_col0" class="data row4 col0" >房贷数与信贷数</td>
      <td id="T_1000e_row4_col1" class="data row4 col1" >同一个样本的信贷数不会低于房贷数</td>
      <td id="T_1000e_row4_col2" class="data row4 col2" >房贷是信贷的其中一种</td>
    </tr>
  </tbody>
</table>




```python
""" 初步数据清洗，针对客观事实，允许在全局数据集进行 """
# 缺失值相关标记
completeness_desc_df, completeness_test_df = add_missing_flag(origin_data)
print_df(completeness_desc_df, precision=2)
```


<style type="text/css">
</style>
<table id="T_7f049">
  <thead>
    <tr>
      <th id="T_7f049_level0_col0" class="col_heading level0 col0" >组别</th>
      <th id="T_7f049_level0_col1" class="col_heading level0 col1" >人数</th>
      <th id="T_7f049_level0_col2" class="col_heading level0 col2" >平均年龄</th>
      <th id="T_7f049_level0_col3" class="col_heading level0 col3" >平均信贷数</th>
      <th id="T_7f049_level0_col4" class="col_heading level0 col4" >平均房贷数</th>
      <th id="T_7f049_level0_col5" class="col_heading level0 col5" >平均违约率(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_7f049_row0_col0" class="data row0 col0" >完整组</td>
      <td id="T_7f049_row0_col1" class="data row0 col1" >120269</td>
      <td id="T_7f049_row0_col2" class="data row0 col2" >51.29</td>
      <td id="T_7f049_row0_col3" class="data row0 col3" >8.76</td>
      <td id="T_7f049_row0_col4" class="data row0 col4" >1.05</td>
      <td id="T_7f049_row0_col5" class="data row0 col5" >6.95</td>
    </tr>
    <tr>
      <td id="T_7f049_row1_col0" class="data row1 col0" >单缺组</td>
      <td id="T_7f049_row1_col1" class="data row1 col1" >25807</td>
      <td id="T_7f049_row1_col2" class="data row1 col2" >55.87</td>
      <td id="T_7f049_row1_col3" class="data row1 col3" >7.46</td>
      <td id="T_7f049_row1_col4" class="data row1 col4" >0.91</td>
      <td id="T_7f049_row1_col5" class="data row1 col5" >5.77</td>
    </tr>
    <tr>
      <td id="T_7f049_row2_col0" class="data row2 col0" >全缺组</td>
      <td id="T_7f049_row2_col1" class="data row2 col1" >3924</td>
      <td id="T_7f049_row2_col2" class="data row2 col2" >59.59</td>
      <td id="T_7f049_row2_col3" class="data row2 col3" >5.6</td>
      <td id="T_7f049_row2_col4" class="data row2 col4" >0.59</td>
      <td id="T_7f049_row2_col5" class="data row2 col5" >4.56</td>
    </tr>
  </tbody>
</table>




```python
print_df(completeness_test_df)
```


<style type="text/css">
</style>
<table id="T_d2aee">
  <thead>
    <tr>
      <th id="T_d2aee_level0_col0" class="col_heading level0 col0" >变量</th>
      <th id="T_d2aee_level0_col1" class="col_heading level0 col1" >检验方法</th>
      <th id="T_d2aee_level0_col2" class="col_heading level0 col2" >统计量</th>
      <th id="T_d2aee_level0_col3" class="col_heading level0 col3" >p 值</th>
      <th id="T_d2aee_level0_col4" class="col_heading level0 col4" >效应值名称</th>
      <th id="T_d2aee_level0_col5" class="col_heading level0 col5" >效应量</th>
      <th id="T_d2aee_level0_col6" class="col_heading level0 col6" >效应解释</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_d2aee_row0_col0" class="data row0 col0" >违约情况</td>
      <td id="T_d2aee_row0_col1" class="data row0 col1" >卡方检验</td>
      <td id="T_d2aee_row0_col2" class="data row0 col2" >76.13</td>
      <td id="T_d2aee_row0_col3" class="data row0 col3" ><0.001***</td>
      <td id="T_d2aee_row0_col4" class="data row0 col4" >Cramer's V</td>
      <td id="T_d2aee_row0_col5" class="data row0 col5" >0.0225</td>
      <td id="T_d2aee_row0_col6" class="data row0 col6" >可忽略</td>
    </tr>
    <tr>
      <td id="T_d2aee_row1_col0" class="data row1 col0" >年龄</td>
      <td id="T_d2aee_row1_col1" class="data row1 col1" >单因素方差分析</td>
      <td id="T_d2aee_row1_col2" class="data row1 col2" >1540.0</td>
      <td id="T_d2aee_row1_col3" class="data row1 col3" ><0.001***</td>
      <td id="T_d2aee_row1_col4" class="data row1 col4" >Cohen's f</td>
      <td id="T_d2aee_row1_col5" class="data row1 col5" >0.1435</td>
      <td id="T_d2aee_row1_col6" class="data row1 col6" >小效应</td>
    </tr>
    <tr>
      <td id="T_d2aee_row2_col0" class="data row2 col0" >信贷数</td>
      <td id="T_d2aee_row2_col1" class="data row2 col1" >Kruskal-Wallis H 检验</td>
      <td id="T_d2aee_row2_col2" class="data row2 col2" >3190.0</td>
      <td id="T_d2aee_row2_col3" class="data row2 col3" ><0.001***</td>
      <td id="T_d2aee_row2_col4" class="data row2 col4" >Epsilon2</td>
      <td id="T_d2aee_row2_col5" class="data row2 col5" >0.0213</td>
      <td id="T_d2aee_row2_col6" class="data row2 col6" >小效应</td>
    </tr>
    <tr>
      <td id="T_d2aee_row3_col0" class="data row3 col0" >房贷数</td>
      <td id="T_d2aee_row3_col1" class="data row3 col1" >Kruskal-Wallis H 检验</td>
      <td id="T_d2aee_row3_col2" class="data row3 col2" >1210.0</td>
      <td id="T_d2aee_row3_col3" class="data row3 col3" ><0.001***</td>
      <td id="T_d2aee_row3_col4" class="data row3 col4" >Epsilon2</td>
      <td id="T_d2aee_row3_col5" class="data row3 col5" >0.008</td>
      <td id="T_d2aee_row3_col6" class="data row3 col6" >可忽略</td>
    </tr>
  </tbody>
</table>




```python
# 逾期次数 96/98 相关标记
blacklist_desc_df, blacklist_test_df = add_blacklist_flag(origin_data)
print_df(blacklist_desc_df, precision=2)
```


<style type="text/css">
</style>
<table id="T_641b0">
  <thead>
    <tr>
      <th id="T_641b0_level0_col0" class="col_heading level0 col0" >组别</th>
      <th id="T_641b0_level0_col1" class="col_heading level0 col1" >人数</th>
      <th id="T_641b0_level0_col2" class="col_heading level0 col2" >平均年龄</th>
      <th id="T_641b0_level0_col3" class="col_heading level0 col3" >平均信贷数</th>
      <th id="T_641b0_level0_col4" class="col_heading level0 col4" >平均房贷数</th>
      <th id="T_641b0_level0_col5" class="col_heading level0 col5" >平均违约率(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_641b0_row0_col0" class="data row0 col0" >有编码组</td>
      <td id="T_641b0_row0_col1" class="data row0 col1" >269</td>
      <td id="T_641b0_row0_col2" class="data row0 col2" >34.25</td>
      <td id="T_641b0_row0_col3" class="data row0 col3" >0.01</td>
      <td id="T_641b0_row0_col4" class="data row0 col4" >0.0</td>
      <td id="T_641b0_row0_col5" class="data row0 col5" >54.65</td>
    </tr>
    <tr>
      <td id="T_641b0_row1_col0" class="data row1 col0" >无编码组</td>
      <td id="T_641b0_row1_col1" class="data row1 col1" >149731</td>
      <td id="T_641b0_row1_col2" class="data row1 col2" >52.33</td>
      <td id="T_641b0_row1_col3" class="data row1 col3" >8.47</td>
      <td id="T_641b0_row1_col4" class="data row1 col4" >1.02</td>
      <td id="T_641b0_row1_col5" class="data row1 col5" >6.6</td>
    </tr>
  </tbody>
</table>




```python
print_df(blacklist_test_df)
```


<style type="text/css">
</style>
<table id="T_56245">
  <thead>
    <tr>
      <th id="T_56245_level0_col0" class="col_heading level0 col0" >变量</th>
      <th id="T_56245_level0_col1" class="col_heading level0 col1" >检验方法</th>
      <th id="T_56245_level0_col2" class="col_heading level0 col2" >统计量</th>
      <th id="T_56245_level0_col3" class="col_heading level0 col3" >p 值</th>
      <th id="T_56245_level0_col4" class="col_heading level0 col4" >效应值名称</th>
      <th id="T_56245_level0_col5" class="col_heading level0 col5" >效应量</th>
      <th id="T_56245_level0_col6" class="col_heading level0 col6" >效应解释</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_56245_row0_col0" class="data row0 col0" >违约情况</td>
      <td id="T_56245_row0_col1" class="data row0 col1" >卡方检验</td>
      <td id="T_56245_row0_col2" class="data row0 col2" >986.0</td>
      <td id="T_56245_row0_col3" class="data row0 col3" ><0.001***</td>
      <td id="T_56245_row0_col4" class="data row0 col4" >Odds Ratio</td>
      <td id="T_56245_row0_col5" class="data row0 col5" >17.0574</td>
      <td id="T_56245_row0_col6" class="data row0 col6" >极大效应</td>
    </tr>
    <tr>
      <td id="T_56245_row1_col0" class="data row1 col0" >年龄</td>
      <td id="T_56245_row1_col1" class="data row1 col1" >Welch's t 检验</td>
      <td id="T_56245_row1_col2" class="data row1 col2" >-22.68</td>
      <td id="T_56245_row1_col3" class="data row1 col3" ><0.001***</td>
      <td id="T_56245_row1_col4" class="data row1 col4" >Cohen's d</td>
      <td id="T_56245_row1_col5" class="data row1 col5" >-1.2977</td>
      <td id="T_56245_row1_col6" class="data row1 col6" >大效应</td>
    </tr>
    <tr>
      <td id="T_56245_row2_col0" class="data row2 col0" >信贷数</td>
      <td id="T_56245_row2_col1" class="data row2 col1" >Mann-Whitney U 检验</td>
      <td id="T_56245_row2_col2" class="data row2 col2" >224000.0</td>
      <td id="T_56245_row2_col3" class="data row2 col3" ><0.001***</td>
      <td id="T_56245_row2_col4" class="data row2 col4" >Cliff's delta</td>
      <td id="T_56245_row2_col5" class="data row2 col5" >-0.9889</td>
      <td id="T_56245_row2_col6" class="data row2 col6" >大效应</td>
    </tr>
    <tr>
      <td id="T_56245_row3_col0" class="data row3 col0" >房贷数</td>
      <td id="T_56245_row3_col1" class="data row3 col1" >Mann-Whitney U 检验</td>
      <td id="T_56245_row3_col2" class="data row3 col2" >7520000.0</td>
      <td id="T_56245_row3_col3" class="data row3 col3" ><0.001***</td>
      <td id="T_56245_row3_col4" class="data row3 col4" >Cliff's delta</td>
      <td id="T_56245_row3_col5" class="data row3 col5" >-0.6265</td>
      <td id="T_56245_row3_col6" class="data row3 col6" >大效应</td>
    </tr>
  </tbody>
</table>




```python
# 月收入异常（极低，0-10）相关标记
zero_vs_low_desc_df, zero_vs_low_test_df, abnormal_vs_normal_desc_df, abnormal_vs_normal_test_df = \
    add_income_anomaly_flag(origin_data)
print_df(zero_vs_low_desc_df, precision=2)
```


<style type="text/css">
</style>
<table id="T_c10f2">
  <thead>
    <tr>
      <th id="T_c10f2_level0_col0" class="col_heading level0 col0" >组别</th>
      <th id="T_c10f2_level0_col1" class="col_heading level0 col1" >人数</th>
      <th id="T_c10f2_level0_col2" class="col_heading level0 col2" >平均年龄</th>
      <th id="T_c10f2_level0_col3" class="col_heading level0 col3" >平均信贷数</th>
      <th id="T_c10f2_level0_col4" class="col_heading level0 col4" >平均房贷数</th>
      <th id="T_c10f2_level0_col5" class="col_heading level0 col5" >平均违约率(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_c10f2_row0_col0" class="data row0 col0" >月收入0组</td>
      <td id="T_c10f2_row0_col1" class="data row0 col1" >1634</td>
      <td id="T_c10f2_row0_col2" class="data row0 col2" >48.34</td>
      <td id="T_c10f2_row0_col3" class="data row0 col3" >7.06</td>
      <td id="T_c10f2_row0_col4" class="data row0 col4" >0.72</td>
      <td id="T_c10f2_row0_col5" class="data row0 col5" >4.04</td>
    </tr>
    <tr>
      <td id="T_c10f2_row1_col0" class="data row1 col0" >月收入1-10组</td>
      <td id="T_c10f2_row1_col1" class="data row1 col1" >619</td>
      <td id="T_c10f2_row1_col2" class="data row1 col2" >45.98</td>
      <td id="T_c10f2_row1_col3" class="data row1 col3" >7.45</td>
      <td id="T_c10f2_row1_col4" class="data row1 col4" >0.81</td>
      <td id="T_c10f2_row1_col5" class="data row1 col5" >2.91</td>
    </tr>
  </tbody>
</table>




```python
print_df(zero_vs_low_test_df)
```


<style type="text/css">
</style>
<table id="T_17646">
  <thead>
    <tr>
      <th id="T_17646_level0_col0" class="col_heading level0 col0" >变量</th>
      <th id="T_17646_level0_col1" class="col_heading level0 col1" >检验方法</th>
      <th id="T_17646_level0_col2" class="col_heading level0 col2" >统计量</th>
      <th id="T_17646_level0_col3" class="col_heading level0 col3" >p 值</th>
      <th id="T_17646_level0_col4" class="col_heading level0 col4" >效应值名称</th>
      <th id="T_17646_level0_col5" class="col_heading level0 col5" >效应量</th>
      <th id="T_17646_level0_col6" class="col_heading level0 col6" >效应解释</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_17646_row0_col0" class="data row0 col0" >违约情况</td>
      <td id="T_17646_row0_col1" class="data row0 col1" >卡方检验</td>
      <td id="T_17646_row0_col2" class="data row0 col2" >1.3</td>
      <td id="T_17646_row0_col3" class="data row0 col3" >0.2540</td>
      <td id="T_17646_row0_col4" class="data row0 col4" >Odds Ratio</td>
      <td id="T_17646_row0_col5" class="data row0 col5" >1.4054</td>
      <td id="T_17646_row0_col6" class="data row0 col6" >可忽略</td>
    </tr>
    <tr>
      <td id="T_17646_row1_col0" class="data row1 col0" >年龄</td>
      <td id="T_17646_row1_col1" class="data row1 col1" >Welch's t 检验</td>
      <td id="T_17646_row1_col2" class="data row1 col2" >3.29</td>
      <td id="T_17646_row1_col3" class="data row1 col3" >0.0010**</td>
      <td id="T_17646_row1_col4" class="data row1 col4" >Cohen's d</td>
      <td id="T_17646_row1_col5" class="data row1 col5" >0.1505</td>
      <td id="T_17646_row1_col6" class="data row1 col6" >小效应</td>
    </tr>
    <tr>
      <td id="T_17646_row2_col0" class="data row2 col0" >信贷数</td>
      <td id="T_17646_row2_col1" class="data row2 col1" >Mann-Whitney U 检验</td>
      <td id="T_17646_row2_col2" class="data row2 col2" >480000.0</td>
      <td id="T_17646_row2_col3" class="data row2 col3" >0.0649</td>
      <td id="T_17646_row2_col4" class="data row2 col4" >Cliff's delta</td>
      <td id="T_17646_row2_col5" class="data row2 col5" >-0.0502</td>
      <td id="T_17646_row2_col6" class="data row2 col6" >可忽略</td>
    </tr>
    <tr>
      <td id="T_17646_row3_col0" class="data row3 col0" >房贷数</td>
      <td id="T_17646_row3_col1" class="data row3 col1" >Mann-Whitney U 检验</td>
      <td id="T_17646_row3_col2" class="data row3 col2" >475000.0</td>
      <td id="T_17646_row3_col3" class="data row3 col3" >0.0162*</td>
      <td id="T_17646_row3_col4" class="data row3 col4" >Cliff's delta</td>
      <td id="T_17646_row3_col5" class="data row3 col5" >-0.0602</td>
      <td id="T_17646_row3_col6" class="data row3 col6" >可忽略</td>
    </tr>
  </tbody>
</table>




```python
print_df(abnormal_vs_normal_desc_df, precision=2)
```


<style type="text/css">
</style>
<table id="T_f7d58">
  <thead>
    <tr>
      <th id="T_f7d58_level0_col0" class="col_heading level0 col0" >组别</th>
      <th id="T_f7d58_level0_col1" class="col_heading level0 col1" >人数</th>
      <th id="T_f7d58_level0_col2" class="col_heading level0 col2" >平均年龄</th>
      <th id="T_f7d58_level0_col3" class="col_heading level0 col3" >平均信贷数</th>
      <th id="T_f7d58_level0_col4" class="col_heading level0 col4" >平均房贷数</th>
      <th id="T_f7d58_level0_col5" class="col_heading level0 col5" >平均违约率(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_f7d58_row0_col0" class="data row0 col0" >月收入异常[0, 10]组</td>
      <td id="T_f7d58_row0_col1" class="data row0 col1" >2253</td>
      <td id="T_f7d58_row0_col2" class="data row0 col2" >47.69</td>
      <td id="T_f7d58_row0_col3" class="data row0 col3" >7.17</td>
      <td id="T_f7d58_row0_col4" class="data row0 col4" >0.74</td>
      <td id="T_f7d58_row0_col5" class="data row0 col5" >3.73</td>
    </tr>
    <tr>
      <td id="T_f7d58_row1_col0" class="data row1 col0" >其余月收入（无NA）组</td>
      <td id="T_f7d58_row1_col1" class="data row1 col1" >118016</td>
      <td id="T_f7d58_row1_col2" class="data row1 col2" >51.36</td>
      <td id="T_f7d58_row1_col3" class="data row1 col3" >8.79</td>
      <td id="T_f7d58_row1_col4" class="data row1 col4" >1.06</td>
      <td id="T_f7d58_row1_col5" class="data row1 col5" >7.01</td>
    </tr>
  </tbody>
</table>




```python
print_df(abnormal_vs_normal_test_df)
```


<style type="text/css">
</style>
<table id="T_e9dcb">
  <thead>
    <tr>
      <th id="T_e9dcb_level0_col0" class="col_heading level0 col0" >变量</th>
      <th id="T_e9dcb_level0_col1" class="col_heading level0 col1" >检验方法</th>
      <th id="T_e9dcb_level0_col2" class="col_heading level0 col2" >统计量</th>
      <th id="T_e9dcb_level0_col3" class="col_heading level0 col3" >p 值</th>
      <th id="T_e9dcb_level0_col4" class="col_heading level0 col4" >效应值名称</th>
      <th id="T_e9dcb_level0_col5" class="col_heading level0 col5" >效应量</th>
      <th id="T_e9dcb_level0_col6" class="col_heading level0 col6" >效应解释</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_e9dcb_row0_col0" class="data row0 col0" >违约情况</td>
      <td id="T_e9dcb_row0_col1" class="data row0 col1" >卡方检验</td>
      <td id="T_e9dcb_row0_col2" class="data row0 col2" >36.32</td>
      <td id="T_e9dcb_row0_col3" class="data row0 col3" ><0.001***</td>
      <td id="T_e9dcb_row0_col4" class="data row0 col4" >Odds Ratio</td>
      <td id="T_e9dcb_row0_col5" class="data row0 col5" >0.5137</td>
      <td id="T_e9dcb_row0_col6" class="data row0 col6" >小效应</td>
    </tr>
    <tr>
      <td id="T_e9dcb_row1_col0" class="data row1 col0" >年龄</td>
      <td id="T_e9dcb_row1_col1" class="data row1 col1" >Welch's t 检验</td>
      <td id="T_e9dcb_row1_col2" class="data row1 col2" >-10.69</td>
      <td id="T_e9dcb_row1_col3" class="data row1 col3" ><0.001***</td>
      <td id="T_e9dcb_row1_col4" class="data row1 col4" >Cohen's d</td>
      <td id="T_e9dcb_row1_col5" class="data row1 col5" >-0.2398</td>
      <td id="T_e9dcb_row1_col6" class="data row1 col6" >中效应</td>
    </tr>
    <tr>
      <td id="T_e9dcb_row2_col0" class="data row2 col0" >信贷数</td>
      <td id="T_e9dcb_row2_col1" class="data row2 col1" >Mann-Whitney U 检验</td>
      <td id="T_e9dcb_row2_col2" class="data row2 col2" >107000000.0</td>
      <td id="T_e9dcb_row2_col3" class="data row2 col3" ><0.001***</td>
      <td id="T_e9dcb_row2_col4" class="data row2 col4" >Cliff's delta</td>
      <td id="T_e9dcb_row2_col5" class="data row2 col5" >-0.1971</td>
      <td id="T_e9dcb_row2_col6" class="data row2 col6" >小效应</td>
    </tr>
    <tr>
      <td id="T_e9dcb_row3_col0" class="data row3 col0" >房贷数</td>
      <td id="T_e9dcb_row3_col1" class="data row3 col1" >Mann-Whitney U 检验</td>
      <td id="T_e9dcb_row3_col2" class="data row3 col2" >111000000.0</td>
      <td id="T_e9dcb_row3_col3" class="data row3 col3" ><0.001***</td>
      <td id="T_e9dcb_row3_col4" class="data row3 col4" >Cliff's delta</td>
      <td id="T_e9dcb_row3_col5" class="data row3 col5" >-0.1674</td>
      <td id="T_e9dcb_row3_col6" class="data row3 col6" >小效应</td>
    </tr>
  </tbody>
</table>




```python
""" 分离测试集 """
# region
random_seed = 50
train_valid_data, test_data = train_test_split(
    origin_data,
    test_size=0.2, 
    random_state=random_seed,
    stratify=origin_data[target]
)
train_valid_data_copy = train_valid_data.copy(); test_data_copy = test_data.copy() # 备份
# endregion
```


```python
""" 筛选值得添加交互标记的交互项 """
is_processing_here1 = True

if is_processing_here1:
    # region
    valuable_intersections_df_list = []
    # 划分训练集、验证集，准备K=5折交叉验证
    random_seed = 100
    folds = create_kfold(train_valid_data, random_seed, n_splits=5)
    for fold_idx in range(len(folds)):
        # 训练集、验证集拆分
        train_idx_list, _ = folds[fold_idx]
        train_data = train_valid_data.iloc[train_idx_list].copy()
        # endregion
        
        """ 统一数据清洗阶段 """
        # region
        silently = False if fold_idx == 0 else True
        # 截断家属数异常大值
        cap_dep_num(train_data, silently=silently)
        # 删除年龄异常小值
        delete_small_age(train_data, silently=silently)
        # 添加信用额度使用率的若干标记
        add_util_flags(train_data, silently=silently)
        # 添加负债率的若干标记
        add_debt_flags(train_data, silently=silently)
        # 添加三种逾期次数的加权求和，权重由逻辑回归模型训练
        beta30, beta60, beta90 = add_late_severity_score(train_data)
        # 添加衍生指标
        add_derived_features(train_data)
        # 整理指标顺序
        train_data = reorder_columns(train_data)
        # endregion
        
        """ 构建全部的一维分箱、二维分箱，计算每个交互项的评判指标 """
        # region
        # 筛选出所有非二元指标
        colnames_not_binary = get_colnames_whether_binary(train_data, is_binary=False)
        # 对所有非布尔值类型的指标建立一维分箱
        for colname in colnames_not_binary:
            _, div_pts = add_bins_1D(train_data, colname)
        # 把一维分箱的具体数值转换为排名序号
        convert_bins_value_to_rank(train_data)
        # 列出通过筛选的一维分箱对应指标的所有无序对
        colname_pairs_not_binary = []
        for i in range(len(colnames_not_binary) - 1):
                for j in range(i + 1, len(colnames_not_binary)):
                    colname_pairs_not_binary.append([colnames_not_binary[i], colnames_not_binary[j]])
        # 对两两配对的每对指标建立二维分箱
        for colname_pair in colname_pairs_not_binary:
            _, bins_2D = add_bins_2D(train_data, colname_pair)
        # 找出有价值添加交互标记的二维分箱
        valuable_intersections_df = find_valuable_intersections(train_data, colname_pairs_not_binary)
        valuable_intersections_df_list.append(valuable_intersections_df)
        # endregion
```

    NumberOfDependents大于10的样本数为1，已删除
    Age小于18的样本数为1，已删除
    RevolvingUtilizationOfUnsecuredLines处于0.75-1的样本数有15437，占比16.08%，违约率18.28%
    RevolvingUtilizationOfUnsecuredLines处于1-5的样本数有1942，占比2.02%，违约率40.37%
    RevolvingUtilizationOfUnsecuredLines处于5以上的样本数有165，占比0.17%，违约率8.48%
    月收入可靠的样本数为75417，占比78.56%
    月收入可靠的样本中，DebtRatio处于0.75-1的样本数有3353，占比4.45%，违约率11.18%
    月收入可靠的样本中，DebtRatio处于1-5的样本数有3137，占比4.16%，违约率13.71%
    月收入可靠的样本中，DebtRatio处于5以上有205，占比0.27%，违约率7.8%
    


```python
# 筛选交互效应为增强的排名前30的二维分箱与交互效应为削弱的排名前10的二维分箱，再做人工复查：删除或合并
if is_processing_here1: # 只在最后一折时做
    top_positive_intersections_df, top_negative_intersections_df = \
        filter_intersections(valuable_intersections_df_list)
    print_df(top_positive_intersections_df)
```


<style type="text/css">
</style>
<table id="T_60630">
  <thead>
    <tr>
      <th id="T_60630_level0_col0" class="col_heading level0 col0" >binname_2D</th>
      <th id="T_60630_level0_col1" class="col_heading level0 col1" >bin1_1D</th>
      <th id="T_60630_level0_col2" class="col_heading level0 col2" >bin2_1D</th>
      <th id="T_60630_level0_col3" class="col_heading level0 col3" >LIFT_mean</th>
      <th id="T_60630_level0_col4" class="col_heading level0 col4" >corrected_LIFT_mean</th>
      <th id="T_60630_level0_col5" class="col_heading level0 col5" >ASR_mean</th>
      <th id="T_60630_level0_col6" class="col_heading level0 col6" >intersection_Z_score_mean</th>
      <th id="T_60630_level0_col7" class="col_heading level0 col7" >mark</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_60630_row0_col0" class="data row0 col0" >late_severity_score_x_short_late_bin</td>
      <td id="T_60630_row0_col1" class="data row0 col1" >2</td>
      <td id="T_60630_row0_col2" class="data row0 col2" >1</td>
      <td id="T_60630_row0_col3" class="data row0 col3" >4.2638</td>
      <td id="T_60630_row0_col4" class="data row0 col4" >3.2334</td>
      <td id="T_60630_row0_col5" class="data row0 col5" >80.1372</td>
      <td id="T_60630_row0_col6" class="data row0 col6" >24.3277</td>
      <td id="T_60630_row0_col7" class="data row0 col7" >23.8104</td>
    </tr>
    <tr>
      <td id="T_60630_row1_col0" class="data row1 col0" >credit_late_density_x_late_severity_score_bin</td>
      <td id="T_60630_row1_col1" class="data row1 col1" >1</td>
      <td id="T_60630_row1_col2" class="data row1 col2" >2</td>
      <td id="T_60630_row1_col3" class="data row1 col3" >3.7253</td>
      <td id="T_60630_row1_col4" class="data row1 col4" >3.3936</td>
      <td id="T_60630_row1_col5" class="data row1 col5" >71.4466</td>
      <td id="T_60630_row1_col6" class="data row1 col6" >25.4601</td>
      <td id="T_60630_row1_col7" class="data row1 col7" >22.2071</td>
    </tr>
    <tr>
      <td id="T_60630_row2_col0" class="data row2 col0" >30-59late_x_short_late_bin</td>
      <td id="T_60630_row2_col1" class="data row2 col1" >1</td>
      <td id="T_60630_row2_col2" class="data row2 col2" >2</td>
      <td id="T_60630_row2_col3" class="data row2 col3" >4.7788</td>
      <td id="T_60630_row2_col4" class="data row2 col4" >2.2715</td>
      <td id="T_60630_row2_col5" class="data row2 col5" >114.6256</td>
      <td id="T_60630_row2_col6" class="data row2 col6" >18.5459</td>
      <td id="T_60630_row2_col7" class="data row2 col7" >17.7695</td>
    </tr>
    <tr>
      <td id="T_60630_row3_col0" class="data row3 col0" >late_severity_score_x_30-59late_bin</td>
      <td id="T_60630_row3_col1" class="data row3 col1" >2</td>
      <td id="T_60630_row3_col2" class="data row3 col2" >1</td>
      <td id="T_60630_row3_col3" class="data row3 col3" >4.0728</td>
      <td id="T_60630_row3_col4" class="data row3 col4" >2.2552</td>
      <td id="T_60630_row3_col5" class="data row3 col5" >119.1423</td>
      <td id="T_60630_row3_col6" class="data row3 col6" >21.6284</td>
      <td id="T_60630_row3_col7" class="data row3 col7" >16.7819</td>
    </tr>
    <tr>
      <td id="T_60630_row4_col0" class="data row4 col0" >age_x_late_severity_score_bin</td>
      <td id="T_60630_row4_col1" class="data row4 col1" >6</td>
      <td id="T_60630_row4_col2" class="data row4 col2" >4</td>
      <td id="T_60630_row4_col3" class="data row4 col3" >6.7486</td>
      <td id="T_60630_row4_col4" class="data row4 col4" >2.6309</td>
      <td id="T_60630_row4_col5" class="data row4 col5" >24.8951</td>
      <td id="T_60630_row4_col6" class="data row4 col6" >11.025</td>
      <td id="T_60630_row4_col7" class="data row4 col7" >14.2508</td>
    </tr>
    <tr>
      <td id="T_60630_row5_col0" class="data row5 col0" >credit_late_density_x_late_severity_score_bin</td>
      <td id="T_60630_row5_col1" class="data row5 col1" >1</td>
      <td id="T_60630_row5_col2" class="data row5 col2" >4</td>
      <td id="T_60630_row5_col3" class="data row5 col3" >6.8878</td>
      <td id="T_60630_row5_col4" class="data row5 col4" >1.7774</td>
      <td id="T_60630_row5_col5" class="data row5 col5" >91.481</td>
      <td id="T_60630_row5_col6" class="data row5 col6" >10.4991</td>
      <td id="T_60630_row5_col7" class="data row5 col7" >11.7862</td>
    </tr>
    <tr>
      <td id="T_60630_row6_col0" class="data row6 col0" >late_severity_score_x_util_bin</td>
      <td id="T_60630_row6_col1" class="data row6 col1" >4</td>
      <td id="T_60630_row6_col2" class="data row6 col2" >2</td>
      <td id="T_60630_row6_col3" class="data row6 col3" >5.4597</td>
      <td id="T_60630_row6_col4" class="data row6 col4" >3.0583</td>
      <td id="T_60630_row6_col5" class="data row6 col5" >14.7351</td>
      <td id="T_60630_row6_col6" class="data row6 col6" >8.4604</td>
      <td id="T_60630_row6_col7" class="data row6 col7" >10.9003</td>
    </tr>
    <tr>
      <td id="T_60630_row7_col0" class="data row7 col0" >credit_pressure_index_x_debt_bin</td>
      <td id="T_60630_row7_col1" class="data row7 col1" >3</td>
      <td id="T_60630_row7_col2" class="data row7 col2" >1</td>
      <td id="T_60630_row7_col3" class="data row7 col3" >2.2937</td>
      <td id="T_60630_row7_col4" class="data row7 col4" >3.8275</td>
      <td id="T_60630_row7_col5" class="data row7 col5" >19.1431</td>
      <td id="T_60630_row7_col6" class="data row7 col6" >26.4715</td>
      <td id="T_60630_row7_col7" class="data row7 col7" >10.7759</td>
    </tr>
    <tr>
      <td id="T_60630_row8_col0" class="data row8 col0" >late_severity_score_x_util_bin</td>
      <td id="T_60630_row8_col1" class="data row8 col1" >4</td>
      <td id="T_60630_row8_col2" class="data row8 col2" >1</td>
      <td id="T_60630_row8_col3" class="data row8 col3" >5.4204</td>
      <td id="T_60630_row8_col4" class="data row8 col4" >2.5902</td>
      <td id="T_60630_row8_col5" class="data row8 col5" >18.9245</td>
      <td id="T_60630_row8_col6" class="data row8 col6" >9.2742</td>
      <td id="T_60630_row8_col7" class="data row8 col7" >10.535</td>
    </tr>
    <tr>
      <td id="T_60630_row9_col0" class="data row9 col0" >credit_late_density_x_30-59late_bin</td>
      <td id="T_60630_row9_col1" class="data row9 col1" >5</td>
      <td id="T_60630_row9_col2" class="data row9 col2" >1</td>
      <td id="T_60630_row9_col3" class="data row9 col3" >6.2682</td>
      <td id="T_60630_row9_col4" class="data row9 col4" >1.781</td>
      <td id="T_60630_row9_col5" class="data row9 col5" >70.3357</td>
      <td id="T_60630_row9_col6" class="data row9 col6" >7.1636</td>
      <td id="T_60630_row9_col7" class="data row9 col7" >8.872</td>
    </tr>
    <tr>
      <td id="T_60630_row10_col0" class="data row10 col0" >late_severity_score_x_short_late_bin</td>
      <td id="T_60630_row10_col1" class="data row10 col1" >4</td>
      <td id="T_60630_row10_col2" class="data row10 col2" >1</td>
      <td id="T_60630_row10_col3" class="data row10 col3" >7.8761</td>
      <td id="T_60630_row10_col4" class="data row10 col4" >1.5517</td>
      <td id="T_60630_row10_col5" class="data row10 col5" >97.334</td>
      <td id="T_60630_row10_col6" class="data row10 col6" >8.0966</td>
      <td id="T_60630_row10_col7" class="data row10 col7" >8.6818</td>
    </tr>
    <tr>
      <td id="T_60630_row11_col0" class="data row11 col0" >credit_pressure_index_x_debt_bin</td>
      <td id="T_60630_row11_col1" class="data row11 col1" >4</td>
      <td id="T_60630_row11_col2" class="data row11 col2" >1</td>
      <td id="T_60630_row11_col3" class="data row11 col3" >3.152</td>
      <td id="T_60630_row11_col4" class="data row11 col4" >3.0465</td>
      <td id="T_60630_row11_col5" class="data row11 col5" >14.8602</td>
      <td id="T_60630_row11_col6" class="data row11 col6" >11.9393</td>
      <td id="T_60630_row11_col7" class="data row11 col7" >8.5588</td>
    </tr>
    <tr>
      <td id="T_60630_row12_col0" class="data row12 col0" >credit_pressure_index_x_util_bin</td>
      <td id="T_60630_row12_col1" class="data row12 col1" >2</td>
      <td id="T_60630_row12_col2" class="data row12 col2" >5</td>
      <td id="T_60630_row12_col3" class="data row12 col3" >3.2738</td>
      <td id="T_60630_row12_col4" class="data row12 col4" >2.7627</td>
      <td id="T_60630_row12_col5" class="data row12 col5" >16.607</td>
      <td id="T_60630_row12_col6" class="data row12 col6" >11.9926</td>
      <td id="T_60630_row12_col7" class="data row12 col7" >8.4124</td>
    </tr>
    <tr>
      <td id="T_60630_row13_col0" class="data row13 col0" >credit_late_density_x_30-59late_bin</td>
      <td id="T_60630_row13_col1" class="data row13 col1" >4</td>
      <td id="T_60630_row13_col2" class="data row13 col2" >1</td>
      <td id="T_60630_row13_col3" class="data row13 col3" >4.3317</td>
      <td id="T_60630_row13_col4" class="data row13 col4" >2.0206</td>
      <td id="T_60630_row13_col5" class="data row13 col5" >48.5418</td>
      <td id="T_60630_row13_col6" class="data row13 col6" >7.6173</td>
      <td id="T_60630_row13_col7" class="data row13 col7" >8.1286</td>
    </tr>
    <tr>
      <td id="T_60630_row14_col0" class="data row14 col0" >age_x_credit_late_density_bin</td>
      <td id="T_60630_row14_col1" class="data row14 col1" >6</td>
      <td id="T_60630_row14_col2" class="data row14 col2" >6</td>
      <td id="T_60630_row14_col3" class="data row14 col3" >5.1746</td>
      <td id="T_60630_row14_col4" class="data row14 col4" >2.3735</td>
      <td id="T_60630_row14_col5" class="data row14 col5" >15.6461</td>
      <td id="T_60630_row14_col6" class="data row14 col6" >7.3246</td>
      <td id="T_60630_row14_col7" class="data row14 col7" >7.7809</td>
    </tr>
    <tr>
      <td id="T_60630_row15_col0" class="data row15 col0" >credit_pressure_index_x_monthly_debt_bin</td>
      <td id="T_60630_row15_col1" class="data row15 col1" >3</td>
      <td id="T_60630_row15_col2" class="data row15 col2" >1</td>
      <td id="T_60630_row15_col3" class="data row15 col3" >2.1288</td>
      <td id="T_60630_row15_col4" class="data row15 col4" >3.1928</td>
      <td id="T_60630_row15_col5" class="data row15 col5" >16.3953</td>
      <td id="T_60630_row15_col6" class="data row15 col6" >21.3071</td>
      <td id="T_60630_row15_col7" class="data row15 col7" >7.505</td>
    </tr>
    <tr>
      <td id="T_60630_row16_col0" class="data row16 col0" >late_severity_score_x_30-59late_bin</td>
      <td id="T_60630_row16_col1" class="data row16 col1" >3</td>
      <td id="T_60630_row16_col2" class="data row16 col2" >1</td>
      <td id="T_60630_row16_col3" class="data row16 col3" >6.8215</td>
      <td id="T_60630_row16_col4" class="data row16 col4" >1.6117</td>
      <td id="T_60630_row16_col5" class="data row16 col5" >76.4769</td>
      <td id="T_60630_row16_col6" class="data row16 col6" >6.2347</td>
      <td id="T_60630_row16_col7" class="data row16 col7" >7.2744</td>
    </tr>
    <tr>
      <td id="T_60630_row17_col0" class="data row17 col0" >credit_pressure_index_x_debt_bin</td>
      <td id="T_60630_row17_col1" class="data row17 col1" >5</td>
      <td id="T_60630_row17_col2" class="data row17 col2" >2</td>
      <td id="T_60630_row17_col3" class="data row17 col3" >4.6716</td>
      <td id="T_60630_row17_col4" class="data row17 col4" >2.3283</td>
      <td id="T_60630_row17_col5" class="data row17 col5" >16.4683</td>
      <td id="T_60630_row17_col6" class="data row17 col6" >7.2784</td>
      <td id="T_60630_row17_col7" class="data row17 col7" >7.2444</td>
    </tr>
    <tr>
      <td id="T_60630_row18_col0" class="data row18 col0" >credit_late_density_x_util_bin</td>
      <td id="T_60630_row18_col1" class="data row18 col1" >6</td>
      <td id="T_60630_row18_col2" class="data row18 col2" >1</td>
      <td id="T_60630_row18_col3" class="data row18 col3" >4.3531</td>
      <td id="T_60630_row18_col4" class="data row18 col4" >2.4284</td>
      <td id="T_60630_row18_col5" class="data row18 col5" >14.3478</td>
      <td id="T_60630_row18_col6" class="data row18 col6" >7.637</td>
      <td id="T_60630_row18_col7" class="data row18 col7" >7.067</td>
    </tr>
    <tr>
      <td id="T_60630_row19_col0" class="data row19 col0" >credit_pressure_index_x_late_severity_score_bin</td>
      <td id="T_60630_row19_col1" class="data row19 col1" >1</td>
      <td id="T_60630_row19_col2" class="data row19 col2" >4</td>
      <td id="T_60630_row19_col3" class="data row19 col3" >5.5301</td>
      <td id="T_60630_row19_col4" class="data row19 col4" >1.992</td>
      <td id="T_60630_row19_col5" class="data row19 col5" >19.0166</td>
      <td id="T_60630_row19_col6" class="data row19 col6" >6.8859</td>
      <td id="T_60630_row19_col7" class="data row19 col7" >6.6975</td>
    </tr>
    <tr>
      <td id="T_60630_row20_col0" class="data row20 col0" >credit_late_density_x_30-59late_bin</td>
      <td id="T_60630_row20_col1" class="data row20 col1" >3</td>
      <td id="T_60630_row20_col2" class="data row20 col2" >1</td>
      <td id="T_60630_row20_col3" class="data row20 col3" >3.4471</td>
      <td id="T_60630_row20_col4" class="data row20 col4" >2.1605</td>
      <td id="T_60630_row20_col5" class="data row20 col5" >34.759</td>
      <td id="T_60630_row20_col6" class="data row20 col6" >7.1461</td>
      <td id="T_60630_row20_col7" class="data row20 col7" >6.6526</td>
    </tr>
    <tr>
      <td id="T_60630_row21_col0" class="data row21 col0" >30-59late_x_90+late_bin</td>
      <td id="T_60630_row21_col1" class="data row21 col1" >1</td>
      <td id="T_60630_row21_col2" class="data row21 col2" >4</td>
      <td id="T_60630_row21_col3" class="data row21 col3" >8.2812</td>
      <td id="T_60630_row21_col4" class="data row21 col4" >1.4609</td>
      <td id="T_60630_row21_col5" class="data row21 col5" >84.4865</td>
      <td id="T_60630_row21_col6" class="data row21 col6" >6.3768</td>
      <td id="T_60630_row21_col7" class="data row21 col7" >6.5861</td>
    </tr>
    <tr>
      <td id="T_60630_row22_col0" class="data row22 col0" >credit_late_density_x_30-59late_bin</td>
      <td id="T_60630_row22_col1" class="data row22 col1" >6</td>
      <td id="T_60630_row22_col2" class="data row22 col2" >1</td>
      <td id="T_60630_row22_col3" class="data row22 col3" >7.7874</td>
      <td id="T_60630_row22_col4" class="data row22 col4" >1.4652</td>
      <td id="T_60630_row22_col5" class="data row22 col5" >98.8764</td>
      <td id="T_60630_row22_col6" class="data row22 col6" >6.0956</td>
      <td id="T_60630_row22_col7" class="data row22 col7" >6.5108</td>
    </tr>
    <tr>
      <td id="T_60630_row23_col0" class="data row23 col0" >90+late_x_short_late_bin</td>
      <td id="T_60630_row23_col1" class="data row23 col1" >4</td>
      <td id="T_60630_row23_col2" class="data row23 col2" >1</td>
      <td id="T_60630_row23_col3" class="data row23 col3" >7.5268</td>
      <td id="T_60630_row23_col4" class="data row23 col4" >1.5589</td>
      <td id="T_60630_row23_col5" class="data row23 col5" >58.9847</td>
      <td id="T_60630_row23_col6" class="data row23 col6" >5.6762</td>
      <td id="T_60630_row23_col7" class="data row23 col7" >6.3441</td>
    </tr>
    <tr>
      <td id="T_60630_row24_col0" class="data row24 col0" >90+late_x_short_late_bin</td>
      <td id="T_60630_row24_col1" class="data row24 col1" >3</td>
      <td id="T_60630_row24_col2" class="data row24 col2" >1</td>
      <td id="T_60630_row24_col3" class="data row24 col3" >6.1946</td>
      <td id="T_60630_row24_col4" class="data row24 col4" >1.5911</td>
      <td id="T_60630_row24_col5" class="data row24 col5" >57.1989</td>
      <td id="T_60630_row24_col6" class="data row24 col6" >6.2803</td>
      <td id="T_60630_row24_col7" class="data row24 col7" >6.297</td>
    </tr>
    <tr>
      <td id="T_60630_row25_col0" class="data row25 col0" >credit_pressure_index_x_debt_bin</td>
      <td id="T_60630_row25_col1" class="data row25 col1" >4</td>
      <td id="T_60630_row25_col2" class="data row25 col2" >2</td>
      <td id="T_60630_row25_col3" class="data row25 col3" >2.257</td>
      <td id="T_60630_row25_col4" class="data row25 col4" >2.1114</td>
      <td id="T_60630_row25_col5" class="data row25 col5" >25.7574</td>
      <td id="T_60630_row25_col6" class="data row25 col6" >19.5803</td>
      <td id="T_60630_row25_col7" class="data row25 col7" >5.8789</td>
    </tr>
    <tr>
      <td id="T_60630_row26_col0" class="data row26 col0" >credit_pressure_index_x_late_severity_score_bin</td>
      <td id="T_60630_row26_col1" class="data row26 col1" >2</td>
      <td id="T_60630_row26_col2" class="data row26 col2" >4</td>
      <td id="T_60630_row26_col3" class="data row26 col3" >5.3778</td>
      <td id="T_60630_row26_col4" class="data row26 col4" >1.9237</td>
      <td id="T_60630_row26_col5" class="data row26 col5" >17.9404</td>
      <td id="T_60630_row26_col6" class="data row26 col6" >6.2858</td>
      <td id="T_60630_row26_col7" class="data row26 col7" >5.8413</td>
    </tr>
    <tr>
      <td id="T_60630_row27_col0" class="data row27 col0" >credit_late_density_x_credit_bin</td>
      <td id="T_60630_row27_col1" class="data row27 col1" >5</td>
      <td id="T_60630_row27_col2" class="data row27 col2" >6</td>
      <td id="T_60630_row27_col3" class="data row27 col3" >6.1942</td>
      <td id="T_60630_row27_col4" class="data row27 col4" >1.65</td>
      <td id="T_60630_row27_col5" class="data row27 col5" >26.8709</td>
      <td id="T_60630_row27_col6" class="data row27 col6" >6.6995</td>
      <td id="T_60630_row27_col7" class="data row27 col7" >5.7167</td>
    </tr>
    <tr>
      <td id="T_60630_row28_col0" class="data row28 col0" >30-59late_x_util_bin</td>
      <td id="T_60630_row28_col1" class="data row28 col1" >4</td>
      <td id="T_60630_row28_col2" class="data row28 col2" >1</td>
      <td id="T_60630_row28_col3" class="data row28 col3" >4.36</td>
      <td id="T_60630_row28_col4" class="data row28 col4" >2.4928</td>
      <td id="T_60630_row28_col5" class="data row28 col5" >10.3534</td>
      <td id="T_60630_row28_col6" class="data row28 col6" >5.7231</td>
      <td id="T_60630_row28_col7" class="data row28 col7" >5.484</td>
    </tr>
    <tr>
      <td id="T_60630_row29_col0" class="data row29 col0" >credit_pressure_index_x_monthly_debt_bin</td>
      <td id="T_60630_row29_col1" class="data row29 col1" >4</td>
      <td id="T_60630_row29_col2" class="data row29 col2" >1</td>
      <td id="T_60630_row29_col3" class="data row29 col3" >2.5822</td>
      <td id="T_60630_row29_col4" class="data row29 col4" >2.2431</td>
      <td id="T_60630_row29_col5" class="data row29 col5" >17.3889</td>
      <td id="T_60630_row29_col6" class="data row29 col6" >12.2298</td>
      <td id="T_60630_row29_col7" class="data row29 col7" >5.48</td>
    </tr>
    <tr>
      <td id="T_60630_row30_col0" class="data row30 col0" >age_x_90+late_bin</td>
      <td id="T_60630_row30_col1" class="data row30 col1" >6</td>
      <td id="T_60630_row30_col2" class="data row30 col2" >2</td>
      <td id="T_60630_row30_col3" class="data row30 col3" >4.3471</td>
      <td id="T_60630_row30_col4" class="data row30 col4" >2.0863</td>
      <td id="T_60630_row30_col5" class="data row30 col5" >14.5769</td>
      <td id="T_60630_row30_col6" class="data row30 col6" >6.5029</td>
      <td id="T_60630_row30_col7" class="data row30 col7" >5.4211</td>
    </tr>
    <tr>
      <td id="T_60630_row31_col0" class="data row31 col0" >dep_x_60-89late_bin</td>
      <td id="T_60630_row31_col1" class="data row31 col1" >-1</td>
      <td id="T_60630_row31_col2" class="data row31 col2" >2</td>
      <td id="T_60630_row31_col3" class="data row31 col3" >7.4945</td>
      <td id="T_60630_row31_col4" class="data row31 col4" >2.0897</td>
      <td id="T_60630_row31_col5" class="data row31 col5" >12.2007</td>
      <td id="T_60630_row31_col6" class="data row31 col6" >4.2419</td>
      <td id="T_60630_row31_col7" class="data row31 col7" >5.366</td>
    </tr>
    <tr>
      <td id="T_60630_row32_col0" class="data row32 col0" >util_x_short_late_bin</td>
      <td id="T_60630_row32_col1" class="data row32 col1" >1</td>
      <td id="T_60630_row32_col2" class="data row32 col2" >4</td>
      <td id="T_60630_row32_col3" class="data row32 col3" >4.3309</td>
      <td id="T_60630_row32_col4" class="data row32 col4" >2.4902</td>
      <td id="T_60630_row32_col5" class="data row32 col5" >9.8782</td>
      <td id="T_60630_row32_col6" class="data row32 col6" >5.4606</td>
      <td id="T_60630_row32_col7" class="data row32 col7" >5.1994</td>
    </tr>
    <tr>
      <td id="T_60630_row33_col0" class="data row33 col0" >credit_pressure_index_x_util_bin</td>
      <td id="T_60630_row33_col1" class="data row33 col1" >1</td>
      <td id="T_60630_row33_col2" class="data row33 col2" >5</td>
      <td id="T_60630_row33_col3" class="data row33 col3" >2.7747</td>
      <td id="T_60630_row33_col4" class="data row33 col4" >2.1975</td>
      <td id="T_60630_row33_col5" class="data row33 col5" >15.4386</td>
      <td id="T_60630_row33_col6" class="data row33 col6" >10.0459</td>
      <td id="T_60630_row33_col7" class="data row33 col7" >5.0738</td>
    </tr>
    <tr>
      <td id="T_60630_row34_col0" class="data row34 col0" >30-59late_x_90+late_bin</td>
      <td id="T_60630_row34_col1" class="data row34 col1" >1</td>
      <td id="T_60630_row34_col2" class="data row34 col2" >3</td>
      <td id="T_60630_row34_col3" class="data row34 col3" >6.4663</td>
      <td id="T_60630_row34_col4" class="data row34 col4" >1.4143</td>
      <td id="T_60630_row34_col5" class="data row34 col5" >75.6968</td>
      <td id="T_60630_row34_col6" class="data row34 col6" >5.8124</td>
      <td id="T_60630_row34_col7" class="data row34 col7" >4.9274</td>
    </tr>
    <tr>
      <td id="T_60630_row35_col0" class="data row35 col0" >age_x_short_late_bin</td>
      <td id="T_60630_row35_col1" class="data row35 col1" >6</td>
      <td id="T_60630_row35_col2" class="data row35 col2" >4</td>
      <td id="T_60630_row35_col3" class="data row35 col3" >4.5095</td>
      <td id="T_60630_row35_col4" class="data row35 col4" >2.2337</td>
      <td id="T_60630_row35_col5" class="data row35 col5" >10.979</td>
      <td id="T_60630_row35_col6" class="data row35 col6" >5.2784</td>
      <td id="T_60630_row35_col7" class="data row35 col7" >4.8249</td>
    </tr>
    <tr>
      <td id="T_60630_row36_col0" class="data row36 col0" >credit_late_density_x_mortgage_bin</td>
      <td id="T_60630_row36_col1" class="data row36 col1" >6</td>
      <td id="T_60630_row36_col2" class="data row36 col2" >2</td>
      <td id="T_60630_row36_col3" class="data row36 col3" >6.8095</td>
      <td id="T_60630_row36_col4" class="data row36 col4" >1.3441</td>
      <td id="T_60630_row36_col5" class="data row36 col5" >58.009</td>
      <td id="T_60630_row36_col6" class="data row36 col6" >7.4018</td>
      <td id="T_60630_row36_col7" class="data row36 col7" >4.6114</td>
    </tr>
    <tr>
      <td id="T_60630_row37_col0" class="data row37 col0" >age_x_30-59late_bin</td>
      <td id="T_60630_row37_col1" class="data row37 col1" >6</td>
      <td id="T_60630_row37_col2" class="data row37 col2" >3</td>
      <td id="T_60630_row37_col3" class="data row37 col3" >3.3606</td>
      <td id="T_60630_row37_col4" class="data row37 col4" >2.3325</td>
      <td id="T_60630_row37_col5" class="data row37 col5" >10.5389</td>
      <td id="T_60630_row37_col6" class="data row37 col6" >6.7145</td>
      <td id="T_60630_row37_col7" class="data row37 col7" >4.604</td>
    </tr>
    <tr>
      <td id="T_60630_row38_col0" class="data row38 col0" >credit_x_30-59late_bin</td>
      <td id="T_60630_row38_col1" class="data row38 col1" >2</td>
      <td id="T_60630_row38_col2" class="data row38 col2" >3</td>
      <td id="T_60630_row38_col3" class="data row38 col3" >5.8561</td>
      <td id="T_60630_row38_col4" class="data row38 col4" >1.5159</td>
      <td id="T_60630_row38_col5" class="data row38 col5" >29.2277</td>
      <td id="T_60630_row38_col6" class="data row38 col6" >6.3633</td>
      <td id="T_60630_row38_col7" class="data row38 col7" >4.5928</td>
    </tr>
    <tr>
      <td id="T_60630_row39_col0" class="data row39 col0" >credit_late_density_x_mortgage_ratio_bin</td>
      <td id="T_60630_row39_col1" class="data row39 col1" >6</td>
      <td id="T_60630_row39_col2" class="data row39 col2" >2</td>
      <td id="T_60630_row39_col3" class="data row39 col3" >7.2972</td>
      <td id="T_60630_row39_col4" class="data row39 col4" >1.5023</td>
      <td id="T_60630_row39_col5" class="data row39 col5" >28.5406</td>
      <td id="T_60630_row39_col6" class="data row39 col6" >5.4342</td>
      <td id="T_60630_row39_col7" class="data row39 col7" >4.5886</td>
    </tr>
    <tr>
      <td id="T_60630_row40_col0" class="data row40 col0" >credit_pressure_index_x_60-89late_bin</td>
      <td id="T_60630_row40_col1" class="data row40 col1" >1</td>
      <td id="T_60630_row40_col2" class="data row40 col2" >2</td>
      <td id="T_60630_row40_col3" class="data row40 col3" >4.1</td>
      <td id="T_60630_row40_col4" class="data row40 col4" >1.9507</td>
      <td id="T_60630_row40_col5" class="data row40 col5" >14.1339</td>
      <td id="T_60630_row40_col6" class="data row40 col6" >6.049</td>
      <td id="T_60630_row40_col7" class="data row40 col7" >4.4944</td>
    </tr>
    <tr>
      <td id="T_60630_row41_col0" class="data row41 col0" >90+late_x_util_bin</td>
      <td id="T_60630_row41_col1" class="data row41 col1" >2</td>
      <td id="T_60630_row41_col2" class="data row41 col2" >2</td>
      <td id="T_60630_row41_col3" class="data row41 col3" >3.7862</td>
      <td id="T_60630_row41_col4" class="data row41 col4" >2.3091</td>
      <td id="T_60630_row41_col5" class="data row41 col5" >10.1787</td>
      <td id="T_60630_row41_col6" class="data row41 col6" >5.6347</td>
      <td id="T_60630_row41_col7" class="data row41 col7" >4.4698</td>
    </tr>
    <tr>
      <td id="T_60630_row42_col0" class="data row42 col0" >util_x_short_late_bin</td>
      <td id="T_60630_row42_col1" class="data row42 col1" >2</td>
      <td id="T_60630_row42_col2" class="data row42 col2" >4</td>
      <td id="T_60630_row42_col3" class="data row42 col3" >3.9735</td>
      <td id="T_60630_row42_col4" class="data row42 col4" >2.6315</td>
      <td id="T_60630_row42_col5" class="data row42 col5" >7.9463</td>
      <td id="T_60630_row42_col6" class="data row42 col6" >4.9622</td>
      <td id="T_60630_row42_col7" class="data row42 col7" >4.432</td>
    </tr>
    <tr>
      <td id="T_60630_row43_col0" class="data row43 col0" >age_x_60-89late_bin</td>
      <td id="T_60630_row43_col1" class="data row43 col1" >6</td>
      <td id="T_60630_row43_col2" class="data row43 col2" >2</td>
      <td id="T_60630_row43_col3" class="data row43 col3" >3.8233</td>
      <td id="T_60630_row43_col4" class="data row43 col4" >2.0158</td>
      <td id="T_60630_row43_col5" class="data row43 col5" >13.2153</td>
      <td id="T_60630_row43_col6" class="data row43 col6" >6.1983</td>
      <td id="T_60630_row43_col7" class="data row43 col7" >4.4273</td>
    </tr>
    <tr>
      <td id="T_60630_row44_col0" class="data row44 col0" >late_severity_score_x_30-59late_bin</td>
      <td id="T_60630_row44_col1" class="data row44 col1" >4</td>
      <td id="T_60630_row44_col2" class="data row44 col2" >1</td>
      <td id="T_60630_row44_col3" class="data row44 col3" >9.1192</td>
      <td id="T_60630_row44_col4" class="data row44 col4" >1.2505</td>
      <td id="T_60630_row44_col5" class="data row44 col5" >163.8022</td>
      <td id="T_60630_row44_col6" class="data row44 col6" >5.7705</td>
      <td id="T_60630_row44_col7" class="data row44 col7" >4.4165</td>
    </tr>
    <tr>
      <td id="T_60630_row45_col0" class="data row45 col0" >credit_late_density_x_credit_bin</td>
      <td id="T_60630_row45_col1" class="data row45 col1" >6</td>
      <td id="T_60630_row45_col2" class="data row45 col2" >6</td>
      <td id="T_60630_row45_col3" class="data row45 col3" >9.261</td>
      <td id="T_60630_row45_col4" class="data row45 col4" >1.5763</td>
      <td id="T_60630_row45_col5" class="data row45 col5" >21.4189</td>
      <td id="T_60630_row45_col6" class="data row45 col6" >4.117</td>
      <td id="T_60630_row45_col7" class="data row45 col7" >4.3922</td>
    </tr>
    <tr>
      <td id="T_60630_row46_col0" class="data row46 col0" >age_x_30-59late_bin</td>
      <td id="T_60630_row46_col1" class="data row46 col1" >6</td>
      <td id="T_60630_row46_col2" class="data row46 col2" >4</td>
      <td id="T_60630_row46_col3" class="data row46 col3" >4.4378</td>
      <td id="T_60630_row46_col4" class="data row46 col4" >2.0698</td>
      <td id="T_60630_row46_col5" class="data row46 col5" >11.2468</td>
      <td id="T_60630_row46_col6" class="data row46 col6" >4.9724</td>
      <td id="T_60630_row46_col7" class="data row46 col7" >4.2078</td>
    </tr>
    <tr>
      <td id="T_60630_row47_col0" class="data row47 col0" >mortgage_ratio_x_90+late_bin</td>
      <td id="T_60630_row47_col1" class="data row47 col1" >3</td>
      <td id="T_60630_row47_col2" class="data row47 col2" >4</td>
      <td id="T_60630_row47_col3" class="data row47 col3" >10.9927</td>
      <td id="T_60630_row47_col4" class="data row47 col4" >1.4553</td>
      <td id="T_60630_row47_col5" class="data row47 col5" >25.023</td>
      <td id="T_60630_row47_col6" class="data row47 col6" >4.0151</td>
      <td id="T_60630_row47_col7" class="data row47 col7" >4.0258</td>
    </tr>
    <tr>
      <td id="T_60630_row48_col0" class="data row48 col0" >credit_late_density_x_30-59late_bin</td>
      <td id="T_60630_row48_col1" class="data row48 col1" >2</td>
      <td id="T_60630_row48_col2" class="data row48 col2" >1</td>
      <td id="T_60630_row48_col3" class="data row48 col3" >2.9961</td>
      <td id="T_60630_row48_col4" class="data row48 col4" >1.9927</td>
      <td id="T_60630_row48_col5" class="data row48 col5" >24.2302</td>
      <td id="T_60630_row48_col6" class="data row48 col6" >5.0696</td>
      <td id="T_60630_row48_col7" class="data row48 col7" >3.9147</td>
    </tr>
    <tr>
      <td id="T_60630_row49_col0" class="data row49 col0" >30-59late_x_short_late_bin</td>
      <td id="T_60630_row49_col1" class="data row49 col1" >1</td>
      <td id="T_60630_row49_col2" class="data row49 col2" >3</td>
      <td id="T_60630_row49_col3" class="data row49 col3" >7.1848</td>
      <td id="T_60630_row49_col4" class="data row49 col4" >1.4358</td>
      <td id="T_60630_row49_col5" class="data row49 col5" >60.6923</td>
      <td id="T_60630_row49_col6" class="data row49 col6" >3.6771</td>
      <td id="T_60630_row49_col7" class="data row49 col7" >3.8136</td>
    </tr>
  </tbody>
</table>




```python
if is_processing_here1:
    print_df(top_negative_intersections_df)
```


<style type="text/css">
</style>
<table id="T_a1c62">
  <thead>
    <tr>
      <th id="T_a1c62_level0_col0" class="col_heading level0 col0" >binname_2D</th>
      <th id="T_a1c62_level0_col1" class="col_heading level0 col1" >bin1_1D</th>
      <th id="T_a1c62_level0_col2" class="col_heading level0 col2" >bin2_1D</th>
      <th id="T_a1c62_level0_col3" class="col_heading level0 col3" >LIFT_mean</th>
      <th id="T_a1c62_level0_col4" class="col_heading level0 col4" >corrected_LIFT_mean</th>
      <th id="T_a1c62_level0_col5" class="col_heading level0 col5" >ASR_mean</th>
      <th id="T_a1c62_level0_col6" class="col_heading level0 col6" >intersection_Z_score_mean</th>
      <th id="T_a1c62_level0_col7" class="col_heading level0 col7" >mark</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_a1c62_row0_col0" class="data row0 col0" >credit_x_util_bin</td>
      <td id="T_a1c62_row0_col1" class="data row0 col1" >1</td>
      <td id="T_a1c62_row0_col2" class="data row0 col2" >2</td>
      <td id="T_a1c62_row0_col3" class="data row0 col3" >0.1496</td>
      <td id="T_a1c62_row0_col4" class="data row0 col4" >0.3814</td>
      <td id="T_a1c62_row0_col5" class="data row0 col5" >-11.799</td>
      <td id="T_a1c62_row0_col6" class="data row0 col6" >-4.4999</td>
      <td id="T_a1c62_row0_col7" class="data row0 col7" >6.7965</td>
    </tr>
    <tr>
      <td id="T_a1c62_row1_col0" class="data row1 col0" >monthly_debt_x_mortgage_bin</td>
      <td id="T_a1c62_row1_col1" class="data row1 col1" >1</td>
      <td id="T_a1c62_row1_col2" class="data row1 col2" >2</td>
      <td id="T_a1c62_row1_col3" class="data row1 col3" >0.2177</td>
      <td id="T_a1c62_row1_col4" class="data row1 col4" >0.272</td>
      <td id="T_a1c62_row1_col5" class="data row1 col5" >-6.389</td>
      <td id="T_a1c62_row1_col6" class="data row1 col6" >-4.0067</td>
      <td id="T_a1c62_row1_col7" class="data row1 col7" >5.1102</td>
    </tr>
    <tr>
      <td id="T_a1c62_row2_col0" class="data row2 col0" >debt_x_monthly_debt_bin</td>
      <td id="T_a1c62_row2_col1" class="data row2 col1" >2</td>
      <td id="T_a1c62_row2_col2" class="data row2 col2" >4</td>
      <td id="T_a1c62_row2_col3" class="data row2 col3" >0.2857</td>
      <td id="T_a1c62_row2_col4" class="data row2 col4" >0.3742</td>
      <td id="T_a1c62_row2_col5" class="data row2 col5" >-9.6805</td>
      <td id="T_a1c62_row2_col6" class="data row2 col6" >-6.0931</td>
      <td id="T_a1c62_row2_col7" class="data row2 col7" >5.0507</td>
    </tr>
    <tr>
      <td id="T_a1c62_row3_col0" class="data row3 col0" >credit_pressure_index_x_mortgage_bin</td>
      <td id="T_a1c62_row3_col1" class="data row3 col1" >2</td>
      <td id="T_a1c62_row3_col2" class="data row3 col2" >2</td>
      <td id="T_a1c62_row3_col3" class="data row3 col3" >0.1789</td>
      <td id="T_a1c62_row3_col4" class="data row3 col4" >0.574</td>
      <td id="T_a1c62_row3_col5" class="data row3 col5" >-22.5493</td>
      <td id="T_a1c62_row3_col6" class="data row3 col6" >-4.88</td>
      <td id="T_a1c62_row3_col7" class="data row3 col7" >4.719</td>
    </tr>
    <tr>
      <td id="T_a1c62_row4_col0" class="data row4 col0" >monthly_debt_x_util_bin</td>
      <td id="T_a1c62_row4_col1" class="data row4 col1" >1</td>
      <td id="T_a1c62_row4_col2" class="data row4 col2" >2</td>
      <td id="T_a1c62_row4_col3" class="data row4 col3" >0.1338</td>
      <td id="T_a1c62_row4_col4" class="data row4 col4" >0.5509</td>
      <td id="T_a1c62_row4_col5" class="data row4 col5" >-16.5686</td>
      <td id="T_a1c62_row4_col6" class="data row4 col6" >-3.505</td>
      <td id="T_a1c62_row4_col7" class="data row4 col7" >4.2223</td>
    </tr>
    <tr>
      <td id="T_a1c62_row5_col0" class="data row5 col0" >credit_pressure_index_x_debt_bin</td>
      <td id="T_a1c62_row5_col1" class="data row5 col1" >2</td>
      <td id="T_a1c62_row5_col2" class="data row5 col2" >2</td>
      <td id="T_a1c62_row5_col3" class="data row5 col3" >0.1895</td>
      <td id="T_a1c62_row5_col4" class="data row5 col4" >0.5685</td>
      <td id="T_a1c62_row5_col5" class="data row5 col5" >-18.4695</td>
      <td id="T_a1c62_row5_col6" class="data row5 col6" >-4.5981</td>
      <td id="T_a1c62_row5_col7" class="data row5 col7" >4.1792</td>
    </tr>
    <tr>
      <td id="T_a1c62_row6_col0" class="data row6 col0" >credit_pressure_index_x_monthly_debt_bin</td>
      <td id="T_a1c62_row6_col1" class="data row6 col1" >3</td>
      <td id="T_a1c62_row6_col2" class="data row6 col2" >4</td>
      <td id="T_a1c62_row6_col3" class="data row6 col3" >0.304</td>
      <td id="T_a1c62_row6_col4" class="data row6 col4" >0.4937</td>
      <td id="T_a1c62_row6_col5" class="data row6 col5" >-13.5895</td>
      <td id="T_a1c62_row6_col6" class="data row6 col6" >-6.3463</td>
      <td id="T_a1c62_row6_col7" class="data row6 col7" >4.0522</td>
    </tr>
    <tr>
      <td id="T_a1c62_row7_col0" class="data row7 col0" >credit_x_util_bin</td>
      <td id="T_a1c62_row7_col1" class="data row7 col1" >1</td>
      <td id="T_a1c62_row7_col2" class="data row7 col2" >3</td>
      <td id="T_a1c62_row7_col3" class="data row7 col3" >0.2915</td>
      <td id="T_a1c62_row7_col4" class="data row7 col4" >0.4126</td>
      <td id="T_a1c62_row7_col5" class="data row7 col5" >-9.1023</td>
      <td id="T_a1c62_row7_col6" class="data row7 col6" >-5.3599</td>
      <td id="T_a1c62_row7_col7" class="data row7 col7" >4.0462</td>
    </tr>
    <tr>
      <td id="T_a1c62_row8_col0" class="data row8 col0" >credit_pressure_index_x_monthly_debt_bin</td>
      <td id="T_a1c62_row8_col1" class="data row8 col1" >2</td>
      <td id="T_a1c62_row8_col2" class="data row8 col2" >3</td>
      <td id="T_a1c62_row8_col3" class="data row8 col3" >0.1863</td>
      <td id="T_a1c62_row8_col4" class="data row8 col4" >0.5533</td>
      <td id="T_a1c62_row8_col5" class="data row8 col5" >-16.1564</td>
      <td id="T_a1c62_row8_col6" class="data row8 col6" >-4.1662</td>
      <td id="T_a1c62_row8_col7" class="data row8 col7" >3.9496</td>
    </tr>
    <tr>
      <td id="T_a1c62_row9_col0" class="data row9 col0" >age_x_monthly_debt_bin</td>
      <td id="T_a1c62_row9_col1" class="data row9 col1" >6</td>
      <td id="T_a1c62_row9_col2" class="data row9 col2" >1</td>
      <td id="T_a1c62_row9_col3" class="data row9 col3" >0.1975</td>
      <td id="T_a1c62_row9_col4" class="data row9 col4" >0.5642</td>
      <td id="T_a1c62_row9_col5" class="data row9 col5" >-16.3917</td>
      <td id="T_a1c62_row9_col6" class="data row9 col6" >-4.4449</td>
      <td id="T_a1c62_row9_col7" class="data row9 col7" >3.8729</td>
    </tr>
  </tbody>
</table>




```python
""" 列出值得添加的交互标记 """
# region
# 增强效应交互项
positive_interactions_df = pd.DataFrame({
    # 交互标记名称
    'signal_name': [
        # 参与交互的两个一维分箱都是只与逾期次数相关的，后续记为第一组
        'late_severity_mid_x_short_late_low_signal',
        'late_severity_high_x_short_late_low_signal',
        '30-59late_low_x_short_late_mid_high_signal',
        '90+late_mid_high_x_short_late_low_signal',
        # 参与交互的两个一维分箱一个只与逾期次数相关，另一个也涉及逾期次数，后续记为第二组
        '30-59late_low_x_credit_late_density_mid_high_signal',
        'age_high_x_credit_late_density_high_signal',
        'util_low_x_credit_late_density_high_signal',
        'mortgage_ratio_mid_x_credit_late_density_high_signal',
        'mortgage_mid_x_credit_late_density_high_signal',
        # 参与交互的两个一维分箱有且只有一个与逾期次数相关，后续记为第三组
        'dep_missing_x_60-89late_mid_signal',
        'age_high_x_late_severity_high_signal',
        'age_high_x_90+late_mid_signal',
        'util_low_mid_x_late_severity_high_signal',
        'util_mid_x_90+late_mid_signal',
        'age_high_x_60-89late_mid_signal',
        '30-59late_mid_x_credit_mid_signal',
        # 参与交互的两个一维分箱都不与逾期次数相关，后续记为第四组
        'debt_low_x_credit_pressure_mid_signal',
        'debt_mid_x_credit_pressure_high_signal',
        'util_high_x_credit_pressure_low_signal',
        'monthly_debt_low_x_credit_pressure_high_signal',
        'late_severity_high_x_credit_pressure_low_signal'
    ],
    # 参与交互的第一个一维分箱排名序号列名
    'binrank_colname1': [
        # 第一组
        'late_severity_score_binrank', 'late_severity_score_binrank', '30-59late_binrank',
        '90+late_binrank',
        # 第二组
        '30-59late_binrank', 'age_binrank', 'util_binrank',
        'mortgage_ratio_binrank', 'mortgage_binrank',
        # 第三组
        'dep_binrank', 'age_binrank', 'age_binrank', 
        'util_binrank', 'util_binrank', 'age_binrank', '30-59late_binrank',
        # 第四组
        'debt_binrank', 'debt_binrank',
        'util_binrank', 'monthly_debt_binrank',
        'late_severity_score_binrank'
    ],
    # 参与交互的第二个一维分箱排名序号列名
    'binrank_colname2': [
        # 第一组
        'short_late_binrank', 'short_late_binrank', 'short_late_binrank',
        'short_late_binrank',
        # 第二组
        'credit_late_density_binrank', 'credit_late_density_binrank', 'credit_late_density_binrank',
        'credit_late_density_binrank', 'credit_late_density_binrank',
        # 第三组
        '60-89late_binrank', 'late_severity_score_binrank', '90+late_binrank', 
        'late_severity_score_binrank','90+late_binrank', '60-89late_binrank', 'credit_binrank',
        # 第四组
        'credit_pressure_index_binrank', 'credit_pressure_index_binrank', 
        'credit_pressure_index_binrank', 'credit_pressure_index_binrank', 
        'credit_pressure_index_binrank'
    ],
    # 参与交互的第一个一维分箱的排名序号跨度
    'binrank1': [
        # 第一组
        '2_2', '4_4', '1_1',
        '3_4',
        # 第二组
        '1_1', '5_6', '1_1',
        '2_2', '2_2',
        # 第三组
        '-1_-1', '6_6', '6_6', 
        '1_2', '2_2', '6_6', '3_3',
        # 第四组
        '1_1', '2_2', 
        '5_5', '1_1',
        '4_4'
    ],
    # 参与交互的第二个一维分箱的排名序号跨度
    'binrank2': [
        # 第一组
        '1_1', '1_1', '2_4',
        '1_1',
        # 第二组
        '4_6', '6_6', '6_6',
        '6_6', '6_6',
        # 第三组
        '2_2', '4_4', '2_3', 
        '4_4', '2_2', '2_2', '2_2',
        # 第四组
        '3_4', '4_5',
        '1_2', '3_4',
        '1_2'
    ]
})# 削弱效应交互项
# 削弱效应交互项
negative_interactions_df = pd.DataFrame({
    # 交互标记名称
    'signal_name': [
        'credit_low_x_util_mid_signal',
        'monthly_debt_low_x_mortgage_ratio_mid_signal',
        'credit_pressure_mid_x_debt_mid_signal',
        'monthly_debt_high_x_debt_mid_signal',
        'credit_pressure_mid_x_mortgage_mid_signal',
        'monthly_debt_low_x_mortgage_mid_signal',
        'monthly_debt_low_x_age_high_signal',
        'monthly_debt_high_x_credit_pressure_mid_high_signal',
        'monthly_debt_mid_x_credit_pressure_mid_signal',
        'age_high_x_debt_low_signal'
    ],
    # 参与交互的第一个一维分箱排名序号列名
    'binrank_colname1': [
        'credit_binrank', 'monthly_debt_binrank', 'credit_pressure_index_binrank',
        'monthly_debt_binrank', 'credit_pressure_index_binrank', 'monthly_debt_binrank',
        'monthly_debt_binrank', 'monthly_debt_binrank', 'monthly_debt_binrank', 'age_binrank'
    ],
    # 参与交互的第二个一维分箱排名序号列名
    'binrank_colname2': [
        'util_binrank', 'mortgage_ratio_binrank', 'debt_binrank',
        'debt_binrank', 'mortgage_binrank', 'mortgage_binrank',
        'age_binrank', 'credit_pressure_index_binrank', 'credit_pressure_index_binrank', 'debt_binrank'
    ],
    # 参与交互的第一个一维分箱的排名序号跨度
    'binrank1': [
        '1_1', '1_1', '2_2',
        '4_4', '2_2', '1_1',
        '1_1', '4_4', '3_3', '6_6'
    ],
    # 参与交互的第二个一维分箱的排名序号跨度
    'binrank2': [
        '2_2', '2_2', '2_2',
        '2_2', '2_2', '2_2',
        '6_6', '3_3', '2_2', '1_1'
    ]
})
# 释放内存
cleanup_vars([
    'train_data', 'folds', 'train_idx_list',
    'valuable_intersections_df', 'valuable_intersections_df_list',
    'top_positive_intersections_df', 'top_negative_intersections_df',
    'colnames_not_binary', 'colname_pairs_not_binary',
    'div_pts', 'bins_2D',
    'beta30', 'beta60', 'beta90',
])
# endregion
```


```python
""" 筛选值得保留的标记、一维分箱与二维分箱 """
is_processing_here2 = True

if is_processing_here2:
# region
    binary_columns_test_df_list = []; bins_1D_test_df_list = []; bins_2D_test_df_list = []
    # 划分训练集、验证集，准备K=5折交叉验证
    random_seed = 1000
    folds = create_kfold(train_valid_data, random_seed, n_splits=5)
    for fold_idx in range(len(folds)):
        # 训练集、验证集拆分
        train_idx_list, valid_idx_list = folds[fold_idx]
        train_data = train_valid_data.iloc[train_idx_list].copy()
        valid_data = train_valid_data.iloc[valid_idx_list].copy()
        # endregion
        
        """ 统一数据清洗阶段 """
        # region
        # 截断家属数异常大值
        cap_dep_num(train_data); cap_dep_num(valid_data)
        # 删除年龄异常小值
        delete_small_age(train_data); delete_small_age(valid_data)
        # 添加信用额度使用率的若干标记
        add_util_flags(train_data); add_util_flags(valid_data)
        # 添加负债率的若干标记
        add_debt_flags(train_data); add_debt_flags(valid_data)
        # 添加三种逾期次数的加权求和，权重由逻辑回归模型训练
        beta30, beta60, beta90 = add_late_severity_score(train_data)
        add_late_severity_score(valid_data, [beta30, beta60, beta90])
        # 添加衍生指标
        add_derived_features(train_data); add_derived_features(valid_data)
        # 筛选出所有非二元指标
        colnames_not_binary = get_colnames_whether_binary(train_data, is_binary=False)
        # 对所有非布尔值类型的指标建立一维分箱
        for colname in colnames_not_binary:
            _, div_pts = add_bins_1D(train_data, colname)
            add_bins_1D(valid_data, colname, fixed_div_pts=div_pts)
        # 把一维分箱的具体数值转换为排名序号
        convert_bins_value_to_rank(train_data); convert_bins_value_to_rank(valid_data)
        # 添加交互标记
        add_intersection_signals(train_data, positive_interactions_df, negative_interactions_df)
        add_intersection_signals(valid_data, positive_interactions_df, negative_interactions_df)
        # 整理指标顺序
        train_data = reorder_columns(train_data); valid_data = reorder_columns(valid_data)
        # endregion
        
        """ 整理标记、一维分箱、二维分箱的评判指标 """
        # region
        # 找出所有二元列
        colnames_binary = get_colnames_whether_binary(train_data, is_binary=True)
        # 计算二元列的风险比率、IV
        binary_columns_test_df = test_binary_columns(train_data, colnames_binary)
        binary_columns_test_df_list.append(binary_columns_test_df)
        # 计算一维分箱的WOE、IV和PSI，做第一轮筛选
        bins_1D_test_df, iv_1D_dict = test_bins_1D(train_data, valid_data, colnames_not_binary)
        bins_1D_test_df_list.append(bins_1D_test_df)
        # 列出所有一维分箱对应指标的所有无序对
        colname_pairs_not_binary = []
        for i in range(len(colnames_not_binary) - 1):
                for j in range(i + 1, len(colnames_not_binary)):
                    colname_pairs_not_binary.append([colnames_not_binary[i], colnames_not_binary[j]])
        # 对两两配对的每对指标建立二维分箱
        for colname_pair in colname_pairs_not_binary:
            _, bins_2D = add_bins_2D(train_data, colname_pair)
            add_bins_2D(valid_data, colname_pair, fixed_bins_2D=bins_2D)
        # 计算二维分箱的WOE、IV、IV增量、IS、PSI，做第二轮筛选
        bins_2D_test_df = test_bins_2D(train_data, valid_data, colname_pairs_not_binary, iv_1D_dict)
        bins_2D_test_df_list.append(bins_2D_test_df)
        # endregion
```


```python
# 展示统一数据清洗完毕时的全体指标名称（不含只起到过渡作用的分箱相关变量）
if is_processing_here2:
    colname_general_df = display_current_colnames_general(origin_data, train_data)
    print_df(colname_general_df)
```


<style type="text/css">
</style>
<table id="T_7819c">
  <thead>
    <tr>
      <th id="T_7819c_level0_col0" class="col_heading level0 col0" >原始指标</th>
      <th id="T_7819c_level0_col1" class="col_heading level0 col1" >数据质量标记</th>
      <th id="T_7819c_level0_col2" class="col_heading level0 col2" >衍生指标</th>
      <th id="T_7819c_level0_col3" class="col_heading level0 col3" >用户行为标记</th>
      <th id="T_7819c_level0_col4" class="col_heading level0 col4" >交互标记</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_7819c_row0_col0" class="data row0 col0" >SeriousDlqin2yrs（目标变量）</td>
      <td id="T_7819c_row0_col1" class="data row0 col1" >blacklist_flag</td>
      <td id="T_7819c_row0_col2" class="data row0 col2" >credit_late_density</td>
      <td id="T_7819c_row0_col3" class="data row0 col3" >has_no_credit</td>
      <td id="T_7819c_row0_col4" class="data row0 col4" >30-59late_low_x_credit_late_density_mid_high_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row1_col0" class="data row1 col0" >Age</td>
      <td id="T_7819c_row1_col1" class="data row1 col1" >both_missing_flag</td>
      <td id="T_7819c_row1_col2" class="data row1 col2" >credit_pressure_index</td>
      <td id="T_7819c_row1_col3" class="data row1 col3" >has_serious_late</td>
      <td id="T_7819c_row1_col4" class="data row1 col4" >30-59late_low_x_short_late_mid_high_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row2_col0" class="data row2 col0" >DebtRatio</td>
      <td id="T_7819c_row2_col1" class="data row2 col1" >debt_anomaly_flag</td>
      <td id="T_7819c_row2_col2" class="data row2 col2" >free_cashflow_income</td>
      <td id="T_7819c_row2_col3" class="data row2 col3" >has_short_late</td>
      <td id="T_7819c_row2_col4" class="data row2 col4" >30-59late_mid_x_credit_mid_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row3_col0" class="data row3 col0" >MonthlyIncome</td>
      <td id="T_7819c_row3_col1" class="data row3 col1" >income_anomaly_flag</td>
      <td id="T_7819c_row3_col2" class="data row3 col2" >income_per_dep</td>
      <td id="T_7819c_row3_col3" class="data row3 col3" >has_short_late_but_no_credit</td>
      <td id="T_7819c_row3_col4" class="data row3 col4" >90+late_mid_high_x_short_late_low_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row4_col0" class="data row4 col0" >NumberOfDependents</td>
      <td id="T_7819c_row4_col1" class="data row4 col1" >single_missing_flag</td>
      <td id="T_7819c_row4_col2" class="data row4 col2" >late_severity_score</td>
      <td id="T_7819c_row4_col3" class="data row4 col3" >is_debt_high</td>
      <td id="T_7819c_row4_col4" class="data row4 col4" >age_high_x_60-89late_mid_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row5_col0" class="data row5 col0" >NumberOfOpenCreditLinesAndLoans</td>
      <td id="T_7819c_row5_col1" class="data row5 col1" >util_anomaly_flag</td>
      <td id="T_7819c_row5_col2" class="data row5 col2" >monthly_debt</td>
      <td id="T_7819c_row5_col3" class="data row5 col3" >is_debt_overlimit</td>
      <td id="T_7819c_row5_col4" class="data row5 col4" >age_high_x_90+late_mid_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row6_col0" class="data row6 col0" >NumberOfTime30-59DaysPastDueNotWorse</td>
      <td id="T_7819c_row6_col1" class="data row6 col1" ></td>
      <td id="T_7819c_row6_col2" class="data row6 col2" >mortgage_ratio</td>
      <td id="T_7819c_row6_col3" class="data row6 col3" >is_util_high</td>
      <td id="T_7819c_row6_col4" class="data row6 col4" >age_high_x_credit_late_density_high_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row7_col0" class="data row7 col0" >NumberOfTime60-89DaysPastDueNotWorse</td>
      <td id="T_7819c_row7_col1" class="data row7 col1" ></td>
      <td id="T_7819c_row7_col2" class="data row7 col2" >short_late</td>
      <td id="T_7819c_row7_col3" class="data row7 col3" >is_util_overlimit</td>
      <td id="T_7819c_row7_col4" class="data row7 col4" >age_high_x_debt_low_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row8_col0" class="data row8 col0" >NumberOfTimes90DaysLate</td>
      <td id="T_7819c_row8_col1" class="data row8 col1" ></td>
      <td id="T_7819c_row8_col2" class="data row8 col2" ></td>
      <td id="T_7819c_row8_col3" class="data row8 col3" ></td>
      <td id="T_7819c_row8_col4" class="data row8 col4" >age_high_x_late_severity_high_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row9_col0" class="data row9 col0" >NumberRealEstateLoansOrLines</td>
      <td id="T_7819c_row9_col1" class="data row9 col1" ></td>
      <td id="T_7819c_row9_col2" class="data row9 col2" ></td>
      <td id="T_7819c_row9_col3" class="data row9 col3" ></td>
      <td id="T_7819c_row9_col4" class="data row9 col4" >credit_low_x_util_mid_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row10_col0" class="data row10 col0" >RevolvingUtilizationOfUnsecuredLines</td>
      <td id="T_7819c_row10_col1" class="data row10 col1" ></td>
      <td id="T_7819c_row10_col2" class="data row10 col2" ></td>
      <td id="T_7819c_row10_col3" class="data row10 col3" ></td>
      <td id="T_7819c_row10_col4" class="data row10 col4" >credit_pressure_mid_x_debt_mid_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row11_col0" class="data row11 col0" ></td>
      <td id="T_7819c_row11_col1" class="data row11 col1" ></td>
      <td id="T_7819c_row11_col2" class="data row11 col2" ></td>
      <td id="T_7819c_row11_col3" class="data row11 col3" ></td>
      <td id="T_7819c_row11_col4" class="data row11 col4" >credit_pressure_mid_x_mortgage_mid_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row12_col0" class="data row12 col0" ></td>
      <td id="T_7819c_row12_col1" class="data row12 col1" ></td>
      <td id="T_7819c_row12_col2" class="data row12 col2" ></td>
      <td id="T_7819c_row12_col3" class="data row12 col3" ></td>
      <td id="T_7819c_row12_col4" class="data row12 col4" >debt_low_x_credit_pressure_mid_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row13_col0" class="data row13 col0" ></td>
      <td id="T_7819c_row13_col1" class="data row13 col1" ></td>
      <td id="T_7819c_row13_col2" class="data row13 col2" ></td>
      <td id="T_7819c_row13_col3" class="data row13 col3" ></td>
      <td id="T_7819c_row13_col4" class="data row13 col4" >debt_mid_x_credit_pressure_high_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row14_col0" class="data row14 col0" ></td>
      <td id="T_7819c_row14_col1" class="data row14 col1" ></td>
      <td id="T_7819c_row14_col2" class="data row14 col2" ></td>
      <td id="T_7819c_row14_col3" class="data row14 col3" ></td>
      <td id="T_7819c_row14_col4" class="data row14 col4" >dep_missing_x_60-89late_mid_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row15_col0" class="data row15 col0" ></td>
      <td id="T_7819c_row15_col1" class="data row15 col1" ></td>
      <td id="T_7819c_row15_col2" class="data row15 col2" ></td>
      <td id="T_7819c_row15_col3" class="data row15 col3" ></td>
      <td id="T_7819c_row15_col4" class="data row15 col4" >late_severity_high_x_credit_pressure_low_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row16_col0" class="data row16 col0" ></td>
      <td id="T_7819c_row16_col1" class="data row16 col1" ></td>
      <td id="T_7819c_row16_col2" class="data row16 col2" ></td>
      <td id="T_7819c_row16_col3" class="data row16 col3" ></td>
      <td id="T_7819c_row16_col4" class="data row16 col4" >late_severity_high_x_short_late_low_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row17_col0" class="data row17 col0" ></td>
      <td id="T_7819c_row17_col1" class="data row17 col1" ></td>
      <td id="T_7819c_row17_col2" class="data row17 col2" ></td>
      <td id="T_7819c_row17_col3" class="data row17 col3" ></td>
      <td id="T_7819c_row17_col4" class="data row17 col4" >late_severity_mid_x_short_late_low_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row18_col0" class="data row18 col0" ></td>
      <td id="T_7819c_row18_col1" class="data row18 col1" ></td>
      <td id="T_7819c_row18_col2" class="data row18 col2" ></td>
      <td id="T_7819c_row18_col3" class="data row18 col3" ></td>
      <td id="T_7819c_row18_col4" class="data row18 col4" >monthly_debt_high_x_credit_pressure_mid_high_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row19_col0" class="data row19 col0" ></td>
      <td id="T_7819c_row19_col1" class="data row19 col1" ></td>
      <td id="T_7819c_row19_col2" class="data row19 col2" ></td>
      <td id="T_7819c_row19_col3" class="data row19 col3" ></td>
      <td id="T_7819c_row19_col4" class="data row19 col4" >monthly_debt_high_x_debt_mid_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row20_col0" class="data row20 col0" ></td>
      <td id="T_7819c_row20_col1" class="data row20 col1" ></td>
      <td id="T_7819c_row20_col2" class="data row20 col2" ></td>
      <td id="T_7819c_row20_col3" class="data row20 col3" ></td>
      <td id="T_7819c_row20_col4" class="data row20 col4" >monthly_debt_low_x_age_high_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row21_col0" class="data row21 col0" ></td>
      <td id="T_7819c_row21_col1" class="data row21 col1" ></td>
      <td id="T_7819c_row21_col2" class="data row21 col2" ></td>
      <td id="T_7819c_row21_col3" class="data row21 col3" ></td>
      <td id="T_7819c_row21_col4" class="data row21 col4" >monthly_debt_low_x_credit_pressure_high_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row22_col0" class="data row22 col0" ></td>
      <td id="T_7819c_row22_col1" class="data row22 col1" ></td>
      <td id="T_7819c_row22_col2" class="data row22 col2" ></td>
      <td id="T_7819c_row22_col3" class="data row22 col3" ></td>
      <td id="T_7819c_row22_col4" class="data row22 col4" >monthly_debt_low_x_mortgage_mid_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row23_col0" class="data row23 col0" ></td>
      <td id="T_7819c_row23_col1" class="data row23 col1" ></td>
      <td id="T_7819c_row23_col2" class="data row23 col2" ></td>
      <td id="T_7819c_row23_col3" class="data row23 col3" ></td>
      <td id="T_7819c_row23_col4" class="data row23 col4" >monthly_debt_low_x_mortgage_ratio_mid_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row24_col0" class="data row24 col0" ></td>
      <td id="T_7819c_row24_col1" class="data row24 col1" ></td>
      <td id="T_7819c_row24_col2" class="data row24 col2" ></td>
      <td id="T_7819c_row24_col3" class="data row24 col3" ></td>
      <td id="T_7819c_row24_col4" class="data row24 col4" >monthly_debt_mid_x_credit_pressure_mid_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row25_col0" class="data row25 col0" ></td>
      <td id="T_7819c_row25_col1" class="data row25 col1" ></td>
      <td id="T_7819c_row25_col2" class="data row25 col2" ></td>
      <td id="T_7819c_row25_col3" class="data row25 col3" ></td>
      <td id="T_7819c_row25_col4" class="data row25 col4" >mortgage_mid_x_credit_late_density_high_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row26_col0" class="data row26 col0" ></td>
      <td id="T_7819c_row26_col1" class="data row26 col1" ></td>
      <td id="T_7819c_row26_col2" class="data row26 col2" ></td>
      <td id="T_7819c_row26_col3" class="data row26 col3" ></td>
      <td id="T_7819c_row26_col4" class="data row26 col4" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row27_col0" class="data row27 col0" ></td>
      <td id="T_7819c_row27_col1" class="data row27 col1" ></td>
      <td id="T_7819c_row27_col2" class="data row27 col2" ></td>
      <td id="T_7819c_row27_col3" class="data row27 col3" ></td>
      <td id="T_7819c_row27_col4" class="data row27 col4" >util_high_x_credit_pressure_low_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row28_col0" class="data row28 col0" ></td>
      <td id="T_7819c_row28_col1" class="data row28 col1" ></td>
      <td id="T_7819c_row28_col2" class="data row28 col2" ></td>
      <td id="T_7819c_row28_col3" class="data row28 col3" ></td>
      <td id="T_7819c_row28_col4" class="data row28 col4" >util_low_mid_x_late_severity_high_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row29_col0" class="data row29 col0" ></td>
      <td id="T_7819c_row29_col1" class="data row29 col1" ></td>
      <td id="T_7819c_row29_col2" class="data row29 col2" ></td>
      <td id="T_7819c_row29_col3" class="data row29 col3" ></td>
      <td id="T_7819c_row29_col4" class="data row29 col4" >util_low_x_credit_late_density_high_signal</td>
    </tr>
    <tr>
      <td id="T_7819c_row30_col0" class="data row30 col0" ></td>
      <td id="T_7819c_row30_col1" class="data row30 col1" ></td>
      <td id="T_7819c_row30_col2" class="data row30 col2" ></td>
      <td id="T_7819c_row30_col3" class="data row30 col3" ></td>
      <td id="T_7819c_row30_col4" class="data row30 col4" >util_mid_x_90+late_mid_signal</td>
    </tr>
  </tbody>
</table>




```python
# 筛选出可以投入训练的标记、一维分箱、二维分箱
if is_processing_here2:
    binary_columns_filter_df = filter_binary_columns(binary_columns_test_df_list)
    bins_1D_filter_df = filter_bins_1D(bins_1D_test_df_list)
    passed_bins_2D_df = filter_bins_2D(bins_2D_test_df_list)
    print_df(binary_columns_filter_df)
```


<style type="text/css">
</style>
<table id="T_31131">
  <thead>
    <tr>
      <th id="T_31131_level0_col0" class="col_heading level0 col0" >colname</th>
      <th id="T_31131_level0_col1" class="col_heading level0 col1" >count_marked_mean</th>
      <th id="T_31131_level0_col2" class="col_heading level0 col2" >count_normal_mean</th>
      <th id="T_31131_level0_col3" class="col_heading level0 col3" >RR_mean</th>
      <th id="T_31131_level0_col4" class="col_heading level0 col4" >IV_mean</th>
      <th id="T_31131_level0_col5" class="col_heading level0 col5" >is_passed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_31131_row0_col0" class="data row0 col0" >blacklist_flag</td>
      <td id="T_31131_row0_col1" class="data row0 col1" >177.6</td>
      <td id="T_31131_row0_col2" class="data row0 col2" >95820.8</td>
      <td id="T_31131_row0_col3" class="data row0 col3" >8.4113</td>
      <td id="T_31131_row0_col4" class="data row0 col4" >0.0415</td>
      <td id="T_31131_row0_col5" class="data row0 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row1_col0" class="data row1 col0" >both_missing_flag</td>
      <td id="T_31131_row1_col1" class="data row1 col1" >2513.6</td>
      <td id="T_31131_row1_col2" class="data row1 col2" >93484.8</td>
      <td id="T_31131_row1_col3" class="data row1 col3" >0.7234</td>
      <td id="T_31131_row1_col4" class="data row1 col4" >0.0026</td>
      <td id="T_31131_row1_col5" class="data row1 col5" >False</td>
    </tr>
    <tr>
      <td id="T_31131_row2_col0" class="data row2 col0" >debt_anomaly_flag</td>
      <td id="T_31131_row2_col1" class="data row2 col1" >199.2</td>
      <td id="T_31131_row2_col2" class="data row2 col2" >95799.2</td>
      <td id="T_31131_row2_col3" class="data row2 col3" >1.0167</td>
      <td id="T_31131_row2_col4" class="data row2 col4" >0.0001</td>
      <td id="T_31131_row2_col5" class="data row2 col5" >False</td>
    </tr>
    <tr>
      <td id="T_31131_row3_col0" class="data row3 col0" >income_anomaly_flag</td>
      <td id="T_31131_row3_col1" class="data row3 col1" >1454.4</td>
      <td id="T_31131_row3_col2" class="data row3 col2" >94544.0</td>
      <td id="T_31131_row3_col3" class="data row3 col3" >0.5396</td>
      <td id="T_31131_row3_col4" class="data row3 col4" >0.0049</td>
      <td id="T_31131_row3_col5" class="data row3 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row4_col0" class="data row4 col0" >single_missing_flag</td>
      <td id="T_31131_row4_col1" class="data row4 col1" >16641.6</td>
      <td id="T_31131_row4_col2" class="data row4 col2" >79356.8</td>
      <td id="T_31131_row4_col3" class="data row4 col3" >0.8373</td>
      <td id="T_31131_row4_col4" class="data row4 col4" >0.0049</td>
      <td id="T_31131_row4_col5" class="data row4 col5" >False</td>
    </tr>
    <tr>
      <td id="T_31131_row5_col0" class="data row5 col0" >util_anomaly_flag</td>
      <td id="T_31131_row5_col1" class="data row5 col1" >162.4</td>
      <td id="T_31131_row5_col2" class="data row5 col2" >95836.0</td>
      <td id="T_31131_row5_col3" class="data row5 col3" >1.2519</td>
      <td id="T_31131_row5_col4" class="data row5 col4" >0.0001</td>
      <td id="T_31131_row5_col5" class="data row5 col5" >False</td>
    </tr>
    <tr>
      <td id="T_31131_row6_col0" class="data row6 col0" >has_no_credit</td>
      <td id="T_31131_row6_col1" class="data row6 col1" >1200.8</td>
      <td id="T_31131_row6_col2" class="data row6 col2" >94797.6</td>
      <td id="T_31131_row6_col3" class="data row6 col3" >3.9909</td>
      <td id="T_31131_row6_col4" class="data row6 col4" >0.0617</td>
      <td id="T_31131_row6_col5" class="data row6 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row7_col0" class="data row7 col0" >has_serious_late</td>
      <td id="T_31131_row7_col1" class="data row7 col1" >5184.8</td>
      <td id="T_31131_row7_col2" class="data row7 col2" >90636.0</td>
      <td id="T_31131_row7_col3" class="data row7 col3" >8.9432</td>
      <td id="T_31131_row7_col4" class="data row7 col4" >0.8476</td>
      <td id="T_31131_row7_col5" class="data row7 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row8_col0" class="data row8 col0" >has_short_late</td>
      <td id="T_31131_row8_col1" class="data row8 col1" >17222.4</td>
      <td id="T_31131_row8_col2" class="data row8 col2" >78598.4</td>
      <td id="T_31131_row8_col3" class="data row8 col3" >6.2186</td>
      <td id="T_31131_row8_col4" class="data row8 col4" >0.8969</td>
      <td id="T_31131_row8_col5" class="data row8 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row9_col0" class="data row9 col0" >has_short_late_but_no_credit</td>
      <td id="T_31131_row9_col1" class="data row9 col1" >119.2</td>
      <td id="T_31131_row9_col2" class="data row9 col2" >95879.2</td>
      <td id="T_31131_row9_col3" class="data row9 col3" >4.6431</td>
      <td id="T_31131_row9_col4" class="data row9 col4" >0.0088</td>
      <td id="T_31131_row9_col5" class="data row9 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row10_col0" class="data row10 col0" >is_debt_high</td>
      <td id="T_31131_row10_col1" class="data row10 col1" >3336.8</td>
      <td id="T_31131_row10_col2" class="data row10 col2" >92661.6</td>
      <td id="T_31131_row10_col3" class="data row10 col3" >1.7559</td>
      <td id="T_31131_row10_col4" class="data row10 col4" >0.0164</td>
      <td id="T_31131_row10_col5" class="data row10 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row11_col0" class="data row11 col0" >is_debt_overlimit</td>
      <td id="T_31131_row11_col1" class="data row11 col1" >3145.6</td>
      <td id="T_31131_row11_col2" class="data row11 col2" >92852.8</td>
      <td id="T_31131_row11_col3" class="data row11 col3" >2.0506</td>
      <td id="T_31131_row11_col4" class="data row11 col4" >0.0274</td>
      <td id="T_31131_row11_col5" class="data row11 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row12_col0" class="data row12 col0" >is_util_high</td>
      <td id="T_31131_row12_col1" class="data row12 col1" >15491.2</td>
      <td id="T_31131_row12_col2" class="data row12 col2" >80507.2</td>
      <td id="T_31131_row12_col3" class="data row12 col3" >4.0585</td>
      <td id="T_31131_row12_col4" class="data row12 col4" >0.4619</td>
      <td id="T_31131_row12_col5" class="data row12 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row13_col0" class="data row13 col0" >is_util_overlimit</td>
      <td id="T_31131_row13_col1" class="data row13 col1" >1971.2</td>
      <td id="T_31131_row13_col2" class="data row13 col2" >94027.2</td>
      <td id="T_31131_row13_col3" class="data row13 col3" >6.6162</td>
      <td id="T_31131_row13_col4" class="data row13 col4" >0.2532</td>
      <td id="T_31131_row13_col5" class="data row13 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row14_col0" class="data row14 col0" >30-59late_low_x_credit_late_density_mid_high_signal</td>
      <td id="T_31131_row14_col1" class="data row14 col1" >1270.4</td>
      <td id="T_31131_row14_col2" class="data row14 col2" >94728.0</td>
      <td id="T_31131_row14_col3" class="data row14 col3" >5.0206</td>
      <td id="T_31131_row14_col4" class="data row14 col4" >0.1032</td>
      <td id="T_31131_row14_col5" class="data row14 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row15_col0" class="data row15 col0" >30-59late_low_x_short_late_mid_high_signal</td>
      <td id="T_31131_row15_col1" class="data row15 col1" >2020.0</td>
      <td id="T_31131_row15_col2" class="data row15 col2" >93978.4</td>
      <td id="T_31131_row15_col3" class="data row15 col3" >4.2475</td>
      <td id="T_31131_row15_col4" class="data row15 col4" >0.1135</td>
      <td id="T_31131_row15_col5" class="data row15 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row16_col0" class="data row16 col0" >30-59late_mid_x_credit_mid_signal</td>
      <td id="T_31131_row16_col1" class="data row16 col1" >445.6</td>
      <td id="T_31131_row16_col2" class="data row16 col2" >95552.8</td>
      <td id="T_31131_row16_col3" class="data row16 col3" >5.9072</td>
      <td id="T_31131_row16_col4" class="data row16 col4" >0.0522</td>
      <td id="T_31131_row16_col5" class="data row16 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row17_col0" class="data row17 col0" >90+late_mid_high_x_short_late_low_signal</td>
      <td id="T_31131_row17_col1" class="data row17 col1" >548.0</td>
      <td id="T_31131_row17_col2" class="data row17 col2" >95450.4</td>
      <td id="T_31131_row17_col3" class="data row17 col3" >6.8606</td>
      <td id="T_31131_row17_col4" class="data row17 col4" >0.0844</td>
      <td id="T_31131_row17_col5" class="data row17 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row18_col0" class="data row18 col0" >age_high_x_60-89late_mid_signal</td>
      <td id="T_31131_row18_col1" class="data row18 col1" >284.8</td>
      <td id="T_31131_row18_col2" class="data row18 col2" >95713.6</td>
      <td id="T_31131_row18_col3" class="data row18 col3" >3.5127</td>
      <td id="T_31131_row18_col4" class="data row18 col4" >0.0115</td>
      <td id="T_31131_row18_col5" class="data row18 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row19_col0" class="data row19 col0" >age_high_x_90+late_mid_signal</td>
      <td id="T_31131_row19_col1" class="data row19 col1" >308.0</td>
      <td id="T_31131_row19_col2" class="data row19 col2" >95690.4</td>
      <td id="T_31131_row19_col3" class="data row19 col3" >4.6365</td>
      <td id="T_31131_row19_col4" class="data row19 col4" >0.0226</td>
      <td id="T_31131_row19_col5" class="data row19 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row20_col0" class="data row20 col0" >age_high_x_credit_late_density_high_signal</td>
      <td id="T_31131_row20_col1" class="data row20 col1" >534.6</td>
      <td id="T_31131_row20_col2" class="data row20 col2" >95463.8</td>
      <td id="T_31131_row20_col3" class="data row20 col3" >5.5048</td>
      <td id="T_31131_row20_col4" class="data row20 col4" >0.0542</td>
      <td id="T_31131_row20_col5" class="data row20 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row21_col0" class="data row21 col0" >age_high_x_debt_low_signal</td>
      <td id="T_31131_row21_col1" class="data row21 col1" >4346.6</td>
      <td id="T_31131_row21_col2" class="data row21 col2" >91651.8</td>
      <td id="T_31131_row21_col3" class="data row21 col3" >0.1776</td>
      <td id="T_31131_row21_col4" class="data row21 col4" >0.0708</td>
      <td id="T_31131_row21_col5" class="data row21 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row22_col0" class="data row22 col0" >age_high_x_late_severity_high_signal</td>
      <td id="T_31131_row22_col1" class="data row22 col1" >222.2</td>
      <td id="T_31131_row22_col2" class="data row22 col2" >95776.2</td>
      <td id="T_31131_row22_col3" class="data row22 col3" >6.7465</td>
      <td id="T_31131_row22_col4" class="data row22 col4" >0.0341</td>
      <td id="T_31131_row22_col5" class="data row22 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row23_col0" class="data row23 col0" >credit_low_x_util_mid_signal</td>
      <td id="T_31131_row23_col1" class="data row23 col1" >1962.2</td>
      <td id="T_31131_row23_col2" class="data row23 col2" >94036.2</td>
      <td id="T_31131_row23_col3" class="data row23 col3" >0.1467</td>
      <td id="T_31131_row23_col4" class="data row23 col4" >0.0369</td>
      <td id="T_31131_row23_col5" class="data row23 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row24_col0" class="data row24 col0" >credit_pressure_mid_x_debt_mid_signal</td>
      <td id="T_31131_row24_col1" class="data row24 col1" >4749.6</td>
      <td id="T_31131_row24_col2" class="data row24 col2" >91248.8</td>
      <td id="T_31131_row24_col3" class="data row24 col3" >0.1893</td>
      <td id="T_31131_row24_col4" class="data row24 col4" >0.0734</td>
      <td id="T_31131_row24_col5" class="data row24 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row25_col0" class="data row25 col0" >credit_pressure_mid_x_mortgage_mid_signal</td>
      <td id="T_31131_row25_col1" class="data row25 col1" >6133.8</td>
      <td id="T_31131_row25_col2" class="data row25 col2" >89864.6</td>
      <td id="T_31131_row25_col3" class="data row25 col3" >0.17</td>
      <td id="T_31131_row25_col4" class="data row25 col4" >0.1031</td>
      <td id="T_31131_row25_col5" class="data row25 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row26_col0" class="data row26 col0" >debt_low_x_credit_pressure_mid_signal</td>
      <td id="T_31131_row26_col1" class="data row26 col1" >2437.6</td>
      <td id="T_31131_row26_col2" class="data row26 col2" >93560.8</td>
      <td id="T_31131_row26_col3" class="data row26 col3" >2.6743</td>
      <td id="T_31131_row26_col4" class="data row26 col4" >0.0471</td>
      <td id="T_31131_row26_col5" class="data row26 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row27_col0" class="data row27 col0" >debt_mid_x_credit_pressure_high_signal</td>
      <td id="T_31131_row27_col1" class="data row27 col1" >4018.8</td>
      <td id="T_31131_row27_col2" class="data row27 col2" >91979.6</td>
      <td id="T_31131_row27_col3" class="data row27 col3" >2.6549</td>
      <td id="T_31131_row27_col4" class="data row27 col4" >0.0728</td>
      <td id="T_31131_row27_col5" class="data row27 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row28_col0" class="data row28 col0" >dep_missing_x_60-89late_mid_signal</td>
      <td id="T_31131_row28_col1" class="data row28 col1" >54.4</td>
      <td id="T_31131_row28_col2" class="data row28 col2" >95944.0</td>
      <td id="T_31131_row28_col3" class="data row28 col3" >6.8434</td>
      <td id="T_31131_row28_col4" class="data row28 col4" >0.0087</td>
      <td id="T_31131_row28_col5" class="data row28 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row29_col0" class="data row29 col0" >late_severity_high_x_credit_pressure_low_signal</td>
      <td id="T_31131_row29_col1" class="data row29 col1" >402.4</td>
      <td id="T_31131_row29_col2" class="data row29 col2" >95596.0</td>
      <td id="T_31131_row29_col3" class="data row29 col3" >5.5065</td>
      <td id="T_31131_row29_col4" class="data row29 col4" >0.0413</td>
      <td id="T_31131_row29_col5" class="data row29 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row30_col0" class="data row30 col0" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_31131_row30_col1" class="data row30 col1" >483.0</td>
      <td id="T_31131_row30_col2" class="data row30 col2" >95515.4</td>
      <td id="T_31131_row30_col3" class="data row30 col3" >6.9236</td>
      <td id="T_31131_row30_col4" class="data row30 col4" >0.0756</td>
      <td id="T_31131_row30_col5" class="data row30 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row31_col0" class="data row31 col0" >late_severity_mid_x_short_late_low_signal</td>
      <td id="T_31131_row31_col1" class="data row31 col1" >1486.4</td>
      <td id="T_31131_row31_col2" class="data row31 col2" >94512.0</td>
      <td id="T_31131_row31_col3" class="data row31 col3" >3.7443</td>
      <td id="T_31131_row31_col4" class="data row31 col4" >0.0657</td>
      <td id="T_31131_row31_col5" class="data row31 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row32_col0" class="data row32 col0" >monthly_debt_high_x_credit_pressure_mid_high_signal</td>
      <td id="T_31131_row32_col1" class="data row32 col1" >3483.0</td>
      <td id="T_31131_row32_col2" class="data row32 col2" >92515.4</td>
      <td id="T_31131_row32_col3" class="data row32 col3" >0.3132</td>
      <td id="T_31131_row32_col4" class="data row32 col4" >0.0321</td>
      <td id="T_31131_row32_col5" class="data row32 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row33_col0" class="data row33 col0" >monthly_debt_high_x_debt_mid_signal</td>
      <td id="T_31131_row33_col1" class="data row33 col1" >1675.2</td>
      <td id="T_31131_row33_col2" class="data row33 col2" >94323.2</td>
      <td id="T_31131_row33_col3" class="data row33 col3" >0.2966</td>
      <td id="T_31131_row33_col4" class="data row33 col4" >0.0166</td>
      <td id="T_31131_row33_col5" class="data row33 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row34_col0" class="data row34 col0" >monthly_debt_low_x_age_high_signal</td>
      <td id="T_31131_row34_col1" class="data row34 col1" >4346.0</td>
      <td id="T_31131_row34_col2" class="data row34 col2" >91652.4</td>
      <td id="T_31131_row34_col3" class="data row34 col3" >0.1896</td>
      <td id="T_31131_row34_col4" class="data row34 col4" >0.0672</td>
      <td id="T_31131_row34_col5" class="data row34 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row35_col0" class="data row35 col0" >monthly_debt_low_x_credit_pressure_high_signal</td>
      <td id="T_31131_row35_col1" class="data row35 col1" >3031.6</td>
      <td id="T_31131_row35_col2" class="data row35 col2" >92966.8</td>
      <td id="T_31131_row35_col3" class="data row35 col3" >2.5194</td>
      <td id="T_31131_row35_col4" class="data row35 col4" >0.0491</td>
      <td id="T_31131_row35_col5" class="data row35 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row36_col0" class="data row36 col0" >monthly_debt_low_x_mortgage_mid_signal</td>
      <td id="T_31131_row36_col1" class="data row36 col1" >542.4</td>
      <td id="T_31131_row36_col2" class="data row36 col2" >95456.0</td>
      <td id="T_31131_row36_col3" class="data row36 col3" >0.2084</td>
      <td id="T_31131_row36_col4" class="data row36 col4" >0.0078</td>
      <td id="T_31131_row36_col5" class="data row36 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row37_col0" class="data row37 col0" >monthly_debt_low_x_mortgage_ratio_mid_signal</td>
      <td id="T_31131_row37_col1" class="data row37 col1" >167.6</td>
      <td id="T_31131_row37_col2" class="data row37 col2" >95830.8</td>
      <td id="T_31131_row37_col3" class="data row37 col3" >0.1075</td>
      <td id="T_31131_row37_col4" class="data row37 col4" >0.009</td>
      <td id="T_31131_row37_col5" class="data row37 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row38_col0" class="data row38 col0" >monthly_debt_mid_x_credit_pressure_mid_signal</td>
      <td id="T_31131_row38_col1" class="data row38 col1" >3604.6</td>
      <td id="T_31131_row38_col2" class="data row38 col2" >92393.8</td>
      <td id="T_31131_row38_col3" class="data row38 col3" >0.1867</td>
      <td id="T_31131_row38_col4" class="data row38 col4" >0.0564</td>
      <td id="T_31131_row38_col5" class="data row38 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row39_col0" class="data row39 col0" >mortgage_mid_x_credit_late_density_high_signal</td>
      <td id="T_31131_row39_col1" class="data row39 col1" >956.6</td>
      <td id="T_31131_row39_col2" class="data row39 col2" >95041.8</td>
      <td id="T_31131_row39_col3" class="data row39 col3" >6.9506</td>
      <td id="T_31131_row39_col4" class="data row39 col4" >0.1454</td>
      <td id="T_31131_row39_col5" class="data row39 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row40_col0" class="data row40 col0" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_31131_row40_col1" class="data row40 col1" >250.6</td>
      <td id="T_31131_row40_col2" class="data row40 col2" >95747.8</td>
      <td id="T_31131_row40_col3" class="data row40 col3" >7.3</td>
      <td id="T_31131_row40_col4" class="data row40 col4" >0.0445</td>
      <td id="T_31131_row40_col5" class="data row40 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row41_col0" class="data row41 col0" >util_high_x_credit_pressure_low_signal</td>
      <td id="T_31131_row41_col1" class="data row41 col1" >1301.8</td>
      <td id="T_31131_row41_col2" class="data row41 col2" >94696.6</td>
      <td id="T_31131_row41_col3" class="data row41 col3" >3.0648</td>
      <td id="T_31131_row41_col4" class="data row41 col4" >0.0367</td>
      <td id="T_31131_row41_col5" class="data row41 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row42_col0" class="data row42 col0" >util_low_mid_x_late_severity_high_signal</td>
      <td id="T_31131_row42_col1" class="data row42 col1" >335.0</td>
      <td id="T_31131_row42_col2" class="data row42 col2" >95663.4</td>
      <td id="T_31131_row42_col3" class="data row42 col3" >5.452</td>
      <td id="T_31131_row42_col4" class="data row42 col4" >0.0339</td>
      <td id="T_31131_row42_col5" class="data row42 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row43_col0" class="data row43 col0" >util_low_x_credit_late_density_high_signal</td>
      <td id="T_31131_row43_col1" class="data row43 col1" >215.6</td>
      <td id="T_31131_row43_col2" class="data row43 col2" >95782.8</td>
      <td id="T_31131_row43_col3" class="data row43 col3" >4.2841</td>
      <td id="T_31131_row43_col4" class="data row43 col4" >0.0135</td>
      <td id="T_31131_row43_col5" class="data row43 col5" >True</td>
    </tr>
    <tr>
      <td id="T_31131_row44_col0" class="data row44 col0" >util_mid_x_90+late_mid_signal</td>
      <td id="T_31131_row44_col1" class="data row44 col1" >181.4</td>
      <td id="T_31131_row44_col2" class="data row44 col2" >95817.0</td>
      <td id="T_31131_row44_col3" class="data row44 col3" >3.2114</td>
      <td id="T_31131_row44_col4" class="data row44 col4" >0.006</td>
      <td id="T_31131_row44_col5" class="data row44 col5" >True</td>
    </tr>
  </tbody>
</table>




```python
if is_processing_here2:
    print_df(bins_1D_filter_df)
```


<style type="text/css">
</style>
<table id="T_52f7c">
  <thead>
    <tr>
      <th id="T_52f7c_level0_col0" class="col_heading level0 col0" >binname_1D</th>
      <th id="T_52f7c_level0_col1" class="col_heading level0 col1" >IV_mean</th>
      <th id="T_52f7c_level0_col2" class="col_heading level0 col2" >PSI_upper</th>
      <th id="T_52f7c_level0_col3" class="col_heading level0 col3" >p_value_upper</th>
      <th id="T_52f7c_level0_col4" class="col_heading level0 col4" >is_passed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_52f7c_row0_col0" class="data row0 col0" >30-59late_bin</td>
      <td id="T_52f7c_row0_col1" class="data row0 col1" >0.763</td>
      <td id="T_52f7c_row0_col2" class="data row0 col2" >0.0005</td>
      <td id="T_52f7c_row0_col3" class="data row0 col3" ><0.001***</td>
      <td id="T_52f7c_row0_col4" class="data row0 col4" >True</td>
    </tr>
    <tr>
      <td id="T_52f7c_row1_col0" class="data row1 col0" >60-89late_bin</td>
      <td id="T_52f7c_row1_col1" class="data row1 col1" >0.6087</td>
      <td id="T_52f7c_row1_col2" class="data row1 col2" >0.0005</td>
      <td id="T_52f7c_row1_col3" class="data row1 col3" ><0.001***</td>
      <td id="T_52f7c_row1_col4" class="data row1 col4" >True</td>
    </tr>
    <tr>
      <td id="T_52f7c_row2_col0" class="data row2 col0" >90+late_bin</td>
      <td id="T_52f7c_row2_col1" class="data row2 col1" >0.8858</td>
      <td id="T_52f7c_row2_col2" class="data row2 col2" >0.0006</td>
      <td id="T_52f7c_row2_col3" class="data row2 col3" ><0.001***</td>
      <td id="T_52f7c_row2_col4" class="data row2 col4" >True</td>
    </tr>
    <tr>
      <td id="T_52f7c_row3_col0" class="data row3 col0" >age_bin</td>
      <td id="T_52f7c_row3_col1" class="data row3 col1" >0.2452</td>
      <td id="T_52f7c_row3_col2" class="data row3 col2" >0.0003</td>
      <td id="T_52f7c_row3_col3" class="data row3 col3" ><0.001***</td>
      <td id="T_52f7c_row3_col4" class="data row3 col4" >True</td>
    </tr>
    <tr>
      <td id="T_52f7c_row4_col0" class="data row4 col0" >credit_bin</td>
      <td id="T_52f7c_row4_col1" class="data row4 col1" >0.0647</td>
      <td id="T_52f7c_row4_col2" class="data row4 col2" >0.0006</td>
      <td id="T_52f7c_row4_col3" class="data row4 col3" ><0.001***</td>
      <td id="T_52f7c_row4_col4" class="data row4 col4" >True</td>
    </tr>
    <tr>
      <td id="T_52f7c_row5_col0" class="data row5 col0" >credit_late_density_bin</td>
      <td id="T_52f7c_row5_col1" class="data row5 col1" >1.0512</td>
      <td id="T_52f7c_row5_col2" class="data row5 col2" >0.0008</td>
      <td id="T_52f7c_row5_col3" class="data row5 col3" ><0.001***</td>
      <td id="T_52f7c_row5_col4" class="data row5 col4" >True</td>
    </tr>
    <tr>
      <td id="T_52f7c_row6_col0" class="data row6 col0" >credit_pressure_index_bin</td>
      <td id="T_52f7c_row6_col1" class="data row6 col1" >0.4506</td>
      <td id="T_52f7c_row6_col2" class="data row6 col2" >0.0006</td>
      <td id="T_52f7c_row6_col3" class="data row6 col3" ><0.001***</td>
      <td id="T_52f7c_row6_col4" class="data row6 col4" >True</td>
    </tr>
    <tr>
      <td id="T_52f7c_row7_col0" class="data row7 col0" >debt_bin</td>
      <td id="T_52f7c_row7_col1" class="data row7 col1" >0.0756</td>
      <td id="T_52f7c_row7_col2" class="data row7 col2" >0.0007</td>
      <td id="T_52f7c_row7_col3" class="data row7 col3" ><0.001***</td>
      <td id="T_52f7c_row7_col4" class="data row7 col4" >True</td>
    </tr>
    <tr>
      <td id="T_52f7c_row8_col0" class="data row8 col0" >dep_bin</td>
      <td id="T_52f7c_row8_col1" class="data row8 col1" >0.0338</td>
      <td id="T_52f7c_row8_col2" class="data row8 col2" >0.0006</td>
      <td id="T_52f7c_row8_col3" class="data row8 col3" ><0.001***</td>
      <td id="T_52f7c_row8_col4" class="data row8 col4" >False</td>
    </tr>
    <tr>
      <td id="T_52f7c_row9_col0" class="data row9 col0" >free_cashflow_income_bin</td>
      <td id="T_52f7c_row9_col1" class="data row9 col1" >0.1351</td>
      <td id="T_52f7c_row9_col2" class="data row9 col2" >0.0012</td>
      <td id="T_52f7c_row9_col3" class="data row9 col3" ><0.001***</td>
      <td id="T_52f7c_row9_col4" class="data row9 col4" >True</td>
    </tr>
    <tr>
      <td id="T_52f7c_row10_col0" class="data row10 col0" >income_bin</td>
      <td id="T_52f7c_row10_col1" class="data row10 col1" >0.0794</td>
      <td id="T_52f7c_row10_col2" class="data row10 col2" >0.0005</td>
      <td id="T_52f7c_row10_col3" class="data row10 col3" ><0.001***</td>
      <td id="T_52f7c_row10_col4" class="data row10 col4" >True</td>
    </tr>
    <tr>
      <td id="T_52f7c_row11_col0" class="data row11 col0" >income_per_dep_bin</td>
      <td id="T_52f7c_row11_col1" class="data row11 col1" >0.1065</td>
      <td id="T_52f7c_row11_col2" class="data row11 col2" >0.0005</td>
      <td id="T_52f7c_row11_col3" class="data row11 col3" ><0.001***</td>
      <td id="T_52f7c_row11_col4" class="data row11 col4" >True</td>
    </tr>
    <tr>
      <td id="T_52f7c_row12_col0" class="data row12 col0" >late_severity_score_bin</td>
      <td id="T_52f7c_row12_col1" class="data row12 col1" >1.4526</td>
      <td id="T_52f7c_row12_col2" class="data row12 col2" >0.0005</td>
      <td id="T_52f7c_row12_col3" class="data row12 col3" ><0.001***</td>
      <td id="T_52f7c_row12_col4" class="data row12 col4" >True</td>
    </tr>
    <tr>
      <td id="T_52f7c_row13_col0" class="data row13 col0" >monthly_debt_bin</td>
      <td id="T_52f7c_row13_col1" class="data row13 col1" >0.0277</td>
      <td id="T_52f7c_row13_col2" class="data row13 col2" >0.0008</td>
      <td id="T_52f7c_row13_col3" class="data row13 col3" ><0.001***</td>
      <td id="T_52f7c_row13_col4" class="data row13 col4" >False</td>
    </tr>
    <tr>
      <td id="T_52f7c_row14_col0" class="data row14 col0" >mortgage_bin</td>
      <td id="T_52f7c_row14_col1" class="data row14 col1" >0.0609</td>
      <td id="T_52f7c_row14_col2" class="data row14 col2" >0.0006</td>
      <td id="T_52f7c_row14_col3" class="data row14 col3" ><0.001***</td>
      <td id="T_52f7c_row14_col4" class="data row14 col4" >True</td>
    </tr>
    <tr>
      <td id="T_52f7c_row15_col0" class="data row15 col0" >mortgage_ratio_bin</td>
      <td id="T_52f7c_row15_col1" class="data row15 col1" >0.0306</td>
      <td id="T_52f7c_row15_col2" class="data row15 col2" >0.0002</td>
      <td id="T_52f7c_row15_col3" class="data row15 col3" ><0.001***</td>
      <td id="T_52f7c_row15_col4" class="data row15 col4" >False</td>
    </tr>
    <tr>
      <td id="T_52f7c_row16_col0" class="data row16 col0" >short_late_bin</td>
      <td id="T_52f7c_row16_col1" class="data row16 col1" >1.0411</td>
      <td id="T_52f7c_row16_col2" class="data row16 col2" >0.0005</td>
      <td id="T_52f7c_row16_col3" class="data row16 col3" ><0.001***</td>
      <td id="T_52f7c_row16_col4" class="data row16 col4" >True</td>
    </tr>
    <tr>
      <td id="T_52f7c_row17_col0" class="data row17 col0" >util_bin</td>
      <td id="T_52f7c_row17_col1" class="data row17 col1" >1.0578</td>
      <td id="T_52f7c_row17_col2" class="data row17 col2" >0.0004</td>
      <td id="T_52f7c_row17_col3" class="data row17 col3" ><0.001***</td>
      <td id="T_52f7c_row17_col4" class="data row17 col4" >True</td>
    </tr>
  </tbody>
</table>




```python
if is_processing_here2:
    print_df(passed_bins_2D_df)
```


<style type="text/css">
</style>
<table id="T_d3cf9">
  <thead>
    <tr>
      <th id="T_d3cf9_level0_col0" class="col_heading level0 col0" >binname_2D</th>
      <th id="T_d3cf9_level0_col1" class="col_heading level0 col1" >IV_mean</th>
      <th id="T_d3cf9_level0_col2" class="col_heading level0 col2" >delta_IV_mean</th>
      <th id="T_d3cf9_level0_col3" class="col_heading level0 col3" >IS_mean</th>
      <th id="T_d3cf9_level0_col4" class="col_heading level0 col4" >PSI_upper</th>
      <th id="T_d3cf9_level0_col5" class="col_heading level0 col5" >p_value_upper</th>
      <th id="T_d3cf9_level0_col6" class="col_heading level0 col6" >RR_mean</th>
      <th id="T_d3cf9_level0_col7" class="col_heading level0 col7" >RV_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_d3cf9_row0_col0" class="data row0 col0" >credit_x_credit_pressure_index_bin</td>
      <td id="T_d3cf9_row0_col1" class="data row0 col1" >0.6221</td>
      <td id="T_d3cf9_row0_col2" class="data row0 col2" >0.1716</td>
      <td id="T_d3cf9_row0_col3" class="data row0 col3" >1.2073</td>
      <td id="T_d3cf9_row0_col4" class="data row0 col4" >0.0027</td>
      <td id="T_d3cf9_row0_col5" class="data row0 col5" ><0.001***</td>
      <td id="T_d3cf9_row0_col6" class="data row0 col6" >0.1955</td>
      <td id="T_d3cf9_row0_col7" class="data row0 col7" >0.0026</td>
    </tr>
    <tr>
      <td id="T_d3cf9_row1_col0" class="data row1 col0" >credit_x_monthly_debt_bin</td>
      <td id="T_d3cf9_row1_col1" class="data row1 col1" >0.1405</td>
      <td id="T_d3cf9_row1_col2" class="data row1 col2" >0.0758</td>
      <td id="T_d3cf9_row1_col3" class="data row1 col3" >1.5203</td>
      <td id="T_d3cf9_row1_col4" class="data row1 col4" >0.0022</td>
      <td id="T_d3cf9_row1_col5" class="data row1 col5" ><0.001***</td>
      <td id="T_d3cf9_row1_col6" class="data row1 col6" >0.1109</td>
      <td id="T_d3cf9_row1_col7" class="data row1 col7" >0.0006</td>
    </tr>
    <tr>
      <td id="T_d3cf9_row2_col0" class="data row2 col0" >debt_x_credit_bin</td>
      <td id="T_d3cf9_row2_col1" class="data row2 col1" >0.2131</td>
      <td id="T_d3cf9_row2_col2" class="data row2 col2" >0.1376</td>
      <td id="T_d3cf9_row2_col3" class="data row2 col3" >1.5194</td>
      <td id="T_d3cf9_row2_col4" class="data row2 col4" >0.0034</td>
      <td id="T_d3cf9_row2_col5" class="data row2 col5" ><0.001***</td>
      <td id="T_d3cf9_row2_col6" class="data row2 col6" >0.1332</td>
      <td id="T_d3cf9_row2_col7" class="data row2 col7" >0.0009</td>
    </tr>
    <tr>
      <td id="T_d3cf9_row3_col0" class="data row3 col0" >debt_x_credit_pressure_index_bin</td>
      <td id="T_d3cf9_row3_col1" class="data row3 col1" >0.6606</td>
      <td id="T_d3cf9_row3_col2" class="data row3 col2" >0.2101</td>
      <td id="T_d3cf9_row3_col3" class="data row3 col3" >1.2557</td>
      <td id="T_d3cf9_row3_col4" class="data row3 col4" >0.0022</td>
      <td id="T_d3cf9_row3_col5" class="data row3 col5" ><0.001***</td>
      <td id="T_d3cf9_row3_col6" class="data row3 col6" >0.3175</td>
      <td id="T_d3cf9_row3_col7" class="data row3 col7" >0.0028</td>
    </tr>
    <tr>
      <td id="T_d3cf9_row4_col0" class="data row4 col0" >debt_x_monthly_debt_bin</td>
      <td id="T_d3cf9_row4_col1" class="data row4 col1" >0.1378</td>
      <td id="T_d3cf9_row4_col2" class="data row4 col2" >0.0622</td>
      <td id="T_d3cf9_row4_col3" class="data row4 col3" >1.335</td>
      <td id="T_d3cf9_row4_col4" class="data row4 col4" >0.0025</td>
      <td id="T_d3cf9_row4_col5" class="data row4 col5" ><0.001***</td>
      <td id="T_d3cf9_row4_col6" class="data row4 col6" >0.1178</td>
      <td id="T_d3cf9_row4_col7" class="data row4 col7" >0.0005</td>
    </tr>
    <tr>
      <td id="T_d3cf9_row5_col0" class="data row5 col0" >debt_x_mortgage_bin</td>
      <td id="T_d3cf9_row5_col1" class="data row5 col1" >0.2406</td>
      <td id="T_d3cf9_row5_col2" class="data row5 col2" >0.165</td>
      <td id="T_d3cf9_row5_col3" class="data row5 col3" >1.7629</td>
      <td id="T_d3cf9_row5_col4" class="data row5 col4" >0.0019</td>
      <td id="T_d3cf9_row5_col5" class="data row5 col5" ><0.001***</td>
      <td id="T_d3cf9_row5_col6" class="data row5 col6" >0.1384</td>
      <td id="T_d3cf9_row5_col7" class="data row5 col7" >0.0009</td>
    </tr>
    <tr>
      <td id="T_d3cf9_row6_col0" class="data row6 col0" >debt_x_mortgage_ratio_bin</td>
      <td id="T_d3cf9_row6_col1" class="data row6 col1" >0.1774</td>
      <td id="T_d3cf9_row6_col2" class="data row6 col2" >0.1018</td>
      <td id="T_d3cf9_row6_col3" class="data row6 col3" >1.6716</td>
      <td id="T_d3cf9_row6_col4" class="data row6 col4" >0.0019</td>
      <td id="T_d3cf9_row6_col5" class="data row6 col5" ><0.001***</td>
      <td id="T_d3cf9_row6_col6" class="data row6 col6" >0.1021</td>
      <td id="T_d3cf9_row6_col7" class="data row6 col7" >0.0006</td>
    </tr>
    <tr>
      <td id="T_d3cf9_row7_col0" class="data row7 col0" >dep_x_credit_bin</td>
      <td id="T_d3cf9_row7_col1" class="data row7 col1" >0.1218</td>
      <td id="T_d3cf9_row7_col2" class="data row7 col2" >0.057</td>
      <td id="T_d3cf9_row7_col3" class="data row7 col3" >1.2355</td>
      <td id="T_d3cf9_row7_col4" class="data row7 col4" >0.0024</td>
      <td id="T_d3cf9_row7_col5" class="data row7 col5" ><0.001***</td>
      <td id="T_d3cf9_row7_col6" class="data row7 col6" >0.1776</td>
      <td id="T_d3cf9_row7_col7" class="data row7 col7" >0.0006</td>
    </tr>
    <tr>
      <td id="T_d3cf9_row8_col0" class="data row8 col0" >dep_x_mortgage_bin</td>
      <td id="T_d3cf9_row8_col1" class="data row8 col1" >0.1157</td>
      <td id="T_d3cf9_row8_col2" class="data row8 col2" >0.0548</td>
      <td id="T_d3cf9_row8_col3" class="data row8 col3" >1.2209</td>
      <td id="T_d3cf9_row8_col4" class="data row8 col4" >0.0013</td>
      <td id="T_d3cf9_row8_col5" class="data row8 col5" ><0.001***</td>
      <td id="T_d3cf9_row8_col6" class="data row8 col6" >0.1067</td>
      <td id="T_d3cf9_row8_col7" class="data row8 col7" >0.0005</td>
    </tr>
    <tr>
      <td id="T_d3cf9_row9_col0" class="data row9 col0" >income_x_monthly_debt_bin</td>
      <td id="T_d3cf9_row9_col1" class="data row9 col1" >0.1413</td>
      <td id="T_d3cf9_row9_col2" class="data row9 col2" >0.0618</td>
      <td id="T_d3cf9_row9_col3" class="data row9 col3" >1.3203</td>
      <td id="T_d3cf9_row9_col4" class="data row9 col4" >0.0019</td>
      <td id="T_d3cf9_row9_col5" class="data row9 col5" ><0.001***</td>
      <td id="T_d3cf9_row9_col6" class="data row9 col6" >0.1215</td>
      <td id="T_d3cf9_row9_col7" class="data row9 col7" >0.0005</td>
    </tr>
    <tr>
      <td id="T_d3cf9_row10_col0" class="data row10 col0" >monthly_debt_x_credit_pressure_index_bin</td>
      <td id="T_d3cf9_row10_col1" class="data row10 col1" >0.6269</td>
      <td id="T_d3cf9_row10_col2" class="data row10 col2" >0.1763</td>
      <td id="T_d3cf9_row10_col3" class="data row10 col3" >1.3106</td>
      <td id="T_d3cf9_row10_col4" class="data row10 col4" >0.0019</td>
      <td id="T_d3cf9_row10_col5" class="data row10 col5" ><0.001***</td>
      <td id="T_d3cf9_row10_col6" class="data row10 col6" >0.2066</td>
      <td id="T_d3cf9_row10_col7" class="data row10 col7" >0.0027</td>
    </tr>
    <tr>
      <td id="T_d3cf9_row11_col0" class="data row11 col0" >mortgage_x_credit_pressure_index_bin</td>
      <td id="T_d3cf9_row11_col1" class="data row11 col1" >0.6475</td>
      <td id="T_d3cf9_row11_col2" class="data row11 col2" >0.1969</td>
      <td id="T_d3cf9_row11_col3" class="data row11 col3" >1.2659</td>
      <td id="T_d3cf9_row11_col4" class="data row11 col4" >0.002</td>
      <td id="T_d3cf9_row11_col5" class="data row11 col5" ><0.001***</td>
      <td id="T_d3cf9_row11_col6" class="data row11 col6" >0.2132</td>
      <td id="T_d3cf9_row11_col7" class="data row11 col7" >0.0027</td>
    </tr>
    <tr>
      <td id="T_d3cf9_row12_col0" class="data row12 col0" >mortgage_x_monthly_debt_bin</td>
      <td id="T_d3cf9_row12_col1" class="data row12 col1" >0.1374</td>
      <td id="T_d3cf9_row12_col2" class="data row12 col2" >0.0765</td>
      <td id="T_d3cf9_row12_col3" class="data row12 col3" >1.5504</td>
      <td id="T_d3cf9_row12_col4" class="data row12 col4" >0.0016</td>
      <td id="T_d3cf9_row12_col5" class="data row12 col5" ><0.001***</td>
      <td id="T_d3cf9_row12_col6" class="data row12 col6" >0.1224</td>
      <td id="T_d3cf9_row12_col7" class="data row12 col7" >0.0005</td>
    </tr>
  </tbody>
</table>




```python
""" 列出通过筛选的标记、一维分箱、二维分箱 """
# region
# 交互标记的相关数据表
interaction_signals_test_df = pd.DataFrame({
    'signal colname': [
        '30-59late_low_x_credit_late_density_mid_high_signal',
        '30-59late_low_x_short_late_mid_high_signal',
        '30-59late_mid_x_credit_mid_signal',
        '90+late_mid_high_x_short_late_low_signal',
        'age_high_x_60-89late_mid_signal',
        'age_high_x_90+late_mid_signal',
        'age_high_x_credit_late_density_high_signal',
        'age_high_x_debt_low_signal',
        'age_high_x_late_severity_high_signal',
        'credit_low_x_util_mid_signal',
        'credit_pressure_mid_x_debt_mid_signal',
        'credit_pressure_mid_x_mortgage_mid_signal',
        'debt_low_x_credit_pressure_mid_signal',
        'debt_mid_x_credit_pressure_high_signal',
        'dep_missing_x_60-89late_mid_signal',
        'late_severity_high_x_credit_pressure_low_signal',
        'late_severity_high_x_short_late_low_signal',
        'late_severity_mid_x_short_late_low_signal',
        'monthly_debt_high_x_credit_pressure_mid_high_signal',
        'monthly_debt_high_x_debt_mid_signal',
        'monthly_debt_low_x_age_high_signal',
        'monthly_debt_low_x_credit_pressure_high_signal',
        'monthly_debt_low_x_mortgage_mid_signal',
        'monthly_debt_low_x_mortgage_ratio_mid_signal',
        'monthly_debt_mid_x_credit_pressure_mid_signal',
        'mortgage_mid_x_credit_late_density_high_signal',
        'mortgage_ratio_mid_x_credit_late_density_high_signal',
        'util_high_x_credit_pressure_low_signal',
        'util_low_mid_x_late_severity_high_signal',
        'util_low_x_credit_late_density_high_signal',
        'util_mid_x_90+late_mid_signal'
    ],
    'count marked': [
        1594.4, 2559.4, 549.6, 670.4, 356.0, 366.4, 714.4, 5405.8, 278.4,
        2404.8, 5945.2, 7693.0, 3049.0, 5038.8, 67.2, 518.6, 670.4, 1865.6,
        4325.2, 2079.8, 5407.8, 3844.4, 699.6, 204.4, 4502.8, 1290.4, 332.0,
        1636.2, 408.0, 287.4, 232.0
    ],
    'count normal': [
        118403.2, 117438.2, 119448.0, 119327.2, 119641.6, 119631.2, 119283.2, 114591.8, 119719.2,
        117592.8, 114052.4, 112304.6, 116948.6, 114958.8, 119930.4, 119479.0, 119327.2, 118132.0,
        115672.4, 117917.8, 114589.8, 116153.2, 119298.0, 119793.2, 115494.8, 118707.2, 119665.6,
        118361.4, 119589.6, 119710.2, 119765.6
    ],
    'RR': [
        5.097823, 4.286049, 5.877717, 6.705545, 3.454014, 4.489173, 5.541378, 0.17482, 6.485335,
        0.120923, 0.186938, 0.177958, 2.58384, 2.624934, 6.257462, 5.650464, 6.705545, 3.675369,
        0.316487, 0.314162, 0.191971, 2.507337, 0.255186, 0.117621, 0.202404, 6.714629, 6.963397,
        3.080146, 5.38985, 4.051518, 3.303112
    ],
    'IV': [
        0.106663, 0.116951, 0.051023, 0.079228, 0.01106, 0.020132, 0.058689, 0.071377, 0.031728,
        0.041168, 0.074248, 0.099769, 0.043014, 0.070817, 0.007285, 0.044717, 0.079228, 0.063386,
        0.031333, 0.015315, 0.066196, 0.049119, 0.006663, 0.003679, 0.05276, 0.146641, 0.043129,
        0.03737, 0.03227, 0.012746, 0.006525
    ]
})
# 通过筛选的标记
passed_binary_colnames = [
    # 数据质量标记
    'blacklist_flag',
    'income_anomaly_flag',
    # 用户行为标记
    'has_no_credit',
    'has_serious_late',
    'has_short_late',
    'has_short_late_but_no_credit',
    'is_debt_high',
    'is_debt_overlimit',
    'is_util_high',
    'is_util_overlimit',
    # 交互标记
    '30-59late_low_x_credit_late_density_mid_high_signal',
    '30-59late_low_x_short_late_mid_high_signal',
    '30-59late_mid_x_credit_mid_signal',
    '90+late_mid_high_x_short_late_low_signal',
    'age_high_x_60-89late_mid_signal',
    'age_high_x_90+late_mid_signal',
    'age_high_x_credit_late_density_high_signal',
    'age_high_x_debt_low_signal',
    'age_high_x_late_severity_high_signal',
    'credit_low_x_util_mid_signal',
    'credit_pressure_mid_x_debt_mid_signal',
    'credit_pressure_mid_x_mortgage_mid_signal',
    'debt_low_x_credit_pressure_mid_signal',
    'debt_mid_x_credit_pressure_high_signal',
    'dep_missing_x_60-89late_mid_signal',
    'late_severity_high_x_credit_pressure_low_signal',
    'late_severity_high_x_short_late_low_signal',
    'late_severity_mid_x_short_late_low_signal',
    'monthly_debt_high_x_credit_pressure_mid_high_signal',
    'monthly_debt_high_x_debt_mid_signal',
    'monthly_debt_low_x_age_high_signal',
    'monthly_debt_low_x_credit_pressure_high_signal',
    'monthly_debt_low_x_mortgage_mid_signal',
    'monthly_debt_low_x_mortgage_ratio_mid_signal',
    'monthly_debt_mid_x_credit_pressure_mid_signal',
    'mortgage_mid_x_credit_late_density_high_signal',
    'mortgage_ratio_mid_x_credit_late_density_high_signal',
    'util_high_x_credit_pressure_low_signal',
    'util_low_mid_x_late_severity_high_signal',
    'util_low_x_credit_late_density_high_signal',
    'util_mid_x_90+late_mid_signal'
]
# 通过筛选的一维分箱
passed_binnames_1D = [
    '30-59late_bin',
    '60-89late_bin',
    '90+late_bin',
    'age_bin',
    'credit_bin',
    'credit_late_density_bin',
    'credit_pressure_index_bin',
    'debt_bin',
    'free_cashflow_income_bin',
    'income_bin',
    'income_per_dep_bin',
    'late_severity_score_bin',
    'mortgage_bin',
    'short_late_bin',
    'util_bin'
]
# 因IV低而未通过筛选的一维分箱
low_iv_binnames_1D = [
    'dep_bin', 'monthly_debt_bin', 'mortgage_ratio_bin'
]
# 通过筛选的二维分箱
passed_binnames_2D = [
    'credit_x_credit_pressure_index_bin',
    'dep_x_credit_bin',
    'credit_x_monthly_debt_bin',
    'debt_x_credit_bin',
    'debt_x_credit_pressure_index_bin',
    'debt_x_monthly_debt_bin',
    'debt_x_mortgage_bin',
    'debt_x_mortgage_ratio_bin',
    'income_x_monthly_debt_bin',
    'monthly_debt_x_credit_pressure_index_bin',
    'mortgage_x_credit_pressure_index_bin',
    'dep_x_mortgage_bin',
    'mortgage_x_monthly_debt_bin'
]
# 释放内存
cleanup_vars([
    'train_data', 'valid_data', 'folds',
    'train_idx_list', 'valid_idx_list',
    'train_data_woe_lr',
    'binary_columns_test_df_list', 'bins_1D_test_df_list', 'bins_2D_test_df_list',
    'binary_columns_test_df', 'bins_1D_test_df', 'bins_2D_test_df',
    'binary_columns_filter_df', 'bins_1D_filter_df', 'passed_bins_2D_df',
    'colnames_binary', 'colnames_not_binary', 'colname_pairs_not_binary',
    'iv_1D_dict', 'div_pts', 'bins_2D',
    'beta30', 'beta60', 'beta90',
])
# endregion
```


```python
""" 对raw LR模型的调试 """
is_processing_here3 = True

if is_processing_here3:
    # region
    raw_lr_fit_goodness_df_list = []
    # 划分训练集、验证集，准备K=5折交叉验证
    random_seed = 200
    folds = create_kfold(train_valid_data, random_seed, n_splits=5)
    for fold_idx in range(len(folds)):
        # 训练集、验证集拆分
        train_idx_list, valid_idx_list = folds[fold_idx]
        train_data = train_valid_data.iloc[train_idx_list].copy()
        valid_data = train_valid_data.iloc[valid_idx_list].copy()
        train_data_copy = train_data.copy(); valid_data_copy = valid_data.copy() # 备份
        # endregion
        
        """ 统一数据清洗阶段 """
        # region
        # 截断家属数异常大值
        cap_dep_num(train_data); cap_dep_num(valid_data)
        # 删除年龄异常小值
        delete_small_age(train_data); delete_small_age(valid_data)
        # 添加信用额度使用率的若干标记
        add_util_flags(train_data); add_util_flags(valid_data)
        # 添加负债率的若干标记
        add_debt_flags(train_data); add_debt_flags(valid_data)
        # 添加三种逾期次数的加权求和，权重由逻辑回归模型训练
        beta30, beta60, beta90 = add_late_severity_score(train_data)
        add_late_severity_score(valid_data, [beta30, beta60, beta90])
        # 添加衍生指标
        add_derived_features(train_data); add_derived_features(valid_data)
        # 筛选出所有非二元指标
        colnames_not_binary = get_colnames_whether_binary(train_data, is_binary=False)
        # 对所有非布尔值类型的指标建立一维分箱
        for colname in colnames_not_binary:
            _, div_pts = add_bins_1D(train_data, colname)
            add_bins_1D(valid_data, colname, fixed_div_pts=div_pts)
        # 把一维分箱的具体数值转换为排名序号
        convert_bins_value_to_rank(train_data); convert_bins_value_to_rank(valid_data)
        # 添加交互标记
        add_intersection_signals(train_data, positive_interactions_df, negative_interactions_df)
        add_intersection_signals(valid_data, positive_interactions_df, negative_interactions_df)
        # 暂时删除一维分箱列以及其对应的排名序号列
        colnames_without_bins = [colname for colname in train_data.columns if '_bin' not in colname]
        train_data = train_data[colnames_without_bins]; valid_data = valid_data[colnames_without_bins]
        # 整理指标顺序
        train_data = reorder_columns(train_data); valid_data = reorder_columns(valid_data)
        # endregion
        
        """ 对raw LR建立特化表达 """
        # region
        train_data_raw_lr = train_data.copy(); valid_data_raw_lr = valid_data.copy()
        # 缺失值、异常值全部赋值为0
        train_data_raw_lr = train_data_raw_lr.clip(lower=0)
        valid_data_raw_lr = valid_data_raw_lr.clip(lower=0)
        # 添加年龄、负债率、信用额度使用率的中心化三列以及中心化的年龄*负债率、年龄*信用额度使用率两列
        centering_means = fit_centering_means(train_data_raw_lr)
        apply_centered_features(train_data_raw_lr, centering_means)
        apply_centered_features(valid_data_raw_lr, centering_means)
        # 对有长尾的指标取对数
        colnames_to_log = fit_log_feature_colnames(train_data_raw_lr)
        add_log_features(train_data_raw_lr, fixed_colnames=colnames_to_log)
        add_log_features(valid_data_raw_lr, fixed_colnames=colnames_to_log)
        # 筛选将要投入训练的列名
        selected_colnames = select_colnames_raw_lr(train_data_raw_lr)
        train_data_raw_lr = train_data_raw_lr[selected_colnames]
        valid_data_raw_lr = valid_data_raw_lr[selected_colnames]
        # endregion
        
        """ 训练raw LR """
        # region
        # 拆分目标变量和自变量
        y_train = train_data[target]; y_valid = valid_data[target]
        X_train_raw_lr = train_data_raw_lr.drop(target, axis=1, inplace=False)
        X_valid_raw_lr = valid_data_raw_lr.drop(target, axis=1, inplace=False)
        # 以下是可能尝试参与拟合的多种指标序列
        # 原始指标
        original_features = [
            'age_centered', 'util_centered', 'debt_centered',
            'income_log', dep, credit, mortgage,
            late30, late60, late90
        ]
        # 数据质量标记
        quality_flags = [
            'single_missing_flag', 'both_missing_flag', 'blacklist_flag',
            'income_anomaly_flag', 'debt_anomaly_flag', 'util_anomaly_flag'
        ]
        # 衍生指标
        derived_features = [
            'free_cashflow_income_log', 'income_per_dep_log', 'monthly_debt_log',
            'credit_pressure_index', 'mortgage_ratio', 'credit_late_density',
            'short_late', 'late_severity_score'
        ]
        # 用户行为标记
        risk_signals = [
            'has_no_credit', 'has_short_late_but_no_credit', 'has_serious_late', 'has_short_late',
            'is_debt_high', 'is_debt_overlimit', 'is_util_high', 'is_util_overlimit'
        ]
        # 削弱效应第1-5名
        negative_1_5 = [
            'monthly_debt_low_x_mortgage_ratio_mid_signal',
            'credit_low_x_util_mid_signal',
            'credit_pressure_mid_x_mortgage_mid_signal',
            'credit_pressure_mid_x_debt_mid_signal',
            'monthly_debt_low_x_age_high_signal'
        ]
        # 增强效应第1-5名
        positive_1_5 = [
            'mortgage_ratio_mid_x_credit_late_density_high_signal',
            'mortgage_mid_x_credit_late_density_high_signal',
            '90+late_mid_high_x_short_late_low_signal',
            'late_severity_high_x_short_late_low_signal',
            'age_high_x_late_severity_high_signal'
        ]
        # 增强效应第6-10名
        positive_6_10 = [
            'dep_missing_x_60-89late_mid_signal',
            '30-59late_mid_x_credit_mid_signal',
            'late_severity_high_x_credit_pressure_low_signal',
            'age_high_x_credit_late_density_high_signal',
            'util_low_mid_x_late_severity_high_signal'
        ]
        # 增强效应第11-15名
        positive_11_15 = [
            '30-59late_low_x_credit_late_density_mid_high_signal',
            '30-59late_low_x_short_late_mid_high_signal',
            'age_high_x_90+late_mid_signal',
            'util_low_x_credit_late_density_high_signal',
            'late_severity_mid_x_short_late_low_signal'
        ]
        # 以下是不同的进行拟合的指标组合
        colnames_to_fit0 = original_features + quality_flags
        colnames_to_fit1 = colnames_to_fit0 + derived_features + risk_signals
        colnames_to_fit2 = colnames_to_fit1 + positive_1_5
        colnames_to_fit3 = colnames_to_fit1 + negative_1_5
        colnames_to_fit4 = colnames_to_fit2 + positive_6_10
        colnames_to_fit5 = colnames_to_fit4 + positive_11_15
        colnames_to_fit_list = [
            colnames_to_fit0, colnames_to_fit1, colnames_to_fit2,
            colnames_to_fit3, colnames_to_fit4, colnames_to_fit5
        ]
        # 正则化强度的倒数（参数C）的尝试列表
        c_list = [0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2]
        # 构建raw LR拟合效果的数据表
        raw_lr_fit_goodness_df = grid_search_lr\
            (X_train_raw_lr, y_train, X_valid_raw_lr, y_valid, colnames_to_fit_list, c_list)
        raw_lr_fit_goodness_df_list.append(raw_lr_fit_goodness_df)
        # 确定最佳的拟合指标序号与参数C
        colnames_to_fit = colnames_to_fit1
        X_train_raw_lr = X_train_raw_lr[colnames_to_fit]; X_valid_raw_lr = X_valid_raw_lr[colnames_to_fit]
        c = 1
        # 以下是尝试舍弃以改善VIF的指标序列
        colnames_to_drop0 = []
        colnames_to_drop1 = [late30, late60, 'income_log', 'free_cashflow_income_log']
        colnames_to_drop2 = colnames_to_drop1 + [late90]
        colnames_to_drop3 = colnames_to_drop1 + ['monthly_debt_log']
        colnames_to_drop4 = colnames_to_drop1 + [late90, 'monthly_debt_log']
        colnames_to_drop_list = [
            colnames_to_drop0, colnames_to_drop1, colnames_to_drop2, colnames_to_drop3, colnames_to_drop4
        ]
        # endregion

```


```python
# 展示对raw LR建立特化表达完毕后的即将投入训练的列名
if is_processing_here3:
    colname_raw_lr_df = display_current_colnames_raw_lr(train_valid_data, train_data, train_data_raw_lr)
    print_df(colname_raw_lr_df)
```


<style type="text/css">
</style>
<table id="T_40efc">
  <thead>
    <tr>
      <th id="T_40efc_level0_col0" class="col_heading level0 col0" >raw LR中新增指标</th>
      <th id="T_40efc_level0_col1" class="col_heading level0 col1" >raw LR中删去原始指标</th>
      <th id="T_40efc_level0_col2" class="col_heading level0 col2" >raw LR中删去衍生指标</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_40efc_row0_col0" class="data row0 col0" >income_per_dep_log</td>
      <td id="T_40efc_row0_col1" class="data row0 col1" >RevolvingUtilizationOfUnsecuredLines</td>
      <td id="T_40efc_row0_col2" class="data row0 col2" >income_per_dep</td>
    </tr>
    <tr>
      <td id="T_40efc_row1_col0" class="data row1 col0" >free_cashflow_income_log</td>
      <td id="T_40efc_row1_col1" class="data row1 col1" >Age</td>
      <td id="T_40efc_row1_col2" class="data row1 col2" >monthly_debt</td>
    </tr>
    <tr>
      <td id="T_40efc_row2_col0" class="data row2 col0" >income_log</td>
      <td id="T_40efc_row2_col1" class="data row2 col1" >DebtRatio</td>
      <td id="T_40efc_row2_col2" class="data row2 col2" >free_cashflow_income</td>
    </tr>
    <tr>
      <td id="T_40efc_row3_col0" class="data row3 col0" >age_centered_*_debt_centered</td>
      <td id="T_40efc_row3_col1" class="data row3 col1" >MonthlyIncome</td>
      <td id="T_40efc_row3_col2" class="data row3 col2" ></td>
    </tr>
    <tr>
      <td id="T_40efc_row4_col0" class="data row4 col0" >util_centered</td>
      <td id="T_40efc_row4_col1" class="data row4 col1" ></td>
      <td id="T_40efc_row4_col2" class="data row4 col2" ></td>
    </tr>
    <tr>
      <td id="T_40efc_row5_col0" class="data row5 col0" >age_centered_*_util_centered</td>
      <td id="T_40efc_row5_col1" class="data row5 col1" ></td>
      <td id="T_40efc_row5_col2" class="data row5 col2" ></td>
    </tr>
    <tr>
      <td id="T_40efc_row6_col0" class="data row6 col0" >monthly_debt_log</td>
      <td id="T_40efc_row6_col1" class="data row6 col1" ></td>
      <td id="T_40efc_row6_col2" class="data row6 col2" ></td>
    </tr>
    <tr>
      <td id="T_40efc_row7_col0" class="data row7 col0" >debt_centered</td>
      <td id="T_40efc_row7_col1" class="data row7 col1" ></td>
      <td id="T_40efc_row7_col2" class="data row7 col2" ></td>
    </tr>
    <tr>
      <td id="T_40efc_row8_col0" class="data row8 col0" >age_centered</td>
      <td id="T_40efc_row8_col1" class="data row8 col1" ></td>
      <td id="T_40efc_row8_col2" class="data row8 col2" ></td>
    </tr>
  </tbody>
</table>




```python
# 网格搜索找出最合适的拟合指标方案和正则化强度的倒数C
if is_processing_here3:
    raw_lr_auc_ks_df, raw_lr_coef_sign_stability_df = test_fit_goodness_lr(raw_lr_fit_goodness_df_list)
    print_df(raw_lr_auc_ks_df)
```


<style type="text/css">
</style>
<table id="T_c5d8c">
  <thead>
    <tr>
      <th id="T_c5d8c_level0_col0" class="col_heading level0 col0" >fit columns index</th>
      <th id="T_c5d8c_level0_col1" class="col_heading level0 col1" >C value</th>
      <th id="T_c5d8c_level0_col2" class="col_heading level0 col2" >valid_AUC_mean</th>
      <th id="T_c5d8c_level0_col3" class="col_heading level0 col3" >AUC_gap_mean</th>
      <th id="T_c5d8c_level0_col4" class="col_heading level0 col4" >valid_KS_mean</th>
      <th id="T_c5d8c_level0_col5" class="col_heading level0 col5" >KS_gap_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_c5d8c_row0_col0" class="data row0 col0" >5.0</td>
      <td id="T_c5d8c_row0_col1" class="data row0 col1" >2.0</td>
      <td id="T_c5d8c_row0_col2" class="data row0 col2" >0.8613</td>
      <td id="T_c5d8c_row0_col3" class="data row0 col3" >0.0006</td>
      <td id="T_c5d8c_row0_col4" class="data row0 col4" >0.5698</td>
      <td id="T_c5d8c_row0_col5" class="data row0 col5" >-0.0018</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row1_col0" class="data row1 col0" >5.0</td>
      <td id="T_c5d8c_row1_col1" class="data row1 col1" >1.0</td>
      <td id="T_c5d8c_row1_col2" class="data row1 col2" >0.8613</td>
      <td id="T_c5d8c_row1_col3" class="data row1 col3" >0.0006</td>
      <td id="T_c5d8c_row1_col4" class="data row1 col4" >0.5701</td>
      <td id="T_c5d8c_row1_col5" class="data row1 col5" >-0.0021</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row2_col0" class="data row2 col0" >5.0</td>
      <td id="T_c5d8c_row2_col1" class="data row2 col1" >0.5</td>
      <td id="T_c5d8c_row2_col2" class="data row2 col2" >0.8612</td>
      <td id="T_c5d8c_row2_col3" class="data row2 col3" >0.0006</td>
      <td id="T_c5d8c_row2_col4" class="data row2 col4" >0.5697</td>
      <td id="T_c5d8c_row2_col5" class="data row2 col5" >-0.002</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row3_col0" class="data row3 col0" >5.0</td>
      <td id="T_c5d8c_row3_col1" class="data row3 col1" >0.2</td>
      <td id="T_c5d8c_row3_col2" class="data row3 col2" >0.8612</td>
      <td id="T_c5d8c_row3_col3" class="data row3 col3" >0.0005</td>
      <td id="T_c5d8c_row3_col4" class="data row3 col4" >0.569</td>
      <td id="T_c5d8c_row3_col5" class="data row3 col5" >-0.002</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row4_col0" class="data row4 col0" >5.0</td>
      <td id="T_c5d8c_row4_col1" class="data row4 col1" >0.1</td>
      <td id="T_c5d8c_row4_col2" class="data row4 col2" >0.861</td>
      <td id="T_c5d8c_row4_col3" class="data row4 col3" >0.0005</td>
      <td id="T_c5d8c_row4_col4" class="data row4 col4" >0.5678</td>
      <td id="T_c5d8c_row4_col5" class="data row4 col5" >-0.0013</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row5_col0" class="data row5 col0" >5.0</td>
      <td id="T_c5d8c_row5_col1" class="data row5 col1" >0.05</td>
      <td id="T_c5d8c_row5_col2" class="data row5 col2" >0.8607</td>
      <td id="T_c5d8c_row5_col3" class="data row5 col3" >0.0005</td>
      <td id="T_c5d8c_row5_col4" class="data row5 col4" >0.567</td>
      <td id="T_c5d8c_row5_col5" class="data row5 col5" >-0.0012</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row6_col0" class="data row6 col0" >4.0</td>
      <td id="T_c5d8c_row6_col1" class="data row6 col1" >2.0</td>
      <td id="T_c5d8c_row6_col2" class="data row6 col2" >0.8604</td>
      <td id="T_c5d8c_row6_col3" class="data row6 col3" >0.0005</td>
      <td id="T_c5d8c_row6_col4" class="data row6 col4" >0.565</td>
      <td id="T_c5d8c_row6_col5" class="data row6 col5" >-0.0019</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row7_col0" class="data row7 col0" >4.0</td>
      <td id="T_c5d8c_row7_col1" class="data row7 col1" >1.0</td>
      <td id="T_c5d8c_row7_col2" class="data row7 col2" >0.8604</td>
      <td id="T_c5d8c_row7_col3" class="data row7 col3" >0.0005</td>
      <td id="T_c5d8c_row7_col4" class="data row7 col4" >0.5652</td>
      <td id="T_c5d8c_row7_col5" class="data row7 col5" >-0.0021</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row8_col0" class="data row8 col0" >4.0</td>
      <td id="T_c5d8c_row8_col1" class="data row8 col1" >0.5</td>
      <td id="T_c5d8c_row8_col2" class="data row8 col2" >0.8604</td>
      <td id="T_c5d8c_row8_col3" class="data row8 col3" >0.0005</td>
      <td id="T_c5d8c_row8_col4" class="data row8 col4" >0.5651</td>
      <td id="T_c5d8c_row8_col5" class="data row8 col5" >-0.0022</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row9_col0" class="data row9 col0" >2.0</td>
      <td id="T_c5d8c_row9_col1" class="data row9 col1" >2.0</td>
      <td id="T_c5d8c_row9_col2" class="data row9 col2" >0.8603</td>
      <td id="T_c5d8c_row9_col3" class="data row9 col3" >0.0005</td>
      <td id="T_c5d8c_row9_col4" class="data row9 col4" >0.5657</td>
      <td id="T_c5d8c_row9_col5" class="data row9 col5" >-0.0021</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row10_col0" class="data row10 col0" >5.0</td>
      <td id="T_c5d8c_row10_col1" class="data row10 col1" >0.025</td>
      <td id="T_c5d8c_row10_col2" class="data row10 col2" >0.8603</td>
      <td id="T_c5d8c_row10_col3" class="data row10 col3" >0.0004</td>
      <td id="T_c5d8c_row10_col4" class="data row10 col4" >0.5666</td>
      <td id="T_c5d8c_row10_col5" class="data row10 col5" >-0.0015</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row11_col0" class="data row11 col0" >2.0</td>
      <td id="T_c5d8c_row11_col1" class="data row11 col1" >1.0</td>
      <td id="T_c5d8c_row11_col2" class="data row11 col2" >0.8602</td>
      <td id="T_c5d8c_row11_col3" class="data row11 col3" >0.0006</td>
      <td id="T_c5d8c_row11_col4" class="data row11 col4" >0.5657</td>
      <td id="T_c5d8c_row11_col5" class="data row11 col5" >-0.002</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row12_col0" class="data row12 col0" >2.0</td>
      <td id="T_c5d8c_row12_col1" class="data row12 col1" >0.5</td>
      <td id="T_c5d8c_row12_col2" class="data row12 col2" >0.8602</td>
      <td id="T_c5d8c_row12_col3" class="data row12 col3" >0.0005</td>
      <td id="T_c5d8c_row12_col4" class="data row12 col4" >0.5655</td>
      <td id="T_c5d8c_row12_col5" class="data row12 col5" >-0.002</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row13_col0" class="data row13 col0" >3.0</td>
      <td id="T_c5d8c_row13_col1" class="data row13 col1" >1.0</td>
      <td id="T_c5d8c_row13_col2" class="data row13 col2" >0.8601</td>
      <td id="T_c5d8c_row13_col3" class="data row13 col3" >0.0006</td>
      <td id="T_c5d8c_row13_col4" class="data row13 col4" >0.5642</td>
      <td id="T_c5d8c_row13_col5" class="data row13 col5" >-0.0003</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row14_col0" class="data row14 col0" >4.0</td>
      <td id="T_c5d8c_row14_col1" class="data row14 col1" >0.2</td>
      <td id="T_c5d8c_row14_col2" class="data row14 col2" >0.8601</td>
      <td id="T_c5d8c_row14_col3" class="data row14 col3" >0.0005</td>
      <td id="T_c5d8c_row14_col4" class="data row14 col4" >0.5645</td>
      <td id="T_c5d8c_row14_col5" class="data row14 col5" >-0.0018</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row15_col0" class="data row15 col0" >3.0</td>
      <td id="T_c5d8c_row15_col1" class="data row15 col1" >2.0</td>
      <td id="T_c5d8c_row15_col2" class="data row15 col2" >0.8601</td>
      <td id="T_c5d8c_row15_col3" class="data row15 col3" >0.0006</td>
      <td id="T_c5d8c_row15_col4" class="data row15 col4" >0.5639</td>
      <td id="T_c5d8c_row15_col5" class="data row15 col5" >-0.0004</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row16_col0" class="data row16 col0" >1.0</td>
      <td id="T_c5d8c_row16_col1" class="data row16 col1" >2.0</td>
      <td id="T_c5d8c_row16_col2" class="data row16 col2" >0.8601</td>
      <td id="T_c5d8c_row16_col3" class="data row16 col3" >0.0005</td>
      <td id="T_c5d8c_row16_col4" class="data row16 col4" >0.5653</td>
      <td id="T_c5d8c_row16_col5" class="data row16 col5" >-0.0017</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row17_col0" class="data row17 col0" >1.0</td>
      <td id="T_c5d8c_row17_col1" class="data row17 col1" >0.5</td>
      <td id="T_c5d8c_row17_col2" class="data row17 col2" >0.86</td>
      <td id="T_c5d8c_row17_col3" class="data row17 col3" >0.0005</td>
      <td id="T_c5d8c_row17_col4" class="data row17 col4" >0.5658</td>
      <td id="T_c5d8c_row17_col5" class="data row17 col5" >-0.002</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row18_col0" class="data row18 col0" >4.0</td>
      <td id="T_c5d8c_row18_col1" class="data row18 col1" >0.1</td>
      <td id="T_c5d8c_row18_col2" class="data row18 col2" >0.86</td>
      <td id="T_c5d8c_row18_col3" class="data row18 col3" >0.0005</td>
      <td id="T_c5d8c_row18_col4" class="data row18 col4" >0.564</td>
      <td id="T_c5d8c_row18_col5" class="data row18 col5" >-0.0019</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row19_col0" class="data row19 col0" >2.0</td>
      <td id="T_c5d8c_row19_col1" class="data row19 col1" >0.2</td>
      <td id="T_c5d8c_row19_col2" class="data row19 col2" >0.86</td>
      <td id="T_c5d8c_row19_col3" class="data row19 col3" >0.0005</td>
      <td id="T_c5d8c_row19_col4" class="data row19 col4" >0.5646</td>
      <td id="T_c5d8c_row19_col5" class="data row19 col5" >-0.0013</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row20_col0" class="data row20 col0" >3.0</td>
      <td id="T_c5d8c_row20_col1" class="data row20 col1" >0.5</td>
      <td id="T_c5d8c_row20_col2" class="data row20 col2" >0.86</td>
      <td id="T_c5d8c_row20_col3" class="data row20 col3" >0.0006</td>
      <td id="T_c5d8c_row20_col4" class="data row20 col4" >0.5639</td>
      <td id="T_c5d8c_row20_col5" class="data row20 col5" >-0.0006</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row21_col0" class="data row21 col0" >1.0</td>
      <td id="T_c5d8c_row21_col1" class="data row21 col1" >1.0</td>
      <td id="T_c5d8c_row21_col2" class="data row21 col2" >0.8599</td>
      <td id="T_c5d8c_row21_col3" class="data row21 col3" >0.0006</td>
      <td id="T_c5d8c_row21_col4" class="data row21 col4" >0.5651</td>
      <td id="T_c5d8c_row21_col5" class="data row21 col5" >-0.0015</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row22_col0" class="data row22 col0" >3.0</td>
      <td id="T_c5d8c_row22_col1" class="data row22 col1" >0.2</td>
      <td id="T_c5d8c_row22_col2" class="data row22 col2" >0.8599</td>
      <td id="T_c5d8c_row22_col3" class="data row22 col3" >0.0006</td>
      <td id="T_c5d8c_row22_col4" class="data row22 col4" >0.5642</td>
      <td id="T_c5d8c_row22_col5" class="data row22 col5" >-0.001</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row23_col0" class="data row23 col0" >2.0</td>
      <td id="T_c5d8c_row23_col1" class="data row23 col1" >0.1</td>
      <td id="T_c5d8c_row23_col2" class="data row23 col2" >0.8598</td>
      <td id="T_c5d8c_row23_col3" class="data row23 col3" >0.0005</td>
      <td id="T_c5d8c_row23_col4" class="data row23 col4" >0.5638</td>
      <td id="T_c5d8c_row23_col5" class="data row23 col5" >-0.0011</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row24_col0" class="data row24 col0" >1.0</td>
      <td id="T_c5d8c_row24_col1" class="data row24 col1" >0.2</td>
      <td id="T_c5d8c_row24_col2" class="data row24 col2" >0.8598</td>
      <td id="T_c5d8c_row24_col3" class="data row24 col3" >0.0005</td>
      <td id="T_c5d8c_row24_col4" class="data row24 col4" >0.5645</td>
      <td id="T_c5d8c_row24_col5" class="data row24 col5" >-0.0009</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row25_col0" class="data row25 col0" >3.0</td>
      <td id="T_c5d8c_row25_col1" class="data row25 col1" >0.1</td>
      <td id="T_c5d8c_row25_col2" class="data row25 col2" >0.8598</td>
      <td id="T_c5d8c_row25_col3" class="data row25 col3" >0.0005</td>
      <td id="T_c5d8c_row25_col4" class="data row25 col4" >0.564</td>
      <td id="T_c5d8c_row25_col5" class="data row25 col5" >-0.0013</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row26_col0" class="data row26 col0" >4.0</td>
      <td id="T_c5d8c_row26_col1" class="data row26 col1" >0.05</td>
      <td id="T_c5d8c_row26_col2" class="data row26 col2" >0.8597</td>
      <td id="T_c5d8c_row26_col3" class="data row26 col3" >0.0005</td>
      <td id="T_c5d8c_row26_col4" class="data row26 col4" >0.5634</td>
      <td id="T_c5d8c_row26_col5" class="data row26 col5" >-0.0017</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row27_col0" class="data row27 col0" >3.0</td>
      <td id="T_c5d8c_row27_col1" class="data row27 col1" >0.05</td>
      <td id="T_c5d8c_row27_col2" class="data row27 col2" >0.8596</td>
      <td id="T_c5d8c_row27_col3" class="data row27 col3" >0.0004</td>
      <td id="T_c5d8c_row27_col4" class="data row27 col4" >0.5637</td>
      <td id="T_c5d8c_row27_col5" class="data row27 col5" >-0.002</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row28_col0" class="data row28 col0" >1.0</td>
      <td id="T_c5d8c_row28_col1" class="data row28 col1" >0.1</td>
      <td id="T_c5d8c_row28_col2" class="data row28 col2" >0.8596</td>
      <td id="T_c5d8c_row28_col3" class="data row28 col3" >0.0005</td>
      <td id="T_c5d8c_row28_col4" class="data row28 col4" >0.5634</td>
      <td id="T_c5d8c_row28_col5" class="data row28 col5" >-0.0008</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row29_col0" class="data row29 col0" >2.0</td>
      <td id="T_c5d8c_row29_col1" class="data row29 col1" >0.05</td>
      <td id="T_c5d8c_row29_col2" class="data row29 col2" >0.8595</td>
      <td id="T_c5d8c_row29_col3" class="data row29 col3" >0.0004</td>
      <td id="T_c5d8c_row29_col4" class="data row29 col4" >0.5636</td>
      <td id="T_c5d8c_row29_col5" class="data row29 col5" >-0.0016</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row30_col0" class="data row30 col0" >5.0</td>
      <td id="T_c5d8c_row30_col1" class="data row30 col1" >0.01</td>
      <td id="T_c5d8c_row30_col2" class="data row30 col2" >0.8593</td>
      <td id="T_c5d8c_row30_col3" class="data row30 col3" >0.0004</td>
      <td id="T_c5d8c_row30_col4" class="data row30 col4" >0.566</td>
      <td id="T_c5d8c_row30_col5" class="data row30 col5" >-0.0018</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row31_col0" class="data row31 col0" >3.0</td>
      <td id="T_c5d8c_row31_col1" class="data row31 col1" >0.025</td>
      <td id="T_c5d8c_row31_col2" class="data row31 col2" >0.8593</td>
      <td id="T_c5d8c_row31_col3" class="data row31 col3" >0.0004</td>
      <td id="T_c5d8c_row31_col4" class="data row31 col4" >0.5636</td>
      <td id="T_c5d8c_row31_col5" class="data row31 col5" >-0.0027</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row32_col0" class="data row32 col0" >1.0</td>
      <td id="T_c5d8c_row32_col1" class="data row32 col1" >0.05</td>
      <td id="T_c5d8c_row32_col2" class="data row32 col2" >0.8593</td>
      <td id="T_c5d8c_row32_col3" class="data row32 col3" >0.0004</td>
      <td id="T_c5d8c_row32_col4" class="data row32 col4" >0.563</td>
      <td id="T_c5d8c_row32_col5" class="data row32 col5" >-0.0009</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row33_col0" class="data row33 col0" >4.0</td>
      <td id="T_c5d8c_row33_col1" class="data row33 col1" >0.025</td>
      <td id="T_c5d8c_row33_col2" class="data row33 col2" >0.8593</td>
      <td id="T_c5d8c_row33_col3" class="data row33 col3" >0.0004</td>
      <td id="T_c5d8c_row33_col4" class="data row33 col4" >0.5624</td>
      <td id="T_c5d8c_row33_col5" class="data row33 col5" >-0.0013</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row34_col0" class="data row34 col0" >2.0</td>
      <td id="T_c5d8c_row34_col1" class="data row34 col1" >0.025</td>
      <td id="T_c5d8c_row34_col2" class="data row34 col2" >0.8591</td>
      <td id="T_c5d8c_row34_col3" class="data row34 col3" >0.0004</td>
      <td id="T_c5d8c_row34_col4" class="data row34 col4" >0.5618</td>
      <td id="T_c5d8c_row34_col5" class="data row34 col5" >-0.0008</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row35_col0" class="data row35 col0" >1.0</td>
      <td id="T_c5d8c_row35_col1" class="data row35 col1" >0.025</td>
      <td id="T_c5d8c_row35_col2" class="data row35 col2" >0.8589</td>
      <td id="T_c5d8c_row35_col3" class="data row35 col3" >0.0004</td>
      <td id="T_c5d8c_row35_col4" class="data row35 col4" >0.563</td>
      <td id="T_c5d8c_row35_col5" class="data row35 col5" >-0.002</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row36_col0" class="data row36 col0" >3.0</td>
      <td id="T_c5d8c_row36_col1" class="data row36 col1" >0.01</td>
      <td id="T_c5d8c_row36_col2" class="data row36 col2" >0.8586</td>
      <td id="T_c5d8c_row36_col3" class="data row36 col3" >0.0003</td>
      <td id="T_c5d8c_row36_col4" class="data row36 col4" >0.5628</td>
      <td id="T_c5d8c_row36_col5" class="data row36 col5" >-0.002</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row37_col0" class="data row37 col0" >4.0</td>
      <td id="T_c5d8c_row37_col1" class="data row37 col1" >0.01</td>
      <td id="T_c5d8c_row37_col2" class="data row37 col2" >0.8584</td>
      <td id="T_c5d8c_row37_col3" class="data row37 col3" >0.0003</td>
      <td id="T_c5d8c_row37_col4" class="data row37 col4" >0.5617</td>
      <td id="T_c5d8c_row37_col5" class="data row37 col5" >-0.0018</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row38_col0" class="data row38 col0" >2.0</td>
      <td id="T_c5d8c_row38_col1" class="data row38 col1" >0.01</td>
      <td id="T_c5d8c_row38_col2" class="data row38 col2" >0.8583</td>
      <td id="T_c5d8c_row38_col3" class="data row38 col3" >0.0004</td>
      <td id="T_c5d8c_row38_col4" class="data row38 col4" >0.5619</td>
      <td id="T_c5d8c_row38_col5" class="data row38 col5" >-0.0018</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row39_col0" class="data row39 col0" >1.0</td>
      <td id="T_c5d8c_row39_col1" class="data row39 col1" >0.01</td>
      <td id="T_c5d8c_row39_col2" class="data row39 col2" >0.8581</td>
      <td id="T_c5d8c_row39_col3" class="data row39 col3" >0.0003</td>
      <td id="T_c5d8c_row39_col4" class="data row39 col4" >0.562</td>
      <td id="T_c5d8c_row39_col5" class="data row39 col5" >-0.0018</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row40_col0" class="data row40 col0" >0.0</td>
      <td id="T_c5d8c_row40_col1" class="data row40 col1" >2.0</td>
      <td id="T_c5d8c_row40_col2" class="data row40 col2" >0.8532</td>
      <td id="T_c5d8c_row40_col3" class="data row40 col3" >0.0002</td>
      <td id="T_c5d8c_row40_col4" class="data row40 col4" >0.5478</td>
      <td id="T_c5d8c_row40_col5" class="data row40 col5" >-0.0017</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row41_col0" class="data row41 col0" >0.0</td>
      <td id="T_c5d8c_row41_col1" class="data row41 col1" >1.0</td>
      <td id="T_c5d8c_row41_col2" class="data row41 col2" >0.8531</td>
      <td id="T_c5d8c_row41_col3" class="data row41 col3" >0.0003</td>
      <td id="T_c5d8c_row41_col4" class="data row41 col4" >0.5476</td>
      <td id="T_c5d8c_row41_col5" class="data row41 col5" >-0.0015</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row42_col0" class="data row42 col0" >0.0</td>
      <td id="T_c5d8c_row42_col1" class="data row42 col1" >0.5</td>
      <td id="T_c5d8c_row42_col2" class="data row42 col2" >0.853</td>
      <td id="T_c5d8c_row42_col3" class="data row42 col3" >0.0003</td>
      <td id="T_c5d8c_row42_col4" class="data row42 col4" >0.5472</td>
      <td id="T_c5d8c_row42_col5" class="data row42 col5" >-0.0013</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row43_col0" class="data row43 col0" >0.0</td>
      <td id="T_c5d8c_row43_col1" class="data row43 col1" >0.2</td>
      <td id="T_c5d8c_row43_col2" class="data row43 col2" >0.853</td>
      <td id="T_c5d8c_row43_col3" class="data row43 col3" >0.0003</td>
      <td id="T_c5d8c_row43_col4" class="data row43 col4" >0.547</td>
      <td id="T_c5d8c_row43_col5" class="data row43 col5" >-0.0011</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row44_col0" class="data row44 col0" >0.0</td>
      <td id="T_c5d8c_row44_col1" class="data row44 col1" >0.1</td>
      <td id="T_c5d8c_row44_col2" class="data row44 col2" >0.8529</td>
      <td id="T_c5d8c_row44_col3" class="data row44 col3" >0.0002</td>
      <td id="T_c5d8c_row44_col4" class="data row44 col4" >0.5474</td>
      <td id="T_c5d8c_row44_col5" class="data row44 col5" >-0.0018</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row45_col0" class="data row45 col0" >0.0</td>
      <td id="T_c5d8c_row45_col1" class="data row45 col1" >0.05</td>
      <td id="T_c5d8c_row45_col2" class="data row45 col2" >0.8527</td>
      <td id="T_c5d8c_row45_col3" class="data row45 col3" >0.0002</td>
      <td id="T_c5d8c_row45_col4" class="data row45 col4" >0.5475</td>
      <td id="T_c5d8c_row45_col5" class="data row45 col5" >-0.0014</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row46_col0" class="data row46 col0" >0.0</td>
      <td id="T_c5d8c_row46_col1" class="data row46 col1" >0.025</td>
      <td id="T_c5d8c_row46_col2" class="data row46 col2" >0.8524</td>
      <td id="T_c5d8c_row46_col3" class="data row46 col3" >0.0002</td>
      <td id="T_c5d8c_row46_col4" class="data row46 col4" >0.5479</td>
      <td id="T_c5d8c_row46_col5" class="data row46 col5" >-0.0013</td>
    </tr>
    <tr>
      <td id="T_c5d8c_row47_col0" class="data row47 col0" >0.0</td>
      <td id="T_c5d8c_row47_col1" class="data row47 col1" >0.01</td>
      <td id="T_c5d8c_row47_col2" class="data row47 col2" >0.8522</td>
      <td id="T_c5d8c_row47_col3" class="data row47 col3" >0.0002</td>
      <td id="T_c5d8c_row47_col4" class="data row47 col4" >0.5506</td>
      <td id="T_c5d8c_row47_col5" class="data row47 col5" >-0.0025</td>
    </tr>
  </tbody>
</table>




```python
if is_processing_here3:
    print_df(raw_lr_coef_sign_stability_df)
```


<style type="text/css">
</style>
<table id="T_6fb3c">
  <thead>
    <tr>
      <th id="T_6fb3c_level0_col0" class="col_heading level0 col0" >fit columns index</th>
      <th id="T_6fb3c_level0_col1" class="col_heading level0 col1" >C value</th>
      <th id="T_6fb3c_level0_col2" class="col_heading level0 col2" >colname</th>
      <th id="T_6fb3c_level0_col3" class="col_heading level0 col3" >sign stabilty</th>
      <th id="T_6fb3c_level0_col4" class="col_heading level0 col4" >std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_6fb3c_row0_col0" class="data row0 col0" >0.0</td>
      <td id="T_6fb3c_row0_col1" class="data row0 col1" >0.01</td>
      <td id="T_6fb3c_row0_col2" class="data row0 col2" >both_missing_flag</td>
      <td id="T_6fb3c_row0_col3" class="data row0 col3" >-0.2</td>
      <td id="T_6fb3c_row0_col4" class="data row0 col4" >0.0177</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row1_col0" class="data row1 col0" >1.0</td>
      <td id="T_6fb3c_row1_col1" class="data row1 col1" >0.01</td>
      <td id="T_6fb3c_row1_col2" class="data row1 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row1_col3" class="data row1 col3" >-0.2</td>
      <td id="T_6fb3c_row1_col4" class="data row1 col4" >0.0089</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row2_col0" class="data row2 col0" >1.0</td>
      <td id="T_6fb3c_row2_col1" class="data row2 col1" >0.025</td>
      <td id="T_6fb3c_row2_col2" class="data row2 col2" >credit_late_density</td>
      <td id="T_6fb3c_row2_col3" class="data row2 col3" >0.2</td>
      <td id="T_6fb3c_row2_col4" class="data row2 col4" >0.0249</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row3_col0" class="data row3 col0" >1.0</td>
      <td id="T_6fb3c_row3_col1" class="data row3 col1" >0.05</td>
      <td id="T_6fb3c_row3_col2" class="data row3 col2" >credit_late_density</td>
      <td id="T_6fb3c_row3_col3" class="data row3 col3" >-0.2</td>
      <td id="T_6fb3c_row3_col4" class="data row3 col4" >0.0277</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row4_col0" class="data row4 col0" >1.0</td>
      <td id="T_6fb3c_row4_col1" class="data row4 col1" >0.1</td>
      <td id="T_6fb3c_row4_col2" class="data row4 col2" >credit_late_density</td>
      <td id="T_6fb3c_row4_col3" class="data row4 col3" >0.2</td>
      <td id="T_6fb3c_row4_col4" class="data row4 col4" >0.0416</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row5_col0" class="data row5 col0" >1.0</td>
      <td id="T_6fb3c_row5_col1" class="data row5 col1" >0.2</td>
      <td id="T_6fb3c_row5_col2" class="data row5 col2" >is_debt_high</td>
      <td id="T_6fb3c_row5_col3" class="data row5 col3" >0.2</td>
      <td id="T_6fb3c_row5_col4" class="data row5 col4" >0.0515</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row6_col0" class="data row6 col0" >1.0</td>
      <td id="T_6fb3c_row6_col1" class="data row6 col1" >0.2</td>
      <td id="T_6fb3c_row6_col2" class="data row6 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row6_col3" class="data row6 col3" >-0.2</td>
      <td id="T_6fb3c_row6_col4" class="data row6 col4" >0.0278</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row7_col0" class="data row7 col0" >1.0</td>
      <td id="T_6fb3c_row7_col1" class="data row7 col1" >0.2</td>
      <td id="T_6fb3c_row7_col2" class="data row7 col2" >credit_late_density</td>
      <td id="T_6fb3c_row7_col3" class="data row7 col3" >-0.2</td>
      <td id="T_6fb3c_row7_col4" class="data row7 col4" >0.0283</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row8_col0" class="data row8 col0" >1.0</td>
      <td id="T_6fb3c_row8_col1" class="data row8 col1" >0.5</td>
      <td id="T_6fb3c_row8_col2" class="data row8 col2" >is_debt_high</td>
      <td id="T_6fb3c_row8_col3" class="data row8 col3" >-0.2</td>
      <td id="T_6fb3c_row8_col4" class="data row8 col4" >0.0564</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row9_col0" class="data row9 col0" >1.0</td>
      <td id="T_6fb3c_row9_col1" class="data row9 col1" >0.5</td>
      <td id="T_6fb3c_row9_col2" class="data row9 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row9_col3" class="data row9 col3" >-0.2</td>
      <td id="T_6fb3c_row9_col4" class="data row9 col4" >0.0192</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row10_col0" class="data row10 col0" >1.0</td>
      <td id="T_6fb3c_row10_col1" class="data row10 col1" >1.0</td>
      <td id="T_6fb3c_row10_col2" class="data row10 col2" >is_debt_high</td>
      <td id="T_6fb3c_row10_col3" class="data row10 col3" >-0.6</td>
      <td id="T_6fb3c_row10_col4" class="data row10 col4" >0.062</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row11_col0" class="data row11 col0" >1.0</td>
      <td id="T_6fb3c_row11_col1" class="data row11 col1" >1.0</td>
      <td id="T_6fb3c_row11_col2" class="data row11 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row11_col3" class="data row11 col3" >-0.2</td>
      <td id="T_6fb3c_row11_col4" class="data row11 col4" >0.0337</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row12_col0" class="data row12 col0" >1.0</td>
      <td id="T_6fb3c_row12_col1" class="data row12 col1" >2.0</td>
      <td id="T_6fb3c_row12_col2" class="data row12 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row12_col3" class="data row12 col3" >0.2</td>
      <td id="T_6fb3c_row12_col4" class="data row12 col4" >0.0153</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row13_col0" class="data row13 col0" >2.0</td>
      <td id="T_6fb3c_row13_col1" class="data row13 col1" >0.01</td>
      <td id="T_6fb3c_row13_col2" class="data row13 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row13_col3" class="data row13 col3" >-0.2</td>
      <td id="T_6fb3c_row13_col4" class="data row13 col4" >0.0109</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row14_col0" class="data row14 col0" >2.0</td>
      <td id="T_6fb3c_row14_col1" class="data row14 col1" >0.01</td>
      <td id="T_6fb3c_row14_col2" class="data row14 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row14_col3" class="data row14 col3" >0.2</td>
      <td id="T_6fb3c_row14_col4" class="data row14 col4" >0.0303</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row15_col0" class="data row15 col0" >2.0</td>
      <td id="T_6fb3c_row15_col1" class="data row15 col1" >0.025</td>
      <td id="T_6fb3c_row15_col2" class="data row15 col2" >credit_late_density</td>
      <td id="T_6fb3c_row15_col3" class="data row15 col3" >0.2</td>
      <td id="T_6fb3c_row15_col4" class="data row15 col4" >0.0316</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row16_col0" class="data row16 col0" >2.0</td>
      <td id="T_6fb3c_row16_col1" class="data row16 col1" >0.025</td>
      <td id="T_6fb3c_row16_col2" class="data row16 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row16_col3" class="data row16 col3" >0.2</td>
      <td id="T_6fb3c_row16_col4" class="data row16 col4" >0.0557</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row17_col0" class="data row17 col0" >2.0</td>
      <td id="T_6fb3c_row17_col1" class="data row17 col1" >0.05</td>
      <td id="T_6fb3c_row17_col2" class="data row17 col2" >credit_late_density</td>
      <td id="T_6fb3c_row17_col3" class="data row17 col3" >-0.2</td>
      <td id="T_6fb3c_row17_col4" class="data row17 col4" >0.0392</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row18_col0" class="data row18 col0" >2.0</td>
      <td id="T_6fb3c_row18_col1" class="data row18 col1" >0.05</td>
      <td id="T_6fb3c_row18_col2" class="data row18 col2" >monthly_debt_log</td>
      <td id="T_6fb3c_row18_col3" class="data row18 col3" >-0.2</td>
      <td id="T_6fb3c_row18_col4" class="data row18 col4" >0.0059</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row19_col0" class="data row19 col0" >2.0</td>
      <td id="T_6fb3c_row19_col1" class="data row19 col1" >0.05</td>
      <td id="T_6fb3c_row19_col2" class="data row19 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row19_col3" class="data row19 col3" >-0.6</td>
      <td id="T_6fb3c_row19_col4" class="data row19 col4" >0.0757</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row20_col0" class="data row20 col0" >2.0</td>
      <td id="T_6fb3c_row20_col1" class="data row20 col1" >0.05</td>
      <td id="T_6fb3c_row20_col2" class="data row20 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row20_col3" class="data row20 col3" >0.6</td>
      <td id="T_6fb3c_row20_col4" class="data row20 col4" >0.1978</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row21_col0" class="data row21 col0" >2.0</td>
      <td id="T_6fb3c_row21_col1" class="data row21 col1" >0.1</td>
      <td id="T_6fb3c_row21_col2" class="data row21 col2" >credit_late_density</td>
      <td id="T_6fb3c_row21_col3" class="data row21 col3" >0.2</td>
      <td id="T_6fb3c_row21_col4" class="data row21 col4" >0.0422</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row22_col0" class="data row22 col0" >2.0</td>
      <td id="T_6fb3c_row22_col1" class="data row22 col1" >0.1</td>
      <td id="T_6fb3c_row22_col2" class="data row22 col2" >monthly_debt_log</td>
      <td id="T_6fb3c_row22_col3" class="data row22 col3" >-0.2</td>
      <td id="T_6fb3c_row22_col4" class="data row22 col4" >0.0086</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row23_col0" class="data row23 col0" >2.0</td>
      <td id="T_6fb3c_row23_col1" class="data row23 col1" >0.1</td>
      <td id="T_6fb3c_row23_col2" class="data row23 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row23_col3" class="data row23 col3" >-0.6</td>
      <td id="T_6fb3c_row23_col4" class="data row23 col4" >0.0922</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row24_col0" class="data row24 col0" >2.0</td>
      <td id="T_6fb3c_row24_col1" class="data row24 col1" >0.1</td>
      <td id="T_6fb3c_row24_col2" class="data row24 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row24_col3" class="data row24 col3" >0.6</td>
      <td id="T_6fb3c_row24_col4" class="data row24 col4" >0.2048</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row25_col0" class="data row25 col0" >2.0</td>
      <td id="T_6fb3c_row25_col1" class="data row25 col1" >0.2</td>
      <td id="T_6fb3c_row25_col2" class="data row25 col2" >is_debt_high</td>
      <td id="T_6fb3c_row25_col3" class="data row25 col3" >0.2</td>
      <td id="T_6fb3c_row25_col4" class="data row25 col4" >0.0554</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row26_col0" class="data row26 col0" >2.0</td>
      <td id="T_6fb3c_row26_col1" class="data row26 col1" >0.2</td>
      <td id="T_6fb3c_row26_col2" class="data row26 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row26_col3" class="data row26 col3" >-0.2</td>
      <td id="T_6fb3c_row26_col4" class="data row26 col4" >0.0303</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row27_col0" class="data row27 col0" >2.0</td>
      <td id="T_6fb3c_row27_col1" class="data row27 col1" >0.2</td>
      <td id="T_6fb3c_row27_col2" class="data row27 col2" >credit_late_density</td>
      <td id="T_6fb3c_row27_col3" class="data row27 col3" >0.2</td>
      <td id="T_6fb3c_row27_col4" class="data row27 col4" >0.0421</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row28_col0" class="data row28 col0" >2.0</td>
      <td id="T_6fb3c_row28_col1" class="data row28 col1" >0.2</td>
      <td id="T_6fb3c_row28_col2" class="data row28 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row28_col3" class="data row28 col3" >-0.6</td>
      <td id="T_6fb3c_row28_col4" class="data row28 col4" >0.1126</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row29_col0" class="data row29 col0" >2.0</td>
      <td id="T_6fb3c_row29_col1" class="data row29 col1" >0.2</td>
      <td id="T_6fb3c_row29_col2" class="data row29 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row29_col3" class="data row29 col3" >0.6</td>
      <td id="T_6fb3c_row29_col4" class="data row29 col4" >0.2524</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row30_col0" class="data row30 col0" >2.0</td>
      <td id="T_6fb3c_row30_col1" class="data row30 col1" >0.5</td>
      <td id="T_6fb3c_row30_col2" class="data row30 col2" >is_debt_high</td>
      <td id="T_6fb3c_row30_col3" class="data row30 col3" >-0.2</td>
      <td id="T_6fb3c_row30_col4" class="data row30 col4" >0.0586</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row31_col0" class="data row31 col0" >2.0</td>
      <td id="T_6fb3c_row31_col1" class="data row31 col1" >0.5</td>
      <td id="T_6fb3c_row31_col2" class="data row31 col2" >credit_late_density</td>
      <td id="T_6fb3c_row31_col3" class="data row31 col3" >-0.2</td>
      <td id="T_6fb3c_row31_col4" class="data row31 col4" >0.039</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row32_col0" class="data row32 col0" >2.0</td>
      <td id="T_6fb3c_row32_col1" class="data row32 col1" >0.5</td>
      <td id="T_6fb3c_row32_col2" class="data row32 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row32_col3" class="data row32 col3" >-0.6</td>
      <td id="T_6fb3c_row32_col4" class="data row32 col4" >0.1198</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row33_col0" class="data row33 col0" >2.0</td>
      <td id="T_6fb3c_row33_col1" class="data row33 col1" >0.5</td>
      <td id="T_6fb3c_row33_col2" class="data row33 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row33_col3" class="data row33 col3" >0.2</td>
      <td id="T_6fb3c_row33_col4" class="data row33 col4" >0.2757</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row34_col0" class="data row34 col0" >2.0</td>
      <td id="T_6fb3c_row34_col1" class="data row34 col1" >1.0</td>
      <td id="T_6fb3c_row34_col2" class="data row34 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row34_col3" class="data row34 col3" >0.2</td>
      <td id="T_6fb3c_row34_col4" class="data row34 col4" >0.0242</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row35_col0" class="data row35 col0" >2.0</td>
      <td id="T_6fb3c_row35_col1" class="data row35 col1" >1.0</td>
      <td id="T_6fb3c_row35_col2" class="data row35 col2" >credit_late_density</td>
      <td id="T_6fb3c_row35_col3" class="data row35 col3" >0.2</td>
      <td id="T_6fb3c_row35_col4" class="data row35 col4" >0.0414</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row36_col0" class="data row36 col0" >2.0</td>
      <td id="T_6fb3c_row36_col1" class="data row36 col1" >1.0</td>
      <td id="T_6fb3c_row36_col2" class="data row36 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row36_col3" class="data row36 col3" >-0.6</td>
      <td id="T_6fb3c_row36_col4" class="data row36 col4" >0.1209</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row37_col0" class="data row37 col0" >2.0</td>
      <td id="T_6fb3c_row37_col1" class="data row37 col1" >1.0</td>
      <td id="T_6fb3c_row37_col2" class="data row37 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row37_col3" class="data row37 col3" >0.6</td>
      <td id="T_6fb3c_row37_col4" class="data row37 col4" >0.2901</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row38_col0" class="data row38 col0" >2.0</td>
      <td id="T_6fb3c_row38_col1" class="data row38 col1" >2.0</td>
      <td id="T_6fb3c_row38_col2" class="data row38 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row38_col3" class="data row38 col3" >0.2</td>
      <td id="T_6fb3c_row38_col4" class="data row38 col4" >0.0292</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row39_col0" class="data row39 col0" >2.0</td>
      <td id="T_6fb3c_row39_col1" class="data row39 col1" >2.0</td>
      <td id="T_6fb3c_row39_col2" class="data row39 col2" >credit_late_density</td>
      <td id="T_6fb3c_row39_col3" class="data row39 col3" >0.2</td>
      <td id="T_6fb3c_row39_col4" class="data row39 col4" >0.0421</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row40_col0" class="data row40 col0" >2.0</td>
      <td id="T_6fb3c_row40_col1" class="data row40 col1" >2.0</td>
      <td id="T_6fb3c_row40_col2" class="data row40 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row40_col3" class="data row40 col3" >0.6</td>
      <td id="T_6fb3c_row40_col4" class="data row40 col4" >0.2965</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row41_col0" class="data row41 col0" >3.0</td>
      <td id="T_6fb3c_row41_col1" class="data row41 col1" >0.01</td>
      <td id="T_6fb3c_row41_col2" class="data row41 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row41_col3" class="data row41 col3" >0.2</td>
      <td id="T_6fb3c_row41_col4" class="data row41 col4" >0.0101</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row42_col0" class="data row42 col0" >3.0</td>
      <td id="T_6fb3c_row42_col1" class="data row42 col1" >0.01</td>
      <td id="T_6fb3c_row42_col2" class="data row42 col2" >both_missing_flag</td>
      <td id="T_6fb3c_row42_col3" class="data row42 col3" >0.2</td>
      <td id="T_6fb3c_row42_col4" class="data row42 col4" >0.0135</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row43_col0" class="data row43 col0" >3.0</td>
      <td id="T_6fb3c_row43_col1" class="data row43 col1" >0.025</td>
      <td id="T_6fb3c_row43_col2" class="data row43 col2" >credit_late_density</td>
      <td id="T_6fb3c_row43_col3" class="data row43 col3" >0.2</td>
      <td id="T_6fb3c_row43_col4" class="data row43 col4" >0.0249</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row44_col0" class="data row44 col0" >3.0</td>
      <td id="T_6fb3c_row44_col1" class="data row44 col1" >0.05</td>
      <td id="T_6fb3c_row44_col2" class="data row44 col2" >credit_late_density</td>
      <td id="T_6fb3c_row44_col3" class="data row44 col3" >0.2</td>
      <td id="T_6fb3c_row44_col4" class="data row44 col4" >0.0231</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row45_col0" class="data row45 col0" >3.0</td>
      <td id="T_6fb3c_row45_col1" class="data row45 col1" >0.2</td>
      <td id="T_6fb3c_row45_col2" class="data row45 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row45_col3" class="data row45 col3" >-0.2</td>
      <td id="T_6fb3c_row45_col4" class="data row45 col4" >0.0218</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row46_col0" class="data row46 col0" >3.0</td>
      <td id="T_6fb3c_row46_col1" class="data row46 col1" >0.5</td>
      <td id="T_6fb3c_row46_col2" class="data row46 col2" >is_debt_high</td>
      <td id="T_6fb3c_row46_col3" class="data row46 col3" >0.2</td>
      <td id="T_6fb3c_row46_col4" class="data row46 col4" >0.0623</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row47_col0" class="data row47 col0" >3.0</td>
      <td id="T_6fb3c_row47_col1" class="data row47 col1" >0.5</td>
      <td id="T_6fb3c_row47_col2" class="data row47 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row47_col3" class="data row47 col3" >-0.2</td>
      <td id="T_6fb3c_row47_col4" class="data row47 col4" >0.0198</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row48_col0" class="data row48 col0" >3.0</td>
      <td id="T_6fb3c_row48_col1" class="data row48 col1" >1.0</td>
      <td id="T_6fb3c_row48_col2" class="data row48 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row48_col3" class="data row48 col3" >0.2</td>
      <td id="T_6fb3c_row48_col4" class="data row48 col4" >0.0207</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row49_col0" class="data row49 col0" >3.0</td>
      <td id="T_6fb3c_row49_col1" class="data row49 col1" >2.0</td>
      <td id="T_6fb3c_row49_col2" class="data row49 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row49_col3" class="data row49 col3" >0.2</td>
      <td id="T_6fb3c_row49_col4" class="data row49 col4" >0.0209</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row50_col0" class="data row50 col0" >3.0</td>
      <td id="T_6fb3c_row50_col1" class="data row50 col1" >2.0</td>
      <td id="T_6fb3c_row50_col2" class="data row50 col2" >credit_late_density</td>
      <td id="T_6fb3c_row50_col3" class="data row50 col3" >-0.2</td>
      <td id="T_6fb3c_row50_col4" class="data row50 col4" >0.0437</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row51_col0" class="data row51 col0" >4.0</td>
      <td id="T_6fb3c_row51_col1" class="data row51 col1" >0.01</td>
      <td id="T_6fb3c_row51_col2" class="data row51 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row51_col3" class="data row51 col3" >-0.2</td>
      <td id="T_6fb3c_row51_col4" class="data row51 col4" >0.0099</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row52_col0" class="data row52 col0" >4.0</td>
      <td id="T_6fb3c_row52_col1" class="data row52 col1" >0.01</td>
      <td id="T_6fb3c_row52_col2" class="data row52 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row52_col3" class="data row52 col3" >0.2</td>
      <td id="T_6fb3c_row52_col4" class="data row52 col4" >0.0301</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row53_col0" class="data row53 col0" >4.0</td>
      <td id="T_6fb3c_row53_col1" class="data row53 col1" >0.025</td>
      <td id="T_6fb3c_row53_col2" class="data row53 col2" >credit_late_density</td>
      <td id="T_6fb3c_row53_col3" class="data row53 col3" >0.2</td>
      <td id="T_6fb3c_row53_col4" class="data row53 col4" >0.03</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row54_col0" class="data row54 col0" >4.0</td>
      <td id="T_6fb3c_row54_col1" class="data row54 col1" >0.025</td>
      <td id="T_6fb3c_row54_col2" class="data row54 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row54_col3" class="data row54 col3" >0.2</td>
      <td id="T_6fb3c_row54_col4" class="data row54 col4" >0.0574</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row55_col0" class="data row55 col0" >4.0</td>
      <td id="T_6fb3c_row55_col1" class="data row55 col1" >0.05</td>
      <td id="T_6fb3c_row55_col2" class="data row55 col2" >late_severity_high_x_credit_pressure_low_signal</td>
      <td id="T_6fb3c_row55_col3" class="data row55 col3" >0.2</td>
      <td id="T_6fb3c_row55_col4" class="data row55 col4" >0.0395</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row56_col0" class="data row56 col0" >4.0</td>
      <td id="T_6fb3c_row56_col1" class="data row56 col1" >0.05</td>
      <td id="T_6fb3c_row56_col2" class="data row56 col2" >credit_late_density</td>
      <td id="T_6fb3c_row56_col3" class="data row56 col3" >0.2</td>
      <td id="T_6fb3c_row56_col4" class="data row56 col4" >0.0274</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row57_col0" class="data row57 col0" >4.0</td>
      <td id="T_6fb3c_row57_col1" class="data row57 col1" >0.05</td>
      <td id="T_6fb3c_row57_col2" class="data row57 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row57_col3" class="data row57 col3" >0.2</td>
      <td id="T_6fb3c_row57_col4" class="data row57 col4" >0.0783</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row58_col0" class="data row58 col0" >4.0</td>
      <td id="T_6fb3c_row58_col1" class="data row58 col1" >0.05</td>
      <td id="T_6fb3c_row58_col2" class="data row58 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row58_col3" class="data row58 col3" >0.6</td>
      <td id="T_6fb3c_row58_col4" class="data row58 col4" >0.1891</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row59_col0" class="data row59 col0" >4.0</td>
      <td id="T_6fb3c_row59_col1" class="data row59 col1" >0.1</td>
      <td id="T_6fb3c_row59_col2" class="data row59 col2" >credit_late_density</td>
      <td id="T_6fb3c_row59_col3" class="data row59 col3" >0.2</td>
      <td id="T_6fb3c_row59_col4" class="data row59 col4" >0.0405</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row60_col0" class="data row60 col0" >4.0</td>
      <td id="T_6fb3c_row60_col1" class="data row60 col1" >0.1</td>
      <td id="T_6fb3c_row60_col2" class="data row60 col2" >monthly_debt_log</td>
      <td id="T_6fb3c_row60_col3" class="data row60 col3" >-0.2</td>
      <td id="T_6fb3c_row60_col4" class="data row60 col4" >0.0066</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row61_col0" class="data row61 col0" >4.0</td>
      <td id="T_6fb3c_row61_col1" class="data row61 col1" >0.1</td>
      <td id="T_6fb3c_row61_col2" class="data row61 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row61_col3" class="data row61 col3" >0.6</td>
      <td id="T_6fb3c_row61_col4" class="data row61 col4" >0.094</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row62_col0" class="data row62 col0" >4.0</td>
      <td id="T_6fb3c_row62_col1" class="data row62 col1" >0.1</td>
      <td id="T_6fb3c_row62_col2" class="data row62 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row62_col3" class="data row62 col3" >0.6</td>
      <td id="T_6fb3c_row62_col4" class="data row62 col4" >0.2243</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row63_col0" class="data row63 col0" >4.0</td>
      <td id="T_6fb3c_row63_col1" class="data row63 col1" >0.2</td>
      <td id="T_6fb3c_row63_col2" class="data row63 col2" >is_debt_high</td>
      <td id="T_6fb3c_row63_col3" class="data row63 col3" >0.2</td>
      <td id="T_6fb3c_row63_col4" class="data row63 col4" >0.0535</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row64_col0" class="data row64 col0" >4.0</td>
      <td id="T_6fb3c_row64_col1" class="data row64 col1" >0.2</td>
      <td id="T_6fb3c_row64_col2" class="data row64 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row64_col3" class="data row64 col3" >-0.2</td>
      <td id="T_6fb3c_row64_col4" class="data row64 col4" >0.0235</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row65_col0" class="data row65 col0" >4.0</td>
      <td id="T_6fb3c_row65_col1" class="data row65 col1" >0.2</td>
      <td id="T_6fb3c_row65_col2" class="data row65 col2" >monthly_debt_log</td>
      <td id="T_6fb3c_row65_col3" class="data row65 col3" >-0.2</td>
      <td id="T_6fb3c_row65_col4" class="data row65 col4" >0.0079</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row66_col0" class="data row66 col0" >4.0</td>
      <td id="T_6fb3c_row66_col1" class="data row66 col1" >0.2</td>
      <td id="T_6fb3c_row66_col2" class="data row66 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row66_col3" class="data row66 col3" >0.6</td>
      <td id="T_6fb3c_row66_col4" class="data row66 col4" >0.112</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row67_col0" class="data row67 col0" >4.0</td>
      <td id="T_6fb3c_row67_col1" class="data row67 col1" >0.2</td>
      <td id="T_6fb3c_row67_col2" class="data row67 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row67_col3" class="data row67 col3" >0.6</td>
      <td id="T_6fb3c_row67_col4" class="data row67 col4" >0.2515</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row68_col0" class="data row68 col0" >4.0</td>
      <td id="T_6fb3c_row68_col1" class="data row68 col1" >0.5</td>
      <td id="T_6fb3c_row68_col2" class="data row68 col2" >is_debt_high</td>
      <td id="T_6fb3c_row68_col3" class="data row68 col3" >-0.2</td>
      <td id="T_6fb3c_row68_col4" class="data row68 col4" >0.0558</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row69_col0" class="data row69 col0" >4.0</td>
      <td id="T_6fb3c_row69_col1" class="data row69 col1" >0.5</td>
      <td id="T_6fb3c_row69_col2" class="data row69 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row69_col3" class="data row69 col3" >0.2</td>
      <td id="T_6fb3c_row69_col4" class="data row69 col4" >0.0307</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row70_col0" class="data row70 col0" >4.0</td>
      <td id="T_6fb3c_row70_col1" class="data row70 col1" >0.5</td>
      <td id="T_6fb3c_row70_col2" class="data row70 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row70_col3" class="data row70 col3" >0.6</td>
      <td id="T_6fb3c_row70_col4" class="data row70 col4" >0.1279</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row71_col0" class="data row71 col0" >4.0</td>
      <td id="T_6fb3c_row71_col1" class="data row71 col1" >0.5</td>
      <td id="T_6fb3c_row71_col2" class="data row71 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row71_col3" class="data row71 col3" >0.6</td>
      <td id="T_6fb3c_row71_col4" class="data row71 col4" >0.2751</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row72_col0" class="data row72 col0" >4.0</td>
      <td id="T_6fb3c_row72_col1" class="data row72 col1" >1.0</td>
      <td id="T_6fb3c_row72_col2" class="data row72 col2" >is_debt_high</td>
      <td id="T_6fb3c_row72_col3" class="data row72 col3" >-0.6</td>
      <td id="T_6fb3c_row72_col4" class="data row72 col4" >0.0629</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row73_col0" class="data row73 col0" >4.0</td>
      <td id="T_6fb3c_row73_col1" class="data row73 col1" >1.0</td>
      <td id="T_6fb3c_row73_col2" class="data row73 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row73_col3" class="data row73 col3" >0.6</td>
      <td id="T_6fb3c_row73_col4" class="data row73 col4" >0.1222</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row74_col0" class="data row74 col0" >4.0</td>
      <td id="T_6fb3c_row74_col1" class="data row74 col1" >1.0</td>
      <td id="T_6fb3c_row74_col2" class="data row74 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row74_col3" class="data row74 col3" >0.2</td>
      <td id="T_6fb3c_row74_col4" class="data row74 col4" >0.2794</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row75_col0" class="data row75 col0" >4.0</td>
      <td id="T_6fb3c_row75_col1" class="data row75 col1" >2.0</td>
      <td id="T_6fb3c_row75_col2" class="data row75 col2" >is_debt_high</td>
      <td id="T_6fb3c_row75_col3" class="data row75 col3" >-0.6</td>
      <td id="T_6fb3c_row75_col4" class="data row75 col4" >0.0717</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row76_col0" class="data row76 col0" >4.0</td>
      <td id="T_6fb3c_row76_col1" class="data row76 col1" >2.0</td>
      <td id="T_6fb3c_row76_col2" class="data row76 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row76_col3" class="data row76 col3" >-0.2</td>
      <td id="T_6fb3c_row76_col4" class="data row76 col4" >0.0259</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row77_col0" class="data row77 col0" >4.0</td>
      <td id="T_6fb3c_row77_col1" class="data row77 col1" >2.0</td>
      <td id="T_6fb3c_row77_col2" class="data row77 col2" >credit_late_density</td>
      <td id="T_6fb3c_row77_col3" class="data row77 col3" >-0.2</td>
      <td id="T_6fb3c_row77_col4" class="data row77 col4" >0.043</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row78_col0" class="data row78 col0" >4.0</td>
      <td id="T_6fb3c_row78_col1" class="data row78 col1" >2.0</td>
      <td id="T_6fb3c_row78_col2" class="data row78 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row78_col3" class="data row78 col3" >0.6</td>
      <td id="T_6fb3c_row78_col4" class="data row78 col4" >0.129</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row79_col0" class="data row79 col0" >4.0</td>
      <td id="T_6fb3c_row79_col1" class="data row79 col1" >2.0</td>
      <td id="T_6fb3c_row79_col2" class="data row79 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row79_col3" class="data row79 col3" >0.2</td>
      <td id="T_6fb3c_row79_col4" class="data row79 col4" >0.2822</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row80_col0" class="data row80 col0" >5.0</td>
      <td id="T_6fb3c_row80_col1" class="data row80 col1" >0.01</td>
      <td id="T_6fb3c_row80_col2" class="data row80 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row80_col3" class="data row80 col3" >-0.2</td>
      <td id="T_6fb3c_row80_col4" class="data row80 col4" >0.0108</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row81_col0" class="data row81 col0" >5.0</td>
      <td id="T_6fb3c_row81_col1" class="data row81 col1" >0.01</td>
      <td id="T_6fb3c_row81_col2" class="data row81 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row81_col3" class="data row81 col3" >0.2</td>
      <td id="T_6fb3c_row81_col4" class="data row81 col4" >0.0317</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row82_col0" class="data row82 col0" >5.0</td>
      <td id="T_6fb3c_row82_col1" class="data row82 col1" >0.025</td>
      <td id="T_6fb3c_row82_col2" class="data row82 col2" >monthly_debt_log</td>
      <td id="T_6fb3c_row82_col3" class="data row82 col3" >0.2</td>
      <td id="T_6fb3c_row82_col4" class="data row82 col4" >0.005</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row83_col0" class="data row83 col0" >5.0</td>
      <td id="T_6fb3c_row83_col1" class="data row83 col1" >0.025</td>
      <td id="T_6fb3c_row83_col2" class="data row83 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row83_col3" class="data row83 col3" >0.6</td>
      <td id="T_6fb3c_row83_col4" class="data row83 col4" >0.0551</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row84_col0" class="data row84 col0" >5.0</td>
      <td id="T_6fb3c_row84_col1" class="data row84 col1" >0.05</td>
      <td id="T_6fb3c_row84_col2" class="data row84 col2" >late_severity_high_x_credit_pressure_low_signal</td>
      <td id="T_6fb3c_row84_col3" class="data row84 col3" >-0.2</td>
      <td id="T_6fb3c_row84_col4" class="data row84 col4" >0.0278</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row85_col0" class="data row85 col0" >5.0</td>
      <td id="T_6fb3c_row85_col1" class="data row85 col1" >0.05</td>
      <td id="T_6fb3c_row85_col2" class="data row85 col2" >monthly_debt_log</td>
      <td id="T_6fb3c_row85_col3" class="data row85 col3" >0.2</td>
      <td id="T_6fb3c_row85_col4" class="data row85 col4" >0.0061</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row86_col0" class="data row86 col0" >5.0</td>
      <td id="T_6fb3c_row86_col1" class="data row86 col1" >0.05</td>
      <td id="T_6fb3c_row86_col2" class="data row86 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row86_col3" class="data row86 col3" >0.6</td>
      <td id="T_6fb3c_row86_col4" class="data row86 col4" >0.075</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row87_col0" class="data row87 col0" >5.0</td>
      <td id="T_6fb3c_row87_col1" class="data row87 col1" >0.05</td>
      <td id="T_6fb3c_row87_col2" class="data row87 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row87_col3" class="data row87 col3" >0.6</td>
      <td id="T_6fb3c_row87_col4" class="data row87 col4" >0.2582</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row88_col0" class="data row88 col0" >5.0</td>
      <td id="T_6fb3c_row88_col1" class="data row88 col1" >0.1</td>
      <td id="T_6fb3c_row88_col2" class="data row88 col2" >monthly_debt_log</td>
      <td id="T_6fb3c_row88_col3" class="data row88 col3" >0.2</td>
      <td id="T_6fb3c_row88_col4" class="data row88 col4" >0.0071</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row89_col0" class="data row89 col0" >5.0</td>
      <td id="T_6fb3c_row89_col1" class="data row89 col1" >0.1</td>
      <td id="T_6fb3c_row89_col2" class="data row89 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row89_col3" class="data row89 col3" >0.6</td>
      <td id="T_6fb3c_row89_col4" class="data row89 col4" >0.0926</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row90_col0" class="data row90 col0" >5.0</td>
      <td id="T_6fb3c_row90_col1" class="data row90 col1" >0.1</td>
      <td id="T_6fb3c_row90_col2" class="data row90 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row90_col3" class="data row90 col3" >0.2</td>
      <td id="T_6fb3c_row90_col4" class="data row90 col4" >0.3074</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row91_col0" class="data row91 col0" >5.0</td>
      <td id="T_6fb3c_row91_col1" class="data row91 col1" >0.2</td>
      <td id="T_6fb3c_row91_col2" class="data row91 col2" >is_debt_high</td>
      <td id="T_6fb3c_row91_col3" class="data row91 col3" >0.2</td>
      <td id="T_6fb3c_row91_col4" class="data row91 col4" >0.068</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row92_col0" class="data row92 col0" >5.0</td>
      <td id="T_6fb3c_row92_col1" class="data row92 col1" >0.2</td>
      <td id="T_6fb3c_row92_col2" class="data row92 col2" >monthly_debt_log</td>
      <td id="T_6fb3c_row92_col3" class="data row92 col3" >0.2</td>
      <td id="T_6fb3c_row92_col4" class="data row92 col4" >0.0058</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row93_col0" class="data row93 col0" >5.0</td>
      <td id="T_6fb3c_row93_col1" class="data row93 col1" >0.2</td>
      <td id="T_6fb3c_row93_col2" class="data row93 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row93_col3" class="data row93 col3" >0.6</td>
      <td id="T_6fb3c_row93_col4" class="data row93 col4" >0.1043</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row94_col0" class="data row94 col0" >5.0</td>
      <td id="T_6fb3c_row94_col1" class="data row94 col1" >0.2</td>
      <td id="T_6fb3c_row94_col2" class="data row94 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row94_col3" class="data row94 col3" >0.2</td>
      <td id="T_6fb3c_row94_col4" class="data row94 col4" >0.3628</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row95_col0" class="data row95 col0" >5.0</td>
      <td id="T_6fb3c_row95_col1" class="data row95 col1" >0.5</td>
      <td id="T_6fb3c_row95_col2" class="data row95 col2" >is_debt_high</td>
      <td id="T_6fb3c_row95_col3" class="data row95 col3" >-0.2</td>
      <td id="T_6fb3c_row95_col4" class="data row95 col4" >0.0546</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row96_col0" class="data row96 col0" >5.0</td>
      <td id="T_6fb3c_row96_col1" class="data row96 col1" >0.5</td>
      <td id="T_6fb3c_row96_col2" class="data row96 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row96_col3" class="data row96 col3" >0.2</td>
      <td id="T_6fb3c_row96_col4" class="data row96 col4" >0.0233</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row97_col0" class="data row97 col0" >5.0</td>
      <td id="T_6fb3c_row97_col1" class="data row97 col1" >0.5</td>
      <td id="T_6fb3c_row97_col2" class="data row97 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row97_col3" class="data row97 col3" >0.6</td>
      <td id="T_6fb3c_row97_col4" class="data row97 col4" >0.1184</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row98_col0" class="data row98 col0" >5.0</td>
      <td id="T_6fb3c_row98_col1" class="data row98 col1" >0.5</td>
      <td id="T_6fb3c_row98_col2" class="data row98 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row98_col3" class="data row98 col3" >0.2</td>
      <td id="T_6fb3c_row98_col4" class="data row98 col4" >0.4108</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row99_col0" class="data row99 col0" >5.0</td>
      <td id="T_6fb3c_row99_col1" class="data row99 col1" >1.0</td>
      <td id="T_6fb3c_row99_col2" class="data row99 col2" >is_debt_high</td>
      <td id="T_6fb3c_row99_col3" class="data row99 col3" >-0.2</td>
      <td id="T_6fb3c_row99_col4" class="data row99 col4" >0.0747</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row100_col0" class="data row100 col0" >5.0</td>
      <td id="T_6fb3c_row100_col1" class="data row100 col1" >1.0</td>
      <td id="T_6fb3c_row100_col2" class="data row100 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row100_col3" class="data row100 col3" >-0.2</td>
      <td id="T_6fb3c_row100_col4" class="data row100 col4" >0.0188</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row101_col0" class="data row101 col0" >5.0</td>
      <td id="T_6fb3c_row101_col1" class="data row101 col1" >1.0</td>
      <td id="T_6fb3c_row101_col2" class="data row101 col2" >30-59late_low_x_credit_late_density_mid_high_signal</td>
      <td id="T_6fb3c_row101_col3" class="data row101 col3" >0.6</td>
      <td id="T_6fb3c_row101_col4" class="data row101 col4" >0.0546</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row102_col0" class="data row102 col0" >5.0</td>
      <td id="T_6fb3c_row102_col1" class="data row102 col1" >1.0</td>
      <td id="T_6fb3c_row102_col2" class="data row102 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row102_col3" class="data row102 col3" >0.6</td>
      <td id="T_6fb3c_row102_col4" class="data row102 col4" >0.1148</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row103_col0" class="data row103 col0" >5.0</td>
      <td id="T_6fb3c_row103_col1" class="data row103 col1" >1.0</td>
      <td id="T_6fb3c_row103_col2" class="data row103 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row103_col3" class="data row103 col3" >0.2</td>
      <td id="T_6fb3c_row103_col4" class="data row103 col4" >0.4248</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row104_col0" class="data row104 col0" >5.0</td>
      <td id="T_6fb3c_row104_col1" class="data row104 col1" >2.0</td>
      <td id="T_6fb3c_row104_col2" class="data row104 col2" >income_per_dep_log</td>
      <td id="T_6fb3c_row104_col3" class="data row104 col3" >0.2</td>
      <td id="T_6fb3c_row104_col4" class="data row104 col4" >0.0177</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row105_col0" class="data row105 col0" >5.0</td>
      <td id="T_6fb3c_row105_col1" class="data row105 col1" >2.0</td>
      <td id="T_6fb3c_row105_col2" class="data row105 col2" >30-59late_low_x_credit_late_density_mid_high_signal</td>
      <td id="T_6fb3c_row105_col3" class="data row105 col3" >0.6</td>
      <td id="T_6fb3c_row105_col4" class="data row105 col4" >0.0552</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row106_col0" class="data row106 col0" >5.0</td>
      <td id="T_6fb3c_row106_col1" class="data row106 col1" >2.0</td>
      <td id="T_6fb3c_row106_col2" class="data row106 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal</td>
      <td id="T_6fb3c_row106_col3" class="data row106 col3" >0.6</td>
      <td id="T_6fb3c_row106_col4" class="data row106 col4" >0.1182</td>
    </tr>
    <tr>
      <td id="T_6fb3c_row107_col0" class="data row107 col0" >5.0</td>
      <td id="T_6fb3c_row107_col1" class="data row107 col1" >2.0</td>
      <td id="T_6fb3c_row107_col2" class="data row107 col2" >late_severity_high_x_short_late_low_signal</td>
      <td id="T_6fb3c_row107_col3" class="data row107 col3" >0.2</td>
      <td id="T_6fb3c_row107_col4" class="data row107 col4" >0.415</td>
    </tr>
  </tbody>
</table>




```python
# 构建不同舍弃指标方案的VIF测试结果表
if is_processing_here3:
    raw_lr_vif_test_df = test_vif_after_drop(X_train_raw_lr, colnames_to_drop_list)
    print_df(raw_lr_vif_test_df)
```


<style type="text/css">
</style>
<table id="T_0ae90">
  <thead>
    <tr>
      <th id="T_0ae90_level0_col0" class="col_heading level0 col0" >drop colnames index</th>
      <th id="T_0ae90_level0_col1" class="col_heading level0 col1" >VIF&gt;20 count</th>
      <th id="T_0ae90_level0_col2" class="col_heading level0 col2" >10&lt;VIF&lt;=20 count</th>
      <th id="T_0ae90_level0_col3" class="col_heading level0 col3" >max VIF</th>
      <th id="T_0ae90_level0_col4" class="col_heading level0 col4" >max VIF colname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_0ae90_row0_col0" class="data row0 col0" >0</td>
      <td id="T_0ae90_row0_col1" class="data row0 col1" >10</td>
      <td id="T_0ae90_row0_col2" class="data row0 col2" >1</td>
      <td id="T_0ae90_row0_col3" class="data row0 col3" >inf</td>
      <td id="T_0ae90_row0_col4" class="data row0 col4" >NumberOfTime30-59DaysPastDueNotWorse</td>
    </tr>
    <tr>
      <td id="T_0ae90_row1_col0" class="data row1 col0" >1</td>
      <td id="T_0ae90_row1_col1" class="data row1 col1" >5</td>
      <td id="T_0ae90_row1_col2" class="data row1 col2" >0</td>
      <td id="T_0ae90_row1_col3" class="data row1 col3" >255.1396</td>
      <td id="T_0ae90_row1_col4" class="data row1 col4" >late_severity_score</td>
    </tr>
    <tr>
      <td id="T_0ae90_row2_col0" class="data row2 col0" >2</td>
      <td id="T_0ae90_row2_col1" class="data row2 col1" >2</td>
      <td id="T_0ae90_row2_col2" class="data row2 col2" >0</td>
      <td id="T_0ae90_row2_col3" class="data row2 col3" >42.78</td>
      <td id="T_0ae90_row2_col4" class="data row2 col4" >monthly_debt_log</td>
    </tr>
    <tr>
      <td id="T_0ae90_row3_col0" class="data row3 col0" >3</td>
      <td id="T_0ae90_row3_col1" class="data row3 col1" >3</td>
      <td id="T_0ae90_row3_col2" class="data row3 col2" >0</td>
      <td id="T_0ae90_row3_col3" class="data row3 col3" >240.9799</td>
      <td id="T_0ae90_row3_col4" class="data row3 col4" >late_severity_score</td>
    </tr>
    <tr>
      <td id="T_0ae90_row4_col0" class="data row4 col0" >4</td>
      <td id="T_0ae90_row4_col1" class="data row4 col1" >0</td>
      <td id="T_0ae90_row4_col2" class="data row4 col2" >2</td>
      <td id="T_0ae90_row4_col3" class="data row4 col3" >13.558</td>
      <td id="T_0ae90_row4_col4" class="data row4 col4" >late_severity_score</td>
    </tr>
  </tbody>
</table>




```python
""" 确定对于raw LR的最佳拟合指标与参数C """
# region
colnames_to_fit_raw_lr = [
        # 原始指标
        'age_centered', 'util_centered', 'debt_centered',
        'income_log', dep, credit, mortgage,
        late30, late60, late90,
        # 数据质量标记
        'single_missing_flag', 'both_missing_flag', 'blacklist_flag',
        'income_anomaly_flag', 'debt_anomaly_flag', 'util_anomaly_flag',
        # 衍生指标
        'free_cashflow_income_log', 'income_per_dep_log', 'monthly_debt_log',
        'credit_pressure_index', 'mortgage_ratio', 'credit_late_density',
        'short_late', 'late_severity_score',
        # 用户行为标记
        'has_no_credit', 'has_short_late_but_no_credit', 'has_serious_late', 'has_short_late',
        'is_debt_high', 'is_debt_overlimit', 'is_util_high', 'is_util_overlimit'
]
c_raw_lr = 1
colnames_to_drop_raw_lr = [
    late30, late60, late90, 
    'income_log', 'free_cashflow_income_log', 'monthly_debt_log'
]
# 检测固定最佳的删除指标后模型拟合效果的变化情况
if is_processing_here3:
    raw_lr_fit_goodness_after_drop_test_df = test_fit_goodness_lr_after_drop(
        X_train_raw_lr, y_train, X_valid_raw_lr, y_valid, 
        colnames_to_fit_raw_lr, c_raw_lr, colnames_to_drop_raw_lr
    )
    print_df(raw_lr_fit_goodness_after_drop_test_df)
# 简化变量
colnames_to_fit_raw_lr = \
    [colname for colname in colnames_to_fit_raw_lr if colname not in colnames_to_drop_raw_lr]
# 释放内存
cleanup_vars([
    'train_data', 'valid_data', 'train_data_copy', 'valid_data_copy',
    'train_data_raw_lr', 'valid_data_raw_lr',
    'X_train_raw_lr', 'X_valid_raw_lr', 'y_train', 'y_valid',
    'folds', 'train_idx_list', 'valid_idx_list',
    'raw_lr_fit_goodness_df_list', 'raw_lr_fit_goodness_df',
    'raw_lr_auc_ks_df', 'raw_lr_coef_sign_stability_df',
    'raw_lr_fit_goodness_after_drop_test_df', 'raw_lr_vif_test_df', 'vif_test_df',
    'centering_means', 'colnames_to_log', 'selected_colnames',
    'colnames_not_binary', 'div_pts',
    'original_features', 'quality_flags', 'derived_features', 'risk_signals',
    'negative_1_5', 'positive_1_5', 'positive_6_10', 'positive_11_15',
    'colnames_to_fit0', 'colnames_to_fit1', 'colnames_to_fit2',
    'colnames_to_fit3', 'colnames_to_fit4', 'colnames_to_fit5',
    'colnames_to_fit_list', 'colnames_to_fit',
    'c_list', 'c',
    'colnames_to_drop0', 'colnames_to_drop1', 'colnames_to_drop2',
    'colnames_to_drop3', 'colnames_to_drop4', 'colnames_to_drop_list',
    'beta30', 'beta60', 'beta90',
], close_plots=False)
# endregion

```


<style type="text/css">
</style>
<table id="T_fa48f">
  <thead>
    <tr>
      <th id="T_fa48f_level0_col0" class="col_heading level0 col0" >status</th>
      <th id="T_fa48f_level0_col1" class="col_heading level0 col1" >valid AUC</th>
      <th id="T_fa48f_level0_col2" class="col_heading level0 col2" >AUC gap</th>
      <th id="T_fa48f_level0_col3" class="col_heading level0 col3" >valid KS</th>
      <th id="T_fa48f_level0_col4" class="col_heading level0 col4" >KS gap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_fa48f_row0_col0" class="data row0 col0" >retained</td>
      <td id="T_fa48f_row0_col1" class="data row0 col1" >0.8639</td>
      <td id="T_fa48f_row0_col2" class="data row0 col2" >-0.0043</td>
      <td id="T_fa48f_row0_col3" class="data row0 col3" >0.5763</td>
      <td id="T_fa48f_row0_col4" class="data row0 col4" >-0.0139</td>
    </tr>
    <tr>
      <td id="T_fa48f_row1_col0" class="data row1 col0" >dropped</td>
      <td id="T_fa48f_row1_col1" class="data row1 col1" >0.8628</td>
      <td id="T_fa48f_row1_col2" class="data row1 col2" >-0.0044</td>
      <td id="T_fa48f_row1_col3" class="data row1 col3" >0.5753</td>
      <td id="T_fa48f_row1_col4" class="data row1 col4" >-0.0161</td>
    </tr>
    <tr>
      <td id="T_fa48f_row2_col0" class="data row2 col0" >difference</td>
      <td id="T_fa48f_row2_col1" class="data row2 col1" >0.001</td>
      <td id="T_fa48f_row2_col2" class="data row2 col2" >0.0001</td>
      <td id="T_fa48f_row2_col3" class="data row2 col3" >0.001</td>
      <td id="T_fa48f_row2_col4" class="data row2 col4" >0.0023</td>
    </tr>
  </tbody>
</table>




```python
""" 对WOE LR模型的调试 """
is_processing_here4 = True

if is_processing_here4:
    # region
    woe_lr_fit_goodness_df_list = []
    # 划分训练集、验证集，准备K=5折交叉验证
    random_seed = 300
    folds = create_kfold(train_valid_data, random_seed, n_splits=5)
    for fold_idx in range(len(folds)):
        # 训练集、验证集拆分
        train_idx_list, valid_idx_list = folds[fold_idx]
        train_data = train_valid_data.iloc[train_idx_list].copy()
        valid_data = train_valid_data.iloc[valid_idx_list].copy()
        train_data_copy = train_data.copy(); valid_data_copy = valid_data.copy() # 备份
    # endregion
        
        """ 统一数据清洗阶段 """
        # region
        # 截断家属数异常大值
        cap_dep_num(train_data); cap_dep_num(valid_data)
        # 删除年龄异常小值
        delete_small_age(train_data); delete_small_age(valid_data)
        # 添加信用额度使用率的若干标记
        add_util_flags(train_data); add_util_flags(valid_data)
        # 添加负债率的若干标记
        add_debt_flags(train_data); add_debt_flags(valid_data)
        # 添加三种逾期次数的加权求和，权重由逻辑回归模型训练
        beta30, beta60, beta90 = add_late_severity_score(train_data)
        add_late_severity_score(valid_data, [beta30, beta60, beta90])
        # 添加衍生指标
        add_derived_features(train_data); add_derived_features(valid_data)
        # 筛选出所有非二元指标
        colnames_not_binary = get_colnames_whether_binary(train_data, is_binary=False)
        # 对所有非布尔值类型的指标建立一维分箱
        for colname in colnames_not_binary:
            _, div_pts = add_bins_1D(train_data, colname)
            add_bins_1D(valid_data, colname, fixed_div_pts=div_pts)
        # 把一维分箱的具体数值转换为排名序号
        convert_bins_value_to_rank(train_data); convert_bins_value_to_rank(valid_data)
        # 添加交互标记
        add_intersection_signals(train_data, positive_interactions_df, negative_interactions_df)
        add_intersection_signals(valid_data, positive_interactions_df, negative_interactions_df)
        # 暂时删除一维分箱列以及其对应的排名序号列
        colnames_without_bins = [colname for colname in train_data.columns if '_bin' not in colname]
        train_data = train_data[colnames_without_bins]; valid_data = valid_data[colnames_without_bins]
        # 整理指标顺序
        train_data = reorder_columns(train_data); valid_data = reorder_columns(valid_data)
        # endregion
        
        """ 对WOE LR建立特化表达 """
        # region
        train_data_woe_lr = train_data.copy(); valid_data_woe_lr = valid_data.copy()
        # 找出所有的非二元列
        colnames_not_binary = get_colnames_whether_binary(train_data_woe_lr, is_binary=False)
        # 对所有非布尔值类型的指标建立一维分箱（虽然构建了理应被淘汰的一维分箱，但后续控制其进入模型训练即可）
        for colname in colnames_not_binary:
            _, div_pts = add_bins_1D(train_data_woe_lr, colname)
            add_bins_1D(valid_data_woe_lr, colname, fixed_div_pts=div_pts)
        # 列出所有一维分箱对应指标的所有无序对
        colname_pairs_not_binary = []
        for i in range(len(colnames_not_binary) - 1):
                for j in range(i + 1, len(colnames_not_binary)):
                    colname_pairs_not_binary.append([colnames_not_binary[i], colnames_not_binary[j]])
        # 只建立通过筛选的二维分箱
        for colname_pair in colname_pairs_not_binary:
            _, bins_2D = add_bins_2D(train_data_woe_lr, colname_pair, constraint=passed_binnames_2D)
            add_bins_2D\
                (valid_data_woe_lr, colname_pair, fixed_bins_2D=bins_2D, constraint=passed_binnames_2D)
        # 把所有标记列、一维分箱列、二维分箱列转换为WOE列
        binnames_map_to_woe = \
            passed_binary_colnames + passed_binnames_1D + low_iv_binnames_1D + passed_binnames_2D
        for colname in binnames_map_to_woe:
            woe_map = fit_woe_mapping(train_data_woe_lr, colname)
            apply_woe_mapping(train_data_woe_lr, colname, woe_map)
            apply_woe_mapping(valid_data_woe_lr, colname, woe_map)
        # 在数据集中只留下投入训练的WOE列
        woe_colnames = [colname for colname in train_data_woe_lr.columns if colname.endswith('_woe')]
        train_data_woe_lr = train_data_woe_lr[[target] + woe_colnames]
        valid_data_woe_lr = valid_data_woe_lr[[target] + woe_colnames]
        # 重排指标列名
        train_data_woe_lr = reorder_columns(train_data_woe_lr)
        valid_data_woe_lr = reorder_columns(valid_data_woe_lr)
        # endregion
        
        """ 训练WOE LR """
        # region
        # 拆分目标变量和自变量
        y_train = train_data[target]; y_valid = valid_data[target]
        X_train_woe_lr = train_data_woe_lr.drop(target, axis=1, inplace=False)
        X_valid_woe_lr = valid_data_woe_lr.drop(target, axis=1, inplace=False)
        # 以下是可能尝试参与拟合的多种指标序列（已补充"_woe"后缀）
        # 原始指标
        original_features = [
            'age_bin_woe', 'util_bin_woe', 'debt_bin_woe',
            'income_bin_woe', 'credit_bin_woe', 'mortgage_bin_woe',
            '30-59late_bin_woe', '60-89late_bin_woe', '90+late_bin_woe'
        ]
        # 衍生指标
        derived_features = [
            'free_cashflow_income_bin_woe', 'income_per_dep_bin_woe',
            'credit_pressure_index_bin_woe', 'credit_late_density_bin_woe',
            'short_late_bin_woe', 'late_severity_score_bin_woe'
        ]
        # 含有较强风险信息的标记
        risk_signals = [
            'has_no_credit_woe', 'has_short_late_but_no_credit_woe', 
            'has_serious_late_woe', 'has_short_late_woe', 'blacklist_flag_woe',
            'is_debt_high_woe', 'is_debt_overlimit_woe', 
            'is_util_high_woe', 'is_util_overlimit_woe'
        ]
        # 二维分箱
        bins_2D = [
            'credit_x_credit_pressure_index_bin_woe',
            'dep_x_credit_bin_woe',
            'credit_x_monthly_debt_bin_woe',
            'debt_x_credit_bin_woe',
            'debt_x_credit_pressure_index_bin_woe',
            'debt_x_monthly_debt_bin_woe',
            'debt_x_mortgage_bin_woe',
            'debt_x_mortgage_ratio_bin_woe',
            'income_x_monthly_debt_bin_woe',
            'monthly_debt_x_credit_pressure_index_bin_woe',
            'mortgage_x_credit_pressure_index_bin_woe',
            'dep_x_mortgage_bin_woe',
            'mortgage_x_monthly_debt_bin_woe'
        ]
        # 低信息价值指标
        low_iv_features = [
            'dep_bin_woe', 'monthly_debt_bin_woe', 'mortgage_ratio_bin_woe'
        ]
        # 增强效应第1-5名
        positive_1_5 = [
            'mortgage_ratio_mid_x_credit_late_density_high_signal_woe',
            'mortgage_mid_x_credit_late_density_high_signal_woe',
            '90+late_mid_high_x_short_late_low_signal_woe',
            'late_severity_high_x_short_late_low_signal_woe',
            'age_high_x_late_severity_high_signal_woe'
        ]
        # 以下是不同的进行拟合的指标组合
        colnames_to_fit0 = original_features
        colnames_to_fit1 = colnames_to_fit0 + derived_features
        colnames_to_fit2 = colnames_to_fit1 + risk_signals
        colnames_to_fit3 = colnames_to_fit2 + bins_2D
        colnames_to_fit4 = colnames_to_fit2 + low_iv_features
        colnames_to_fit5 = colnames_to_fit2 + positive_1_5
        colnames_to_fit_list = [
            colnames_to_fit0, colnames_to_fit1, colnames_to_fit2, 
            colnames_to_fit3, colnames_to_fit4, colnames_to_fit5
        ]
        # 正则化强度的倒数（参数C）的尝试列表
        c_list = [0.01, 0.025, 0.1, 0.2, 0.5, 1, 2]
        # 构建WOE LR拟合效果的数据表
        woe_lr_fit_goodness_df = grid_search_lr\
            (X_train_woe_lr, y_train, X_valid_woe_lr, y_valid, colnames_to_fit_list, c_list)
        woe_lr_fit_goodness_df_list.append(woe_lr_fit_goodness_df)
        # 用最佳的拟合指标去精简数据集
        colnames_to_fit = colnames_to_fit3
        X_train_woe_lr = X_train_woe_lr[colnames_to_fit3]; X_valid_woe_lr = X_valid_woe_lr[colnames_to_fit3]
        c = 0.025
        # 以下是尝试舍弃以改善VIF的指标序列
        colnames_to_drop0 = []
        colnames_to_drop1 = [
            '30-59late_bin_woe', '60-89late_bin_woe', '90+late_bin_woe',
            'income_bin_woe', 'free_cashflow_income_bin_woe'
        ]
        colnames_to_drop2 = colnames_to_drop1 + [
            'short_late_bin_woe',
            'credit_x_credit_pressure_index_bin_woe', 'mortgage_x_credit_pressure_index_bin_woe',
            'monthly_debt_x_credit_pressure_index_bin_woe'
        ]
        colnames_to_drop3 = colnames_to_drop2 + ['has_short_late_woe', 'credit_late_density_bin_woe']
        colnames_to_drop_list = [
            colnames_to_drop0, colnames_to_drop1, colnames_to_drop2, colnames_to_drop3
        ]
        # endregion

```


```python
# 展示对WOE LR建立特化表达完毕后的即将投入训练的列名
if is_processing_here4:
    colname_woe_lr_df = display_current_colnames_woe_lr(train_data, train_data_woe_lr)
    print_df(colname_woe_lr_df)
```


<style type="text/css">
</style>
<table id="T_05271">
  <thead>
    <tr>
      <th id="T_05271_level0_col0" class="col_heading level0 col0" >WOE LR中未构建WOE的标记</th>
      <th id="T_05271_level0_col1" class="col_heading level0 col1" >WOE LR中构建了WOE的二维分箱</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_05271_row0_col0" class="data row0 col0" >both_missing_flag</td>
      <td id="T_05271_row0_col1" class="data row0 col1" >credit_x_credit_pressure_index_bin</td>
    </tr>
    <tr>
      <td id="T_05271_row1_col0" class="data row1 col0" >debt_anomaly_flag</td>
      <td id="T_05271_row1_col1" class="data row1 col1" >credit_x_monthly_debt_bin</td>
    </tr>
    <tr>
      <td id="T_05271_row2_col0" class="data row2 col0" >single_missing_flag</td>
      <td id="T_05271_row2_col1" class="data row2 col1" >debt_x_credit_bin</td>
    </tr>
    <tr>
      <td id="T_05271_row3_col0" class="data row3 col0" >util_anomaly_flag</td>
      <td id="T_05271_row3_col1" class="data row3 col1" >debt_x_credit_pressure_index_bin</td>
    </tr>
    <tr>
      <td id="T_05271_row4_col0" class="data row4 col0" ></td>
      <td id="T_05271_row4_col1" class="data row4 col1" >debt_x_monthly_debt_bin</td>
    </tr>
    <tr>
      <td id="T_05271_row5_col0" class="data row5 col0" ></td>
      <td id="T_05271_row5_col1" class="data row5 col1" >debt_x_mortgage_bin</td>
    </tr>
    <tr>
      <td id="T_05271_row6_col0" class="data row6 col0" ></td>
      <td id="T_05271_row6_col1" class="data row6 col1" >debt_x_mortgage_ratio_bin</td>
    </tr>
    <tr>
      <td id="T_05271_row7_col0" class="data row7 col0" ></td>
      <td id="T_05271_row7_col1" class="data row7 col1" >dep_x_credit_bin</td>
    </tr>
    <tr>
      <td id="T_05271_row8_col0" class="data row8 col0" ></td>
      <td id="T_05271_row8_col1" class="data row8 col1" >dep_x_mortgage_bin</td>
    </tr>
    <tr>
      <td id="T_05271_row9_col0" class="data row9 col0" ></td>
      <td id="T_05271_row9_col1" class="data row9 col1" >income_x_monthly_debt_bin</td>
    </tr>
    <tr>
      <td id="T_05271_row10_col0" class="data row10 col0" ></td>
      <td id="T_05271_row10_col1" class="data row10 col1" >monthly_debt_x_credit_pressure_index_bin</td>
    </tr>
    <tr>
      <td id="T_05271_row11_col0" class="data row11 col0" ></td>
      <td id="T_05271_row11_col1" class="data row11 col1" >mortgage_x_credit_pressure_index_bin</td>
    </tr>
    <tr>
      <td id="T_05271_row12_col0" class="data row12 col0" ></td>
      <td id="T_05271_row12_col1" class="data row12 col1" >mortgage_x_monthly_debt_bin</td>
    </tr>
  </tbody>
</table>




```python
# 网格搜索找出最合适的拟合指标方案和正则化强度的倒数C
if is_processing_here4:
    woe_lr_auc_ks_df, woe_lr_coef_sign_stability_df = test_fit_goodness_lr(woe_lr_fit_goodness_df_list)
    print_df(woe_lr_auc_ks_df)
```


<style type="text/css">
</style>
<table id="T_0d962">
  <thead>
    <tr>
      <th id="T_0d962_level0_col0" class="col_heading level0 col0" >fit columns index</th>
      <th id="T_0d962_level0_col1" class="col_heading level0 col1" >C value</th>
      <th id="T_0d962_level0_col2" class="col_heading level0 col2" >valid_AUC_mean</th>
      <th id="T_0d962_level0_col3" class="col_heading level0 col3" >AUC_gap_mean</th>
      <th id="T_0d962_level0_col4" class="col_heading level0 col4" >valid_KS_mean</th>
      <th id="T_0d962_level0_col5" class="col_heading level0 col5" >KS_gap_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_0d962_row0_col0" class="data row0 col0" >3.0</td>
      <td id="T_0d962_row0_col1" class="data row0 col1" >0.025</td>
      <td id="T_0d962_row0_col2" class="data row0 col2" >0.8601</td>
      <td id="T_0d962_row0_col3" class="data row0 col3" >0.0012</td>
      <td id="T_0d962_row0_col4" class="data row0 col4" >0.568</td>
      <td id="T_0d962_row0_col5" class="data row0 col5" >0.0006</td>
    </tr>
    <tr>
      <td id="T_0d962_row1_col0" class="data row1 col0" >3.0</td>
      <td id="T_0d962_row1_col1" class="data row1 col1" >0.1</td>
      <td id="T_0d962_row1_col2" class="data row1 col2" >0.8601</td>
      <td id="T_0d962_row1_col3" class="data row1 col3" >0.0013</td>
      <td id="T_0d962_row1_col4" class="data row1 col4" >0.5672</td>
      <td id="T_0d962_row1_col5" class="data row1 col5" >0.0009</td>
    </tr>
    <tr>
      <td id="T_0d962_row2_col0" class="data row2 col0" >3.0</td>
      <td id="T_0d962_row2_col1" class="data row2 col1" >0.2</td>
      <td id="T_0d962_row2_col2" class="data row2 col2" >0.86</td>
      <td id="T_0d962_row2_col3" class="data row2 col3" >0.0013</td>
      <td id="T_0d962_row2_col4" class="data row2 col4" >0.567</td>
      <td id="T_0d962_row2_col5" class="data row2 col5" >0.0007</td>
    </tr>
    <tr>
      <td id="T_0d962_row3_col0" class="data row3 col0" >3.0</td>
      <td id="T_0d962_row3_col1" class="data row3 col1" >0.5</td>
      <td id="T_0d962_row3_col2" class="data row3 col2" >0.86</td>
      <td id="T_0d962_row3_col3" class="data row3 col3" >0.0013</td>
      <td id="T_0d962_row3_col4" class="data row3 col4" >0.5669</td>
      <td id="T_0d962_row3_col5" class="data row3 col5" >0.0008</td>
    </tr>
    <tr>
      <td id="T_0d962_row4_col0" class="data row4 col0" >4.0</td>
      <td id="T_0d962_row4_col1" class="data row4 col1" >0.025</td>
      <td id="T_0d962_row4_col2" class="data row4 col2" >0.86</td>
      <td id="T_0d962_row4_col3" class="data row4 col3" >0.0008</td>
      <td id="T_0d962_row4_col4" class="data row4 col4" >0.5682</td>
      <td id="T_0d962_row4_col5" class="data row4 col5" >-0.0015</td>
    </tr>
    <tr>
      <td id="T_0d962_row5_col0" class="data row5 col0" >3.0</td>
      <td id="T_0d962_row5_col1" class="data row5 col1" >0.01</td>
      <td id="T_0d962_row5_col2" class="data row5 col2" >0.8599</td>
      <td id="T_0d962_row5_col3" class="data row5 col3" >0.0011</td>
      <td id="T_0d962_row5_col4" class="data row5 col4" >0.5673</td>
      <td id="T_0d962_row5_col5" class="data row5 col5" >0.0008</td>
    </tr>
    <tr>
      <td id="T_0d962_row6_col0" class="data row6 col0" >4.0</td>
      <td id="T_0d962_row6_col1" class="data row6 col1" >0.1</td>
      <td id="T_0d962_row6_col2" class="data row6 col2" >0.8599</td>
      <td id="T_0d962_row6_col3" class="data row6 col3" >0.0008</td>
      <td id="T_0d962_row6_col4" class="data row6 col4" >0.5684</td>
      <td id="T_0d962_row6_col5" class="data row6 col5" >-0.0014</td>
    </tr>
    <tr>
      <td id="T_0d962_row7_col0" class="data row7 col0" >3.0</td>
      <td id="T_0d962_row7_col1" class="data row7 col1" >2.0</td>
      <td id="T_0d962_row7_col2" class="data row7 col2" >0.8599</td>
      <td id="T_0d962_row7_col3" class="data row7 col3" >0.0014</td>
      <td id="T_0d962_row7_col4" class="data row7 col4" >0.5667</td>
      <td id="T_0d962_row7_col5" class="data row7 col5" >0.0009</td>
    </tr>
    <tr>
      <td id="T_0d962_row8_col0" class="data row8 col0" >3.0</td>
      <td id="T_0d962_row8_col1" class="data row8 col1" >1.0</td>
      <td id="T_0d962_row8_col2" class="data row8 col2" >0.8599</td>
      <td id="T_0d962_row8_col3" class="data row8 col3" >0.0014</td>
      <td id="T_0d962_row8_col4" class="data row8 col4" >0.5668</td>
      <td id="T_0d962_row8_col5" class="data row8 col5" >0.0008</td>
    </tr>
    <tr>
      <td id="T_0d962_row9_col0" class="data row9 col0" >5.0</td>
      <td id="T_0d962_row9_col1" class="data row9 col1" >0.025</td>
      <td id="T_0d962_row9_col2" class="data row9 col2" >0.8599</td>
      <td id="T_0d962_row9_col3" class="data row9 col3" >0.0008</td>
      <td id="T_0d962_row9_col4" class="data row9 col4" >0.5668</td>
      <td id="T_0d962_row9_col5" class="data row9 col5" >-0.0007</td>
    </tr>
    <tr>
      <td id="T_0d962_row10_col0" class="data row10 col0" >4.0</td>
      <td id="T_0d962_row10_col1" class="data row10 col1" >0.5</td>
      <td id="T_0d962_row10_col2" class="data row10 col2" >0.8599</td>
      <td id="T_0d962_row10_col3" class="data row10 col3" >0.0009</td>
      <td id="T_0d962_row10_col4" class="data row10 col4" >0.5676</td>
      <td id="T_0d962_row10_col5" class="data row10 col5" >-0.0005</td>
    </tr>
    <tr>
      <td id="T_0d962_row11_col0" class="data row11 col0" >4.0</td>
      <td id="T_0d962_row11_col1" class="data row11 col1" >1.0</td>
      <td id="T_0d962_row11_col2" class="data row11 col2" >0.8599</td>
      <td id="T_0d962_row11_col3" class="data row11 col3" >0.0008</td>
      <td id="T_0d962_row11_col4" class="data row11 col4" >0.5678</td>
      <td id="T_0d962_row11_col5" class="data row11 col5" >-0.0008</td>
    </tr>
    <tr>
      <td id="T_0d962_row12_col0" class="data row12 col0" >4.0</td>
      <td id="T_0d962_row12_col1" class="data row12 col1" >0.2</td>
      <td id="T_0d962_row12_col2" class="data row12 col2" >0.8599</td>
      <td id="T_0d962_row12_col3" class="data row12 col3" >0.0009</td>
      <td id="T_0d962_row12_col4" class="data row12 col4" >0.5682</td>
      <td id="T_0d962_row12_col5" class="data row12 col5" >-0.0013</td>
    </tr>
    <tr>
      <td id="T_0d962_row13_col0" class="data row13 col0" >4.0</td>
      <td id="T_0d962_row13_col1" class="data row13 col1" >2.0</td>
      <td id="T_0d962_row13_col2" class="data row13 col2" >0.8599</td>
      <td id="T_0d962_row13_col3" class="data row13 col3" >0.0009</td>
      <td id="T_0d962_row13_col4" class="data row13 col4" >0.5677</td>
      <td id="T_0d962_row13_col5" class="data row13 col5" >-0.0007</td>
    </tr>
    <tr>
      <td id="T_0d962_row14_col0" class="data row14 col0" >2.0</td>
      <td id="T_0d962_row14_col1" class="data row14 col1" >0.025</td>
      <td id="T_0d962_row14_col2" class="data row14 col2" >0.8599</td>
      <td id="T_0d962_row14_col3" class="data row14 col3" >0.0007</td>
      <td id="T_0d962_row14_col4" class="data row14 col4" >0.5675</td>
      <td id="T_0d962_row14_col5" class="data row14 col5" >-0.0009</td>
    </tr>
    <tr>
      <td id="T_0d962_row15_col0" class="data row15 col0" >5.0</td>
      <td id="T_0d962_row15_col1" class="data row15 col1" >0.1</td>
      <td id="T_0d962_row15_col2" class="data row15 col2" >0.8599</td>
      <td id="T_0d962_row15_col3" class="data row15 col3" >0.0008</td>
      <td id="T_0d962_row15_col4" class="data row15 col4" >0.5665</td>
      <td id="T_0d962_row15_col5" class="data row15 col5" >-0.0006</td>
    </tr>
    <tr>
      <td id="T_0d962_row16_col0" class="data row16 col0" >2.0</td>
      <td id="T_0d962_row16_col1" class="data row16 col1" >0.1</td>
      <td id="T_0d962_row16_col2" class="data row16 col2" >0.8599</td>
      <td id="T_0d962_row16_col3" class="data row16 col3" >0.0008</td>
      <td id="T_0d962_row16_col4" class="data row16 col4" >0.5673</td>
      <td id="T_0d962_row16_col5" class="data row16 col5" >-0.001</td>
    </tr>
    <tr>
      <td id="T_0d962_row17_col0" class="data row17 col0" >5.0</td>
      <td id="T_0d962_row17_col1" class="data row17 col1" >0.2</td>
      <td id="T_0d962_row17_col2" class="data row17 col2" >0.8599</td>
      <td id="T_0d962_row17_col3" class="data row17 col3" >0.0008</td>
      <td id="T_0d962_row17_col4" class="data row17 col4" >0.5668</td>
      <td id="T_0d962_row17_col5" class="data row17 col5" >-0.0008</td>
    </tr>
    <tr>
      <td id="T_0d962_row18_col0" class="data row18 col0" >2.0</td>
      <td id="T_0d962_row18_col1" class="data row18 col1" >1.0</td>
      <td id="T_0d962_row18_col2" class="data row18 col2" >0.8599</td>
      <td id="T_0d962_row18_col3" class="data row18 col3" >0.0008</td>
      <td id="T_0d962_row18_col4" class="data row18 col4" >0.567</td>
      <td id="T_0d962_row18_col5" class="data row18 col5" >-0.0006</td>
    </tr>
    <tr>
      <td id="T_0d962_row19_col0" class="data row19 col0" >2.0</td>
      <td id="T_0d962_row19_col1" class="data row19 col1" >2.0</td>
      <td id="T_0d962_row19_col2" class="data row19 col2" >0.8599</td>
      <td id="T_0d962_row19_col3" class="data row19 col3" >0.0008</td>
      <td id="T_0d962_row19_col4" class="data row19 col4" >0.5668</td>
      <td id="T_0d962_row19_col5" class="data row19 col5" >-0.0005</td>
    </tr>
    <tr>
      <td id="T_0d962_row20_col0" class="data row20 col0" >2.0</td>
      <td id="T_0d962_row20_col1" class="data row20 col1" >0.2</td>
      <td id="T_0d962_row20_col2" class="data row20 col2" >0.8599</td>
      <td id="T_0d962_row20_col3" class="data row20 col3" >0.0008</td>
      <td id="T_0d962_row20_col4" class="data row20 col4" >0.567</td>
      <td id="T_0d962_row20_col5" class="data row20 col5" >-0.0008</td>
    </tr>
    <tr>
      <td id="T_0d962_row21_col0" class="data row21 col0" >5.0</td>
      <td id="T_0d962_row21_col1" class="data row21 col1" >0.5</td>
      <td id="T_0d962_row21_col2" class="data row21 col2" >0.8598</td>
      <td id="T_0d962_row21_col3" class="data row21 col3" >0.0008</td>
      <td id="T_0d962_row21_col4" class="data row21 col4" >0.5668</td>
      <td id="T_0d962_row21_col5" class="data row21 col5" >-0.0009</td>
    </tr>
    <tr>
      <td id="T_0d962_row22_col0" class="data row22 col0" >5.0</td>
      <td id="T_0d962_row22_col1" class="data row22 col1" >2.0</td>
      <td id="T_0d962_row22_col2" class="data row22 col2" >0.8598</td>
      <td id="T_0d962_row22_col3" class="data row22 col3" >0.0008</td>
      <td id="T_0d962_row22_col4" class="data row22 col4" >0.5666</td>
      <td id="T_0d962_row22_col5" class="data row22 col5" >-0.0007</td>
    </tr>
    <tr>
      <td id="T_0d962_row23_col0" class="data row23 col0" >5.0</td>
      <td id="T_0d962_row23_col1" class="data row23 col1" >1.0</td>
      <td id="T_0d962_row23_col2" class="data row23 col2" >0.8598</td>
      <td id="T_0d962_row23_col3" class="data row23 col3" >0.0008</td>
      <td id="T_0d962_row23_col4" class="data row23 col4" >0.5667</td>
      <td id="T_0d962_row23_col5" class="data row23 col5" >-0.0008</td>
    </tr>
    <tr>
      <td id="T_0d962_row24_col0" class="data row24 col0" >2.0</td>
      <td id="T_0d962_row24_col1" class="data row24 col1" >0.5</td>
      <td id="T_0d962_row24_col2" class="data row24 col2" >0.8598</td>
      <td id="T_0d962_row24_col3" class="data row24 col3" >0.0008</td>
      <td id="T_0d962_row24_col4" class="data row24 col4" >0.567</td>
      <td id="T_0d962_row24_col5" class="data row24 col5" >-0.0006</td>
    </tr>
    <tr>
      <td id="T_0d962_row25_col0" class="data row25 col0" >5.0</td>
      <td id="T_0d962_row25_col1" class="data row25 col1" >0.01</td>
      <td id="T_0d962_row25_col2" class="data row25 col2" >0.8598</td>
      <td id="T_0d962_row25_col3" class="data row25 col3" >0.0007</td>
      <td id="T_0d962_row25_col4" class="data row25 col4" >0.5659</td>
      <td id="T_0d962_row25_col5" class="data row25 col5" >-0.0001</td>
    </tr>
    <tr>
      <td id="T_0d962_row26_col0" class="data row26 col0" >4.0</td>
      <td id="T_0d962_row26_col1" class="data row26 col1" >0.01</td>
      <td id="T_0d962_row26_col2" class="data row26 col2" >0.8598</td>
      <td id="T_0d962_row26_col3" class="data row26 col3" >0.0007</td>
      <td id="T_0d962_row26_col4" class="data row26 col4" >0.5664</td>
      <td id="T_0d962_row26_col5" class="data row26 col5" >-0.0006</td>
    </tr>
    <tr>
      <td id="T_0d962_row27_col0" class="data row27 col0" >2.0</td>
      <td id="T_0d962_row27_col1" class="data row27 col1" >0.01</td>
      <td id="T_0d962_row27_col2" class="data row27 col2" >0.8597</td>
      <td id="T_0d962_row27_col3" class="data row27 col3" >0.0007</td>
      <td id="T_0d962_row27_col4" class="data row27 col4" >0.5667</td>
      <td id="T_0d962_row27_col5" class="data row27 col5" >-0.0007</td>
    </tr>
    <tr>
      <td id="T_0d962_row28_col0" class="data row28 col0" >1.0</td>
      <td id="T_0d962_row28_col1" class="data row28 col1" >0.2</td>
      <td id="T_0d962_row28_col2" class="data row28 col2" >0.8588</td>
      <td id="T_0d962_row28_col3" class="data row28 col3" >0.0007</td>
      <td id="T_0d962_row28_col4" class="data row28 col4" >0.5668</td>
      <td id="T_0d962_row28_col5" class="data row28 col5" >-0.0005</td>
    </tr>
    <tr>
      <td id="T_0d962_row29_col0" class="data row29 col0" >1.0</td>
      <td id="T_0d962_row29_col1" class="data row29 col1" >0.1</td>
      <td id="T_0d962_row29_col2" class="data row29 col2" >0.8588</td>
      <td id="T_0d962_row29_col3" class="data row29 col3" >0.0007</td>
      <td id="T_0d962_row29_col4" class="data row29 col4" >0.5664</td>
      <td id="T_0d962_row29_col5" class="data row29 col5" >-0.0001</td>
    </tr>
    <tr>
      <td id="T_0d962_row30_col0" class="data row30 col0" >1.0</td>
      <td id="T_0d962_row30_col1" class="data row30 col1" >0.025</td>
      <td id="T_0d962_row30_col2" class="data row30 col2" >0.8588</td>
      <td id="T_0d962_row30_col3" class="data row30 col3" >0.0007</td>
      <td id="T_0d962_row30_col4" class="data row30 col4" >0.5662</td>
      <td id="T_0d962_row30_col5" class="data row30 col5" >-0.0006</td>
    </tr>
    <tr>
      <td id="T_0d962_row31_col0" class="data row31 col0" >1.0</td>
      <td id="T_0d962_row31_col1" class="data row31 col1" >1.0</td>
      <td id="T_0d962_row31_col2" class="data row31 col2" >0.8588</td>
      <td id="T_0d962_row31_col3" class="data row31 col3" >0.0008</td>
      <td id="T_0d962_row31_col4" class="data row31 col4" >0.567</td>
      <td id="T_0d962_row31_col5" class="data row31 col5" >-0.0005</td>
    </tr>
    <tr>
      <td id="T_0d962_row32_col0" class="data row32 col0" >1.0</td>
      <td id="T_0d962_row32_col1" class="data row32 col1" >0.5</td>
      <td id="T_0d962_row32_col2" class="data row32 col2" >0.8588</td>
      <td id="T_0d962_row32_col3" class="data row32 col3" >0.0008</td>
      <td id="T_0d962_row32_col4" class="data row32 col4" >0.5667</td>
      <td id="T_0d962_row32_col5" class="data row32 col5" >-0.0003</td>
    </tr>
    <tr>
      <td id="T_0d962_row33_col0" class="data row33 col0" >1.0</td>
      <td id="T_0d962_row33_col1" class="data row33 col1" >2.0</td>
      <td id="T_0d962_row33_col2" class="data row33 col2" >0.8588</td>
      <td id="T_0d962_row33_col3" class="data row33 col3" >0.0008</td>
      <td id="T_0d962_row33_col4" class="data row33 col4" >0.567</td>
      <td id="T_0d962_row33_col5" class="data row33 col5" >-0.0005</td>
    </tr>
    <tr>
      <td id="T_0d962_row34_col0" class="data row34 col0" >1.0</td>
      <td id="T_0d962_row34_col1" class="data row34 col1" >0.01</td>
      <td id="T_0d962_row34_col2" class="data row34 col2" >0.8586</td>
      <td id="T_0d962_row34_col3" class="data row34 col3" >0.0006</td>
      <td id="T_0d962_row34_col4" class="data row34 col4" >0.5657</td>
      <td id="T_0d962_row34_col5" class="data row34 col5" >-0.0002</td>
    </tr>
    <tr>
      <td id="T_0d962_row35_col0" class="data row35 col0" >0.0</td>
      <td id="T_0d962_row35_col1" class="data row35 col1" >0.025</td>
      <td id="T_0d962_row35_col2" class="data row35 col2" >0.8562</td>
      <td id="T_0d962_row35_col3" class="data row35 col3" >0.0006</td>
      <td id="T_0d962_row35_col4" class="data row35 col4" >0.5594</td>
      <td id="T_0d962_row35_col5" class="data row35 col5" >-0.0015</td>
    </tr>
    <tr>
      <td id="T_0d962_row36_col0" class="data row36 col0" >0.0</td>
      <td id="T_0d962_row36_col1" class="data row36 col1" >0.1</td>
      <td id="T_0d962_row36_col2" class="data row36 col2" >0.8562</td>
      <td id="T_0d962_row36_col3" class="data row36 col3" >0.0006</td>
      <td id="T_0d962_row36_col4" class="data row36 col4" >0.5593</td>
      <td id="T_0d962_row36_col5" class="data row36 col5" >-0.0009</td>
    </tr>
    <tr>
      <td id="T_0d962_row37_col0" class="data row37 col0" >0.0</td>
      <td id="T_0d962_row37_col1" class="data row37 col1" >0.2</td>
      <td id="T_0d962_row37_col2" class="data row37 col2" >0.8561</td>
      <td id="T_0d962_row37_col3" class="data row37 col3" >0.0006</td>
      <td id="T_0d962_row37_col4" class="data row37 col4" >0.5595</td>
      <td id="T_0d962_row37_col5" class="data row37 col5" >-0.0011</td>
    </tr>
    <tr>
      <td id="T_0d962_row38_col0" class="data row38 col0" >0.0</td>
      <td id="T_0d962_row38_col1" class="data row38 col1" >0.01</td>
      <td id="T_0d962_row38_col2" class="data row38 col2" >0.8561</td>
      <td id="T_0d962_row38_col3" class="data row38 col3" >0.0005</td>
      <td id="T_0d962_row38_col4" class="data row38 col4" >0.5583</td>
      <td id="T_0d962_row38_col5" class="data row38 col5" >-0.0021</td>
    </tr>
    <tr>
      <td id="T_0d962_row39_col0" class="data row39 col0" >0.0</td>
      <td id="T_0d962_row39_col1" class="data row39 col1" >0.5</td>
      <td id="T_0d962_row39_col2" class="data row39 col2" >0.8561</td>
      <td id="T_0d962_row39_col3" class="data row39 col3" >0.0006</td>
      <td id="T_0d962_row39_col4" class="data row39 col4" >0.5593</td>
      <td id="T_0d962_row39_col5" class="data row39 col5" >-0.0008</td>
    </tr>
    <tr>
      <td id="T_0d962_row40_col0" class="data row40 col0" >0.0</td>
      <td id="T_0d962_row40_col1" class="data row40 col1" >1.0</td>
      <td id="T_0d962_row40_col2" class="data row40 col2" >0.8561</td>
      <td id="T_0d962_row40_col3" class="data row40 col3" >0.0006</td>
      <td id="T_0d962_row40_col4" class="data row40 col4" >0.5594</td>
      <td id="T_0d962_row40_col5" class="data row40 col5" >-0.001</td>
    </tr>
    <tr>
      <td id="T_0d962_row41_col0" class="data row41 col0" >0.0</td>
      <td id="T_0d962_row41_col1" class="data row41 col1" >2.0</td>
      <td id="T_0d962_row41_col2" class="data row41 col2" >0.8561</td>
      <td id="T_0d962_row41_col3" class="data row41 col3" >0.0006</td>
      <td id="T_0d962_row41_col4" class="data row41 col4" >0.5596</td>
      <td id="T_0d962_row41_col5" class="data row41 col5" >-0.0011</td>
    </tr>
  </tbody>
</table>




```python
if is_processing_here4:
    print_df(woe_lr_coef_sign_stability_df)
```


<style type="text/css">
</style>
<table id="T_40d0f">
  <thead>
    <tr>
      <th id="T_40d0f_level0_col0" class="col_heading level0 col0" >fit columns index</th>
      <th id="T_40d0f_level0_col1" class="col_heading level0 col1" >C value</th>
      <th id="T_40d0f_level0_col2" class="col_heading level0 col2" >colname</th>
      <th id="T_40d0f_level0_col3" class="col_heading level0 col3" >sign stabilty</th>
      <th id="T_40d0f_level0_col4" class="col_heading level0 col4" >std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_40d0f_row0_col0" class="data row0 col0" >1.0</td>
      <td id="T_40d0f_row0_col1" class="data row0 col1" >0.1</td>
      <td id="T_40d0f_row0_col2" class="data row0 col2" >credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row0_col3" class="data row0 col3" >0.2</td>
      <td id="T_40d0f_row0_col4" class="data row0 col4" >0.0195</td>
    </tr>
    <tr>
      <td id="T_40d0f_row1_col0" class="data row1 col0" >1.0</td>
      <td id="T_40d0f_row1_col1" class="data row1 col1" >0.1</td>
      <td id="T_40d0f_row1_col2" class="data row1 col2" >credit_bin_woe</td>
      <td id="T_40d0f_row1_col3" class="data row1 col3" >-0.2</td>
      <td id="T_40d0f_row1_col4" class="data row1 col4" >0.0281</td>
    </tr>
    <tr>
      <td id="T_40d0f_row2_col0" class="data row2 col0" >1.0</td>
      <td id="T_40d0f_row2_col1" class="data row2 col1" >0.2</td>
      <td id="T_40d0f_row2_col2" class="data row2 col2" >credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row2_col3" class="data row2 col3" >0.2</td>
      <td id="T_40d0f_row2_col4" class="data row2 col4" >0.0178</td>
    </tr>
    <tr>
      <td id="T_40d0f_row3_col0" class="data row3 col0" >1.0</td>
      <td id="T_40d0f_row3_col1" class="data row3 col1" >0.2</td>
      <td id="T_40d0f_row3_col2" class="data row3 col2" >credit_bin_woe</td>
      <td id="T_40d0f_row3_col3" class="data row3 col3" >-0.2</td>
      <td id="T_40d0f_row3_col4" class="data row3 col4" >0.0227</td>
    </tr>
    <tr>
      <td id="T_40d0f_row4_col0" class="data row4 col0" >1.0</td>
      <td id="T_40d0f_row4_col1" class="data row4 col1" >0.5</td>
      <td id="T_40d0f_row4_col2" class="data row4 col2" >credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row4_col3" class="data row4 col3" >0.2</td>
      <td id="T_40d0f_row4_col4" class="data row4 col4" >0.0172</td>
    </tr>
    <tr>
      <td id="T_40d0f_row5_col0" class="data row5 col0" >1.0</td>
      <td id="T_40d0f_row5_col1" class="data row5 col1" >0.5</td>
      <td id="T_40d0f_row5_col2" class="data row5 col2" >credit_bin_woe</td>
      <td id="T_40d0f_row5_col3" class="data row5 col3" >-0.2</td>
      <td id="T_40d0f_row5_col4" class="data row5 col4" >0.0266</td>
    </tr>
    <tr>
      <td id="T_40d0f_row6_col0" class="data row6 col0" >1.0</td>
      <td id="T_40d0f_row6_col1" class="data row6 col1" >1.0</td>
      <td id="T_40d0f_row6_col2" class="data row6 col2" >credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row6_col3" class="data row6 col3" >0.2</td>
      <td id="T_40d0f_row6_col4" class="data row6 col4" >0.0173</td>
    </tr>
    <tr>
      <td id="T_40d0f_row7_col0" class="data row7 col0" >1.0</td>
      <td id="T_40d0f_row7_col1" class="data row7 col1" >1.0</td>
      <td id="T_40d0f_row7_col2" class="data row7 col2" >credit_bin_woe</td>
      <td id="T_40d0f_row7_col3" class="data row7 col3" >-0.2</td>
      <td id="T_40d0f_row7_col4" class="data row7 col4" >0.0264</td>
    </tr>
    <tr>
      <td id="T_40d0f_row8_col0" class="data row8 col0" >1.0</td>
      <td id="T_40d0f_row8_col1" class="data row8 col1" >2.0</td>
      <td id="T_40d0f_row8_col2" class="data row8 col2" >credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row8_col3" class="data row8 col3" >0.2</td>
      <td id="T_40d0f_row8_col4" class="data row8 col4" >0.0173</td>
    </tr>
    <tr>
      <td id="T_40d0f_row9_col0" class="data row9 col0" >1.0</td>
      <td id="T_40d0f_row9_col1" class="data row9 col1" >2.0</td>
      <td id="T_40d0f_row9_col2" class="data row9 col2" >credit_bin_woe</td>
      <td id="T_40d0f_row9_col3" class="data row9 col3" >-0.2</td>
      <td id="T_40d0f_row9_col4" class="data row9 col4" >0.0264</td>
    </tr>
    <tr>
      <td id="T_40d0f_row10_col0" class="data row10 col0" >2.0</td>
      <td id="T_40d0f_row10_col1" class="data row10 col1" >0.01</td>
      <td id="T_40d0f_row10_col2" class="data row10 col2" >is_util_high_woe</td>
      <td id="T_40d0f_row10_col3" class="data row10 col3" >-0.2</td>
      <td id="T_40d0f_row10_col4" class="data row10 col4" >0.0145</td>
    </tr>
    <tr>
      <td id="T_40d0f_row11_col0" class="data row11 col0" >2.0</td>
      <td id="T_40d0f_row11_col1" class="data row11 col1" >0.1</td>
      <td id="T_40d0f_row11_col2" class="data row11 col2" >credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row11_col3" class="data row11 col3" >0.2</td>
      <td id="T_40d0f_row11_col4" class="data row11 col4" >0.0121</td>
    </tr>
    <tr>
      <td id="T_40d0f_row12_col0" class="data row12 col0" >2.0</td>
      <td id="T_40d0f_row12_col1" class="data row12 col1" >0.2</td>
      <td id="T_40d0f_row12_col2" class="data row12 col2" >credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row12_col3" class="data row12 col3" >0.2</td>
      <td id="T_40d0f_row12_col4" class="data row12 col4" >0.013</td>
    </tr>
    <tr>
      <td id="T_40d0f_row13_col0" class="data row13 col0" >3.0</td>
      <td id="T_40d0f_row13_col1" class="data row13 col1" >0.01</td>
      <td id="T_40d0f_row13_col2" class="data row13 col2" >is_util_high_woe</td>
      <td id="T_40d0f_row13_col3" class="data row13 col3" >0.2</td>
      <td id="T_40d0f_row13_col4" class="data row13 col4" >0.0121</td>
    </tr>
    <tr>
      <td id="T_40d0f_row14_col0" class="data row14 col0" >3.0</td>
      <td id="T_40d0f_row14_col1" class="data row14 col1" >0.01</td>
      <td id="T_40d0f_row14_col2" class="data row14 col2" >credit_bin_woe</td>
      <td id="T_40d0f_row14_col3" class="data row14 col3" >0.2</td>
      <td id="T_40d0f_row14_col4" class="data row14 col4" >0.0181</td>
    </tr>
    <tr>
      <td id="T_40d0f_row15_col0" class="data row15 col0" >3.0</td>
      <td id="T_40d0f_row15_col1" class="data row15 col1" >0.01</td>
      <td id="T_40d0f_row15_col2" class="data row15 col2" >debt_x_mortgage_bin_woe</td>
      <td id="T_40d0f_row15_col3" class="data row15 col3" >-0.2</td>
      <td id="T_40d0f_row15_col4" class="data row15 col4" >0.0277</td>
    </tr>
    <tr>
      <td id="T_40d0f_row16_col0" class="data row16 col0" >3.0</td>
      <td id="T_40d0f_row16_col1" class="data row16 col1" >0.025</td>
      <td id="T_40d0f_row16_col2" class="data row16 col2" >credit_bin_woe</td>
      <td id="T_40d0f_row16_col3" class="data row16 col3" >-0.2</td>
      <td id="T_40d0f_row16_col4" class="data row16 col4" >0.0282</td>
    </tr>
    <tr>
      <td id="T_40d0f_row17_col0" class="data row17 col0" >3.0</td>
      <td id="T_40d0f_row17_col1" class="data row17 col1" >0.025</td>
      <td id="T_40d0f_row17_col2" class="data row17 col2" >debt_x_mortgage_bin_woe</td>
      <td id="T_40d0f_row17_col3" class="data row17 col3" >0.2</td>
      <td id="T_40d0f_row17_col4" class="data row17 col4" >0.041</td>
    </tr>
    <tr>
      <td id="T_40d0f_row18_col0" class="data row18 col0" >3.0</td>
      <td id="T_40d0f_row18_col1" class="data row18 col1" >0.025</td>
      <td id="T_40d0f_row18_col2" class="data row18 col2" >debt_x_credit_bin_woe</td>
      <td id="T_40d0f_row18_col3" class="data row18 col3" >-0.2</td>
      <td id="T_40d0f_row18_col4" class="data row18 col4" >0.0212</td>
    </tr>
    <tr>
      <td id="T_40d0f_row19_col0" class="data row19 col0" >3.0</td>
      <td id="T_40d0f_row19_col1" class="data row19 col1" >0.1</td>
      <td id="T_40d0f_row19_col2" class="data row19 col2" >credit_x_monthly_debt_bin_woe</td>
      <td id="T_40d0f_row19_col3" class="data row19 col3" >0.6</td>
      <td id="T_40d0f_row19_col4" class="data row19 col4" >0.0599</td>
    </tr>
    <tr>
      <td id="T_40d0f_row20_col0" class="data row20 col0" >3.0</td>
      <td id="T_40d0f_row20_col1" class="data row20 col1" >0.1</td>
      <td id="T_40d0f_row20_col2" class="data row20 col2" >mortgage_x_monthly_debt_bin_woe</td>
      <td id="T_40d0f_row20_col3" class="data row20 col3" >0.2</td>
      <td id="T_40d0f_row20_col4" class="data row20 col4" >0.03</td>
    </tr>
    <tr>
      <td id="T_40d0f_row21_col0" class="data row21 col0" >3.0</td>
      <td id="T_40d0f_row21_col1" class="data row21 col1" >0.1</td>
      <td id="T_40d0f_row21_col2" class="data row21 col2" >credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row21_col3" class="data row21 col3" >0.2</td>
      <td id="T_40d0f_row21_col4" class="data row21 col4" >0.0271</td>
    </tr>
    <tr>
      <td id="T_40d0f_row22_col0" class="data row22 col0" >3.0</td>
      <td id="T_40d0f_row22_col1" class="data row22 col1" >0.1</td>
      <td id="T_40d0f_row22_col2" class="data row22 col2" >mortgage_x_credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row22_col3" class="data row22 col3" >-0.6</td>
      <td id="T_40d0f_row22_col4" class="data row22 col4" >0.0649</td>
    </tr>
    <tr>
      <td id="T_40d0f_row23_col0" class="data row23 col0" >3.0</td>
      <td id="T_40d0f_row23_col1" class="data row23 col1" >0.1</td>
      <td id="T_40d0f_row23_col2" class="data row23 col2" >credit_bin_woe</td>
      <td id="T_40d0f_row23_col3" class="data row23 col3" >-0.2</td>
      <td id="T_40d0f_row23_col4" class="data row23 col4" >0.0453</td>
    </tr>
    <tr>
      <td id="T_40d0f_row24_col0" class="data row24 col0" >3.0</td>
      <td id="T_40d0f_row24_col1" class="data row24 col1" >0.1</td>
      <td id="T_40d0f_row24_col2" class="data row24 col2" >debt_x_mortgage_bin_woe</td>
      <td id="T_40d0f_row24_col3" class="data row24 col3" >0.2</td>
      <td id="T_40d0f_row24_col4" class="data row24 col4" >0.0546</td>
    </tr>
    <tr>
      <td id="T_40d0f_row25_col0" class="data row25 col0" >3.0</td>
      <td id="T_40d0f_row25_col1" class="data row25 col1" >0.2</td>
      <td id="T_40d0f_row25_col2" class="data row25 col2" >credit_x_monthly_debt_bin_woe</td>
      <td id="T_40d0f_row25_col3" class="data row25 col3" >0.6</td>
      <td id="T_40d0f_row25_col4" class="data row25 col4" >0.0602</td>
    </tr>
    <tr>
      <td id="T_40d0f_row26_col0" class="data row26 col0" >3.0</td>
      <td id="T_40d0f_row26_col1" class="data row26 col1" >0.2</td>
      <td id="T_40d0f_row26_col2" class="data row26 col2" >credit_x_credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row26_col3" class="data row26 col3" >0.2</td>
      <td id="T_40d0f_row26_col4" class="data row26 col4" >0.037</td>
    </tr>
    <tr>
      <td id="T_40d0f_row27_col0" class="data row27 col0" >3.0</td>
      <td id="T_40d0f_row27_col1" class="data row27 col1" >0.2</td>
      <td id="T_40d0f_row27_col2" class="data row27 col2" >mortgage_x_monthly_debt_bin_woe</td>
      <td id="T_40d0f_row27_col3" class="data row27 col3" >0.2</td>
      <td id="T_40d0f_row27_col4" class="data row27 col4" >0.0328</td>
    </tr>
    <tr>
      <td id="T_40d0f_row28_col0" class="data row28 col0" >3.0</td>
      <td id="T_40d0f_row28_col1" class="data row28 col1" >0.2</td>
      <td id="T_40d0f_row28_col2" class="data row28 col2" >debt_x_mortgage_ratio_bin_woe</td>
      <td id="T_40d0f_row28_col3" class="data row28 col3" >-0.2</td>
      <td id="T_40d0f_row28_col4" class="data row28 col4" >0.0391</td>
    </tr>
    <tr>
      <td id="T_40d0f_row29_col0" class="data row29 col0" >3.0</td>
      <td id="T_40d0f_row29_col1" class="data row29 col1" >0.2</td>
      <td id="T_40d0f_row29_col2" class="data row29 col2" >mortgage_x_credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row29_col3" class="data row29 col3" >-0.6</td>
      <td id="T_40d0f_row29_col4" class="data row29 col4" >0.0687</td>
    </tr>
    <tr>
      <td id="T_40d0f_row30_col0" class="data row30 col0" >3.0</td>
      <td id="T_40d0f_row30_col1" class="data row30 col1" >0.2</td>
      <td id="T_40d0f_row30_col2" class="data row30 col2" >credit_bin_woe</td>
      <td id="T_40d0f_row30_col3" class="data row30 col3" >-0.2</td>
      <td id="T_40d0f_row30_col4" class="data row30 col4" >0.0562</td>
    </tr>
    <tr>
      <td id="T_40d0f_row31_col0" class="data row31 col0" >3.0</td>
      <td id="T_40d0f_row31_col1" class="data row31 col1" >0.2</td>
      <td id="T_40d0f_row31_col2" class="data row31 col2" >debt_x_mortgage_bin_woe</td>
      <td id="T_40d0f_row31_col3" class="data row31 col3" >0.2</td>
      <td id="T_40d0f_row31_col4" class="data row31 col4" >0.0538</td>
    </tr>
    <tr>
      <td id="T_40d0f_row32_col0" class="data row32 col0" >3.0</td>
      <td id="T_40d0f_row32_col1" class="data row32 col1" >0.5</td>
      <td id="T_40d0f_row32_col2" class="data row32 col2" >credit_x_monthly_debt_bin_woe</td>
      <td id="T_40d0f_row32_col3" class="data row32 col3" >0.6</td>
      <td id="T_40d0f_row32_col4" class="data row32 col4" >0.0731</td>
    </tr>
    <tr>
      <td id="T_40d0f_row33_col0" class="data row33 col0" >3.0</td>
      <td id="T_40d0f_row33_col1" class="data row33 col1" >0.5</td>
      <td id="T_40d0f_row33_col2" class="data row33 col2" >credit_x_credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row33_col3" class="data row33 col3" >0.2</td>
      <td id="T_40d0f_row33_col4" class="data row33 col4" >0.0365</td>
    </tr>
    <tr>
      <td id="T_40d0f_row34_col0" class="data row34 col0" >3.0</td>
      <td id="T_40d0f_row34_col1" class="data row34 col1" >0.5</td>
      <td id="T_40d0f_row34_col2" class="data row34 col2" >credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row34_col3" class="data row34 col3" >0.2</td>
      <td id="T_40d0f_row34_col4" class="data row34 col4" >0.0331</td>
    </tr>
    <tr>
      <td id="T_40d0f_row35_col0" class="data row35 col0" >3.0</td>
      <td id="T_40d0f_row35_col1" class="data row35 col1" >0.5</td>
      <td id="T_40d0f_row35_col2" class="data row35 col2" >debt_x_mortgage_ratio_bin_woe</td>
      <td id="T_40d0f_row35_col3" class="data row35 col3" >-0.2</td>
      <td id="T_40d0f_row35_col4" class="data row35 col4" >0.0431</td>
    </tr>
    <tr>
      <td id="T_40d0f_row36_col0" class="data row36 col0" >3.0</td>
      <td id="T_40d0f_row36_col1" class="data row36 col1" >0.5</td>
      <td id="T_40d0f_row36_col2" class="data row36 col2" >mortgage_x_credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row36_col3" class="data row36 col3" >-0.6</td>
      <td id="T_40d0f_row36_col4" class="data row36 col4" >0.0704</td>
    </tr>
    <tr>
      <td id="T_40d0f_row37_col0" class="data row37 col0" >3.0</td>
      <td id="T_40d0f_row37_col1" class="data row37 col1" >0.5</td>
      <td id="T_40d0f_row37_col2" class="data row37 col2" >credit_bin_woe</td>
      <td id="T_40d0f_row37_col3" class="data row37 col3" >0.2</td>
      <td id="T_40d0f_row37_col4" class="data row37 col4" >0.064</td>
    </tr>
    <tr>
      <td id="T_40d0f_row38_col0" class="data row38 col0" >3.0</td>
      <td id="T_40d0f_row38_col1" class="data row38 col1" >0.5</td>
      <td id="T_40d0f_row38_col2" class="data row38 col2" >income_per_dep_bin_woe</td>
      <td id="T_40d0f_row38_col3" class="data row38 col3" >0.6</td>
      <td id="T_40d0f_row38_col4" class="data row38 col4" >0.0794</td>
    </tr>
    <tr>
      <td id="T_40d0f_row39_col0" class="data row39 col0" >3.0</td>
      <td id="T_40d0f_row39_col1" class="data row39 col1" >0.5</td>
      <td id="T_40d0f_row39_col2" class="data row39 col2" >debt_x_mortgage_bin_woe</td>
      <td id="T_40d0f_row39_col3" class="data row39 col3" >0.2</td>
      <td id="T_40d0f_row39_col4" class="data row39 col4" >0.0551</td>
    </tr>
    <tr>
      <td id="T_40d0f_row40_col0" class="data row40 col0" >3.0</td>
      <td id="T_40d0f_row40_col1" class="data row40 col1" >1.0</td>
      <td id="T_40d0f_row40_col2" class="data row40 col2" >credit_x_monthly_debt_bin_woe</td>
      <td id="T_40d0f_row40_col3" class="data row40 col3" >0.6</td>
      <td id="T_40d0f_row40_col4" class="data row40 col4" >0.0727</td>
    </tr>
    <tr>
      <td id="T_40d0f_row41_col0" class="data row41 col0" >3.0</td>
      <td id="T_40d0f_row41_col1" class="data row41 col1" >1.0</td>
      <td id="T_40d0f_row41_col2" class="data row41 col2" >credit_x_credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row41_col3" class="data row41 col3" >0.2</td>
      <td id="T_40d0f_row41_col4" class="data row41 col4" >0.0355</td>
    </tr>
    <tr>
      <td id="T_40d0f_row42_col0" class="data row42 col0" >3.0</td>
      <td id="T_40d0f_row42_col1" class="data row42 col1" >1.0</td>
      <td id="T_40d0f_row42_col2" class="data row42 col2" >credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row42_col3" class="data row42 col3" >0.2</td>
      <td id="T_40d0f_row42_col4" class="data row42 col4" >0.0388</td>
    </tr>
    <tr>
      <td id="T_40d0f_row43_col0" class="data row43 col0" >3.0</td>
      <td id="T_40d0f_row43_col1" class="data row43 col1" >1.0</td>
      <td id="T_40d0f_row43_col2" class="data row43 col2" >debt_x_mortgage_ratio_bin_woe</td>
      <td id="T_40d0f_row43_col3" class="data row43 col3" >-0.2</td>
      <td id="T_40d0f_row43_col4" class="data row43 col4" >0.0482</td>
    </tr>
    <tr>
      <td id="T_40d0f_row44_col0" class="data row44 col0" >3.0</td>
      <td id="T_40d0f_row44_col1" class="data row44 col1" >1.0</td>
      <td id="T_40d0f_row44_col2" class="data row44 col2" >mortgage_x_credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row44_col3" class="data row44 col3" >-0.6</td>
      <td id="T_40d0f_row44_col4" class="data row44 col4" >0.0653</td>
    </tr>
    <tr>
      <td id="T_40d0f_row45_col0" class="data row45 col0" >3.0</td>
      <td id="T_40d0f_row45_col1" class="data row45 col1" >1.0</td>
      <td id="T_40d0f_row45_col2" class="data row45 col2" >credit_bin_woe</td>
      <td id="T_40d0f_row45_col3" class="data row45 col3" >0.2</td>
      <td id="T_40d0f_row45_col4" class="data row45 col4" >0.0955</td>
    </tr>
    <tr>
      <td id="T_40d0f_row46_col0" class="data row46 col0" >3.0</td>
      <td id="T_40d0f_row46_col1" class="data row46 col1" >1.0</td>
      <td id="T_40d0f_row46_col2" class="data row46 col2" >income_per_dep_bin_woe</td>
      <td id="T_40d0f_row46_col3" class="data row46 col3" >0.6</td>
      <td id="T_40d0f_row46_col4" class="data row46 col4" >0.0863</td>
    </tr>
    <tr>
      <td id="T_40d0f_row47_col0" class="data row47 col0" >3.0</td>
      <td id="T_40d0f_row47_col1" class="data row47 col1" >1.0</td>
      <td id="T_40d0f_row47_col2" class="data row47 col2" >debt_x_mortgage_bin_woe</td>
      <td id="T_40d0f_row47_col3" class="data row47 col3" >0.2</td>
      <td id="T_40d0f_row47_col4" class="data row47 col4" >0.0515</td>
    </tr>
    <tr>
      <td id="T_40d0f_row48_col0" class="data row48 col0" >3.0</td>
      <td id="T_40d0f_row48_col1" class="data row48 col1" >2.0</td>
      <td id="T_40d0f_row48_col2" class="data row48 col2" >credit_x_monthly_debt_bin_woe</td>
      <td id="T_40d0f_row48_col3" class="data row48 col3" >0.6</td>
      <td id="T_40d0f_row48_col4" class="data row48 col4" >0.0748</td>
    </tr>
    <tr>
      <td id="T_40d0f_row49_col0" class="data row49 col0" >3.0</td>
      <td id="T_40d0f_row49_col1" class="data row49 col1" >2.0</td>
      <td id="T_40d0f_row49_col2" class="data row49 col2" >credit_x_credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row49_col3" class="data row49 col3" >0.2</td>
      <td id="T_40d0f_row49_col4" class="data row49 col4" >0.035</td>
    </tr>
    <tr>
      <td id="T_40d0f_row50_col0" class="data row50 col0" >3.0</td>
      <td id="T_40d0f_row50_col1" class="data row50 col1" >2.0</td>
      <td id="T_40d0f_row50_col2" class="data row50 col2" >credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row50_col3" class="data row50 col3" >0.2</td>
      <td id="T_40d0f_row50_col4" class="data row50 col4" >0.0399</td>
    </tr>
    <tr>
      <td id="T_40d0f_row51_col0" class="data row51 col0" >3.0</td>
      <td id="T_40d0f_row51_col1" class="data row51 col1" >2.0</td>
      <td id="T_40d0f_row51_col2" class="data row51 col2" >debt_x_mortgage_ratio_bin_woe</td>
      <td id="T_40d0f_row51_col3" class="data row51 col3" >-0.2</td>
      <td id="T_40d0f_row51_col4" class="data row51 col4" >0.0481</td>
    </tr>
    <tr>
      <td id="T_40d0f_row52_col0" class="data row52 col0" >3.0</td>
      <td id="T_40d0f_row52_col1" class="data row52 col1" >2.0</td>
      <td id="T_40d0f_row52_col2" class="data row52 col2" >dep_x_mortgage_bin_woe</td>
      <td id="T_40d0f_row52_col3" class="data row52 col3" >-0.6</td>
      <td id="T_40d0f_row52_col4" class="data row52 col4" >0.0706</td>
    </tr>
    <tr>
      <td id="T_40d0f_row53_col0" class="data row53 col0" >3.0</td>
      <td id="T_40d0f_row53_col1" class="data row53 col1" >2.0</td>
      <td id="T_40d0f_row53_col2" class="data row53 col2" >mortgage_x_credit_pressure_index_bin_woe</td>
      <td id="T_40d0f_row53_col3" class="data row53 col3" >-0.6</td>
      <td id="T_40d0f_row53_col4" class="data row53 col4" >0.0658</td>
    </tr>
    <tr>
      <td id="T_40d0f_row54_col0" class="data row54 col0" >3.0</td>
      <td id="T_40d0f_row54_col1" class="data row54 col1" >2.0</td>
      <td id="T_40d0f_row54_col2" class="data row54 col2" >credit_bin_woe</td>
      <td id="T_40d0f_row54_col3" class="data row54 col3" >0.2</td>
      <td id="T_40d0f_row54_col4" class="data row54 col4" >0.1027</td>
    </tr>
    <tr>
      <td id="T_40d0f_row55_col0" class="data row55 col0" >3.0</td>
      <td id="T_40d0f_row55_col1" class="data row55 col1" >2.0</td>
      <td id="T_40d0f_row55_col2" class="data row55 col2" >income_per_dep_bin_woe</td>
      <td id="T_40d0f_row55_col3" class="data row55 col3" >0.6</td>
      <td id="T_40d0f_row55_col4" class="data row55 col4" >0.0876</td>
    </tr>
    <tr>
      <td id="T_40d0f_row56_col0" class="data row56 col0" >3.0</td>
      <td id="T_40d0f_row56_col1" class="data row56 col1" >2.0</td>
      <td id="T_40d0f_row56_col2" class="data row56 col2" >debt_x_mortgage_bin_woe</td>
      <td id="T_40d0f_row56_col3" class="data row56 col3" >0.2</td>
      <td id="T_40d0f_row56_col4" class="data row56 col4" >0.052</td>
    </tr>
    <tr>
      <td id="T_40d0f_row57_col0" class="data row57 col0" >4.0</td>
      <td id="T_40d0f_row57_col1" class="data row57 col1" >0.01</td>
      <td id="T_40d0f_row57_col2" class="data row57 col2" >is_util_high_woe</td>
      <td id="T_40d0f_row57_col3" class="data row57 col3" >-0.2</td>
      <td id="T_40d0f_row57_col4" class="data row57 col4" >0.0121</td>
    </tr>
    <tr>
      <td id="T_40d0f_row58_col0" class="data row58 col0" >4.0</td>
      <td id="T_40d0f_row58_col1" class="data row58 col1" >0.01</td>
      <td id="T_40d0f_row58_col2" class="data row58 col2" >income_per_dep_bin_woe</td>
      <td id="T_40d0f_row58_col3" class="data row58 col3" >-0.2</td>
      <td id="T_40d0f_row58_col4" class="data row58 col4" >0.0256</td>
    </tr>
    <tr>
      <td id="T_40d0f_row59_col0" class="data row59 col0" >4.0</td>
      <td id="T_40d0f_row59_col1" class="data row59 col1" >0.1</td>
      <td id="T_40d0f_row59_col2" class="data row59 col2" >income_per_dep_bin_woe</td>
      <td id="T_40d0f_row59_col3" class="data row59 col3" >0.6</td>
      <td id="T_40d0f_row59_col4" class="data row59 col4" >0.0562</td>
    </tr>
    <tr>
      <td id="T_40d0f_row60_col0" class="data row60 col0" >4.0</td>
      <td id="T_40d0f_row60_col1" class="data row60 col1" >0.2</td>
      <td id="T_40d0f_row60_col2" class="data row60 col2" >monthly_debt_bin_woe</td>
      <td id="T_40d0f_row60_col3" class="data row60 col3" >-0.2</td>
      <td id="T_40d0f_row60_col4" class="data row60 col4" >0.038</td>
    </tr>
    <tr>
      <td id="T_40d0f_row61_col0" class="data row61 col0" >4.0</td>
      <td id="T_40d0f_row61_col1" class="data row61 col1" >0.2</td>
      <td id="T_40d0f_row61_col2" class="data row61 col2" >income_per_dep_bin_woe</td>
      <td id="T_40d0f_row61_col3" class="data row61 col3" >0.6</td>
      <td id="T_40d0f_row61_col4" class="data row61 col4" >0.0657</td>
    </tr>
    <tr>
      <td id="T_40d0f_row62_col0" class="data row62 col0" >4.0</td>
      <td id="T_40d0f_row62_col1" class="data row62 col1" >0.5</td>
      <td id="T_40d0f_row62_col2" class="data row62 col2" >monthly_debt_bin_woe</td>
      <td id="T_40d0f_row62_col3" class="data row62 col3" >-0.2</td>
      <td id="T_40d0f_row62_col4" class="data row62 col4" >0.0394</td>
    </tr>
    <tr>
      <td id="T_40d0f_row63_col0" class="data row63 col0" >4.0</td>
      <td id="T_40d0f_row63_col1" class="data row63 col1" >0.5</td>
      <td id="T_40d0f_row63_col2" class="data row63 col2" >income_per_dep_bin_woe</td>
      <td id="T_40d0f_row63_col3" class="data row63 col3" >0.6</td>
      <td id="T_40d0f_row63_col4" class="data row63 col4" >0.0674</td>
    </tr>
    <tr>
      <td id="T_40d0f_row64_col0" class="data row64 col0" >4.0</td>
      <td id="T_40d0f_row64_col1" class="data row64 col1" >1.0</td>
      <td id="T_40d0f_row64_col2" class="data row64 col2" >monthly_debt_bin_woe</td>
      <td id="T_40d0f_row64_col3" class="data row64 col3" >-0.2</td>
      <td id="T_40d0f_row64_col4" class="data row64 col4" >0.0392</td>
    </tr>
    <tr>
      <td id="T_40d0f_row65_col0" class="data row65 col0" >4.0</td>
      <td id="T_40d0f_row65_col1" class="data row65 col1" >1.0</td>
      <td id="T_40d0f_row65_col2" class="data row65 col2" >income_per_dep_bin_woe</td>
      <td id="T_40d0f_row65_col3" class="data row65 col3" >0.6</td>
      <td id="T_40d0f_row65_col4" class="data row65 col4" >0.0682</td>
    </tr>
    <tr>
      <td id="T_40d0f_row66_col0" class="data row66 col0" >4.0</td>
      <td id="T_40d0f_row66_col1" class="data row66 col1" >2.0</td>
      <td id="T_40d0f_row66_col2" class="data row66 col2" >monthly_debt_bin_woe</td>
      <td id="T_40d0f_row66_col3" class="data row66 col3" >-0.2</td>
      <td id="T_40d0f_row66_col4" class="data row66 col4" >0.0394</td>
    </tr>
    <tr>
      <td id="T_40d0f_row67_col0" class="data row67 col0" >4.0</td>
      <td id="T_40d0f_row67_col1" class="data row67 col1" >2.0</td>
      <td id="T_40d0f_row67_col2" class="data row67 col2" >income_per_dep_bin_woe</td>
      <td id="T_40d0f_row67_col3" class="data row67 col3" >0.6</td>
      <td id="T_40d0f_row67_col4" class="data row67 col4" >0.0684</td>
    </tr>
    <tr>
      <td id="T_40d0f_row68_col0" class="data row68 col0" >5.0</td>
      <td id="T_40d0f_row68_col1" class="data row68 col1" >0.01</td>
      <td id="T_40d0f_row68_col2" class="data row68 col2" >90+late_mid_high_x_short_late_low_signal_woe</td>
      <td id="T_40d0f_row68_col3" class="data row68 col3" >-0.2</td>
      <td id="T_40d0f_row68_col4" class="data row68 col4" >0.055</td>
    </tr>
    <tr>
      <td id="T_40d0f_row69_col0" class="data row69 col0" >5.0</td>
      <td id="T_40d0f_row69_col1" class="data row69 col1" >0.01</td>
      <td id="T_40d0f_row69_col2" class="data row69 col2" >is_util_high_woe</td>
      <td id="T_40d0f_row69_col3" class="data row69 col3" >0.2</td>
      <td id="T_40d0f_row69_col4" class="data row69 col4" >0.012</td>
    </tr>
    <tr>
      <td id="T_40d0f_row70_col0" class="data row70 col0" >5.0</td>
      <td id="T_40d0f_row70_col1" class="data row70 col1" >0.025</td>
      <td id="T_40d0f_row70_col2" class="data row70 col2" >90+late_mid_high_x_short_late_low_signal_woe</td>
      <td id="T_40d0f_row70_col3" class="data row70 col3" >0.2</td>
      <td id="T_40d0f_row70_col4" class="data row70 col4" >0.0695</td>
    </tr>
    <tr>
      <td id="T_40d0f_row71_col0" class="data row71 col0" >5.0</td>
      <td id="T_40d0f_row71_col1" class="data row71 col1" >0.025</td>
      <td id="T_40d0f_row71_col2" class="data row71 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal_woe</td>
      <td id="T_40d0f_row71_col3" class="data row71 col3" >-0.2</td>
      <td id="T_40d0f_row71_col4" class="data row71 col4" >0.0338</td>
    </tr>
    <tr>
      <td id="T_40d0f_row72_col0" class="data row72 col0" >5.0</td>
      <td id="T_40d0f_row72_col1" class="data row72 col1" >0.1</td>
      <td id="T_40d0f_row72_col2" class="data row72 col2" >90+late_mid_high_x_short_late_low_signal_woe</td>
      <td id="T_40d0f_row72_col3" class="data row72 col3" >0.2</td>
      <td id="T_40d0f_row72_col4" class="data row72 col4" >0.0772</td>
    </tr>
    <tr>
      <td id="T_40d0f_row73_col0" class="data row73 col0" >5.0</td>
      <td id="T_40d0f_row73_col1" class="data row73 col1" >0.1</td>
      <td id="T_40d0f_row73_col2" class="data row73 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal_woe</td>
      <td id="T_40d0f_row73_col3" class="data row73 col3" >-0.2</td>
      <td id="T_40d0f_row73_col4" class="data row73 col4" >0.0385</td>
    </tr>
    <tr>
      <td id="T_40d0f_row74_col0" class="data row74 col0" >5.0</td>
      <td id="T_40d0f_row74_col1" class="data row74 col1" >0.2</td>
      <td id="T_40d0f_row74_col2" class="data row74 col2" >90+late_mid_high_x_short_late_low_signal_woe</td>
      <td id="T_40d0f_row74_col3" class="data row74 col3" >0.2</td>
      <td id="T_40d0f_row74_col4" class="data row74 col4" >0.0796</td>
    </tr>
    <tr>
      <td id="T_40d0f_row75_col0" class="data row75 col0" >5.0</td>
      <td id="T_40d0f_row75_col1" class="data row75 col1" >0.2</td>
      <td id="T_40d0f_row75_col2" class="data row75 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal_woe</td>
      <td id="T_40d0f_row75_col3" class="data row75 col3" >0.2</td>
      <td id="T_40d0f_row75_col4" class="data row75 col4" >0.0405</td>
    </tr>
    <tr>
      <td id="T_40d0f_row76_col0" class="data row76 col0" >5.0</td>
      <td id="T_40d0f_row76_col1" class="data row76 col1" >0.5</td>
      <td id="T_40d0f_row76_col2" class="data row76 col2" >90+late_mid_high_x_short_late_low_signal_woe</td>
      <td id="T_40d0f_row76_col3" class="data row76 col3" >0.2</td>
      <td id="T_40d0f_row76_col4" class="data row76 col4" >0.0736</td>
    </tr>
    <tr>
      <td id="T_40d0f_row77_col0" class="data row77 col0" >5.0</td>
      <td id="T_40d0f_row77_col1" class="data row77 col1" >0.5</td>
      <td id="T_40d0f_row77_col2" class="data row77 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal_woe</td>
      <td id="T_40d0f_row77_col3" class="data row77 col3" >-0.2</td>
      <td id="T_40d0f_row77_col4" class="data row77 col4" >0.0407</td>
    </tr>
    <tr>
      <td id="T_40d0f_row78_col0" class="data row78 col0" >5.0</td>
      <td id="T_40d0f_row78_col1" class="data row78 col1" >1.0</td>
      <td id="T_40d0f_row78_col2" class="data row78 col2" >90+late_mid_high_x_short_late_low_signal_woe</td>
      <td id="T_40d0f_row78_col3" class="data row78 col3" >0.2</td>
      <td id="T_40d0f_row78_col4" class="data row78 col4" >0.0752</td>
    </tr>
    <tr>
      <td id="T_40d0f_row79_col0" class="data row79 col0" >5.0</td>
      <td id="T_40d0f_row79_col1" class="data row79 col1" >1.0</td>
      <td id="T_40d0f_row79_col2" class="data row79 col2" >has_serious_late_woe</td>
      <td id="T_40d0f_row79_col3" class="data row79 col3" >-0.2</td>
      <td id="T_40d0f_row79_col4" class="data row79 col4" >0.0252</td>
    </tr>
    <tr>
      <td id="T_40d0f_row80_col0" class="data row80 col0" >5.0</td>
      <td id="T_40d0f_row80_col1" class="data row80 col1" >1.0</td>
      <td id="T_40d0f_row80_col2" class="data row80 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal_woe</td>
      <td id="T_40d0f_row80_col3" class="data row80 col3" >-0.2</td>
      <td id="T_40d0f_row80_col4" class="data row80 col4" >0.0403</td>
    </tr>
    <tr>
      <td id="T_40d0f_row81_col0" class="data row81 col0" >5.0</td>
      <td id="T_40d0f_row81_col1" class="data row81 col1" >2.0</td>
      <td id="T_40d0f_row81_col2" class="data row81 col2" >90+late_mid_high_x_short_late_low_signal_woe</td>
      <td id="T_40d0f_row81_col3" class="data row81 col3" >0.2</td>
      <td id="T_40d0f_row81_col4" class="data row81 col4" >0.0748</td>
    </tr>
    <tr>
      <td id="T_40d0f_row82_col0" class="data row82 col0" >5.0</td>
      <td id="T_40d0f_row82_col1" class="data row82 col1" >2.0</td>
      <td id="T_40d0f_row82_col2" class="data row82 col2" >has_serious_late_woe</td>
      <td id="T_40d0f_row82_col3" class="data row82 col3" >-0.2</td>
      <td id="T_40d0f_row82_col4" class="data row82 col4" >0.0257</td>
    </tr>
    <tr>
      <td id="T_40d0f_row83_col0" class="data row83 col0" >5.0</td>
      <td id="T_40d0f_row83_col1" class="data row83 col1" >2.0</td>
      <td id="T_40d0f_row83_col2" class="data row83 col2" >mortgage_ratio_mid_x_credit_late_density_high_signal_woe</td>
      <td id="T_40d0f_row83_col3" class="data row83 col3" >-0.2</td>
      <td id="T_40d0f_row83_col4" class="data row83 col4" >0.0411</td>
    </tr>
  </tbody>
</table>




```python
# 构建不同舍弃指标方案的VIF测试结果表（测试样本仅来自第五折）
if is_processing_here4:
    vif_test_df = test_vif_after_drop(X_train_woe_lr, colnames_to_drop_list)
    print_df(vif_test_df)
```


<style type="text/css">
</style>
<table id="T_a1225">
  <thead>
    <tr>
      <th id="T_a1225_level0_col0" class="col_heading level0 col0" >drop colnames index</th>
      <th id="T_a1225_level0_col1" class="col_heading level0 col1" >VIF&gt;20 count</th>
      <th id="T_a1225_level0_col2" class="col_heading level0 col2" >10&lt;VIF&lt;=20 count</th>
      <th id="T_a1225_level0_col3" class="col_heading level0 col3" >max VIF</th>
      <th id="T_a1225_level0_col4" class="col_heading level0 col4" >max VIF colname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_a1225_row0_col0" class="data row0 col0" >0</td>
      <td id="T_a1225_row0_col1" class="data row0 col1" >9</td>
      <td id="T_a1225_row0_col2" class="data row0 col2" >4</td>
      <td id="T_a1225_row0_col3" class="data row0 col3" >120.7214</td>
      <td id="T_a1225_row0_col4" class="data row0 col4" >short_late_bin_woe</td>
    </tr>
    <tr>
      <td id="T_a1225_row1_col0" class="data row1 col0" >1</td>
      <td id="T_a1225_row1_col1" class="data row1 col1" >6</td>
      <td id="T_a1225_row1_col2" class="data row1 col2" >3</td>
      <td id="T_a1225_row1_col3" class="data row1 col3" >35.5205</td>
      <td id="T_a1225_row1_col4" class="data row1 col4" >credit_x_credit_pressure_index_bin_woe</td>
    </tr>
    <tr>
      <td id="T_a1225_row2_col0" class="data row2 col0" >2</td>
      <td id="T_a1225_row2_col1" class="data row2 col1" >1</td>
      <td id="T_a1225_row2_col2" class="data row2 col2" >3</td>
      <td id="T_a1225_row2_col3" class="data row2 col3" >21.1303</td>
      <td id="T_a1225_row2_col4" class="data row2 col4" >late_severity_score_bin_woe</td>
    </tr>
    <tr>
      <td id="T_a1225_row3_col0" class="data row3 col0" >3</td>
      <td id="T_a1225_row3_col1" class="data row3 col1" >0</td>
      <td id="T_a1225_row3_col2" class="data row3 col2" >1</td>
      <td id="T_a1225_row3_col3" class="data row3 col3" >10.7201</td>
      <td id="T_a1225_row3_col4" class="data row3 col4" >debt_x_mortgage_bin_woe</td>
    </tr>
  </tbody>
</table>




```python
""" 确定对于WOE LR的最佳拟合指标与参数C """
# region
colnames_to_fit_woe_lr = [
    # 原始指标
    'age_bin_woe', 'util_bin_woe', 'debt_bin_woe',
    'income_bin_woe', 'credit_bin_woe', 'mortgage_bin_woe',
    '30-59late_bin_woe', '60-89late_bin_woe', '90+late_bin_woe',
    # 衍生指标
    'free_cashflow_income_bin_woe', 'income_per_dep_bin_woe',
    'credit_pressure_index_bin_woe', 'credit_late_density_bin_woe',
    'short_late_bin_woe', 'late_severity_score_bin_woe',
    # 含有较强风险信息的标记
    'has_no_credit_woe', 'has_short_late_but_no_credit_woe', 
    'has_serious_late_woe', 'has_short_late_woe', 'blacklist_flag_woe',
    'is_debt_high_woe', 'is_debt_overlimit_woe', 
    'is_util_high_woe', 'is_util_overlimit_woe',
    # 二维分箱
    'credit_x_credit_pressure_index_bin_woe',
    'dep_x_credit_bin_woe',
    'credit_x_monthly_debt_bin_woe',
    'debt_x_credit_bin_woe',
    'debt_x_credit_pressure_index_bin_woe',
    'debt_x_monthly_debt_bin_woe',
    'debt_x_mortgage_bin_woe',
    'debt_x_mortgage_ratio_bin_woe',
    'income_x_monthly_debt_bin_woe',
    'monthly_debt_x_credit_pressure_index_bin_woe',
    'mortgage_x_credit_pressure_index_bin_woe',
    'dep_x_mortgage_bin_woe',
    'mortgage_x_monthly_debt_bin_woe'
]
c_woe_lr = 0.025
colnames_to_drop_woe_lr = [
    '30-59late_bin_woe', '60-89late_bin_woe', '90+late_bin_woe',
    'short_late_bin_woe', 'has_short_late_woe',
    'credit_late_density_bin_woe', 'income_bin_woe', 'free_cashflow_income_bin_woe',
    'credit_x_credit_pressure_index_bin_woe',
    'mortgage_x_credit_pressure_index_bin_woe',
    'monthly_debt_x_credit_pressure_index_bin_woe'
]
# 检测固定最佳的删除指标后模型拟合效果的变化情况
if is_processing_here4:
    woe_lr_fit_goodness_after_drop_test_df = test_fit_goodness_lr_after_drop(
        X_train_woe_lr, y_train, X_valid_woe_lr, y_valid, 
        colnames_to_fit_woe_lr, c_woe_lr, colnames_to_drop_woe_lr
    )
    print_df(woe_lr_fit_goodness_after_drop_test_df)
# 简化变量
colnames_to_fit_woe_lr = \
    [colname for colname in colnames_to_fit_woe_lr if colname not in colnames_to_drop_woe_lr]
# 释放内存
cleanup_vars([
    'train_data', 'valid_data', 'train_data_copy', 'valid_data_copy',
    'train_data_woe_lr', 'valid_data_woe_lr',
    'X_train_woe_lr', 'X_valid_woe_lr', 'y_train', 'y_valid',
    'folds', 'train_idx_list', 'valid_idx_list',
    'woe_lr_fit_goodness_df_list', 'woe_lr_fit_goodness_df',
    'woe_lr_auc_ks_df', 'woe_lr_coef_sign_stability_df',
    'woe_lr_fit_goodness_after_drop_test_df', 'vif_test_df',
    'colnames_not_binary', 'colname_pairs_not_binary',
    'binnames_map_to_woe', 'woe_colnames',
    'div_pts', 'bins_2D',
    'original_features', 'derived_features', 'risk_signals',
    'low_iv_features', 'positive_1_5',
    'colnames_to_fit0', 'colnames_to_fit1', 'colnames_to_fit2',
    'colnames_to_fit3', 'colnames_to_fit4', 'colnames_to_fit5',
    'colnames_to_fit_list', 'colnames_to_fit',
    'c_list', 'c',
    'colnames_to_drop0', 'colnames_to_drop1', 'colnames_to_drop2',
    'colnames_to_drop3', 'colnames_to_drop_list',
    'woe_map', 'beta30', 'beta60', 'beta90',
], close_plots=False)
# endregion
```


<style type="text/css">
</style>
<table id="T_00ff5">
  <thead>
    <tr>
      <th id="T_00ff5_level0_col0" class="col_heading level0 col0" >status</th>
      <th id="T_00ff5_level0_col1" class="col_heading level0 col1" >valid AUC</th>
      <th id="T_00ff5_level0_col2" class="col_heading level0 col2" >AUC gap</th>
      <th id="T_00ff5_level0_col3" class="col_heading level0 col3" >valid KS</th>
      <th id="T_00ff5_level0_col4" class="col_heading level0 col4" >KS gap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_00ff5_row0_col0" class="data row0 col0" >retained</td>
      <td id="T_00ff5_row0_col1" class="data row0 col1" >0.8563</td>
      <td id="T_00ff5_row0_col2" class="data row0 col2" >0.0061</td>
      <td id="T_00ff5_row0_col3" class="data row0 col3" >0.5665</td>
      <td id="T_00ff5_row0_col4" class="data row0 col4" >0.0032</td>
    </tr>
    <tr>
      <td id="T_00ff5_row1_col0" class="data row1 col0" >dropped</td>
      <td id="T_00ff5_row1_col1" class="data row1 col1" >0.855</td>
      <td id="T_00ff5_row1_col2" class="data row1 col2" >0.0064</td>
      <td id="T_00ff5_row1_col3" class="data row1 col3" >0.5655</td>
      <td id="T_00ff5_row1_col4" class="data row1 col4" >0.0011</td>
    </tr>
    <tr>
      <td id="T_00ff5_row2_col0" class="data row2 col0" >difference</td>
      <td id="T_00ff5_row2_col1" class="data row2 col1" >0.0012</td>
      <td id="T_00ff5_row2_col2" class="data row2 col2" >-0.0003</td>
      <td id="T_00ff5_row2_col3" class="data row2 col3" >0.001</td>
      <td id="T_00ff5_row2_col4" class="data row2 col4" >0.0021</td>
    </tr>
  </tbody>
</table>




```python
""" 对XGBoost模型的调试 """
is_processing_here5 = True

if is_processing_here5:
    # region
    xgb_auc_ks_round1_df_list = []
    xgb_top_recall_df_list = []
    xgb_auc_ks_round2_df_list = []
    xgb_auc_ks_round3_df_list = []
    xgb_auc_ks_round4_df_list = []
    xgb_auc_ks_round5_df_list = []
    # 划分训练集、验证集，准备K=5折交叉验证
    random_seed = 400
    folds = create_kfold(train_valid_data, random_seed, n_splits=5)
    for fold_idx in range(len(folds)):
        # 训练集、验证集拆分
        train_idx_list, valid_idx_list = folds[fold_idx]
        train_data = train_valid_data.iloc[train_idx_list].copy()
        valid_data = train_valid_data.iloc[valid_idx_list].copy()
        train_data_copy = train_data.copy(); valid_data_copy = valid_data.copy() # 备份
        # endregion
        
        """ 统一数据清洗阶段 """
        # region
        # 截断家属数异常大值
        cap_dep_num(train_data); cap_dep_num(valid_data)
        # 删除年龄异常小值
        delete_small_age(train_data); delete_small_age(valid_data)
        # 添加信用额度使用率的若干标记
        add_util_flags(train_data); add_util_flags(valid_data)
        # 添加负债率的若干标记
        add_debt_flags(train_data); add_debt_flags(valid_data)
        # 添加三种逾期次数的加权求和，权重由逻辑回归模型训练
        beta30, beta60, beta90 = add_late_severity_score(train_data)
        add_late_severity_score(valid_data, [beta30, beta60, beta90])
        # 添加衍生指标
        add_derived_features(train_data); add_derived_features(valid_data)
        # 筛选出所有非二元指标
        colnames_not_binary = get_colnames_whether_binary(train_data, is_binary=False)
        # 对所有非布尔值类型的指标建立一维分箱
        for colname in colnames_not_binary:
            _, div_pts = add_bins_1D(train_data, colname)
            add_bins_1D(valid_data, colname, fixed_div_pts=div_pts)
        # 把一维分箱的具体数值转换为排名序号
        convert_bins_value_to_rank(train_data); convert_bins_value_to_rank(valid_data)
        # 添加交互标记
        add_intersection_signals(train_data, positive_interactions_df, negative_interactions_df)
        add_intersection_signals(valid_data, positive_interactions_df, negative_interactions_df)
        # 暂时删除一维分箱列以及其对应的排名序号列
        colnames_without_bins = [colname for colname in train_data.columns if '_bin' not in colname]
        train_data = train_data[colnames_without_bins]; valid_data = valid_data[colnames_without_bins]
        # endregion
        
        """ 对XGBoost建立特化表达 """
        # region
        train_data_xgb = train_data.copy(); valid_data_xgb = valid_data.copy()
        # 将所有表示数据不可靠的负整数转换为NA
        train_data_xgb = train_data_xgb.mask(train_data_xgb < 0, np.nan)
        valid_data_xgb = valid_data_xgb.mask(valid_data_xgb < 0, np.nan)
        # endregion
        
        """ 训练XGBoost模型 """
        # region
        # 拆分目标变量和自变量
        y_train = train_data[target]; y_valid = valid_data[target]
        X_train_xgb = train_data_xgb.drop(target, axis=1, inplace=False)
        X_valid_xgb = valid_data_xgb.drop(target, axis=1, inplace=False)
        # 第一轮网格搜索（对参与拟合的指标）
        # 原始指标
        original_features = [
            age, util, debt,
            income, dep, credit, mortgage,
            late30, late60, late90
        ]
        # 数据质量标记
        quality_flags = [
            'single_missing_flag', 'both_missing_flag', 'blacklist_flag',
            'income_anomaly_flag', 'debt_anomaly_flag', 'util_anomaly_flag'
        ]
        # 衍生指标
        derived_features = [
            'free_cashflow_income', 'income_per_dep', 'monthly_debt',
            'credit_pressure_index', 'mortgage_ratio', 'credit_late_density',
            'short_late', 'late_severity_score'
        ]
        # 用户行为标记
        risk_signals = [
            'has_no_credit', 'has_short_late_but_no_credit', 'has_serious_late', 'has_short_late',
            'is_debt_high', 'is_debt_overlimit', 'is_util_high', 'is_util_overlimit'
        ]
        # 削弱效应第1-5名
        negative_1_5 = [
            'monthly_debt_low_x_mortgage_ratio_mid_signal',
            'credit_low_x_util_mid_signal',
            'credit_pressure_mid_x_mortgage_mid_signal',
            'credit_pressure_mid_x_debt_mid_signal',
            'monthly_debt_low_x_age_high_signal'
        ]
        # 增强效应第1-5名
        positive_1_5 = [
            'mortgage_ratio_mid_x_credit_late_density_high_signal',
            'mortgage_mid_x_credit_late_density_high_signal',
            '90+late_mid_high_x_short_late_low_signal',
            'late_severity_high_x_short_late_low_signal',
            'age_high_x_late_severity_high_signal'
        ]
        # 增强效应第6-10名
        positive_6_10 = [
            'dep_missing_x_60-89late_mid_signal',
            '30-59late_mid_x_credit_mid_signal',
            'late_severity_high_x_credit_pressure_low_signal',
            'age_high_x_credit_late_density_high_signal',
            'util_low_mid_x_late_severity_high_signal'
        ]
        # 即将尝试的拟合指标组合
        colnames_to_fit0 = original_features
        colnames_to_fit1 = colnames_to_fit0 + quality_flags
        colnames_to_fit2 = colnames_to_fit1 + derived_features
        colnames_to_fit3 = colnames_to_fit2 + risk_signals
        colnames_to_fit4 = colnames_to_fit2 + negative_1_5
        colnames_to_fit5 = colnames_to_fit2 + positive_1_5
        colnames_to_fit_list_round1 = [
            colnames_to_fit0, colnames_to_fit1, colnames_to_fit2, 
            colnames_to_fit3, colnames_to_fit4, colnames_to_fit5, 
        ]
        xgb_auc_ks_round1_df = grid_search_colnames_xgb(
            X_train_xgb, y_train, X_valid_xgb, y_valid, 
            colnames_to_fit_list_round1
        )
        xgb_auc_ks_round1_df_list.append(xgb_auc_ks_round1_df)
        # 在第一轮网格搜索的基础上，探索哪些衍生指标与用户行为标记值得被保留
        xgb_top_recall_df = grid_search_colnames_upgraded_xgb(
            X_train_xgb, y_train, X_valid_xgb, y_valid,
            base_colnames=original_features + quality_flags,
            colnames_to_try=derived_features + risk_signals
        )
        xgb_top_recall_df_list.append(xgb_top_recall_df)
        # 确定保留下来的衍生指标
        derived_features_retained = [
            'free_cashflow_income', 'credit_late_density', 'credit_pressure_index',
            'mortgage_ratio', 'late_severity_score'
        ]
        colnames_to_fit = original_features + derived_features_retained
        X_train_xgb = X_train_xgb[colnames_to_fit]; X_valid_xgb = X_valid_xgb[colnames_to_fit]
        # 确定下来的最佳参数的储存字典
        best_para_dict = {}
        # 第二轮网格搜索（对模型参数）
        paras_round2 = ['max_depth', 'min_child_weight']
        max_depth_list = [2, 3, 4, 5, 6, 7]
        min_child_weight_list = [1, 3, 5, 7, 10]
        xgb_auc_ks_round2_df = grid_search_paras_xgb(
            X_train_xgb, y_train, X_valid_xgb, y_valid, 
            paras_round2, max_depth_list, min_child_weight_list
        )
        xgb_auc_ks_round2_df_list.append(xgb_auc_ks_round2_df)
        # 确定最佳的max_depth和min_child_weight
        best_para_dict['max_depth'] = 3
        best_para_dict['min_child_weight'] = 7
        # 第三轮网格搜索（对模型参数）
        paras_round3 = ['learning_rate', 'n_estimators']
        learning_rate_list = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
        n_estimators_list = [300, 500, 800, 1000, 1500]
        xgb_auc_ks_round3_df = grid_search_paras_xgb(
            X_train_xgb, y_train, X_valid_xgb, y_valid, 
            paras_round3, learning_rate_list, n_estimators_list, best_para_dict
        )
        xgb_auc_ks_round3_df_list.append(xgb_auc_ks_round3_df)
        # 确定最佳的learning_rate和n_estimators
        best_para_dict['learning_rate'] = 0.03
        best_para_dict['n_estimators'] = 800
        # 第四轮网格搜索（对模型参数）
        paras_round4 = ['subsample', 'colsample_bytree']
        subsample_list = [0.6, 0.7, 0.8, 0.9]
        colsample_bytree_list = [0.6, 0.7, 0.8, 0.9]
        xgb_auc_ks_round4_df = grid_search_paras_xgb(
            X_train_xgb, y_train, X_valid_xgb, y_valid, 
            paras_round4, subsample_list, colsample_bytree_list, best_para_dict
        )
        xgb_auc_ks_round4_df_list.append(xgb_auc_ks_round4_df)
        # 确定最佳的subsample和colsample_bytree
        best_para_dict['subsample'] = 0.7
        best_para_dict['colsample_bytree'] = 0.6
        # # 第五轮网格搜索（对模型参数）
        paras_round5 = ['reg_alpha', 'reg_lambda']
        reg_alpha_list = [0, 0.1, 0.25, 0.5, 1]
        reg_lambda_list = [1, 2, 4, 7, 10]
        xgb_auc_ks_round5_df = grid_search_paras_xgb(
            X_train_xgb, y_train, X_valid_xgb, y_valid, 
            paras_round5, reg_alpha_list, reg_lambda_list, best_para_dict
        )
        xgb_auc_ks_round5_df_list.append(xgb_auc_ks_round5_df)
        # 确定最佳的reg_alpha和reg_lambda
        best_para_dict['reg_alpha'] = 1
        best_para_dict['reg_lambda'] = 2
        # endregion

```


```python
# 整合五折交叉验证得到的五个XGBoost第1轮网格搜索的结果表
if is_processing_here5:
    xgb_fit_goodness_round1_df = test_fit_goodness_xgb(xgb_auc_ks_round1_df_list)
    print_df(xgb_fit_goodness_round1_df)
```


<style type="text/css">
</style>
<table id="T_e7420">
  <thead>
    <tr>
      <th id="T_e7420_level0_col0" class="col_heading level0 col0" >fit columns index</th>
      <th id="T_e7420_level0_col1" class="col_heading level0 col1" >valid_AUC_mean</th>
      <th id="T_e7420_level0_col2" class="col_heading level0 col2" >AUC_gap_mean</th>
      <th id="T_e7420_level0_col3" class="col_heading level0 col3" >valid_KS_mean</th>
      <th id="T_e7420_level0_col4" class="col_heading level0 col4" >KS_gap_mean</th>
      <th id="T_e7420_level0_col5" class="col_heading level0 col5" >best_iteration_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_e7420_row0_col0" class="data row0 col0" >5.0</td>
      <td id="T_e7420_row0_col1" class="data row0 col1" >0.8663</td>
      <td id="T_e7420_row0_col2" class="data row0 col2" >0.0159</td>
      <td id="T_e7420_row0_col3" class="data row0 col3" >0.5794</td>
      <td id="T_e7420_row0_col4" class="data row0 col4" >0.025</td>
      <td id="T_e7420_row0_col5" class="data row0 col5" >124.0</td>
    </tr>
    <tr>
      <td id="T_e7420_row1_col0" class="data row1 col0" >2.0</td>
      <td id="T_e7420_row1_col1" class="data row1 col1" >0.8663</td>
      <td id="T_e7420_row1_col2" class="data row1 col2" >0.0162</td>
      <td id="T_e7420_row1_col3" class="data row1 col3" >0.5776</td>
      <td id="T_e7420_row1_col4" class="data row1 col4" >0.0271</td>
      <td id="T_e7420_row1_col5" class="data row1 col5" >124.2</td>
    </tr>
    <tr>
      <td id="T_e7420_row2_col0" class="data row2 col0" >4.0</td>
      <td id="T_e7420_row2_col1" class="data row2 col1" >0.8662</td>
      <td id="T_e7420_row2_col2" class="data row2 col2" >0.0152</td>
      <td id="T_e7420_row2_col3" class="data row2 col3" >0.5781</td>
      <td id="T_e7420_row2_col4" class="data row2 col4" >0.0248</td>
      <td id="T_e7420_row2_col5" class="data row2 col5" >113.6</td>
    </tr>
    <tr>
      <td id="T_e7420_row3_col0" class="data row3 col0" >3.0</td>
      <td id="T_e7420_row3_col1" class="data row3 col1" >0.8662</td>
      <td id="T_e7420_row3_col2" class="data row3 col2" >0.0159</td>
      <td id="T_e7420_row3_col3" class="data row3 col3" >0.5773</td>
      <td id="T_e7420_row3_col4" class="data row3 col4" >0.0269</td>
      <td id="T_e7420_row3_col5" class="data row3 col5" >122.6</td>
    </tr>
    <tr>
      <td id="T_e7420_row4_col0" class="data row4 col0" >1.0</td>
      <td id="T_e7420_row4_col1" class="data row4 col1" >0.8661</td>
      <td id="T_e7420_row4_col2" class="data row4 col2" >0.0125</td>
      <td id="T_e7420_row4_col3" class="data row4 col3" >0.5788</td>
      <td id="T_e7420_row4_col4" class="data row4 col4" >0.02</td>
      <td id="T_e7420_row4_col5" class="data row4 col5" >128.8</td>
    </tr>
    <tr>
      <td id="T_e7420_row5_col0" class="data row5 col0" >0.0</td>
      <td id="T_e7420_row5_col1" class="data row5 col1" >0.8658</td>
      <td id="T_e7420_row5_col2" class="data row5 col2" >0.0121</td>
      <td id="T_e7420_row5_col3" class="data row5 col3" >0.5781</td>
      <td id="T_e7420_row5_col4" class="data row5 col4" >0.0183</td>
      <td id="T_e7420_row5_col5" class="data row5 col5" >116.6</td>
    </tr>
  </tbody>
</table>




```python
# 整合五折交叉验证得到的五个XGBoost第1轮网格搜索（进阶版）的结果表
if is_processing_here5:
    xgb_fit_goodness_round1_upgraded_df = test_fit_goodness_upgraded_xgb(xgb_top_recall_df_list)
    print_df(xgb_fit_goodness_round1_upgraded_df)
```


<style type="text/css">
</style>
<table id="T_ec426">
  <thead>
    <tr>
      <th id="T_ec426_level0_col0" class="col_heading level0 col0" >added feature</th>
      <th id="T_ec426_level0_col1" class="col_heading level0 col1" >top5_recall_mean</th>
      <th id="T_ec426_level0_col2" class="col_heading level0 col2" >top10_recall_mean</th>
      <th id="T_ec426_level0_col3" class="col_heading level0 col3" >top20_recall_mean</th>
      <th id="T_ec426_level0_col4" class="col_heading level0 col4" >top5_recall_gap_mean</th>
      <th id="T_ec426_level0_col5" class="col_heading level0 col5" >top10_recall_gap_mean</th>
      <th id="T_ec426_level0_col6" class="col_heading level0 col6" >top20_recall_gap_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_ec426_row0_col0" class="data row0 col0" >credit_late_density</td>
      <td id="T_ec426_row0_col1" class="data row0 col1" >0.3602</td>
      <td id="T_ec426_row0_col2" class="data row0 col2" >0.5491</td>
      <td id="T_ec426_row0_col3" class="data row0 col3" >0.7285</td>
      <td id="T_ec426_row0_col4" class="data row0 col4" >-0.0026</td>
      <td id="T_ec426_row0_col5" class="data row0 col5" >-0.0004</td>
      <td id="T_ec426_row0_col6" class="data row0 col6" >-0.0006</td>
    </tr>
    <tr>
      <td id="T_ec426_row1_col0" class="data row1 col0" >credit_pressure_index</td>
      <td id="T_ec426_row1_col1" class="data row1 col1" >0.3611</td>
      <td id="T_ec426_row1_col2" class="data row1 col2" >0.5497</td>
      <td id="T_ec426_row1_col3" class="data row1 col3" >0.7281</td>
      <td id="T_ec426_row1_col4" class="data row1 col4" >-0.0017</td>
      <td id="T_ec426_row1_col5" class="data row1 col5" >0.0002</td>
      <td id="T_ec426_row1_col6" class="data row1 col6" >-0.001</td>
    </tr>
    <tr>
      <td id="T_ec426_row2_col0" class="data row2 col0" >free_cashflow_income</td>
      <td id="T_ec426_row2_col1" class="data row2 col1" >0.3607</td>
      <td id="T_ec426_row2_col2" class="data row2 col2" >0.5493</td>
      <td id="T_ec426_row2_col3" class="data row2 col3" >0.7267</td>
      <td id="T_ec426_row2_col4" class="data row2 col4" >-0.0021</td>
      <td id="T_ec426_row2_col5" class="data row2 col5" >-0.0001</td>
      <td id="T_ec426_row2_col6" class="data row2 col6" >-0.0024</td>
    </tr>
    <tr>
      <td id="T_ec426_row3_col0" class="data row3 col0" >late_severity_score</td>
      <td id="T_ec426_row3_col1" class="data row3 col1" >0.3635</td>
      <td id="T_ec426_row3_col2" class="data row3 col2" >0.5514</td>
      <td id="T_ec426_row3_col3" class="data row3 col3" >0.7306</td>
      <td id="T_ec426_row3_col4" class="data row3 col4" >0.0007</td>
      <td id="T_ec426_row3_col5" class="data row3 col5" >0.002</td>
      <td id="T_ec426_row3_col6" class="data row3 col6" >0.0015</td>
    </tr>
    <tr>
      <td id="T_ec426_row4_col0" class="data row4 col0" >mortgage_ratio</td>
      <td id="T_ec426_row4_col1" class="data row4 col1" >0.3614</td>
      <td id="T_ec426_row4_col2" class="data row4 col2" >0.5511</td>
      <td id="T_ec426_row4_col3" class="data row4 col3" >0.7268</td>
      <td id="T_ec426_row4_col4" class="data row4 col4" >-0.0014</td>
      <td id="T_ec426_row4_col5" class="data row4 col5" >0.0016</td>
      <td id="T_ec426_row4_col6" class="data row4 col6" >-0.0022</td>
    </tr>
  </tbody>
</table>




```python
# 整合五折交叉验证得到的五个XGBoost第2轮网格搜索的结果表
if is_processing_here5:
    xgb_fit_goodness_round2_df = test_fit_goodness_xgb(xgb_auc_ks_round2_df_list)
    print_df(xgb_fit_goodness_round2_df)
```


<style type="text/css">
</style>
<table id="T_5d348">
  <thead>
    <tr>
      <th id="T_5d348_level0_col0" class="col_heading level0 col0" >max_depth</th>
      <th id="T_5d348_level0_col1" class="col_heading level0 col1" >min_child_weight</th>
      <th id="T_5d348_level0_col2" class="col_heading level0 col2" >valid_AUC_mean</th>
      <th id="T_5d348_level0_col3" class="col_heading level0 col3" >AUC_gap_mean</th>
      <th id="T_5d348_level0_col4" class="col_heading level0 col4" >valid_KS_mean</th>
      <th id="T_5d348_level0_col5" class="col_heading level0 col5" >KS_gap_mean</th>
      <th id="T_5d348_level0_col6" class="col_heading level0 col6" >best_iteration_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_5d348_row0_col0" class="data row0 col0" >4.0</td>
      <td id="T_5d348_row0_col1" class="data row0 col1" >1.0</td>
      <td id="T_5d348_row0_col2" class="data row0 col2" >0.8662</td>
      <td id="T_5d348_row0_col3" class="data row0 col3" >0.0093</td>
      <td id="T_5d348_row0_col4" class="data row0 col4" >0.577</td>
      <td id="T_5d348_row0_col5" class="data row0 col5" >0.0144</td>
      <td id="T_5d348_row0_col6" class="data row0 col6" >136.6</td>
    </tr>
    <tr>
      <td id="T_5d348_row1_col0" class="data row1 col0" >4.0</td>
      <td id="T_5d348_row1_col1" class="data row1 col1" >5.0</td>
      <td id="T_5d348_row1_col2" class="data row1 col2" >0.8662</td>
      <td id="T_5d348_row1_col3" class="data row1 col3" >0.0093</td>
      <td id="T_5d348_row1_col4" class="data row1 col4" >0.5773</td>
      <td id="T_5d348_row1_col5" class="data row1 col5" >0.0148</td>
      <td id="T_5d348_row1_col6" class="data row1 col6" >147.8</td>
    </tr>
    <tr>
      <td id="T_5d348_row2_col0" class="data row2 col0" >4.0</td>
      <td id="T_5d348_row2_col1" class="data row2 col1" >3.0</td>
      <td id="T_5d348_row2_col2" class="data row2 col2" >0.8661</td>
      <td id="T_5d348_row2_col3" class="data row2 col3" >0.0093</td>
      <td id="T_5d348_row2_col4" class="data row2 col4" >0.5765</td>
      <td id="T_5d348_row2_col5" class="data row2 col5" >0.0152</td>
      <td id="T_5d348_row2_col6" class="data row2 col6" >140.8</td>
    </tr>
    <tr>
      <td id="T_5d348_row3_col0" class="data row3 col0" >4.0</td>
      <td id="T_5d348_row3_col1" class="data row3 col1" >10.0</td>
      <td id="T_5d348_row3_col2" class="data row3 col2" >0.8661</td>
      <td id="T_5d348_row3_col3" class="data row3 col3" >0.0096</td>
      <td id="T_5d348_row3_col4" class="data row3 col4" >0.5775</td>
      <td id="T_5d348_row3_col5" class="data row3 col5" >0.015</td>
      <td id="T_5d348_row3_col6" class="data row3 col6" >161.8</td>
    </tr>
    <tr>
      <td id="T_5d348_row4_col0" class="data row4 col0" >4.0</td>
      <td id="T_5d348_row4_col1" class="data row4 col1" >7.0</td>
      <td id="T_5d348_row4_col2" class="data row4 col2" >0.8661</td>
      <td id="T_5d348_row4_col3" class="data row4 col3" >0.0087</td>
      <td id="T_5d348_row4_col4" class="data row4 col4" >0.5773</td>
      <td id="T_5d348_row4_col5" class="data row4 col5" >0.0133</td>
      <td id="T_5d348_row4_col6" class="data row4 col6" >131.6</td>
    </tr>
    <tr>
      <td id="T_5d348_row5_col0" class="data row5 col0" >3.0</td>
      <td id="T_5d348_row5_col1" class="data row5 col1" >3.0</td>
      <td id="T_5d348_row5_col2" class="data row5 col2" >0.866</td>
      <td id="T_5d348_row5_col3" class="data row5 col3" >0.0066</td>
      <td id="T_5d348_row5_col4" class="data row5 col4" >0.5761</td>
      <td id="T_5d348_row5_col5" class="data row5 col5" >0.011</td>
      <td id="T_5d348_row5_col6" class="data row5 col6" >220.0</td>
    </tr>
    <tr>
      <td id="T_5d348_row6_col0" class="data row6 col0" >5.0</td>
      <td id="T_5d348_row6_col1" class="data row6 col1" >3.0</td>
      <td id="T_5d348_row6_col2" class="data row6 col2" >0.866</td>
      <td id="T_5d348_row6_col3" class="data row6 col3" >0.0152</td>
      <td id="T_5d348_row6_col4" class="data row6 col4" >0.5775</td>
      <td id="T_5d348_row6_col5" class="data row6 col5" >0.0245</td>
      <td id="T_5d348_row6_col6" class="data row6 col6" >123.2</td>
    </tr>
    <tr>
      <td id="T_5d348_row7_col0" class="data row7 col0" >5.0</td>
      <td id="T_5d348_row7_col1" class="data row7 col1" >5.0</td>
      <td id="T_5d348_row7_col2" class="data row7 col2" >0.866</td>
      <td id="T_5d348_row7_col3" class="data row7 col3" >0.0146</td>
      <td id="T_5d348_row7_col4" class="data row7 col4" >0.5778</td>
      <td id="T_5d348_row7_col5" class="data row7 col5" >0.0234</td>
      <td id="T_5d348_row7_col6" class="data row7 col6" >122.0</td>
    </tr>
    <tr>
      <td id="T_5d348_row8_col0" class="data row8 col0" >3.0</td>
      <td id="T_5d348_row8_col1" class="data row8 col1" >1.0</td>
      <td id="T_5d348_row8_col2" class="data row8 col2" >0.866</td>
      <td id="T_5d348_row8_col3" class="data row8 col3" >0.0071</td>
      <td id="T_5d348_row8_col4" class="data row8 col4" >0.5764</td>
      <td id="T_5d348_row8_col5" class="data row8 col5" >0.0114</td>
      <td id="T_5d348_row8_col6" class="data row8 col6" >240.4</td>
    </tr>
    <tr>
      <td id="T_5d348_row9_col0" class="data row9 col0" >5.0</td>
      <td id="T_5d348_row9_col1" class="data row9 col1" >7.0</td>
      <td id="T_5d348_row9_col2" class="data row9 col2" >0.866</td>
      <td id="T_5d348_row9_col3" class="data row9 col3" >0.0136</td>
      <td id="T_5d348_row9_col4" class="data row9 col4" >0.5786</td>
      <td id="T_5d348_row9_col5" class="data row9 col5" >0.0209</td>
      <td id="T_5d348_row9_col6" class="data row9 col6" >113.4</td>
    </tr>
    <tr>
      <td id="T_5d348_row10_col0" class="data row10 col0" >5.0</td>
      <td id="T_5d348_row10_col1" class="data row10 col1" >10.0</td>
      <td id="T_5d348_row10_col2" class="data row10 col2" >0.8659</td>
      <td id="T_5d348_row10_col3" class="data row10 col3" >0.0138</td>
      <td id="T_5d348_row10_col4" class="data row10 col4" >0.5762</td>
      <td id="T_5d348_row10_col5" class="data row10 col5" >0.024</td>
      <td id="T_5d348_row10_col6" class="data row10 col6" >124.8</td>
    </tr>
    <tr>
      <td id="T_5d348_row11_col0" class="data row11 col0" >3.0</td>
      <td id="T_5d348_row11_col1" class="data row11 col1" >7.0</td>
      <td id="T_5d348_row11_col2" class="data row11 col2" >0.8659</td>
      <td id="T_5d348_row11_col3" class="data row11 col3" >0.0061</td>
      <td id="T_5d348_row11_col4" class="data row11 col4" >0.5758</td>
      <td id="T_5d348_row11_col5" class="data row11 col5" >0.0105</td>
      <td id="T_5d348_row11_col6" class="data row11 col6" >195.4</td>
    </tr>
    <tr>
      <td id="T_5d348_row12_col0" class="data row12 col0" >3.0</td>
      <td id="T_5d348_row12_col1" class="data row12 col1" >5.0</td>
      <td id="T_5d348_row12_col2" class="data row12 col2" >0.8659</td>
      <td id="T_5d348_row12_col3" class="data row12 col3" >0.0059</td>
      <td id="T_5d348_row12_col4" class="data row12 col4" >0.576</td>
      <td id="T_5d348_row12_col5" class="data row12 col5" >0.0098</td>
      <td id="T_5d348_row12_col6" class="data row12 col6" >186.2</td>
    </tr>
    <tr>
      <td id="T_5d348_row13_col0" class="data row13 col0" >5.0</td>
      <td id="T_5d348_row13_col1" class="data row13 col1" >1.0</td>
      <td id="T_5d348_row13_col2" class="data row13 col2" >0.8658</td>
      <td id="T_5d348_row13_col3" class="data row13 col3" >0.0151</td>
      <td id="T_5d348_row13_col4" class="data row13 col4" >0.5765</td>
      <td id="T_5d348_row13_col5" class="data row13 col5" >0.025</td>
      <td id="T_5d348_row13_col6" class="data row13 col6" >110.4</td>
    </tr>
    <tr>
      <td id="T_5d348_row14_col0" class="data row14 col0" >3.0</td>
      <td id="T_5d348_row14_col1" class="data row14 col1" >10.0</td>
      <td id="T_5d348_row14_col2" class="data row14 col2" >0.8658</td>
      <td id="T_5d348_row14_col3" class="data row14 col3" >0.0058</td>
      <td id="T_5d348_row14_col4" class="data row14 col4" >0.5758</td>
      <td id="T_5d348_row14_col5" class="data row14 col5" >0.0096</td>
      <td id="T_5d348_row14_col6" class="data row14 col6" >186.0</td>
    </tr>
    <tr>
      <td id="T_5d348_row15_col0" class="data row15 col0" >6.0</td>
      <td id="T_5d348_row15_col1" class="data row15 col1" >10.0</td>
      <td id="T_5d348_row15_col2" class="data row15 col2" >0.8655</td>
      <td id="T_5d348_row15_col3" class="data row15 col3" >0.0185</td>
      <td id="T_5d348_row15_col4" class="data row15 col4" >0.5761</td>
      <td id="T_5d348_row15_col5" class="data row15 col5" >0.0321</td>
      <td id="T_5d348_row15_col6" class="data row15 col6" >95.6</td>
    </tr>
    <tr>
      <td id="T_5d348_row16_col0" class="data row16 col0" >6.0</td>
      <td id="T_5d348_row16_col1" class="data row16 col1" >5.0</td>
      <td id="T_5d348_row16_col2" class="data row16 col2" >0.8655</td>
      <td id="T_5d348_row16_col3" class="data row16 col3" >0.0198</td>
      <td id="T_5d348_row16_col4" class="data row16 col4" >0.577</td>
      <td id="T_5d348_row16_col5" class="data row16 col5" >0.0338</td>
      <td id="T_5d348_row16_col6" class="data row16 col6" >90.0</td>
    </tr>
    <tr>
      <td id="T_5d348_row17_col0" class="data row17 col0" >6.0</td>
      <td id="T_5d348_row17_col1" class="data row17 col1" >3.0</td>
      <td id="T_5d348_row17_col2" class="data row17 col2" >0.8654</td>
      <td id="T_5d348_row17_col3" class="data row17 col3" >0.0209</td>
      <td id="T_5d348_row17_col4" class="data row17 col4" >0.5765</td>
      <td id="T_5d348_row17_col5" class="data row17 col5" >0.0355</td>
      <td id="T_5d348_row17_col6" class="data row17 col6" >89.4</td>
    </tr>
    <tr>
      <td id="T_5d348_row18_col0" class="data row18 col0" >6.0</td>
      <td id="T_5d348_row18_col1" class="data row18 col1" >1.0</td>
      <td id="T_5d348_row18_col2" class="data row18 col2" >0.8653</td>
      <td id="T_5d348_row18_col3" class="data row18 col3" >0.0202</td>
      <td id="T_5d348_row18_col4" class="data row18 col4" >0.5763</td>
      <td id="T_5d348_row18_col5" class="data row18 col5" >0.0339</td>
      <td id="T_5d348_row18_col6" class="data row18 col6" >73.6</td>
    </tr>
    <tr>
      <td id="T_5d348_row19_col0" class="data row19 col0" >6.0</td>
      <td id="T_5d348_row19_col1" class="data row19 col1" >7.0</td>
      <td id="T_5d348_row19_col2" class="data row19 col2" >0.8653</td>
      <td id="T_5d348_row19_col3" class="data row19 col3" >0.018</td>
      <td id="T_5d348_row19_col4" class="data row19 col4" >0.5766</td>
      <td id="T_5d348_row19_col5" class="data row19 col5" >0.0306</td>
      <td id="T_5d348_row19_col6" class="data row19 col6" >83.6</td>
    </tr>
    <tr>
      <td id="T_5d348_row20_col0" class="data row20 col0" >2.0</td>
      <td id="T_5d348_row20_col1" class="data row20 col1" >7.0</td>
      <td id="T_5d348_row20_col2" class="data row20 col2" >0.8652</td>
      <td id="T_5d348_row20_col3" class="data row20 col3" >0.0045</td>
      <td id="T_5d348_row20_col4" class="data row20 col4" >0.5768</td>
      <td id="T_5d348_row20_col5" class="data row20 col5" >0.0063</td>
      <td id="T_5d348_row20_col6" class="data row20 col6" >351.6</td>
    </tr>
    <tr>
      <td id="T_5d348_row21_col0" class="data row21 col0" >2.0</td>
      <td id="T_5d348_row21_col1" class="data row21 col1" >5.0</td>
      <td id="T_5d348_row21_col2" class="data row21 col2" >0.8652</td>
      <td id="T_5d348_row21_col3" class="data row21 col3" >0.0046</td>
      <td id="T_5d348_row21_col4" class="data row21 col4" >0.5768</td>
      <td id="T_5d348_row21_col5" class="data row21 col5" >0.0062</td>
      <td id="T_5d348_row21_col6" class="data row21 col6" >353.6</td>
    </tr>
    <tr>
      <td id="T_5d348_row22_col0" class="data row22 col0" >2.0</td>
      <td id="T_5d348_row22_col1" class="data row22 col1" >10.0</td>
      <td id="T_5d348_row22_col2" class="data row22 col2" >0.865</td>
      <td id="T_5d348_row22_col3" class="data row22 col3" >0.0041</td>
      <td id="T_5d348_row22_col4" class="data row22 col4" >0.5766</td>
      <td id="T_5d348_row22_col5" class="data row22 col5" >0.0059</td>
      <td id="T_5d348_row22_col6" class="data row22 col6" >318.2</td>
    </tr>
    <tr>
      <td id="T_5d348_row23_col0" class="data row23 col0" >2.0</td>
      <td id="T_5d348_row23_col1" class="data row23 col1" >3.0</td>
      <td id="T_5d348_row23_col2" class="data row23 col2" >0.865</td>
      <td id="T_5d348_row23_col3" class="data row23 col3" >0.0044</td>
      <td id="T_5d348_row23_col4" class="data row23 col4" >0.5772</td>
      <td id="T_5d348_row23_col5" class="data row23 col5" >0.0054</td>
      <td id="T_5d348_row23_col6" class="data row23 col6" >319.2</td>
    </tr>
    <tr>
      <td id="T_5d348_row24_col0" class="data row24 col0" >7.0</td>
      <td id="T_5d348_row24_col1" class="data row24 col1" >10.0</td>
      <td id="T_5d348_row24_col2" class="data row24 col2" >0.8648</td>
      <td id="T_5d348_row24_col3" class="data row24 col3" >0.0174</td>
      <td id="T_5d348_row24_col4" class="data row24 col4" >0.5755</td>
      <td id="T_5d348_row24_col5" class="data row24 col5" >0.031</td>
      <td id="T_5d348_row24_col6" class="data row24 col6" >41.4</td>
    </tr>
    <tr>
      <td id="T_5d348_row25_col0" class="data row25 col0" >7.0</td>
      <td id="T_5d348_row25_col1" class="data row25 col1" >3.0</td>
      <td id="T_5d348_row25_col2" class="data row25 col2" >0.8648</td>
      <td id="T_5d348_row25_col3" class="data row25 col3" >0.023</td>
      <td id="T_5d348_row25_col4" class="data row25 col4" >0.5752</td>
      <td id="T_5d348_row25_col5" class="data row25 col5" >0.0424</td>
      <td id="T_5d348_row25_col6" class="data row25 col6" >49.8</td>
    </tr>
    <tr>
      <td id="T_5d348_row26_col0" class="data row26 col0" >2.0</td>
      <td id="T_5d348_row26_col1" class="data row26 col1" >1.0</td>
      <td id="T_5d348_row26_col2" class="data row26 col2" >0.8648</td>
      <td id="T_5d348_row26_col3" class="data row26 col3" >0.0041</td>
      <td id="T_5d348_row26_col4" class="data row26 col4" >0.5767</td>
      <td id="T_5d348_row26_col5" class="data row26 col5" >0.0051</td>
      <td id="T_5d348_row26_col6" class="data row26 col6" >272.4</td>
    </tr>
    <tr>
      <td id="T_5d348_row27_col0" class="data row27 col0" >7.0</td>
      <td id="T_5d348_row27_col1" class="data row27 col1" >7.0</td>
      <td id="T_5d348_row27_col2" class="data row27 col2" >0.8647</td>
      <td id="T_5d348_row27_col3" class="data row27 col3" >0.0197</td>
      <td id="T_5d348_row27_col4" class="data row27 col4" >0.575</td>
      <td id="T_5d348_row27_col5" class="data row27 col5" >0.0363</td>
      <td id="T_5d348_row27_col6" class="data row27 col6" >44.0</td>
    </tr>
    <tr>
      <td id="T_5d348_row28_col0" class="data row28 col0" >7.0</td>
      <td id="T_5d348_row28_col1" class="data row28 col1" >5.0</td>
      <td id="T_5d348_row28_col2" class="data row28 col2" >0.8646</td>
      <td id="T_5d348_row28_col3" class="data row28 col3" >0.0196</td>
      <td id="T_5d348_row28_col4" class="data row28 col4" >0.5765</td>
      <td id="T_5d348_row28_col5" class="data row28 col5" >0.0348</td>
      <td id="T_5d348_row28_col6" class="data row28 col6" >39.6</td>
    </tr>
    <tr>
      <td id="T_5d348_row29_col0" class="data row29 col0" >7.0</td>
      <td id="T_5d348_row29_col1" class="data row29 col1" >1.0</td>
      <td id="T_5d348_row29_col2" class="data row29 col2" >0.8646</td>
      <td id="T_5d348_row29_col3" class="data row29 col3" >0.0225</td>
      <td id="T_5d348_row29_col4" class="data row29 col4" >0.5745</td>
      <td id="T_5d348_row29_col5" class="data row29 col5" >0.0426</td>
      <td id="T_5d348_row29_col6" class="data row29 col6" >38.6</td>
    </tr>
  </tbody>
</table>




```python
# 整合五折交叉验证得到的五个XGBoost第3轮网格搜索的结果表
if is_processing_here5:
    xgb_fit_goodness_round3_df = test_fit_goodness_xgb(xgb_auc_ks_round3_df_list)
    print_df(xgb_fit_goodness_round3_df)
```


<style type="text/css">
</style>
<table id="T_c1415">
  <thead>
    <tr>
      <th id="T_c1415_level0_col0" class="col_heading level0 col0" >learning_rate</th>
      <th id="T_c1415_level0_col1" class="col_heading level0 col1" >n_estimators</th>
      <th id="T_c1415_level0_col2" class="col_heading level0 col2" >valid_AUC_mean</th>
      <th id="T_c1415_level0_col3" class="col_heading level0 col3" >AUC_gap_mean</th>
      <th id="T_c1415_level0_col4" class="col_heading level0 col4" >valid_KS_mean</th>
      <th id="T_c1415_level0_col5" class="col_heading level0 col5" >KS_gap_mean</th>
      <th id="T_c1415_level0_col6" class="col_heading level0 col6" >best_iteration_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_c1415_row0_col0" class="data row0 col0" >0.03</td>
      <td id="T_c1415_row0_col1" class="data row0 col1" >500.0</td>
      <td id="T_c1415_row0_col2" class="data row0 col2" >0.866</td>
      <td id="T_c1415_row0_col3" class="data row0 col3" >0.0064</td>
      <td id="T_c1415_row0_col4" class="data row0 col4" >0.5768</td>
      <td id="T_c1415_row0_col5" class="data row0 col5" >0.0101</td>
      <td id="T_c1415_row0_col6" class="data row0 col6" >345.0</td>
    </tr>
    <tr>
      <td id="T_c1415_row1_col0" class="data row1 col0" >0.03</td>
      <td id="T_c1415_row1_col1" class="data row1 col1" >800.0</td>
      <td id="T_c1415_row1_col2" class="data row1 col2" >0.866</td>
      <td id="T_c1415_row1_col3" class="data row1 col3" >0.0064</td>
      <td id="T_c1415_row1_col4" class="data row1 col4" >0.5768</td>
      <td id="T_c1415_row1_col5" class="data row1 col5" >0.0101</td>
      <td id="T_c1415_row1_col6" class="data row1 col6" >345.0</td>
    </tr>
    <tr>
      <td id="T_c1415_row2_col0" class="data row2 col0" >0.03</td>
      <td id="T_c1415_row2_col1" class="data row2 col1" >1000.0</td>
      <td id="T_c1415_row2_col2" class="data row2 col2" >0.866</td>
      <td id="T_c1415_row2_col3" class="data row2 col3" >0.0064</td>
      <td id="T_c1415_row2_col4" class="data row2 col4" >0.5768</td>
      <td id="T_c1415_row2_col5" class="data row2 col5" >0.0101</td>
      <td id="T_c1415_row2_col6" class="data row2 col6" >345.0</td>
    </tr>
    <tr>
      <td id="T_c1415_row3_col0" class="data row3 col0" >0.03</td>
      <td id="T_c1415_row3_col1" class="data row3 col1" >1500.0</td>
      <td id="T_c1415_row3_col2" class="data row3 col2" >0.866</td>
      <td id="T_c1415_row3_col3" class="data row3 col3" >0.0064</td>
      <td id="T_c1415_row3_col4" class="data row3 col4" >0.5768</td>
      <td id="T_c1415_row3_col5" class="data row3 col5" >0.0101</td>
      <td id="T_c1415_row3_col6" class="data row3 col6" >345.0</td>
    </tr>
    <tr>
      <td id="T_c1415_row4_col0" class="data row4 col0" >0.1</td>
      <td id="T_c1415_row4_col1" class="data row4 col1" >300.0</td>
      <td id="T_c1415_row4_col2" class="data row4 col2" >0.8659</td>
      <td id="T_c1415_row4_col3" class="data row4 col3" >0.0071</td>
      <td id="T_c1415_row4_col4" class="data row4 col4" >0.5767</td>
      <td id="T_c1415_row4_col5" class="data row4 col5" >0.0112</td>
      <td id="T_c1415_row4_col6" class="data row4 col6" >125.2</td>
    </tr>
    <tr>
      <td id="T_c1415_row5_col0" class="data row5 col0" >0.1</td>
      <td id="T_c1415_row5_col1" class="data row5 col1" >500.0</td>
      <td id="T_c1415_row5_col2" class="data row5 col2" >0.8659</td>
      <td id="T_c1415_row5_col3" class="data row5 col3" >0.0071</td>
      <td id="T_c1415_row5_col4" class="data row5 col4" >0.5767</td>
      <td id="T_c1415_row5_col5" class="data row5 col5" >0.0112</td>
      <td id="T_c1415_row5_col6" class="data row5 col6" >125.2</td>
    </tr>
    <tr>
      <td id="T_c1415_row6_col0" class="data row6 col0" >0.1</td>
      <td id="T_c1415_row6_col1" class="data row6 col1" >800.0</td>
      <td id="T_c1415_row6_col2" class="data row6 col2" >0.8659</td>
      <td id="T_c1415_row6_col3" class="data row6 col3" >0.0071</td>
      <td id="T_c1415_row6_col4" class="data row6 col4" >0.5767</td>
      <td id="T_c1415_row6_col5" class="data row6 col5" >0.0112</td>
      <td id="T_c1415_row6_col6" class="data row6 col6" >125.2</td>
    </tr>
    <tr>
      <td id="T_c1415_row7_col0" class="data row7 col0" >0.1</td>
      <td id="T_c1415_row7_col1" class="data row7 col1" >1000.0</td>
      <td id="T_c1415_row7_col2" class="data row7 col2" >0.8659</td>
      <td id="T_c1415_row7_col3" class="data row7 col3" >0.0071</td>
      <td id="T_c1415_row7_col4" class="data row7 col4" >0.5767</td>
      <td id="T_c1415_row7_col5" class="data row7 col5" >0.0112</td>
      <td id="T_c1415_row7_col6" class="data row7 col6" >125.2</td>
    </tr>
    <tr>
      <td id="T_c1415_row8_col0" class="data row8 col0" >0.1</td>
      <td id="T_c1415_row8_col1" class="data row8 col1" >1500.0</td>
      <td id="T_c1415_row8_col2" class="data row8 col2" >0.8659</td>
      <td id="T_c1415_row8_col3" class="data row8 col3" >0.0071</td>
      <td id="T_c1415_row8_col4" class="data row8 col4" >0.5767</td>
      <td id="T_c1415_row8_col5" class="data row8 col5" >0.0112</td>
      <td id="T_c1415_row8_col6" class="data row8 col6" >125.2</td>
    </tr>
    <tr>
      <td id="T_c1415_row9_col0" class="data row9 col0" >0.05</td>
      <td id="T_c1415_row9_col1" class="data row9 col1" >300.0</td>
      <td id="T_c1415_row9_col2" class="data row9 col2" >0.8659</td>
      <td id="T_c1415_row9_col3" class="data row9 col3" >0.0061</td>
      <td id="T_c1415_row9_col4" class="data row9 col4" >0.5758</td>
      <td id="T_c1415_row9_col5" class="data row9 col5" >0.0105</td>
      <td id="T_c1415_row9_col6" class="data row9 col6" >195.4</td>
    </tr>
    <tr>
      <td id="T_c1415_row10_col0" class="data row10 col0" >0.05</td>
      <td id="T_c1415_row10_col1" class="data row10 col1" >500.0</td>
      <td id="T_c1415_row10_col2" class="data row10 col2" >0.8659</td>
      <td id="T_c1415_row10_col3" class="data row10 col3" >0.0061</td>
      <td id="T_c1415_row10_col4" class="data row10 col4" >0.5758</td>
      <td id="T_c1415_row10_col5" class="data row10 col5" >0.0105</td>
      <td id="T_c1415_row10_col6" class="data row10 col6" >195.4</td>
    </tr>
    <tr>
      <td id="T_c1415_row11_col0" class="data row11 col0" >0.05</td>
      <td id="T_c1415_row11_col1" class="data row11 col1" >800.0</td>
      <td id="T_c1415_row11_col2" class="data row11 col2" >0.8659</td>
      <td id="T_c1415_row11_col3" class="data row11 col3" >0.0061</td>
      <td id="T_c1415_row11_col4" class="data row11 col4" >0.5758</td>
      <td id="T_c1415_row11_col5" class="data row11 col5" >0.0105</td>
      <td id="T_c1415_row11_col6" class="data row11 col6" >195.4</td>
    </tr>
    <tr>
      <td id="T_c1415_row12_col0" class="data row12 col0" >0.05</td>
      <td id="T_c1415_row12_col1" class="data row12 col1" >1000.0</td>
      <td id="T_c1415_row12_col2" class="data row12 col2" >0.8659</td>
      <td id="T_c1415_row12_col3" class="data row12 col3" >0.0061</td>
      <td id="T_c1415_row12_col4" class="data row12 col4" >0.5758</td>
      <td id="T_c1415_row12_col5" class="data row12 col5" >0.0105</td>
      <td id="T_c1415_row12_col6" class="data row12 col6" >195.4</td>
    </tr>
    <tr>
      <td id="T_c1415_row13_col0" class="data row13 col0" >0.05</td>
      <td id="T_c1415_row13_col1" class="data row13 col1" >1500.0</td>
      <td id="T_c1415_row13_col2" class="data row13 col2" >0.8659</td>
      <td id="T_c1415_row13_col3" class="data row13 col3" >0.0061</td>
      <td id="T_c1415_row13_col4" class="data row13 col4" >0.5758</td>
      <td id="T_c1415_row13_col5" class="data row13 col5" >0.0105</td>
      <td id="T_c1415_row13_col6" class="data row13 col6" >195.4</td>
    </tr>
    <tr>
      <td id="T_c1415_row14_col0" class="data row14 col0" >0.03</td>
      <td id="T_c1415_row14_col1" class="data row14 col1" >300.0</td>
      <td id="T_c1415_row14_col2" class="data row14 col2" >0.8658</td>
      <td id="T_c1415_row14_col3" class="data row14 col3" >0.0057</td>
      <td id="T_c1415_row14_col4" class="data row14 col4" >0.577</td>
      <td id="T_c1415_row14_col5" class="data row14 col5" >0.0083</td>
      <td id="T_c1415_row14_col6" class="data row14 col6" >288.6</td>
    </tr>
    <tr>
      <td id="T_c1415_row15_col0" class="data row15 col0" >0.2</td>
      <td id="T_c1415_row15_col1" class="data row15 col1" >300.0</td>
      <td id="T_c1415_row15_col2" class="data row15 col2" >0.8655</td>
      <td id="T_c1415_row15_col3" class="data row15 col3" >0.0073</td>
      <td id="T_c1415_row15_col4" class="data row15 col4" >0.5769</td>
      <td id="T_c1415_row15_col5" class="data row15 col5" >0.0103</td>
      <td id="T_c1415_row15_col6" class="data row15 col6" >66.2</td>
    </tr>
    <tr>
      <td id="T_c1415_row16_col0" class="data row16 col0" >0.2</td>
      <td id="T_c1415_row16_col1" class="data row16 col1" >500.0</td>
      <td id="T_c1415_row16_col2" class="data row16 col2" >0.8655</td>
      <td id="T_c1415_row16_col3" class="data row16 col3" >0.0073</td>
      <td id="T_c1415_row16_col4" class="data row16 col4" >0.5769</td>
      <td id="T_c1415_row16_col5" class="data row16 col5" >0.0103</td>
      <td id="T_c1415_row16_col6" class="data row16 col6" >66.2</td>
    </tr>
    <tr>
      <td id="T_c1415_row17_col0" class="data row17 col0" >0.2</td>
      <td id="T_c1415_row17_col1" class="data row17 col1" >800.0</td>
      <td id="T_c1415_row17_col2" class="data row17 col2" >0.8655</td>
      <td id="T_c1415_row17_col3" class="data row17 col3" >0.0073</td>
      <td id="T_c1415_row17_col4" class="data row17 col4" >0.5769</td>
      <td id="T_c1415_row17_col5" class="data row17 col5" >0.0103</td>
      <td id="T_c1415_row17_col6" class="data row17 col6" >66.2</td>
    </tr>
    <tr>
      <td id="T_c1415_row18_col0" class="data row18 col0" >0.2</td>
      <td id="T_c1415_row18_col1" class="data row18 col1" >1000.0</td>
      <td id="T_c1415_row18_col2" class="data row18 col2" >0.8655</td>
      <td id="T_c1415_row18_col3" class="data row18 col3" >0.0073</td>
      <td id="T_c1415_row18_col4" class="data row18 col4" >0.5769</td>
      <td id="T_c1415_row18_col5" class="data row18 col5" >0.0103</td>
      <td id="T_c1415_row18_col6" class="data row18 col6" >66.2</td>
    </tr>
    <tr>
      <td id="T_c1415_row19_col0" class="data row19 col0" >0.2</td>
      <td id="T_c1415_row19_col1" class="data row19 col1" >1500.0</td>
      <td id="T_c1415_row19_col2" class="data row19 col2" >0.8655</td>
      <td id="T_c1415_row19_col3" class="data row19 col3" >0.0073</td>
      <td id="T_c1415_row19_col4" class="data row19 col4" >0.5769</td>
      <td id="T_c1415_row19_col5" class="data row19 col5" >0.0103</td>
      <td id="T_c1415_row19_col6" class="data row19 col6" >66.2</td>
    </tr>
    <tr>
      <td id="T_c1415_row20_col0" class="data row20 col0" >0.01</td>
      <td id="T_c1415_row20_col1" class="data row20 col1" >1500.0</td>
      <td id="T_c1415_row20_col2" class="data row20 col2" >0.8641</td>
      <td id="T_c1415_row20_col3" class="data row20 col3" >0.0048</td>
      <td id="T_c1415_row20_col4" class="data row20 col4" >0.5728</td>
      <td id="T_c1415_row20_col5" class="data row20 col5" >0.0077</td>
      <td id="T_c1415_row20_col6" class="data row20 col6" >734.0</td>
    </tr>
    <tr>
      <td id="T_c1415_row21_col0" class="data row21 col0" >0.01</td>
      <td id="T_c1415_row21_col1" class="data row21 col1" >1000.0</td>
      <td id="T_c1415_row21_col2" class="data row21 col2" >0.864</td>
      <td id="T_c1415_row21_col3" class="data row21 col3" >0.0047</td>
      <td id="T_c1415_row21_col4" class="data row21 col4" >0.5727</td>
      <td id="T_c1415_row21_col5" class="data row21 col5" >0.0078</td>
      <td id="T_c1415_row21_col6" class="data row21 col6" >715.4</td>
    </tr>
    <tr>
      <td id="T_c1415_row22_col0" class="data row22 col0" >0.01</td>
      <td id="T_c1415_row22_col1" class="data row22 col1" >800.0</td>
      <td id="T_c1415_row22_col2" class="data row22 col2" >0.8639</td>
      <td id="T_c1415_row22_col3" class="data row22 col3" >0.0044</td>
      <td id="T_c1415_row22_col4" class="data row22 col4" >0.5726</td>
      <td id="T_c1415_row22_col5" class="data row22 col5" >0.0069</td>
      <td id="T_c1415_row22_col6" class="data row22 col6" >638.6</td>
    </tr>
    <tr>
      <td id="T_c1415_row23_col0" class="data row23 col0" >0.5</td>
      <td id="T_c1415_row23_col1" class="data row23 col1" >300.0</td>
      <td id="T_c1415_row23_col2" class="data row23 col2" >0.8635</td>
      <td id="T_c1415_row23_col3" class="data row23 col3" >0.0066</td>
      <td id="T_c1415_row23_col4" class="data row23 col4" >0.5726</td>
      <td id="T_c1415_row23_col5" class="data row23 col5" >0.0085</td>
      <td id="T_c1415_row23_col6" class="data row23 col6" >26.6</td>
    </tr>
    <tr>
      <td id="T_c1415_row24_col0" class="data row24 col0" >0.5</td>
      <td id="T_c1415_row24_col1" class="data row24 col1" >500.0</td>
      <td id="T_c1415_row24_col2" class="data row24 col2" >0.8635</td>
      <td id="T_c1415_row24_col3" class="data row24 col3" >0.0066</td>
      <td id="T_c1415_row24_col4" class="data row24 col4" >0.5726</td>
      <td id="T_c1415_row24_col5" class="data row24 col5" >0.0085</td>
      <td id="T_c1415_row24_col6" class="data row24 col6" >26.6</td>
    </tr>
    <tr>
      <td id="T_c1415_row25_col0" class="data row25 col0" >0.5</td>
      <td id="T_c1415_row25_col1" class="data row25 col1" >800.0</td>
      <td id="T_c1415_row25_col2" class="data row25 col2" >0.8635</td>
      <td id="T_c1415_row25_col3" class="data row25 col3" >0.0066</td>
      <td id="T_c1415_row25_col4" class="data row25 col4" >0.5726</td>
      <td id="T_c1415_row25_col5" class="data row25 col5" >0.0085</td>
      <td id="T_c1415_row25_col6" class="data row25 col6" >26.6</td>
    </tr>
    <tr>
      <td id="T_c1415_row26_col0" class="data row26 col0" >0.5</td>
      <td id="T_c1415_row26_col1" class="data row26 col1" >1000.0</td>
      <td id="T_c1415_row26_col2" class="data row26 col2" >0.8635</td>
      <td id="T_c1415_row26_col3" class="data row26 col3" >0.0066</td>
      <td id="T_c1415_row26_col4" class="data row26 col4" >0.5726</td>
      <td id="T_c1415_row26_col5" class="data row26 col5" >0.0085</td>
      <td id="T_c1415_row26_col6" class="data row26 col6" >26.6</td>
    </tr>
    <tr>
      <td id="T_c1415_row27_col0" class="data row27 col0" >0.5</td>
      <td id="T_c1415_row27_col1" class="data row27 col1" >1500.0</td>
      <td id="T_c1415_row27_col2" class="data row27 col2" >0.8635</td>
      <td id="T_c1415_row27_col3" class="data row27 col3" >0.0066</td>
      <td id="T_c1415_row27_col4" class="data row27 col4" >0.5726</td>
      <td id="T_c1415_row27_col5" class="data row27 col5" >0.0085</td>
      <td id="T_c1415_row27_col6" class="data row27 col6" >26.6</td>
    </tr>
    <tr>
      <td id="T_c1415_row28_col0" class="data row28 col0" >0.01</td>
      <td id="T_c1415_row28_col1" class="data row28 col1" >500.0</td>
      <td id="T_c1415_row28_col2" class="data row28 col2" >0.8631</td>
      <td id="T_c1415_row28_col3" class="data row28 col3" >0.0033</td>
      <td id="T_c1415_row28_col4" class="data row28 col4" >0.5716</td>
      <td id="T_c1415_row28_col5" class="data row28 col5" >0.0047</td>
      <td id="T_c1415_row28_col6" class="data row28 col6" >411.8</td>
    </tr>
    <tr>
      <td id="T_c1415_row29_col0" class="data row29 col0" >0.01</td>
      <td id="T_c1415_row29_col1" class="data row29 col1" >300.0</td>
      <td id="T_c1415_row29_col2" class="data row29 col2" >0.8615</td>
      <td id="T_c1415_row29_col3" class="data row29 col3" >0.0024</td>
      <td id="T_c1415_row29_col4" class="data row29 col4" >0.5679</td>
      <td id="T_c1415_row29_col5" class="data row29 col5" >0.003</td>
      <td id="T_c1415_row29_col6" class="data row29 col6" >252.2</td>
    </tr>
  </tbody>
</table>




```python
# 整合五折交叉验证得到的五个XGBoost第4轮网格搜索的结果表
if is_processing_here5:
    xgb_fit_goodness_round4_df = test_fit_goodness_xgb(xgb_auc_ks_round4_df_list)
    print_df(xgb_fit_goodness_round4_df)
```


<style type="text/css">
</style>
<table id="T_8afba">
  <thead>
    <tr>
      <th id="T_8afba_level0_col0" class="col_heading level0 col0" >subsample</th>
      <th id="T_8afba_level0_col1" class="col_heading level0 col1" >colsample_bytree</th>
      <th id="T_8afba_level0_col2" class="col_heading level0 col2" >valid_AUC_mean</th>
      <th id="T_8afba_level0_col3" class="col_heading level0 col3" >AUC_gap_mean</th>
      <th id="T_8afba_level0_col4" class="col_heading level0 col4" >valid_KS_mean</th>
      <th id="T_8afba_level0_col5" class="col_heading level0 col5" >KS_gap_mean</th>
      <th id="T_8afba_level0_col6" class="col_heading level0 col6" >best_iteration_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_8afba_row0_col0" class="data row0 col0" >0.6</td>
      <td id="T_8afba_row0_col1" class="data row0 col1" >0.9</td>
      <td id="T_8afba_row0_col2" class="data row0 col2" >0.8662</td>
      <td id="T_8afba_row0_col3" class="data row0 col3" >0.0059</td>
      <td id="T_8afba_row0_col4" class="data row0 col4" >0.5772</td>
      <td id="T_8afba_row0_col5" class="data row0 col5" >0.0098</td>
      <td id="T_8afba_row0_col6" class="data row0 col6" >336.2</td>
    </tr>
    <tr>
      <td id="T_8afba_row1_col0" class="data row1 col0" >0.7</td>
      <td id="T_8afba_row1_col1" class="data row1 col1" >0.8</td>
      <td id="T_8afba_row1_col2" class="data row1 col2" >0.8661</td>
      <td id="T_8afba_row1_col3" class="data row1 col3" >0.0064</td>
      <td id="T_8afba_row1_col4" class="data row1 col4" >0.5777</td>
      <td id="T_8afba_row1_col5" class="data row1 col5" >0.009</td>
      <td id="T_8afba_row1_col6" class="data row1 col6" >357.6</td>
    </tr>
    <tr>
      <td id="T_8afba_row2_col0" class="data row2 col0" >0.9</td>
      <td id="T_8afba_row2_col1" class="data row2 col1" >0.8</td>
      <td id="T_8afba_row2_col2" class="data row2 col2" >0.866</td>
      <td id="T_8afba_row2_col3" class="data row2 col3" >0.0066</td>
      <td id="T_8afba_row2_col4" class="data row2 col4" >0.5763</td>
      <td id="T_8afba_row2_col5" class="data row2 col5" >0.0108</td>
      <td id="T_8afba_row2_col6" class="data row2 col6" >382.4</td>
    </tr>
    <tr>
      <td id="T_8afba_row3_col0" class="data row3 col0" >0.8</td>
      <td id="T_8afba_row3_col1" class="data row3 col1" >0.9</td>
      <td id="T_8afba_row3_col2" class="data row3 col2" >0.866</td>
      <td id="T_8afba_row3_col3" class="data row3 col3" >0.007</td>
      <td id="T_8afba_row3_col4" class="data row3 col4" >0.5768</td>
      <td id="T_8afba_row3_col5" class="data row3 col5" >0.0111</td>
      <td id="T_8afba_row3_col6" class="data row3 col6" >396.4</td>
    </tr>
    <tr>
      <td id="T_8afba_row4_col0" class="data row4 col0" >0.7</td>
      <td id="T_8afba_row4_col1" class="data row4 col1" >0.9</td>
      <td id="T_8afba_row4_col2" class="data row4 col2" >0.866</td>
      <td id="T_8afba_row4_col3" class="data row4 col3" >0.0065</td>
      <td id="T_8afba_row4_col4" class="data row4 col4" >0.5764</td>
      <td id="T_8afba_row4_col5" class="data row4 col5" >0.0102</td>
      <td id="T_8afba_row4_col6" class="data row4 col6" >356.6</td>
    </tr>
    <tr>
      <td id="T_8afba_row5_col0" class="data row5 col0" >0.6</td>
      <td id="T_8afba_row5_col1" class="data row5 col1" >0.6</td>
      <td id="T_8afba_row5_col2" class="data row5 col2" >0.866</td>
      <td id="T_8afba_row5_col3" class="data row5 col3" >0.0059</td>
      <td id="T_8afba_row5_col4" class="data row5 col4" >0.5765</td>
      <td id="T_8afba_row5_col5" class="data row5 col5" >0.0092</td>
      <td id="T_8afba_row5_col6" class="data row5 col6" >348.2</td>
    </tr>
    <tr>
      <td id="T_8afba_row6_col0" class="data row6 col0" >0.8</td>
      <td id="T_8afba_row6_col1" class="data row6 col1" >0.8</td>
      <td id="T_8afba_row6_col2" class="data row6 col2" >0.866</td>
      <td id="T_8afba_row6_col3" class="data row6 col3" >0.0064</td>
      <td id="T_8afba_row6_col4" class="data row6 col4" >0.5768</td>
      <td id="T_8afba_row6_col5" class="data row6 col5" >0.0101</td>
      <td id="T_8afba_row6_col6" class="data row6 col6" >345.0</td>
    </tr>
    <tr>
      <td id="T_8afba_row7_col0" class="data row7 col0" >0.8</td>
      <td id="T_8afba_row7_col1" class="data row7 col1" >0.6</td>
      <td id="T_8afba_row7_col2" class="data row7 col2" >0.866</td>
      <td id="T_8afba_row7_col3" class="data row7 col3" >0.0067</td>
      <td id="T_8afba_row7_col4" class="data row7 col4" >0.5767</td>
      <td id="T_8afba_row7_col5" class="data row7 col5" >0.0105</td>
      <td id="T_8afba_row7_col6" class="data row7 col6" >395.2</td>
    </tr>
    <tr>
      <td id="T_8afba_row8_col0" class="data row8 col0" >0.9</td>
      <td id="T_8afba_row8_col1" class="data row8 col1" >0.9</td>
      <td id="T_8afba_row8_col2" class="data row8 col2" >0.866</td>
      <td id="T_8afba_row8_col3" class="data row8 col3" >0.0065</td>
      <td id="T_8afba_row8_col4" class="data row8 col4" >0.5766</td>
      <td id="T_8afba_row8_col5" class="data row8 col5" >0.0102</td>
      <td id="T_8afba_row8_col6" class="data row8 col6" >359.8</td>
    </tr>
    <tr>
      <td id="T_8afba_row9_col0" class="data row9 col0" >0.6</td>
      <td id="T_8afba_row9_col1" class="data row9 col1" >0.8</td>
      <td id="T_8afba_row9_col2" class="data row9 col2" >0.866</td>
      <td id="T_8afba_row9_col3" class="data row9 col3" >0.0059</td>
      <td id="T_8afba_row9_col4" class="data row9 col4" >0.5779</td>
      <td id="T_8afba_row9_col5" class="data row9 col5" >0.008</td>
      <td id="T_8afba_row9_col6" class="data row9 col6" >319.6</td>
    </tr>
    <tr>
      <td id="T_8afba_row10_col0" class="data row10 col0" >0.8</td>
      <td id="T_8afba_row10_col1" class="data row10 col1" >0.7</td>
      <td id="T_8afba_row10_col2" class="data row10 col2" >0.8659</td>
      <td id="T_8afba_row10_col3" class="data row10 col3" >0.0066</td>
      <td id="T_8afba_row10_col4" class="data row10 col4" >0.5772</td>
      <td id="T_8afba_row10_col5" class="data row10 col5" >0.0098</td>
      <td id="T_8afba_row10_col6" class="data row10 col6" >378.2</td>
    </tr>
    <tr>
      <td id="T_8afba_row11_col0" class="data row11 col0" >0.9</td>
      <td id="T_8afba_row11_col1" class="data row11 col1" >0.6</td>
      <td id="T_8afba_row11_col2" class="data row11 col2" >0.8659</td>
      <td id="T_8afba_row11_col3" class="data row11 col3" >0.0066</td>
      <td id="T_8afba_row11_col4" class="data row11 col4" >0.5766</td>
      <td id="T_8afba_row11_col5" class="data row11 col5" >0.0104</td>
      <td id="T_8afba_row11_col6" class="data row11 col6" >384.8</td>
    </tr>
    <tr>
      <td id="T_8afba_row12_col0" class="data row12 col0" >0.6</td>
      <td id="T_8afba_row12_col1" class="data row12 col1" >0.7</td>
      <td id="T_8afba_row12_col2" class="data row12 col2" >0.8658</td>
      <td id="T_8afba_row12_col3" class="data row12 col3" >0.0052</td>
      <td id="T_8afba_row12_col4" class="data row12 col4" >0.5767</td>
      <td id="T_8afba_row12_col5" class="data row12 col5" >0.0077</td>
      <td id="T_8afba_row12_col6" class="data row12 col6" >280.8</td>
    </tr>
    <tr>
      <td id="T_8afba_row13_col0" class="data row13 col0" >0.9</td>
      <td id="T_8afba_row13_col1" class="data row13 col1" >0.7</td>
      <td id="T_8afba_row13_col2" class="data row13 col2" >0.8658</td>
      <td id="T_8afba_row13_col3" class="data row13 col3" >0.0063</td>
      <td id="T_8afba_row13_col4" class="data row13 col4" >0.577</td>
      <td id="T_8afba_row13_col5" class="data row13 col5" >0.0093</td>
      <td id="T_8afba_row13_col6" class="data row13 col6" >342.2</td>
    </tr>
    <tr>
      <td id="T_8afba_row14_col0" class="data row14 col0" >0.7</td>
      <td id="T_8afba_row14_col1" class="data row14 col1" >0.6</td>
      <td id="T_8afba_row14_col2" class="data row14 col2" >0.8658</td>
      <td id="T_8afba_row14_col3" class="data row14 col3" >0.0056</td>
      <td id="T_8afba_row14_col4" class="data row14 col4" >0.5767</td>
      <td id="T_8afba_row14_col5" class="data row14 col5" >0.0083</td>
      <td id="T_8afba_row14_col6" class="data row14 col6" >309.4</td>
    </tr>
    <tr>
      <td id="T_8afba_row15_col0" class="data row15 col0" >0.7</td>
      <td id="T_8afba_row15_col1" class="data row15 col1" >0.7</td>
      <td id="T_8afba_row15_col2" class="data row15 col2" >0.8658</td>
      <td id="T_8afba_row15_col3" class="data row15 col3" >0.0055</td>
      <td id="T_8afba_row15_col4" class="data row15 col4" >0.5768</td>
      <td id="T_8afba_row15_col5" class="data row15 col5" >0.0086</td>
      <td id="T_8afba_row15_col6" class="data row15 col6" >295.2</td>
    </tr>
  </tbody>
</table>




```python
# 整合五折交叉验证得到的五个XGBoost第5轮网格搜索的结果表
if is_processing_here5:
    xgb_fit_goodness_round5_df = test_fit_goodness_xgb(xgb_auc_ks_round5_df_list)
    print_df(xgb_fit_goodness_round5_df)
```


<style type="text/css">
</style>
<table id="T_db7a1">
  <thead>
    <tr>
      <th id="T_db7a1_level0_col0" class="col_heading level0 col0" >reg_alpha</th>
      <th id="T_db7a1_level0_col1" class="col_heading level0 col1" >reg_lambda</th>
      <th id="T_db7a1_level0_col2" class="col_heading level0 col2" >valid_AUC_mean</th>
      <th id="T_db7a1_level0_col3" class="col_heading level0 col3" >AUC_gap_mean</th>
      <th id="T_db7a1_level0_col4" class="col_heading level0 col4" >valid_KS_mean</th>
      <th id="T_db7a1_level0_col5" class="col_heading level0 col5" >KS_gap_mean</th>
      <th id="T_db7a1_level0_col6" class="col_heading level0 col6" >best_iteration_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_db7a1_row0_col0" class="data row0 col0" >0.0</td>
      <td id="T_db7a1_row0_col1" class="data row0 col1" >4.0</td>
      <td id="T_db7a1_row0_col2" class="data row0 col2" >0.8661</td>
      <td id="T_db7a1_row0_col3" class="data row0 col3" >0.0056</td>
      <td id="T_db7a1_row0_col4" class="data row0 col4" >0.5766</td>
      <td id="T_db7a1_row0_col5" class="data row0 col5" >0.009</td>
      <td id="T_db7a1_row0_col6" class="data row0 col6" >346.4</td>
    </tr>
    <tr>
      <td id="T_db7a1_row1_col0" class="data row1 col0" >0.1</td>
      <td id="T_db7a1_row1_col1" class="data row1 col1" >4.0</td>
      <td id="T_db7a1_row1_col2" class="data row1 col2" >0.866</td>
      <td id="T_db7a1_row1_col3" class="data row1 col3" >0.0057</td>
      <td id="T_db7a1_row1_col4" class="data row1 col4" >0.576</td>
      <td id="T_db7a1_row1_col5" class="data row1 col5" >0.0099</td>
      <td id="T_db7a1_row1_col6" class="data row1 col6" >346.8</td>
    </tr>
    <tr>
      <td id="T_db7a1_row2_col0" class="data row2 col0" >0.25</td>
      <td id="T_db7a1_row2_col1" class="data row2 col1" >4.0</td>
      <td id="T_db7a1_row2_col2" class="data row2 col2" >0.866</td>
      <td id="T_db7a1_row2_col3" class="data row2 col3" >0.0055</td>
      <td id="T_db7a1_row2_col4" class="data row2 col4" >0.5755</td>
      <td id="T_db7a1_row2_col5" class="data row2 col5" >0.0097</td>
      <td id="T_db7a1_row2_col6" class="data row2 col6" >341.2</td>
    </tr>
    <tr>
      <td id="T_db7a1_row3_col0" class="data row3 col0" >1.0</td>
      <td id="T_db7a1_row3_col1" class="data row3 col1" >4.0</td>
      <td id="T_db7a1_row3_col2" class="data row3 col2" >0.866</td>
      <td id="T_db7a1_row3_col3" class="data row3 col3" >0.006</td>
      <td id="T_db7a1_row3_col4" class="data row3 col4" >0.5759</td>
      <td id="T_db7a1_row3_col5" class="data row3 col5" >0.0102</td>
      <td id="T_db7a1_row3_col6" class="data row3 col6" >379.6</td>
    </tr>
    <tr>
      <td id="T_db7a1_row4_col0" class="data row4 col0" >0.1</td>
      <td id="T_db7a1_row4_col1" class="data row4 col1" >2.0</td>
      <td id="T_db7a1_row4_col2" class="data row4 col2" >0.866</td>
      <td id="T_db7a1_row4_col3" class="data row4 col3" >0.0059</td>
      <td id="T_db7a1_row4_col4" class="data row4 col4" >0.5762</td>
      <td id="T_db7a1_row4_col5" class="data row4 col5" >0.0097</td>
      <td id="T_db7a1_row4_col6" class="data row4 col6" >351.0</td>
    </tr>
    <tr>
      <td id="T_db7a1_row5_col0" class="data row5 col0" >1.0</td>
      <td id="T_db7a1_row5_col1" class="data row5 col1" >1.0</td>
      <td id="T_db7a1_row5_col2" class="data row5 col2" >0.866</td>
      <td id="T_db7a1_row5_col3" class="data row5 col3" >0.0057</td>
      <td id="T_db7a1_row5_col4" class="data row5 col4" >0.576</td>
      <td id="T_db7a1_row5_col5" class="data row5 col5" >0.0097</td>
      <td id="T_db7a1_row5_col6" class="data row5 col6" >336.2</td>
    </tr>
    <tr>
      <td id="T_db7a1_row6_col0" class="data row6 col0" >0.25</td>
      <td id="T_db7a1_row6_col1" class="data row6 col1" >7.0</td>
      <td id="T_db7a1_row6_col2" class="data row6 col2" >0.866</td>
      <td id="T_db7a1_row6_col3" class="data row6 col3" >0.0057</td>
      <td id="T_db7a1_row6_col4" class="data row6 col4" >0.5757</td>
      <td id="T_db7a1_row6_col5" class="data row6 col5" >0.0105</td>
      <td id="T_db7a1_row6_col6" class="data row6 col6" >363.0</td>
    </tr>
    <tr>
      <td id="T_db7a1_row7_col0" class="data row7 col0" >1.0</td>
      <td id="T_db7a1_row7_col1" class="data row7 col1" >2.0</td>
      <td id="T_db7a1_row7_col2" class="data row7 col2" >0.866</td>
      <td id="T_db7a1_row7_col3" class="data row7 col3" >0.0058</td>
      <td id="T_db7a1_row7_col4" class="data row7 col4" >0.5758</td>
      <td id="T_db7a1_row7_col5" class="data row7 col5" >0.0102</td>
      <td id="T_db7a1_row7_col6" class="data row7 col6" >343.4</td>
    </tr>
    <tr>
      <td id="T_db7a1_row8_col0" class="data row8 col0" >1.0</td>
      <td id="T_db7a1_row8_col1" class="data row8 col1" >10.0</td>
      <td id="T_db7a1_row8_col2" class="data row8 col2" >0.866</td>
      <td id="T_db7a1_row8_col3" class="data row8 col3" >0.0056</td>
      <td id="T_db7a1_row8_col4" class="data row8 col4" >0.5768</td>
      <td id="T_db7a1_row8_col5" class="data row8 col5" >0.0087</td>
      <td id="T_db7a1_row8_col6" class="data row8 col6" >376.4</td>
    </tr>
    <tr>
      <td id="T_db7a1_row9_col0" class="data row9 col0" >0.5</td>
      <td id="T_db7a1_row9_col1" class="data row9 col1" >4.0</td>
      <td id="T_db7a1_row9_col2" class="data row9 col2" >0.8659</td>
      <td id="T_db7a1_row9_col3" class="data row9 col3" >0.0057</td>
      <td id="T_db7a1_row9_col4" class="data row9 col4" >0.5756</td>
      <td id="T_db7a1_row9_col5" class="data row9 col5" >0.0101</td>
      <td id="T_db7a1_row9_col6" class="data row9 col6" >340.4</td>
    </tr>
    <tr>
      <td id="T_db7a1_row10_col0" class="data row10 col0" >1.0</td>
      <td id="T_db7a1_row10_col1" class="data row10 col1" >7.0</td>
      <td id="T_db7a1_row10_col2" class="data row10 col2" >0.8659</td>
      <td id="T_db7a1_row10_col3" class="data row10 col3" >0.0057</td>
      <td id="T_db7a1_row10_col4" class="data row10 col4" >0.5762</td>
      <td id="T_db7a1_row10_col5" class="data row10 col5" >0.0096</td>
      <td id="T_db7a1_row10_col6" class="data row10 col6" >362.0</td>
    </tr>
    <tr>
      <td id="T_db7a1_row11_col0" class="data row11 col0" >0.0</td>
      <td id="T_db7a1_row11_col1" class="data row11 col1" >2.0</td>
      <td id="T_db7a1_row11_col2" class="data row11 col2" >0.8659</td>
      <td id="T_db7a1_row11_col3" class="data row11 col3" >0.0056</td>
      <td id="T_db7a1_row11_col4" class="data row11 col4" >0.5766</td>
      <td id="T_db7a1_row11_col5" class="data row11 col5" >0.0086</td>
      <td id="T_db7a1_row11_col6" class="data row11 col6" >316.2</td>
    </tr>
    <tr>
      <td id="T_db7a1_row12_col0" class="data row12 col0" >0.1</td>
      <td id="T_db7a1_row12_col1" class="data row12 col1" >7.0</td>
      <td id="T_db7a1_row12_col2" class="data row12 col2" >0.8659</td>
      <td id="T_db7a1_row12_col3" class="data row12 col3" >0.0055</td>
      <td id="T_db7a1_row12_col4" class="data row12 col4" >0.5758</td>
      <td id="T_db7a1_row12_col5" class="data row12 col5" >0.0091</td>
      <td id="T_db7a1_row12_col6" class="data row12 col6" >338.4</td>
    </tr>
    <tr>
      <td id="T_db7a1_row13_col0" class="data row13 col0" >0.5</td>
      <td id="T_db7a1_row13_col1" class="data row13 col1" >2.0</td>
      <td id="T_db7a1_row13_col2" class="data row13 col2" >0.8659</td>
      <td id="T_db7a1_row13_col3" class="data row13 col3" >0.0055</td>
      <td id="T_db7a1_row13_col4" class="data row13 col4" >0.5765</td>
      <td id="T_db7a1_row13_col5" class="data row13 col5" >0.0085</td>
      <td id="T_db7a1_row13_col6" class="data row13 col6" >311.0</td>
    </tr>
    <tr>
      <td id="T_db7a1_row14_col0" class="data row14 col0" >0.25</td>
      <td id="T_db7a1_row14_col1" class="data row14 col1" >1.0</td>
      <td id="T_db7a1_row14_col2" class="data row14 col2" >0.8658</td>
      <td id="T_db7a1_row14_col3" class="data row14 col3" >0.0055</td>
      <td id="T_db7a1_row14_col4" class="data row14 col4" >0.577</td>
      <td id="T_db7a1_row14_col5" class="data row14 col5" >0.0082</td>
      <td id="T_db7a1_row14_col6" class="data row14 col6" >303.0</td>
    </tr>
    <tr>
      <td id="T_db7a1_row15_col0" class="data row15 col0" >0.0</td>
      <td id="T_db7a1_row15_col1" class="data row15 col1" >1.0</td>
      <td id="T_db7a1_row15_col2" class="data row15 col2" >0.8658</td>
      <td id="T_db7a1_row15_col3" class="data row15 col3" >0.0056</td>
      <td id="T_db7a1_row15_col4" class="data row15 col4" >0.5767</td>
      <td id="T_db7a1_row15_col5" class="data row15 col5" >0.0083</td>
      <td id="T_db7a1_row15_col6" class="data row15 col6" >309.4</td>
    </tr>
    <tr>
      <td id="T_db7a1_row16_col0" class="data row16 col0" >0.25</td>
      <td id="T_db7a1_row16_col1" class="data row16 col1" >2.0</td>
      <td id="T_db7a1_row16_col2" class="data row16 col2" >0.8658</td>
      <td id="T_db7a1_row16_col3" class="data row16 col3" >0.0054</td>
      <td id="T_db7a1_row16_col4" class="data row16 col4" >0.5765</td>
      <td id="T_db7a1_row16_col5" class="data row16 col5" >0.0083</td>
      <td id="T_db7a1_row16_col6" class="data row16 col6" >304.2</td>
    </tr>
    <tr>
      <td id="T_db7a1_row17_col0" class="data row17 col0" >0.5</td>
      <td id="T_db7a1_row17_col1" class="data row17 col1" >7.0</td>
      <td id="T_db7a1_row17_col2" class="data row17 col2" >0.8658</td>
      <td id="T_db7a1_row17_col3" class="data row17 col3" >0.0051</td>
      <td id="T_db7a1_row17_col4" class="data row17 col4" >0.5761</td>
      <td id="T_db7a1_row17_col5" class="data row17 col5" >0.0086</td>
      <td id="T_db7a1_row17_col6" class="data row17 col6" >326.8</td>
    </tr>
    <tr>
      <td id="T_db7a1_row18_col0" class="data row18 col0" >0.5</td>
      <td id="T_db7a1_row18_col1" class="data row18 col1" >1.0</td>
      <td id="T_db7a1_row18_col2" class="data row18 col2" >0.8658</td>
      <td id="T_db7a1_row18_col3" class="data row18 col3" >0.0053</td>
      <td id="T_db7a1_row18_col4" class="data row18 col4" >0.5765</td>
      <td id="T_db7a1_row18_col5" class="data row18 col5" >0.0083</td>
      <td id="T_db7a1_row18_col6" class="data row18 col6" >292.0</td>
    </tr>
    <tr>
      <td id="T_db7a1_row19_col0" class="data row19 col0" >0.0</td>
      <td id="T_db7a1_row19_col1" class="data row19 col1" >7.0</td>
      <td id="T_db7a1_row19_col2" class="data row19 col2" >0.8657</td>
      <td id="T_db7a1_row19_col3" class="data row19 col3" >0.0053</td>
      <td id="T_db7a1_row19_col4" class="data row19 col4" >0.5756</td>
      <td id="T_db7a1_row19_col5" class="data row19 col5" >0.0089</td>
      <td id="T_db7a1_row19_col6" class="data row19 col6" >313.0</td>
    </tr>
    <tr>
      <td id="T_db7a1_row20_col0" class="data row20 col0" >0.1</td>
      <td id="T_db7a1_row20_col1" class="data row20 col1" >1.0</td>
      <td id="T_db7a1_row20_col2" class="data row20 col2" >0.8657</td>
      <td id="T_db7a1_row20_col3" class="data row20 col3" >0.005</td>
      <td id="T_db7a1_row20_col4" class="data row20 col4" >0.5762</td>
      <td id="T_db7a1_row20_col5" class="data row20 col5" >0.008</td>
      <td id="T_db7a1_row20_col6" class="data row20 col6" >270.8</td>
    </tr>
    <tr>
      <td id="T_db7a1_row21_col0" class="data row21 col0" >0.1</td>
      <td id="T_db7a1_row21_col1" class="data row21 col1" >10.0</td>
      <td id="T_db7a1_row21_col2" class="data row21 col2" >0.8641</td>
      <td id="T_db7a1_row21_col3" class="data row21 col3" >0.0046</td>
      <td id="T_db7a1_row21_col4" class="data row21 col4" >0.5738</td>
      <td id="T_db7a1_row21_col5" class="data row21 col5" >0.0066</td>
      <td id="T_db7a1_row21_col6" class="data row21 col6" >286.8</td>
    </tr>
    <tr>
      <td id="T_db7a1_row22_col0" class="data row22 col0" >0.0</td>
      <td id="T_db7a1_row22_col1" class="data row22 col1" >10.0</td>
      <td id="T_db7a1_row22_col2" class="data row22 col2" >0.864</td>
      <td id="T_db7a1_row22_col3" class="data row22 col3" >0.0043</td>
      <td id="T_db7a1_row22_col4" class="data row22 col4" >0.574</td>
      <td id="T_db7a1_row22_col5" class="data row22 col5" >0.0057</td>
      <td id="T_db7a1_row22_col6" class="data row22 col6" >261.4</td>
    </tr>
    <tr>
      <td id="T_db7a1_row23_col0" class="data row23 col0" >0.5</td>
      <td id="T_db7a1_row23_col1" class="data row23 col1" >10.0</td>
      <td id="T_db7a1_row23_col2" class="data row23 col2" >0.8639</td>
      <td id="T_db7a1_row23_col3" class="data row23 col3" >0.0043</td>
      <td id="T_db7a1_row23_col4" class="data row23 col4" >0.5737</td>
      <td id="T_db7a1_row23_col5" class="data row23 col5" >0.0059</td>
      <td id="T_db7a1_row23_col6" class="data row23 col6" >263.8</td>
    </tr>
    <tr>
      <td id="T_db7a1_row24_col0" class="data row24 col0" >0.25</td>
      <td id="T_db7a1_row24_col1" class="data row24 col1" >10.0</td>
      <td id="T_db7a1_row24_col2" class="data row24 col2" >0.8639</td>
      <td id="T_db7a1_row24_col3" class="data row24 col3" >0.0043</td>
      <td id="T_db7a1_row24_col4" class="data row24 col4" >0.5738</td>
      <td id="T_db7a1_row24_col5" class="data row24 col5" >0.0054</td>
      <td id="T_db7a1_row24_col6" class="data row24 col6" >264.2</td>
    </tr>
  </tbody>
</table>




```python
""" 确定对于XGBoost的最佳拟合指标与参数组合 """
# region
# 最佳拟合指标
colnames_to_fit_xgb = [
    # 原始指标
    age, util, debt,
    income, dep, credit, mortgage,
    late30, late60, late90,
    # 衍生指标
    'free_cashflow_income', 'credit_pressure_index', 'credit_late_density',
    'mortgage_ratio', 'late_severity_score'
]
# 最佳模型can
best_para_xgb_dict = dict(
    max_depth = 3,
    min_child_weight = 7,
    learning_rate = 0.03,
    n_estimators = 800,
    subsample = 0.7,
    colsample_bytree = 0.6,
    reg_alpha = 1,
    reg_lambda = 2
)
# 释放内存
cleanup_vars([
    'train_data', 'valid_data', 'train_data_copy', 'valid_data_copy',
    'train_data_xgb', 'valid_data_xgb',
    'X_train_xgb', 'X_valid_xgb', 'y_train', 'y_valid',
    'folds', 'train_idx_list', 'valid_idx_list',
    'xgb_auc_ks_round1_df_list', 'xgb_top_recall_df_list',
    'xgb_auc_ks_round2_df_list', 'xgb_auc_ks_round3_df_list',
    'xgb_auc_ks_round4_df_list', 'xgb_auc_ks_round5_df_list',
    'xgb_auc_ks_round1_df', 'xgb_top_recall_df',
    'xgb_auc_ks_round2_df', 'xgb_auc_ks_round3_df',
    'xgb_auc_ks_round4_df', 'xgb_auc_ks_round5_df',
    'xgb_top_recall_aggregate_df',
    'xgb_fit_goodness_round1_df', 'xgb_fit_goodness_round1_upgraded_df', 
    'xgb_fit_goodness_round2_df', 'xgb_fit_goodness_round3_df', 
    'xgb_fit_goodness_round4_df', 'xgb_fit_goodness_round5_df', 
    'colnames_not_binary', 'div_pts',
    'original_features', 'quality_flags', 'derived_features', 'risk_signals',
    'negative_1_5', 'positive_1_5', 'positive_6_10',
    'colnames_to_fit0', 'colnames_to_fit1', 'colnames_to_fit2',
    'colnames_to_fit3', 'colnames_to_fit4', 'colnames_to_fit5',
    'colnames_to_fit_list_round1', 'colnames_to_fit',
    'derived_features_retained',
    'paras_round2', 'paras_round3', 'paras_round4', 'paras_round5',
    'max_depth_list', 'min_child_weight_list',
    'learning_rate_list', 'n_estimators_list',
    'subsample_list', 'colsample_bytree_list',
    'reg_alpha_list', 'reg_lambda_list',
    'best_para_dict',
    'beta30', 'beta60', 'beta90',
], close_plots=False)
# endregion
```


```python
""" 使用各模型的最佳参数组合，以原训练集与验证集的并集做拟合，观察在测试集中的拟合效果 """
is_processing_here6 = True
train_data = train_valid_data

if is_processing_here6:
    """ 统一数据清洗阶段 """
    # region
    # 截断家属数异常大值
    cap_dep_num(train_data); cap_dep_num(test_data)
    # 删除年龄异常小值
    delete_small_age(train_data); delete_small_age(test_data)
    # 添加信用额度使用率的若干标记
    add_util_flags(train_data); add_util_flags(test_data)
    # 添加负债率的若干标记
    add_debt_flags(train_data); add_debt_flags(test_data)
    # 添加三种逾期次数的加权求和，权重由逻辑回归模型训练
    beta30, beta60, beta90 = add_late_severity_score(train_data)
    add_late_severity_score(test_data, [beta30, beta60, beta90])
    # 添加衍生指标
    add_derived_features(train_data); add_derived_features(test_data)
    # 筛选出所有非二元指标
    colnames_not_binary = get_colnames_whether_binary(train_data, is_binary=False)
    # 对所有非布尔值类型的指标建立一维分箱
    for colname in colnames_not_binary:
        _, div_pts = add_bins_1D(train_data, colname)
        add_bins_1D(test_data, colname, fixed_div_pts=div_pts)
    # 把一维分箱的具体数值转换为排名序号
    convert_bins_value_to_rank(train_data); convert_bins_value_to_rank(test_data)
    # 添加交互标记
    add_intersection_signals(train_data, positive_interactions_df, negative_interactions_df)
    add_intersection_signals(test_data, positive_interactions_df, negative_interactions_df)
    # 暂时删除一维分箱列以及其对应的排名序号列
    colnames_without_bins = [colname for colname in train_data.columns if '_bin' not in colname]
    train_data = train_data[colnames_without_bins]; test_data = test_data[colnames_without_bins]
    # 整理指标顺序
    train_data = reorder_columns(train_data); test_data = reorder_columns(test_data)
    # endregion
    
    """ 对raw LR建立特化表达 """
    # region
    train_data_raw_lr = train_data.copy(); test_data_raw_lr = test_data.copy()
    # 缺失值、异常值全部赋值为0
    train_data_raw_lr = train_data_raw_lr.clip(lower=0)
    test_data_raw_lr = test_data_raw_lr.clip(lower=0)
    # 添加年龄、负债率、信用额度使用率的中心化三列以及中心化的年龄*负债率、年龄*信用额度使用率两列
    centering_means = fit_centering_means(train_data_raw_lr)
    apply_centered_features(train_data_raw_lr, centering_means)
    apply_centered_features(test_data_raw_lr, centering_means)
    # 对有长尾的指标取对数
    colnames_to_log = fit_log_feature_colnames(train_data_raw_lr)
    add_log_features(train_data_raw_lr, fixed_colnames=colnames_to_log)
    add_log_features(test_data_raw_lr, fixed_colnames=colnames_to_log)
    # 筛选将要投入训练的列名
    selected_colnames = select_colnames_raw_lr(train_data_raw_lr)
    train_data_raw_lr = train_data_raw_lr[selected_colnames]
    test_data_raw_lr = test_data_raw_lr[selected_colnames]
    # endregion
    
    """ 对WOE LR建立特化表达 """
    # region
    train_data_woe_lr = train_data.copy(); test_data_woe_lr = test_data.copy()
    # 找出所有的非二元列
    colnames_not_binary = get_colnames_whether_binary(train_data_woe_lr, is_binary=False)
    # 对所有非布尔值类型的指标建立一维分箱（虽然构建了理应被淘汰的一维分箱，但后续控制其进入模型训练即可）
    for colname in colnames_not_binary:
        _, div_pts = add_bins_1D(train_data_woe_lr, colname)
        add_bins_1D(test_data_woe_lr, colname, fixed_div_pts=div_pts)
    # 列出所有一维分箱对应指标的所有无序对
    colname_pairs_not_binary = []
    for i in range(len(colnames_not_binary) - 1):
            for j in range(i + 1, len(colnames_not_binary)):
                colname_pairs_not_binary.append([colnames_not_binary[i], colnames_not_binary[j]])
    # 只建立通过筛选的二维分箱
    for colname_pair in colname_pairs_not_binary:
        _, bins_2D = add_bins_2D(train_data_woe_lr, colname_pair, constraint=passed_binnames_2D)
        add_bins_2D(test_data_woe_lr, colname_pair, fixed_bins_2D=bins_2D, constraint=passed_binnames_2D)
    # 把所有标记列、一维分箱列、二维分箱列转换为WOE列
    binnames_map_to_woe = \
        passed_binary_colnames + passed_binnames_1D + low_iv_binnames_1D + passed_binnames_2D
    for colname in binnames_map_to_woe:
        woe_map = fit_woe_mapping(train_data_woe_lr, colname)
        apply_woe_mapping(train_data_woe_lr, colname, woe_map)
        apply_woe_mapping(test_data_woe_lr, colname, woe_map)
    # 在数据集中只留下投入训练的WOE列
    woe_colnames = [colname for colname in train_data_woe_lr.columns if colname.endswith('_woe')]
    train_data_woe_lr = train_data_woe_lr[[target] + woe_colnames]
    test_data_woe_lr = test_data_woe_lr[[target] + woe_colnames]
    # 重排指标列名
    train_data_woe_lr = reorder_columns(train_data_woe_lr)
    test_data_woe_lr = reorder_columns(test_data_woe_lr)
    # endregion
    
    """ 对XGBoost建立特化表达 """
    # region
    train_data_xgb = train_data.copy(); test_data_xgb = test_data.copy()
    # 将所有表示数据不可靠的负整数转换为NA
    train_data_xgb = train_data_xgb.mask(train_data_xgb < 0, np.nan)
    test_data_xgb = test_data_xgb.mask(test_data_xgb < 0, np.nan)
    # endregion
    
    """ 按确定好的最佳参数分别构建三种模型 """
    # region
    # 拆分目标变量和自变量
    y_train = train_data[target]; y_test = test_data[target]
    X_train_raw_lr = train_data_raw_lr.drop(target, axis=1, inplace=False)
    X_test_raw_lr = test_data_raw_lr.drop(target, axis=1, inplace=False)
    X_train_woe_lr = train_data_woe_lr.drop(target, axis=1, inplace=False)
    X_test_woe_lr = test_data_woe_lr.drop(target, axis=1, inplace=False)
    X_train_xgb = train_data_xgb.drop(target, axis=1, inplace=False)
    X_test_xgb = test_data_xgb.drop(target, axis=1, inplace=False)
    # 释放内存
    if 'train_data' in dir() and 'test_data' in dir():
        del train_data; del test_data
    
    # 构建raw LR
    X_train_raw_lr = X_train_raw_lr[colnames_to_fit_raw_lr]
    X_test_raw_lr = X_test_raw_lr[colnames_to_fit_raw_lr]
    raw_lr_model = LogisticRegression(
        penalty='l2',
        C=c_raw_lr,
        solver='lbfgs',
        max_iter=1000,
        fit_intercept=True
    ) # 设置LR参数
    raw_lr_model.fit(X_train_raw_lr, y_train) # 拟合模型
    
    # 构建WOE LR和评分卡模型
    X_train_woe_lr = X_train_woe_lr[colnames_to_fit_woe_lr]
    X_test_woe_lr = X_test_woe_lr[colnames_to_fit_woe_lr]
    woe_lr_model = LogisticRegression(
        penalty='l2',
        C=c_woe_lr,
        solver='lbfgs',
        max_iter=1000,
        fit_intercept=True
    ) # 设置LR参数
    woe_lr_model.fit(X_train_woe_lr, y_train) # 拟合模型
    sc_model = ScoreCard() # 初始化评分卡类
    sc_model.fit(woe_lr_model, X_train_woe_lr) # 拟合评分卡模型
    score_bins_train_df = sc_model.show_score_bins(X_train_woe_lr, y_train)
    score_bins_test_df = sc_model.show_score_bins(X_test_woe_lr, y_test)
    
    # 构建XGBoost
    X_train_xgb = X_train_xgb[colnames_to_fit_xgb]
    X_test_xgb = X_test_xgb[colnames_to_fit_xgb]
    xgb_model = xgb.XGBClassifier(
        **best_para_xgb_dict,
        random_state=500,
        eval_metric='auc',
        use_label_encoder=False
    ) # 设置XGBoost参数
    xgb_model.fit(
        X_train_xgb, y_train,
        verbose=False
    ) # 拟合模型
    # endregion

```


```python
""" 横向对比三种模型的拟合效果 """
# 量化地对比三种模型的拟合效果
if is_processing_here6:
    model_comparison_df = quantify_model_comparison(
        y_train, y_test,
        X_train_raw_lr, X_test_raw_lr, X_train_woe_lr, X_test_woe_lr, X_train_xgb, X_test_xgb,
        raw_lr_model, sc_model, xgb_model
    )
    print_df(model_comparison_df)
```


<style type="text/css">
</style>
<table id="T_e62da">
  <thead>
    <tr>
      <th id="T_e62da_level0_col0" class="col_heading level0 col0" >model</th>
      <th id="T_e62da_level0_col1" class="col_heading level0 col1" >test AUC</th>
      <th id="T_e62da_level0_col2" class="col_heading level0 col2" >AUC gap</th>
      <th id="T_e62da_level0_col3" class="col_heading level0 col3" >test KS</th>
      <th id="T_e62da_level0_col4" class="col_heading level0 col4" >KS gap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_e62da_row0_col0" class="data row0 col0" >raw LR</td>
      <td id="T_e62da_row0_col1" class="data row0 col1" >0.8581</td>
      <td id="T_e62da_row0_col2" class="data row0 col2" >0.0012</td>
      <td id="T_e62da_row0_col3" class="data row0 col3" >0.5671</td>
      <td id="T_e62da_row0_col4" class="data row0 col4" >-0.0065</td>
    </tr>
    <tr>
      <td id="T_e62da_row1_col0" class="data row1 col0" >WOE LR</td>
      <td id="T_e62da_row1_col1" class="data row1 col1" >0.8596</td>
      <td id="T_e62da_row1_col2" class="data row1 col2" >0.0003</td>
      <td id="T_e62da_row1_col3" class="data row1 col3" >0.5698</td>
      <td id="T_e62da_row1_col4" class="data row1 col4" >-0.0041</td>
    </tr>
    <tr>
      <td id="T_e62da_row2_col0" class="data row2 col0" >XGBoost</td>
      <td id="T_e62da_row2_col1" class="data row2 col1" >0.8661</td>
      <td id="T_e62da_row2_col2" class="data row2 col2" >0.0092</td>
      <td id="T_e62da_row2_col3" class="data row2 col3" >0.5827</td>
      <td id="T_e62da_row2_col4" class="data row2 col4" >0.0094</td>
    </tr>
  </tbody>
</table>




```python
# 可视化地对比三种模型的拟合效果
if is_processing_here6:
    fig_list = visualize_fit_goodness(
        y_test, 
        X_test_raw_lr, X_test_woe_lr, X_test_xgb,
        raw_lr_model, sc_model, xgb_model
    )
    fig_list
```


    
![png](reports/operation_full_output_files/reports/operation_full_output_47_0.png)
    



    
![png](reports/operation_full_output_files/reports/operation_full_output_47_1.png)
    



    
![png](reports/operation_full_output_files/reports/operation_full_output_47_2.png)
    



```python
# 释放最终模块中不再需要的大型中间对象
if is_processing_here6:
    cleanup_vars([
        'train_data_raw_lr', 'test_data_raw_lr',
        'train_data_woe_lr', 'test_data_woe_lr',
        'train_data_xgb', 'test_data_xgb',
        'X_train_raw_lr', 'X_test_raw_lr',
        'X_train_woe_lr', 'X_test_woe_lr',
        'X_train_xgb', 'X_test_xgb',
        'y_train', 'y_test',
        'score_bins_train_df', 'score_bins_test_df',
        'centering_means', 'colnames_to_log', 'selected_colnames',
        'colnames_not_binary', 'colname_pairs_not_binary',
        'binnames_map_to_woe', 'woe_colnames',
        'div_pts', 'bins_2D',
        'beta30', 'beta60', 'beta90',
    ], close_plots=False)

```
