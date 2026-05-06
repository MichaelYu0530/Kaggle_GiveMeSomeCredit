# Linux Python 3.13 运行报错

```text
Traceback (most recent call last):
  File "/home/father_yuyue/projects/Kaggle_GiveMeSomeCredits/.venv_py313/lib/python3.13/site-packages/pandas/core/internals/blocks.py", line 1115, in setitem
    casted = np_can_hold_element(values.dtype, value)
  File "/home/father_yuyue/projects/Kaggle_GiveMeSomeCredits/.venv_py313/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1705, in np_can_hold_element
    raise LossySetitemError
pandas.errors.LossySetitemError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 12, in <module>
  File "operation.py", line 505, in <module>
    add_derived_features(train_data); add_derived_features(valid_data)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/tmp/gmsc_py313_run_qV8X6Y/pipeline.py", line 716, in add_derived_features
    data.loc[mask_income_reliable, 'income_per_dep'] = \
    ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/father_yuyue/projects/Kaggle_GiveMeSomeCredits/.venv_py313/lib/python3.13/site-packages/pandas/core/indexing.py", line 938, in __setitem__
    iloc._setitem_with_indexer(indexer, value, self.name)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/father_yuyue/projects/Kaggle_GiveMeSomeCredits/.venv_py313/lib/python3.13/site-packages/pandas/core/indexing.py", line 1953, in _setitem_with_indexer
    self._setitem_with_indexer_split_path(indexer, value, name)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/home/father_yuyue/projects/Kaggle_GiveMeSomeCredits/.venv_py313/lib/python3.13/site-packages/pandas/core/indexing.py", line 1997, in _setitem_with_indexer_split_path
    self._setitem_single_column(ilocs[0], value, pi)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/home/father_yuyue/projects/Kaggle_GiveMeSomeCredits/.venv_py313/lib/python3.13/site-packages/pandas/core/indexing.py", line 2181, in _setitem_single_column
    self.obj._mgr.column_setitem(loc, plane_indexer, value)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/father_yuyue/projects/Kaggle_GiveMeSomeCredits/.venv_py313/lib/python3.13/site-packages/pandas/core/internals/managers.py", line 1520, in column_setitem
    new_mgr = col_mgr.setitem((idx,), value)
  File "/home/father_yuyue/projects/Kaggle_GiveMeSomeCredits/.venv_py313/lib/python3.13/site-packages/pandas/core/internals/managers.py", line 604, in setitem
    return self.apply("setitem", indexer=indexer, value=value)
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/father_yuyue/projects/Kaggle_GiveMeSomeCredits/.venv_py313/lib/python3.13/site-packages/pandas/core/internals/managers.py", line 442, in apply
    applied = getattr(b, f)(**kwargs)
  File "/home/father_yuyue/projects/Kaggle_GiveMeSomeCredits/.venv_py313/lib/python3.13/site-packages/pandas/core/internals/blocks.py", line 1118, in setitem
    nb = self.coerce_to_target_dtype(value, raise_on_upcast=True)
  File "/home/father_yuyue/projects/Kaggle_GiveMeSomeCredits/.venv_py313/lib/python3.13/site-packages/pandas/core/internals/blocks.py", line 468, in coerce_to_target_dtype
    raise TypeError(f"Invalid value '{other}' for dtype '{self.values.dtype}'")
TypeError: Invalid value '[1202.33333333 1160.         1246.         ... 2600.         3601.
 3833.        ]' for dtype 'int64'

```
