"""Legacy imports for the archived research script and notebook.

New code should import the responsible gmsc module explicitly.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency

from gmsc.schema import *
from gmsc.data import *
from gmsc.cleaning import *
from gmsc.features import *
from gmsc.binning import *
from gmsc.transforms import *
from gmsc.woe import *
from gmsc.scorecard import *
from gmsc.selection import *
from gmsc.legacy_income import *
from gmsc.statistics import *
from gmsc.metrics import *
from gmsc.interpretability import *
from gmsc.summaries import *
from gmsc.plotting_eda import *
from gmsc.plotting_evaluation import *
