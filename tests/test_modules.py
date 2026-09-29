"""Small data contracts for the extracted modeling modules."""

import unittest

import numpy as np
import pandas as pd

from gmsc.binning import add_bins_1D
from gmsc.metrics import calculate_auc_ks
from gmsc.schema import age, target
from gmsc.woe import apply_woe_mapping, fit_woe_mapping


class ModuleContracts(unittest.TestCase):
    def test_training_bins_are_applied_without_refitting(self):
        train = pd.DataFrame({age: [21, 25, 30, 40, 50, 60, 70, 80]})
        valid = pd.DataFrame({age: [25, 55, 75]})
        name, edges = add_bins_1D(train, age, n_bins=4)
        add_bins_1D(valid, age, fixed_div_pts=edges)
        self.assertEqual(name, "age_bin")
        self.assertEqual(train[name].cat.categories.tolist(), valid[name].cat.categories.tolist())
        self.assertTrue(valid[name].notna().all())

    def test_woe_validation_does_not_require_labels(self):
        train = pd.DataFrame({target: [0, 1, 0, 1, 0, 1], "risk_flag": [0, 0, 0, 1, 1, 1]})
        valid = pd.DataFrame({"risk_flag": [1, 0, 2]})
        mapping = fit_woe_mapping(train, "risk_flag")
        apply_woe_mapping(valid, "risk_flag", mapping)
        self.assertEqual(valid.loc[0, "risk_flag_woe"], mapping[1])
        self.assertEqual(valid.loc[2, "risk_flag_woe"], 0.0)

    def test_auc_ks_on_perfect_rank(self):
        auc, ks = calculate_auc_ks(pd.Series([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertAlmostEqual(auc, 1.0)
        self.assertAlmostEqual(ks, 1.0)


if __name__ == "__main__":
    unittest.main()
