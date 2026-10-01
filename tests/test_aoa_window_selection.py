"""Missing observations must extend a window without becoming estimated data."""
import unittest

import numpy as np
import pandas as pd

from aoa_pipeline import (
    generate_aoa_candidates,
    longest_valid_epoch_segment,
    summarize_spoof_windows,
    valid_epoch_segments,
    select_joint_source_aoa,
    select_joint_source_aoa_segments,
)


class WindowSelectionTests(unittest.TestCase):
    def test_two_missing_epochs_join_v9_blocks_without_filling_them(self):
        mask = pd.Series(True, index=range(34))
        mask.loc[[19, 20]] = False
        self.assertEqual(longest_valid_epoch_segment(mask), list(range(19)))
        self.assertEqual(longest_valid_epoch_segment(mask, max_gap_epochs=1), list(range(19)))
        self.assertEqual(longest_valid_epoch_segment(mask, max_gap_epochs=2), list(range(34)))
        self.assertEqual(int(mask.sum()), 32)
        self.assertFalse(mask.loc[19])
        self.assertFalse(mask.loc[20])

    def test_long_gaps_split_windows_and_endpoints_are_never_padded(self):
        mask = pd.Series([False, True, True, False, False, True, False,
                          False, False, True, True, True, False])
        self.assertEqual(longest_valid_epoch_segment(mask, max_gap_epochs=2), list(range(1, 6)))
        self.assertEqual(longest_valid_epoch_segment(mask), [9, 10, 11])
        self.assertEqual(longest_valid_epoch_segment(pd.Series([False, False])), [])

    def test_equal_span_prefers_more_measured_epochs(self):
        mask = pd.Series([True, False, True, False, False, False, True, True, True])
        self.assertEqual(longest_valid_epoch_segment(mask, max_gap_epochs=1), [6, 7, 8])

    def test_every_window_is_listed_in_time_order(self):
        # sp-partial-28-09: the spoof group before and after a cold start.
        mask = pd.Series([True] * 4 + [False] * 3 + [True] * 6 + [False, True])
        self.assertEqual(
            valid_epoch_segments(mask), [[0, 1, 2, 3], list(range(7, 13)), [14]]
        )
        self.assertEqual(
            valid_epoch_segments(mask, max_gap_epochs=1), [[0, 1, 2, 3], list(range(7, 15))]
        )
        tow = pd.Series(np.arange(len(mask)) + 1000.0)
        windows = summarize_spoof_windows(mask, tow, min_window_epochs=2)
        self.assertEqual(windows["window"].tolist(), [1, 2, 3])
        self.assertEqual(windows["start_tow"].tolist(), [1000.0, 1007.0, 1014.0])
        self.assertEqual(windows["solvable"].tolist(), [True, True, False])
        self.assertEqual(summarize_spoof_windows(pd.Series([False]), tow).shape[0], 0)

    def test_gap_policy_rejects_invalid_values(self):
        for limit in (-1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                longest_valid_epoch_segment(pd.Series([True]), max_gap_epochs=limit)


class PartialGroupSelectionTests(unittest.TestCase):
    @staticmethod
    def candidates():
        # A known 38.3-degree common source, reference 66.6 degrees, D/lambda=1.
        dd = np.cos(np.deg2rad(38.3)) - np.cos(np.deg2rad(66.6))
        fractional = pd.DataFrame(dd, index=range(8), columns=[10, 23, 24, 25])
        fractional.loc[1, [24, 25]] = np.nan  # Two PRNs still support an estimate.
        fractional.loc[3, [23, 24, 25]] = np.nan  # One is below the declared minimum.
        fractional.loc[[4, 5], :] = np.nan  # A two-epoch gap.
        return generate_aoa_candidates(fractional, alpha_r_deg=66.6)

    def test_partial_epochs_keep_only_measured_prns_and_restart_blocks(self):
        aoa, branch, ambiguity = select_joint_source_aoa_segments(
            self.candidates(), (10, 23, 24, 25), min_valid_spoof_prns=2, fixed_m=0,
        )
        aoa = aoa.reindex(range(8))
        self.assertEqual(aoa.loc[1].notna().sum(), 2)
        self.assertTrue(aoa.loc[[3, 4, 5]].isna().all().all())
        self.assertEqual(int(aoa.notna().sum().sum()), 18)
        np.testing.assert_allclose(aoa.to_numpy()[aoa.notna()], 38.3)
        self.assertTrue((branch.eq(0) | branch.isna()).all().all())
        self.assertEqual(ambiguity.groupby("segment_id")["epoch_index"].agg(list).tolist(),
                         [[0], [1], [2], [6, 7]])
        self.assertEqual(ambiguity.loc[ambiguity.epoch_index.eq(1), "active_svids"].iloc[0], "10,23")

    def test_no_branch_constraint_crosses_a_gap(self):
        candidates = self.candidates()
        # Force only the mirror branch in the first block; later blocks still have both.
        candidates = candidates.loc[~(candidates.epoch_index.eq(0) & candidates.branch.ne(-1))]
        _, branches, _ = select_joint_source_aoa_segments(
            candidates, (10, 23, 24, 25), min_valid_spoof_prns=2,
        )
        self.assertTrue(branches.loc[0].eq(-1).all())
        self.assertTrue(branches.loc[6].eq(0).all())
        later = candidates.loc[candidates.epoch_index.isin([6, 7])]
        _, expected = select_joint_source_aoa(later, (10, 23, 24, 25), return_branches=True)
        pd.testing.assert_frame_equal(branches.loc[[6, 7]], expected)

    def test_default_requires_all_prns(self):
        selected, _, _ = select_joint_source_aoa_segments(self.candidates(), (10, 23, 24, 25))
        self.assertTrue(selected.loc[1].isna().all())
        self.assertEqual(selected.dropna(how="all").index.tolist(), [0, 2, 6, 7])

    def test_minimum_prn_policy_rejects_invalid_values(self):
        for minimum in (0, 5, True, 2.5):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                select_joint_source_aoa_segments(
                    self.candidates(), (10, 23, 24, 25), min_valid_spoof_prns=minimum,
                )


if __name__ == "__main__":
    unittest.main()
