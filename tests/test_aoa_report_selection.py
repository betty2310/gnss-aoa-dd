"""Scientific-data invariants for the compacted publication display."""
from itertools import combinations
import unittest

import numpy as np
import pandas as pd

from aoa_figure_report import break_at_wraps, select_compacted_report_epochs, wrapping_prns


class CompactedReportSelectionTests(unittest.TestCase):
    def epochs(self, size):
        return pd.DataFrame({
            "epoch_index": np.arange(size),
            "source_epoch_index": 300 + np.arange(size),
            "rcvTow_rx1": 12345.001 + np.arange(size),
        })

    def test_smooth_subsequence_matches_exhaustive_search_and_preserves_values(self):
        fractional = pd.DataFrame({10: [0.0, 0.03, -0.3, 0.02, 0.025, 0.027, 0.5]})
        proposed = pd.DataFrame({10: [90., 91., 179., 92., 93., 92., 2.]})
        ephemeris = pd.DataFrame({10: [90.] * 7})
        originals = [frame.copy(deep=True) for frame in (fractional, proposed, ephemeris)]
        index, selection, audit = select_compacted_report_epochs(
            self.epochs(7), fractional, proposed, ephemeris, spoof_svids=(10,), count=4,
        )

        def rank(path):
            jumps = sum(
                (fractional.loc[b, 10] - fractional.loc[a, 10]) ** 2
                + ((proposed.loc[b, 10] - proposed.loc[a, 10]) / 180.) ** 2
                for a, b in zip(path, path[1:])
            )
            return jumps, path[-1] - path[0], path

        expected = min(combinations(range(7), 4), key=rank)
        self.assertEqual(tuple(index), expected)
        self.assertEqual(audit.loc[audit.selected_for_report, "source_epoch_index"].tolist(),
                         [300 + value for value in expected])
        self.assertAlmostEqual(selection["normalized_squared_steps"], rank(expected)[0])
        for before, after in zip(originals, (fractional, proposed, ephemeris)):
            pd.testing.assert_frame_equal(before, after)

    def test_coverage_has_priority_over_artificially_quiet_sparse_epochs(self):
        proposed = pd.DataFrame({10: [0., 0., 100.], 23: [0., np.nan, 100.]})
        index, selection, _ = select_compacted_report_epochs(
            self.epochs(3), proposed / 360., proposed, proposed,
            spoof_svids=(10, 23), count=2,
        )
        self.assertEqual(index.tolist(), [0, 2])
        self.assertEqual(selection["selected_paired_spoof_measurements"], 4)

    def test_all_valid_samples_required_keeps_discrete_epochs_and_prn_nans(self):
        proposed = pd.DataFrame({10: [90., np.nan, 91., 92.], 23: [90., np.nan, np.nan, 92.]})
        index, selection, audit = select_compacted_report_epochs(
            self.epochs(4), proposed / 360., proposed, proposed,
            spoof_svids=(10, 23), count=3,
        )
        self.assertEqual(index.tolist(), [0, 2, 3])
        self.assertTrue(np.isnan(proposed.loc[index, 23].iloc[1]))
        self.assertEqual(selection["eligible_epoch_count"], 3)
        self.assertFalse(audit.loc[1, "eligible_for_selection"])
        with self.assertRaisesRegex(ValueError, "only 2 valid spoof epochs"):
            select_compacted_report_epochs(
                self.epochs(4), proposed / 360., proposed, proposed,
                spoof_svids=(10, 23), count=3, minimum_spoof_prns=2,
            )

    def test_tied_sequences_use_shortest_span_then_earliest_epochs(self):
        frame = pd.DataFrame({10: [90., 90., 90., 90., 90.]})
        index, _, _ = select_compacted_report_epochs(
            self.epochs(5), frame / 360., frame, frame, spoof_svids=(10,), count=3,
        )
        self.assertEqual(index.tolist(), [0, 1, 2])

    def test_rejects_invalid_budget_and_non_1hz_axis(self):
        frame = pd.DataFrame({10: [90., 91., 92.]})
        for count in (0, -1, 1.5, True, 4):
            with self.subTest(count=count), self.assertRaises(ValueError):
                select_compacted_report_epochs(
                    self.epochs(3), frame / 360., frame, frame, spoof_svids=(10,), count=count,
                )
        epochs = self.epochs(3)
        epochs.loc[2, "rcvTow_rx1"] += 1
        with self.assertRaisesRegex(ValueError, "regular 1 Hz"):
            select_compacted_report_epochs(
                epochs, frame / 360., frame, frame, spoof_svids=(10,), count=2,
            )


class WrapExclusionTests(unittest.TestCase):
    def test_wrapping_prns_finds_half_cycle_jumps_across_gaps(self):
        fractional = pd.DataFrame({
            2: [0.47, -0.49, 0.47, np.nan],     # flips across +-0.5
            4: [-0.49, np.nan, 0.49, -0.48],    # flips across a dropout
            7: [0.12, 0.14, np.nan, 0.13],      # stays on one side
        })
        self.assertEqual(wrapping_prns(fractional), [2, 4])

    def test_break_at_wraps_keeps_every_sample_and_drops_the_crossing_line(self):
        x = np.array([0., 1., 2., 3.])
        y = np.array([0.48, -0.49, -0.47, 0.49])
        xs, ys, positions = break_at_wraps(x, y)
        np.testing.assert_array_equal(ys[positions], y)
        np.testing.assert_array_equal(xs[positions], x)
        self.assertEqual(int(np.isnan(ys).sum()), 2)


if __name__ == "__main__":
    unittest.main()
