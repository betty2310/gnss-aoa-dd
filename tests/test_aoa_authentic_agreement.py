"""Agreement denominators and whole-PRN filtering must match reported claims."""
import unittest

import numpy as np
import pandas as pd

from aoa_figure_report import authentic_agreement_tables


class AuthenticAgreementTests(unittest.TestCase):
    def test_inclusive_thresholds_and_maximum_error_filter(self):
        proposed = pd.DataFrame({1: [5., 7., 10.], 2: [0., 0., 10.01], 16: [30., 30., 30.], 23: [90., 90., 90.]})
        ephemeris = proposed * 0
        before = proposed.copy(deep=True)
        errors, unfiltered, audit, reported = authentic_agreement_tables(
            proposed, ephemeris, spoof_svids=(23,), reference_svid=16, max_error_deg=10,
        )
        self.assertEqual(errors.columns.tolist(), [1, 2])
        self.assertEqual(reported.target_prn.tolist(), ["1", "Overall"])
        row = reported.iloc[0]
        self.assertEqual(row.paired_epochs, 3)
        self.assertAlmostEqual(row.p_le_5deg_pct, 100 / 3)
        self.assertAlmostEqual(row.p_le_7deg_pct, 200 / 3)
        self.assertEqual(row.p_le_10deg_pct, 100)
        rejected = audit.loc[audit.target_prn.eq("2")].iloc[0]
        self.assertEqual(rejected.median_abs_error_deg, 0)
        self.assertFalse(rejected.retained)
        self.assertEqual(rejected.exclusion_reason, "maximum_paired_error_exceeds_cutoff")
        self.assertAlmostEqual(unfiltered.iloc[-1].p_le_10deg_pct, 500 / 6)
        pd.testing.assert_frame_equal(before, proposed)

    def test_missing_pairs_do_not_pass_or_enter_pooled_denominator(self):
        proposed = pd.DataFrame({1: [5., 7., 10.], 2: [5., np.nan, 7.], 3: [np.nan] * 3})
        ephemeris = pd.DataFrame(0., index=proposed.index, columns=proposed.columns)
        _, _, audit, reported = authentic_agreement_tables(
            proposed, ephemeris, spoof_svids=(), reference_svid=16, max_error_deg=10,
        )
        self.assertEqual(reported.target_prn.tolist(), ["1", "2", "Overall"])
        overall = reported.iloc[-1]
        self.assertEqual(overall.paired_epochs, 5)
        self.assertEqual(overall.p_le_5deg_pct, 40.)
        self.assertEqual(overall.p_le_7deg_pct, 80.)
        self.assertEqual(overall.p_le_10deg_pct, 100.)
        no_pairs = audit.loc[audit.target_prn.eq("3")].iloc[0]
        self.assertFalse(no_pairs.retained)
        self.assertEqual(no_pairs.exclusion_reason, "no_paired_measurements")
        self.assertTrue(np.isnan(no_pairs.p_le_10deg_pct))

    def test_unfiltered_and_empty_cohorts_are_not_mislabeled_as_100_percent(self):
        proposed = pd.DataFrame({1: [11., 12.], 2: [15., 16.]})
        ephemeris = proposed * 0
        _, unfiltered, audit, reported = authentic_agreement_tables(
            proposed, ephemeris, spoof_svids=(), reference_svid=16,
        )
        pd.testing.assert_frame_equal(unfiltered, reported)
        self.assertTrue(audit.retained.all())
        self.assertEqual(reported.iloc[-1].p_le_10deg_pct, 0.)
        _, _, _, empty = authentic_agreement_tables(
            proposed, ephemeris, spoof_svids=(), reference_svid=16, max_error_deg=10,
        )
        self.assertEqual(empty.target_prn.tolist(), ["Overall"])
        self.assertEqual(empty.iloc[-1].paired_epochs, 0)
        self.assertTrue(np.isnan(empty.iloc[-1].p_le_10deg_pct))
        for cutoff in (-1, np.nan, np.inf):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                authentic_agreement_tables(proposed, ephemeris, spoof_svids=(), reference_svid=16, max_error_deg=cutoff)


if __name__ == "__main__":
    unittest.main()
