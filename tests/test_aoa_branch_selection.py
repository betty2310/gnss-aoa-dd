"""Tests for the Eq. 21 branch selection, its ambiguity reporting and the rig checks.

The synthetic captures below build fractional double differences from a known source
direction ``alpha_s`` and reference anchor ``alpha_r`` exactly as Eq. 18 would produce them,
so every test knows which branch is physically right.
"""
import unittest

import numpy as np
import pandas as pd

from aoa_pipeline import (
    _describe_segments,
    _true_segments,
    build_best_m_all_satellite_aoa,
    fit_baseline_azimuth,
    generate_aoa_candidates,
    select_aoa_by_reference,
    select_aoa_by_reference_per_epoch,
    select_aoa_dp,
    select_joint_source_aoa,
    summarize_branch_candidates,
    summarize_pairwise_single_difference_collapse,
    summarize_per_epoch_branches,
    summarize_spoof_branch_detection,
)

SOURCE_AOA_DEG = 38.3
MIRROR_AOA_DEG = 102.5  # arccos(cos(38.3 deg) - 1)
REFERENCE_AOA_DEG = 66.6
SPOOF_SVIDS = (10, 23, 24)


def synthetic_candidates(
    *,
    source_aoa_deg: float = SOURCE_AOA_DEG,
    reference_aoa_deg: float = REFERENCE_AOA_DEG,
    svids=SPOOF_SVIDS,
    epochs: int = 12,
    noise_cycles: float = 0.01,
    baseline_over_wavelength: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    geometry = baseline_over_wavelength * (
        np.cos(np.deg2rad(source_aoa_deg)) - np.cos(np.deg2rad(reference_aoa_deg))
    )
    frame = {}
    for svid in svids:
        double_difference = geometry + rng.normal(0.0, noise_cycles, size=epochs)
        frame[svid] = double_difference - np.round(double_difference)
    fractional = pd.DataFrame(frame, index=pd.Index(range(epochs), name="epoch_index"))
    return generate_aoa_candidates(
        fractional,
        alpha_r_deg=reference_aoa_deg,
        baseline_over_wavelength=baseline_over_wavelength,
    )


class JointSourceBranchSelectionTests(unittest.TestCase):
    def test_explicit_m_selects_v4_and_v5_branches_without_resolving_ambiguity(self):
        for source_angle, anchor, branch in [(38.3, 66.6, 0), (143.0, 62.1, -1)]:
            with self.subTest(branch=branch):
                candidates = synthetic_candidates(
                    source_aoa_deg=source_angle, reference_aoa_deg=anchor
                )
                selected, branches, ambiguity = select_joint_source_aoa(
                    candidates, SPOOF_SVIDS, fixed_m=branch,
                    return_branches=True, return_ambiguity=True,
                )
                self.assertTrue(branches.eq(branch).all().all())
                self.assertTrue(ambiguity["chosen_by"].eq("explicit_m").all())
                self.assertTrue(ambiguity["ambiguous"].all())
                self.assertTrue(ambiguity["fixed_m"].eq(branch).all())
                self.assertAlmostEqual(float(selected.median().median()), source_angle, delta=1.5)
                expected = candidates.loc[candidates["branch"].eq(branch)].pivot(
                    index="epoch_index", columns="svid", values="candidate_aoa_deg"
                )
                np.testing.assert_allclose(selected, expected)
                self.assertNotEqual(ambiguity["alternative_branches"].iloc[0], "")

    def test_explicit_m_rejects_invalid_branches_and_conflicting_prior(self):
        candidates = synthetic_candidates()
        for kwargs in [{"fixed_m": 2}, {"fixed_m": 0.5}, {"fixed_m": True},
                       {"fixed_m": 0, "prior_aoa_deg": 30}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_joint_source_aoa(candidates, SPOOF_SVIDS, **kwargs)

    def test_best_m_export_keeps_joint_choice_and_missing_observations(self):
        dp = pd.DataFrame({10: [78.0, 79.0], 16: [110.0, 111.0]})
        eph = pd.DataFrame({10: [80.0, 81.0], 16: [60.0, 61.0]})
        joint = pd.DataFrame({10: [143.0, np.nan]})
        combined = build_best_m_all_satellite_aoa(dp, eph, (10,), joint_source_aoa=joint)
        np.testing.assert_allclose(combined[10], joint[10], equal_nan=True)
        np.testing.assert_allclose(combined[16], eph[16])
        np.testing.assert_allclose(dp[10], [78.0, 79.0])

    def test_per_epoch_reference_branch_follows_the_half_cycle_wrap(self):
        # 30-1 PRN 2: cos(alpha_i) - cos(alpha_r) = -1.5, so the fractional DD flips between
        # about +0.47 and -0.49. A constant m jumps a full cycle (~90 vs ~165 deg) at each flip.
        alpha_r = 57.94
        frac = pd.DataFrame({2: [0.47, -0.49, 0.47, -0.47, 0.30]})
        eph = pd.DataFrame({2: [165.8] * 5}).rename_axis("epoch_index")
        candidates = generate_aoa_candidates(frac, alpha_r)

        constant = select_aoa_by_reference(candidates, eph, min_valid_fraction=0.6)
        self.assertGreater(constant[2].max() - constant[2].min(), 60.0)

        aoa, branch = select_aoa_by_reference_per_epoch(candidates, eph)
        self.assertEqual(branch[2].tolist()[:4], [-2.0, -1.0, -2.0, -1.0])
        self.assertTrue((aoa[2].iloc[:4] > 150.0).all())
        # The nearest branch is cos = -1.17: outside [-1, 1], so NaN, not the valid branch at ~98 deg.
        self.assertTrue(np.isnan(aoa[2].iloc[4]))
        self.assertEqual(branch[2].iloc[4], -2.0)
        summary = summarize_per_epoch_branches(branch, aoa)
        self.assertEqual(summary.loc[0, ["branch", "branches", "branch_switches", "valid_epochs"]].tolist(),
                         [-2, "-2;-1", 3, 4])

    def test_one_wavelength_branches_tie_in_direction_cosine_and_stay_unresolved(self):
        candidates = synthetic_candidates()

        selected, branches, ambiguity = select_joint_source_aoa(
            candidates, SPOOF_SVIDS, return_branches=True, return_ambiguity=True
        )

        first = ambiguity.iloc[0]
        self.assertTrue(bool(first["ambiguous"]))
        self.assertEqual(first["chosen_by"], "unresolved")
        self.assertNotEqual(first["tied_shifts"], "")
        # Both branches are fully valid, so their spread and step costs are identical.
        np.testing.assert_allclose(
            ambiguity["dp_cost"], ambiguity["alternative_cost"], atol=1e-9
        )
        self.assertEqual(float(first["alternative_mean_excess"]), 0.0)
        # Without a prior the smallest |m| combination is reported, not the one nearer 90 deg.
        self.assertTrue(branches.eq(0).all().all())
        self.assertAlmostEqual(float(selected.median().median()), SOURCE_AOA_DEG, delta=1.5)
        self.assertEqual(first["alternative_branches"], "-1,-1,-1")

        summary = summarize_branch_candidates(candidates, SPOOF_SVIDS)
        self.assertEqual(len(summary), 2 * len(SPOOF_SVIDS))
        self.assertTrue(summary["epoch_count"].eq(12).all())
        mirror = summary.loc[summary["branch"].eq(-1), "median_aoa_deg"]
        self.assertTrue(np.allclose(mirror, MIRROR_AOA_DEG, atol=1.5))

    def test_prior_breaks_the_tie_toward_the_declared_direction(self):
        candidates = synthetic_candidates()

        near_source, _, ambiguity = select_joint_source_aoa(
            candidates,
            SPOOF_SVIDS,
            prior_aoa_deg=30.0,
            return_branches=True,
            return_ambiguity=True,
        )
        self.assertEqual(ambiguity["chosen_by"].iloc[0], "prior")
        self.assertAlmostEqual(float(near_source.median().median()), SOURCE_AOA_DEG, delta=1.5)

        near_mirror, branches, ambiguity = select_joint_source_aoa(
            candidates,
            SPOOF_SVIDS,
            prior_aoa_deg=110.0,
            return_branches=True,
            return_ambiguity=True,
        )
        self.assertEqual(ambiguity["chosen_by"].iloc[0], "prior")
        self.assertTrue(branches.eq(-1).all().all())
        self.assertAlmostEqual(float(near_mirror.median().median()), MIRROR_AOA_DEG, delta=1.5)
        self.assertEqual(ambiguity["alternative_branches"].iloc[0], "0,0,0")

    def test_half_wavelength_baseline_leaves_one_valid_branch_chosen_by_data(self):
        candidates = synthetic_candidates(baseline_over_wavelength=0.5)

        selected, _, ambiguity = select_joint_source_aoa(
            candidates, SPOOF_SVIDS, return_branches=True, return_ambiguity=True
        )

        first = ambiguity.iloc[0]
        self.assertFalse(bool(first["ambiguous"]))
        self.assertEqual(first["chosen_by"], "data")
        self.assertGreater(
            float(first["alternative_mean_excess"]), float(first["tie_tolerance"])
        )
        summary = summarize_branch_candidates(candidates, SPOOF_SVIDS)
        self.assertEqual(len(summary), len(SPOOF_SVIDS))
        self.assertAlmostEqual(float(selected.median().median()), SOURCE_AOA_DEG, delta=1.5)

    def test_degree_metric_reproduces_the_legacy_preference_for_90_deg(self):
        candidates = synthetic_candidates()

        legacy, branches, ambiguity = select_joint_source_aoa(
            candidates,
            SPOOF_SVIDS,
            metric="aoa_deg",
            return_branches=True,
            return_ambiguity=True,
        )

        # The same phase noise gives a smaller degree spread near 90 deg, so the legacy
        # metric picks the mirror branch; the tie is still reported.
        self.assertTrue(branches.eq(-1).all().all())
        self.assertAlmostEqual(float(legacy.median().median()), MIRROR_AOA_DEG, delta=1.5)
        self.assertEqual(ambiguity["chosen_by"].iloc[0], "unresolved")
        self.assertTrue(bool(ambiguity["ambiguous"].iloc[0]))

    def test_invalid_arguments_are_rejected(self):
        candidates = synthetic_candidates()
        with self.assertRaises(ValueError):
            select_joint_source_aoa(candidates, SPOOF_SVIDS, metric="radians")
        with self.assertRaises(ValueError):
            select_joint_source_aoa(candidates, SPOOF_SVIDS, tie_tolerance=-1.0)
        with self.assertRaises(ValueError):
            select_joint_source_aoa(candidates, ())


class PerSatelliteDpMetricTests(unittest.TestCase):
    def test_direction_cosine_metric_breaks_exact_ties_toward_smallest_abs_m(self):
        candidates = synthetic_candidates(svids=(10,))

        selected, branches = select_aoa_dp(candidates, return_branches=True)

        self.assertTrue(branches[10].eq(0).all())
        self.assertAlmostEqual(float(selected[10].median()), SOURCE_AOA_DEG, delta=1.5)

    def test_degree_metric_keeps_the_branch_nearer_90_deg(self):
        candidates = synthetic_candidates(svids=(10,))

        selected, branches = select_aoa_dp(
            candidates, return_branches=True, metric="aoa_deg"
        )

        self.assertTrue(branches[10].eq(-1).all())
        self.assertAlmostEqual(float(selected[10].median()), MIRROR_AOA_DEG, delta=1.5)


class SpoofBranchDetectionTests(unittest.TestCase):
    def test_flags_are_reported_per_branch(self):
        candidates = synthetic_candidates()
        branch_candidates = summarize_branch_candidates(candidates, SPOOF_SVIDS)
        ephemeris = pd.DataFrame(
            {10: 124.4, 23: 90.4, 24: 100.0},
            index=pd.Index(range(12), name="epoch_index"),
        )

        detection = summarize_spoof_branch_detection(
            branch_candidates,
            ephemeris,
            selected_branch_by_svid={10: 0, 23: 0, 24: 0},
            alternative_branch_by_svid={10: -1, 23: -1, 24: -1},
            chosen_by="unresolved",
            threshold_deg=10.0,
        )

        detection = detection.set_index("svid")
        self.assertTrue(detection["spoof_flag_selected"].all())
        self.assertTrue(bool(detection.loc[10, "spoof_flag_all_branches"]))
        self.assertTrue(bool(detection.loc[23, "spoof_flag_all_branches"]))
        # PRN 24's mirror branch (102.5 deg) agrees with a 100 deg ephemeris.
        self.assertFalse(bool(detection.loc[24, "spoof_flag_all_branches"]))
        self.assertAlmostEqual(
            float(detection.loc[24, "alternative_aoa_deg"]), MIRROR_AOA_DEG, delta=1.5
        )
        self.assertEqual(detection["chosen_by"].iloc[0], "unresolved")


class BaselineAzimuthFitTests(unittest.TestCase):
    @staticmethod
    def synthetic_rig(true_azimuth_deg: float, order_sign: float = 1.0, seed: int = 1):
        rng = np.random.default_rng(seed)
        epochs = 240
        index = pd.Index(range(epochs), name="epoch_index")
        geometry = {
            3: (40.0, 120.0),
            8: (65.0, 200.0),
            16: (25.0, 300.0),
            27: (55.0, 20.0),
            31: (15.0, 250.0),
        }
        elevation = pd.DataFrame(
            {svid: np.full(epochs, elev) + np.linspace(0, 2, epochs) for svid, (elev, _) in geometry.items()},
            index=index,
        )
        azimuth = pd.DataFrame(
            {svid: np.full(epochs, azim) + np.linspace(0, 3, epochs) for svid, (_, azim) in geometry.items()},
            index=index,
        )
        receiver_clock = rng.uniform(-0.5, 0.5, size=epochs)
        single_difference = {}
        for svid in geometry:
            cos_alpha = np.cos(np.deg2rad(elevation[svid])) * np.cos(
                np.deg2rad(azimuth[svid] - true_azimuth_deg)
            )
            single_difference[svid] = order_sign * (
                cos_alpha + receiver_clock + rng.integers(-3, 4)
            ) + rng.normal(0.0, 0.02, size=epochs)
        # A spoofed PRN whose single difference does not follow its ephemeris geometry.
        single_difference[24] = rng.uniform(-0.5, 0.5, size=epochs)
        elevation[24] = 30.0
        azimuth[24] = 90.0
        return pd.DataFrame(single_difference, index=index), elevation, azimuth

    def test_recovers_the_baseline_azimuth_and_confirms_a_consistent_configuration(self):
        single_difference, elevation, azimuth = self.synthetic_rig(352.0)

        fit = fit_baseline_azimuth(
            single_difference,
            elevation,
            azimuth,
            exclude_svids=(24,),
            configured_azimuth_deg=0.0,
        )

        self.assertEqual(fit.authentic_svids, (3, 8, 16, 27, 31))
        self.assertLessEqual(abs(fit.fitted_azimuth_deg - 352.0), 1.0)
        self.assertLess(fit.fitted_residual_cycles, 0.05)
        self.assertTrue(fit.consistent)
        self.assertIn("consistent", fit.recommendation)
        self.assertEqual(len(fit.to_frame()), 1)
        self.assertTrue((fit.per_svid_residual_cycles < 0.05).all())

    def test_swapped_receiver_order_is_reported_as_the_opposite_polarity(self):
        single_difference, elevation, azimuth = self.synthetic_rig(0.0, order_sign=-1.0)

        fit = fit_baseline_azimuth(
            single_difference,
            elevation,
            azimuth,
            exclude_svids=(24,),
            configured_azimuth_deg=0.0,
        )

        self.assertLessEqual(abs(((fit.fitted_azimuth_deg - 180.0 + 180.0) % 360.0) - 180.0), 1.0)
        self.assertFalse(fit.consistent)
        self.assertIn("other way round", fit.recommendation)
        self.assertLess(fit.opposite_residual_cycles, fit.configured_residual_cycles)

    def test_requires_three_authentic_satellites(self):
        single_difference, elevation, azimuth = self.synthetic_rig(10.0)
        with self.assertRaises(ValueError):
            fit_baseline_azimuth(
                single_difference,
                elevation,
                azimuth,
                exclude_svids=(24, 3, 8, 16),
            )


class CaptureProbeHelperTests(unittest.TestCase):
    def test_pairwise_single_difference_collapse_is_branch_free(self):
        index = pd.Index(range(50), name="epoch_index")
        base = np.linspace(0.0, 0.4, 50)
        frame = pd.DataFrame(
            {10: base + 3.0, 23: base - 2.0 + 0.01, 24: np.linspace(0.3, -0.2, 50)},
            index=index,
        )

        collapse = summarize_pairwise_single_difference_collapse(frame, (10, 23, 24, 99))

        collapse = collapse.set_index("pair")
        self.assertEqual(len(collapse), 3)
        self.assertAlmostEqual(
            float(collapse.loc["10-23", "median_abs_wrapped_sd_diff_cycles"]), 0.01, places=6
        )
        self.assertGreater(float(collapse.loc["10-24", "median_abs_wrapped_sd_diff_cycles"]), 0.1)

    def test_segments_are_described_with_tow_ranges(self):
        mask = pd.Series([False, True, True, False, True], index=[0, 1, 2, 3, 4])
        tow = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=[0, 1, 2, 3, 4])

        segments = _true_segments(mask)

        self.assertEqual(segments, [(1, 2), (4, 4)])
        self.assertEqual(
            _describe_segments(segments, tow), "TOW 101-102 (2 ep); TOW 104-104 (1 ep)"
        )


if __name__ == "__main__":
    unittest.main()
