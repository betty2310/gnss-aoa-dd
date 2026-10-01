"""Workflow boundaries and provenance checks for saved AoA experiments."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from aoa_experiment import (
    ExperimentConfig, _solve, _use_opposite_reference, angle_reference_summary,
    opposite_reference, replay_experiment, run_experiment, window_configs,
)
from aoa_pipeline import generate_aoa_candidates


class ExperimentWorkflowTests(unittest.TestCase):
    def config(self, **overrides):
        values = dict(receiver1_path="rx1.ubx", receiver2_path="rx2.ubx", spoof_svids=(15, 18, 20, 23))
        values.update(overrides)
        return ExperimentConfig(**values)

    def test_explicit_inputs_round_trip_without_dataset_notes(self):
        config = self.config(name="capture", baseline_azimuth_deg=180, single_difference_order="rx1-rx2")
        restored = ExperimentConfig(**json.loads(json.dumps(config.to_dict())))
        self.assertEqual(config, restored)

    def test_rejects_invalid_geometry_and_conflicting_branches(self):
        for overrides in [
            {"receiver2_path": "rx1.ubx"}, {"spoof_svids": (15, 15)},
            {"spoof_svids": (0, 33)}, {"baseline_over_wavelength": 0},
            {"baseline_azimuth_deg": float("nan")}, {"reference_svid": 15},
            {"spoof_branch_m": 0, "prior_aoa_deg": 40}, {"name": "../outside"},
            {"start_epoch": -1}, {"min_window_epochs": 0}, {"start_epoch": 1.5},
        ]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.config(**overrides)

    def test_each_solvable_window_becomes_its_own_experiment(self):
        windows = pd.DataFrame({
            "window": [1, 2, 3], "start_epoch": [0, 157, 300], "end_epoch": [95, 290, 301],
            "solvable": [True, True, False],
        })
        checks = SimpleNamespace(probe=SimpleNamespace(spoof_windows=windows))
        configs = window_configs(self.config(name="capture"), checks)
        self.assertEqual(sorted(configs), [1, 2])
        self.assertEqual((configs[2].name, configs[2].start_epoch, configs[2].max_epochs),
                         ("capture/window_02", 157, 134))
        clipped = window_configs(self.config(start_epoch=50, max_epochs=150, min_window_epochs=10), checks)
        self.assertEqual({k: (c.start_epoch, c.max_epochs) for k, c in clipped.items()},
                         {1: (50, 46), 2: (157, 43)})

    def test_input_edits_require_a_new_capture_check(self):
        checks = SimpleNamespace(config=self.config())
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            run_experiment(self.config(single_difference_order="rx1-rx2"), checks)

    def test_changed_raw_file_blocks_export_replay_before_solving(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "receiver.ubx"
            raw.write_bytes(b"original")
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            manifest = Path(directory) / "experiment.json"
            manifest.write_text(json.dumps({
                "schema_version": 1, "status": "completed",
                "sources_sha256": {str(raw): digest},
            }))
            raw.write_bytes(b"different capture")
            with patch("aoa_experiment._solve") as solve:
                with self.assertRaisesRegex(ValueError, "input or solver changed"):
                    replay_experiment(manifest)
                solve.assert_not_called()

    def test_changed_saved_results_block_export_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "aoa.csv"
            result.write_text("PRN,angle\n15,40\n")
            manifest = Path(directory) / "experiment.json"
            manifest.write_text(json.dumps({
                "schema_version": 1, "status": "completed", "sources_sha256": {},
                "result_files": {"joint_aoa": str(result)}, "result_sha256": {"joint_aoa": "wrong"},
            }))
            with self.assertRaisesRegex(ValueError, "Saved experiment result changed"):
                replay_experiment(manifest)

    def test_expected_reference_requires_a_valid_angle(self):
        for overrides in [
            {"angle_reference": "automatic"}, {"angle_reference": "closest_to_expected"},
            {"expected_aoa_deg": -1}, {"expected_aoa_deg": 181},
            {"expected_aoa_deg": float("nan")},
        ]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.config(**overrides)

    def test_reference_preference_does_not_force_the_angle(self):
        config = self.config(angle_reference="closest_to_expected", expected_aoa_deg=40)
        self.assertTrue(_use_opposite_reference(config, 144.31))
        self.assertFalse(_use_opposite_reference(config, 35.69))
        self.assertFalse(_use_opposite_reference(config, 90))

    def test_complete_reversal_changes_sign_geometry_and_declared_branch(self):
        config = self.config(baseline_azimuth_deg=350, spoof_branch_m=-1)
        flipped = opposite_reference(config)
        self.assertEqual(flipped.single_difference_order, "rx1-rx2")
        self.assertEqual(flipped.baseline_azimuth_deg, 170)
        self.assertEqual(flipped.spoof_branch_m, 1)
        self.assertEqual(opposite_reference(self.config(prior_aoa_deg=140)).prior_aoa_deg, 40)
        self.assertEqual(opposite_reference(flipped), config)

    def test_dd_equation_is_supplementary_only_after_complete_reversal(self):
        fractional = pd.DataFrame({15: [0.10, 0.12], 18: [0.11, 0.09]})
        before = generate_aoa_candidates(fractional, 153, branch_values=(0,))
        after = generate_aoa_candidates(-fractional, 27, branch_values=(0,))
        np.testing.assert_allclose(after.candidate_aoa_deg, 180-before.candidate_aoa_deg)
        order_only = generate_aoa_candidates(-fractional, 153, branch_values=(0,))
        self.assertFalse(np.allclose(order_only.candidate_aoa_deg, 180-before.candidate_aoa_deg))

    def test_expected_reference_preserves_samples_and_records_unconfirmed_orientation(self):
        config = self.config(angle_reference="closest_to_expected", expected_aoa_deg=40)
        before = pd.DataFrame({15: [143., 144.], 18: [144., 145.]})
        calls = []

        def solve(config, usable):
            calls.append(config)
            values = before if config.single_difference_order == "rx2-rx1" else 180-before
            return SimpleNamespace(config=config.dataset_config(usable), joint_source_aoa_deg=values)

        with patch("aoa_experiment._solve_configured", side_effect=solve):
            result = _solve(config, config.spoof_svids, save_outputs=False)
        pd.testing.assert_frame_equal(result.joint_source_aoa_deg, 180-before)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call.prior_aoa_deg is None for call in calls))
        summary = angle_reference_summary(config, result)
        self.assertEqual(summary["reported_aoa_deg"], 36)
        self.assertEqual(summary["selected_by"], "expected_angle_preference")
        self.assertEqual(summary["reference_direction_under_configured_hypothesis"], "south")
        self.assertFalse(summary["physical_orientation_confirmed"])

    def test_mismatching_reversed_branch_is_rejected(self):
        config = self.config(angle_reference="opposite")
        result = SimpleNamespace(joint_source_aoa_deg=pd.DataFrame({15: [144.]}))
        with patch("aoa_experiment._solve_configured", return_value=result):
            with self.assertRaises(AssertionError):
                _solve(config, config.spoof_svids, save_outputs=False)


if __name__ == "__main__":
    unittest.main()
