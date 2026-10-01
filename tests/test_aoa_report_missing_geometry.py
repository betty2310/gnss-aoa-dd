"""A missing ephemeris angle must not discard a valid spoof DD measurement."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from aoa_figure_report import FIGURE_STEMS, export_paper_figures, load_figure_design


class MissingGeometryReportTests(unittest.TestCase):
    def test_export_keeps_unpaired_spoof_dd_and_reports_zero_comparison_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("rx1.ubx", "rx2.ubx"):
                (root / name).write_bytes(b"synthetic provenance fixture")
            # Two UBX paths plus an explicit spoof group do not require note.txt.
            epochs = pd.DataFrame({
                "epoch_index": [0, 1], "source_epoch_index": [0, 1],
                "rcvTow_rx1": [100., 101.],
            })
            joint = pd.DataFrame({15: [40., 41.], 23: [40.5, 41.5]})
            branches = pd.DataFrame({15: [0, 0], 23: [0, 0]})
            proposed = joint.assign(**{"26": [100., 101.]})
            proposed.columns = proposed.columns.astype(int)
            # PRN 23 is absent, not merely a column of NaNs.
            ephemeris = pd.DataFrame({15: [60., 60.], 26: [101., 102.]})
            run = SimpleNamespace(
                epochs=epochs, reference_svid=10,
                fractional_dd_cycles=proposed / 360.,
            )
            quality = epochs.assign(meets_spoof_prn_requirement=True)
            scenario = SimpleNamespace(
                all_satellite_result=run, probe_result=run,
                spoof_svids=(15, 23), best_m_all_satellite_aoa_deg=proposed,
                ephemeris_aoa_rx1_deg=ephemeris,
                joint_source_aoa_deg=joint, joint_source_branch=branches,
                joint_source_ambiguity=pd.DataFrame({
                    "chosen_by": ["unresolved", "unresolved"],
                    "ambiguous": [True, True],
                }),
                window_quality=quality,
                run_summary=pd.DataFrame({
                    "min_valid_spoof_prns": [2], "max_gap_epochs": [0],
                    "reference_alpha_deg": [90.], "spoof_branch_m": [None],
                }),
                spoof_branch_candidates=pd.DataFrame({"svid": [15, 23]}),
                spoof_branch_detection=pd.DataFrame({"svid": [15, 23]}),
                ephemeris_aligned_branch=pd.DataFrame({"svid": [26], "branch": [0]}),
                ephemeris_aligned_branch_by_epoch=pd.DataFrame(),
                config=SimpleNamespace(
                    name="synthetic", receiver1_path=root / "rx1.ubx",
                    receiver2_path=root / "rx2.ubx", single_difference_order="rx2-rx1",
                    navsat_azimuth_bias_deg=0., baseline_over_wavelength=1.,
                ),
            )
            design = load_figure_design()
            design.update(formats=["png"], dpi=72)
            output = root / "report"
            manifest = export_paper_figures(
                scenario, output, declared_spoof_svids=(15, 23),
                dataset_dir=root, design=design, show=False,
            )
            self.assertEqual(manifest["unpaired_spoof_svids"], [23])
            self.assertEqual(manifest["omitted_spoof_svids"], [])
            self.assertEqual(manifest["paired_aoa_epoch_count_by_prn"][23], 0)
            self.assertNotIn(23, manifest["comparable_prns"])
            all_phase = pd.read_csv(output / "data/spoof_joint_source_aoa_all_phase_epochs_deg.csv")
            np.testing.assert_allclose(all_phase["23"], joint[23])
            paired = pd.read_csv(output / "data/alpha_i_proposed_best_m_paired_deg.csv")
            geometry = pd.read_csv(output / "data/alpha_i_ephemeris_deg.csv")
            self.assertTrue(paired["23"].isna().all())
            self.assertTrue(geometry["23"].isna().all())
            self.assertEqual(len(manifest["figure_files"]), 4)
            self.assertEqual({Path(path).stem for path in manifest["figure_files"]}, set(FIGURE_STEMS))
            self.assertTrue(all(Path(path).is_file() for path in manifest["figure_files"]))
            standalone = pd.read_csv(output / "data/alpha_i_proposed_best_m_deg.csv")
            np.testing.assert_allclose(standalone["23"], joint[23])


if __name__ == "__main__":
    unittest.main()
