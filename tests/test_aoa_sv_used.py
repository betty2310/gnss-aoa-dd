"""NAV-SAT svUsed status per GPS second and its per-window summary."""
import unittest

import numpy as np
import pandas as pd

from aoa_pipeline import navsat_sv_status_by_second, summarize_sv_used_by_window


def navsat_rows(tow, svid, cno, used):
    return {"rcvTow": tow, "svid": svid, "cno": cno, "svUsed": used}


class SvUsedTests(unittest.TestCase):
    def navsat(self):
        rows = []
        # 1002 is a NAV-SAT dropout; TOW 1.0 is a cold-start row logged before GPS time was known.
        for tow in (1000.0, 1001.0, 1.0, 1003.0):
            rows += [navsat_rows(tow, 10, 45, int(tow >= 1001)), navsat_rows(tow, 9, 40, 1),
                     navsat_rows(tow, 5, 0, 0)]
        return pd.DataFrame(rows)

    def test_status_keeps_dropouts_and_drops_pre_time_rows(self):
        status = navsat_sv_status_by_second(self.navsat())
        self.assertEqual(list(status.index), [1000, 1001, 1002, 1003])
        self.assertEqual(list(status.columns), [9, 10])  # PRN 5 is never tracked
        np.testing.assert_array_equal(status[10].to_numpy(), [0.0, 1.0, np.nan, 1.0])

    def test_leading_rows_before_gps_time_are_dropped(self):
        rows = [navsat_rows(tow, 9, 40, 1) for tow in (1.0, 2.0, 124000.0, 124001.0)]
        status = navsat_sv_status_by_second(pd.DataFrame(rows))
        self.assertEqual(list(status.index), [124000, 124001])

    def test_window_summary_counts_dropouts_as_not_used(self):
        status = {"RX1": navsat_sv_status_by_second(self.navsat())}
        windows = pd.DataFrame({"window": [1], "start_tow": [999.996], "end_tow": [1003.004]})
        summary = summarize_sv_used_by_window(status, windows, spoof_svids=(10, 28)).set_index("svid")
        self.assertEqual(list(summary.index), [10, 28, 9])
        self.assertEqual(summary.loc[10, "role"], "spoof")
        self.assertEqual(summary.loc[10, "used_seconds"], 2)
        self.assertEqual(summary.loc[10, "navsat_seconds"], 3)
        self.assertAlmostEqual(summary.loc[10, "pct_used"], 50.0)
        self.assertEqual(summary.loc[28, "tracked_seconds"], 0)
        self.assertEqual(summary.loc[9, "role"], "authentic")


if __name__ == "__main__":
    unittest.main()
