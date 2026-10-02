"""Authentic dataset: DD-method AoA agrees with the Ephemeris-based AoA for every PRN."""
import unittest
from pathlib import Path

import numpy as np

from authentic_figures import AOA_PRNS, DATA_DIR, dd_method_aoa, ephemeris_aoa, fractional_dd, read_ubx

ROOT = Path(__file__).resolve().parents[1]


class AuthenticDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rx1 = read_ubx(ROOT / DATA_DIR / "raw_data_1.ubx")
        cls.rx2 = read_ubx(ROOT / DATA_DIR / "raw_data_2.ubx")
        cls.psi, cls.reference = fractional_dd(cls.rx1, cls.rx2)

    def test_window_and_reference(self):
        self.assertEqual(self.reference, 10)
        self.assertEqual(len(self.psi), 2000)
        self.assertEqual(self.psi.index[0], 201251)
        self.assertTrue(((self.psi.abs() <= 0.5) | self.psi.isna()).all().all())

    def test_dd_method_aoa_tracks_ephemeris(self):
        seconds = self.psi.index[:1000]
        error = (dd_method_aoa(self.psi.iloc[:1000]) - ephemeris_aoa(self.rx1["NAV-SAT"], seconds, AOA_PRNS)).abs()
        for prn in AOA_PRNS:
            self.assertLess(np.nanmean(error[prn]), 7.0, f"PRN {prn}")


if __name__ == "__main__":
    unittest.main()
