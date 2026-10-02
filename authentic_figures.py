"""Authentic-dataset figures: fractional DD and Ephemeris-based vs. DD-method AoA.

Reads dataset/authentic/ (2000 s of a static, authentic-sky capture; see its note.txt) and
writes two side-by-side figures without captions, plus the plotted values as CSV:

* ``dd_authentic_side_by_side``  - fractional DD, full 2000 s | first 200 s.
* ``aoa_authentic_side_by_side`` - Ephemeris-based AoA | DD-method AoA, first 1000 s.

Reference satellite: PRN 10 (the lowest PRN tracked by both receivers in the first epoch).
Epochs are paired by GPS time; a PRN missing in either receiver, or a missing NAV-SAT
epoch, is a gap, never interpolated.

    uv run python authentic_figures.py                  # paper style (aoa_figure_style.json)
    uv run python authentic_figures.py --style archive  # style of the original manuscript plots
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyubx2 import UBXReader

from aoa_figure_report import draw_prn_traces, load_figure_design, prn_styles, save_figure

DATA_DIR = Path("dataset/authentic")
OUT_DIR = Path("outputs/figures/authentic")

DD_EXCLUDED_PRNS = {12}  # tracked only intermittently in this window
ALPHA_R_DEG = 53.89629  # reference-angle anchor: baseline-relative AoA of PRN 10
BASELINE_AZIMUTH_DEG = 16.0  # baseline direction, clockwise from north
AOA_PRNS = [15, 18, 23, 24, 25, 28, 32]
# Integer branch m per PRN in cos(alpha_i) = psi_i + cos(alpha_r) + m, psi_i = DD(rx1 - rx2).
BRANCH_M = {15: 0, 23: 0, 24: 0, 32: 0, 18: -1, 25: -1, 28: -1}


def read_ubx(path: Path) -> dict[str, dict[int, object]]:
    """RXM-RAWX, NAV-PVT and NAV-SAT keyed by whole GPS second."""
    out = {"RXM-RAWX": {}, "NAV-PVT": {}, "NAV-SAT": {}}
    with path.open("rb") as stream:
        for _, msg in UBXReader(stream, protfilter=2):
            if msg is None or msg.identity not in out:
                continue
            tow = msg.rcvTow if msg.identity == "RXM-RAWX" else msg.iTOW / 1000.0
            out[msg.identity][round(tow)] = msg
    return out


def l1_carrier_cycles(rawx, pvt) -> dict[int, float]:
    """GPS L1 C/A carrier phase per PRN, shifted to the PVT epoch with Doppler (cycles)."""
    phase = {}
    for i in range(1, rawx.numMeas + 1):
        if getattr(rawx, f"gnssId_{i:02}") == 0 and getattr(rawx, f"sigId_{i:02}") == 0:
            phase[getattr(rawx, f"svId_{i:02}")] = (
                getattr(rawx, f"cpMes_{i:02}") + pvt.nano * 1e-9 * getattr(rawx, f"doMes_{i:02}")
            )
    return phase


def fractional_dd(rx1, rx2) -> tuple[pd.DataFrame, int]:
    """psi_i = DD(rx1 - rx2) minus its nearest integer, per GPS second x PRN."""
    seconds = sorted(set(rx1["RXM-RAWX"]) & set(rx2["RXM-RAWX"]) & set(rx1["NAV-PVT"]) & set(rx2["NAV-PVT"]))
    sd = {}
    for s in seconds:
        p1 = l1_carrier_cycles(rx1["RXM-RAWX"][s], rx1["NAV-PVT"][s])
        p2 = l1_carrier_cycles(rx2["RXM-RAWX"][s], rx2["NAV-PVT"][s])
        sd[s] = {prn: p1[prn] - p2[prn] for prn in p1.keys() & p2.keys() if p1[prn] and p2[prn]}
    sd = pd.DataFrame.from_dict(sd, orient="index").sort_index().sort_index(axis=1)
    reference = int(sd.iloc[0].dropna().index.min())
    dd = sd.drop(columns=reference).sub(sd[reference], axis=0)
    return dd - np.round(dd), reference


def ephemeris_aoa(navsat: dict[int, object], seconds, prns) -> pd.DataFrame:
    """alpha_i = acos(cos(elev) cos(azim - baseline azimuth)) from NAV-SAT, per GPS second."""
    rows = {}
    for s in seconds:
        msg, row = navsat.get(s), {}
        if msg is not None:
            for i in range(1, msg.numSvs + 1):
                prn = getattr(msg, f"svId_{i:02d}")
                if getattr(msg, f"gnssId_{i:02d}") == 0 and prn in prns:
                    el = math.radians(getattr(msg, f"elev_{i:02d}"))
                    az = math.radians(getattr(msg, f"azim_{i:02d}") - BASELINE_AZIMUTH_DEG)
                    row[prn] = math.degrees(math.acos(math.cos(el) * math.cos(az)))
        rows[s] = row
    return pd.DataFrame.from_dict(rows, orient="index").reindex(index=list(seconds), columns=prns)


def dd_method_aoa(psi: pd.DataFrame) -> pd.DataFrame:
    cos_r = math.cos(math.radians(ALPHA_R_DEG))
    return pd.DataFrame(
        {prn: np.degrees(np.arccos(np.clip(psi[prn] + cos_r + BRANCH_M[prn], -1, 1))) for prn in AOA_PRNS}
    )


def plot_paper(t, dd_plot, t_aoa, eph, aoa, zoom: int, xlabel_out: Path, formats) -> None:
    design = load_figure_design()
    design["formats"] = formats
    xlabel = design["elapsed_axis_label"]
    with plt.rc_context(design["rc_params"]):
        styles = prn_styles(list(dd_plot), (), design)
        fig, axes = plt.subplots(1, 2, figsize=design["comparison_figsize"], sharey=True)
        for ax, n in zip(axes, [len(t), zoom]):
            draw_prn_traces(ax, t[:n], dd_plot.iloc[:n], declared=(), styles=styles, design=design,
                            xlabel=xlabel, break_wraps=True)
            ax.set_ylim(-0.55, 0.55)
            ax.set_yticks(np.arange(-0.5, 0.51, 0.25))
        axes[0].set_ylabel("Fractional double difference (cycles)")
        save_figure(fig, axes, xlabel_out, "dd_authentic_side_by_side", design, show=False)

        styles = prn_styles(AOA_PRNS, (), design)
        fig, axes = plt.subplots(1, 2, figsize=design["comparison_figsize"], sharex=True, sharey=True)
        for ax, frame in zip(axes, [eph, aoa]):
            draw_prn_traces(ax, t_aoa, frame, declared=(), styles=styles, design=design, xlabel=xlabel)
            ax.set_ylim(0, 180)
            ax.set_yticks(np.arange(0, 181, 30))
        axes[0].set_ylabel(r"$\alpha_i$ (degree)")
        save_figure(fig, axes, xlabel_out, "aoa_authentic_side_by_side", design, show=False)


def plot_archive(t, dd_plot, t_aoa, eph, aoa, zoom: int, out: Path, formats) -> None:
    """Look of the original manuscript plots (seaborn grids, default colour cycle)."""
    def save(fig, stem):
        fig.tight_layout()
        for ext in formats:
            fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, n in zip(axes, [len(t), zoom]):
            for prn in dd_plot:
                ax.plot(t[:n], dd_plot[prn].iloc[:n], label=f"SV {prn}", alpha=0.9)
            ax.set(xlabel="Time (s)", ylabel="Cycle", ylim=(-0.5, 0.5))
            ax.legend(ncol=4, fontsize=8, loc="upper right")
        save(fig, "dd_authentic_side_by_side")
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        for ax, frame, loc in [(axes[0], eph, "upper left"), (axes[1], aoa, "lower left")]:
            for prn in frame:
                ax.plot(t_aoa, frame[prn], label=f"SV {prn}", alpha=0.9)
            ax.set(xlabel="Time (s)", ylabel=r"$\alpha_i$ (in degree)")
            ax.legend(ncol=4, fontsize=8, loc=loc)
        axes[1].tick_params(labelleft=True)
        save(fig, "aoa_authentic_side_by_side")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--style", choices=["paper", "archive"], default="paper")
    parser.add_argument("--zoom-seconds", type=int, default=200)
    parser.add_argument("--aoa-seconds", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "eps"])
    args = parser.parse_args()

    rx1, rx2 = read_ubx(DATA_DIR / "raw_data_1.ubx"), read_ubx(DATA_DIR / "raw_data_2.ubx")
    psi, reference = fractional_dd(rx1, rx2)
    seconds = psi.index.to_numpy()
    t = (seconds - seconds[0]).astype(float)
    # The manuscript's DD plot is the rx2 - rx1 orientation; re-wrap to [-0.5, 0.5].
    dd_plot = psi.drop(columns=[c for c in DD_EXCLUDED_PRNS if c in psi]).mul(-1)
    dd_plot -= np.round(dd_plot)
    dd_plot = dd_plot.dropna(axis=1, how="all")

    n = args.aoa_seconds
    eph = ephemeris_aoa(rx1["NAV-SAT"], seconds[:n], AOA_PRNS)
    aoa = dd_method_aoa(psi.iloc[:n])

    out = args.out / args.style
    out.mkdir(parents=True, exist_ok=True)
    plot = plot_paper if args.style == "paper" else plot_archive
    plot(t, dd_plot, t[:n], eph, aoa, args.zoom_seconds, out, args.formats)

    dd_plot.set_axis(t, axis=0).rename_axis("time_s").to_csv(out / "fractional_dd_rx2_minus_rx1.csv")
    pd.concat({"ephemeris": eph.set_axis(t[:n]), "dd_method": aoa.set_axis(t[:n])}, axis=1).rename_axis(
        "time_s").to_csv(out / "aoa_deg.csv")
    print(f"reference PRN {reference}; {len(seconds)} epochs (TOW {seconds[0]}-{seconds[-1]}); "
          f"PRNs {list(dd_plot)}; NAV-SAT gaps in AoA window {int(eph.isna().all(axis=1).sum())}")
    err = (aoa - eph).abs()
    print("mean |DD-method - ephemeris| AoA (deg): " + ", ".join(f"PRN {p} {err[p].mean():.1f}" for p in AOA_PRNS))
    print(f"saved -> {out}/")


if __name__ == "__main__":
    main()
