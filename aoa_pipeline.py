from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pyubx2 import UBXReader

try:
    from pyubx2 import ERR_IGNORE
except ImportError:  # pragma: no cover - depends on pyubx2 version
    ERR_IGNORE = None


RAWX_COLUMNS = [
    "rcvTow",
    "week",
    "measurement_index",
    "gnssId",
    "svid",
    "sigId",
    "cpMes",
    "prMes",
    "doMes",
    "cno",
    "locktime",
    "cp_valid",
    "pr_valid",
    "half_cycle",
    "cpStd",
]

NAVSAT_COLUMNS = [
    "rcvTow",
    "iTOW",
    "navsat_index",
    "gnssId",
    "svid",
    "cno",
    "elev",
    "azim",
    "qualityInd",
    "svUsed",
]

NAVPVT_COLUMNS = [
    "rcvTow",
    "iTOW",
    "fixType",
    "numSV",
    "lon",
    "lat",
    "height",
    "hMSL",
    "hAcc",
    "vAcc",
    "nano",
]


@dataclass
class DatasetConfig:
    name: str
    receiver1_path: str | Path
    receiver2_path: str | Path
    alpha_r_deg: float | None = None
    reference_svid: int | None = None
    excluded_reference_svids: tuple[int, ...] | None = None
    reference_required_svids: tuple[int, ...] | None = None
    selected_svids: tuple[int, ...] | None = None
    excluded_analysis_svids: tuple[int, ...] | None = None
    start_index: int = 0
    length: int | None = 500
    gnss_id: int = 0
    sig_id: int = 0
    sync_tow_decimals: int | None = None
    use_navpvt_time_correction: bool = True
    navpvt_tow_decimals: int = 0
    baseline_over_wavelength: float = 1.0
    single_difference_order: str = "rx1-rx2"
    branch_values: tuple[int, ...] = (-2, -1, 0, 1, 2)
    # Quantity whose epoch-to-epoch step the Algorithm 1 DP minimises: "direction_cosine"
    # (cos alpha, where the Eq. 21 branches are exact unit shifts of each other, so equally
    # smooth branches tie) or "aoa_deg" (degrees; legacy, favours the branch nearer 90 deg).
    branch_metric: str = "direction_cosine"
    navsat_azimuth_bias_deg: float | None = None
    output_root: str | Path = Path("outputs/aoa_paper")

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.name


@dataclass
class UbxObservationTables:
    rawx: pd.DataFrame
    navsat: pd.DataFrame
    navpvt: pd.DataFrame
    # UBX message identity -> count over the parsed part of the log (all identities,
    # not only the three tables above); empty for tables derived by sync/slice.
    message_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class SyncedObservations:
    epochs: pd.DataFrame
    rx1: UbxObservationTables
    rx2: UbxObservationTables


@dataclass
class CarrierPhaseTables:
    rx1_cycles: pd.DataFrame
    rx2_cycles: pd.DataFrame
    combined_cycles: pd.DataFrame
    mean_cno: pd.DataFrame


@dataclass
class AoaRunResult:
    config: DatasetConfig
    epochs: pd.DataFrame
    reference_svid: int
    reference_selection: pd.DataFrame
    carrier_phase_cycles: pd.DataFrame
    carrier_phase_rx1_cycles: pd.DataFrame
    carrier_phase_rx2_cycles: pd.DataFrame
    single_difference_cycles: pd.DataFrame
    double_difference_cycles: pd.DataFrame
    rounded_dd_cycles: pd.DataFrame
    fractional_dd_cycles: pd.DataFrame
    candidate_aoa_deg: pd.DataFrame
    selected_aoa_deg: pd.DataFrame
    selected_branch: pd.DataFrame
    collapse_metrics_by_epoch: pd.DataFrame
    collapse_metrics: pd.DataFrame
    aoa_collapse_metrics_by_epoch: pd.DataFrame
    aoa_collapse_metrics: pd.DataFrame
    navsat_aoa_diagnostic_deg: pd.DataFrame
    ephemeris_aoa_deg: pd.DataFrame
    output_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class PaperScenarioResult:
    config: DatasetConfig
    partial_spoof_enabled: bool
    spoof_svids: tuple[int, ...]
    probe_result: AoaRunResult | None
    analysis_result: AoaRunResult
    all_satellite_result: AoaRunResult
    reference_alpha_by_epoch: pd.DataFrame
    joint_source_aoa_deg: pd.DataFrame
    joint_source_branch: pd.DataFrame
    pairwise_convergence: pd.DataFrame
    cluster_summary: pd.DataFrame
    all_satellite_dd_method_aoa_deg: pd.DataFrame
    ephemeris_aoa_rx1_deg: pd.DataFrame
    ephemeris_aoa_rx2_deg: pd.DataFrame
    all_satellite_valid_m_candidates: pd.DataFrame
    all_satellite_valid_m_summary: pd.DataFrame
    ephemeris_aligned_dd_method_aoa_deg: pd.DataFrame
    ephemeris_aligned_branch: pd.DataFrame
    best_m_all_satellite_aoa_deg: pd.DataFrame
    aoa_method_equivalence_summary: pd.DataFrame
    sv_used_rx1: pd.DataFrame
    sv_used_rx2: pd.DataFrame
    sv_used_summary: pd.DataFrame
    run_summary: pd.DataFrame
    output_paths: dict[str, Path] = field(default_factory=dict)
    # Partial-spoof only: per-epoch branch ambiguity of the joint-source selection, every
    # valid Eq. 21 branch of the spoof PRNs, and the per-PRN detection table (both branches
    # against the ephemeris-based AoA).
    joint_source_ambiguity: pd.DataFrame = field(default_factory=pd.DataFrame)
    spoof_branch_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    spoof_branch_detection: pd.DataFrame = field(default_factory=pd.DataFrame)
    prior_aoa_deg: float | None = None
    window_quality: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Authentic SVs: per-epoch branch m closest to the ephemeris-based AoA (the "best m" plot);
    # ``ephemeris_aligned_branch`` summarizes it per SVID.
    ephemeris_aligned_branch_by_epoch: pd.DataFrame = field(default_factory=pd.DataFrame)


def _make_ubx_reader(stream) -> UBXReader:
    kwargs = {"protfilter": 2}
    if ERR_IGNORE is not None:
        kwargs["quitonerror"] = ERR_IGNORE
    return UBXReader(stream, **kwargs)


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise_svids(svids: Sequence[int] | None) -> tuple[int, ...] | None:
    if svids is None:
        return None
    return tuple(dict.fromkeys(int(svid) for svid in svids))


def _rawx_rows(parsed_data, gnss_id: int, sig_id: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    num_meas = _int_or_none(getattr(parsed_data, "numMeas", None))
    if num_meas is None:
        num_meas = 0
        while hasattr(parsed_data, f"gnssId_{num_meas + 1:02}"):
            num_meas += 1

    for index in range(1, num_meas + 1):
        gnss = _int_or_none(getattr(parsed_data, f"gnssId_{index:02}", None))
        sig = _int_or_none(getattr(parsed_data, f"sigId_{index:02}", None))
        if gnss != gnss_id or sig != sig_id:
            continue

        rows.append(
            {
                "rcvTow": _float_or_nan(parsed_data.rcvTow),
                "week": _int_or_none(getattr(parsed_data, "week", None)),
                "measurement_index": index,
                "gnssId": gnss,
                "svid": _int_or_none(getattr(parsed_data, f"svId_{index:02}", None)),
                "sigId": sig,
                "cpMes": _float_or_nan(getattr(parsed_data, f"cpMes_{index:02}", None)),
                "prMes": _float_or_nan(getattr(parsed_data, f"prMes_{index:02}", None)),
                "doMes": _float_or_nan(getattr(parsed_data, f"doMes_{index:02}", None)),
                "cno": _float_or_nan(getattr(parsed_data, f"cno_{index:02}", None)),
                "locktime": _float_or_nan(
                    getattr(parsed_data, f"locktime_{index:02}", None)
                ),
                "cp_valid": _int_or_none(
                    getattr(parsed_data, f"cpValid_{index:02}", None)
                ),
                "pr_valid": _int_or_none(
                    getattr(parsed_data, f"prValid_{index:02}", None)
                ),
                "half_cycle": _int_or_none(
                    getattr(parsed_data, f"halfCyc_{index:02}", None)
                ),
                "cpStd": _float_or_nan(getattr(parsed_data, f"cpStd_{index:02}", None)),
            }
        )
    return rows


def _navsat_rows(parsed_data, gnss_id: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rcv_tow = _float_or_nan(getattr(parsed_data, "iTOW", float("nan"))) / 1000.0
    num_svs = _int_or_none(getattr(parsed_data, "numSvs", None))
    if num_svs is None:
        num_svs = 0
        while hasattr(parsed_data, f"gnssId_{num_svs + 1:02}"):
            num_svs += 1

    for index in range(1, num_svs + 1):
        gnss = _int_or_none(getattr(parsed_data, f"gnssId_{index:02}", None))
        if gnss != gnss_id:
            continue

        rows.append(
            {
                "rcvTow": rcv_tow,
                "iTOW": _int_or_none(getattr(parsed_data, "iTOW", None)),
                "navsat_index": index,
                "gnssId": gnss,
                "svid": _int_or_none(getattr(parsed_data, f"svId_{index:02}", None)),
                "cno": _float_or_nan(getattr(parsed_data, f"cno_{index:02}", None)),
                "elev": _float_or_nan(getattr(parsed_data, f"elev_{index:02}", None)),
                "azim": _float_or_nan(getattr(parsed_data, f"azim_{index:02}", None)),
                "qualityInd": _int_or_none(
                    getattr(parsed_data, f"qualityInd_{index:02}", None)
                ),
                "svUsed": _int_or_none(
                    getattr(parsed_data, f"svUsed_{index:02}", None)
                ),
            }
        )
    return rows


def _navpvt_row(parsed_data) -> dict[str, object]:
    i_tow = _int_or_none(getattr(parsed_data, "iTOW", None))
    return {
        "rcvTow": (i_tow / 1000.0) if i_tow is not None else float("nan"),
        "iTOW": i_tow,
        "fixType": _int_or_none(getattr(parsed_data, "fixType", None)),
        "numSV": _int_or_none(getattr(parsed_data, "numSV", None)),
        "lon": _float_or_nan(getattr(parsed_data, "lon", None)),
        "lat": _float_or_nan(getattr(parsed_data, "lat", None)),
        "height": _float_or_nan(getattr(parsed_data, "height", None)),
        "hMSL": _float_or_nan(getattr(parsed_data, "hMSL", None)),
        "hAcc": _float_or_nan(getattr(parsed_data, "hAcc", None)),
        "vAcc": _float_or_nan(getattr(parsed_data, "vAcc", None)),
        "nano": _float_or_nan(getattr(parsed_data, "nano", None)),
    }


def read_ubx_observations(
    path: str | Path,
    gnss_id: int = 0,
    sig_id: int = 0,
    max_rawx_epochs: int | None = None,
) -> UbxObservationTables:
    """Parse UBX RXM-RAWX, NAV-SAT, and NAV-PVT records into DataFrames."""
    ubx_path = Path(path)
    if not ubx_path.exists():
        raise FileNotFoundError(ubx_path)

    rawx_rows: list[dict[str, object]] = []
    navsat_rows: list[dict[str, object]] = []
    navpvt_rows: list[dict[str, object]] = []
    message_counts: dict[str, int] = {}
    rawx_epoch_count = 0

    with ubx_path.open("rb") as stream:
        for _, parsed_data in _make_ubx_reader(stream):
            identity = getattr(parsed_data, "identity", "")
            message_counts[identity] = message_counts.get(identity, 0) + 1
            if identity == "RXM-RAWX":
                rawx_epoch_count += 1
                rawx_rows.extend(
                    _rawx_rows(parsed_data, gnss_id=gnss_id, sig_id=sig_id)
                )
            elif identity == "NAV-SAT":
                navsat_rows.extend(_navsat_rows(parsed_data, gnss_id=gnss_id))
            elif identity == "NAV-PVT":
                navpvt_rows.append(_navpvt_row(parsed_data))
            if max_rawx_epochs is not None and rawx_epoch_count >= max_rawx_epochs:
                break

    rawx = pd.DataFrame(rawx_rows, columns=RAWX_COLUMNS)
    navsat = pd.DataFrame(navsat_rows, columns=NAVSAT_COLUMNS)
    navpvt = pd.DataFrame(navpvt_rows, columns=NAVPVT_COLUMNS)
    return UbxObservationTables(
        rawx=rawx, navsat=navsat, navpvt=navpvt, message_counts=message_counts
    )


def _with_sync_tow(rawx: pd.DataFrame, tow_round_decimals: int | None) -> pd.DataFrame:
    data = rawx.copy()
    if tow_round_decimals is None:
        data["sync_tow"] = data["rcvTow"]
    else:
        data["sync_tow"] = data["rcvTow"].round(tow_round_decimals)
    return data


def _sync_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    if "week" in left.columns and "week" in right.columns:
        if left["week"].notna().any() and right["week"].notna().any():
            return ["week", "sync_tow"]
    return ["sync_tow"]


def sync_receiver_epochs(
    obs1: UbxObservationTables,
    obs2: UbxObservationTables,
    tow_round_decimals: int | None = None,
) -> SyncedObservations:
    """Synchronize two receivers by the intersection of RAWX TOW epoch keys."""
    if obs1.rawx.empty or obs2.rawx.empty:
        raise ValueError("Both receivers must contain RXM-RAWX observations.")

    rx1_rawx_all = _with_sync_tow(obs1.rawx, tow_round_decimals)
    rx2_rawx_all = _with_sync_tow(obs2.rawx, tow_round_decimals)
    keys = _sync_keys(rx1_rawx_all, rx2_rawx_all)
    left_epochs = (
        rx1_rawx_all[keys + ["rcvTow"]]
        .drop_duplicates()
        .rename(columns={"rcvTow": "rcvTow_rx1"})
    )
    right_epochs = (
        rx2_rawx_all[keys + ["rcvTow"]]
        .drop_duplicates()
        .rename(columns={"rcvTow": "rcvTow_rx2"})
    )
    epochs = (
        left_epochs.merge(right_epochs, how="inner", on=keys)
        .sort_values(keys)
        .reset_index(drop=True)
    )
    epochs.insert(0, "epoch_index", np.arange(len(epochs), dtype=int))
    if epochs.empty:
        raise ValueError("No RAWX TOW overlap exists between receivers.")
    epochs["tow_delta_s"] = epochs["rcvTow_rx1"] - epochs["rcvTow_rx2"]

    rx1_rawx = rx1_rawx_all.merge(epochs[keys + ["epoch_index"]], how="inner", on=keys)
    rx2_rawx = rx2_rawx_all.merge(epochs[keys + ["epoch_index"]], how="inner", on=keys)

    return SyncedObservations(
        epochs=epochs,
        rx1=UbxObservationTables(rawx=rx1_rawx, navsat=obs1.navsat, navpvt=obs1.navpvt),
        rx2=UbxObservationTables(rawx=rx2_rawx, navsat=obs2.navsat, navpvt=obs2.navpvt),
    )


def _slice_synced_epochs(
    synced: SyncedObservations,
    start_index: int,
    length: int | None,
) -> SyncedObservations:
    if start_index < 0:
        raise ValueError("start_index must be non-negative.")

    stop = None if length is None else start_index + length
    selected = synced.epochs.iloc[start_index:stop].copy()
    if selected.empty:
        raise ValueError("The requested epoch window is empty.")

    old_to_new = selected[["epoch_index"]].copy()
    old_to_new = old_to_new.rename(columns={"epoch_index": "source_epoch_index"})
    old_to_new["epoch_index"] = np.arange(len(old_to_new), dtype=int)

    selected = selected.rename(columns={"epoch_index": "source_epoch_index"})
    selected = selected.merge(old_to_new, how="inner", on="source_epoch_index")
    leading_cols = ["epoch_index", "source_epoch_index"]
    other_cols = [col for col in selected.columns if col not in leading_cols]
    selected = selected[leading_cols + other_cols]

    def remap(rawx: pd.DataFrame) -> pd.DataFrame:
        mapped = rawx.rename(columns={"epoch_index": "source_epoch_index"})
        mapped = mapped.merge(old_to_new, how="inner", on="source_epoch_index")
        leading = ["epoch_index", "source_epoch_index"]
        rest = [col for col in mapped.columns if col not in leading]
        return mapped[leading + rest]

    return SyncedObservations(
        epochs=selected.reset_index(drop=True),
        rx1=UbxObservationTables(
            rawx=remap(synced.rx1.rawx),
            navsat=synced.rx1.navsat,
            navpvt=synced.rx1.navpvt,
        ),
        rx2=UbxObservationTables(
            rawx=remap(synced.rx2.rawx),
            navsat=synced.rx2.navsat,
            navpvt=synced.rx2.navpvt,
        ),
    )


def _pivot_rawx(
    rawx: pd.DataFrame,
    value_col: str,
    epoch_index: pd.Index,
    require_valid_phase: bool = False,
) -> pd.DataFrame:
    if rawx.empty:
        return pd.DataFrame(index=epoch_index)

    valid = rawx[value_col].notna() & rawx["svid"].notna()
    if require_valid_phase:
        if rawx["cp_valid"].notna().any():
            valid &= rawx["cp_valid"].fillna(0).astype(int).eq(1)
        # Eq. 18 rounds DD to whole cycles, so unresolved half-cycle ambiguity
        # cannot be treated as a valid carrier-phase measurement.
        if rawx["half_cycle"].notna().any():
            valid &= rawx["half_cycle"].fillna(0).astype(int).eq(1)

    pivot = rawx.loc[valid].pivot_table(
        index="epoch_index",
        columns="svid",
        values=value_col,
        aggfunc="first",
    )
    pivot = pivot.reindex(epoch_index)
    pivot.index.name = "epoch_index"
    pivot.columns = [int(col) for col in pivot.columns]
    return pivot.sort_index(axis=1)


def _attach_navpvt_time_correction(
    rawx: pd.DataFrame,
    navpvt: pd.DataFrame,
    tow_round_decimals: int,
    use_navpvt_time_correction: bool,
) -> pd.DataFrame:
    data = rawx.copy()
    data["carrier_phase_correction_cycles"] = 0.0
    data["navpvt_nano"] = np.nan
    if not use_navpvt_time_correction:
        data["carrier_phase_cycles"] = data["cpMes"]
        return data

    if navpvt.empty:
        data["carrier_phase_correction_cycles"] = np.nan
        data["carrier_phase_cycles"] = np.nan
        return data

    data["_navpvt_sync_tow"] = data["rcvTow"].round(tow_round_decimals)
    if "fixType" in navpvt.columns:
        # Without a GNSS fix (0 = none, 1 = dead reckoning only) the receiver reports
        # nano = 0 instead of its clock offset, so Eq. 8 cannot be applied. Such epochs
        # (e.g. a cold start) become carrier-phase dropouts, not measurements.
        navpvt = navpvt.loc[~navpvt["fixType"].isin((0, 1))]
    nano = navpvt[["rcvTow", "nano"]].dropna(subset=["rcvTow", "nano"]).copy()
    nano["_navpvt_sync_tow"] = nano["rcvTow"].round(tow_round_decimals)
    nano = (
        nano.sort_values("rcvTow")
        .drop_duplicates("_navpvt_sync_tow", keep="first")
        .loc[:, ["_navpvt_sync_tow", "nano"]]
        .rename(columns={"nano": "navpvt_nano"})
    )

    data = data.merge(
        nano, how="left", on="_navpvt_sync_tow", suffixes=("", "_matched")
    )
    data["navpvt_nano"] = data["navpvt_nano_matched"].combine_first(data["navpvt_nano"])
    data = data.drop(columns=["navpvt_nano_matched", "_navpvt_sync_tow"])
    data["carrier_phase_correction_cycles"] = data["navpvt_nano"] * 1e-9 * data["doMes"]
    data["carrier_phase_cycles"] = (
        data["cpMes"] + data["carrier_phase_correction_cycles"]
    )
    return data


def _filter_carrier_phase_columns(
    tables: CarrierPhaseTables,
    svids: Sequence[int] | None,
) -> CarrierPhaseTables:
    if svids is None:
        return tables
    columns = [svid for svid in _normalise_svids(svids) or ()]
    return CarrierPhaseTables(
        rx1_cycles=tables.rx1_cycles.reindex(columns=columns),
        rx2_cycles=tables.rx2_cycles.reindex(columns=columns),
        combined_cycles=tables.combined_cycles.reindex(
            columns=pd.MultiIndex.from_product([["rx1", "rx2"], columns])
        ),
        mean_cno=tables.mean_cno.reindex(columns=columns),
    )


def build_carrier_phase_table(
    synced_epochs: SyncedObservations,
    use_navpvt_time_correction: bool = True,
    navpvt_tow_decimals: int = 0,
) -> CarrierPhaseTables:
    """Build RX1/RX2 GPS L1 carrier-phase tables indexed by synchronized epoch."""
    epoch_index = pd.Index(synced_epochs.epochs["epoch_index"], name="epoch_index")
    rx1_rawx = _attach_navpvt_time_correction(
        synced_epochs.rx1.rawx,
        synced_epochs.rx1.navpvt,
        navpvt_tow_decimals,
        use_navpvt_time_correction,
    )
    rx2_rawx = _attach_navpvt_time_correction(
        synced_epochs.rx2.rawx,
        synced_epochs.rx2.navpvt,
        navpvt_tow_decimals,
        use_navpvt_time_correction,
    )
    rx1 = _pivot_rawx(rx1_rawx, "carrier_phase_cycles", epoch_index, True)
    rx2 = _pivot_rawx(rx2_rawx, "carrier_phase_cycles", epoch_index, True)
    cno1 = _pivot_rawx(rx1_rawx, "cno", epoch_index)
    cno2 = _pivot_rawx(rx2_rawx, "cno", epoch_index)

    columns = sorted(set(rx1.columns) | set(rx2.columns))
    rx1 = rx1.reindex(columns=columns)
    rx2 = rx2.reindex(columns=columns)
    cno1 = cno1.reindex(columns=columns)
    cno2 = cno2.reindex(columns=columns)

    combined = pd.concat({"rx1": rx1, "rx2": rx2}, axis=1)
    mean_cno = pd.concat([cno1, cno2]).groupby(level=0).mean().reindex(epoch_index)
    mean_cno = mean_cno.reindex(columns=columns)
    return CarrierPhaseTables(
        rx1_cycles=rx1,
        rx2_cycles=rx2,
        combined_cycles=combined,
        mean_cno=mean_cno,
    )


def _longest_true_run(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def select_reference_svid(
    carrier_phase: CarrierPhaseTables,
    cno: pd.DataFrame | None = None,
    override: int | None = None,
    exclude_svids: Sequence[int] | None = None,
    required_svids: Sequence[int] | None = None,
) -> tuple[int, pd.DataFrame]:
    """Select a reference satellite, honoring an explicit override first.

    ``override`` is authoritative: it may name an SVID in ``exclude_svids``. This
    lets experiments deliberately use any observed PRN (including a known spoof
    PRN) as the DD reference. Exclusions only constrain automatic selection.
    """
    rx1 = carrier_phase.rx1_cycles
    rx2 = carrier_phase.rx2_cycles
    common_valid = rx1.notna() & rx2.notna()
    excluded = set(_normalise_svids(exclude_svids) or ())
    required = _normalise_svids(required_svids) or ()

    required_mask = pd.Series(True, index=common_valid.index)
    missing_required = [svid for svid in required if svid not in common_valid.columns]
    if missing_required:
        raise ValueError(
            f"required_svids are not present in both receivers: {missing_required}"
        )
    for svid in required:
        required_mask &= common_valid[svid]

    if override is not None:
        if override not in common_valid.columns:
            raise ValueError(
                f"reference_svid={override} is not present in both receivers."
            )
        if not (common_valid[override] & required_mask).any():
            raise ValueError(
                f"reference_svid={override} has no common valid epochs with required_svids."
            )
        selected = int(override)
    else:
        rows: list[dict[str, object]] = []
        single_diff = rx1 - rx2
        for svid in common_valid.columns:
            if int(svid) in excluded:
                continue
            mask = common_valid[svid] & required_mask
            valid_count = int(mask.sum())
            if valid_count == 0:
                continue
            phase_step = single_diff.loc[mask, svid].diff().abs()
            jitter = float(phase_step.median(skipna=True))
            if np.isnan(jitter):
                jitter = float("inf")
            cno_mean = float("nan")
            if cno is not None and svid in cno.columns:
                cno_mean = float(cno.loc[mask, svid].mean(skipna=True))
            rows.append(
                {
                    "svid": int(svid),
                    "valid_epoch_count": valid_count,
                    "longest_continuous_run": _longest_true_run(mask.to_numpy()),
                    "mean_cno": cno_mean,
                    "median_abs_phase_step_cycles": jitter,
                }
            )

        if not rows:
            raise ValueError(
                "No satellite has common valid carrier phase in both receivers."
            )

        ranking = pd.DataFrame(rows)
        ranking["_mean_cno_sort"] = ranking["mean_cno"].fillna(-np.inf)
        ranking["_jitter_sort"] = ranking["median_abs_phase_step_cycles"].replace(
            np.inf, np.nan
        )
        ranking["_jitter_sort"] = ranking["_jitter_sort"].fillna(np.inf)
        ranking = ranking.sort_values(
            by=[
                "longest_continuous_run",
                "valid_epoch_count",
                "_mean_cno_sort",
                "_jitter_sort",
                "svid",
            ],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)
        selected = int(ranking.loc[0, "svid"])

    selection = pd.DataFrame()
    if override is not None:
        selected_mask = common_valid[selected] & required_mask
        selection = pd.DataFrame(
            [
                {
                    "svid": selected,
                    "valid_epoch_count": int(selected_mask.sum()),
                    "longest_continuous_run": _longest_true_run(
                        selected_mask.to_numpy()
                    ),
                    "mean_cno": (
                        float(
                            cno.loc[selected_mask, selected].mean(skipna=True)
                        )
                        if cno is not None and selected in cno.columns
                        else float("nan")
                    ),
                    "median_abs_phase_step_cycles": float("nan"),
                    "manual_override": True,
                }
            ]
        )
    else:
        selection = ranking.drop(columns=["_mean_cno_sort", "_jitter_sort"])
        selection["manual_override"] = False
    return selected, selection


def compute_single_difference(
    carrier_phase_rx1: CarrierPhaseTables | pd.DataFrame,
    carrier_phase_rx2: pd.DataFrame | None = None,
    order: str = "rx1-rx2",
) -> pd.DataFrame:
    """Compute receiver single differences in carrier cycles.

    ``order="rx1-rx2"`` (default) uses ``RX1 - RX2``; ``order="rx2-rx1"`` uses ``RX2 - RX1``,
    which negates the single and double differences. Reversing the order alone does not
    make the solved AoA supplementary: the reference geometry and integer branch must
    also be reversed consistently. With that complete baseline reversal, the same
    source is described by ``180 - alpha`` instead of ``alpha``.
    """
    if order not in ("rx1-rx2", "rx2-rx1"):
        raise ValueError(f"order must be 'rx1-rx2' or 'rx2-rx1', got {order!r}")

    if isinstance(carrier_phase_rx1, CarrierPhaseTables):
        rx1 = carrier_phase_rx1.rx1_cycles
        rx2 = carrier_phase_rx1.rx2_cycles
    else:
        if carrier_phase_rx2 is None:
            raise ValueError("carrier_phase_rx2 is required when passing DataFrames.")
        rx1 = carrier_phase_rx1
        rx2 = carrier_phase_rx2

    columns = sorted(set(rx1.columns) | set(rx2.columns))
    result = rx1.reindex(columns=columns) - rx2.reindex(columns=columns)
    if order == "rx2-rx1":
        result = -result
    result.index.name = "epoch_index"
    return result


def compute_double_difference(
    single_difference: pd.DataFrame,
    reference_svid: int,
) -> pd.DataFrame:
    """Compute satellite double differences in carrier cycles using a reference SV."""
    if reference_svid not in single_difference.columns:
        raise ValueError(
            f"Reference SVID {reference_svid} is not in single differences."
        )
    result = single_difference.subtract(single_difference[reference_svid], axis=0)
    result.index.name = "epoch_index"
    return result


def fractional_round_dd(
    double_difference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Eq. 18: fractional DD = DD cycles - round(DD cycles)."""
    rounded = double_difference.round()
    fractional = double_difference - rounded
    rounded.index.name = "epoch_index"
    fractional.index.name = "epoch_index"
    return rounded, fractional


def generate_aoa_candidates(
    fractional_dd: pd.DataFrame,
    alpha_r_deg: float | pd.Series,
    branch_values: Iterable[int] = (-2, -1, 0, 1, 2),
    baseline_over_wavelength: float = 1.0,
) -> pd.DataFrame:
    """Generate Eq. 21 AoA branch candidates from fractional DD values.

    ``alpha_r_deg`` may be a scalar anchor or an epoch-indexed ephemeris AoA series.
    The latter implements the journal method's time-varying reference direction.
    """
    if baseline_over_wavelength == 0:
        raise ValueError("baseline_over_wavelength must be non-zero.")

    frac = fractional_dd.copy()
    frac.index.name = "epoch_index"
    long = frac.reset_index().melt(
        id_vars=["epoch_index"],
        var_name="svid",
        value_name="fractional_dd_cycles",
    )
    long["svid"] = long["svid"].astype(int)
    if isinstance(alpha_r_deg, pd.Series):
        reference_alpha = pd.to_numeric(alpha_r_deg, errors="coerce").reindex(frac.index)
        reference_alpha.index.name = "epoch_index"
        long["reference_alpha_deg"] = long["epoch_index"].map(reference_alpha)
    else:
        long["reference_alpha_deg"] = float(alpha_r_deg)
    long["reference_cos_alpha"] = np.cos(
        np.deg2rad(long["reference_alpha_deg"].astype(float))
    )

    frames: list[pd.DataFrame] = []
    for branch in branch_values:
        candidate = long.copy()
        candidate["branch"] = int(branch)
        candidate["candidate_cos_alpha"] = (
            candidate["fractional_dd_cycles"] + int(branch)
        ) / baseline_over_wavelength + candidate["reference_cos_alpha"]
        valid = candidate["fractional_dd_cycles"].notna() & candidate[
            "candidate_cos_alpha"
        ].between(-1.0, 1.0, inclusive="both")
        clipped = candidate["candidate_cos_alpha"].clip(-1.0, 1.0)
        candidate["candidate_aoa_deg"] = np.where(
            valid,
            np.rad2deg(np.arccos(clipped)),
            np.nan,
        )
        candidate["valid_candidate"] = valid
        frames.append(candidate)

    return pd.concat(frames, ignore_index=True).drop(columns="reference_cos_alpha")


def _valid_segments(index_values: list[int], valid_mask: pd.Series) -> list[list[int]]:
    segments: list[list[int]] = []
    current: list[int] = []
    previous: int | None = None
    for epoch in index_values:
        valid = bool(valid_mask.loc[epoch])
        if not valid:
            if current:
                segments.append(current)
                current = []
            previous = None
            continue
        if previous is None or epoch == previous + 1:
            current.append(epoch)
        else:
            if current:
                segments.append(current)
            current = [epoch]
        previous = epoch
    if current:
        segments.append(current)
    return segments


BRANCH_METRICS = ("direction_cosine", "aoa_deg")


def _validate_branch_metric(metric: str) -> None:
    if metric not in BRANCH_METRICS:
        raise ValueError(f"metric must be one of {BRANCH_METRICS}, got {metric!r}")


def _dp_select_segment(
    values: np.ndarray,
    cost_values: np.ndarray | None = None,
    tie_break: np.ndarray | None = None,
    tie_tolerance: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Algorithm 1 on one continuous segment of candidates.

    ``values`` (epochs x branches) holds the candidate AoA in degrees that is returned.
    ``cost_values`` (same shape, defaults to ``values``) holds the numbers whose absolute
    epoch-to-epoch step the DP minimises; passing direction cosines makes the Eq. 21
    branches exact unit shifts of each other, so equally smooth branches tie instead of
    the branch nearer 90 deg winning. ``tie_break`` gives one secondary key per column
    (for example ``|m|``): when several end states lie within ``tie_tolerance`` of the
    minimum total cost, the smallest key wins instead of the first column.
    """
    cost_source = values if cost_values is None else cost_values
    if cost_source.shape != values.shape:
        raise ValueError("cost_values must have the same shape as values.")
    rows, cols = values.shape
    costs = np.full((rows, cols), np.inf, dtype=float)
    back = np.full((rows, cols), -1, dtype=int)
    valid_first = np.isfinite(cost_source[0])
    costs[0, valid_first] = 0.0

    for row in range(1, rows):
        previous_values = cost_source[row - 1]
        for current_col in range(cols):
            current_value = cost_source[row, current_col]
            if not np.isfinite(current_value):
                continue
            transition = costs[row - 1] + np.abs(current_value - previous_values)
            if not np.isfinite(transition).any():
                continue
            best_previous = int(np.nanargmin(transition))
            costs[row, current_col] = transition[best_previous]
            back[row, current_col] = best_previous

    end_costs = costs[-1]
    finite_end = np.isfinite(end_costs)
    if not finite_end.any():
        return np.full(rows, np.nan), np.full(rows, np.nan)
    min_cost = float(end_costs[finite_end].min())
    tied = np.flatnonzero(finite_end & (end_costs <= min_cost + tie_tolerance))
    end_col = int(tied[0])
    if tie_break is not None and len(tied) > 1:
        keys = np.asarray(tie_break, dtype=float)[tied]
        end_col = int(tied[int(np.argmin(keys))])

    selected_cols = np.full(rows, end_col, dtype=int)
    for row in range(rows - 1, 0, -1):
        selected_cols[row - 1] = back[row, selected_cols[row]]

    selected_values = values[np.arange(rows), selected_cols]
    return selected_values, selected_cols.astype(float)


def select_aoa_dp(
    candidates: pd.DataFrame,
    reference_svid: int | None = None,
    reference_angle_deg: float | pd.Series | None = None,
    return_branches: bool = False,
    *,
    metric: str = "direction_cosine",
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Select a smooth AoA sequence per satellite using Algorithm 1.

    ``metric`` is the quantity whose epoch-to-epoch change the DP minimises.
    ``"direction_cosine"`` (default) scores ``cos(alpha)``: the Eq. 21 branches then differ
    by exactly ``1 / (D / lambda)`` at every epoch, so equally smooth branches tie and the
    tie goes to the smallest ``|m|``. ``"aoa_deg"`` scores degrees, the original behaviour;
    arccos maps the same phase noise to smaller degree steps near 90 deg, so that metric
    favours the branch nearer 90 deg whatever the true direction.
    """
    _validate_branch_metric(metric)
    required = {"epoch_index", "svid", "branch", "candidate_aoa_deg"}
    if metric == "direction_cosine":
        required.add("candidate_cos_alpha")
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates is missing required columns: {sorted(missing)}")

    all_epochs = sorted(int(epoch) for epoch in candidates["epoch_index"].unique())
    all_svids = sorted(int(svid) for svid in candidates["svid"].unique())
    branch_values = sorted(int(branch) for branch in candidates["branch"].unique())
    tie_break = np.abs(np.asarray(branch_values, dtype=float))
    selected = pd.DataFrame(index=all_epochs, columns=all_svids, dtype=float)
    branches = pd.DataFrame(index=all_epochs, columns=all_svids, dtype=float)
    selected.index.name = "epoch_index"
    branches.index.name = "epoch_index"

    for svid in all_svids:
        subset = candidates.loc[candidates["svid"].eq(svid)]
        pivot = subset.pivot_table(
            index="epoch_index",
            columns="branch",
            values="candidate_aoa_deg",
            aggfunc="first",
        )
        pivot = pivot.reindex(index=all_epochs, columns=branch_values)
        cost_pivot = None
        if metric == "direction_cosine":
            cost_pivot = subset.pivot_table(
                index="epoch_index",
                columns="branch",
                values="candidate_cos_alpha",
                aggfunc="first",
            )
            cost_pivot = cost_pivot.reindex(index=all_epochs, columns=branch_values)
            cost_pivot = cost_pivot.where(pivot.notna())
        valid_mask = pivot.notna().any(axis=1)
        for segment in _valid_segments(all_epochs, valid_mask):
            values = pivot.loc[segment].to_numpy(dtype=float)
            cost_values = (
                None
                if cost_pivot is None
                else cost_pivot.loc[segment].to_numpy(dtype=float)
            )
            segment_values, selected_cols = _dp_select_segment(
                values, cost_values, tie_break
            )
            selected.loc[segment, svid] = segment_values
            for epoch, col in zip(segment, selected_cols):
                if np.isfinite(col):
                    branches.loc[epoch, svid] = branch_values[int(col)]

    if reference_svid is not None and reference_svid in selected.columns:
        if reference_angle_deg is not None:
            ref_mask = candidates.loc[
                candidates["svid"].eq(reference_svid)
                & candidates["fractional_dd_cycles"].notna()
            ]
            valid_ref_epochs = sorted(
                int(epoch) for epoch in ref_mask["epoch_index"].unique()
            )
            if isinstance(reference_angle_deg, pd.Series):
                reference_values = pd.to_numeric(
                    reference_angle_deg, errors="coerce"
                ).reindex(valid_ref_epochs)
                selected.loc[valid_ref_epochs, reference_svid] = reference_values.to_numpy()
            else:
                selected.loc[valid_ref_epochs, reference_svid] = float(reference_angle_deg)
            branches.loc[valid_ref_epochs, reference_svid] = 0.0

    if return_branches:
        return selected, branches
    return selected


def candidate_aoa_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    """Per-satellite table of valid branch candidates -- the 'maybe' AoA values.

    For each satellite and branch ``m``, report the median candidate AoA over the epochs where
    that branch is valid (``cos(alpha)`` within ``[-1, 1]``) and the count of such epochs.
    Because ``cos(alpha)`` is bounded to ``[-1, 1]``, extreme branches (``m = +-2``) are usually
    rare or impossible, so most satellites show only one or two candidate angles. Pick the
    correct branch afterwards (e.g. with :func:`select_aoa_by_reference`).
    """
    required = {"svid", "branch", "candidate_aoa_deg"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates is missing required columns: {sorted(missing)}")

    valid = candidates.dropna(subset=["candidate_aoa_deg"]).copy()
    if valid.empty:
        return pd.DataFrame()
    valid["svid"] = valid["svid"].astype(int)
    valid["branch"] = valid["branch"].astype(int)

    aoa = valid.pivot_table(
        index="svid", columns="branch", values="candidate_aoa_deg", aggfunc="median"
    ).rename(columns=lambda m: f"aoa_m{int(m):+d}")
    count = valid.pivot_table(
        index="svid", columns="branch", values="candidate_aoa_deg", aggfunc="count"
    ).rename(columns=lambda m: f"n_m{int(m):+d}")
    summary = aoa.join(count)
    summary.index.name = "svid"
    return summary.sort_index()


def select_aoa_by_reference(
    candidates: pd.DataFrame,
    reference_aoa: pd.DataFrame,
    return_branches: bool = False,
    min_valid_fraction: float = 0.8,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Resolve one constant branch per satellite by matching a reference AoA.

    Legacy/comparison selection: the paper flow uses :func:`select_aoa_by_reference_per_epoch`
    for authentic SVs, because a constant ``m`` breaks when the fractional DD crosses +-0.5.
    Still used by the pairwise classifier's ``branch_selection_method="best_m"``.

    For each satellite, pick one branch ``m`` that is valid across most of the satellite's
    valid epochs and has the smallest median error to ``reference_aoa`` (e.g.
    ephemeris-based AoA), then return that branch's candidate AoA at every epoch. This
    resolves the branch ambiguity from geometry instead of from time-smoothness, so the
    absolute level is anchored per satellite.

    Note: matching the branch to the ephemeris biases spoofed PRNs toward agreement, so the
    returned AoA is for authentic-vs-ephemeris comparison, not a standalone spoof test (use the
    fractional-DD residual against a known baseline for detection).
    """
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in the interval (0, 1].")

    valid = candidates.dropna(subset=["candidate_aoa_deg"]).copy()
    valid["svid"] = valid["svid"].astype(int)
    valid["branch"] = valid["branch"].astype(int)

    all_epochs = sorted(int(epoch) for epoch in candidates["epoch_index"].unique())
    svids = sorted(int(svid) for svid in valid["svid"].unique())
    reference_median = reference_aoa.median()

    selected = pd.DataFrame(index=all_epochs, columns=svids, dtype=float)
    selected.index.name = "epoch_index"
    chosen: dict[int, int] = {}
    for svid in svids:
        if svid not in reference_median.index or not np.isfinite(reference_median[svid]):
            continue
        subset = valid.loc[valid["svid"].eq(svid)]
        all_valid_epochs = int(subset["epoch_index"].nunique())
        min_epoch_count = max(1, int(np.ceil(all_valid_epochs * min_valid_fraction)))
        branch_epoch_counts = subset.groupby("branch")["epoch_index"].nunique()
        eligible_branches = branch_epoch_counts.loc[
            branch_epoch_counts.ge(min_epoch_count)
        ].index
        if len(eligible_branches) == 0:
            eligible_branches = branch_epoch_counts.loc[
                branch_epoch_counts.eq(branch_epoch_counts.max())
            ].index

        branch_scores: dict[int, float] = {}
        reference = reference_aoa[svid].rename("reference_aoa_deg")
        for branch in eligible_branches:
            branch_subset = subset.loc[subset["branch"].eq(int(branch))]
            comparison = branch_subset.merge(
                reference.reset_index(), on="epoch_index", how="left"
            ).dropna(subset=["reference_aoa_deg"])
            if comparison.empty:
                continue
            branch_scores[int(branch)] = float(
                (
                    comparison["candidate_aoa_deg"]
                    - comparison["reference_aoa_deg"]
                )
                .abs()
                .median()
            )
        if not branch_scores:
            continue

        best_branch = min(branch_scores, key=branch_scores.get)
        chosen[svid] = best_branch
        series = subset.loc[subset["branch"].eq(best_branch)].set_index("epoch_index")[
            "candidate_aoa_deg"
        ]
        selected[svid] = series.reindex(all_epochs)

    if return_branches:
        return selected, pd.Series(chosen, name="chosen_branch").sort_index()
    return selected


def select_aoa_by_reference_per_epoch(
    candidates: pd.DataFrame,
    reference_aoa: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick the branch ``m`` at every epoch whose direction cosine is closest to the reference.

    A constant ``m`` (:func:`select_aoa_by_reference`) is only valid while the fractional DD
    stays on one side of the +-0.5 cycle wrap: when a satellite's ``cos(alpha_i) - cos(alpha_r)``
    is near a half-integer, noise flips the rounded integer and the same ``m`` lands one full
    cycle away (e.g. 30-1 PRN 2 jumping between 90 and 165 deg). Matching
    ``cos(reference_aoa)`` per epoch follows the wrap. The nearest branch is taken over all
    branches, valid or not; when it falls outside [-1, 1] the epoch stays NaN rather than
    switching to a valid branch one cycle away.

    For authentic satellites only: like :func:`select_aoa_by_reference`, this ties the DD-method
    AoA to the ephemeris under test, so it is a geometry-conditioned comparison, not a spoof test.
    Returns ``(aoa_deg, branch)``, both epoch x SVID.
    """
    frame = candidates.dropna(subset=["fractional_dd_cycles", "candidate_cos_alpha"]).copy()
    frame["svid"] = frame["svid"].astype(int)
    frame["branch"] = frame["branch"].astype(int)
    all_epochs = sorted(int(epoch) for epoch in candidates["epoch_index"].unique())
    reference = reference_aoa.copy()
    reference.columns = [int(column) for column in reference.columns]
    reference_long = (
        reference.rename_axis("epoch_index").reset_index()
        .melt(id_vars="epoch_index", var_name="svid", value_name="reference_aoa_deg")
        .dropna(subset=["reference_aoa_deg"])
    )
    reference_long["svid"] = reference_long["svid"].astype(int)
    frame = frame.merge(reference_long, on=["epoch_index", "svid"], how="inner")
    svids = sorted(int(svid) for svid in frame["svid"].unique())
    empty = pd.DataFrame(index=pd.Index(all_epochs, name="epoch_index"), columns=svids, dtype=float)
    if frame.empty:
        return empty, empty.copy()
    frame["cos_error"] = (
        frame["candidate_cos_alpha"] - np.cos(np.deg2rad(frame["reference_aoa_deg"]))
    ).abs()
    best = frame.loc[frame.groupby(["epoch_index", "svid"])["cos_error"].idxmin()]
    aoa = best.pivot(index="epoch_index", columns="svid", values="candidate_aoa_deg")
    branch = best.pivot(index="epoch_index", columns="svid", values="branch")
    return (
        aoa.reindex(index=empty.index, columns=svids).astype(float),
        branch.reindex(index=empty.index, columns=svids).astype(float),
    )


def valid_m_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return the physically valid AoA candidates for each SVID/epoch/branch."""
    required = {
        "epoch_index",
        "svid",
        "branch",
        "fractional_dd_cycles",
        "candidate_cos_alpha",
        "candidate_aoa_deg",
        "valid_candidate",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates is missing required columns: {sorted(missing)}")

    valid = candidates.loc[candidates["valid_candidate"]].copy()
    valid = valid.dropna(subset=["candidate_aoa_deg"])
    if valid.empty:
        return valid
    valid["epoch_index"] = valid["epoch_index"].astype(int)
    valid["svid"] = valid["svid"].astype(int)
    valid["branch"] = valid["branch"].astype(int)
    return valid.sort_values(["svid", "branch", "epoch_index"]).reset_index(drop=True)


def summarize_valid_m_candidates(
    candidates: pd.DataFrame,
    *,
    reference_aoa: pd.DataFrame | None = None,
    selected_branch: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize every physically valid integer branch ``m`` per SVID."""
    valid = valid_m_candidates(candidates)
    if valid.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | None]] = []
    for (svid, branch), subset in valid.groupby(["svid", "branch"], sort=True):
        row: dict[str, float | int | None] = {
            "svid": int(svid),
            "branch": int(branch),
            "valid_epoch_count": int(len(subset)),
            "median_candidate_aoa_deg": float(subset["candidate_aoa_deg"].median()),
            "mean_candidate_aoa_deg": float(subset["candidate_aoa_deg"].mean()),
            "min_candidate_aoa_deg": float(subset["candidate_aoa_deg"].min()),
            "max_candidate_aoa_deg": float(subset["candidate_aoa_deg"].max()),
            "median_candidate_cos_alpha": float(
                subset["candidate_cos_alpha"].median()
            ),
        }

        if reference_aoa is not None and int(svid) in reference_aoa.columns:
            reference = reference_aoa[int(svid)].rename("reference_aoa_deg")
            comparison = subset.merge(
                reference.reset_index(), on="epoch_index", how="left"
            ).dropna(subset=["reference_aoa_deg"])
            if not comparison.empty:
                abs_error = (
                    comparison["candidate_aoa_deg"]
                    - comparison["reference_aoa_deg"]
                ).abs()
                row.update(
                    {
                        "reference_epoch_count": int(len(comparison)),
                        "median_reference_aoa_deg": float(
                            comparison["reference_aoa_deg"].median()
                        ),
                        "median_abs_error_to_reference_deg": float(
                            abs_error.median()
                        ),
                        "mean_abs_error_to_reference_deg": float(abs_error.mean()),
                        "within_10deg_fraction": float((abs_error <= 10.0).mean()),
                    }
                )

        if selected_branch is not None and int(svid) in selected_branch.columns:
            branch_series = selected_branch[int(svid)].reindex(subset["epoch_index"])
            row["dp_selected_epoch_fraction"] = float(
                branch_series.eq(int(branch)).mean()
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["svid", "branch"]).reset_index(drop=True)


def _branch_series_to_frame(branches: pd.Series) -> pd.DataFrame:
    """Convert a per-SVID branch Series into a stable CSV-friendly table."""
    if branches.empty:
        return pd.DataFrame(columns=["svid", "branch"])
    frame = branches.rename_axis("svid").reset_index()
    frame.columns = ["svid", "branch"]
    frame["svid"] = frame["svid"].astype(int)
    frame["branch"] = frame["branch"].astype(int)
    return frame.sort_values("svid").reset_index(drop=True)


def summarize_per_epoch_branches(
    branch_by_epoch: pd.DataFrame, aoa: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-SVID summary of per-epoch branches: most common ``m``, every ``m`` used, switches.

    ``branch`` is the most common ``m`` over epochs with a valid AoA (all epochs when ``aoa`` is
    omitted); ``branches`` lists every ``m`` used there; ``branch_switches`` counts changes
    between consecutive valid epochs (a switch across the +-0.5 cycle wrap is expected).
    """
    rows = []
    for column in branch_by_epoch.columns:
        branch = branch_by_epoch[column]
        if aoa is not None:
            branch = branch.where(aoa[column].reindex(branch.index).notna()) if column in aoa else branch.iloc[:0]
        branch = branch.dropna().astype(int)
        if branch.empty:
            continue
        rows.append({
            "svid": int(column),
            "branch": int(branch.mode().iloc[0]),
            "branches": ";".join(str(value) for value in sorted(branch.unique())),
            "branch_switches": int(branch.diff().fillna(0).ne(0).sum()),
            "valid_epochs": int(len(branch)),
        })
    if not rows:
        return pd.DataFrame(columns=["svid", "branch", "branches", "branch_switches", "valid_epochs"])
    return pd.DataFrame(rows).sort_values("svid").reset_index(drop=True)


def summarize_aoa_method_equivalence(
    *,
    dp_aoa: pd.DataFrame,
    dp_branch: pd.DataFrame | None = None,
    ephemeris_aligned_aoa: pd.DataFrame,
    ephemeris_aoa: pd.DataFrame,
    ephemeris_aligned_branch: pd.Series,
    per_epoch_aligned_aoa: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare DD-method AoA variants against ephemeris-based AoA per SVID."""

    dp_branch_by_svid = pd.Series(dtype=float)
    if dp_branch is not None:
        branch_modes: dict[int, float] = {}
        for column in dp_branch.columns:
            series = dp_branch[column].dropna()
            if not series.empty:
                branch_modes[int(column)] = float(series.mode().iloc[0])
        dp_branch_by_svid = pd.Series(branch_modes)

    def summarize_one(
        method: str,
        dd_aoa: pd.DataFrame,
        branch_by_svid: pd.Series | None = None,
    ) -> list[dict[str, float | int | str | None]]:
        rows: list[dict[str, float | int | str | None]] = []
        for svid in sorted(int(column) for column in dd_aoa.columns):
            if svid not in ephemeris_aoa.columns:
                continue
            joined = pd.concat(
                [
                    dd_aoa[svid].rename("dd_aoa_deg"),
                    ephemeris_aoa[svid].rename("ephemeris_aoa_deg"),
                ],
                axis=1,
            ).dropna()
            if joined.empty:
                continue
            abs_error = (
                joined["dd_aoa_deg"] - joined["ephemeris_aoa_deg"]
            ).abs()
            branch_value: int | None = None
            if branch_by_svid is not None and svid in branch_by_svid.index:
                branch_value = int(branch_by_svid.loc[svid])
            rows.append(
                {
                    "method": method,
                    "svid": int(svid),
                    "branch": branch_value,
                    "epoch_count": int(len(joined)),
                    "median_abs_error_deg": float(abs_error.median()),
                    "mean_abs_error_deg": float(abs_error.mean()),
                    "p90_abs_error_deg": float(abs_error.quantile(0.9)),
                    "within_5deg_fraction": float((abs_error <= 5.0).mean()),
                    "within_10deg_fraction": float((abs_error <= 10.0).mean()),
                }
            )
        return rows

    rows = summarize_one("dp_smoothness", dp_aoa, dp_branch_by_svid)
    rows.extend(
        summarize_one(
            "ephemeris_aligned_constant_m",
            ephemeris_aligned_aoa,
            ephemeris_aligned_branch,
        )
    )
    if per_epoch_aligned_aoa is not None:
        rows.extend(summarize_one("ephemeris_aligned_per_epoch_m", per_epoch_aligned_aoa))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["svid", "method"]).reset_index(drop=True)


def build_best_m_all_satellite_aoa(
    dp_aoa: pd.DataFrame,
    ephemeris_aligned_aoa: pd.DataFrame,
    spoof_svids: Sequence[int] = (),
    *,
    joint_source_aoa: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Use the best (ephemeris-aligned) branch ``m`` for authentic SVs, DP branch for spoofed SVs.

    The time-smoothness DP (:func:`select_aoa_dp`) sometimes locks onto the wrong integer branch
    ``m`` for an authentic satellite, which inflates its DD-vs-ephemeris error even though the
    signal is genuine (e.g. SV24/SV25 in the partial-spoof set). For authentic SVs this swaps in
    the ephemeris-aligned AoA -- :func:`run_paper_scenario` passes the per-epoch selection
    (:func:`select_aoa_by_reference_per_epoch`), which resolves the branch from geometry and
    follows the +-0.5 cycle wrap.

    Spoofed SVs use the joint-source selection when supplied, otherwise the DP branch:
    ephemeris-aligning a spoofed PRN biases it
    toward the very ephemeris under test and scatters it off the common-source collapse, so the
    spoof signature is preserved by leaving those columns untouched. Authentic SVs without an
    ephemeris reference (no NAV-SAT AoA) also fall back to the DP branch.
    """
    spoof = set(_normalise_svids(spoof_svids) or ())
    ephemeris_by_svid = {
        int(column): column for column in ephemeris_aligned_aoa.columns
    }
    combined = dp_aoa.copy()
    for column in combined.columns:
        svid = int(column)
        if svid in spoof and joint_source_aoa is not None and svid in joint_source_aoa:
            combined[column] = joint_source_aoa[svid].reindex(combined.index)
            continue
        if svid in spoof or svid not in ephemeris_by_svid:
            continue
        combined[column] = ephemeris_aligned_aoa[ephemeris_by_svid[svid]].reindex(
            combined.index
        )
    if combined.empty:
        return combined
    return sort_svid_columns(combined)


def compute_collapse_metrics(
    selected_aoa: pd.DataFrame,
    thresholds_deg: Iterable[float] = (5.0, 10.0, 15.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute cross-satellite AoA dispersion metrics for a DD-only run."""
    valid_count = selected_aoa.count(axis=1)
    epoch_median = selected_aoa.median(axis=1, skipna=True)
    epoch_min = selected_aoa.min(axis=1, skipna=True)
    epoch_max = selected_aoa.max(axis=1, skipna=True)
    dispersion = epoch_max - epoch_min
    mad = selected_aoa.subtract(epoch_median, axis=0).abs().median(axis=1, skipna=True)

    per_epoch = pd.DataFrame(
        {
            "valid_satellite_count": valid_count,
            "median_aoa_deg": epoch_median,
            "min_aoa_deg": epoch_min,
            "max_aoa_deg": epoch_max,
            "max_minus_min_dispersion_deg": dispersion,
            "median_absolute_deviation_deg": mad,
        }
    )
    per_epoch.index.name = "epoch_index"

    usable = per_epoch.loc[per_epoch["valid_satellite_count"].ge(2)]
    row: dict[str, object] = {
        "usable_epoch_count": int(len(usable)),
        "total_epoch_count": int(len(per_epoch)),
        "median_dispersion_deg": float(
            usable["max_minus_min_dispersion_deg"].median(skipna=True)
        )
        if not usable.empty
        else float("nan"),
        "p90_dispersion_deg": float(
            usable["max_minus_min_dispersion_deg"].quantile(0.9)
        )
        if not usable.empty
        else float("nan"),
        "median_mad_deg": float(
            usable["median_absolute_deviation_deg"].median(skipna=True)
        )
        if not usable.empty
        else float("nan"),
    }
    for threshold in thresholds_deg:
        key = f"pct_epochs_dispersion_le_{int(threshold)}deg"
        row[key] = (
            float((usable["max_minus_min_dispersion_deg"] <= threshold).mean() * 100.0)
            if not usable.empty
            else float("nan")
        )

    summary = pd.DataFrame([row])
    return per_epoch, summary


def compute_relative_dd_collapse_metrics(
    fractional_dd: pd.DataFrame,
    thresholds_cycles: Iterable[float] = (0.05, 0.1, 0.15),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute cross-satellite collapse from fractional DD values in cycles."""
    valid_count = fractional_dd.count(axis=1)
    epoch_median = fractional_dd.median(axis=1, skipna=True)
    epoch_min = fractional_dd.min(axis=1, skipna=True)
    epoch_max = fractional_dd.max(axis=1, skipna=True)
    dispersion = epoch_max - epoch_min
    mad = fractional_dd.subtract(epoch_median, axis=0).abs().median(axis=1, skipna=True)

    per_epoch = pd.DataFrame(
        {
            "valid_satellite_count": valid_count,
            "median_fractional_dd_cycles": epoch_median,
            "min_fractional_dd_cycles": epoch_min,
            "max_fractional_dd_cycles": epoch_max,
            "max_minus_min_dispersion_cycles": dispersion,
            "median_absolute_deviation_cycles": mad,
        }
    )
    per_epoch.index.name = "epoch_index"

    usable = per_epoch.loc[per_epoch["valid_satellite_count"].ge(2)]
    row: dict[str, object] = {
        "usable_epoch_count": int(len(usable)),
        "total_epoch_count": int(len(per_epoch)),
        "median_dispersion_cycles": float(
            usable["max_minus_min_dispersion_cycles"].median(skipna=True)
        )
        if not usable.empty
        else float("nan"),
        "p90_dispersion_cycles": float(
            usable["max_minus_min_dispersion_cycles"].quantile(0.9)
        )
        if not usable.empty
        else float("nan"),
        "median_mad_cycles": float(
            usable["median_absolute_deviation_cycles"].median(skipna=True)
        )
        if not usable.empty
        else float("nan"),
    }
    for threshold in thresholds_cycles:
        threshold_key = str(threshold).replace(".", "p")
        key = f"pct_epochs_dispersion_le_{threshold_key}_cycles"
        row[key] = (
            float(
                (usable["max_minus_min_dispersion_cycles"] <= threshold).mean() * 100.0
            )
            if not usable.empty
            else float("nan")
        )

    summary = pd.DataFrame([row])
    return per_epoch, summary


def _drop_invalid_navsat_geometry(navsat: pd.DataFrame) -> pd.DataFrame:
    """Drop NAV-SAT rows whose elevation/azimuth are out of range.

    u-blox NAV-SAT marks elevation as 'unknown' when outside [-90, 90] deg (e.g. the sentinel
    -91) and azimuth outside [0, 360] deg; such rows would otherwise yield a spurious
    ephemeris AoA (for example SV13 in this dataset).
    """
    elev = navsat["elev"].astype(float)
    azim = navsat["azim"].astype(float)
    return navsat.loc[elev.abs().le(90.0) & azim.between(0.0, 360.0)]


def compute_navsat_aoa(
    navsat: pd.DataFrame,
    baseline_azimuth_bias_deg: float | None = None,
) -> pd.DataFrame:
    """Compute receiver-reported NAV-SAT AoA diagnostic from azimuth/elevation."""
    if navsat.empty:
        return pd.DataFrame()

    bias = (
        0.0 if baseline_azimuth_bias_deg is None else float(baseline_azimuth_bias_deg)
    )
    data = navsat.dropna(subset=["rcvTow", "svid", "elev", "azim"]).copy()
    data = _drop_invalid_navsat_geometry(data)
    if data.empty:
        return pd.DataFrame()

    data["navsat_epoch_index"] = pd.factorize(data["rcvTow"], sort=True)[0]
    elevation_rad = np.deg2rad(data["elev"].astype(float))
    azimuth_rad = np.deg2rad(data["azim"].astype(float) - bias)
    cos_alpha = np.cos(elevation_rad) * np.cos(azimuth_rad)
    data["navsat_aoa_deg"] = np.rad2deg(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
    diagnostic = data.pivot_table(
        index="navsat_epoch_index",
        columns="svid",
        values="navsat_aoa_deg",
        aggfunc="first",
    )
    diagnostic.columns = [int(col) for col in diagnostic.columns]
    diagnostic.index.name = "epoch_index"
    return diagnostic.sort_index(axis=1)


def compute_reference_alpha_by_epoch(
    epochs: pd.DataFrame,
    navsat: pd.DataFrame,
    reference_svid: int,
    baseline_azimuth_bias_deg: float | None = None,
    navsat_tow_decimals: int = 0,
    epoch_tow_column: str = "rcvTow_rx1",
) -> pd.DataFrame:
    """Match a reference satellite's NAV-SAT AoA to synchronized RAWX epochs."""
    required_epoch_columns = {"epoch_index", epoch_tow_column}
    missing_epoch_columns = required_epoch_columns - set(epochs.columns)
    if missing_epoch_columns:
        raise ValueError(
            f"epochs is missing required columns: {sorted(missing_epoch_columns)}"
        )
    if navsat.empty:
        raise ValueError("NAV-SAT observations are required for reference AoA.")

    bias = (
        0.0 if baseline_azimuth_bias_deg is None else float(baseline_azimuth_bias_deg)
    )
    nav = navsat.loc[navsat["svid"].eq(int(reference_svid))].copy()
    nav = nav.dropna(subset=["rcvTow", "elev", "azim"])
    nav = _drop_invalid_navsat_geometry(nav)
    if nav.empty:
        raise ValueError(
            f"No valid NAV-SAT AoA rows found for reference SV {reference_svid}."
        )

    nav["_navsat_sync_tow"] = nav["rcvTow"].round(navsat_tow_decimals)
    nav = nav.sort_values("rcvTow").drop_duplicates("_navsat_sync_tow", keep="first")
    elevation_rad = np.deg2rad(nav["elev"].astype(float))
    azimuth_rad = np.deg2rad(nav["azim"].astype(float) - bias)
    cos_alpha = np.cos(elevation_rad) * np.cos(azimuth_rad)
    nav["reference_alpha_deg"] = np.rad2deg(
        np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    )

    epoch_data = epochs[["epoch_index", epoch_tow_column]].copy()
    epoch_data["_navsat_sync_tow"] = epoch_data[epoch_tow_column].round(
        navsat_tow_decimals
    )
    matched = epoch_data.merge(
        nav[
            [
                "_navsat_sync_tow",
                "rcvTow",
                "svid",
                "elev",
                "azim",
                "cno",
                "svUsed",
                "reference_alpha_deg",
            ]
        ],
        how="left",
        on="_navsat_sync_tow",
    )
    matched = matched.rename(
        columns={
            "rcvTow": "navsat_rcvTow",
            "svid": "reference_svid",
            "elev": "reference_elev_deg",
            "azim": "reference_azim_deg",
            "cno": "reference_cno",
            "svUsed": "reference_svUsed",
        }
    )
    return matched.drop(columns=["_navsat_sync_tow"])


def valid_epoch_segments(
    valid_mask: pd.Series, *, max_gap_epochs: int = 0
) -> list[list[int]]:
    """Return every window in time order, retaining at most ``max_gap_epochs`` per gap.

    Both endpoints of each window must be valid. Internal invalid epochs remain in
    the returned window, without changing the mask or filling any measurements.
    """
    if isinstance(max_gap_epochs, bool) or not isinstance(max_gap_epochs, (int, np.integer)) or max_gap_epochs < 0:
        raise ValueError("max_gap_epochs must be a non-negative integer.")
    if not valid_mask.index.is_unique or not valid_mask.index.is_monotonic_increasing:
        raise ValueError("Epoch indices must be unique and increasing.")
    segments: list[list[int]] = []
    current: list[int] = []
    previous: int | None = None

    for epoch, valid in valid_mask.fillna(False).items():
        epoch = int(epoch)
        if not bool(valid):
            continue

        if previous is None or epoch - previous - 1 <= max_gap_epochs:
            current.append(epoch)
        else:
            if current:
                segments.append(current)
            current = [epoch]
        previous = epoch

    if current:
        segments.append(current)
    return [
        [int(epoch) for epoch in valid_mask.index if segment[0] <= epoch <= segment[-1]]
        for segment in segments
    ]


def longest_valid_epoch_segment(
    valid_mask: pd.Series, *, max_gap_epochs: int = 0
) -> list[int]:
    """Return the longest window of :func:`valid_epoch_segments`.

    Rank by span, then valid-epoch count, then earliest start. Zero preserves
    strict selection.
    """
    segments = valid_epoch_segments(valid_mask, max_gap_epochs=max_gap_epochs)
    if not segments:
        return []
    return max(
        segments,
        key=lambda epochs: (epochs[-1] - epochs[0], int(valid_mask.reindex(epochs).fillna(False).sum())),
    )


def _minimum_valid_spoof_prns(minimum: int | None, group_size: int) -> int:
    if minimum is None:
        return group_size
    if isinstance(minimum, bool) or not isinstance(minimum, (int, np.integer)) or not 1 <= minimum <= group_size:
        raise ValueError(f"min_valid_spoof_prns must be between 1 and {group_size}, or None for all PRNs.")
    return int(minimum)


def sort_svid_columns(
    frame: pd.DataFrame,
    selected_svids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Sort regular SVID columns numerically and optionally reindex to a subset."""
    result = frame.copy()
    result.columns = [int(column) for column in result.columns]
    result = result.sort_index(axis=1)
    selected = _normalise_svids(selected_svids)
    if selected is not None:
        result = result.reindex(columns=list(selected))
    result.index.name = "epoch_index"
    return result


def _navsat_epoch_merge(
    epochs: pd.DataFrame,
    navsat: pd.DataFrame,
    *,
    epoch_tow_column: str,
    navsat_tow_decimals: int,
) -> pd.DataFrame:
    required = {"epoch_index", epoch_tow_column}
    missing = required - set(epochs.columns)
    if missing:
        raise ValueError(f"epochs is missing required columns: {sorted(missing)}")

    epoch_data = epochs[["epoch_index", epoch_tow_column]].copy()
    epoch_data["_navsat_sync_tow"] = epoch_data[epoch_tow_column].round(
        navsat_tow_decimals
    )

    if navsat.empty:
        return epoch_data.assign(svid=np.nan)

    nav = navsat.dropna(subset=["rcvTow", "svid"]).copy()
    nav["svid"] = nav["svid"].astype(int)
    nav["_navsat_sync_tow"] = nav["rcvTow"].round(navsat_tow_decimals)
    nav = nav.sort_values("rcvTow").drop_duplicates(
        ["_navsat_sync_tow", "svid"], keep="first"
    )
    return epoch_data.merge(nav, how="left", on="_navsat_sync_tow")


def compute_navsat_aoa_by_synced_epoch(
    epochs: pd.DataFrame,
    navsat: pd.DataFrame,
    *,
    epoch_tow_column: str,
    navsat_tow_decimals: int = 0,
    baseline_azimuth_bias_deg: float | None = None,
    selected_svids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Compute NAV-SAT ephemeris-style AoA matched to synchronized RAWX epochs."""
    matched = _navsat_epoch_merge(
        epochs,
        navsat,
        epoch_tow_column=epoch_tow_column,
        navsat_tow_decimals=navsat_tow_decimals,
    )
    matched = matched.dropna(subset=["svid", "elev", "azim"]).copy()
    matched = _drop_invalid_navsat_geometry(matched)
    if matched.empty:
        return pd.DataFrame(index=epochs["epoch_index"].astype(int))

    bias = (
        0.0 if baseline_azimuth_bias_deg is None else float(baseline_azimuth_bias_deg)
    )
    elevation_rad = np.deg2rad(matched["elev"].astype(float))
    azimuth_rad = np.deg2rad(matched["azim"].astype(float) - bias)
    cos_alpha = np.cos(elevation_rad) * np.cos(azimuth_rad)
    matched["ephemeris_aoa_deg"] = np.rad2deg(
        np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    )
    pivot = matched.pivot_table(
        index="epoch_index",
        columns="svid",
        values="ephemeris_aoa_deg",
        aggfunc="first",
    )
    pivot = pivot.reindex(epochs["epoch_index"].astype(int))
    return sort_svid_columns(pivot, selected_svids)


def compute_navsat_sv_used_by_synced_epoch(
    epochs: pd.DataFrame,
    navsat: pd.DataFrame,
    *,
    epoch_tow_column: str,
    navsat_tow_decimals: int = 0,
    selected_svids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Compute NAV-SAT svUsed matched to synchronized RAWX epochs."""
    matched = _navsat_epoch_merge(
        epochs,
        navsat,
        epoch_tow_column=epoch_tow_column,
        navsat_tow_decimals=navsat_tow_decimals,
    )
    matched = matched.dropna(subset=["svid", "svUsed"]).copy()
    if matched.empty:
        return pd.DataFrame(index=epochs["epoch_index"].astype(int))

    matched["svUsed"] = pd.to_numeric(matched["svUsed"], errors="coerce")
    pivot = matched.pivot_table(
        index="epoch_index",
        columns="svid",
        values="svUsed",
        aggfunc="max",
    )
    pivot = pivot.reindex(epochs["epoch_index"].astype(int))
    return sort_svid_columns(pivot, selected_svids)


def compute_navsat_sv_used(
    navsat: pd.DataFrame,
    selected_svids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Compute receiver-native NAV-SAT svUsed over every NAV-SAT epoch."""
    if navsat.empty:
        return pd.DataFrame()

    data = navsat.dropna(subset=["rcvTow", "svid", "svUsed"]).copy()
    if data.empty:
        return pd.DataFrame()

    data["svid"] = data["svid"].astype(int)
    data["svUsed"] = pd.to_numeric(data["svUsed"], errors="coerce")
    data["navsat_epoch_index"] = pd.factorize(data["rcvTow"], sort=True)[0]
    pivot = data.pivot_table(
        index="navsat_epoch_index",
        columns="svid",
        values="svUsed",
        aggfunc="max",
    )
    return sort_svid_columns(pivot, selected_svids)


def summarize_sv_used(
    sv_used_by_receiver: dict[str, pd.DataFrame],
    svids: Sequence[int],
) -> pd.DataFrame:
    """Summarize whether selected satellites were used in receiver navigation."""
    selected_svids = _normalise_svids(svids) or ()
    rows: list[dict[str, object]] = []
    for receiver, frame in sv_used_by_receiver.items():
        for svid in selected_svids:
            series = frame.get(int(svid), pd.Series(index=frame.index, dtype=float))
            observed = series.notna()
            used = series.fillna(0).astype(float).gt(0)
            used_epochs = series.index[used].to_list()
            rows.append(
                {
                    "receiver": receiver,
                    "svid": int(svid),
                    "total_epoch_count": int(len(series)),
                    "observed_epoch_count": int(observed.sum()),
                    "used_epoch_count": int((observed & used).sum()),
                    "pct_observed_epochs_used": float(
                        used.loc[observed].mean() * 100.0
                    )
                    if observed.any()
                    else float("nan"),
                    "first_used_epoch_index": int(used_epochs[0])
                    if used_epochs
                    else None,
                    "last_used_epoch_index": int(used_epochs[-1])
                    if used_epochs
                    else None,
                }
            )
    return pd.DataFrame(rows)


SV_STATUS_NOT_TRACKED, SV_STATUS_TRACKED, SV_STATUS_USED = -1.0, 0.0, 1.0


def navsat_sv_status_by_second(
    navsat: pd.DataFrame, *, max_forward_jump_s: float = 3600.0
) -> pd.DataFrame:
    """NAV-SAT status of every PRN per whole GPS second.

    Values: -1 not tracked (not listed, or C/N0 = 0), 0 tracked but not used in the
    navigation solution, 1 used (``svUsed``). Seconds without any NAV-SAT message stay
    NaN, so dropouts remain visible. Rows logged before the receiver knew GPS time
    (after a cold start ``rcvTow`` restarts from a few seconds) are dropped: in file
    order, rows below the running maximum TOW, and rows before a forward TOW jump of
    more than ``max_forward_jump_s``.
    """
    if navsat.empty:
        return pd.DataFrame()
    data = navsat.dropna(subset=["rcvTow", "svid"]).copy()
    data = data.loc[data["rcvTow"].ge(data["rcvTow"].cummax())]
    jumps = np.flatnonzero(data["rcvTow"].diff().gt(max_forward_jump_s).to_numpy())
    if jumps.size:
        data = data.iloc[jumps[-1]:]
    if data.empty:
        return pd.DataFrame()
    data["tow_s"] = data["rcvTow"].round().astype(int)
    data["svid"] = data["svid"].astype(int)
    used = pd.to_numeric(data["svUsed"], errors="coerce").gt(0)
    tracked = pd.to_numeric(data["cno"], errors="coerce").gt(0)
    data["status"] = np.select(
        [used, tracked], [SV_STATUS_USED, SV_STATUS_TRACKED], SV_STATUS_NOT_TRACKED
    )
    status = data.pivot_table(index="tow_s", columns="svid", values="status", aggfunc="max")
    status = status.fillna(SV_STATUS_NOT_TRACKED)
    status = status.loc[:, status.gt(SV_STATUS_NOT_TRACKED).any()]
    seconds = pd.RangeIndex(int(status.index.min()), int(status.index.max()) + 1, name="tow_s")
    return status.reindex(seconds).sort_index(axis=1)


def summarize_sv_used_by_window(
    status_by_receiver: Mapping[str, pd.DataFrame],
    windows: pd.DataFrame,
    spoof_svids: Sequence[int],
) -> pd.DataFrame:
    """Per window, receiver and PRN: seconds tracked and used in the navigation solution.

    ``windows`` needs ``window``, ``start_tow`` and ``end_tow``. Spoof PRNs are always
    listed; authentic PRNs only when tracked inside the window. ``pct_used`` is over
    the window's seconds, so NAV-SAT dropouts count as not used.
    """
    spoof = _normalise_svids(spoof_svids) or ()
    rows: list[dict[str, object]] = []
    for window in windows.itertuples(index=False):
        first, last = round(float(window.start_tow)), round(float(window.end_tow))
        for receiver, status in status_by_receiver.items():
            span = status.loc[(status.index >= first) & (status.index <= last)]
            tracked_svids = [int(svid) for svid in span.columns if span[svid].ge(0).any()]
            for svid in list(spoof) + sorted(set(tracked_svids) - set(spoof)):
                series = span[svid] if svid in span.columns else pd.Series(dtype=float)
                seconds = last - first + 1
                used = int(series.eq(SV_STATUS_USED).sum())
                rows.append({
                    "window": int(window.window),
                    "receiver": receiver,
                    "svid": svid,
                    "role": "spoof" if svid in spoof else "authentic",
                    "window_seconds": seconds,
                    "navsat_seconds": int(span.notna().any(axis=1).sum()),
                    "tracked_seconds": int(series.ge(SV_STATUS_TRACKED).sum()),
                    "used_seconds": used,
                    "pct_used": 100.0 * used / seconds,
                })
    return pd.DataFrame(rows)


def _branch_step_direction_cosine(candidates: pd.DataFrame) -> float:
    """Direction-cosine step between adjacent Eq. 21 branches, i.e. ``1 / (D / lambda)``."""
    data = candidates.dropna(subset=["candidate_cos_alpha"])
    if data.empty:
        return 1.0
    ordered = data.sort_values(["epoch_index", "svid", "branch"])
    grouped = ordered.groupby(["epoch_index", "svid"], sort=False)
    step = grouped["candidate_cos_alpha"].diff() / grouped["branch"].diff()
    step = step.replace([np.inf, -np.inf], np.nan).dropna()
    return float(step.median()) if not step.empty else 1.0


def _format_branches(branches: Sequence[int]) -> str:
    return ",".join(str(int(branch)) for branch in branches)


def select_joint_source_aoa(
    candidates: pd.DataFrame,
    svids: Sequence[int],
    *,
    metric: str = "direction_cosine",
    prior_aoa_deg: float | None = None,
    fixed_m: int | None = None,
    prior_weight: float = 1.0,
    tie_tolerance: float | None = None,
    spread_weight: float = 4.0,
    smooth_weight: float = 1.0,
    max_states_per_epoch: int = 50_000,
    return_branches: bool = False,
    return_ambiguity: bool = False,
):
    """Select Eq. 21 branches jointly for satellites expected to share one RF source.

    Every epoch has one state per branch combination of the group. The DP minimises
    ``spread_weight * (max - min over the group) + smooth_weight * mean |step from the
    previous epoch|``, both measured in ``metric`` units (see :func:`select_aoa_dp`).

    With a one-wavelength baseline the fractional DD fixes ``cos(alpha)`` only modulo one,
    so shifting every satellite's branch by the same integer yields a second combination
    with identical spread and step costs: a static capture cannot tell the two apart. The
    function therefore evaluates every uniform shift of the DP path. An alternative counts
    as tied when its direction cosines leave ``[-1, 1]`` by at most ``tie_tolerance`` on
    average over the window (``None``: three times the median per-epoch group spread plus
    step in direction cosine, at least 0.01). The tie is broken in this order and recorded
    in the ambiguity table as ``chosen_by``:

    - ``"explicit_m"``: a declared ``fixed_m`` selects a fully valid uniform shift of the
      joint path with that branch at every epoch and SVID. It does not resolve the data's
      ambiguity. Invalid fixed branches raise instead of clipping or dropping observations.
    - ``"data"``: no tied alternative, the DP path stands;
    - ``"prior"``: ``prior_aoa_deg`` picks, among the DP path and the tied alternatives, the
      combination minimising ``mean excess + prior_weight * |cos(median AoA) - cos(prior)|``;
    - ``"unresolved"``: no prior. With ``metric="direction_cosine"`` the fully valid tied
      combination with the smallest mean ``|m|`` is reported; with ``"aoa_deg"`` the DP path
      (the legacy choice, nearer 90 deg). Report both branches side by side in that case.

    Returns the selected AoA (epochs x svids). ``return_branches`` adds the selected branch
    ``m`` per epoch and satellite; ``return_ambiguity`` adds a per-epoch table with the DP
    cost, the cost and out-of-range excess of the closest alternative combination, the
    branch shifts (relative to the selection), the tied shifts and ``chosen_by``.
    """
    _validate_branch_metric(metric)
    required = {
        "epoch_index",
        "svid",
        "branch",
        "candidate_aoa_deg",
        "candidate_cos_alpha",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates is missing required columns: {sorted(missing)}")

    selected_svids = _normalise_svids(svids) or ()
    if not selected_svids:
        raise ValueError("At least one SVID is required.")
    if prior_weight < 0:
        raise ValueError("prior_weight must be non-negative.")
    if tie_tolerance is not None and tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative.")
    if fixed_m is not None:
        if isinstance(fixed_m, bool) or not isinstance(fixed_m, (int, np.integer)):
            raise ValueError("fixed_m must be an integer or None.")
        if prior_aoa_deg is not None:
            raise ValueError("Declare either fixed_m or prior_aoa_deg, not both.")

    branch_values = sorted(int(branch) for branch in candidates["branch"].unique())
    if len(branch_values) ** len(selected_svids) > max_states_per_epoch:
        raise ValueError(
            "Too many branch combinations for joint AoA selection; reduce svids "
            "or increase max_states_per_epoch."
        )

    group = candidates.loc[candidates["svid"].isin(selected_svids)]
    unit = _branch_step_direction_cosine(group)
    per_epoch: dict[int, dict[int, dict[int, tuple[float, float]]]] = {}
    for row in group.loc[group["candidate_aoa_deg"].notna()].itertuples():
        per_epoch.setdefault(int(row.epoch_index), {}).setdefault(int(row.svid), {})[
            int(row.branch)
        ] = (float(row.candidate_aoa_deg), float(row.candidate_cos_alpha))
    use_cosine = metric == "direction_cosine"

    states_by_epoch: list[list[dict[str, object]]] = []
    valid_epochs: list[int] = []
    for epoch in sorted(per_epoch):
        per_sat = per_epoch[epoch]
        if any(svid not in per_sat for svid in selected_svids):
            continue
        epoch_states: list[dict[str, object]] = []
        for branch_combo in product(branch_values, repeat=len(selected_svids)):
            try:
                pairs = [
                    per_sat[svid][branch]
                    for svid, branch in zip(selected_svids, branch_combo)
                ]
            except KeyError:
                continue
            aoa = np.asarray([pair[0] for pair in pairs], dtype=float)
            cos_alpha = np.asarray([pair[1] for pair in pairs], dtype=float)
            score = cos_alpha if use_cosine else aoa
            epoch_states.append(
                {
                    "branches": tuple(int(branch) for branch in branch_combo),
                    "aoa": aoa,
                    "cos": cos_alpha,
                    "score": score,
                    "spread": float(score.max() - score.min()),
                }
            )
        if epoch_states:
            states_by_epoch.append(epoch_states)
            valid_epochs.append(epoch)

    selected = pd.DataFrame(index=valid_epochs, columns=selected_svids, dtype=float)
    branches = pd.DataFrame(index=valid_epochs, columns=selected_svids, dtype=float)
    selected.index.name = "epoch_index"
    branches.index.name = "epoch_index"
    ambiguity_columns = [
        "epoch_index",
        "dp_branches",
        "dp_cost",
        "selected_branches",
        "selected_cost",
        "alternative_shift",
        "alternative_branches",
        "alternative_cost",
        "alternative_excess",
        "alternative_mean_excess",
        "tied_shifts",
        "ambiguous",
        "chosen_by",
        "tie_tolerance",
        "metric",
        "prior_aoa_deg",
        "fixed_m",
    ]
    if not states_by_epoch:
        ambiguity = pd.DataFrame(columns=ambiguity_columns)
        outputs: list[pd.DataFrame] = [selected]
        if return_branches:
            outputs.append(branches)
        if return_ambiguity:
            outputs.append(ambiguity)
        return tuple(outputs) if len(outputs) > 1 else selected

    costs: list[np.ndarray] = [
        np.asarray(
            [spread_weight * float(state["spread"]) for state in states_by_epoch[0]],
            dtype=float,
        )
    ]
    backs: list[np.ndarray] = [np.full(len(states_by_epoch[0]), -1, dtype=int)]
    for epoch_position in range(1, len(states_by_epoch)):
        previous_states = states_by_epoch[epoch_position - 1]
        current_states = states_by_epoch[epoch_position]
        epoch_cost = np.full(len(current_states), np.inf, dtype=float)
        epoch_back = np.full(len(current_states), -1, dtype=int)
        for current_index, current_state in enumerate(current_states):
            current_values = current_state["score"]
            transition = np.asarray(
                [
                    np.mean(np.abs(current_values - previous_state["score"]))
                    for previous_state in previous_states
                ],
                dtype=float,
            )
            total = (
                costs[epoch_position - 1]
                + smooth_weight * transition
                + spread_weight * float(current_state["spread"])
            )
            best_previous = int(np.nanargmin(total))
            epoch_cost[current_index] = total[best_previous]
            epoch_back[current_index] = best_previous
        costs.append(epoch_cost)
        backs.append(epoch_back)

    path = [int(np.nanargmin(costs[-1]))]
    for epoch_position in range(len(states_by_epoch) - 1, 0, -1):
        path.append(int(backs[epoch_position][path[-1]]))
    path = list(reversed(path))

    dp_states = [states_by_epoch[position][index] for position, index in enumerate(path)]
    dp_branches = np.asarray([state["branches"] for state in dp_states], dtype=int)
    dp_aoa = np.asarray([state["aoa"] for state in dp_states], dtype=float)
    dp_cos = np.asarray([state["cos"] for state in dp_states], dtype=float)

    def per_epoch_cost(score: np.ndarray) -> np.ndarray:
        spread = score.max(axis=1) - score.min(axis=1)
        step = np.zeros(len(score), dtype=float)
        if len(score) > 1:
            step[1:] = np.abs(np.diff(score, axis=0)).mean(axis=1)
        return spread_weight * spread + smooth_weight * step

    def evaluate_shift(shift: int) -> dict[str, object]:
        raw_cos = dp_cos + shift * unit
        excess = np.clip(np.abs(raw_cos) - 1.0, 0.0, None).max(axis=1)
        clipped_cos = np.clip(raw_cos, -1.0, 1.0)
        clipped_aoa = np.degrees(np.arccos(clipped_cos))
        score = clipped_cos if use_cosine else clipped_aoa
        return {
            "shift": int(shift),
            "branches": dp_branches + int(shift),
            "excess": excess,
            "mean_excess": float(excess.mean()),
            "fully_valid": bool(excess.max() == 0.0),
            "cost": per_epoch_cost(score),
            "median_aoa_deg": float(np.median(clipped_aoa)),
            "mean_abs_branch": float(np.abs(dp_branches + int(shift)).mean()),
        }

    cos_spread = dp_cos.max(axis=1) - dp_cos.min(axis=1)
    cos_step = np.zeros(len(dp_cos), dtype=float)
    if len(dp_cos) > 1:
        cos_step[1:] = np.abs(np.diff(dp_cos, axis=0)).mean(axis=1)
    if tie_tolerance is None:
        tolerance = max(3.0 * float(np.median(cos_spread + cos_step)), 0.01)
    else:
        tolerance = float(tie_tolerance)

    allowed = set(branch_values)
    shifts = [
        shift
        for shift in range(
            min(branch_values) - int(dp_branches.max()),
            max(branch_values) - int(dp_branches.min()) + 1,
        )
        if shift != 0 and set(np.unique(dp_branches + shift).tolist()) <= allowed
    ]
    evaluations = {0: evaluate_shift(0)}
    for shift in shifts:
        evaluations[shift] = evaluate_shift(shift)
    tied = [shift for shift in shifts if evaluations[shift]["mean_excess"] <= tolerance]
    ambiguous = bool(tied)
    options = [0, *tied]

    if fixed_m is not None:
        matching = [
            shift for shift, evaluation in evaluations.items()
            if evaluation["fully_valid"]
            and np.all(evaluation["branches"] == fixed_m)
        ]
        if not matching:
            raise ValueError(
                f"fixed_m={fixed_m} is not a fully valid uniform branch of the joint-source "
                "path at every epoch and satellite. Inspect the branch candidates."
            )
        selected_shift = matching[0]
        chosen_by = "explicit_m"
    elif not ambiguous:
        selected_shift = 0
        chosen_by = "data"
    elif prior_aoa_deg is not None:
        prior_cos = float(np.cos(np.deg2rad(float(prior_aoa_deg))))
        selected_shift = min(
            options,
            key=lambda shift: (
                float(evaluations[shift]["mean_excess"])
                + prior_weight
                * abs(
                    float(np.cos(np.deg2rad(evaluations[shift]["median_aoa_deg"])))
                    - prior_cos
                ),
                abs(shift),
            ),
        )
        chosen_by = "prior"
    else:
        chosen_by = "unresolved"
        if use_cosine:
            selectable = [
                shift for shift in options if evaluations[shift]["fully_valid"]
            ] or [0]
            selected_shift = min(
                selectable,
                key=lambda shift: (
                    float(evaluations[shift]["mean_abs_branch"]),
                    abs(shift),
                ),
            )
        else:
            selected_shift = 0

    others = [shift for shift in options if shift != selected_shift]
    if others:
        alternative = min(
            others,
            key=lambda shift: (
                float(evaluations[shift]["mean_excess"]),
                abs(shift - selected_shift),
            ),
        )
    elif shifts:
        alternative = min(
            shifts,
            key=lambda shift: (float(evaluations[shift]["mean_excess"]), abs(shift)),
        )
    else:
        alternative = None

    selected_branches = dp_branches + selected_shift
    for position, epoch in enumerate(valid_epochs):
        for column, svid in enumerate(selected_svids):
            branch = int(selected_branches[position, column])
            pair = per_epoch[epoch].get(svid, {}).get(branch)
            branches.loc[epoch, svid] = branch
            selected.loc[epoch, svid] = np.nan if pair is None else pair[0]

    selected_evaluation = evaluations[selected_shift]
    alternative_evaluation = None if alternative is None else evaluations[alternative]
    tied_relative = [shift - selected_shift for shift in options if shift != selected_shift]
    ambiguity = pd.DataFrame(
        {
            "epoch_index": valid_epochs,
            "dp_branches": [_format_branches(row) for row in dp_branches],
            "dp_cost": evaluations[0]["cost"],
            "selected_branches": [_format_branches(row) for row in selected_branches],
            "selected_cost": selected_evaluation["cost"],
            "alternative_shift": (
                np.nan if alternative is None else int(alternative - selected_shift)
            ),
            "alternative_branches": (
                ""
                if alternative_evaluation is None
                else [
                    _format_branches(row) for row in alternative_evaluation["branches"]
                ]
            ),
            "alternative_cost": (
                np.nan if alternative_evaluation is None else alternative_evaluation["cost"]
            ),
            "alternative_excess": (
                np.nan
                if alternative_evaluation is None
                else alternative_evaluation["excess"]
            ),
            "alternative_mean_excess": (
                np.nan
                if alternative_evaluation is None
                else alternative_evaluation["mean_excess"]
            ),
            "tied_shifts": ",".join(str(int(shift)) for shift in tied_relative),
            "ambiguous": ambiguous,
            "chosen_by": chosen_by,
            "tie_tolerance": tolerance,
            "metric": metric,
            "prior_aoa_deg": np.nan if prior_aoa_deg is None else float(prior_aoa_deg),
            "fixed_m": np.nan if fixed_m is None else int(fixed_m),
        },
        columns=ambiguity_columns,
    )

    outputs = [selected]
    if return_branches:
        outputs.append(branches)
    if return_ambiguity:
        outputs.append(ambiguity)
    return tuple(outputs) if len(outputs) > 1 else selected


def select_joint_source_aoa_segments(
    candidates: pd.DataFrame,
    svids: Sequence[int],
    *,
    min_valid_spoof_prns: int | None = None,
    **selection_options,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select available PRNs jointly, restarting at gaps or membership changes.

    Each block calls the existing complete-group selector independently. Missing
    PRNs and epochs stay NaN; no smoothness constraint crosses a reacquisition.
    The ambiguity table records each block's PRNs and branch decision.
    """
    svids = _normalise_svids(svids) or ()
    if not svids:
        raise ValueError("At least one SVID is required.")
    minimum = _minimum_valid_spoof_prns(min_valid_spoof_prns, len(svids))
    epochs = sorted(int(epoch) for epoch in candidates["epoch_index"].unique())
    selected = pd.DataFrame(index=epochs, columns=svids, dtype=float)
    branches = selected.copy()
    selected.index.name = branches.index.name = "epoch_index"
    valid = candidates.loc[
        candidates["svid"].isin(svids) & candidates["candidate_aoa_deg"].notna()
    ]
    available = valid.groupby("epoch_index")["svid"].agg(
        lambda values: tuple(svid for svid in svids if svid in set(values))
    )
    blocks: list[tuple[tuple[int, ...], list[int]]] = []
    previous: int | None = None
    for epoch, members in available.items():
        epoch = int(epoch)
        if len(members) < minimum:
            previous = None
            continue
        if blocks and previous is not None and epoch == previous + 1 and members == blocks[-1][0]:
            blocks[-1][1].append(epoch)
        else:
            blocks.append((members, [epoch]))
        previous = epoch
    ambiguity_parts = []
    for segment_id, (members, block_epochs) in enumerate(blocks):
        block = candidates.loc[candidates["epoch_index"].isin(block_epochs)]
        aoa, branch, ambiguity = select_joint_source_aoa(
            block, members, return_branches=True, return_ambiguity=True,
            **selection_options,
        )
        selected.loc[aoa.index, list(members)] = aoa
        branches.loc[branch.index, list(members)] = branch
        ambiguity["segment_id"] = segment_id
        ambiguity["active_svids"] = ",".join(map(str, members))
        ambiguity_parts.append(ambiguity)
    ambiguity = pd.concat(ambiguity_parts, ignore_index=True) if ambiguity_parts else pd.DataFrame()
    return selected, branches, ambiguity


def summarize_branch_candidates(
    candidates: pd.DataFrame,
    svids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Per satellite and Eq. 21 branch ``m``: epoch count, median/mean/std AoA, median cos.

    Lists every physically valid branch, so the one-wavelength mirror is visible next to
    whatever :func:`select_joint_source_aoa` or :func:`select_aoa_dp` picked.
    """
    required = {"epoch_index", "svid", "branch", "candidate_aoa_deg", "candidate_cos_alpha"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates is missing required columns: {sorted(missing)}")
    columns = [
        "svid",
        "branch",
        "epoch_count",
        "median_aoa_deg",
        "mean_aoa_deg",
        "std_aoa_deg",
        "min_aoa_deg",
        "max_aoa_deg",
        "median_cos_alpha",
    ]
    valid = candidates.loc[candidates["candidate_aoa_deg"].notna()]
    selected = _normalise_svids(svids)
    if selected is not None:
        valid = valid.loc[valid["svid"].isin(selected)]
    if valid.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        valid.groupby(["svid", "branch"])
        .agg(
            epoch_count=("epoch_index", "nunique"),
            median_aoa_deg=("candidate_aoa_deg", "median"),
            mean_aoa_deg=("candidate_aoa_deg", "mean"),
            std_aoa_deg=("candidate_aoa_deg", "std"),
            min_aoa_deg=("candidate_aoa_deg", "min"),
            max_aoa_deg=("candidate_aoa_deg", "max"),
            median_cos_alpha=("candidate_cos_alpha", "median"),
        )
        .reset_index()
    )
    summary["svid"] = summary["svid"].astype(int)
    summary["branch"] = summary["branch"].astype(int)
    return summary.sort_values(["svid", "branch"]).reset_index(drop=True)[columns]


def summarize_spoof_branch_detection(
    branch_candidates: pd.DataFrame,
    ephemeris_aoa_deg: pd.DataFrame,
    *,
    selected_branch_by_svid: Mapping[int, int],
    alternative_branch_by_svid: Mapping[int, int] | None = None,
    chosen_by: str = "unresolved",
    threshold_deg: float = 10.0,
    min_branch_epoch_fraction: float = 0.5,
) -> pd.DataFrame:
    """Per spoof PRN: ephemeris-based AoA against the DD-method AoA of both Eq. 21 branches.

    ``spoof_flag_selected`` is the paper's threshold test on the selected branch.
    ``spoof_flag_all_branches`` is true only when every valid branch (with at least
    ``min_branch_epoch_fraction`` of the PRN's best branch epoch count) disagrees with the
    ephemeris by more than ``threshold_deg``, i.e. the verdict does not depend on how the
    one-wavelength ambiguity was resolved. Both flags are ``<NA>`` when the PRN has no
    ephemeris-based AoA inside the window.
    """
    rows: list[dict[str, object]] = []
    alternatives = dict(alternative_branch_by_svid or {})
    ephemeris_columns = {int(column): column for column in ephemeris_aoa_deg.columns}
    for svid, selected_branch in sorted(
        (int(svid), int(branch)) for svid, branch in selected_branch_by_svid.items()
    ):
        per_branch = branch_candidates.loc[branch_candidates["svid"].eq(svid)].set_index(
            "branch"
        )
        ephemeris = pd.Series(dtype=float)
        if svid in ephemeris_columns:
            ephemeris = ephemeris_aoa_deg[ephemeris_columns[svid]].dropna()
        ephemeris_median = float(ephemeris.median()) if not ephemeris.empty else np.nan

        def branch_stats(branch: int | None) -> tuple[float, float, int]:
            if branch is None or int(branch) not in per_branch.index:
                return np.nan, np.nan, 0
            entry = per_branch.loc[int(branch)]
            return (
                float(entry["median_aoa_deg"]),
                float(entry["std_aoa_deg"]),
                int(entry["epoch_count"]),
            )

        alternative_branch = alternatives.get(svid)
        selected_aoa, selected_std, selected_epochs = branch_stats(selected_branch)
        alternative_aoa, alternative_std, alternative_epochs = branch_stats(
            None if alternative_branch is None else int(alternative_branch)
        )
        abs_diff_selected = abs(selected_aoa - ephemeris_median)
        abs_diff_alternative = abs(alternative_aoa - ephemeris_median)

        considered: list[str] = []
        all_disagree = not per_branch.empty and np.isfinite(ephemeris_median)
        if not per_branch.empty:
            minimum_epochs = min_branch_epoch_fraction * float(per_branch["epoch_count"].max())
            for branch, entry in per_branch.sort_index().iterrows():
                if float(entry["epoch_count"]) < minimum_epochs:
                    continue
                difference = abs(float(entry["median_aoa_deg"]) - ephemeris_median)
                considered.append(f"m={int(branch)}:{float(entry['median_aoa_deg']):.1f}")
                if not (difference > threshold_deg):
                    all_disagree = False
        rows.append(
            {
                "svid": svid,
                "ephemeris_aoa_deg": ephemeris_median,
                "ephemeris_epochs": int(len(ephemeris)),
                "selected_m": selected_branch,
                "selected_aoa_deg": selected_aoa,
                "selected_std_deg": selected_std,
                "selected_epochs": selected_epochs,
                "alternative_m": (
                    np.nan if alternative_branch is None else int(alternative_branch)
                ),
                "alternative_aoa_deg": alternative_aoa,
                "alternative_std_deg": alternative_std,
                "alternative_epochs": alternative_epochs,
                "chosen_by": chosen_by,
                "abs_diff_selected_deg": abs_diff_selected,
                "abs_diff_alternative_deg": abs_diff_alternative,
                "threshold_deg": float(threshold_deg),
                "spoof_flag_selected": (
                    bool(abs_diff_selected > threshold_deg)
                    if np.isfinite(abs_diff_selected)
                    else pd.NA
                ),
                "spoof_flag_all_branches": (
                    bool(all_disagree) if np.isfinite(ephemeris_median) else pd.NA
                ),
                "branches_considered": "|".join(considered),
            }
        )
    detection = pd.DataFrame(rows)
    for column in ("spoof_flag_selected", "spoof_flag_all_branches"):
        if column in detection.columns:
            detection[column] = detection[column].astype("boolean")
    return detection


def summarize_pairwise_single_difference_collapse(
    single_difference_cycles: pd.DataFrame,
    svids: Sequence[int],
) -> pd.DataFrame:
    """Median ``|wrap(SD_i - SD_j)|`` per pair of a spoof group over every common epoch.

    Two signals from one source have equal single differences up to an integer, so this
    branch-free, anchor-free number is the common-source evidence; it needs no reference
    satellite, no reference-angle anchor and no ephemeris.
    """
    columns = {int(column): column for column in single_difference_cycles.columns}
    present = [svid for svid in (_normalise_svids(svids) or ()) if svid in columns]
    rows: list[dict[str, object]] = []
    for left_svid, right_svid in combinations(present, 2):
        difference = wrap_cycles(
            single_difference_cycles[columns[left_svid]]
            - single_difference_cycles[columns[right_svid]]
        ).abs().dropna()
        rows.append(
            {
                "pair": f"{left_svid}-{right_svid}",
                "left_svid": left_svid,
                "right_svid": right_svid,
                "epoch_count": int(len(difference)),
                "median_abs_wrapped_sd_diff_cycles": (
                    float(difference.median()) if len(difference) else np.nan
                ),
                "p90_abs_wrapped_sd_diff_cycles": (
                    float(difference.quantile(0.9)) if len(difference) else np.nan
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "pair",
            "left_svid",
            "right_svid",
            "epoch_count",
            "median_abs_wrapped_sd_diff_cycles",
            "p90_abs_wrapped_sd_diff_cycles",
        ],
    )


def compute_pairwise_aoa_convergence(
    selected_aoa: pd.DataFrame,
    threshold_deg: float = 5.0,
) -> pd.DataFrame:
    """Summarize pairwise AoA convergence for selected spoofing satellites."""
    columns = [int(column) for column in selected_aoa.columns]
    rows: list[dict[str, object]] = []
    for left_index, left_svid in enumerate(columns):
        for right_svid in columns[left_index + 1 :]:
            diff = (
                selected_aoa[left_svid] - selected_aoa[right_svid]
            ).abs().dropna()
            rows.append(
                {
                    "pair": f"{left_svid}-{right_svid}",
                    "left_svid": left_svid,
                    "right_svid": right_svid,
                    "epoch_count": int(len(diff)),
                    "median_abs_aoa_diff_deg": float(diff.median())
                    if not diff.empty
                    else float("nan"),
                    "mean_abs_aoa_diff_deg": float(diff.mean())
                    if not diff.empty
                    else float("nan"),
                    "p90_abs_aoa_diff_deg": float(diff.quantile(0.9))
                    if not diff.empty
                    else float("nan"),
                    "max_abs_aoa_diff_deg": float(diff.max())
                    if not diff.empty
                    else float("nan"),
                    "pct_epochs_aoa_diff_le_threshold": float(
                        (diff <= threshold_deg).mean() * 100.0
                    )
                    if not diff.empty
                    else float("nan"),
                    "threshold_deg": float(threshold_deg),
                }
            )
    return pd.DataFrame(rows)


def summarize_converged_aoa_clusters(
    selected_aoa: pd.DataFrame,
    threshold_deg: float = 5.0,
) -> pd.DataFrame:
    """Find connected AoA clusters and summarize their median angle."""
    columns = [int(column) for column in selected_aoa.columns]
    if not columns:
        return pd.DataFrame()

    pairwise = compute_pairwise_aoa_convergence(selected_aoa, threshold_deg)
    adjacency = {svid: {svid} for svid in columns}
    for row in pairwise.itertuples():
        if (
            pd.notna(row.median_abs_aoa_diff_deg)
            and float(row.median_abs_aoa_diff_deg) <= threshold_deg
        ):
            adjacency[int(row.left_svid)].add(int(row.right_svid))
            adjacency[int(row.right_svid)].add(int(row.left_svid))

    visited: set[int] = set()
    clusters: list[list[int]] = []
    for svid in columns:
        if svid in visited:
            continue
        stack = [svid]
        cluster: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)
            stack.extend(sorted(adjacency[current] - visited))
        clusters.append(sorted(cluster))

    rows: list[dict[str, object]] = []
    for cluster in clusters:
        data = selected_aoa[cluster].dropna(how="all")
        epoch_median = data.median(axis=1, skipna=True)
        spread = data.max(axis=1, skipna=True) - data.min(axis=1, skipna=True)
        rows.append(
            {
                "cluster_svids": ",".join(str(svid) for svid in cluster),
                "cluster_size": int(len(cluster)),
                "epoch_count": int(epoch_median.notna().sum()),
                "median_aoa_deg": float(epoch_median.median(skipna=True)),
                "mean_aoa_deg": float(epoch_median.mean(skipna=True)),
                "std_epoch_median_aoa_deg": float(epoch_median.std(skipna=True)),
                "median_spread_deg": float(spread.median(skipna=True)),
                "p90_spread_deg": float(spread.quantile(0.9)),
                "threshold_deg": float(threshold_deg),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["cluster_size", "epoch_count"], ascending=[False, False]
    )


def reconstruct_multi_reference_aoa(
    single_difference_cycles: pd.DataFrame,
    ephemeris_aoa_deg: pd.DataFrame,
    *,
    reference_svids: Sequence[int] | None = None,
    selected_svids: Sequence[int] | None = None,
    min_target_epochs: int = 10,
    branch_values: Iterable[int] = (-2, -1, 0, 1, 2),
    baseline_over_wavelength: float = 1.0,
    branch_selection_method: str = "dp",
    branch_metric: str = "aoa_deg",
) -> pd.DataFrame:
    """Reconstruct DD AoAs for every eligible DD reference.

    ``branch_selection_method="dp"`` applies temporal DP and backtracking without using target
    ephemeris to select an ambiguity branch. ``"best_m"`` instead selects, for every satellite,
    the constant branch with the smallest median AoA error to that satellite's ephemeris track.
    Best-m uses no authenticity labels, but is an ephemeris-assisted diagnostic rather than an
    independent spoof detector because the same ephemeris is later used in the residual.

    ``branch_metric`` is passed to :func:`select_aoa_dp`. It defaults to the legacy degree
    metric so the DP-versus-best-m comparison in the pairwise classifier notebooks keeps its
    recorded behaviour; the paper flow uses ``"direction_cosine"``.
    """
    if min_target_epochs < 1:
        raise ValueError("min_target_epochs must be at least 1.")
    if branch_selection_method not in {"dp", "best_m"}:
        raise ValueError("branch_selection_method must be 'dp' or 'best_m'.")
    _validate_branch_metric(branch_metric)

    single_difference = sort_svid_columns(single_difference_cycles)
    ephemeris = sort_svid_columns(ephemeris_aoa_deg).reindex(single_difference.index)
    available = sorted(set(single_difference.columns) & set(ephemeris.columns))
    requested_targets = _normalise_svids(selected_svids)
    if requested_targets is not None:
        available = [svid for svid in available if svid in requested_targets]

    eligible = [
        svid
        for svid in available
        if int(single_difference[svid].notna().sum()) >= min_target_epochs
        and int(ephemeris[svid].notna().sum()) >= min_target_epochs
    ]
    if len(eligible) < 3:
        raise ValueError(
            "At least three satellites with sufficient DD and ephemeris observations are "
            "required for pairwise geometry classification."
        )

    requested_references = _normalise_svids(reference_svids)
    if requested_references is None:
        references = eligible
    else:
        references = [svid for svid in requested_references if svid in eligible]
    if not references:
        raise ValueError("No requested DD reference has sufficient observations.")

    rows: list[pd.DataFrame] = []
    for reference_svid in references:
        double_difference = compute_double_difference(
            single_difference.reindex(columns=eligible), reference_svid
        )
        _, fractional_dd = fractional_round_dd(double_difference)
        reference_aoa = ephemeris[reference_svid]
        candidates = generate_aoa_candidates(
            fractional_dd,
            alpha_r_deg=reference_aoa,
            branch_values=branch_values,
            baseline_over_wavelength=baseline_over_wavelength,
        )
        if branch_selection_method == "dp":
            selected_aoa, selected_branch = select_aoa_dp(
                candidates,
                reference_svid=reference_svid,
                reference_angle_deg=reference_aoa,
                return_branches=True,
                metric=branch_metric,
            )
        else:
            selected_aoa, selected_branch_by_svid = select_aoa_by_reference(
                candidates,
                ephemeris.reindex(columns=eligible),
                return_branches=True,
            )
            selected_branch = pd.DataFrame(
                np.nan,
                index=selected_aoa.index,
                columns=selected_aoa.columns,
                dtype=float,
            )
            selected_branch.index.name = "epoch_index"
            for svid, branch in selected_branch_by_svid.items():
                valid_epochs = selected_aoa[int(svid)].notna()
                selected_branch.loc[valid_epochs, int(svid)] = int(branch)

        aoa_long = selected_aoa.reset_index().melt(
            id_vars="epoch_index", var_name="svid", value_name="dd_aoa_deg"
        )
        branch_long = selected_branch.reset_index().melt(
            id_vars="epoch_index", var_name="svid", value_name="selected_branch"
        )
        ephemeris_long = ephemeris.reindex(columns=eligible).reset_index().melt(
            id_vars="epoch_index", var_name="svid", value_name="ephemeris_aoa_deg"
        )
        reference_rows = aoa_long.merge(
            branch_long, on=["epoch_index", "svid"], how="left"
        ).merge(ephemeris_long, on=["epoch_index", "svid"], how="left")
        reference_rows.insert(0, "reference_svid", int(reference_svid))
        reference_rows.insert(1, "branch_selection_method", branch_selection_method)
        reference_rows["svid"] = reference_rows["svid"].astype(int)
        reference_rows["dd_direction_cosine"] = np.cos(
            np.deg2rad(reference_rows["dd_aoa_deg"])
        )
        reference_rows["ephemeris_direction_cosine"] = np.cos(
            np.deg2rad(reference_rows["ephemeris_aoa_deg"])
        )
        reference_rows["signed_geometry_error"] = (
            reference_rows["dd_direction_cosine"]
            - reference_rows["ephemeris_direction_cosine"]
        )
        rows.append(reference_rows)

    return pd.concat(rows, ignore_index=True).sort_values(
        ["reference_svid", "epoch_index", "svid"]
    ).reset_index(drop=True)


def wrap_cycles(values: float | np.ndarray | pd.Series) -> float | np.ndarray | pd.Series:
    """Wrap carrier-cycle values to the half-open interval [-0.5, 0.5)."""
    return (values + 0.5) % 1.0 - 0.5


def compute_ephemeris_predicted_pair_separation_cycles(
    left_aoa_deg: float | np.ndarray | pd.Series,
    right_aoa_deg: float | np.ndarray | pd.Series,
    *,
    baseline_over_wavelength: float,
) -> float | np.ndarray | pd.Series:
    """Project two ephemeris AoAs into observable fractional carrier separation."""
    if baseline_over_wavelength <= 0:
        raise ValueError("baseline_over_wavelength must be positive.")
    left_direction_cosine = np.cos(np.deg2rad(left_aoa_deg))
    right_direction_cosine = np.cos(np.deg2rad(right_aoa_deg))
    predicted_pair_cycles = baseline_over_wavelength * (
        left_direction_cosine - right_direction_cosine
    )
    return abs(wrap_cycles(predicted_pair_cycles))


def compute_wrapped_pairwise_residuals(
    single_difference_cycles: pd.DataFrame,
    ephemeris_aoa_deg: pd.DataFrame,
    *,
    selected_svids: Sequence[int] | None = None,
    min_target_epochs: int = 10,
    baseline_over_wavelength: float = 1.0,
) -> pd.DataFrame:
    """Compute branch-invariant pairwise geometry and common-source residuals.

    Subtracting two receiver single differences gives an inter-satellite double difference.
    Wrapping that measurement-minus-ephemeris difference modulo one cycle removes the unknown
    integer ambiguity exactly, so neither temporal DP nor ephemeris-aligned best-m is needed.
    """
    if min_target_epochs < 1:
        raise ValueError("min_target_epochs must be at least 1.")
    if baseline_over_wavelength <= 0:
        raise ValueError("baseline_over_wavelength must be positive.")

    single_difference = sort_svid_columns(single_difference_cycles)
    ephemeris = sort_svid_columns(ephemeris_aoa_deg).reindex(single_difference.index)
    available = sorted(set(single_difference.columns) & set(ephemeris.columns))
    requested_svids = _normalise_svids(selected_svids)
    if requested_svids is not None:
        available = [svid for svid in available if svid in requested_svids]
    eligible = [
        int(svid)
        for svid in available
        if int(single_difference[svid].notna().sum()) >= min_target_epochs
        and int(ephemeris[svid].notna().sum()) >= min_target_epochs
    ]
    if len(eligible) < 2:
        raise ValueError(
            "At least two satellites with sufficient single-difference and ephemeris "
            "observations are required."
        )

    ephemeris_direction_cosine = np.cos(np.deg2rad(ephemeris.reindex(columns=eligible)))
    rows: list[pd.DataFrame] = []
    for left_svid, right_svid in combinations(eligible, 2):
        pair = pd.DataFrame(
            {
                "measured_pair_cycles": (
                    single_difference[left_svid] - single_difference[right_svid]
                ),
                "expected_pair_cycles": baseline_over_wavelength
                * (
                    ephemeris_direction_cosine[left_svid]
                    - ephemeris_direction_cosine[right_svid]
                ),
            }
        ).dropna()
        if pair.empty:
            continue
        pair["measured_pair_cycles_wrapped"] = wrap_cycles(
            pair["measured_pair_cycles"]
        )
        pair["expected_pair_cycles_wrapped"] = wrap_cycles(
            pair["expected_pair_cycles"]
        )
        pair["signed_wrapped_geometry_residual_cycles"] = wrap_cycles(
            pair["measured_pair_cycles"] - pair["expected_pair_cycles"]
        )
        pair["wrapped_geometry_residual_cycles"] = pair[
            "signed_wrapped_geometry_residual_cycles"
        ].abs()
        pair["common_source_collapse_cycles"] = pair[
            "measured_pair_cycles_wrapped"
        ].abs()
        pair["wrapped_expected_separation_cycles"] = pair[
            "expected_pair_cycles_wrapped"
        ].abs()
        pair["common_source_incompatibility_cycles"] = (
            pair["wrapped_expected_separation_cycles"]
            - pair["common_source_collapse_cycles"]
        )
        pair = pair.reset_index().rename(columns={pair.index.name or "index": "epoch_index"})
        pair.insert(1, "left_svid", int(left_svid))
        pair.insert(2, "right_svid", int(right_svid))
        rows.append(pair)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["left_svid", "right_svid", "epoch_index"]
    ).reset_index(drop=True)


def aggregate_wrapped_pairwise_residuals(
    epoch_residuals: pd.DataFrame,
    *,
    min_epochs_per_pair: int = 10,
) -> pd.DataFrame:
    """Aggregate branch-invariant pairwise residuals over epochs."""
    if min_epochs_per_pair < 1:
        raise ValueError("min_epochs_per_pair must be at least 1.")
    required = {
        "left_svid",
        "right_svid",
        "wrapped_geometry_residual_cycles",
        "common_source_collapse_cycles",
        "wrapped_expected_separation_cycles",
        "common_source_incompatibility_cycles",
    }
    missing = required - set(epoch_residuals.columns)
    if missing:
        raise ValueError(f"epoch_residuals is missing required columns: {sorted(missing)}")
    if epoch_residuals.empty:
        return pd.DataFrame()

    aggregated = (
        epoch_residuals.groupby(["left_svid", "right_svid"], as_index=False)
        .agg(
            median_wrapped_geometry_residual=(
                "wrapped_geometry_residual_cycles",
                "median",
            ),
            mean_wrapped_geometry_residual=(
                "wrapped_geometry_residual_cycles",
                "mean",
            ),
            p90_wrapped_geometry_residual=(
                "wrapped_geometry_residual_cycles",
                lambda values: values.quantile(0.9),
            ),
            median_common_source_collapse=(
                "common_source_collapse_cycles",
                "median",
            ),
            mean_common_source_collapse=(
                "common_source_collapse_cycles",
                "mean",
            ),
            p90_common_source_collapse=(
                "common_source_collapse_cycles",
                lambda values: values.quantile(0.9),
            ),
            median_expected_separation=(
                "wrapped_expected_separation_cycles",
                "median",
            ),
            median_common_source_incompatibility=(
                "common_source_incompatibility_cycles",
                "median",
            ),
            epoch_count=("wrapped_geometry_residual_cycles", "count"),
        )
        .loc[lambda frame: frame["epoch_count"].ge(min_epochs_per_pair)]
        .reset_index(drop=True)
    )
    aggregated["pair"] = aggregated.apply(
        lambda row: f"PRN {int(row.left_svid)}–PRN {int(row.right_svid)}", axis=1
    )
    return aggregated


def find_pairwise_cliques(
    aggregated_pairs: pd.DataFrame,
    *,
    edge_value_column: str,
    edge_threshold: float,
    min_clique_size: int = 3,
) -> pd.DataFrame:
    """Find maximal complete subgraphs whose pair values do not exceed a threshold."""
    if edge_threshold < 0:
        raise ValueError("edge_threshold must be non-negative.")
    if min_clique_size < 2:
        raise ValueError("min_clique_size must be at least 2.")
    required = {"left_svid", "right_svid", edge_value_column}
    missing = required - set(aggregated_pairs.columns)
    if missing:
        raise ValueError(f"aggregated_pairs is missing required columns: {sorted(missing)}")

    edge_values = {
        tuple(sorted((int(row.left_svid), int(row.right_svid)))): float(
            getattr(row, edge_value_column)
        )
        for row in aggregated_pairs.itertuples()
        if np.isfinite(float(getattr(row, edge_value_column)))
    }
    svids = sorted({svid for edge in edge_values for svid in edge})
    complete_subsets: list[tuple[int, ...]] = []
    for size in range(min_clique_size, len(svids) + 1):
        for subset in combinations(svids, size):
            if all(
                edge_values.get((left_svid, right_svid), np.inf) <= edge_threshold
                for left_svid, right_svid in combinations(subset, 2)
            ):
                complete_subsets.append(tuple(int(svid) for svid in subset))

    maximal_subsets = [
        subset
        for subset in complete_subsets
        if not any(set(subset) < set(other) for other in complete_subsets)
    ]
    rows: list[dict[str, object]] = []
    for subset in maximal_subsets:
        values = [
            edge_values[(left_svid, right_svid)]
            for left_svid, right_svid in combinations(subset, 2)
        ]
        rows.append(
            {
                "clique_svids": subset,
                "clique_size": len(subset),
                "edge_count": len(values),
                "median_edge_value": float(np.median(values)),
                "max_edge_value": float(np.max(values)),
                "edge_threshold": float(edge_threshold),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "clique_svids",
                "clique_size",
                "edge_count",
                "median_edge_value",
                "max_edge_value",
                "edge_threshold",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["clique_size", "median_edge_value"], ascending=[False, True]
    ).reset_index(drop=True)


def compute_pairwise_geometry_residuals(
    geometry_tracks: pd.DataFrame,
    *,
    residual_domain: str = "direction_cosine",
) -> pd.DataFrame:
    """Compute pairwise geometry residuals per epoch and DD reference.

    ``residual_domain="direction_cosine"`` reproduces journal Eq. 32. ``"degrees"``
    applies the same difference-of-relative-geometry construction directly to the reconstructed
    and ephemeris AoAs. These AoAs are arccos outputs in [0, 180] degrees, so circular wrapping
    is not required.
    """
    if residual_domain == "direction_cosine":
        measured_column = "dd_direction_cosine"
        expected_column = "ephemeris_direction_cosine"
        measured_output = "measured_relative_direction_cosine"
        expected_output = "expected_relative_direction_cosine"
        residual_unit = "direction_cosine"
    elif residual_domain == "degrees":
        measured_column = "dd_aoa_deg"
        expected_column = "ephemeris_aoa_deg"
        measured_output = "measured_relative_aoa_deg"
        expected_output = "expected_relative_aoa_deg"
        residual_unit = "deg"
    else:
        raise ValueError(
            "residual_domain must be 'direction_cosine' or 'degrees'."
        )

    required = {
        "reference_svid",
        "epoch_index",
        "svid",
        measured_column,
        expected_column,
    }
    missing = required - set(geometry_tracks.columns)
    if missing:
        raise ValueError(f"geometry_tracks is missing required columns: {sorted(missing)}")

    all_svids = sorted(int(svid) for svid in geometry_tracks["svid"].unique())
    rows: list[pd.DataFrame] = []
    for reference_svid, reference_rows in geometry_tracks.groupby(
        "reference_svid", sort=True
    ):
        reference_svid = int(reference_svid)
        targets = [svid for svid in all_svids if svid != reference_svid]
        for left_svid, right_svid in combinations(targets, 2):
            left = reference_rows.loc[
                reference_rows["svid"].eq(left_svid),
                ["epoch_index", measured_column, expected_column],
            ].set_index("epoch_index")
            right = reference_rows.loc[
                reference_rows["svid"].eq(right_svid),
                ["epoch_index", measured_column, expected_column],
            ].set_index("epoch_index")
            paired = left.join(right, how="inner", lsuffix="_left", rsuffix="_right")
            paired = paired.dropna()
            if paired.empty:
                continue
            measured_relative = (
                paired[f"{measured_column}_left"]
                - paired[f"{measured_column}_right"]
            )
            expected_relative = (
                paired[f"{expected_column}_left"]
                - paired[f"{expected_column}_right"]
            )
            pair_rows = pd.DataFrame(
                {
                    "reference_svid": reference_svid,
                    "epoch_index": paired.index.astype(int),
                    "left_svid": left_svid,
                    "right_svid": right_svid,
                    "residual_domain": residual_domain,
                    "residual_unit": residual_unit,
                    measured_output: measured_relative,
                    expected_output: expected_relative,
                    "pairwise_residual": (measured_relative - expected_relative).abs(),
                }
            )
            rows.append(pair_rows.reset_index(drop=True))

    columns = [
        "reference_svid",
        "epoch_index",
        "left_svid",
        "right_svid",
        "residual_domain",
        "residual_unit",
        measured_output,
        expected_output,
        "pairwise_residual",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["left_svid", "right_svid", "reference_svid", "epoch_index"]
    ).reset_index(drop=True)


def aggregate_pairwise_geometry_residuals(
    epoch_residuals: pd.DataFrame,
    *,
    min_epochs_per_reference: int = 10,
    min_references_per_pair: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply journal Eqs. 33-34: median over epochs, then valid DD references."""
    if min_epochs_per_reference < 1 or min_references_per_pair < 1:
        raise ValueError("Minimum epoch and reference counts must be at least 1.")
    required = {
        "reference_svid",
        "left_svid",
        "right_svid",
        "pairwise_residual",
    }
    missing = required - set(epoch_residuals.columns)
    if missing:
        raise ValueError(f"epoch_residuals is missing required columns: {sorted(missing)}")
    if epoch_residuals.empty:
        return pd.DataFrame(), pd.DataFrame()

    residual_metadata: dict[str, object] = {}
    for column in ("residual_domain", "residual_unit"):
        if column not in epoch_residuals:
            continue
        values = epoch_residuals[column].dropna().unique()
        if len(values) != 1:
            raise ValueError(f"epoch_residuals must contain exactly one {column} value.")
        residual_metadata[column] = values[0]

    by_reference = (
        epoch_residuals.groupby(
            ["reference_svid", "left_svid", "right_svid"], as_index=False
        )["pairwise_residual"]
        .agg(
            temporal_median_residual="median",
            mean_residual="mean",
            p90_residual=lambda values: values.quantile(0.9),
            epoch_count="count",
        )
        .loc[lambda frame: frame["epoch_count"].ge(min_epochs_per_reference)]
        .reset_index(drop=True)
    )
    if by_reference.empty:
        return by_reference, pd.DataFrame()
    for position, (column, value) in enumerate(residual_metadata.items(), start=3):
        by_reference.insert(position, column, value)

    aggregated = (
        by_reference.groupby(["left_svid", "right_svid"], as_index=False)
        .agg(
            aggregated_pairwise_residual=("temporal_median_residual", "median"),
            min_reference_residual=("temporal_median_residual", "min"),
            max_reference_residual=("temporal_median_residual", "max"),
            reference_count=("reference_svid", "count"),
            epoch_observation_count=("epoch_count", "sum"),
        )
        .loc[lambda frame: frame["reference_count"].ge(min_references_per_pair)]
        .reset_index(drop=True)
    )
    aggregated["pair"] = aggregated.apply(
        lambda row: f"PRN {int(row.left_svid)}–PRN {int(row.right_svid)}", axis=1
    )
    for position, (column, value) in enumerate(residual_metadata.items(), start=2):
        aggregated.insert(position, column, value)
    return by_reference, aggregated


def calibrate_pairwise_threshold(
    authentic_aggregated_residuals: pd.DataFrame,
    *,
    percentile: float = 95.0,
) -> float:
    """Calibrate journal threshold tau_p from an independent authentic dataset."""
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between 0 and 100.")
    column = "aggregated_pairwise_residual"
    if column not in authentic_aggregated_residuals:
        raise ValueError(f"authentic_aggregated_residuals must contain {column!r}.")
    values = authentic_aggregated_residuals[column].dropna().astype(float)
    if values.empty:
        raise ValueError("No authentic pairwise residuals are available for calibration.")
    return float(values.quantile(percentile / 100.0))


def estimate_pairwise_threshold_gmm(
    aggregated_residuals: pd.DataFrame,
    *,
    value_column: str = "aggregated_pairwise_residual",
    random_state: int = 0,
) -> tuple[float, pd.DataFrame]:
    """Estimate an exploratory threshold from a two-component residual mixture.

    This is a label-free diagnostic for datasets without an authentic calibration recording;
    it is not equivalent to the journal's calibrated ``tau_p``. The returned threshold is the
    intersection of the two weighted Gaussian densities between their fitted means.
    """
    from sklearn.mixture import GaussianMixture

    if value_column not in aggregated_residuals:
        raise ValueError(f"aggregated_residuals must contain {value_column!r}.")
    values = (
        aggregated_residuals[value_column]
        .dropna()
        .astype(float)
        .to_numpy()
        .reshape(-1, 1)
    )
    if len(values) < 6:
        raise ValueError("At least six pairwise residuals are required for GMM estimation.")

    one_component = GaussianMixture(
        n_components=1,
        covariance_type="full",
        n_init=20,
        random_state=random_state,
        reg_covar=1e-6,
    ).fit(values)
    two_component = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=20,
        random_state=random_state,
        reg_covar=1e-6,
    ).fit(values)

    order = np.argsort(two_component.means_.ravel())
    means = two_component.means_.ravel()[order]
    standard_deviations = np.sqrt(two_component.covariances_.reshape(-1))[order]
    weights = two_component.weights_.ravel()[order]
    grid = np.linspace(float(means[0]), float(means[1]), 20_001)
    densities = []
    for mean, standard_deviation, weight in zip(means, standard_deviations, weights):
        density = (
            weight
            / (standard_deviation * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((grid - mean) / standard_deviation) ** 2)
        )
        densities.append(density)
    threshold = float(grid[int(np.argmin(np.abs(densities[0] - densities[1])))])

    bic_one = float(one_component.bic(values))
    bic_two = float(two_component.bic(values))
    diagnostics = pd.DataFrame(
        [
            {
                "threshold_mode": "unsupervised_gmm_exploratory",
                "value_column": value_column,
                "pair_count": int(len(values)),
                "estimated_threshold": threshold,
                "low_component_mean": float(means[0]),
                "low_component_std": float(standard_deviations[0]),
                "low_component_weight": float(weights[0]),
                "high_component_mean": float(means[1]),
                "high_component_std": float(standard_deviations[1]),
                "high_component_weight": float(weights[1]),
                "one_component_bic": bic_one,
                "two_component_bic": bic_two,
                "bic_improvement_for_two_components": bic_one - bic_two,
            }
        ]
    )
    return threshold, diagnostics


def _symmetric_pairwise_matrix(
    aggregated_residuals: pd.DataFrame,
    value_column: str,
    *,
    diagonal: float | None = None,
) -> pd.DataFrame:
    svids = sorted(
        set(aggregated_residuals["left_svid"].astype(int))
        | set(aggregated_residuals["right_svid"].astype(int))
    )
    matrix = pd.DataFrame(np.nan, index=svids, columns=svids, dtype=float)
    matrix.index.name = "svid"
    matrix.columns.name = "svid"
    for row in aggregated_residuals.itertuples():
        left_svid = int(row.left_svid)
        right_svid = int(row.right_svid)
        value = float(getattr(row, value_column))
        matrix.loc[left_svid, right_svid] = value
        matrix.loc[right_svid, left_svid] = value
    if diagonal is not None:
        for svid in svids:
            matrix.loc[svid, svid] = float(diagonal)
    return matrix


def classify_pairwise_geometry(
    aggregated_residuals: pd.DataFrame,
    *,
    pairwise_threshold: float,
    score_threshold: float = 0.5,
    min_pair_coverage: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply journal Eqs. 37-39 and return scores, residual matrix, and A_ij."""
    if pairwise_threshold < 0:
        raise ValueError("pairwise_threshold must be non-negative.")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1.")
    if not 0.0 <= min_pair_coverage <= 1.0:
        raise ValueError("min_pair_coverage must be between 0 and 1.")
    required = {"left_svid", "right_svid", "aggregated_pairwise_residual"}
    missing = required - set(aggregated_residuals.columns)
    if missing:
        raise ValueError(
            f"aggregated_residuals is missing required columns: {sorted(missing)}"
        )
    if aggregated_residuals.empty:
        raise ValueError("No aggregated residuals are available for classification.")

    residual_matrix = _symmetric_pairwise_matrix(
        aggregated_residuals, "aggregated_pairwise_residual", diagonal=0.0
    )
    consistency_matrix = residual_matrix.le(float(pairwise_threshold)).astype(float)
    consistency_matrix = consistency_matrix.where(residual_matrix.notna())
    for svid in consistency_matrix.index:
        consistency_matrix.loc[svid, svid] = np.nan

    possible_pairs = max(len(residual_matrix) - 1, 1)
    rows: list[dict[str, object]] = []
    for svid in residual_matrix.index:
        valid = consistency_matrix.loc[svid].dropna()
        valid_pair_count = int(len(valid))
        consistent_pair_count = int(valid.eq(1.0).sum())
        coverage = valid_pair_count / possible_pairs
        consistency_score = consistent_pair_count / possible_pairs
        available_pair_score = (
            consistent_pair_count / valid_pair_count
            if valid_pair_count
            else float("nan")
        )
        if coverage < min_pair_coverage:
            decision = "Undetermined"
        elif consistency_score >= score_threshold:
            decision = "Authentic"
        else:
            decision = "Counterfeit"
        rows.append(
            {
                "svid": int(svid),
                "consistent_pair_count": consistent_pair_count,
                "valid_pair_count": valid_pair_count,
                "possible_pair_count": possible_pairs,
                "pair_coverage": coverage,
                "consistency_score": consistency_score,
                "available_pair_score": available_pair_score,
                "pairwise_threshold": float(pairwise_threshold),
                "score_threshold": float(score_threshold),
                "decision": decision,
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        ["consistency_score", "svid"], ascending=[False, True]
    ).reset_index(drop=True)
    return scores, residual_matrix, consistency_matrix


UBX_KEY_MESSAGES = ("RXM-RAWX", "NAV-PVT", "NAV-SAT", "NAV-SIG", "RXM-SFRBX")


def compute_navsat_geometry_by_synced_epoch(
    epochs: pd.DataFrame,
    navsat: pd.DataFrame,
    *,
    epoch_tow_column: str,
    navsat_tow_decimals: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NAV-SAT elevation and azimuth (deg) per synchronized epoch and SVID."""
    epoch_index = pd.Index(epochs["epoch_index"].astype(int), name="epoch_index")
    if navsat.empty:
        empty = pd.DataFrame(index=epoch_index)
        return empty, empty.copy()
    matched = _navsat_epoch_merge(
        epochs,
        navsat,
        epoch_tow_column=epoch_tow_column,
        navsat_tow_decimals=navsat_tow_decimals,
    )
    matched = matched.dropna(subset=["svid", "elev", "azim"]).copy()
    if matched.empty:
        empty = pd.DataFrame(index=epoch_index)
        return empty, empty.copy()
    matched = _drop_invalid_navsat_geometry(matched)
    matched["svid"] = matched["svid"].astype(int)
    elevation = matched.pivot_table(
        index="epoch_index", columns="svid", values="elev", aggfunc="first"
    ).reindex(epoch_index)
    azimuth = matched.pivot_table(
        index="epoch_index", columns="svid", values="azim", aggfunc="first"
    ).reindex(epoch_index)
    return sort_svid_columns(elevation), sort_svid_columns(azimuth)


@dataclass
class BaselineAzimuthFit:
    """Antenna-baseline azimuth (and D/lambda) fitted from the authentic satellites."""

    authentic_svids: tuple[int, ...]
    baseline_over_wavelength: float
    configured_azimuth_deg: float
    fitted_azimuth_deg: float
    fitted_residual_cycles: float
    configured_residual_cycles: float
    opposite_residual_cycles: float
    azimuth_difference_deg: float
    consistent: bool
    recommendation: str
    per_svid_residual_cycles: pd.Series
    best_per_ratio: pd.DataFrame
    grid: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        """One-row summary for display next to the configured rig parameters."""
        return pd.DataFrame(
            [
                {
                    "authentic_svids": ",".join(str(svid) for svid in self.authentic_svids),
                    "baseline_over_wavelength": self.baseline_over_wavelength,
                    "configured_azimuth_deg": self.configured_azimuth_deg,
                    "fitted_azimuth_deg": self.fitted_azimuth_deg,
                    "azimuth_difference_deg": self.azimuth_difference_deg,
                    "fitted_residual_cycles": self.fitted_residual_cycles,
                    "configured_residual_cycles": self.configured_residual_cycles,
                    "opposite_residual_cycles": self.opposite_residual_cycles,
                    "consistent": self.consistent,
                    "recommendation": self.recommendation,
                }
            ]
        )


def fit_baseline_azimuth(
    single_difference_cycles: pd.DataFrame,
    navsat_elev_deg: pd.DataFrame,
    navsat_azim_deg: pd.DataFrame,
    *,
    exclude_svids: Sequence[int] | None = None,
    min_epochs: int = 50,
    baseline_over_wavelength: float = 1.0,
    ratio_grid: Sequence[float] = (0.9, 0.95, 1.0, 1.05, 1.1),
    azimuth_step_deg: float = 1.0,
    configured_azimuth_deg: float = 0.0,
    azimuth_tolerance_deg: float = 30.0,
) -> BaselineAzimuthFit:
    """Fit the antenna-baseline azimuth from the authentic satellites.

    For authentic PRNs ``i, j`` the wrapped inter-satellite double difference must equal
    ``wrap(k (cos a_i - cos a_j))`` with ``cos a = cos(el) cos(az - az_b)``; this uses no
    reference satellite, no reference-angle anchor and no Eq. 21 branch. A grid search over
    ``az_b`` (and over the D/lambda values in ``ratio_grid``) reports the azimuth that fits
    the single differences *as computed with the configured receiver order*. A fit near
    ``configured_azimuth_deg`` confirms the rig polarity; a fit near ``configured + 180``
    means the receivers are wired the other way round (set ``navsat_azimuth_bias_deg`` to
    the fitted value or swap ``single_difference_order``). Random data give a median
    residual near 0.25 cycles; this rig fits at 0.05-0.09 cycles.
    """
    if min_epochs < 1:
        raise ValueError("min_epochs must be at least 1.")
    if baseline_over_wavelength <= 0:
        raise ValueError("baseline_over_wavelength must be positive.")
    excluded = set(_normalise_svids(exclude_svids) or ())
    single_difference = sort_svid_columns(single_difference_cycles)
    elevation = sort_svid_columns(navsat_elev_deg).reindex(single_difference.index)
    azimuth = sort_svid_columns(navsat_azim_deg).reindex(single_difference.index)
    authentic = [
        int(svid)
        for svid in single_difference.columns
        if int(svid) not in excluded
        and int(svid) in elevation.columns
        and int(svid) in azimuth.columns
        and int(
            (
                single_difference[svid].notna()
                & elevation[int(svid)].notna()
                & azimuth[int(svid)].notna()
            ).sum()
        )
        >= min_epochs
    ]
    if len(authentic) < 3:
        raise ValueError(
            f"Need at least 3 authentic PRNs with >= {min_epochs} epochs of single "
            f"difference and NAV-SAT geometry; found {authentic}."
        )

    measured_sd = single_difference[authentic].to_numpy(dtype=float)
    elevation_rad = np.deg2rad(elevation[authentic].to_numpy(dtype=float))
    azimuth_rad = np.deg2rad(azimuth[authentic].to_numpy(dtype=float))
    pairs = list(combinations(range(len(authentic)), 2))
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    measured = measured_sd[:, left] - measured_sd[:, right]

    def residuals(azimuth_deg: float, ratio: float) -> np.ndarray:
        cos_alpha = np.cos(elevation_rad) * np.cos(azimuth_rad - np.deg2rad(azimuth_deg))
        expected = cos_alpha[:, left] - cos_alpha[:, right]
        return np.abs(wrap_cycles(measured - ratio * expected))

    def median_residual(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else float("nan")

    azimuth_grid = np.arange(0.0, 360.0, float(azimuth_step_deg))
    ratios = sorted({float(ratio) for ratio in ratio_grid} | {float(baseline_over_wavelength)})
    rows: list[dict[str, float]] = []
    for ratio in ratios:
        for azimuth_deg in azimuth_grid:
            values = residuals(float(azimuth_deg), ratio)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "baseline_over_wavelength": ratio,
                    "azimuth_deg": float(azimuth_deg),
                    "median_abs_residual_cycles": median_residual(values),
                    "pct_le_0p05_cycles": (
                        float(100.0 * np.mean(finite <= 0.05)) if finite.size else float("nan")
                    ),
                }
            )
    grid = pd.DataFrame(rows)
    best_per_ratio = grid.loc[
        grid.groupby("baseline_over_wavelength")["median_abs_residual_cycles"].idxmin()
    ].reset_index(drop=True)
    at_ratio = grid.loc[grid["baseline_over_wavelength"].eq(float(baseline_over_wavelength))]
    best = at_ratio.loc[at_ratio["median_abs_residual_cycles"].idxmin()]
    fitted_azimuth = float(best["azimuth_deg"])
    fitted_residual = float(best["median_abs_residual_cycles"])
    configured = float(configured_azimuth_deg) % 360.0
    configured_residual = median_residual(residuals(configured, float(baseline_over_wavelength)))
    opposite_residual = median_residual(
        residuals((configured + 180.0) % 360.0, float(baseline_over_wavelength))
    )
    best_values = residuals(fitted_azimuth, float(baseline_over_wavelength))
    per_svid = {}
    for position, svid in enumerate(authentic):
        columns = [index for index, pair in enumerate(pairs) if position in pair]
        per_svid[svid] = median_residual(best_values[:, columns])
    per_svid_residual = pd.Series(per_svid, name="median_abs_residual_cycles")
    per_svid_residual.index.name = "svid"

    difference = ((fitted_azimuth - configured + 180.0) % 360.0) - 180.0
    if abs(difference) <= azimuth_tolerance_deg:
        consistent = True
        recommendation = (
            f"configured baseline azimuth {configured:.0f} deg is consistent with the "
            f"authentic PRNs (fit {fitted_azimuth:.0f} deg, median residual "
            f"{fitted_residual:.3f} cycles at the fit, {configured_residual:.3f} as configured)."
        )
    elif abs(abs(difference) - 180.0) <= azimuth_tolerance_deg:
        consistent = False
        recommendation = (
            f"receivers are wired the other way round: the authentic PRNs fit baseline "
            f"azimuth {fitted_azimuth:.0f} deg, not {configured:.0f} deg (residual "
            f"{fitted_residual:.3f} vs {configured_residual:.3f} cycles). Set "
            f"navsat_azimuth_bias_deg={fitted_azimuth:.0f}, or keep the bias and swap "
            f"single_difference_order. Absolute angles computed with the current setting "
            f"are wrong."
        )
    else:
        consistent = False
        recommendation = (
            f"the authentic PRNs fit baseline azimuth {fitted_azimuth:.0f} deg, "
            f"{difference:+.0f} deg from the configured {configured:.0f} deg (residual "
            f"{fitted_residual:.3f} vs {configured_residual:.3f} cycles). Set "
            f"navsat_azimuth_bias_deg={fitted_azimuth:.0f} before reading absolute angles."
        )
    return BaselineAzimuthFit(
        authentic_svids=tuple(authentic),
        baseline_over_wavelength=float(baseline_over_wavelength),
        configured_azimuth_deg=configured,
        fitted_azimuth_deg=fitted_azimuth,
        fitted_residual_cycles=fitted_residual,
        configured_residual_cycles=configured_residual,
        opposite_residual_cycles=opposite_residual,
        azimuth_difference_deg=float(difference),
        consistent=consistent,
        recommendation=recommendation,
        per_svid_residual_cycles=per_svid_residual,
        best_per_ratio=best_per_ratio,
        grid=grid,
    )


def _true_segments(mask: pd.Series) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for index, flag in mask.items():
        index = int(index)
        if bool(flag):
            if start is None:
                start = index
            previous = index
        elif start is not None:
            segments.append((start, int(previous)))
            start = None
    if start is not None:
        segments.append((start, int(previous)))
    return segments


def _describe_segments(segments: Sequence[tuple[int, int]], tow: pd.Series | None) -> str:
    parts: list[str] = []
    for first, last in segments:
        count = last - first + 1
        if tow is not None and first in tow.index and last in tow.index:
            parts.append(f"TOW {tow.loc[first]:.0f}-{tow.loc[last]:.0f} ({count} ep)")
        else:
            parts.append(f"epochs {first}-{last} ({count} ep)")
    return "; ".join(parts)


SPOOF_WINDOW_COLUMNS = [
    "window", "start_epoch", "end_epoch", "epochs", "valid_epochs",
    "start_tow", "end_tow", "solvable",
]


def summarize_spoof_windows(
    valid_mask: pd.Series,
    tow: pd.Series,
    *,
    max_gap_epochs: int = 0,
    min_window_epochs: int = 2,
) -> pd.DataFrame:
    """List every spoof-group window of a capture in time order.

    ``valid_mask`` marks synchronized epochs that meet the spoof-PRN count policy;
    windows split at gaps longer than ``max_gap_epochs``. A window is ``solvable``
    when it holds at least ``min_window_epochs`` valid epochs. Epoch numbers are
    synchronized-epoch positions, i.e. ``DatasetConfig.start_index`` values.
    """
    rows = []
    for number, epochs in enumerate(
        valid_epoch_segments(valid_mask, max_gap_epochs=max_gap_epochs), start=1
    ):
        valid = int(valid_mask.reindex(epochs).fillna(False).sum())
        rows.append({
            "window": number,
            "start_epoch": epochs[0],
            "end_epoch": epochs[-1],
            "epochs": len(epochs),
            "valid_epochs": valid,
            "start_tow": float(tow.get(epochs[0], np.nan)),
            "end_tow": float(tow.get(epochs[-1], np.nan)),
            "solvable": valid >= min_window_epochs,
        })
    return pd.DataFrame(rows, columns=SPOOF_WINDOW_COLUMNS)


@dataclass
class CaptureProbe:
    """Data check of a two-receiver capture before the DD-method AoA flow."""

    receiver1_path: Path
    receiver2_path: Path
    spoof_svids: tuple[int, ...]
    message_inventory: pd.DataFrame
    observations: tuple[UbxObservationTables, UbxObservationTables]
    synced: SyncedObservations | None
    carrier_phase: CarrierPhaseTables | None
    common_valid_phase: pd.DataFrame
    receiver_svid_summary: pd.DataFrame
    spoof_phase_summary: pd.DataFrame
    usable_spoof_svids: tuple[int, ...]
    all_spoof_valid_epoch_count: int
    longest_all_spoof_valid_run: int
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    spoof_windows: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def ok(self) -> bool:
        return not self.problems

    def raise_for_problems(self) -> None:
        if self.problems:
            raise ValueError(
                "Capture is not usable for the DD-method AoA flow:\n- "
                + "\n- ".join(self.problems)
            )


def probe_capture(
    receiver1_path: str | Path,
    receiver2_path: str | Path,
    *,
    spoof_svids: Sequence[int] | None = None,
    gnss_id: int = 0,
    sig_id: int = 0,
    sync_tow_decimals: int | None = 0,
    use_navpvt_time_correction: bool = True,
    navpvt_tow_decimals: int = 0,
    thin_window_epochs: int = 5,
    min_window_epochs: int = 2,
    min_valid_spoof_prns: int | None = None,
    max_gap_epochs: int = 0,
) -> CaptureProbe:
    """Check that a two-receiver capture can feed the DD-method AoA flow.

    Reports the UBX message inventory per receiver, per-PRN tracking quality (RAWX rows,
    rows with ``cpValid=1`` and ``halfCyc=1``, C/N0, lock time, ``svUsed``, NAV-SAT
    geometry) and, for the spoof group, the epochs where every spoof PRN has valid carrier
    phase in both receivers -- the window the partial-spoof flow needs. ``problems`` lists
    what makes the capture unusable (no RXM-RAWX, no NAV-PVT, no Receiver 1 NAV-SAT, fewer
    than ``min_window_epochs`` continuous epochs with every usable spoof PRN valid in both
    receivers); ``warnings`` lists what shrinks it (spoof PRNs dropped, a window shorter
    than ``thin_window_epochs``).
    """
    spoof = _normalise_svids(spoof_svids) or ()
    paths = (Path(receiver1_path), Path(receiver2_path))
    labels = ("Receiver 1", "Receiver 2")
    problems: list[str] = []
    warnings: list[str] = []
    observations = tuple(
        read_ubx_observations(path, gnss_id=gnss_id, sig_id=sig_id) for path in paths
    )

    inventory_rows: list[dict[str, object]] = []
    for label, path, obs in zip(labels, paths, observations):
        row: dict[str, object] = {
            "receiver": label,
            "file": path.name,
            "size_kb": round(path.stat().st_size / 1024.0, 1),
        }
        for key in UBX_KEY_MESSAGES:
            row[key] = int(obs.message_counts.get(key, 0))
        row["rawx_epochs"] = int(obs.rawx["rcvTow"].nunique()) if not obs.rawx.empty else 0
        row["rawx_span_s"] = (
            float(obs.rawx["rcvTow"].max() - obs.rawx["rcvTow"].min())
            if not obs.rawx.empty
            else 0.0
        )
        inventory_rows.append(row)
        if obs.rawx.empty:
            problems.append(
                f"{label} ({path.name}) has no RXM-RAWX for GPS L1: it contributes no "
                "carrier phase, so no double difference can be formed. Enable RXM-RAWX "
                "output on that receiver."
            )
        elif obs.navpvt.empty and use_navpvt_time_correction:
            problems.append(
                f"{label} ({path.name}) has no NAV-PVT: the Eq. 8 time correction "
                "(nano * doMes) cannot be applied and every carrier phase becomes NaN. "
                "Enable NAV-PVT or set use_navpvt_time_correction=False."
            )
        if obs.navsat.empty:
            if label == "Receiver 1":
                problems.append(
                    f"{label} ({path.name}) has no NAV-SAT: the reference-angle anchor and "
                    "the ephemeris-based AoA both need Receiver 1 NAV-SAT azimuth/elevation."
                )
            else:
                warnings.append(
                    f"{label} ({path.name}) has no NAV-SAT: the Receiver 2 ephemeris "
                    "diagnostic and svUsed timeline will be empty."
                )
    message_inventory = pd.DataFrame(inventory_rows).set_index("receiver")

    svid_frames: list[pd.DataFrame] = []
    for label, obs in zip(labels, observations):
        if obs.rawx.empty:
            continue
        rawx = obs.rawx
        usable = rawx["cp_valid"].fillna(0).astype(int).eq(1) & rawx["half_cycle"].fillna(
            0
        ).astype(int).eq(1)
        per_svid = rawx.assign(usable_phase=usable).groupby("svid").agg(
            rawx_rows=("cpMes", "size"),
            usable_phase_rows=("usable_phase", "sum"),
            mean_cno=("cno", "mean"),
            max_locktime_s=("locktime", lambda values: float(values.max()) / 1000.0),
        )
        if not obs.navsat.empty:
            navsat = obs.navsat.groupby("svid").agg(
                sv_used_epochs=("svUsed", "sum"),
                median_elev_deg=("elev", "median"),
                median_azim_deg=("azim", "median"),
            )
            per_svid = per_svid.join(navsat, how="left")
        per_svid.insert(0, "receiver", label)
        per_svid["spoof"] = per_svid.index.isin(spoof)
        svid_frames.append(per_svid.reset_index())
    receiver_svid_summary = (
        pd.concat(svid_frames, ignore_index=True) if svid_frames else pd.DataFrame()
    )
    if not receiver_svid_summary.empty:
        receiver_svid_summary["svid"] = receiver_svid_summary["svid"].astype(int)

    spoof_columns = [
        "svid",
        "rx1_valid_epochs",
        "rx2_valid_epochs",
        "common_valid_epochs",
        "longest_common_run",
        "usable",
        "common_segments",
        "rx1_segments",
        "rx2_segments",
    ]
    if any(obs.rawx.empty for obs in observations):
        return CaptureProbe(
            receiver1_path=paths[0],
            receiver2_path=paths[1],
            spoof_svids=spoof,
            message_inventory=message_inventory,
            observations=observations,
            synced=None,
            carrier_phase=None,
            common_valid_phase=pd.DataFrame(),
            receiver_svid_summary=receiver_svid_summary,
            spoof_phase_summary=pd.DataFrame(columns=spoof_columns),
            usable_spoof_svids=(),
            all_spoof_valid_epoch_count=0,
            longest_all_spoof_valid_run=0,
            problems=problems,
            warnings=warnings,
        )

    try:
        synced = sync_receiver_epochs(
            observations[0], observations[1], tow_round_decimals=sync_tow_decimals
        )
    except ValueError as error:
        problems.append(f"receiver synchronization failed: {error}")
        return CaptureProbe(
            receiver1_path=paths[0],
            receiver2_path=paths[1],
            spoof_svids=spoof,
            message_inventory=message_inventory,
            observations=observations,
            synced=None,
            carrier_phase=None,
            common_valid_phase=pd.DataFrame(),
            receiver_svid_summary=receiver_svid_summary,
            spoof_phase_summary=pd.DataFrame(columns=spoof_columns),
            usable_spoof_svids=(),
            all_spoof_valid_epoch_count=0,
            longest_all_spoof_valid_run=0,
            problems=problems,
            warnings=warnings,
        )
    carrier_phase = build_carrier_phase_table(
        synced,
        use_navpvt_time_correction=use_navpvt_time_correction,
        navpvt_tow_decimals=navpvt_tow_decimals,
    )
    common_valid = carrier_phase.rx1_cycles.notna() & carrier_phase.rx2_cycles.notna()
    tow = synced.epochs.set_index("epoch_index")["rcvTow_rx1"]
    false_column = pd.Series(False, index=common_valid.index)

    spoof_rows: list[dict[str, object]] = []
    for svid in spoof:
        rx1_valid = (
            carrier_phase.rx1_cycles[svid].notna()
            if svid in carrier_phase.rx1_cycles.columns
            else false_column
        )
        rx2_valid = (
            carrier_phase.rx2_cycles[svid].notna()
            if svid in carrier_phase.rx2_cycles.columns
            else false_column
        )
        both = common_valid[svid] if svid in common_valid.columns else false_column
        spoof_rows.append(
            {
                "svid": int(svid),
                "rx1_valid_epochs": int(rx1_valid.sum()),
                "rx2_valid_epochs": int(rx2_valid.sum()),
                "common_valid_epochs": int(both.sum()),
                "longest_common_run": len(longest_valid_epoch_segment(both)),
                "usable": bool(both.any()),
                "common_segments": _describe_segments(_true_segments(both), tow),
                "rx1_segments": _describe_segments(_true_segments(rx1_valid), tow),
                "rx2_segments": _describe_segments(_true_segments(rx2_valid), tow),
            }
        )
    spoof_phase_summary = pd.DataFrame(spoof_rows, columns=spoof_columns)
    usable_spoof = tuple(int(row["svid"]) for row in spoof_rows if row["usable"])
    dropped = [svid for svid in spoof if svid not in usable_spoof]
    if spoof and not usable_spoof:
        problems.append(
            f"no spoof PRN of {list(spoof)} has valid carrier phase (cpValid=1, halfCyc=1) "
            "in both receivers at the same epoch; there is nothing to estimate. Check lock "
            "time and halfCyc in receiver_svid_summary."
        )
    elif dropped:
        warnings.append(
            f"spoof PRNs {dropped} never have common valid phase and are dropped from the "
            f"group; the flow continues with {list(usable_spoof)}."
        )
    if usable_spoof:
        all_valid = common_valid.reindex(columns=list(usable_spoof)).all(axis=1)
    else:
        all_valid = false_column
    all_valid_count = int(all_valid.sum())
    longest_run = len(longest_valid_epoch_segment(all_valid))
    policy_mask = all_valid
    minimum = len(usable_spoof)
    if usable_spoof:
        minimum = _minimum_valid_spoof_prns(min_valid_spoof_prns, len(usable_spoof))
        policy_mask = common_valid.reindex(columns=list(usable_spoof)).sum(axis=1).ge(minimum)
    policy_window = longest_valid_epoch_segment(policy_mask, max_gap_epochs=max_gap_epochs)
    valid_window_count = int(policy_mask.reindex(policy_window).sum())
    spoof_windows = summarize_spoof_windows(
        policy_mask, tow, max_gap_epochs=max_gap_epochs, min_window_epochs=min_window_epochs,
    )
    if usable_spoof and valid_window_count < min_window_epochs:
        problems.append(
            f"only {valid_window_count} valid epoch(s) have at least {minimum} usable spoof PRNs "
            f"in one window with gaps <= {max_gap_epochs} epoch(s); the joint-source "
            f"selection needs at least {min_window_epochs}. Check lock time and halfCyc in "
            "receiver_svid_summary."
        )
    elif usable_spoof and valid_window_count < thin_window_epochs:
        warnings.append(
            f"only {valid_window_count} epoch(s) meet the spoof-PRN count policy in "
            "both receivers; the AoA window will be thin."
        )

    return CaptureProbe(
        receiver1_path=paths[0],
        receiver2_path=paths[1],
        spoof_svids=spoof,
        message_inventory=message_inventory,
        observations=observations,
        synced=synced,
        carrier_phase=carrier_phase,
        common_valid_phase=common_valid,
        receiver_svid_summary=receiver_svid_summary,
        spoof_phase_summary=spoof_phase_summary,
        usable_spoof_svids=usable_spoof,
        all_spoof_valid_epoch_count=all_valid_count,
        longest_all_spoof_valid_run=longest_run,
        problems=problems,
        warnings=warnings,
        spoof_windows=spoof_windows,
    )


def run_dataset(config: DatasetConfig) -> AoaRunResult:
    """Run one DD-method AoA analysis from UBX files through collapse metrics."""
    read_limit = None
    if config.length is not None:
        read_limit = config.start_index + config.length + 120
    obs1 = read_ubx_observations(
        config.receiver1_path,
        config.gnss_id,
        config.sig_id,
        max_rawx_epochs=read_limit,
    )
    obs2 = read_ubx_observations(
        config.receiver2_path,
        config.gnss_id,
        config.sig_id,
        max_rawx_epochs=read_limit,
    )
    synced = sync_receiver_epochs(
        obs1, obs2, tow_round_decimals=config.sync_tow_decimals
    )
    synced = _slice_synced_epochs(synced, config.start_index, config.length)

    carrier_phase_full = build_carrier_phase_table(
        synced,
        use_navpvt_time_correction=config.use_navpvt_time_correction,
        navpvt_tow_decimals=config.navpvt_tow_decimals,
    )
    reference_svid, reference_selection = select_reference_svid(
        carrier_phase_full,
        cno=carrier_phase_full.mean_cno,
        override=config.reference_svid,
        exclude_svids=config.excluded_reference_svids,
        required_svids=config.reference_required_svids,
    )

    selected_svids = _normalise_svids(config.selected_svids)
    excluded_analysis_svids = set(_normalise_svids(config.excluded_analysis_svids) or ())
    if selected_svids is None:
        analysis_svids = tuple(
            int(svid)
            for svid in carrier_phase_full.rx1_cycles.columns
            if int(svid) not in excluded_analysis_svids
        )
    else:
        analysis_svids = tuple(
            svid for svid in selected_svids if svid not in excluded_analysis_svids
        )
    phase_svids = tuple(dict.fromkeys((reference_svid, *analysis_svids)))
    carrier_phase = _filter_carrier_phase_columns(carrier_phase_full, phase_svids)

    single_difference = compute_single_difference(
        carrier_phase, order=config.single_difference_order
    )
    double_difference_full = compute_double_difference(
        single_difference, reference_svid
    )
    double_difference = double_difference_full.reindex(columns=list(analysis_svids))
    rounded_dd, fractional_dd = fractional_round_dd(double_difference)
    metrics_by_epoch, metrics_summary = compute_relative_dd_collapse_metrics(
        fractional_dd
    )

    if config.alpha_r_deg is None:
        candidates = pd.DataFrame(
            columns=[
                "epoch_index",
                "svid",
                "fractional_dd_cycles",
                "branch",
                "reference_alpha_deg",
                "candidate_cos_alpha",
                "candidate_aoa_deg",
                "valid_candidate",
            ]
        )
        selected_aoa = pd.DataFrame(index=fractional_dd.index)
        selected_branch = pd.DataFrame(index=fractional_dd.index)
        aoa_metrics_by_epoch = pd.DataFrame(index=fractional_dd.index)
        aoa_metrics_summary = pd.DataFrame()
    else:
        candidates = generate_aoa_candidates(
            fractional_dd,
            alpha_r_deg=config.alpha_r_deg,
            branch_values=config.branch_values,
            baseline_over_wavelength=config.baseline_over_wavelength,
        )
        selected_aoa, selected_branch = select_aoa_dp(
            candidates,
            reference_svid=reference_svid,
            reference_angle_deg=config.alpha_r_deg,
            return_branches=True,
            metric=config.branch_metric,
        )
        aoa_metrics_by_epoch, aoa_metrics_summary = compute_collapse_metrics(
            selected_aoa
        )

    navsat_aoa = compute_navsat_aoa(
        obs1.navsat,
        baseline_azimuth_bias_deg=config.navsat_azimuth_bias_deg,
    )
    ephemeris_aoa = compute_navsat_aoa_by_synced_epoch(
        synced.epochs,
        obs1.navsat,
        epoch_tow_column="rcvTow_rx1",
        navsat_tow_decimals=config.navpvt_tow_decimals,
        baseline_azimuth_bias_deg=config.navsat_azimuth_bias_deg,
    )

    return AoaRunResult(
        config=config,
        epochs=synced.epochs,
        reference_svid=reference_svid,
        reference_selection=reference_selection,
        carrier_phase_cycles=carrier_phase.combined_cycles,
        carrier_phase_rx1_cycles=carrier_phase.rx1_cycles,
        carrier_phase_rx2_cycles=carrier_phase.rx2_cycles,
        single_difference_cycles=single_difference,
        double_difference_cycles=double_difference,
        rounded_dd_cycles=rounded_dd,
        fractional_dd_cycles=fractional_dd,
        candidate_aoa_deg=candidates,
        selected_aoa_deg=selected_aoa,
        selected_branch=selected_branch,
        collapse_metrics_by_epoch=metrics_by_epoch,
        collapse_metrics=metrics_summary,
        aoa_collapse_metrics_by_epoch=aoa_metrics_by_epoch,
        aoa_collapse_metrics=aoa_metrics_summary,
        navsat_aoa_diagnostic_deg=navsat_aoa,
        ephemeris_aoa_deg=ephemeris_aoa,
    )


def _write_csv(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, na_rep="")


def _import_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _plot_wide_dataframe(
    df: pd.DataFrame,
    title: str | None,
    ylabel: str,
    *,
    ylim: tuple[float, float] | None = None,
    highlight_svids: Sequence[int] | None = None,
    highlight_label: str | None = None,
):
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(12, 5))
    highlighted = set(_normalise_svids(highlight_svids) or ())
    for column in df.columns:
        series = df[column]
        if series.notna().any():
            svid = int(column)
            is_highlighted = svid in highlighted
            suffix = (
                f" ({highlight_label})"
                if is_highlighted and highlight_label
                else ""
            )
            ax.plot(
                df.index,
                series,
                label=f"PRN {svid}{suffix}",
                linewidth=2.0 if is_highlighted else 1.25,
                alpha=0.95 if is_highlighted else 0.8,
                marker="o" if is_highlighted else None,
                markersize=2.5,
                markevery=3,
            )
    if title:
        ax.set_title(title)
    ax.set_xlabel("Epoch index")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    if len(df.columns) > 0:
        ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    return fig


def plot_carrier_phase(result: AoaRunResult):
    plt = _import_pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, df, label in [
        (axes[0], result.carrier_phase_rx1_cycles, "Receiver 1"),
        (axes[1], result.carrier_phase_rx2_cycles, "Receiver 2"),
    ]:
        for column in df.columns:
            series = df[column]
            if series.notna().any():
                ax.plot(
                    df.index, series, label=f"SV {column}", linewidth=1.0, alpha=0.8
                )
        ax.set_ylabel("Carrier phase (cycles)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch index")
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    return fig


def plot_single_difference(result: AoaRunResult):
    return _plot_wide_dataframe(
        result.single_difference_cycles,
        "Receiver Single Difference Carrier Phase",
        "Single difference (cycles)",
    )


def plot_double_difference(result: AoaRunResult):
    return _plot_wide_dataframe(
        result.double_difference_cycles,
        f"Double Difference Carrier Phase (reference SV {result.reference_svid})",
        "Double difference (cycles)",
    )


def plot_rounding_steps(result: AoaRunResult):
    return _plot_wide_dataframe(
        result.rounded_dd_cycles,
        "Rounded Double Difference Carrier Phase",
        "Rounded DD (cycles)",
    )


def plot_fractional_dd(
    result: AoaRunResult,
    *,
    highlight_svids: Sequence[int] | None = None,
    title: str | None = "Fractional Double Difference Carrier Phase",
):
    return _plot_wide_dataframe(
        result.fractional_dd_cycles,
        title,
        "Fractional DD (cycles)",
        ylim=(-0.55, 0.55),
        highlight_svids=highlight_svids,
        highlight_label="spoof",
    )


def plot_candidate_cloud(result: AoaRunResult):
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(12, 5))
    data = result.candidate_aoa_deg.dropna(subset=["candidate_aoa_deg"])
    if not data.empty:
        scatter = ax.scatter(
            data["epoch_index"],
            data["candidate_aoa_deg"],
            c=data["svid"],
            s=8,
            alpha=0.35,
            cmap="tab20",
        )
        fig.colorbar(scatter, ax=ax, label="SVID")
    ax.set_title("AoA Branch Candidates")
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Candidate AoA (deg)")
    ax.set_ylim(0, 180)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_valid_m_candidates(
    candidates: pd.DataFrame,
    *,
    reference_aoa: pd.DataFrame | None = None,
    title: str = "Valid AoA Candidates by Branch",
    highlight_svids: Sequence[int] = (),
):
    """Plot all physically valid AoA branches, faceted by SVID."""
    plt = _import_pyplot()
    from matplotlib.lines import Line2D

    data = valid_m_candidates(candidates)
    svids = sorted(int(svid) for svid in data["svid"].unique()) if not data.empty else []
    if not svids:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No valid candidates", ha="center", va="center")
        ax.axis("off")
        return fig

    ncols = min(3, len(svids))
    nrows = int(np.ceil(len(svids) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    branch_colors = {
        -2: "#7f3b08",
        -1: "#2c7fb8",
        0: "#31a354",
        1: "#f16913",
        2: "#756bb1",
    }
    highlight_set = set(int(svid) for svid in highlight_svids)

    for ax, svid in zip(axes_array, svids):
        subset = data.loc[data["svid"].eq(svid)]
        for branch, branch_subset in subset.groupby("branch", sort=True):
            ax.scatter(
                branch_subset["epoch_index"],
                branch_subset["candidate_aoa_deg"],
                s=10,
                alpha=0.55,
                color=branch_colors.get(int(branch), "#636363"),
                label=f"m={int(branch)}",
            )
        if reference_aoa is not None and svid in reference_aoa.columns:
            reference = reference_aoa[svid].dropna()
            if not reference.empty:
                ax.plot(
                    reference.index,
                    reference,
                    color="black",
                    linestyle="--",
                    linewidth=1.4,
                    label="ephemeris",
                )
        suffix = " spoof" if svid in highlight_set else ""
        ax.set_title(f"SV {svid}{suffix}")
        ax.set_ylim(0, 180)
        ax.grid(True, alpha=0.3)

    for ax in axes_array[len(svids) :]:
        ax.axis("off")
    branch_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=f"m={int(branch)}",
            markerfacecolor=branch_colors.get(int(branch), "#636363"),
            markersize=6,
            alpha=0.7,
        )
        for branch in sorted(int(branch) for branch in data["branch"].unique())
    ]
    if reference_aoa is not None:
        branch_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.4,
                label="ephemeris",
            )
        )
    fig.legend(branch_handles, [handle.get_label() for handle in branch_handles], ncol=6)
    fig.suptitle(title, y=0.995)
    fig.supxlabel("Epoch index")
    fig.supylabel("AoA (deg)")
    fig.tight_layout()
    return fig


def plot_aoa_method_comparison(
    dd_aoa: pd.DataFrame,
    ephemeris_aoa: pd.DataFrame,
    *,
    title: str,
    highlight_svids: Sequence[int] = (),
):
    """Plot DD-method AoA against ephemeris AoA for each shared SVID."""
    plt = _import_pyplot()
    shared_svids = sorted(
        int(column)
        for column in dd_aoa.columns
        if int(column) in {int(ephem_column) for ephem_column in ephemeris_aoa.columns}
    )
    if not shared_svids:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No shared SVIDs", ha="center", va="center")
        ax.axis("off")
        return fig

    ncols = min(3, len(shared_svids))
    nrows = int(np.ceil(len(shared_svids) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    highlight_set = set(int(svid) for svid in highlight_svids)

    for ax, svid in zip(axes_array, shared_svids):
        joined = pd.concat(
            [
                dd_aoa[svid].rename("dd_aoa_deg"),
                ephemeris_aoa[svid].rename("ephemeris_aoa_deg"),
            ],
            axis=1,
        )
        if joined["dd_aoa_deg"].notna().any():
            ax.plot(
                joined.index,
                joined["dd_aoa_deg"],
                color="#2c7fb8",
                linewidth=1.4,
                label="DD",
            )
        if joined["ephemeris_aoa_deg"].notna().any():
            ax.plot(
                joined.index,
                joined["ephemeris_aoa_deg"],
                color="black",
                linestyle="--",
                linewidth=1.3,
                label="ephemeris",
            )
        comparison = joined.dropna()
        error_label = ""
        if not comparison.empty:
            median_error = (
                comparison["dd_aoa_deg"] - comparison["ephemeris_aoa_deg"]
            ).abs().median()
            error_label = f", med err {median_error:.1f} deg"
        suffix = " spoof" if svid in highlight_set else ""
        ax.set_title(f"SV {svid}{suffix}{error_label}")
        ax.set_ylim(0, 180)
        ax.grid(True, alpha=0.3)

    for ax in axes_array[len(shared_svids) :]:
        ax.axis("off")
    axes_array[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.supxlabel("Epoch index")
    fig.supylabel("AoA (deg)")
    fig.tight_layout()
    return fig


def plot_selected_aoa_tracks(result: AoaRunResult):
    return _plot_wide_dataframe(
        result.selected_aoa_deg,
        "Selected DD-method AoA Tracks",
        "Selected AoA (deg)",
        ylim=(0, 180),
    )


def plot_collapse_metrics(result: AoaRunResult):
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(12, 4))
    data = result.collapse_metrics_by_epoch
    ax.plot(
        data.index,
        data["max_minus_min_dispersion_cycles"],
        label="max-min dispersion",
        linewidth=1.4,
    )
    ax.plot(
        data.index,
        data["median_absolute_deviation_cycles"],
        label="median absolute deviation",
        linewidth=1.2,
    )
    ax.set_title("Cross-satellite Relative DD Collapse Metrics")
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Cycles")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_aoa_collapse_metrics(result: AoaRunResult):
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(12, 4))
    data = result.aoa_collapse_metrics_by_epoch
    if not data.empty:
        ax.plot(
            data.index,
            data["max_minus_min_dispersion_deg"],
            label="max-min dispersion",
            linewidth=1.4,
        )
        ax.plot(
            data.index,
            data["median_absolute_deviation_deg"],
            label="median absolute deviation",
            linewidth=1.2,
        )
    ax.set_title("Cross-satellite AoA Collapse Metrics")
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Degrees")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_navsat_aoa_diagnostic(result: AoaRunResult):
    return _plot_wide_dataframe(
        result.navsat_aoa_diagnostic_deg,
        "NAV-SAT AoA Diagnostic",
        "Diagnostic AoA (deg)",
        ylim=(0, 180),
    )


def plot_aoa_tracks(
    aoa_tracks: pd.DataFrame,
    *,
    title: str,
    highlight_svids: Sequence[int] = (),
):
    """Plot baseline-relative AoA tracks, emphasizing selected SVIDs."""
    plt = _import_pyplot()
    columns = [int(column) for column in aoa_tracks.columns]
    highlight = set(_normalise_svids(highlight_svids) or ())
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(columns), 1)))

    fig, ax = plt.subplots(figsize=(12, 5))
    for color, svid in zip(colors, columns):
        series = aoa_tracks[svid]
        is_highlighted = svid in highlight
        ax.plot(
            series.index,
            series,
            marker="o" if is_highlighted else None,
            markersize=3 if is_highlighted else 0,
            linewidth=2.4 if is_highlighted else 1.0,
            alpha=1.0 if is_highlighted else 0.45,
            color=color,
            label=f"SV {svid}",
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Baseline-relative AoA (deg)")
    ax.set_ylim(0, 180)
    ax.grid(True, alpha=0.3)
    if columns:
        ax.legend(ncol=6, fontsize=8)
    fig.tight_layout()
    return fig


def plot_spoof_branch_summary(
    detection: pd.DataFrame,
    *,
    prior_aoa_deg: float | None = None,
    cluster_median_aoa_deg: float | None = None,
    title: str = "Spoof PRNs: DD-method AoA on both Eq. 21 branches vs ephemeris-based AoA",
):
    """Per spoof PRN: selected and alternative branch (median +- std), ephemeris +- threshold."""
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(max(6.5, 1.7 * len(detection) + 3.5), 5.0))
    if detection.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No spoof PRN detection table", ha="center", va="center")
        ax.axis("off")
        return fig

    threshold = float(detection["threshold_deg"].iloc[0])
    positions = np.arange(len(detection))
    for position, row in zip(positions, detection.itertuples()):
        if np.isfinite(row.ephemeris_aoa_deg):
            ax.fill_between(
                [position - 0.3, position + 0.3],
                row.ephemeris_aoa_deg - threshold,
                row.ephemeris_aoa_deg + threshold,
                color="#2c7fb8",
                alpha=0.12,
                label=f"ephemeris-based AoA +- {threshold:.0f} deg" if position == 0 else None,
            )
            ax.plot(
                position,
                row.ephemeris_aoa_deg,
                marker="x",
                color="#2c7fb8",
                ms=10,
                mew=2,
                ls="none",
                label="ephemeris-based AoA (real satellite)" if position == 0 else None,
                zorder=2,
            )
        if np.isfinite(row.selected_aoa_deg):
            ax.errorbar(
                position,
                row.selected_aoa_deg,
                yerr=row.selected_std_deg if np.isfinite(row.selected_std_deg) else 0.0,
                fmt="o",
                color="#d7301f",
                capsize=3,
                ms=7,
                label=(
                    f"DD-method AoA, selected branch (chosen_by = {row.chosen_by})"
                    if position == 0
                    else None
                ),
                zorder=3,
            )
            ax.annotate(
                f"m={int(row.selected_m)}",
                (position, row.selected_aoa_deg),
                xytext=(7, 4),
                textcoords="offset points",
                fontsize=8,
                color="#d7301f",
            )
        if np.isfinite(row.alternative_aoa_deg):
            ax.errorbar(
                position,
                row.alternative_aoa_deg,
                yerr=(
                    row.alternative_std_deg if np.isfinite(row.alternative_std_deg) else 0.0
                ),
                fmt="s",
                color="#fdae6b",
                capsize=3,
                ms=7,
                label="DD-method AoA, alternative Eq. 21 branch" if position == 0 else None,
                zorder=3,
            )
            ax.annotate(
                f"m={int(row.alternative_m)}",
                (position, row.alternative_aoa_deg),
                xytext=(7, 4),
                textcoords="offset points",
                fontsize=8,
                color="#b35806",
            )
    if cluster_median_aoa_deg is not None and np.isfinite(cluster_median_aoa_deg):
        ax.axhline(
            float(cluster_median_aoa_deg),
            color="#d7301f",
            lw=0.8,
            ls="--",
            alpha=0.7,
            label=f"cluster median {float(cluster_median_aoa_deg):.1f} deg",
        )
    if prior_aoa_deg is not None and np.isfinite(prior_aoa_deg):
        ax.axhline(
            float(prior_aoa_deg),
            color="#636363",
            lw=1.0,
            ls=":",
            label=f"declared prior {float(prior_aoa_deg):.0f} deg",
        )
    ax.set_xticks(positions, [f"PRN {int(svid)}" for svid in detection["svid"]])
    ax.set_xlim(-0.6, len(detection) - 0.4)
    ax.set_ylim(0, 180)
    ax.set_yticks(range(0, 181, 30))
    ax.set_ylabel("Baseline-relative AoA (deg)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_sv_used_timeline(
    sv_used_by_receiver: dict[str, pd.DataFrame],
    *,
    title: str = "u-blox NAV-SAT svUsed by Receiver",
    highlight_svids: Sequence[int] = (),
):
    """Plot receiver-native svUsed timelines for every NAV-SAT epoch."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    plt = _import_pyplot()
    all_svids = sorted(
        {
            int(column)
            for frame in sv_used_by_receiver.values()
            for column in frame.columns
        }
        | set(_normalise_svids(highlight_svids) or ())
    )
    highlight = set(_normalise_svids(highlight_svids) or ())
    if not all_svids:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No NAV-SAT svUsed data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    receiver_count = max(len(sv_used_by_receiver), 1)
    fig_height = min(
        10.0,
        max(6.0, 2.8 * receiver_count + 0.12 * len(all_svids) * receiver_count),
    )
    fig, axes = plt.subplots(
        receiver_count,
        1,
        figsize=(12, fig_height),
        sharex=False,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    cmap = ListedColormap(["#d8d8d8", "#f2b447", "#277f6a"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    image = None

    for ax, (receiver, frame) in zip(axes, sv_used_by_receiver.items()):
        matrix = frame.reindex(columns=all_svids).T
        values = matrix.where(matrix.notna(), -1).astype(float).clip(lower=-1, upper=1)
        image = ax.imshow(
            values.to_numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        ax.set_title(receiver)
        ax.set_ylabel("SVID")
        ax.set_yticks(range(len(all_svids)))
        ax.set_yticklabels([str(svid) for svid in all_svids])
        for tick, svid in zip(ax.get_yticklabels(), all_svids):
            if svid in highlight:
                tick.set_color("crimson")
                tick.set_fontweight("bold")
        ax.set_xlabel("NAV-SAT epoch index")
        ax.grid(False)

    fig.suptitle(title)
    if image is not None:
        cbar = fig.colorbar(
            image, ax=axes.tolist(), ticks=[-1, 0, 1], fraction=0.025, pad=0.02
        )
        cbar.ax.set_yticklabels(["no NAV-SAT", "not used", "used"])
    return fig


def plot_sv_used_by_tow(
    status_by_receiver: Mapping[str, pd.DataFrame],
    *,
    highlight_svids: Sequence[int] = (),
    windows: pd.DataFrame | None = None,
    min_tracked_seconds: int = 10,
    title: str = "PRNs used in the receiver navigation solution (NAV-SAT svUsed)",
):
    """Per-receiver svUsed timelines on a GPS-TOW axis plus the used-PRN count.

    ``status_by_receiver`` comes from ``navsat_sv_status_by_second``. Highlighted
    (spoof) PRNs are drawn first, above a red line. ``windows`` (``window``,
    ``start_tow``, ``end_tow``) marks the analysis windows. Other PRNs are shown when
    used at least once or tracked for ``min_tracked_seconds`` in some receiver; the
    used-PRN count always includes every PRN. NAV-SAT dropouts stay blank.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    plt = _import_pyplot()
    highlight = list(_normalise_svids(highlight_svids) or ())
    others = sorted(
        {int(svid) for status in status_by_receiver.values() for svid in status.columns}
        - set(highlight)
    )
    shown = sorted(
        {
            int(svid)
            for status in status_by_receiver.values()
            for svid in status.columns
            if int(svid) in others
            and (status[svid].eq(SV_STATUS_USED).any()
                 or status[svid].ge(SV_STATUS_TRACKED).sum() >= min_tracked_seconds)
        }
    )
    svids = highlight + shown
    receivers = list(status_by_receiver)
    fig, axes = plt.subplots(
        len(receivers) + 1, 1, sharex=True, constrained_layout=True,
        figsize=(12, 2.8 + 0.17 * len(svids) * len(receivers) + 2.0),
        gridspec_kw={"height_ratios": [max(len(svids), 1)] * len(receivers) + [10]},
    )
    colors = ["#e6e6e6", "#f2b447", "#277f6a"]
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    count_ax = axes[-1]
    for index, (ax, receiver) in enumerate(zip(axes, receivers)):
        status = status_by_receiver[receiver]
        ax.set_ylabel(f"{receiver}\nPRN")
        if status.empty:
            ax.text(0.5, 0.5, "No NAV-SAT svUsed data", ha="center", va="center", transform=ax.transAxes)
            continue
        present = status.notna().any(axis=1)
        full = status.reindex(columns=highlight + others)
        full = full.where(~present, full.fillna(SV_STATUS_NOT_TRACKED))
        matrix = full[svids]
        tow = matrix.index.to_numpy(dtype=float)
        ax.imshow(
            np.ma.masked_invalid(matrix.T.to_numpy(dtype=float)),
            aspect="auto", interpolation="nearest", cmap=cmap, norm=norm,
            extent=(tow[0] - 0.5, tow[-1] + 0.5, len(svids) - 0.5, -0.5),
        )
        ax.set_yticks(range(len(svids)))
        ax.set_yticklabels([str(svid) for svid in svids], fontsize=7)
        for tick, svid in zip(ax.get_yticklabels(), svids):
            if svid in highlight:
                tick.set_color("crimson")
                tick.set_fontweight("bold")
        if highlight and shown:
            ax.axhline(len(highlight) - 0.5, color="crimson", linewidth=0.8)

        used = full.eq(SV_STATUS_USED)
        color = f"C{index}"
        for group, style, label in ((highlight, "-", "spoof"), (others, "--", "authentic")):
            if group:
                count = used[group].sum(axis=1).where(present)
                count_ax.step(tow, count, where="mid", color=color, linestyle=style,
                              label=f"{receiver} {label}")

    if highlight:
        count_ax.axhline(len(highlight), color="crimson", linestyle=":", linewidth=0.8)
    count_ax.set_ylabel("PRNs used")
    count_ax.set_ylim(bottom=0)
    count_ax.legend(loc="lower right", fontsize=8, ncol=len(receivers))
    count_ax.set_xlabel("GPS TOW (s)")
    count_ax.ticklabel_format(axis="x", useOffset=False, style="plain")

    if windows is not None:
        for window in windows.itertuples(index=False):
            for ax in axes:
                for tow in (window.start_tow, window.end_tow):
                    ax.axvline(float(tow), color="black", linestyle="--", linewidth=0.8)
            axes[0].text(
                (float(window.start_tow) + float(window.end_tow)) / 2, -0.6, f"W{int(window.window)}",
                ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle(title)
    fig.legend(
        handles=[Patch(facecolor="white", edgecolor="0.6", label="no NAV-SAT")]
        + [Patch(color=color, label=label)
           for color, label in zip(colors, ["not tracked", "tracked, not used", "used in fix"])],
        loc="outside lower center", ncol=4, fontsize=8, frameon=False,
    )
    return fig


def save_artifacts(
    result: AoaRunResult, output_dir: str | Path | None = None, *,
    save_figures: bool = True, figure_formats: Sequence[str] = ("png",),
) -> dict[str, Path]:
    """Save analysis tables, with optional diagnostic figure export."""
    directory = Path(output_dir) if output_dir is not None else result.config.output_dir
    directory.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}
    csv_specs = [
        ("epochs.csv", result.epochs, True),
        ("reference_selection.csv", result.reference_selection, False),
        ("carrier_phase_cycles.csv", result.carrier_phase_cycles, True),
        ("single_difference_cycles.csv", result.single_difference_cycles, True),
        ("double_difference_cycles.csv", result.double_difference_cycles, True),
        ("rounded_dd_cycles.csv", result.rounded_dd_cycles, True),
        ("fractional_dd_cycles.csv", result.fractional_dd_cycles, True),
        ("candidate_aoa_deg.csv", result.candidate_aoa_deg, False),
        ("selected_aoa_deg.csv", result.selected_aoa_deg, True),
        ("selected_branch.csv", result.selected_branch, True),
        ("collapse_metrics_by_epoch.csv", result.collapse_metrics_by_epoch, True),
        ("collapse_metrics.csv", result.collapse_metrics, False),
    ]
    if not result.aoa_collapse_metrics.empty:
        csv_specs.extend(
            [
                (
                    "aoa_collapse_metrics_by_epoch.csv",
                    result.aoa_collapse_metrics_by_epoch,
                    True,
                ),
                ("aoa_collapse_metrics.csv", result.aoa_collapse_metrics, False),
            ]
        )
    if not result.navsat_aoa_diagnostic_deg.empty:
        csv_specs.append(
            ("navsat_aoa_diagnostic_deg.csv", result.navsat_aoa_diagnostic_deg, True)
        )

    for filename, frame, include_index in csv_specs:
        path = directory / filename
        _write_csv(frame, path, index=include_index)
        artifacts[filename] = path

    if not save_figures:
        result.output_paths = artifacts
        return artifacts

    plot_specs = [
        ("carrier_phase_cycles.png", plot_carrier_phase),
        ("single_difference_cycles.png", plot_single_difference),
        ("double_difference_cycles.png", plot_double_difference),
        ("rounded_dd_cycles.png", plot_rounding_steps),
        ("fractional_dd_cycles.png", plot_fractional_dd),
        ("candidate_aoa_deg.png", plot_candidate_cloud),
        ("selected_aoa_deg.png", plot_selected_aoa_tracks),
        ("collapse_metrics.png", plot_collapse_metrics),
    ]
    if not result.aoa_collapse_metrics.empty:
        plot_specs.append(("aoa_collapse_metrics.png", plot_aoa_collapse_metrics))
    if not result.navsat_aoa_diagnostic_deg.empty:
        plot_specs.append(("navsat_aoa_diagnostic_deg.png", plot_navsat_aoa_diagnostic))

    plt = _import_pyplot()
    for filename, plotter in plot_specs:
        fig = plotter(result)
        for extension in figure_formats:
            path = directory / f"{Path(filename).stem}.{extension}"
            fig.savefig(path, dpi=180)
            artifacts[path.name] = path
        plt.close(fig)

    result.output_paths = artifacts
    return artifacts


def _combine_svids(*groups: Sequence[int] | None) -> tuple[int, ...]:
    values: list[int] = []
    for group in groups:
        values.extend(_normalise_svids(group) or ())
    return tuple(dict.fromkeys(values))


def _paper_scenario_summary(
    *,
    config: DatasetConfig,
    partial_spoof_enabled: bool,
    spoof_svids: tuple[int, ...],
    analysis_result: AoaRunResult,
    all_satellite_result: AoaRunResult,
    reference_alpha_deg: float | None,
    source_epoch_start: int,
    sv_used_summary: pd.DataFrame,
    prior_aoa_deg: float | None = None,
    joint_source_ambiguity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = [
        {
            "dataset": config.name,
            "partial_spoof_enabled": bool(partial_spoof_enabled),
            "spoof_svids": spoof_svids,
            "excluded_analysis_svids": config.excluded_analysis_svids or (),
            "reference_svid": analysis_result.reference_svid,
            "reference_alpha_deg": reference_alpha_deg,
            "navsat_azimuth_bias_deg": config.navsat_azimuth_bias_deg,
            "single_difference_order": config.single_difference_order,
            "baseline_over_wavelength": config.baseline_over_wavelength,
            "branch_metric": config.branch_metric,
            "prior_aoa_deg": prior_aoa_deg,
            "source_epoch_start": int(source_epoch_start),
            "analysis_epoch_count": len(analysis_result.epochs),
            "all_satellite_dd_tracks": int(
                all_satellite_result.selected_aoa_deg.notna().any(axis=0).sum()
            ),
            "output_dir": str(analysis_result.config.output_dir),
        }
    ]
    if joint_source_ambiguity is not None and not joint_source_ambiguity.empty:
        first = joint_source_ambiguity.iloc[0]
        rows[0].update(
            {
                "joint_source_chosen_by": str(first["chosen_by"]),
                "joint_source_ambiguous": bool(first["ambiguous"]),
                "joint_source_selected_branches": str(first["selected_branches"]),
                "joint_source_alternative_branches": str(first["alternative_branches"]),
                "joint_source_tie_tolerance": float(first["tie_tolerance"]),
            }
        )
    for row in sv_used_summary.itertuples():
        rows[0][f"rx_{row.receiver}_sv{int(row.svid)}_used_epoch_count"] = int(
            row.used_epoch_count
        )
    return pd.DataFrame(rows)


def run_paper_scenario(
    config: DatasetConfig,
    *,
    partial_spoof_enabled: bool = False,
    spoof_svids: Sequence[int] | None = None,
    convergence_threshold_deg: float = 5.0,
    save_outputs: bool = True,
    prior_aoa_deg: float | None = None,
    spoof_branch_m: int | None = None,
    tie_tolerance: float | None = None,
    detection_threshold_deg: float = 10.0,
    min_valid_spoof_prns: int | None = None,
    max_gap_epochs: int = 0,
    save_figures: bool = True,
) -> PaperScenarioResult:
    """Run one paper-style AoA flow for all-spoof or partial-spoof scenarios.

    In the partial-spoof flow the spoof PRNs' Eq. 21 branches are selected jointly with
    :func:`select_joint_source_aoa` using ``config.branch_metric``. ``prior_aoa_deg`` is the
    declared prior that breaks the one-wavelength branch tie (``None`` reports the tie as
    unresolved), ``tie_tolerance`` overrides its automatic tolerance, and
    ``detection_threshold_deg`` is the paper's AoA-difference threshold used in the
    per-PRN detection table (``spoof_branch_detection``).
    ``spoof_branch_m`` declares a fixed Eq. 21 branch for the spoof group, mutually
    exclusive with an AoA prior; the result records ``chosen_by="explicit_m"``.
    ``min_valid_spoof_prns=None, max_gap_epochs=0`` keeps strict window selection.
    Otherwise allow a smaller per-epoch group and bounded internal gaps, retain
    missing measurements, and restart joint branch selection at every gap or
    change of PRN membership. Gap size counts synchronized epoch rows.
    """
    normalized_spoof_svids = _normalise_svids(spoof_svids) or ()
    output_dir = config.output_dir

    full_obs1 = read_ubx_observations(
        config.receiver1_path,
        config.gnss_id,
        config.sig_id,
    )
    full_obs2 = read_ubx_observations(
        config.receiver2_path,
        config.gnss_id,
        config.sig_id,
    )

    sv_used_rx1 = compute_navsat_sv_used(full_obs1.navsat)
    sv_used_rx2 = compute_navsat_sv_used(full_obs2.navsat)
    summary_svids = normalized_spoof_svids or _combine_svids(
        sv_used_rx1.columns, sv_used_rx2.columns
    )
    sv_used_summary = summarize_sv_used(
        {"Receiver 1": sv_used_rx1, "Receiver 2": sv_used_rx2},
        summary_svids,
    )

    probe_result: AoaRunResult | None = None
    reference_alpha_by_epoch = pd.DataFrame()
    joint_source_aoa = pd.DataFrame()
    joint_source_branch = pd.DataFrame()
    joint_source_ambiguity = pd.DataFrame()
    spoof_branch_candidates = pd.DataFrame()
    spoof_branch_detection = pd.DataFrame()
    pairwise_convergence = pd.DataFrame()
    cluster_summary = pd.DataFrame()
    reference_alpha_deg: float | None = config.alpha_r_deg
    source_epoch_start = config.start_index
    window_quality = pd.DataFrame()
    minimum_spoof_prns = len(normalized_spoof_svids)

    if partial_spoof_enabled:
        if not normalized_spoof_svids:
            raise ValueError("spoof_svids is required when partial_spoof_enabled=True.")

        excluded_reference_svids = _combine_svids(
            config.excluded_reference_svids,
            normalized_spoof_svids,
        )
        reference_required_svids = _combine_svids(
            config.reference_required_svids,
            normalized_spoof_svids,
        )
        probe_config = replace(
            config,
            alpha_r_deg=None,
            selected_svids=normalized_spoof_svids,
            excluded_reference_svids=excluded_reference_svids,
            reference_required_svids=reference_required_svids,
        )
        probe_result = run_dataset(probe_config)

        minimum_spoof_prns = _minimum_valid_spoof_prns(
            min_valid_spoof_prns, len(normalized_spoof_svids)
        )
        valid_spoof_mask = (
            probe_result.fractional_dd_cycles.reindex(
                columns=list(normalized_spoof_svids)
            )
            .notna()
            .sum(axis=1)
            .ge(minimum_spoof_prns)
        )
        reference_alpha_by_epoch = compute_reference_alpha_by_epoch(
            probe_result.epochs,
            full_obs1.navsat,
            probe_result.reference_svid,
            baseline_azimuth_bias_deg=config.navsat_azimuth_bias_deg,
            navsat_tow_decimals=config.navpvt_tow_decimals,
            epoch_tow_column="rcvTow_rx1",
        )
        if config.reference_svid is None:
            # The automatic reference needs its NAV-SAT angle at every window epoch;
            # epochs without it (e.g. first fix after a cold start) end the window.
            has_reference_alpha = (
                reference_alpha_by_epoch.set_index("epoch_index")["reference_alpha_deg"]
                .notna()
                .reindex(valid_spoof_mask.index, fill_value=False)
            )
            valid_spoof_mask &= has_reference_alpha
        epoch_segment = longest_valid_epoch_segment(valid_spoof_mask, max_gap_epochs=max_gap_epochs)
        if not epoch_segment:
            raise ValueError(
                "No epoch window meets the requested valid spoof-PRN count and gap "
                "limits with reference NAV-SAT geometry available."
            )

        reference_alpha_window = reference_alpha_by_epoch.loc[
            reference_alpha_by_epoch["epoch_index"].isin(epoch_segment)
        ]
        missing_reference_alpha = reference_alpha_window[
            "reference_alpha_deg"
        ].isna()
        if missing_reference_alpha.all():
            raise ValueError(
                "Reference NAV-SAT AoA is unavailable inside the selected "
                "partial-spoof epoch window."
            )

        reference_alpha_deg = float(
            reference_alpha_window["reference_alpha_deg"].median()
        )
        epoch_lookup = probe_result.epochs.set_index("epoch_index")
        source_epoch_start = int(
            epoch_lookup.loc[epoch_segment[0]].get(
                "source_epoch_index", epoch_segment[0]
            )
        )

        analysis_config = replace(
            probe_config,
            reference_svid=probe_result.reference_svid,
            alpha_r_deg=reference_alpha_deg,
            start_index=source_epoch_start,
            length=len(epoch_segment),
        )
    else:
        analysis_config = config

    analysis_result = run_dataset(analysis_config)
    if save_outputs:
        save_artifacts(analysis_result, save_figures=save_figures)

    if partial_spoof_enabled:
        joint_selector = (
            select_joint_source_aoa_segments
            if min_valid_spoof_prns is not None or max_gap_epochs > 0
            else select_joint_source_aoa
        )
        selection_options = (
            {"min_valid_spoof_prns": minimum_spoof_prns}
            if joint_selector is select_joint_source_aoa_segments
            else {"return_branches": True, "return_ambiguity": True}
        )
        joint_source_aoa, joint_source_branch, joint_source_ambiguity = (
            joint_selector(
                analysis_result.candidate_aoa_deg,
                normalized_spoof_svids,
                metric=config.branch_metric,
                prior_aoa_deg=prior_aoa_deg,
                fixed_m=spoof_branch_m,
                tie_tolerance=tie_tolerance,
                **selection_options,
            )
        )
        # Include every original epoch so NaNs break plotted lines at missing data.
        window_index = analysis_result.epochs["epoch_index"].to_numpy()
        joint_source_aoa = joint_source_aoa.reindex(window_index)
        joint_source_branch = joint_source_branch.reindex(window_index)
        window_quality = analysis_result.epochs.copy()
        window_quality["valid_spoof_prn_count"] = (
            analysis_result.fractional_dd_cycles.reindex(
                index=window_index, columns=list(normalized_spoof_svids)
            ).notna().sum(axis=1).to_numpy()
        )
        window_quality["required_spoof_prn_count"] = minimum_spoof_prns
        window_quality["meets_spoof_prn_requirement"] = window_quality["valid_spoof_prn_count"].ge(minimum_spoof_prns)
        window_quality["joint_aoa_prn_count"] = joint_source_aoa.notna().sum(axis=1).to_numpy()
        spoof_branch_candidates = summarize_branch_candidates(
            analysis_result.candidate_aoa_deg, normalized_spoof_svids
        )
        pairwise_convergence = compute_pairwise_aoa_convergence(
            joint_source_aoa,
            threshold_deg=convergence_threshold_deg,
        )
        cluster_summary = summarize_converged_aoa_clusters(
            joint_source_aoa,
            threshold_deg=convergence_threshold_deg,
        )
        all_satellite_config = replace(analysis_config, selected_svids=None)
        all_satellite_result = run_dataset(all_satellite_config)
    else:
        all_satellite_result = analysis_result

    all_satellite_dd_aoa = sort_svid_columns(
        all_satellite_result.selected_aoa_deg.dropna(axis=1, how="all")
    )
    all_satellite_svids = tuple(int(column) for column in all_satellite_dd_aoa.columns)
    ephemeris_aoa_rx1 = compute_navsat_aoa_by_synced_epoch(
        all_satellite_result.epochs,
        full_obs1.navsat,
        epoch_tow_column="rcvTow_rx1",
        navsat_tow_decimals=config.navpvt_tow_decimals,
        baseline_azimuth_bias_deg=config.navsat_azimuth_bias_deg,
        selected_svids=all_satellite_svids,
    )
    ephemeris_aoa_rx2 = compute_navsat_aoa_by_synced_epoch(
        all_satellite_result.epochs,
        full_obs2.navsat,
        epoch_tow_column="rcvTow_rx2",
        navsat_tow_decimals=config.navpvt_tow_decimals,
        baseline_azimuth_bias_deg=config.navsat_azimuth_bias_deg,
        selected_svids=all_satellite_svids,
    )
    all_satellite_valid_candidates = valid_m_candidates(
        all_satellite_result.candidate_aoa_deg
    )
    all_satellite_valid_m_summary = summarize_valid_m_candidates(
        all_satellite_result.candidate_aoa_deg,
        reference_aoa=ephemeris_aoa_rx1,
        selected_branch=all_satellite_result.selected_branch,
    )
    # Authentic SVs: branch m picked per epoch against the ephemeris-based AoA (the default).
    # The constant-m selection is kept only as a comparison row in the equivalence summary.
    constant_m_aligned_aoa, constant_m_branch = select_aoa_by_reference(
        all_satellite_result.candidate_aoa_deg,
        ephemeris_aoa_rx1,
        return_branches=True,
    )
    ephemeris_aligned_aoa, ephemeris_aligned_branch_by_epoch = select_aoa_by_reference_per_epoch(
        all_satellite_result.candidate_aoa_deg,
        ephemeris_aoa_rx1,
    )
    ephemeris_aligned_aoa = sort_svid_columns(
        ephemeris_aligned_aoa.dropna(axis=1, how="all")
    )
    ephemeris_aligned_branch = summarize_per_epoch_branches(
        ephemeris_aligned_branch_by_epoch, ephemeris_aligned_aoa
    )
    best_m_all_satellite_aoa = build_best_m_all_satellite_aoa(
        all_satellite_dd_aoa,
        ephemeris_aligned_aoa,
        spoof_svids=normalized_spoof_svids,
        joint_source_aoa=joint_source_aoa if partial_spoof_enabled else None,
    )
    aoa_method_equivalence_summary = summarize_aoa_method_equivalence(
        dp_aoa=all_satellite_dd_aoa,
        dp_branch=all_satellite_result.selected_branch,
        ephemeris_aligned_aoa=sort_svid_columns(constant_m_aligned_aoa.dropna(axis=1, how="all")),
        ephemeris_aoa=ephemeris_aoa_rx1,
        ephemeris_aligned_branch=constant_m_branch,
        per_epoch_aligned_aoa=ephemeris_aligned_aoa,
    )

    if partial_spoof_enabled and not joint_source_branch.empty:
        selected_branch_by_svid = {
            int(svid): int(joint_source_branch[svid].dropna().mode().iloc[0])
            for svid in joint_source_branch.columns
            if joint_source_branch[svid].notna().any()
        }
        alternative_branch_by_svid = None
        chosen_by = "unresolved"
        if not joint_source_ambiguity.empty:
            first = joint_source_ambiguity.iloc[0]
            chosen_by = str(first["chosen_by"])
            if np.isfinite(float(first["alternative_shift"])):
                shift = int(first["alternative_shift"])
                alternative_branch_by_svid = {
                    svid: branch + shift for svid, branch in selected_branch_by_svid.items()
                }
        spoof_branch_detection = summarize_spoof_branch_detection(
            spoof_branch_candidates,
            ephemeris_aoa_rx1,
            selected_branch_by_svid=selected_branch_by_svid,
            alternative_branch_by_svid=alternative_branch_by_svid,
            chosen_by=chosen_by,
            threshold_deg=detection_threshold_deg,
        )

    run_summary = _paper_scenario_summary(
        config=config,
        partial_spoof_enabled=partial_spoof_enabled,
        spoof_svids=normalized_spoof_svids,
        analysis_result=analysis_result,
        all_satellite_result=all_satellite_result,
        reference_alpha_deg=reference_alpha_deg,
        source_epoch_start=source_epoch_start,
        sv_used_summary=sv_used_summary,
        prior_aoa_deg=prior_aoa_deg,
        joint_source_ambiguity=joint_source_ambiguity,
    )

    run_summary["spoof_branch_m"] = spoof_branch_m
    run_summary["min_valid_spoof_prns"] = minimum_spoof_prns
    run_summary["max_gap_epochs"] = max_gap_epochs
    run_summary["analysis_span_s"] = float(analysis_result.epochs["rcvTow_rx1"].iloc[-1] - analysis_result.epochs["rcvTow_rx1"].iloc[0])
    if not window_quality.empty:
        valid_epoch_count = int(window_quality["meets_spoof_prn_requirement"].sum())
        run_summary["valid_spoof_epoch_count"] = valid_epoch_count
        run_summary["missing_spoof_epoch_count"] = len(window_quality) - valid_epoch_count

    scenario_result = PaperScenarioResult(
        config=config,
        partial_spoof_enabled=partial_spoof_enabled,
        spoof_svids=normalized_spoof_svids,
        probe_result=probe_result,
        analysis_result=analysis_result,
        all_satellite_result=all_satellite_result,
        reference_alpha_by_epoch=reference_alpha_by_epoch,
        joint_source_aoa_deg=joint_source_aoa,
        joint_source_branch=joint_source_branch,
        pairwise_convergence=pairwise_convergence,
        cluster_summary=cluster_summary,
        all_satellite_dd_method_aoa_deg=all_satellite_dd_aoa,
        ephemeris_aoa_rx1_deg=ephemeris_aoa_rx1,
        ephemeris_aoa_rx2_deg=ephemeris_aoa_rx2,
        all_satellite_valid_m_candidates=all_satellite_valid_candidates,
        all_satellite_valid_m_summary=all_satellite_valid_m_summary,
        ephemeris_aligned_dd_method_aoa_deg=ephemeris_aligned_aoa,
        ephemeris_aligned_branch=ephemeris_aligned_branch,
        ephemeris_aligned_branch_by_epoch=ephemeris_aligned_branch_by_epoch,
        best_m_all_satellite_aoa_deg=best_m_all_satellite_aoa,
        aoa_method_equivalence_summary=aoa_method_equivalence_summary,
        sv_used_rx1=sv_used_rx1,
        sv_used_rx2=sv_used_rx2,
        sv_used_summary=sv_used_summary,
        run_summary=run_summary,
        joint_source_ambiguity=joint_source_ambiguity,
        spoof_branch_candidates=spoof_branch_candidates,
        spoof_branch_detection=spoof_branch_detection,
        prior_aoa_deg=prior_aoa_deg,
        window_quality=window_quality,
    )
    if save_outputs:
        scenario_result.output_paths = save_paper_scenario_artifacts(
            scenario_result, output_dir=output_dir, save_figures=save_figures
        )
    return scenario_result


def save_paper_scenario_artifacts(
    scenario: PaperScenarioResult,
    output_dir: str | Path | None = None,
    *, save_figures: bool = True, figure_formats: Sequence[str] = ("png",),
) -> dict[str, Path]:
    """Save paper-scenario CSVs and figures that are outside the base DD run."""
    directory = Path(output_dir) if output_dir is not None else scenario.config.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    csv_specs = [
        ("paper_scenario_run_summary.csv", scenario.run_summary, False),
        ("paper_window_quality.csv", scenario.window_quality, False),
        (
            "paper_reference_alpha_by_epoch.csv",
            scenario.reference_alpha_by_epoch,
            False,
        ),
        (
            "paper_spoof_joint_source_aoa_deg.csv",
            scenario.joint_source_aoa_deg,
            True,
        ),
        (
            "paper_spoof_joint_source_branch.csv",
            scenario.joint_source_branch,
            True,
        ),
        ("paper_spoof_pairwise_convergence.csv", scenario.pairwise_convergence, False),
        ("paper_spoof_cluster_summary.csv", scenario.cluster_summary, False),
        (
            "paper_spoof_joint_source_ambiguity.csv",
            scenario.joint_source_ambiguity,
            False,
        ),
        ("paper_spoof_branch_candidates.csv", scenario.spoof_branch_candidates, False),
        ("paper_spoof_branch_detection.csv", scenario.spoof_branch_detection, False),
        (
            "paper_all_satellite_dd_method_aoa_deg.csv",
            scenario.all_satellite_dd_method_aoa_deg,
            True,
        ),
        (
            "paper_all_satellite_fractional_dd_cycles.csv",
            sort_svid_columns(
                scenario.all_satellite_result.fractional_dd_cycles.dropna(
                    axis=1, how="all"
                )
            ),
            True,
        ),
        (
            "paper_all_satellite_ephemeris_aoa_rx1_deg.csv",
            scenario.ephemeris_aoa_rx1_deg,
            True,
        ),
        (
            "paper_all_satellite_ephemeris_aoa_rx2_deg.csv",
            scenario.ephemeris_aoa_rx2_deg,
            True,
        ),
        (
            "paper_all_satellite_valid_m_candidates.csv",
            scenario.all_satellite_valid_m_candidates,
            False,
        ),
        (
            "paper_all_satellite_valid_m_summary.csv",
            scenario.all_satellite_valid_m_summary,
            False,
        ),
        (
            "paper_all_satellite_ephemeris_aligned_dd_method_aoa_deg.csv",
            scenario.ephemeris_aligned_dd_method_aoa_deg,
            True,
        ),
        (
            "paper_all_satellite_ephemeris_aligned_branch.csv",
            scenario.ephemeris_aligned_branch,
            False,
        ),
        (
            "paper_all_satellite_ephemeris_aligned_branch_by_epoch.csv",
            scenario.ephemeris_aligned_branch_by_epoch,
            True,
        ),
        (
            "paper_all_satellite_dd_method_aoa_best_m_deg.csv",
            scenario.best_m_all_satellite_aoa_deg,
            True,
        ),
        (
            "paper_all_satellite_aoa_method_equivalence_summary.csv",
            scenario.aoa_method_equivalence_summary,
            False,
        ),
        ("paper_receiver1_sv_used_all_epochs.csv", scenario.sv_used_rx1, True),
        ("paper_receiver2_sv_used_all_epochs.csv", scenario.sv_used_rx2, True),
        ("paper_sv_used_summary.csv", scenario.sv_used_summary, False),
    ]
    for filename, frame, include_index in csv_specs:
        if frame.empty and filename != "paper_reference_alpha_by_epoch.csv":
            continue
        path = directory / filename
        _write_csv(frame, path, index=include_index)
        artifacts[filename] = path

    if not save_figures:
        scenario.output_paths = artifacts
        return artifacts

    plot_specs = [
        (
            "paper_fractional_dd.png",
            lambda: plot_fractional_dd(scenario.analysis_result),
        ),
        (
            "paper_spoof_joint_source_dd_method_aoa.png",
            lambda: plot_aoa_tracks(
                scenario.joint_source_aoa_deg,
                title="Spoof-satellite joint-source DD-method AoA",
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_spoof_branch_summary.png",
            lambda: plot_spoof_branch_summary(
                scenario.spoof_branch_detection,
                prior_aoa_deg=scenario.prior_aoa_deg,
                cluster_median_aoa_deg=(
                    float(scenario.cluster_summary["median_aoa_deg"].iloc[0])
                    if not scenario.cluster_summary.empty
                    else None
                ),
            ),
        ),
        (
            "paper_all_satellite_dd_method_aoa.png",
            lambda: plot_aoa_tracks(
                scenario.all_satellite_dd_method_aoa_deg,
                title="All-satellite DD-method AoA",
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_all_satellite_fractional_dd.png",
            lambda: plot_fractional_dd(
                scenario.all_satellite_result,
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_all_satellite_dd_method_aoa_best_m.png",
            lambda: plot_aoa_tracks(
                scenario.best_m_all_satellite_aoa_deg,
                title=(
                    "All-satellite DD-method AoA "
                    "(authentic: ephemeris-aligned; spoof: joint source)"
                ),
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_all_satellite_ephemeris_aoa_rx1.png",
            lambda: plot_aoa_tracks(
                scenario.ephemeris_aoa_rx1_deg,
                title="All-satellite ephemeris-based AoA (Receiver 1 NAV-SAT)",
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_all_satellite_ephemeris_aoa_rx2.png",
            lambda: plot_aoa_tracks(
                scenario.ephemeris_aoa_rx2_deg,
                title="All-satellite ephemeris-based AoA (Receiver 2 NAV-SAT)",
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_all_satellite_valid_m_candidates.png",
            lambda: plot_valid_m_candidates(
                scenario.all_satellite_result.candidate_aoa_deg,
                reference_aoa=scenario.ephemeris_aoa_rx1_deg,
                title="All valid DD AoA candidates by integer branch",
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_all_satellite_ephemeris_aligned_vs_ephemeris_aoa.png",
            lambda: plot_aoa_method_comparison(
                scenario.ephemeris_aligned_dd_method_aoa_deg,
                scenario.ephemeris_aoa_rx1_deg,
                title="Per-epoch ephemeris-aligned DD AoA vs ephemeris AoA",
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_sv_used_all_epochs_by_receiver.png",
            lambda: plot_sv_used_timeline(
                {"Receiver 1": scenario.sv_used_rx1, "Receiver 2": scenario.sv_used_rx2},
                title="u-blox NAV-SAT svUsed Over All NAV-SAT Epochs",
                highlight_svids=scenario.spoof_svids,
            ),
        ),
    ]
    plt = _import_pyplot()
    for filename, plotter in plot_specs:
        if (
            filename == "paper_spoof_joint_source_dd_method_aoa.png"
            and scenario.joint_source_aoa_deg.empty
        ):
            continue
        if (
            filename == "paper_spoof_branch_summary.png"
            and scenario.spoof_branch_detection.empty
        ):
            continue
        fig = plotter()
        for extension in figure_formats:
            path = directory / f"{Path(filename).stem}.{extension}"
            fig.savefig(path, dpi=180)
            artifacts[path.name] = path
        plt.close(fig)

    scenario.output_paths = artifacts
    return artifacts
