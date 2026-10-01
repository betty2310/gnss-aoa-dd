"""Small experiment API shared by the experiment and figure-export notebooks.

Experiments save tables and a reproducible configuration. Figure export is a
separate step that replays that configuration and verifies the saved results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aoa_pipeline import (
    BaselineAzimuthFit, CaptureProbe, DatasetConfig, PaperScenarioResult,
    compute_navsat_geometry_by_synced_epoch, compute_single_difference,
    fit_baseline_azimuth, probe_capture, run_paper_scenario,
    save_artifacts, save_paper_scenario_artifacts,
    summarize_pairwise_single_difference_collapse,
)


@dataclass(frozen=True)
class ExperimentConfig:
    receiver1_path: str | Path
    receiver2_path: str | Path
    spoof_svids: tuple[int, ...]
    name: str = "experiment"
    output_root: str | Path = "outputs/aoa_experiments"
    single_difference_order: str = "rx2-rx1"
    baseline_azimuth_deg: float = 0.0
    baseline_over_wavelength: float = 1.0
    reference_svid: int | None = None
    spoof_branch_m: int | None = None
    prior_aoa_deg: float | None = None
    angle_reference: str = "configured"
    expected_aoa_deg: float | None = None
    convergence_threshold_deg: float = 5.0
    detection_threshold_deg: float = 10.0
    min_valid_spoof_prns: int | None = None
    max_gap_epochs: int = 0
    max_epochs: int | None = None
    start_epoch: int = 0
    min_window_epochs: int = 2

    def __post_init__(self):
        for field in ("receiver1_path", "receiver2_path", "output_root"):
            object.__setattr__(self, field, Path(getattr(self, field)).expanduser().resolve())
        object.__setattr__(self, "spoof_svids", tuple(self.spoof_svids))
        if not self.spoof_svids or any(
            isinstance(prn, bool) or not isinstance(prn, int) or not 1 <= prn <= 32
            for prn in self.spoof_svids
        ) or len(set(self.spoof_svids)) != len(self.spoof_svids):
            raise ValueError("spoof_svids must contain unique GPS PRNs from 1 to 32.")
        if self.receiver1_path == self.receiver2_path:
            raise ValueError("Receiver 1 and Receiver 2 must use different UBX files.")
        if self.single_difference_order not in ("rx1-rx2", "rx2-rx1"):
            raise ValueError("single_difference_order must be rx1-rx2 or rx2-rx1.")
        if not np.isfinite(self.baseline_over_wavelength) or self.baseline_over_wavelength <= 0:
            raise ValueError("baseline_over_wavelength must be positive and finite.")
        if not np.isfinite(self.baseline_azimuth_deg):
            raise ValueError("baseline_azimuth_deg must be finite.")
        if self.reference_svid in self.spoof_svids:
            raise ValueError("The reference satellite must not be in the spoof group.")
        if self.spoof_branch_m is not None and self.prior_aoa_deg is not None:
            raise ValueError("Declare either spoof_branch_m or prior_aoa_deg, not both.")
        if self.angle_reference not in ("configured", "opposite", "closest_to_expected"):
            raise ValueError("angle_reference must be configured, opposite, or closest_to_expected.")
        if self.expected_aoa_deg is not None and (
            not np.isfinite(self.expected_aoa_deg) or not 0 <= self.expected_aoa_deg <= 180
        ):
            raise ValueError("expected_aoa_deg must be between 0 and 180 degrees.")
        if self.angle_reference == "closest_to_expected" and self.expected_aoa_deg is None:
            raise ValueError("closest_to_expected requires expected_aoa_deg.")
        for field, minimum in (("start_epoch", 0), ("min_window_epochs", 1)):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{field} must be an integer of at least {minimum}.")
        if not self.name or Path(self.name).is_absolute() or ".." in Path(self.name).parts:
            raise ValueError("name must be a nonempty relative run name.")

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.name

    def to_dict(self) -> dict:
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self).items()}

    def dataset_config(self, usable_spoof_svids: tuple[int, ...]) -> DatasetConfig:
        return DatasetConfig(
            name=self.name, receiver1_path=self.receiver1_path,
            receiver2_path=self.receiver2_path, output_root=self.output_root,
            reference_svid=self.reference_svid, selected_svids=usable_spoof_svids,
            excluded_reference_svids=self.spoof_svids,
            reference_required_svids=usable_spoof_svids, start_index=self.start_epoch,
            length=self.max_epochs, sync_tow_decimals=0,
            use_navpvt_time_correction=True,
            single_difference_order=self.single_difference_order,
            baseline_over_wavelength=self.baseline_over_wavelength,
            branch_metric="direction_cosine",
            navsat_azimuth_bias_deg=self.baseline_azimuth_deg,
        )


@dataclass
class ExperimentChecks:
    config: ExperimentConfig
    probe: CaptureProbe
    rig: BaselineAzimuthFit
    phase_collapse: pd.DataFrame
    orientation_hypotheses: pd.DataFrame = field(default_factory=pd.DataFrame)


def check_experiment(config: ExperimentConfig) -> ExperimentChecks:
    """Check tracking and baseline consistency before solving for AoA."""
    probe = probe_capture(
        config.receiver1_path, config.receiver2_path,
        spoof_svids=config.spoof_svids,
        min_valid_spoof_prns=config.min_valid_spoof_prns,
        max_gap_epochs=config.max_gap_epochs,
        min_window_epochs=config.min_window_epochs,
    )
    probe.raise_for_problems()
    sd = compute_single_difference(probe.carrier_phase, order=config.single_difference_order)
    elevation, azimuth = compute_navsat_geometry_by_synced_epoch(
        probe.synced.epochs, probe.observations[0].navsat, epoch_tow_column="rcvTow_rx1",
    )
    rig = fit_baseline_azimuth(
        sd, elevation, azimuth, exclude_svids=config.spoof_svids,
        baseline_over_wavelength=config.baseline_over_wavelength,
        configured_azimuth_deg=config.baseline_azimuth_deg,
    )
    collapse = summarize_pairwise_single_difference_collapse(sd, probe.usable_spoof_svids)
    return ExperimentChecks(config, probe, rig, collapse)


def _solve_configured(config: ExperimentConfig, usable: tuple[int, ...]) -> PaperScenarioResult:
    return run_paper_scenario(
        config.dataset_config(usable), partial_spoof_enabled=True,
        spoof_svids=usable, convergence_threshold_deg=config.convergence_threshold_deg,
        detection_threshold_deg=config.detection_threshold_deg,
        prior_aoa_deg=config.prior_aoa_deg, spoof_branch_m=config.spoof_branch_m,
        min_valid_spoof_prns=config.min_valid_spoof_prns,
        max_gap_epochs=config.max_gap_epochs, save_outputs=False,
        save_figures=False,
    )


def opposite_reference(config: ExperimentConfig) -> ExperimentConfig:
    """Reverse the directed reference while preserving the physical wiring hypothesis."""
    return replace(
        config,
        single_difference_order=("rx1-rx2" if config.single_difference_order == "rx2-rx1" else "rx2-rx1"),
        baseline_azimuth_deg=(config.baseline_azimuth_deg + 180) % 360,
        prior_aoa_deg=None if config.prior_aoa_deg is None else 180 - config.prior_aoa_deg,
        spoof_branch_m=None if config.spoof_branch_m is None else -config.spoof_branch_m,
        angle_reference="configured",
    )


def _use_opposite_reference(config: ExperimentConfig, group_angle: float) -> bool:
    if config.angle_reference == "opposite":
        return True
    if config.angle_reference == "closest_to_expected":
        expected = config.expected_aoa_deg
        # A tie retains the configured reference. This never changes the measured angle
        # to the expected value or selects a different integer-ambiguity branch.
        return abs(180 - group_angle - expected) < abs(group_angle - expected)
    return False


def _solve(config: ExperimentConfig, usable: tuple[int, ...], *, save_outputs: bool) -> PaperScenarioResult:
    scenario = _solve_configured(config, usable)
    group_angle = float(scenario.joint_source_aoa_deg.median(axis=1).median())
    if _use_opposite_reference(config, group_angle):
        reversed_scenario = _solve_configured(opposite_reference(config), usable)
        # Validate a change of reference, not a new branch chosen to fit an expectation.
        np.testing.assert_allclose(
            reversed_scenario.joint_source_aoa_deg.to_numpy(),
            180 - scenario.joint_source_aoa_deg.to_numpy(), rtol=0, atol=1e-8, equal_nan=True,
        )
        scenario = reversed_scenario
    if save_outputs:
        save_artifacts(scenario.analysis_result, save_figures=False)
        save_paper_scenario_artifacts(scenario, save_figures=False)
    return scenario


def angle_reference_summary(config: ExperimentConfig, scenario: PaperScenarioResult) -> dict:
    """Distinguish a reporting-reference preference from physical orientation evidence."""
    reversed_reference = scenario.config.single_difference_order != config.single_difference_order
    angle = float(scenario.joint_source_aoa_deg.median(axis=1).median())
    azimuth = float(scenario.config.navsat_azimuth_bias_deg) % 360
    direction = {0.0: "north", 90.0: "east", 180.0: "south", 270.0: "west"}.get(
        azimuth, f"azimuth {azimuth:g} degrees"
    )
    return {
        "mode": config.angle_reference,
        "selected_reference": "opposite" if reversed_reference else "configured",
        "selected_by": "expected_angle_preference" if config.angle_reference == "closest_to_expected" else "explicit_reference",
        "expected_aoa_deg": config.expected_aoa_deg,
        "expected_reference_direction": "unspecified",
        "reported_aoa_deg": angle,
        "configured_reference_aoa_deg": 180 - angle if reversed_reference else angle,
        "other_reference_aoa_deg": 180 - angle,
        "effective_single_difference_order": scenario.config.single_difference_order,
        "effective_baseline_azimuth_deg": azimuth,
        "reference_direction_under_configured_hypothesis": direction,
        "physical_orientation_confirmed": False,
        "note": "The expected angle selects only the directed reporting reference. It does not establish receiver placement or resolve the integer branch ambiguity.",
    }


def compare_receiver_orientations(
    config: ExperimentConfig, checks: ExperimentChecks, scenario: PaperScenarioResult,
) -> pd.DataFrame:
    """Audit both wiring hypotheses independently of the expected angle preference.

    The second hypothesis reverses geometry with fixed subtraction order. In contrast,
    opposite_reference reverses both and describes the same physical hypothesis.
    """
    other = _solve_configured(replace(
        config, baseline_azimuth_deg=(config.baseline_azimuth_deg + 180) % 360,
        angle_reference="configured", prior_aoa_deg=None, spoof_branch_m=None,
    ), checks.probe.usable_spoof_svids)
    rows = []
    for label, result, residual in [
        ("configured wiring", scenario, checks.rig.configured_residual_cycles),
        ("opposite wiring", other, checks.rig.opposite_residual_cycles),
    ]:
        angle = float(result.joint_source_aoa_deg.median(axis=1).median())
        alternative = float(result.spoof_branch_detection.alternative_aoa_deg.median())
        azimuth = float(result.config.navsat_azimuth_bias_deg) % 360
        from_configured_axis = np.isclose(azimuth, config.baseline_azimuth_deg % 360)
        rows.append({
            "wiring_hypothesis": label,
            "rx1_to_rx2_azimuth_deg": (
                config.baseline_azimuth_deg
                + (180 if config.single_difference_order == "rx2-rx1" else 0)
                + (180 if label == "opposite wiring" else 0)
            ) % 360,
            "aoa_from_configured_axis_deg": angle if from_configured_axis else 180 - angle,
            "aoa_from_opposite_axis_deg": 180 - angle if from_configured_axis else angle,
            "other_branch_from_configured_axis_deg": alternative if from_configured_axis else 180 - alternative,
            "other_branch_from_opposite_axis_deg": 180 - alternative if from_configured_axis else alternative,
            "authentic_residual_cycles": residual,
            "branch_selection": ",".join(sorted(set(result.joint_source_ambiguity.chosen_by))),
        })
    return pd.DataFrame(rows)


def validate_experiment(scenario: PaperScenarioResult, declared_spoof_svids: tuple[int, ...]) -> None:
    """Validate result coverage without forcing an expected angle or branch."""
    usable = set(scenario.spoof_svids)
    if scenario.analysis_result.epochs.empty or scenario.joint_source_aoa_deg.empty:
        raise RuntimeError("The experiment produced no AoA epochs.")
    if set(scenario.joint_source_aoa_deg.columns) != usable:
        raise RuntimeError("The AoA table does not contain every usable spoof PRN.")
    if set(scenario.spoof_branch_detection.svid) != usable:
        raise RuntimeError("The branch table does not contain every usable spoof PRN.")
    if scenario.analysis_result.reference_svid in declared_spoof_svids:
        raise RuntimeError("The reference is a declared spoof PRN.")
    choices = set(scenario.joint_source_ambiguity.chosen_by)
    if not choices <= {"data", "prior", "unresolved", "explicit_m"}:
        raise RuntimeError(f"Unexpected branch-selection provenance: {choices}")
    if scenario.prior_aoa_deg is None and "prior" in choices:
        raise RuntimeError("A branch was attributed to an undeclared prior.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_experiment(config: ExperimentConfig, checks: ExperimentChecks | None = None) -> PaperScenarioResult:
    """Run the scientific pipeline and save tables plus experiment.json, without exporting figures."""
    checks = check_experiment(config) if checks is None else checks
    if checks.config != config:
        raise ValueError("The inputs changed after the data check; run the check again.")
    scenario, orientation = _run_and_save(config, checks)
    if not orientation.empty:
        checks.orientation_hypotheses = orientation
    return scenario


def _run_and_save(
    config: ExperimentConfig, checks: ExperimentChecks,
) -> tuple[PaperScenarioResult, pd.DataFrame]:
    """Solve one epoch range and save its tables plus experiment.json."""
    scenario = _solve(config, checks.probe.usable_spoof_svids, save_outputs=True)
    validate_experiment(scenario, config.spoof_svids)
    orientation = pd.DataFrame()
    if config.angle_reference == "closest_to_expected":
        orientation = compare_receiver_orientations(config, checks, scenario)
    output = config.output_dir
    checks.probe.message_inventory.to_csv(output / "paper_capture_message_inventory.csv")
    checks.probe.spoof_phase_summary.to_csv(output / "paper_capture_spoof_phase.csv", index=False)
    checks.rig.to_frame().to_csv(output / "paper_rig_check.csv", index=False)
    checks.phase_collapse.to_csv(output / "paper_spoof_single_difference_collapse.csv", index=False)
    sources = [config.receiver1_path, config.receiver2_path,
               Path(__file__).resolve(), Path(__file__).with_name("aoa_pipeline.py").resolve()]
    data_files = {
        "joint_aoa": output / "paper_spoof_joint_source_aoa_deg.csv",
        "joint_branch": output / "paper_spoof_joint_source_branch.csv",
        "fractional_dd": output / "paper_all_satellite_fractional_dd_cycles.csv",
    }
    if not orientation.empty:
        data_files["orientation_hypotheses"] = output / "paper_receiver_orientation_hypotheses.csv"
        orientation.to_csv(data_files["orientation_hypotheses"], index=False)
    manifest = {
        "schema_version": 1, "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(), "usable_spoof_svids": list(scenario.spoof_svids),
        "omitted_spoof_svids": sorted(set(config.spoof_svids) - set(scenario.spoof_svids)),
        "reference_svid": scenario.analysis_result.reference_svid,
        "epochs": len(scenario.analysis_result.epochs),
        "group_median_aoa_deg": float(scenario.joint_source_aoa_deg.median(axis=1).median()),
        "chosen_by": sorted(set(scenario.joint_source_ambiguity.chosen_by)),
        "angle_reference": angle_reference_summary(config, scenario),
        "sources_sha256": {str(path): _sha256(path) for path in sources},
        "result_files": {name: str(path) for name, path in data_files.items()},
        "result_sha256": {name: _sha256(path) for name, path in data_files.items()},
        "warnings": checks.probe.warnings,
        "angle_convention": "angle toward the source, relative to the effective directed baseline recorded in angle_reference, in [0, 180] degrees",
    }
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return scenario, orientation


@dataclass
class ExperimentWindows:
    """Every spoof-group window of one capture, each solved as its own experiment."""

    config: ExperimentConfig
    summary: pd.DataFrame
    configs: dict[int, ExperimentConfig]
    scenarios: dict[int, PaperScenarioResult]
    orientation_hypotheses: dict[int, pd.DataFrame]

    def aoa_by_capture_epoch(self) -> pd.DataFrame:
        """Joint-source AoA of all windows on one capture-epoch axis; gaps stay NaN."""
        parts = []
        for window, scenario in self.scenarios.items():
            epochs = scenario.analysis_result.epochs.set_index("epoch_index")
            aoa = scenario.joint_source_aoa_deg.copy()
            aoa.index = epochs.loc[aoa.index, "source_epoch_index"].astype(int).to_numpy()
            parts.append(aoa)
        if not parts:
            return pd.DataFrame()
        combined = pd.concat(parts).sort_index()
        combined = combined.reindex(range(int(combined.index.min()), int(combined.index.max()) + 1))
        combined.index.name = "capture_epoch"
        return combined


def window_configs(config: ExperimentConfig, checks: ExperimentChecks) -> dict[int, ExperimentConfig]:
    """One configuration per solvable window inside the configured epoch range."""
    first = config.start_epoch
    last = None if config.max_epochs is None else first + config.max_epochs - 1
    configs = {}
    for row in checks.probe.spoof_windows.itertuples(index=False):
        if not row.solvable:
            continue
        start = max(int(row.start_epoch), first)
        end = int(row.end_epoch) if last is None else min(int(row.end_epoch), last)
        if end - start + 1 < config.min_window_epochs:
            continue
        configs[int(row.window)] = replace(
            config, name=f"{config.name}/window_{int(row.window):02d}",
            start_epoch=start, max_epochs=end - start + 1,
        )
    return configs


def run_experiment_windows(config: ExperimentConfig, checks: ExperimentChecks | None = None) -> ExperimentWindows:
    """Solve every spoof-group window of a long capture instead of only the longest one.

    Each window is an independent experiment (own reference satellite, anchor and
    branch decision; no selection crosses a reacquisition) saved under
    ``output_dir/window_NN`` with its own experiment.json, so figure export works
    per window. ``output_dir/windows.csv`` lists every window, including skipped
    and failed ones.
    """
    checks = check_experiment(config) if checks is None else checks
    if checks.config != config:
        raise ValueError("The inputs changed after the data check; run the check again.")
    configs = window_configs(config, checks)
    scenarios: dict[int, PaperScenarioResult] = {}
    orientations: dict[int, pd.DataFrame] = {}
    rows = []
    for window in checks.probe.spoof_windows.itertuples(index=False):
        number = int(window.window)
        row = {
            "window": number, "start_tow": window.start_tow, "end_tow": window.end_tow,
            "capture_epochs": f"{window.start_epoch}-{window.end_epoch}",
            "valid_epochs": window.valid_epochs, "status": "skipped", "note": "",
        }
        if number not in configs:
            row["note"] = f"fewer than {config.min_window_epochs} valid epochs in the configured range"
            rows.append(row)
            continue
        window_config = configs[number]
        try:
            scenario, orientation = _run_and_save(window_config, checks)
        except (ValueError, RuntimeError) as error:
            row.update(status="failed", note=str(error))
            rows.append(row)
            continue
        scenarios[number], orientations[number] = scenario, orientation
        reference = angle_reference_summary(window_config, scenario)
        detection = scenario.spoof_branch_detection
        epochs = scenario.analysis_result.epochs
        row.update(
            status="solved", epochs=len(epochs),
            start_tow=float(epochs["rcvTow_rx1"].iloc[0]),
            end_tow=float(epochs["rcvTow_rx1"].iloc[-1]),
            capture_epochs=f"{int(epochs['source_epoch_index'].iloc[0])}-{int(epochs['source_epoch_index'].iloc[-1])}",
            reference_svid=scenario.analysis_result.reference_svid,
            group_aoa_deg=reference["reported_aoa_deg"],
            group_spread_deg=float(scenario.joint_source_aoa_deg.std(axis=1).median()),
            other_branch_deg=float(detection["alternative_aoa_deg"].median()),
            chosen_by=",".join(sorted(set(scenario.joint_source_ambiguity.chosen_by))),
            flagged_all_branches=int(detection["spoof_flag_all_branches"].sum()),
            selected_reference=reference["selected_reference"],
            output_dir=str(window_config.output_dir),
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.output_dir / "windows.csv", index=False)
    return ExperimentWindows(config, summary, configs, scenarios, orientations)


def replay_experiment(manifest_path: str | Path) -> tuple[ExperimentConfig, PaperScenarioResult]:
    """Recompute an unchanged saved run for export and compare with its original tables.

    No notebook kernel state or executable pickle is required. Changed inputs,
    solver code, or saved result tables require a new experiment run.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "completed":
        raise ValueError("Expected a completed experiment.json from AOA_Paper.ipynb.")
    for name, expected in manifest["sources_sha256"].items():
        path = Path(name)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Experiment input or solver changed: {path}. Rerun AOA_Paper.ipynb.")
    for name, filename in manifest["result_files"].items():
        if _sha256(Path(filename)) != manifest["result_sha256"][name]:
            raise ValueError(f"Saved experiment result changed: {filename}. Rerun the experiment.")
    config = ExperimentConfig(**manifest["config"])
    scenario = _solve(config, tuple(manifest["usable_spoof_svids"]), save_outputs=False)
    validate_experiment(scenario, config.spoof_svids)
    expected_frames = {
        "joint_aoa": scenario.joint_source_aoa_deg,
        "joint_branch": scenario.joint_source_branch,
        "fractional_dd": scenario.all_satellite_result.fractional_dd_cycles.dropna(axis=1, how="all"),
    }
    for name, frame in expected_frames.items():
        saved = pd.read_csv(manifest["result_files"][name], index_col=0)
        if list(map(str, frame.columns)) != list(saved.columns) or list(frame.index) != list(saved.index):
            raise RuntimeError(f"Replayed {name} has different satellites or epochs.")
        np.testing.assert_allclose(saved.to_numpy(), frame.to_numpy(), rtol=0, atol=1e-10, equal_nan=True)
    return config, scenario


def result_table(scenario: PaperScenarioResult) -> pd.DataFrame:
    """Compact per-PRN results; keep missing geometry and branch provenance visible."""
    columns = {
        "svid": "PRN", "selected_aoa_deg": "DD AoA (deg)",
        "selected_std_deg": "Std (deg)", "alternative_aoa_deg": "Other branch (deg)",
        "ephemeris_aoa_deg": "Ephemeris (deg)", "selected_epochs": "DD epochs",
        "ephemeris_epochs": "Geometry epochs", "chosen_by": "Branch selection",
        "spoof_flag_all_branches": "Flag on all branches",
    }
    return scenario.spoof_branch_detection[list(columns)].rename(columns=columns)
