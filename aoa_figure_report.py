"""Publication figures and diagnostics, separate from the experiment notebook."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from dataclasses import replace

from aoa_pipeline import (
    PaperScenarioResult, compute_navsat_aoa_by_synced_epoch, read_ubx_observations, run_dataset,
    save_artifacts, save_paper_scenario_artifacts, select_aoa_by_reference_per_epoch,
)


DESIGN_PATH = Path(__file__).with_name("aoa_figure_style.json")
# Kept for callers that previously used this constant for source hashing.
DESIGN_NOTEBOOK = DESIGN_PATH
DESIGN_TAG = "paper-figure-report-design"
FIGURE_STEMS = (
    "figure_01_fractional_double_difference_mixed",
    "figure_02_alpha_i_ephemeris_mixed",
    "figure_03_alpha_i_proposed_best_m_mixed",
    "figure_04_alpha_i_side_by_side_best_m",
)
AGREEMENT_THRESHOLDS_DEG = (5, 7, 10)
AGREEMENT_TABLE_STEM = "table_01_authentic_angular_agreement"
CAPTURE_PRESETS = {
    "v4": {"single_difference_order": "rx2-rx1", "spoof_branch_m": 0},
    "v5": {"single_difference_order": "rx2-rx1", "spoof_branch_m": -1},
    # Authentic-PRN rig checks establish the reversed COM3/COM4 baseline polarity.
    "v8": {"single_difference_order": "rx1-rx2", "spoof_branch_m": None},
    "v9": {"single_difference_order": "rx1-rx2", "spoof_branch_m": None},
}


def read_spoof_svids(folder: str | Path) -> tuple[int, ...]:
    """Read only the explicit spoof-group declaration, not numbers in note tables."""
    path = Path(folder) / "note.txt"
    note = path.read_text(encoding="utf-8")
    match = re.search(r"^\s*spoofs?\s+group\s*:\s*([\d, \t]+)$", note, re.I | re.M)
    if match is None:
        raise ValueError(f"No 'spoofs group: ...' declaration in {path}")
    values = tuple(sorted({int(value) for value in re.findall(r"\d+", match[1])}))
    if not values or not all(1 <= value <= 32 for value in values):
        raise ValueError(f"Invalid GPS PRNs in {path}: {values}")
    return values


def capture_report_preset(folder: str | Path) -> dict:
    folder = Path(folder)
    preset = {"single_difference_order": "rx2-rx1", "spoof_branch_m": None}
    if folder.parent.name == "sp-partial-28-08":
        preset.update(CAPTURE_PRESETS.get(folder.name, {}))
    return {**preset, "spoof_svids": read_spoof_svids(folder)}


def load_figure_design(path: str | Path = DESIGN_PATH) -> dict:
    """Read standalone figure style; old tagged notebook designs remain readable."""
    path = Path(path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".json":
        return notebook
    cells = [cell for cell in notebook["cells"] if DESIGN_TAG in cell.get("metadata", {}).get("tags", [])]
    if len(cells) != 1:
        raise ValueError(f"Expected one '{DESIGN_TAG}' cell in {path}, found {len(cells)}")
    tree = ast.parse("".join(cells[0]["source"]))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PAPER_FIGURE_DESIGN"
            for target in statement.targets
        ):
            return ast.literal_eval(statement.value)
    raise ValueError(f"PAPER_FIGURE_DESIGN literal is missing from {path}")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_angular_errors(errors: pd.DataFrame) -> pd.DataFrame:
    """Per-PRN agreement and a pooled overall rate, excluding missing pairs."""
    def row(label, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        count = len(values)
        result = {
            "target_prn": str(label), "paired_epochs": count,
            "mean_abs_error_deg": float(values.mean()) if count else np.nan,
            "median_abs_error_deg": float(np.median(values)) if count else np.nan,
            "max_abs_error_deg": float(values.max()) if count else np.nan,
        }
        for threshold in AGREEMENT_THRESHOLDS_DEG:
            passed = int(np.count_nonzero(values <= threshold))
            result[f"count_le_{threshold}deg"] = passed
            result[f"p_le_{threshold}deg_pct"] = 100.0 * passed / count if count else np.nan
        return result

    rows = [row(prn, errors[prn]) for prn in errors]
    rows.append(row("Overall", errors.to_numpy().ravel()))
    return pd.DataFrame(rows)


def authentic_agreement_tables(
    proposed: pd.DataFrame,
    ephemeris: pd.DataFrame,
    *,
    spoof_svids: tuple[int, ...],
    reference_svid: int,
    max_error_deg: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter whole authentic PRNs using the maximum paired error, never samples.

    The filter is outcome-conditioned: 100% at the cutoff is guaranteed for the
    retained cohort and must not be described as the unfiltered capture rate.
    """
    if max_error_deg is not None and (not np.isfinite(max_error_deg) or max_error_deg < 0):
        raise ValueError("Authentic PRN error cutoff must be finite and non-negative.")
    excluded = set(spoof_svids) | {reference_svid}
    authentic = sorted((set(proposed.columns) | set(ephemeris.columns)) - excluded)
    dd = proposed.reindex(columns=authentic)
    eph = ephemeris.reindex(index=dd.index, columns=authentic)
    errors = (dd - eph).abs().where(np.isfinite(dd) & np.isfinite(eph))
    unfiltered = summarize_angular_errors(errors)
    audit = unfiltered.iloc[:-1].copy()
    audit["retained"] = audit["paired_epochs"].gt(0)
    audit["exclusion_reason"] = np.where(audit["retained"], "", "no_paired_measurements")
    if max_error_deg is not None:
        exceeds = audit["max_abs_error_deg"].gt(max_error_deg)
        audit.loc[exceeds, "retained"] = False
        audit.loc[exceeds, "exclusion_reason"] = "maximum_paired_error_exceeds_cutoff"
    audit["cutoff_deg"] = max_error_deg
    retained = audit.loc[audit["retained"], "target_prn"].astype(int).tolist()
    filtered = summarize_angular_errors(errors[retained])
    return errors, unfiltered, audit, filtered


def export_agreement_table(summary: pd.DataFrame, output_dir: Path, *, filtered: bool, design: dict, show: bool) -> list[str]:
    """Export a caption-free booktabs-style table and an editable LaTeX tabular."""
    suffix = "filtered" if filtered else "unfiltered"
    stem = f"{AGREEMENT_TABLE_STEM}_{suffix}"
    columns = ["target_prn", *(f"p_le_{value}deg_pct" for value in AGREEMENT_THRESHOLDS_DEG)]
    table_data = summary[columns]
    csv_path = output_dir / f"{stem}.csv"
    table_data.to_csv(csv_path, index=False)

    def percentage(value):
        return f"{value:.1f}%" if np.isfinite(value) else "—"

    cell_text = [[str(row[0]), *(percentage(value) for value in row[1:])]
                 for row in table_data.itertuples(index=False, name=None)]
    table_design = design["agreement_table"]
    rows = len(cell_text) + 1
    height = table_design["row_height_in"] * rows + 2 * table_design["margin_in"]
    paths = [str(csv_path.resolve())]
    with plt.rc_context(design["rc_params"]):
        fig, ax = plt.subplots(figsize=(table_design["width_in"], height))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        bottom = table_design["margin_in"] / height
        top = 1 - bottom
        left, right = table_design["horizontal_bounds"]
        table = ax.table(
            cellText=cell_text,
            colLabels=["Target PRN", *(rf"$P_i({value}^{{\circ}})$" for value in AGREEMENT_THRESHOLDS_DEG)],
            cellLoc="center", colLoc="center", edges="open",
            colWidths=table_design["column_widths"], bbox=[left, bottom, right - left, top - bottom],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(table_design["font_size"])
        for (row, column), cell in table.get_celld().items():
            if row in (0, rows - 1):
                cell.get_text().set_fontweight("bold")
        row_height = (top - bottom) / rows
        ax.hlines([top, bottom], left, right, colors="black", linewidth=table_design["outer_rule_width"])
        ax.hlines([top - row_height, bottom + row_height], left, right,
                  colors="black", linewidth=table_design["inner_rule_width"])
        for extension in design["formats"]:
            path = output_dir / f"{stem}.{extension}"
            fig.savefig(path, format=extension, dpi=design["dpi"], facecolor="white")
            if path.stat().st_size == 0:
                raise RuntimeError(f"Empty agreement table: {path}")
            paths.append(str(path.resolve()))
        if show:
            from IPython.display import display
            display(fig)
        plt.close(fig)

    latex = [
        "% Requires \\usepackage{booktabs}. Caption is intentionally left to the manuscript.",
        "% Rates use paired samples; Overall pools the retained satellite-epoch pairs.",
        "% Authentic PRNs were filtered by maximum error on the displayed epochs." if filtered else "% No error-threshold PRN filter applied.",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"\textbf{Target PRN} & $P_i(5^{\circ})$ & $P_i(7^{\circ})$ & $P_i(10^{\circ})$ \\",
        r"\midrule",
    ]
    for row in cell_text:
        cells = [value.replace("%", r"\%") for value in row]
        if row[0] == "Overall":
            latex.append(r"\midrule")
            cells = [rf"\textbf{{{value}}}" for value in cells]
        latex.append(" & ".join(cells) + r" \\")
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    tex_path = output_dir / f"{stem}.tex"
    tex_path.write_text("\n".join(latex) + "\n", encoding="utf-8")
    paths.append(str(tex_path.resolve()))
    return paths


def select_compacted_report_epochs(
    epochs: pd.DataFrame,
    fractional: pd.DataFrame,
    proposed: pd.DataFrame,
    ephemeris: pd.DataFrame,
    *,
    spoof_svids: tuple[int, ...],
    count: int,
    minimum_spoof_prns: int = 1,
) -> tuple[pd.Index, dict, pd.DataFrame]:
    """Select a chronological display subset without modifying any measurements.

    Dynamic programming finds the lexicographic optimum: maximize paired spoof,
    phase-valid spoof, and total paired measurements; then minimize trace-mask
    changes and squared steps normalized by the three figures' vertical ranges.
    Source span and earliest epoch order break ties. This is a presentation
    selection, not an unbiased sample for estimating accuracy or availability.
    """
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Compacted sample count must be a positive integer.")
    if not 1 <= minimum_spoof_prns <= len(spoof_svids):
        raise ValueError("Minimum spoof PRN count is outside the usable group.")
    index = pd.Index(epochs["epoch_index"], name="epoch_index")
    tow = epochs["rcvTow_rx1"].to_numpy(dtype=float)
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Report epochs must be unique and chronological.")
    if not np.isfinite(tow).all() or (len(tow) > 1 and not np.allclose(np.diff(tow), 1.0, atol=0.01, rtol=0)):
        raise ValueError("A compacted seconds axis requires a regular 1 Hz source timeline.")
    fractional = fractional.reindex(index)
    proposed = proposed.reindex(index)
    ephemeris = ephemeris.reindex(index=index, columns=proposed.columns)
    phase_valid = np.isfinite(proposed)
    paired = phase_valid & np.isfinite(ephemeris)
    phase_count = phase_valid[list(spoof_svids)].sum(axis=1).to_numpy(dtype=int)
    paired_spoof_count = paired[list(spoof_svids)].sum(axis=1).to_numpy(dtype=int)
    paired_count = paired.sum(axis=1).to_numpy(dtype=int)
    eligible = np.flatnonzero(phase_count >= minimum_spoof_prns)
    if len(eligible) < count:
        raise ValueError(f"Requested {count} samples, but only {len(eligible)} valid spoof epochs are available.")

    # Missing values contribute mask changes, never artificial numeric values.
    frames = [fractional, ephemeris.where(paired) / 180.0, proposed / 180.0]
    values = [frame.iloc[eligible].to_numpy(dtype=float) for frame in frames]
    masks = [np.isfinite(value) for value in values]
    transitions, variation = {}, {}
    for end in range(len(eligible)):
        for start in range(end):
            transitions[start, end] = sum(int(np.count_nonzero(mask[start] != mask[end])) for mask in masks)
            variation[start, end] = sum(
                float(np.square(value[end][mask[start] & mask[end]] - value[start][mask[start] & mask[end]]).sum())
                / max(value.shape[1], 1)
                for value, mask in zip(values, masks)
            )

    def node_cost(position):
        source = eligible[position]
        return (-int(paired_spoof_count[source]), -int(phase_count[source]), -int(paired_count[source]))

    # Each state ends at a fixed source epoch, so all objective terms are additive.
    states = {end: (*node_cost(end), 0, 0.0, 0.0, (end,)) for end in range(len(eligible))}
    for length in range(2, count + 1):
        next_states = {}
        for end in range(length - 1, len(eligible)):
            node = node_cost(end)
            choices = []
            for start, previous in states.items():
                if start >= end:
                    continue
                choices.append((
                    *(previous[i] + node[i] for i in range(3)),
                    previous[3] + transitions[start, end],
                    previous[4] + variation[start, end],
                    previous[5] + float(tow[eligible[end]] - tow[eligible[start]]),
                    (*previous[6], end),
                ))
            if choices:
                next_states[end] = min(choices)
        states = next_states
    optimum = min(states.values())
    selected = index[eligible[list(optimum[6])]]
    audit = epochs.copy()
    audit["valid_spoof_prn_count"] = phase_count
    audit["paired_spoof_prn_count"] = paired_spoof_count
    audit["paired_prn_count"] = paired_count
    audit["eligible_for_selection"] = phase_count >= minimum_spoof_prns
    audit["selected_for_report"] = index.isin(selected)
    selection = {
        "policy": "coverage_then_smoothness_chronological_subsequence",
        "objective_priority": [
            "maximize paired spoof measurements", "maximize valid spoof DD measurements",
            "maximize all paired measurements", "minimize changes in trace availability",
            "minimize squared steps: fractional DD / 1 cycle, ephemeris and DD AoA / 180 degrees",
            "minimize original elapsed span", "earliest chronological epoch sequence",
        ],
        "eligible_epoch_count": len(eligible),
        "selected_epoch_count": count,
        "selected_paired_spoof_measurements": -optimum[0],
        "selected_valid_spoof_measurements": -optimum[1],
        "selected_paired_measurements": -optimum[2],
        "trace_availability_changes": optimum[3],
        "normalized_squared_steps": optimum[4],
        "measurement_values_changed": False,
        "interpretation": "Selected for presentation; do not use this subset to estimate unbiased accuracy or tracking duration.",
    }
    return selected, selection, audit


def prn_styles(prns, declared: tuple[int, ...], design: dict) -> dict[int, tuple[str, str | None]]:
    """Colour and marker per PRN; spoof and authentic PRNs use separate palettes."""
    styles = {}
    for is_spoof, palette_key in [(False, "authentic_colors"), (True, "spoof_colors")]:
        members = [prn for prn in sorted(prns) if (prn in declared) == is_spoof]
        for position, prn in enumerate(members):
            palette = design[palette_key]
            marker = design["spoof_markers"][position % len(design["spoof_markers"])] if is_spoof else None
            styles[prn] = (palette[position % len(palette)], marker)
    return styles


def wrapping_prns(fractional: pd.DataFrame) -> list[int]:
    """PRNs whose fractional DD jumps more than half a cycle between consecutive valid samples,
    i.e. crosses the +-0.5 cycle wrap (cos(alpha_i) - cos(alpha_r) near a half-integer)."""
    return [int(prn) for prn in fractional if fractional[prn].dropna().diff().abs().gt(0.5).any()]


def break_at_wraps(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert a NaN between samples that jump more than half a cycle (a +-0.5 wrap), so the
    line is not drawn across the plot. Returns x, y and the positions of the original samples."""
    jumps = np.flatnonzero(np.abs(np.diff(y)) > 0.5) + 1
    positions = np.arange(len(y)) + np.searchsorted(jumps, np.arange(len(y)), side="right")
    return (np.insert(x.astype(float), jumps, (x[jumps - 1] + x[jumps]) / 2),
            np.insert(y.astype(float), jumps, np.nan), positions)


def draw_prn_traces(ax, x, frame: pd.DataFrame, *, declared, styles, design, xlabel, xticks=None,
                    break_wraps: bool = False) -> None:
    """``break_wraps`` (fractional DD in cycles): no line across a +-0.5 wrap; every sample is kept."""
    # Collapsed spoof tracks coincide; sparse markers staggered per PRN keep each one visible.
    step = max(1, round(len(x) / design.get("spoof_marker_count", len(x))))
    marker_every = {prn: (position * step // len(declared), step) for position, prn in enumerate(declared)}
    for prn in frame:
        is_spoof = prn in declared
        color, marker = styles[prn]
        xs, ys, markevery = np.asarray(x), frame[prn].to_numpy(dtype=float), marker_every.get(prn)
        if break_wraps:
            xs, ys, positions = break_at_wraps(xs, ys)
            if markevery is not None:
                start, every = markevery
                markevery = np.zeros(len(xs), dtype=bool)
                markevery[positions[start::every]] = True
        ax.plot(
            xs, ys, color=color,
            lw=design["spoof_linewidth"] if is_spoof else design["authentic_linewidth"],
            ls="-" if is_spoof else "--", marker=marker, markersize=design["markersize"],
            markevery=markevery,
            label=f"PRN {prn}" + (" (spoof)" if is_spoof else ""),
            # Authentic dashes stay on top so one near the spoof direction is not hidden.
            zorder=2 if is_spoof else 3,
        )
    ax.set_xlabel(xlabel)
    if xticks is None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    else:
        ax.set_xticks(xticks)
    ax.grid(True, color=design["grid_color"], linewidth=0.5, alpha=1.0)
    ax.margins(x=0.025)
    ax.set_axisbelow(True)


def save_figure(fig, axes, output_dir: Path, stem: str, design: dict, *, comparison=False, show=True) -> list[str]:
    """Place the shared legend (and panel captions), save every format, and close."""
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=design["legend_anchor"],
        ncol=design["legend_columns"], frameon=False,
        columnspacing=1.2, handlelength=2.3, labelspacing=0.45,
    )
    # Measure the legend and panel captions to keep them clear of the axes.
    fig.canvas.draw()
    legend_box = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    top = legend_box.y0 - design["legend_gap_in"] / fig.get_figheight()
    fig.subplots_adjust(top=top, **design["subplot_margins"])
    if comparison:
        caption_style = design["comparison_captions"]
        caption_bottom = caption_style["bottom_in"] / fig.get_figheight()
        captions = [
            fig.text(
                (ax.get_position().x0 + ax.get_position().x1) / 2, caption_bottom, text,
                ha="center", va="bottom", fontsize=caption_style["font_size"], linespacing=1.2,
            )
            for ax, text in zip(axes, caption_style["texts"], strict=True)
        ]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        caption_top = max(
            caption.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y1
            for caption in captions
        )
        axis_label_height = max(
            ax.get_position().y0 - ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0
            for ax in axes
        )
        fig.subplots_adjust(bottom=max(
            design["subplot_margins"]["bottom"],
            caption_top + caption_style["gap_in"] / fig.get_figheight() + axis_label_height,
        ))
    else:
        fig.subplots_adjust(bottom=max(design["subplot_margins"]["bottom"], 0.19))
    paths = []
    for extension in design["formats"]:
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, format=extension, dpi=design["dpi"], facecolor="white")
        if path.stat().st_size == 0:
            raise RuntimeError(f"Empty figure: {path}")
        paths.append(str(path.resolve()))
    if show:
        from IPython.display import display
        display(fig)
    plt.close(fig)
    return paths


def authentic_after_window(
    scenario: PaperScenarioResult, *, declared: tuple[int, ...], after_tow: float, until_tow: float,
) -> dict | None:
    """Authentic-PRN traces for epochs after the spoof window, up to ``until_tow``.

    Re-runs the all-satellite DD over the whole capture with the window's reference satellite and
    reference-angle anchor (checked to reproduce the window's fractional DD), and picks each
    authentic PRN's branch per epoch from its ephemeris-based AoA, as on the window. Spoof PRNs
    and the reference are left out; missing phase or geometry stays NaN.
    """
    run = scenario.all_satellite_result
    extended = run_dataset(replace(run.config, length=None))
    if extended.reference_svid != run.reference_svid:
        raise RuntimeError("The whole-capture run chose a different reference satellite.")
    window = run.epochs["epoch_index"]
    pd.testing.assert_frame_equal(
        extended.epochs.set_index("epoch_index").loc[window, ["rcvTow_rx1", "rcvTow_rx2"]],
        run.epochs.set_index("epoch_index").loc[window, ["rcvTow_rx1", "rcvTow_rx2"]],
    )
    pd.testing.assert_frame_equal(
        extended.fractional_dd_cycles.reindex(index=window, columns=run.fractional_dd_cycles.columns),
        run.fractional_dd_cycles.reindex(window), check_names=False,
    )
    tow = extended.epochs["rcvTow_rx1"]
    extra = extended.epochs.loc[(tow > after_tow + 1e-6) & (tow <= until_tow + 1e-6)].copy()
    branches = scenario.ephemeris_aligned_branch.set_index("svid")["branch"]
    authentic = [int(prn) for prn in extended.fractional_dd_cycles.columns
                 if int(prn) not in declared and int(prn) != run.reference_svid and int(prn) in branches.index]
    if extra.empty or not authentic:
        return None
    index = pd.Index(extra["epoch_index"], name="epoch_index")
    fractional = extended.fractional_dd_cycles.reindex(index=index, columns=authentic)
    navsat = read_ubx_observations(run.config.receiver1_path, run.config.gnss_id, run.config.sig_id).navsat
    ephemeris = compute_navsat_aoa_by_synced_epoch(
        extra, navsat, epoch_tow_column="rcvTow_rx1", navsat_tow_decimals=run.config.navpvt_tow_decimals,
        baseline_azimuth_bias_deg=run.config.navsat_azimuth_bias_deg, selected_svids=tuple(authentic),
    ).reindex(index=index, columns=authentic)
    candidates = extended.candidate_aoa_deg
    proposed, chosen = select_aoa_by_reference_per_epoch(
        candidates.loc[candidates["epoch_index"].isin(index) & candidates["svid"].isin(authentic)], ephemeris)
    proposed = proposed.reindex(index=index, columns=authentic)
    modes = {prn: chosen[prn].dropna().mode() for prn in authentic if prn in chosen}
    return {"epochs": extra, "fractional": fractional, "ephemeris": ephemeris, "proposed": proposed,
            "authentic_svids": authentic,
            "branches": {prn: int(mode.iloc[0]) for prn, mode in modes.items() if not mode.empty}}


def plot_figure_set(
    x, output_dir: Path, *, design: dict, declared, styles, fractional, standalone_ephemeris,
    standalone_proposed, ephemeris, proposed, unpaired_spoof, xlabel, xticks=None, xlim=None, show=True,
) -> list[str]:
    """Figures 1-4. NaN samples (dropouts, gaps between windows) break the lines; in Figure 1 so
    does a +-0.5 cycle wrap (spoof PRNs; wrapping authentic PRNs are excluded before this)."""
    paths = []
    traces = {"declared": declared, "styles": styles, "design": design, "xlabel": xlabel, "xticks": xticks}
    missing = ", ".join(f"PRN {prn}" for prn in unpaired_spoof)
    with plt.rc_context(design["rc_params"]):
        for stem, frame, ylabel in [
            (FIGURE_STEMS[0], fractional, "Fractional double difference (cycles)"),
            (FIGURE_STEMS[1], standalone_ephemeris, r"$\alpha_i$ (degree)"),
            (FIGURE_STEMS[2], standalone_proposed, r"$\alpha_i$ (degree)"),
        ]:
            fig, ax = plt.subplots(figsize=design["single_figsize"])
            draw_prn_traces(ax, x, frame, **traces, break_wraps=stem == FIGURE_STEMS[0])
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.set_ylabel(ylabel)
            if stem == FIGURE_STEMS[0]:
                ax.set_ylim(-0.55, 0.55)
                ax.set_yticks(np.arange(-0.5, 0.51, 0.25))
            else:
                ax.set_ylim(0, 180)
                ax.set_yticks(np.arange(0, 181, 30))
            if stem == FIGURE_STEMS[1] and unpaired_spoof:
                ax.text(0.03, 0.03, f"No paired geometry: {missing}", transform=ax.transAxes, fontsize=8)
            paths += save_figure(fig, [ax], output_dir, stem, design, show=show)
        fig, axes = plt.subplots(1, 2, figsize=design["comparison_figsize"], sharex=True, sharey=True)
        for ax, frame in zip(axes, [ephemeris, proposed]):
            draw_prn_traces(ax, x, frame, **traces)
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.set_ylim(0, 180)
            ax.set_yticks(np.arange(0, 181, 30))
        axes[0].set_ylabel(r"$\alpha_i$ (degree)")
        if unpaired_spoof:
            axes[0].text(
                0.03, 0.03, f"No paired geometry: {missing}",
                transform=axes[0].transAxes, fontsize=8, va="bottom",
            )
        paths += save_figure(fig, axes, output_dir, FIGURE_STEMS[3], design, comparison=True, show=show)
    return paths


def export_paper_figures(
    scenario: PaperScenarioResult,
    output_dir: str | Path,
    *,
    declared_spoof_svids: tuple[int, ...],
    dataset_dir: str | Path | None = None,
    design: dict | None = None,
    compact_epochs: int | None = None,
    display_duration_s: float | None = None,
    extend_authentic: bool = False,
    authentic_max_error_deg: float | None = None,
    filter_figures: bool = True,
    exclude_wrapping_authentic: bool = True,
    show: bool = True,
) -> dict:
    """Export Figures 1-4, plotted data, and their configuration/provenance.

    ``exclude_wrapping_authentic`` drops every authentic PRN whose fractional DD crosses the
    +-0.5 cycle wrap on the drawn epochs (window and any after-window extension) from all
    figures and tables; spoof PRNs are kept and Figure 1 breaks their lines at the wrap.

    ``display_duration_s`` fixes the time axis at 0..duration for manuscript figures: a longer
    window shows its first ``duration`` seconds, a shorter one is drawn in full and the rest of
    the axis stays empty. ``extend_authentic`` then fills that empty part with the authentic
    PRNs' measurements after the spoof window (see :func:`authentic_after_window`); spoof PRNs
    end with the window, and tables and the group median still use the window only.
    """
    if extend_authentic and display_duration_s is None:
        raise ValueError("extend_authentic needs display_duration_s.")
    if compact_epochs is not None and display_duration_s is not None:
        raise ValueError("Use either compact_epochs or display_duration_s, not both.")
    design = load_figure_design() if design is None else design
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else Path(scenario.config.receiver1_path).parent
    if not design["formats"] or not set(design["formats"]) <= {"png", "pdf", "svg", "eps"}:
        raise ValueError("Figure formats must be selected from png, pdf, svg, eps.")
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    run = scenario.all_satellite_result
    source_index = pd.Index(run.epochs["epoch_index"], name="epoch_index")
    source_tow = run.epochs["rcvTow_rx1"].to_numpy(dtype=float)
    if len(source_tow) > 1 and not np.all(np.diff(source_tow) > 0):
        raise ValueError("Receiver time must increase across the report window.")
    reference = int(run.reference_svid)
    spoof = tuple(int(prn) for prn in scenario.spoof_svids)
    declared = tuple(sorted(int(prn) for prn in declared_spoof_svids))
    fractional = run.fractional_dd_cycles.reindex(source_index).drop(columns=reference, errors="ignore")
    fractional = fractional.dropna(axis=1, how="all")
    proposed = scenario.best_m_all_satellite_aoa_deg.reindex(source_index).drop(columns=reference, errors="ignore")
    ephemeris = scenario.ephemeris_aoa_rx1_deg.reindex(source_index).drop(columns=reference, errors="ignore")
    minimum_prns = int(scenario.run_summary["min_valid_spoof_prns"].iloc[0])
    max_gap_epochs = int(scenario.run_summary["max_gap_epochs"].iloc[0])
    selection = {"policy": "all_epochs_in_analysis_window"}
    index = source_index
    if compact_epochs is not None:
        index, selection, selection_audit = select_compacted_report_epochs(
            run.epochs, fractional, proposed, ephemeris,
            spoof_svids=spoof, count=compact_epochs, minimum_spoof_prns=minimum_prns,
        )
        selection_audit.to_csv(data_dir / "epoch_selection_audit.csv", index=False)
    if display_duration_s is not None:
        index = source_index[source_tow - source_tow[0] <= display_duration_s + 1e-6]
        selection = {
            "policy": "first_seconds_of_analysis_window", "display_duration_s": display_duration_s,
            "analysis_window_elapsed_span_s": float(source_tow[-1] - source_tow[0]),
            "truncated": bool(len(index) < len(source_index)),
        }
    if compact_epochs is not None or display_duration_s is not None or authentic_max_error_deg is not None:
        # Preserve the unselected analysis beside the exact plotted subset.
        source_dir = data_dir / "analysis_window"
        source_dir.mkdir(exist_ok=True)
        for name, frame in [
            ("fractional_double_difference_cycles", fractional),
            ("alpha_i_proposed_best_m_deg", proposed),
            ("alpha_i_ephemeris_deg", ephemeris),
            ("spoof_joint_source_branch", scenario.joint_source_branch),
        ]:
            frame.to_csv(source_dir / f"{name}.csv", index_label="epoch_index")
        run.epochs.to_csv(source_dir / "epochs.csv", index=False)
        scenario.window_quality.to_csv(source_dir / "window_quality.csv", index=False)
    if compact_epochs is not None or display_duration_s is not None:
        fractional, proposed, ephemeris = (frame.reindex(index) for frame in (fractional, proposed, ephemeris))
        fractional = fractional.dropna(axis=1, how="all")
    epochs = run.epochs.set_index("epoch_index", drop=False).loc[index].copy()
    tow = epochs["rcvTow_rx1"].to_numpy(dtype=float)
    elapsed = tow - tow[0]
    plot_time = np.arange(1, len(index) + 1, dtype=float) if compact_epochs is not None else elapsed
    epoch_map = epochs.copy()
    epoch_map.insert(0, "report_sample", np.arange(1, len(index) + 1))
    epoch_map.insert(1, "plot_time_s", plot_time)
    epoch_map["analysis_window_elapsed_s"] = tow - source_tow[0]
    if scenario.probe_result is not None:
        epoch_map["capture_elapsed_s"] = tow - float(scenario.probe_result.epochs["rcvTow_rx1"].iloc[0])
    epoch_map["omitted_source_epochs_before"] = np.r_[0, np.diff(epochs["source_epoch_index"]) - 1]
    epoch_map.to_csv(data_dir / "selected_epoch_map.csv", index=False)
    extension = None
    if extend_authentic and elapsed[-1] < display_duration_s:
        extension = authentic_after_window(scenario, declared=declared, after_tow=tow[-1],
                                           until_tow=tow[0] + display_duration_s)
    # Colors are assigned before any PRN exclusion so every trace keeps its identity.
    style_prns = sorted(set(fractional.columns) | set(spoof) | {
        int(prn) for prn in set(proposed.columns) & set(ephemeris.columns)
        if (proposed[prn].notna() & ephemeris[prn].notna()).any()})
    styles = prn_styles(style_prns, declared, design)
    wrap_excluded: list[int] = []
    if exclude_wrapping_authentic:
        drawn = fractional if extension is None else pd.concat(
            [fractional, extension["fractional"].reindex(columns=fractional.columns)])
        wrap_excluded = sorted(prn for prn in wrapping_prns(drawn) if prn not in set(declared) | set(spoof))
        fractional, proposed, ephemeris = (
            frame.drop(columns=wrap_excluded, errors="ignore") for frame in (fractional, proposed, ephemeris))
    comparable = [
        int(prn) for prn in sorted(set(proposed.columns) & set(ephemeris.columns))
        if (proposed[prn].notna() & ephemeris[prn].notna()).any()
    ]
    # Missing satellite geometry prevents a paired comparison, but does not
    # invalidate that spoof PRN's carrier-phase DD estimate.
    unpaired_spoof = sorted(set(spoof) - set(comparable))
    report_prns = sorted(set(comparable) | set(spoof))
    proposed, ephemeris = (
        frame.reindex(columns=report_prns) for frame in (proposed, ephemeris)
    )
    np.testing.assert_allclose(
        proposed[list(spoof)], scenario.joint_source_aoa_deg.reindex(index)[list(spoof)], equal_nan=True
    )
    proposed_all_phase = proposed.copy()
    standalone_ephemeris = ephemeris.copy()
    common = proposed.notna() & ephemeris.notna()
    proposed, ephemeris = proposed.where(common), ephemeris.where(common)
    standalone_proposed = (
        proposed_all_phase if design.get("standalone_proposed_use_all_phase_epochs", False)
        else proposed
    )

    errors, unfiltered_agreement, authentic_audit, agreement = authentic_agreement_tables(
        proposed, ephemeris, spoof_svids=declared, reference_svid=reference,
        max_error_deg=authentic_max_error_deg,
    )
    retained_authentic = authentic_audit.loc[authentic_audit["retained"], "target_prn"].astype(int).tolist()
    omitted_authentic = authentic_audit.loc[~authentic_audit["retained"], "target_prn"].astype(int).tolist()
    errors.to_csv(data_dir / "authentic_angular_errors_deg_unfiltered.csv", index_label="epoch_index")
    unfiltered_agreement.to_csv(data_dir / "authentic_agreement_unfiltered.csv", index=False)
    authentic_audit.to_csv(data_dir / "authentic_prn_filter_audit.csv", index=False)
    agreement.to_csv(data_dir / "authentic_agreement_reported.csv", index=False)
    if authentic_max_error_deg is not None and not retained_authentic:
        raise ValueError("No authentic PRNs meet the maximum-error cutoff; a 100% table would have no samples.")
    if authentic_max_error_deg is not None and filter_figures:
        # Apply the whole-PRN display filter only after the fixed epoch selection.
        # Colors were assigned before filtering so surviving traces keep their identities.
        keep = set(retained_authentic) | set(declared)
        fractional = fractional[[prn for prn in fractional if prn in keep]]
        comparable = [prn for prn in comparable if prn in keep]
        report_prns = [prn for prn in report_prns if prn in keep]
        proposed, ephemeris, standalone_proposed, standalone_ephemeris, common = (
            frame[report_prns] for frame in (proposed, ephemeris, standalone_proposed, standalone_ephemeris, common)
        )

    selected_joint = scenario.joint_source_aoa_deg.reindex(index)
    selected_branches = scenario.joint_source_branch.reindex(index)
    group_median = float(selected_joint.median(axis=1).median())
    branch_values = sorted(int(value) for value in np.unique(selected_branches.to_numpy()) if np.isfinite(value))
    chosen_by = str(scenario.joint_source_ambiguity["chosen_by"].iloc[0])
    branch_text = ", ".join(str(value) for value in branch_values)
    source_text = {"explicit_m": "declared", "unresolved": "ambiguity unresolved", "prior": "declared prior", "data": "data-selected"}[chosen_by]
    footer = f"Spoof group: m = {branch_text} ({source_text}); median AoA = {group_median:.1f}°"
    spoof_paired_counts = sorted({int(common[prn].sum()) for prn in spoof})
    paired_note = ""
    if min(spoof_paired_counts) < len(index):
        counts = "/".join(map(str, spoof_paired_counts))
        paired_note = f"Spoof AoA comparison: {counts} of {len(index)} epochs; phase/geometry gaps retained."
        footer += "\n" + paired_note
    version = Path(dataset_dir).name
    subtitle = f"{version}  |  Reference PRN {reference}  |  {len(index)} epochs  |  GPS TOW {tow[0]:.0f}–{tow[-1]:.0f} s"
    quality = scenario.window_quality.set_index("epoch_index", drop=False).loc[index]
    valid_phase_epochs = int(quality["meets_spoof_prn_requirement"].sum())
    missing_phase_epochs = len(index) - valid_phase_epochs

    if extension is not None:
        extra_index = pd.Index(extension["epochs"]["epoch_index"], name="epoch_index")
        extra_tow = extension["epochs"]["rcvTow_rx1"].to_numpy(dtype=float)
        extra_ephemeris = extension["ephemeris"].where(extension["proposed"].notna())
        extra_proposed = extension["proposed"].where(extension["ephemeris"].notna())

        def extend(frame, extra):
            return pd.concat([frame, extra.reindex(index=extra_index, columns=frame.columns)])

        fractional = extend(fractional, extension["fractional"])
        standalone_ephemeris = extend(standalone_ephemeris, extension["ephemeris"])
        standalone_proposed = extend(standalone_proposed, extension["proposed"])
        ephemeris, proposed = extend(ephemeris, extra_ephemeris), extend(proposed, extra_proposed)
        common = extend(common, extra_proposed.notna()).fillna(False).astype(bool)
        plot_time = np.r_[plot_time, extra_tow - tow[0]]
        extra_map = extension["epochs"].copy()
        extra_map.insert(0, "report_sample", np.arange(len(epoch_map) + 1, len(epoch_map) + len(extra_map) + 1))
        extra_map.insert(1, "plot_time_s", extra_tow - tow[0])
        extra_map["analysis_window_elapsed_s"] = extra_tow - source_tow[0]
        if "capture_elapsed_s" in epoch_map:
            extra_map["capture_elapsed_s"] = extra_tow - float(scenario.probe_result.epochs["rcvTow_rx1"].iloc[0])
        epoch_map = pd.concat([epoch_map.assign(in_analysis_window=True), extra_map.assign(in_analysis_window=False)],
                              ignore_index=True)
        epoch_map.to_csv(data_dir / "selected_epoch_map.csv", index=False)

    xlim = None
    if compact_epochs is None:
        xlabel, xticks = design["elapsed_axis_label"], None
        if display_duration_s is not None:
            xlim = (-0.025 * display_duration_s, 1.025 * display_duration_s)
            xticks = np.linspace(0, display_duration_s, 7)
    else:
        step = int(design["compacted_tick_step"])
        xlabel, xticks = design["compacted_axis_label"], sorted({1, len(index), *range(step, len(index) + 1, step)})
    paths = plot_figure_set(
        plot_time, output_dir, design=design, declared=declared, styles=styles,
        fractional=fractional, standalone_ephemeris=standalone_ephemeris,
        standalone_proposed=standalone_proposed, ephemeris=ephemeris, proposed=proposed,
        unpaired_spoof=unpaired_spoof, xlabel=xlabel, xticks=xticks, xlim=xlim, show=show,
    )

    table_files = export_agreement_table(unfiltered_agreement, output_dir, filtered=False, design=design, show=show)
    if authentic_max_error_deg is not None:
        table_files += export_agreement_table(agreement, output_dir, filtered=True, design=design, show=show)

    for name, frame in [
        ("fractional_double_difference_cycles", fractional),
        ("alpha_i_ephemeris_deg", ephemeris),
        ("alpha_i_ephemeris_all_geometry_deg", standalone_ephemeris),
        ("alpha_i_proposed_best_m_deg", standalone_proposed),
        ("alpha_i_proposed_best_m_paired_deg", proposed),
        ("spoof_joint_source_branch", selected_branches),
        ("spoof_joint_source_aoa_all_phase_epochs_deg", selected_joint),
        ("paired_valid_aoa_mask", common),
    ]:
        frame.to_csv(data_dir / f"{name}.csv", index_label="epoch_index")
    epochs.to_csv(data_dir / "epochs.csv", index=False)
    quality.to_csv(data_dir / "window_quality.csv", index=False)
    for name, frame in [
        ("branch_selection", scenario.joint_source_ambiguity),
        ("spoof_branch_candidates", scenario.spoof_branch_candidates),
        ("spoof_branch_detection", scenario.spoof_branch_detection),
        ("authentic_ephemeris_aligned_branch", scenario.ephemeris_aligned_branch),
    ]:
        frame.to_csv(data_dir / f"{name}.csv", index=False)
    by_epoch = scenario.ephemeris_aligned_branch_by_epoch
    authentic_columns = [prn for prn in by_epoch.columns if int(prn) not in set(scenario.spoof_svids)]
    by_epoch.reindex(columns=authentic_columns).to_csv(
        data_dir / "authentic_ephemeris_aligned_branch_by_epoch.csv", index_label="epoch_index")

    note_path = Path(dataset_dir) / "note.txt"
    if note_path.is_file():
        (data_dir / "dataset_note.txt").write_text(note_path.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": scenario.config.name,
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "declared_spoof_svids": declared,
        "usable_spoof_svids": spoof,
        "omitted_spoof_svids": sorted(set(declared) - set(spoof)),
        "reference_svid": reference,
        "reference_alpha_deg": float(scenario.run_summary["reference_alpha_deg"].iloc[0]),
        "epochs": len(index),
        "elapsed_span_s": float(elapsed[-1]),
        "plot_axis_mode": ("compacted_1hz_samples" if compact_epochs is not None
                           else "fixed_duration_elapsed_time" if display_duration_s is not None else "actual_elapsed_time"),
        "display_duration_s": display_duration_s,
        "authentic_after_window": None if extension is None else {
            "policy": "authentic PRNs drawn after the spoof window up to the display duration; spoof PRNs end with "
                      "the window; tables, group median and branch choices use the window only",
            "tow_start_s": float(extension["epochs"]["rcvTow_rx1"].iloc[0]),
            "tow_end_s": float(extension["epochs"]["rcvTow_rx1"].iloc[-1]),
            "epochs": len(extension["epochs"]),
            "authentic_svids": extension["authentic_svids"],
            "branches_from_window": extension["branches"],
            "reference_svid": reference,
        },
        "compacted_nominal_duration_s": len(index) if compact_epochs is not None else None,
        "plot_time_start_s": float(plot_time[0]),
        "plot_time_end_s": float(plot_time[-1]),
        "selected_epoch_indices": [int(value) for value in index],
        "selected_source_epoch_indices": epochs["source_epoch_index"].astype(int).tolist(),
        "selected_capture_elapsed_s": epoch_map["capture_elapsed_s"].tolist() if "capture_elapsed_s" in epoch_map else None,
        "epoch_selection": selection,
        "analysis_window_epochs": len(source_index),
        "analysis_window_elapsed_span_s": float(source_tow[-1] - source_tow[0]),
        "analysis_window_valid_spoof_epochs": int(scenario.window_quality["meets_spoof_prn_requirement"].sum()),
        "min_valid_spoof_prns": minimum_prns,
        "max_gap_epochs": max_gap_epochs,
        "valid_spoof_phase_epochs": valid_phase_epochs,
        "missing_spoof_phase_epochs": missing_phase_epochs,
        "joint_aoa_epoch_count": int(selected_joint.notna().any(axis=1).sum()),
        "joint_branch_continuity_policy": (
            "restart at gaps and PRN membership changes"
            if "segment_id" in scenario.joint_source_ambiguity else "one continuous complete group"
        ),
        "tow_start_s": float(tow[0]), "tow_end_s": float(tow[-1]),
        "single_difference_order": scenario.config.single_difference_order,
        "navsat_azimuth_bias_deg": scenario.config.navsat_azimuth_bias_deg,
        "baseline_over_wavelength": scenario.config.baseline_over_wavelength,
        "spoof_branch_m": scenario.run_summary["spoof_branch_m"].iloc[0],
        "displayed_spoof_m_values": branch_values,
        "chosen_by": chosen_by,
        "branch_ambiguous": bool(scenario.joint_source_ambiguity["ambiguous"].any()),
        "group_median_aoa_deg": group_median,
        "group_median_definition": "median over displayed epochs of the median AoA across usable spoof PRNs",
        "authentic_branch_policy": "per-epoch m closest to the ephemeris-based AoA (direction cosine); this is a geometry-conditioned comparison",
        "wrap_excluded_authentic_svids": wrap_excluded,
        "style_prns": [int(prn) for prn in style_prns],
        "wrap_exclusion_policy": (
            "authentic PRNs whose fractional DD jumps more than 0.5 cycle between consecutive valid drawn samples "
            "(a +-0.5 wrap) are removed from every figure and table; spoof PRNs are kept and Figure 1 breaks "
            "their lines at the wrap" if exclude_wrapping_authentic else "no wrap exclusion"
        ),
        "authentic_max_error_deg": authentic_max_error_deg,
        "retained_authentic_svids": retained_authentic,
        "excluded_authentic_svids": omitted_authentic if authentic_max_error_deg is not None else [],
        "authentic_filter_applies_to_figures": authentic_max_error_deg is not None and filter_figures,
        "authentic_prn_filter_policy": (
            "exclude the entire authentic PRN when any paired error on the selected epochs exceeds the cutoff; "
            "apply after epoch selection; leave spoof PRNs and measurement values unchanged; "
            + ("excluded PRNs are removed from the figures" if filter_figures else "figures keep every authentic PRN")
            if authentic_max_error_deg is not None else "no error-threshold PRN filter"
        ),
        "authentic_agreement_thresholds_deg": AGREEMENT_THRESHOLDS_DEG,
        "authentic_agreement_definition": "100 * count(abs(DD AoA - ephemeris AoA) <= threshold) / finite paired count",
        "authentic_agreement_overall_definition": "pooled finite satellite-epoch pairs across retained authentic PRNs",
        "authentic_agreement_overall": agreement.iloc[-1].to_dict(),
        "authentic_agreement_unfiltered_overall": unfiltered_agreement.iloc[-1].to_dict(),
        "authentic_agreement_interpretation": (
            "The cutoff-column 100% is guaranteed by outcome-based PRN exclusion and is not an unfiltered performance rate."
            if authentic_max_error_deg is not None else "Agreement on the displayed epoch selection; authentic branches are geometry-conditioned."
        ),
        "angle_convention": "baseline-relative arccos angle in [0, 180] degrees",
        "comparable_prns": comparable,
        "report_prns": report_prns,
        "unpaired_spoof_svids": unpaired_spoof,
        "paired_aoa_epoch_count_by_prn": {int(prn): int(common[prn].sum()) for prn in report_prns},
        "standalone_proposed_epoch_count_by_prn": {int(prn): int(standalone_proposed[prn].notna().sum()) for prn in report_prns},
        "fractional_dd_prns": [int(value) for value in fractional.columns],
        "missing_data_policy": "no interpolation; NaNs break lines; figure 04 uses paired-valid epochs",
        "time_axis_policy": (
            "N means retained 1 Hz samples, not continuous elapsed seconds or GPS subframes; "
            "chronological samples are drawn at positions 1..N; original timestamps are in selected_epoch_map.csv"
            if compact_epochs is not None else
            f"actual elapsed receiver time on a fixed 0-{display_duration_s:g} s axis; the first "
            f"{display_duration_s:g} s of the analysis window are shown, a shorter window leaves the rest empty; "
            "missing source epochs retain their positions"
            if display_duration_s is not None else "actual elapsed receiver time; missing source epochs retain their positions"
        ),
        "branch_summary_scope": "complete analysis window; branch_selection.csv retains original epoch indices",
        "standalone_proposed_uses_all_phase_epochs": bool(design.get("standalone_proposed_use_all_phase_epochs", False)),
        "figure_design_source": str(DESIGN_PATH.resolve()),
        "figure_annotation_policy": "axes and PRN legend; Figure 4 also has descriptions (a) and (b) below its panels",
        "plot_axis_label": design["compacted_axis_label"] if compact_epochs is not None else design["elapsed_axis_label"],
        "table_annotation_policy": (
            "unfiltered agreement table always; filtered table only with an authentic PRN error cutoff; no captions"
        ),
        "comparison_panel_order": ["ephemeris", "proposed DD"],
        "figure_design": design,
        "sources_sha256": {
            str(Path(path).resolve()): file_sha256(path)
            for path in [scenario.config.receiver1_path, scenario.config.receiver2_path,
                         *([note_path] if note_path.is_file() else []),
                         Path(__file__), Path(__file__).with_name("aoa_pipeline.py"), DESIGN_PATH]
        },
        "figure_count": len(FIGURE_STEMS),
        "figure_file_count": len(paths),
        "figure_files": paths,
        "table_files": table_files,
    }
    # NumPy scalar values can originate in the run-summary frame.
    (output_dir / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=lambda value: value.item(), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    omitted = manifest["omitted_spoof_svids"]
    compaction_note = ""
    if compact_epochs is not None:
        compaction_note = (
            f"This is a **{len(index)}-sample compacted display** of 1 Hz measurements. The axis runs from 1 to "
            f"{len(index)} with the label Epoch; the actual first-to-last source span is {elapsed[-1]:g} seconds. "
            "It does not establish two decoded GPS subframes or continuous tracking for that duration. "
            "Selected samples keep their original chronological order and numeric values. Global omissions "
            "are removed from the display axis; NaNs for individual PRNs still break lines. "
            "`data/selected_epoch_map.csv` maps every plotted position to its capture offset and receiver TOW.\n\n"
            "Selection first maximizes paired spoof, valid spoof DD, and total paired observations, then "
            "minimizes changes in trace availability and normalized squared steps across fractional DD, "
            "ephemeris AoA, and DD AoA. Source span and earliest epoch order break ties. This presentation "
            "selection favors smooth, well-observed samples; its errors or apparent stability must not be "
            "reported as unbiased capture performance. The complete original window is retained in "
            "`data/analysis_window/`, and `data/epoch_selection_audit.csv` records every selected/rejected epoch. "
            "Branch choices and reference geometry were computed on that complete window before selection.\n\n"
        )
    if display_duration_s is not None:
        compaction_note = (
            f"The time axis is fixed at 0-{display_duration_s:g} s for the manuscript. The figures show the first "
            f"{elapsed[-1]:g} s ({len(index)} epochs) of the {source_tow[-1] - source_tow[0]:g} s analysis window"
            + ("" if len(index) < len(source_index) else
               "; the window is shorter" + ("" if extension is not None else ", so the rest of the axis is empty"))
            + ". Nothing is selected by quality. Tables and the group median use the displayed epochs; branch "
            "choices were made on the complete window, which is kept in `data/analysis_window/`.\n\n"
        )
        if extension is not None:
            compaction_note += (
                f"After the spoof window ends the spoof PRNs have no valid phase, but the authentic PRNs are still "
                f"tracked: the figures also draw them from TOW {manifest['authentic_after_window']['tow_start_s']:.0f} "
                f"to {manifest['authentic_after_window']['tow_end_s']:.0f} s ({len(extension['epochs'])} epochs), "
                f"with the same reference PRN {reference}, reference-angle anchor and per-PRN branch as the window. "
                "Those epochs are flagged `in_analysis_window = False` in `data/selected_epoch_map.csv` and are not "
                "used in any table, median or branch choice.\n\n"
            )
    agreement_note = (
        "The agreement CSV reports P_i(5°), P_i(7°), and P_i(10°), using inclusive thresholds on "
        "|DD AoA − ephemeris AoA|. Missing pairs do not enter the denominator. Overall pools all retained "
        "authentic satellite-epoch pairs. The reference and all declared spoof PRNs are excluded from this table. "
        "These supporting rates are retained in `data/authentic_agreement_reported.csv`. "
        f"`{AGREEMENT_TABLE_STEM}_unfiltered.*` tabulates every authentic PRN; "
        f"`{AGREEMENT_TABLE_STEM}_filtered.*` is exported only when an error cutoff is set.\n\n"
    )
    if authentic_max_error_deg is not None:
        agreement_note += (
            f"**Filtered authentic cohort:** retain a PRN only when its maximum paired error on the "
            f"{len(index)} displayed epochs is ≤{authentic_max_error_deg:g}°. Retained: {retained_authentic}. "
            f"Excluded: {omitted_authentic}. The whole PRN is removed from the reported agreement CSV, the filtered table"
            + (" and both panels. " if filter_figures else "; the figures still show every authentic PRN. ") +
            "The selected epochs, spoof tracks, branch choices, and original measurements are unchanged. "
            "A 100% rate at the filtering threshold follows from this exclusion rule; it is not the rate for "
            "all authentic satellites. `data/authentic_angular_errors_deg_unfiltered.csv` retains every original "
            "paired error, `data/authentic_agreement_unfiltered.csv` retains all original rates, and "
            "`data/authentic_prn_filter_audit.csv` records counts, maximum errors, and exclusion reasons.\n\n"
        )
    (output_dir / "README.md").write_text(
        f"# {scenario.config.name}: figure report\n\n"
        f"{subtitle}\n\n{footer}. AoA uses the 0–180° convention.\n\n" + compaction_note + agreement_note +
        f"Declared spoof PRNs: {declared}. Usable: {spoof}. Omitted without usable phase: {omitted or 'none'}.\n\n"
        f"Spoof PRNs without paired geometry: {unpaired_spoof or 'none'}. Their paired curves remain NaN; "
        "their DD estimates are preserved in `data/spoof_joint_source_aoa_all_phase_epochs_deg.csv` "
        "and `data/spoof_branch_candidates.csv`. Missing geometry is not a negative spoof-detection result.\n\n"
        "Figures 1-4 are supplied: fractional DD, ephemeris AoA, proposed DD AoA, and their side-by-side comparison. "
        "Each figure is exported in the configured formats. Figure 3 preserves all valid DD measurements. "
        "Figure 2 preserves all available geometry. The Figure 4 comparison uses identical "
        "PRNs, time samples, colors, and AoA limits in both panels. CSVs in `data/` contain the plotted values.\n\n"
        "Both panels use the paired mask. The paired DD data is `data/alpha_i_proposed_best_m_paired_deg.csv`; "
        "`data/alpha_i_proposed_best_m_deg.csv` also preserves DD measurements without matching ephemeris angles.\n\n"
        "The descriptions below the panels are (a) Ephemeris-derived baseline-referenced angles and "
        "(b) Angles estimated using the proposed wrapped-DD method. Ephemeris AoA is on the left and "
        "proposed DD AoA is on the right, with a shared PRN legend above.\n\n"
        "The spoof group uses the experiment's joint-source branch; "
        "alternatives and selection provenance remain in `data/spoof_branch_detection.csv`. "
        "Authentic PRNs use the ephemeris-aligned constant branch, so their agreement is conditioned on geometry. "
        "The single-wavelength branch ambiguity is retained even when m is declared.\n\n"
        f"The full capture is probed. The analysis window spans {source_tow[-1] - source_tow[0]:g} seconds with "
        f"{manifest['analysis_window_valid_spoof_epochs']} valid group epochs. The figures display "
        f"{valid_phase_epochs} valid group epochs and {missing_phase_epochs} missing group epochs. "
        f"It requires at least {minimum_prns} usable spoof PRNs plus the reference in both receivers, "
        f"allowing at most {max_gap_epochs} consecutive missing epochs per gap. "
        "Missing phase and missing geometry remain NaN and break plotted lines. "
        "The relaxed mode restarts joint branch selection after gaps or membership changes; "
        "`data/branch_selection.csv` records each block. No values are interpolated or changed to match expected angles.\n\n"
        "Design: `aoa_figure_style.json`. Run `AOA_Paper.ipynb` for the experiment and "
        "`AOA_Export_Figures.ipynb` for figure export. "
        "`report_manifest.json` records source hashes, branch policy, and receiver polarity.\n",
        encoding="utf-8",
    )
    return manifest


def write_run_report(path: str | Path, config, scenario: PaperScenarioResult, manifest: dict) -> str:
    """Plain-text summary of one exported run: spoof PRNs, reference, branch, results."""
    from aoa_experiment import result_table

    def prns(values) -> str:
        return ", ".join(str(int(value)) for value in values) or "none"

    reference = manifest["angle_reference"]
    ambiguity = scenario.joint_source_ambiguity
    first = ambiguity.iloc[0]
    branches = scenario.joint_source_branch.stack()
    branch_counts = ", ".join(
        f"m={int(value)}: {int(count)}" for value, count in branches.value_counts().sort_index().items()
    )
    detection = scenario.spoof_branch_detection
    alternative = detection["alternative_aoa_deg"].median()
    flagged = detection.loc[detection["spoof_flag_all_branches"].astype(bool), "svid"]
    not_flagged = detection.loc[~detection["spoof_flag_all_branches"].astype(bool), "svid"]
    authentic = sorted(set(manifest["comparable_prns"]) - set(config.spoof_svids))
    def agreement_text(overall: dict) -> str:
        rates = " / ".join(f"{overall[f'p_le_{value}deg_pct']:.1f}%" for value in AGREEMENT_THRESHOLDS_DEG)
        return (f"{rates} within {'/'.join(map(str, AGREEMENT_THRESHOLDS_DEG))} deg "
                f"({overall['paired_epochs']} pairs, median error {overall['median_abs_error_deg']:.2f} deg)")

    cutoff = manifest["authentic_max_error_deg"]
    agreement_lines = [f"  All authentic PRNs:   {agreement_text(manifest['authentic_agreement_unfiltered_overall'])}"]
    if cutoff is not None:
        agreement_lines += [
            f"  Clean (max <= {cutoff:g} deg): {agreement_text(manifest['authentic_agreement_overall'])}",
            f"    retained PRN {prns(manifest['retained_authentic_svids'])}; "
            f"excluded PRN {prns(manifest['excluded_authentic_svids'])} "
            f"({'also removed from figures' if manifest['authentic_filter_applies_to_figures'] else 'figures keep all PRNs'})",
            "    Filtered rates are conditioned on the outcome; quote the unfiltered rates as capture performance.",
        ]
    epochs = manifest["selected_source_epoch_indices"]
    table = result_table(scenario)
    # The joint track may change m between epochs; its spread differs from the constant-m columns.
    track = scenario.joint_source_aoa_deg
    table.insert(1, "Track median (deg)", table["PRN"].map(track.median()).to_numpy())
    table.insert(2, "Track std (deg)", table["PRN"].map(track.std()).to_numpy())
    table = table.round(2).to_string(index=False)
    lines = [
        f"AoA export report: {config.name}",
        "=" * (19 + len(config.name)),
        f"Created (UTC):        {manifest['created_at_utc']}",
        f"Experiment file:      {manifest['experiment_file']}",
        f"Replay verified:      {'yes' if manifest['experiment_replay_verified'] else 'NO'}",
        "",
        "Capture",
        f"  Dataset dir:        {manifest['dataset_dir']}",
        f"  Receiver 1:         {Path(config.receiver1_path).name}",
        f"  Receiver 2:         {Path(config.receiver2_path).name}",
        *([
            f"  Analysis window:    GPS TOW {manifest['tow_start_s']:.0f}-"
            f"{manifest['tow_start_s'] + manifest['analysis_window_elapsed_span_s']:.0f} s "
            f"({manifest['analysis_window_epochs']} epochs, {manifest['analysis_window_elapsed_span_s']:g} s)",
            f"  Figures show:       capture epochs {epochs[0]}-{epochs[-1]}, GPS TOW "
            f"{manifest['tow_start_s']:.0f}-{manifest['tow_end_s']:.0f} s ({manifest['epochs']} epochs, "
            f"{manifest['elapsed_span_s']:g} s) on a fixed 0-{manifest['display_duration_s']:g} s axis",
            *([
                f"  Authentic only:     GPS TOW {manifest['authentic_after_window']['tow_start_s']:.0f}-"
                f"{manifest['authentic_after_window']['tow_end_s']:.0f} s "
                f"({manifest['authentic_after_window']['epochs']} epochs after the spoof window; PRN "
                f"{prns(manifest['authentic_after_window']['authentic_svids'])}; drawn only, not in tables)",
            ] if manifest.get("authentic_after_window") else []),
        ] if manifest.get("display_duration_s") is not None else [
            f"  Analysis window:    capture epochs {epochs[0]}-{epochs[-1]}, GPS TOW "
            f"{manifest['tow_start_s']:.0f}-{manifest['tow_end_s']:.0f} s "
            f"({manifest['epochs']} epochs, {manifest['analysis_window_elapsed_span_s']:g} s)",
        ]),
        f"  Spoof-group epochs: {manifest['valid_spoof_phase_epochs']} valid, "
        f"{manifest['missing_spoof_phase_epochs']} missing (no interpolation)",
        "",
        "Satellites",
        f"  Spoofed PRNs (declared):  {prns(manifest['declared_spoof_svids'])}",
        f"  Spoofed PRNs (usable):    {prns(manifest['usable_spoof_svids'])}",
        f"  Spoofed PRNs (omitted):   {prns(manifest['omitted_spoof_svids'])}",
        f"  Reference satellite:      PRN {manifest['reference_svid']} "
        f"(ephemeris AoA {manifest['reference_alpha_deg']:.2f} deg)",
        f"  Authentic PRNs compared:  {prns(authentic)}",
        f"  Spoof PRNs w/o geometry:  {prns(manifest['unpaired_spoof_svids'])}",
        "",
        "Configuration",
        f"  Single-difference order:  {reference['effective_single_difference_order']}",
        f"  Baseline azimuth:         {reference['effective_baseline_azimuth_deg']:g} deg "
        f"({reference['reference_direction_under_configured_hypothesis']}, assumed)",
        f"  Baseline / wavelength:    {manifest['baseline_over_wavelength']:g}",
        f"  Angle reference:          {reference['selected_reference']} (selected by {reference['selected_by']})",
        f"  Physical orientation:     {'confirmed' if reference['physical_orientation_confirmed'] else 'NOT confirmed'}",
        f"  Prior AoA:                {config.prior_aoa_deg} deg",
        f"  Declared spoof branch m:  {config.spoof_branch_m}",
        f"  Detection threshold:      {config.detection_threshold_deg:g} deg",
        "",
        "Branch selection (spoof group)",
        f"  chosen_by:                {manifest['chosen_by']} (ambiguous: {'yes' if manifest['branch_ambiguous'] else 'no'}, "
        f"tie tolerance {first['tie_tolerance']:.4f}, metric {first['metric']})",
        f"  Selected branches:        {first['selected_branches']} at window start "
        f"(PRN-epochs per branch: {branch_counts})",
        f"  Alternative branches:     {first['alternative_branches']} "
        f"(uniform shift {int(first['alternative_shift']):+d}, group AoA ~{alternative:.2f} deg)",
        "",
        "Result",
        f"  Spoof-group DD-method AoA (median): {manifest['group_median_aoa_deg']:.2f} deg",
        f"  Opposite directed reference:        {reference['other_reference_aoa_deg']:.2f} deg",
        f"  Flagged on all branches:            PRN {prns(flagged)}",
        f"  Not flagged on all branches:        PRN {prns(not_flagged)} "
        "(ephemeris AoA within threshold of the spoof direction)" if len(not_flagged) else
        "  Not flagged on all branches:        none",
        "",
        table,
        "  Track = selected joint-source AoA per epoch (branch may change between epochs);",
        "  DD AoA / Std / Other branch = constant-m summaries from spoof_branch_detection.",
        "",
        "Authentic DD-method vs ephemeris-based AoA agreement (pooled)",
        *agreement_lines,
        "",
        f"Figures: {manifest['figure_count']} publication, {manifest['diagnostic_figure_count']} diagnostic; "
        f"formats {', '.join(manifest['figure_design']['formats'])}; "
        f"{manifest['total_figure_file_count']} files. Tables: "
        f"{', '.join(sorted({Path(path).stem for path in manifest['table_files']})) or 'none'}. "
        "Plotted data in data/, provenance in report_manifest.json.",
    ]
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    return text


def export_saved_experiment(
    experiment_file: str | Path, output_dir: str | Path, *,
    design: dict | None = None, compact_epochs: int | None = None, display_duration_s: float | None = None,
    extend_authentic: bool = False, authentic_max_error_deg: float | None = None, filter_figures: bool = True,
    include_diagnostics: bool = True, show: bool = True,
) -> dict:
    """Export a verified saved experiment without relying on another notebook's kernel."""
    from aoa_experiment import angle_reference_summary, replay_experiment

    config, scenario = replay_experiment(experiment_file)
    design = load_figure_design() if design is None else design
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = export_paper_figures(
        scenario, output_dir, declared_spoof_svids=config.spoof_svids,
        design=design, compact_epochs=compact_epochs, display_duration_s=display_duration_s,
        extend_authentic=extend_authentic, authentic_max_error_deg=authentic_max_error_deg, filter_figures=filter_figures, show=show,
    )
    diagnostics = []
    if include_diagnostics:
        for files in (
            save_artifacts(scenario.analysis_result, output_dir / "diagnostics/phase",
                           figure_formats=design["formats"]),
            save_paper_scenario_artifacts(scenario, output_dir / "diagnostics/scenario",
                                         figure_formats=design["formats"]),
        ):
            diagnostics.extend(str(path.resolve()) for path in files.values()
                               if path.suffix.lstrip(".") in design["formats"])
    for filename in [*manifest["figure_files"], *diagnostics]:
        path = Path(filename)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty figure export: {path}")
    manifest.update(
        experiment_file=str(Path(experiment_file).resolve()),
        experiment_sha256=file_sha256(experiment_file),
        experiment_replay_verified=True, diagnostic_figure_files=diagnostics,
        diagnostic_figure_count=len(diagnostics) // len(design["formats"]),
        total_figure_file_count=len(manifest["figure_files"]) + len(diagnostics),
        angle_reference=angle_reference_summary(config, scenario),
    )
    orientation_audit = config.output_dir / "paper_receiver_orientation_hypotheses.csv"
    if config.angle_reference == "closest_to_expected" and orientation_audit.is_file():
        (output_dir / "data/receiver_orientation_hypotheses.csv").write_bytes(orientation_audit.read_bytes())
    manifest["sources_sha256"][str(Path(__file__).with_name("aoa_experiment.py").resolve())] = file_sha256(
        Path(__file__).with_name("aoa_experiment.py")
    )
    (output_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    write_run_report(output_dir / "report.txt", config, scenario, manifest)
    with (output_dir / "README.md").open("a", encoding="utf-8") as stream:
        stream.write(
            f"\nThe export replayed `{Path(experiment_file).resolve()}` and matched its saved DD and branch tables. "
            f"There are {len(FIGURE_STEMS)} publication figures and {manifest['diagnostic_figure_count']} "
            f"diagnostic figures, totaling {manifest['total_figure_file_count']} figure-format files. "
            "Diagnostics use the full analysis window and are in `diagnostics/phase/` and `diagnostics/scenario/`.\n"
        )
        reference = manifest["angle_reference"]
        stream.write(
            f"\nAngle reference: **{reference['selected_reference']}** relative to the input convention; "
            f"selected by `{reference['selected_by']}`. Effective subtraction: "
            f"`{reference['effective_single_difference_order']}`; effective baseline azimuth: "
            f"{reference['effective_baseline_azimuth_deg']:g} degrees "
            f"({reference['reference_direction_under_configured_hypothesis']} under the configured wiring hypothesis). "
            f"Expected angle: {reference['expected_aoa_deg']}. {reference['note']} "
            "The receiver-to-antenna mapping is not independently confirmed.\n"
        )
    return manifest


COMBINED_FRAMES = {
    "fractional": "fractional_double_difference_cycles",
    "standalone_ephemeris": "alpha_i_ephemeris_all_geometry_deg",
    "standalone_proposed": "alpha_i_proposed_best_m_deg",
    "ephemeris": "alpha_i_ephemeris_deg",
    "proposed": "alpha_i_proposed_best_m_paired_deg",
}


def export_combined_windows(
    window_dirs, output_dir: str | Path, *, design: dict | None = None, show: bool = True,
) -> dict:
    """Figures 1-4 of every exported window of one capture on a shared capture-time axis.

    Reads each window's plotted CSVs, so the combined traces equal the per-window
    figures. A NaN row between windows leaves the gap blank; nothing is interpolated.
    """
    design = load_figure_design() if design is None else design
    window_dirs = [Path(path) for path in window_dirs]
    output_dir = Path(output_dir).expanduser().resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((path / "report_manifest.json").read_text()) for path in window_dirs]
    for key in ("dataset_dir", "declared_spoof_svids"):
        if len({json.dumps(manifest[key]) for manifest in manifests}) != 1:
            raise ValueError(f"Windows disagree on {key}; combine windows of one capture only.")
    declared = tuple(manifests[0]["declared_spoof_svids"])

    # Each window's capture_elapsed_s restarts at that window's probe; receiver TOW is shared.
    order = np.argsort([manifest["tow_start_s"] for manifest in manifests])
    window_dirs, manifests = [window_dirs[i] for i in order], [manifests[i] for i in order]
    tow_zero = manifests[0]["tow_start_s"]
    parts = {name: [] for name in COMBINED_FRAMES}
    segments = []
    for window_dir, manifest in zip(window_dirs, manifests):
        epoch_map = pd.read_csv(window_dir / "data/selected_epoch_map.csv")
        time = pd.Index(epoch_map["rcvTow_rx1"].to_numpy(dtype=float) - tow_zero, name="elapsed_s")
        if segments and time[0] <= segments[-1]["end_s"]:
            raise ValueError(f"{window_dir.name} overlaps the previous window.")
        for name, stem in COMBINED_FRAMES.items():
            frame = pd.read_csv(window_dir / f"data/{stem}.csv", index_col="epoch_index")
            if list(frame.index) != epoch_map["epoch_index"].tolist():
                raise RuntimeError(f"{window_dir}/data/{stem}.csv does not match selected_epoch_map.csv")
            frame.columns = frame.columns.astype(int)
            frame.index = time
            if segments:
                # One blank sample in the gap breaks every line between windows.
                gap = (segments[-1]["end_s"] + time[0]) / 2
                parts[name].append(pd.DataFrame(np.nan, index=pd.Index([gap], name=time.name), columns=frame.columns))
            parts[name].append(frame)
        segments.append({
            "window": window_dir.name,
            "capture_epochs": f"{manifest['selected_source_epoch_indices'][0]}-{manifest['selected_source_epoch_indices'][-1]}",
            "start_s": float(time[0]), "end_s": float(time[-1]), "epochs": len(time),
            "tow_start_s": manifest["tow_start_s"], "tow_end_s": manifest["tow_end_s"],
            "reference_svid": manifest["reference_svid"], "chosen_by": manifest["chosen_by"],
            "group_median_aoa_deg": manifest["group_median_aoa_deg"],
        })
    frames = {name: pd.concat(frame_parts).sort_index(axis=1) for name, frame_parts in parts.items()}
    x = frames["fractional"].index.to_numpy(dtype=float)
    for frame in frames.values():
        np.testing.assert_array_equal(frame.index.to_numpy(dtype=float), x)

    prns = set().union(*(frame.columns for frame in frames.values()))
    styles = prn_styles(prns, declared, design)
    unpaired = sorted(prn for prn in declared if prn not in frames["proposed"] or frames["proposed"][prn].isna().all())
    paths = plot_figure_set(
        x, output_dir, design=design, declared=declared, styles=styles, unpaired_spoof=unpaired,
        xlabel=design["elapsed_axis_label"], show=show, **frames,
    )
    for name, frame in frames.items():
        frame.to_csv(data_dir / f"{COMBINED_FRAMES[name]}.csv")
    pd.DataFrame(segments).to_csv(data_dir / "window_segments.csv", index=False)
    references = sorted({segment["reference_svid"] for segment in segments})
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": manifests[0]["dataset_dir"],
        "declared_spoof_svids": declared,
        "window_dirs": [str(path.resolve()) for path in window_dirs],
        "window_segments": segments,
        "reference_svids": references,
        "time_axis": "seconds since the first plotted epoch (receiver 1 TOW minus the first window's start TOW)",
        "tow_zero_s": tow_zero,
        "gap_policy": "one NaN sample between windows; the unanalysed interval is left blank",
        "trace_source": "each window's data/*.csv as plotted in its own figures",
        "figure_files": paths,
    }
    (output_dir / "combined_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def combined_report_section(manifest: dict) -> str:
    lines = [
        "Combined figures (all windows on one capture-time axis)",
        "=======================================================",
        f"Folder: {Path(manifest['figure_files'][0]).parent}",
        f"Time axis: seconds since GPS TOW {manifest['tow_zero_s']:.0f} (first window start). "
        "The gap between windows is left blank; no values are interpolated.",
    ]
    for segment in manifest["window_segments"]:
        lines.append(
            f"  {segment['window']}: t = {segment['start_s']:.0f}-{segment['end_s']:.0f} s, "
            f"capture epochs {segment['capture_epochs']} "
            f"(TOW {segment['tow_start_s']:.0f}-{segment['tow_end_s']:.0f}, {segment['epochs']} epochs), "
            f"reference PRN {segment['reference_svid']}, chosen_by {segment['chosen_by']}, "
            f"spoof-group AoA {segment['group_median_aoa_deg']:.2f} deg"
        )
    if len(manifest["reference_svids"]) > 1:
        lines.append("  Note: the reference satellite changes between windows; fractional DD levels are not continuous.")
    return "\n".join(lines) + "\n"
