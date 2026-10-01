# GNSS AoA Paper Methodology Context

Source paper: "GNSS Spoofing Detection by Estimating Angle of Arrival based on Carrier Phase Double Difference Measurements" by Hiep Hoang Van and Thuan Nguyen Dinh.
Zotero item: `BGYM97FT`.

This note is a compact, method-focused digest for LLM context. It summarizes the paper's algorithm and the repo-specific interpretation used in this project.

## Short Paper Summary

The paper proposes a GNSS spoofing detector that estimates a satellite signal's baseline-relative angle of arrival from carrier-phase double differences measured by two closely spaced receivers. The key idea is to compare two estimates of the same angle:

- `Ephemeris-based AoA`: computed from receiver position, broadcast ephemeris, satellite elevation, satellite azimuth, and known antenna baseline direction.
- `DD-method AoA`: computed from carrier-phase double differences between two receivers, using one reference satellite and a one-wavelength antenna baseline.

If the two AoA estimates agree within a threshold, the signal is treated as authentic. If they disagree, the signal is flagged as spoofed. In a single-source spoofing case, DD-method AoAs for spoofed satellites tend to collapse toward a common source direction while ephemeris-based AoAs remain satellite-specific.

## Core Definitions

- `Baseline-relative AoA`: angle between the incoming signal line of sight and the vector from antenna 1 to antenna 2.
- `Reference satellite`: satellite used as the subtraction anchor for double differences.
- `Reference-angle anchor`: the assumed or externally known baseline-relative AoA of the reference satellite.
- `DD-method AoA`: AoA inferred from double-difference carrier phase, not from ephemeris.
- `Ephemeris-based AoA`: AoA predicted from satellite geometry and user position.

## Measurement Model

For satellite `i`, receiver-side carrier phase contains geometric range, integer ambiguity, receiver and satellite clocks, propagation errors, and noise:

```text
phi_i = r_i + N_i * lambda + c * (sat_clock_i - receiver_clock) - iono_i + tropo_i + noise_i
```

With two nearby receivers, compute the receiver single difference for the same satellite:

```text
Delta_phi_i = phi_i_rx1 - phi_i_rx2
```

For closely spaced antennas, ionospheric and tropospheric differences are assumed negligible, and the satellite clock cancels. Receiver clock bias and integer ambiguity remain.

Then compute the satellite double difference using a reference satellite `r`:

```text
DD_phi_i = Delta_phi_i - Delta_phi_r
```

This removes the receiver clock term. The idealized double-difference relation is:

```text
DD_phi_i = D * (cos(alpha_i) - cos(alpha_r)) + lambda * DD_N_i + DD_noise_i
```

where:

- `D` is the baseline length between antennas.
- `lambda` is the carrier wavelength, about 0.19 m for GPS L1.
- `alpha_i` is the baseline-relative AoA for satellite `i`.
- `alpha_r` is the baseline-relative AoA for the reference satellite.
- `DD_N_i` is the remaining double-difference integer ambiguity.

## Ambiguity Handling

The paper's main trick is to set the antenna baseline length equal to one carrier wavelength:

```text
D = lambda
```

Convert double difference to cycles and remove the nearest integer:

```text
Phi_i = DD_phi_i / lambda - round(DD_phi_i / lambda)
```

Because `D / lambda = 1`, the fractional part is linked to:

```text
cos(alpha_i) - cos(alpha_r)
```

This avoids explicit integer ambiguity resolution. However, one still needs a reference-angle anchor, `alpha_r`, to convert the relative DD observation into an absolute baseline-relative AoA.

The remaining branch ambiguity is represented by a small integer `m`:

```text
m in {-2, -1, 0, 1, 2}
cos(alpha_i) = Phi_i + m + cos(alpha_r)
alpha_i = arccos(Phi_i + m + cos(alpha_r))
```

Invalid candidates are rejected when the arccos input falls outside `[-1, 1]`.

## Dynamic Programming Step

At each epoch, each satellite can have multiple valid AoA candidates from the branch set `m in {-2, -1, 0, 1, 2}`. The paper chooses a smooth time sequence using dynamic programming.

For each epoch `k`, select the candidate path that minimizes total change between consecutive epochs:

```text
cost_k(j) = min_i cost_{k-1}(i) + abs(alpha_k_j - alpha_{k-1}_i)
```

This encodes the assumption that true AoA changes smoothly over time. In this repo, smoothing should restart after measurement dropouts rather than bridging gaps.

## Detection Logic

The detector compares:

```text
abs(DD_method_AoA_i - Ephemeris_based_AoA_i)
```

against a threshold `tau`.

Paper interpretation:

- Difference <= `tau`: satellite is consistent with authentic geometry.
- Difference > `tau`: satellite is inconsistent and likely spoofed.

The paper reports that a 10 deg threshold works well in its two demonstration scenarios: clean open-sky data and all-spoofed data.

## Experimental Claims

Clean scenario:

- Two antennas separated by one GPS L1 wavelength.
- Open-sky data collected for about 3.5 hours.
- DD-method AoA is compared against ephemeris-based AoA.
- Reported average agreement rises as threshold is relaxed: about 61% at 5 deg, 85.7% at 7 deg, and 97.7% at 10 deg.

Spoofing scenario:

- Spoofer is directly connected to the two receiver antenna inputs, so all tracked satellite signals are counterfeit.
- DD-method AoAs converge toward a common source-like direction.
- Ephemeris-based AoAs remain satellite-specific.
- With a 10 deg threshold, essentially all spoofed satellites are rejected in the paper's test.

## Important Assumptions

- Two receiver antennas are closely spaced and have a known fixed baseline.
- Baseline length is exactly one carrier wavelength, or the implementation must explicitly account for `D / lambda`.
- The reference satellite has a trusted `alpha_r` when absolute AoA is required.
- For spoofing localization, at least one authentic satellite is needed as a reference.
- The true AoA changes smoothly across adjacent valid epochs.
- Measurement dropouts should remain visible and should break dynamic-programming continuity.
- Multipath and residual phase noise are small enough that rounding remains meaningful.

## Repo-Specific Interpretation

For this repo, avoid claiming full spoofer bearing from the paper algorithm. The implementation target is baseline-relative AoA, not full 2D or 3D direction finding.

Use these project terms consistently:

- `DD-method AoA` for the carrier-phase double-difference estimate.
- `Ephemeris-based AoA` for the geometry-derived comparison value.
- `Reference satellite` for the DD subtraction anchor.
- `Reference-angle anchor` for the known or assumed `alpha_r`.
- `DD-only run` when no trusted `alpha_r` or ephemeris comparison is available.

For all-spoofing datasets without a trusted reference-angle anchor, do not fabricate absolute AoA. Use relative fractional-DD convergence or collapse metrics instead.

## Implementation Mapping

| Paper concept | Repo-friendly artifact |
| --- | --- |
| Carrier phase observable | `carrier_phase_cycles` |
| Receiver single difference | `single_difference_cycles` |
| Satellite double difference | `double_difference_cycles` |
| Rounding operation | `rounded_dd_cycles` |
| Fractional DD after rounding | `fractional_dd_cycles` |
| Branch candidates from `m` values | `candidate_aoa_deg` |
| DP-selected smooth path | `selected_aoa_deg` |
| All-spoofing no-ground-truth diagnostic | fractional-DD collapse metrics |
| Ephemeris comparison detector | AoA difference threshold test |

## Minimal Algorithm

1. Parse synchronized carrier-phase measurements from two receivers.
2. Filter to the target signal, usually GPS L1.
3. Select or override a reference satellite.
4. Compute single differences between receivers for each satellite.
5. Compute double differences against the reference satellite.
6. Convert DD values to cycles and subtract `round(DD)`.
7. If `alpha_r` is trusted, generate AoA candidates for `m in {-2, -1, 0, 1, 2}`.
8. Use dynamic programming to choose a smooth candidate path per continuous segment.
9. Compute ephemeris-based AoA from position, ephemeris, baseline convention, elevation, and azimuth.
10. Compare DD-method AoA and ephemeris-based AoA with a threshold such as 10 deg.
11. For all-spoofing or DD-only runs, report collapse/convergence instead of absolute authenticity claims.

## Cautions for LLM Use

- Do not describe the paper as solving full spoofer localization; it mainly estimates and compares baseline-relative AoA.
- Do not treat receiver-reported satellite azimuth/elevation in an all-spoofing dataset as clean ground truth.
- Do not smooth across missing carrier-phase epochs.
- Do not infer absolute `alpha_i` without a valid `alpha_r`.
- Do not confuse "same DD-method AoA for many satellites" with proof of exact spoofer azimuth; it is evidence of common-source behavior relative to the baseline.
