# GNSS AoA Estimation

This context describes the project language for estimating GNSS signal angle of arrival from two u-blox receiver logs using carrier-phase double differences.

## Language

**Baseline-relative AoA**:
The angle between a satellite signal line-of-sight and the antenna baseline vector.
_Avoid_: full bearing, spoofer azimuth

**DD-method AoA**:
The baseline-relative AoA estimated from carrier-phase double differences between the two receivers.
_Avoid_: ephemeris AoA, solved bearing

**Ephemeris-based AoA**:
The baseline-relative AoA derived from broadcast ephemeris, receiver position, and the known antenna baseline direction.
_Avoid_: recorded AoA, ground-truth bearing

**Carrier-phase measurement**:
The raw per-signal carrier-phase observable logged by the receiver, still containing ambiguity and measurement errors.
_Avoid_: solved ambiguity, resolved phase

**AoA pipeline diagnostic**:
A plot or table that exposes an intermediate step from raw carrier phase to DD-method AoA.
_Avoid_: final result, detector score

**Reference satellite**:
The satellite used as the subtraction anchor for carrier-phase double differences.
_Avoid_: ground truth satellite, fixed PRN

**DD-only run**:
A pipeline run that estimates AoA from receiver double differences without comparing against ephemeris-based AoA.
_Avoid_: full spoofing detector, ephemeris validation

**Reference-angle anchor**:
An assumed or externally known baseline-relative AoA for the reference satellite used to turn DD-only relative candidates into absolute AoA values.
_Avoid_: random angle, inferred absolute AoA

**Measurement dropout**:
A temporary degradation or loss of usable receiver observations that later returns to normal tracking.
_Avoid_: dataset failure, permanent outage

## Relationships

- A **DD-method AoA** estimates one **Baseline-relative AoA** for one satellite at one epoch.
- An **Ephemeris-based AoA** is used as a reference for comparing the **DD-method AoA**.
- A **Carrier-phase measurement** is an input to the double-difference computation, not the final **DD-method AoA**.
- An **AoA pipeline diagnostic** explains how the **DD-method AoA** was produced.
- A **Reference satellite** is selected automatically by default, with the design kept open for later manual input.
- A **DD-only run** is the first implementation target for the current full-spoofing dataset.
- A **DD-only run** needs a **Reference-angle anchor** before its output can be interpreted as absolute **Baseline-relative AoA**.
- A **Measurement dropout** must be visible in **AoA pipeline diagnostics**, not silently smoothed into normal data.
- A **Measurement dropout** breaks the continuity assumption used by DD-method smoothing.

## Example dialogue

> **Dev:** "Are we implementing spoofer localization?"
> **Domain expert:** "No. First implement **DD-method AoA** per satellite per epoch, then compare it with **Ephemeris-based AoA**."

> **Dev:** "Do we only need the final AoA chart?"
> **Domain expert:** "No. We also need **AoA pipeline diagnostics** for carrier phase, single difference, double difference, and candidate selection."

> **Dev:** "Should the user choose the reference satellite?"
> **Domain expert:** "Not for the first run. Select the **Reference satellite** automatically, but keep the implementation open for a manual choice later."

> **Dev:** "Should this dataset compare against ephemeris-based AoA?"
> **Domain expert:** "No. The current dataset is full spoofing, so first run the **DD-only run** and defer ephemeris comparison."

> **Dev:** "Can we choose a random reference angle?"
> **Domain expert:** "No. Without a **Reference-angle anchor**, the DD-only result is relative; a random anchor would create fake absolute AoA."

> **Dev:** "Should we clean up the periods where receiver observations go down and later recover?"
> **Domain expert:** "No. Mark each **Measurement dropout** so the diagnostics show where the AoA pipeline is operating on weak or missing data."

> **Dev:** "Should the smoother bridge across a dropout?"
> **Domain expert:** "No. Preserve the gap and restart smoothing on the next continuous valid segment."

## Flagged ambiguities

- "AoA" was used broadly; resolved: the implementation target is **DD-method AoA**, which is a **Baseline-relative AoA**, not full azimuth/elevation localization.
- "ephemeris-base AOA data" was used to mean data recorded directly by u-blox; resolved: u-blox records navigation and position data from which **Ephemeris-based AoA** can be derived.
- "alpha_dd_deg" in a **DD-only run** needs an explicit reference-angle anchor before it can be interpreted as an absolute **Baseline-relative AoA**.
