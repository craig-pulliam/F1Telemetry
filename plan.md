# F1 2026 ERS Battery Inference Model — Implementation Plan

## Overview

The file `f1_ers_inference.py` contains a monolithic 441-line script combining data loading, an ERS inference model, a test suite, and a runner. This plan restructures the codebase into a well-organized, modular Python package while improving model correctness, performance, and testability.

---

## Phase 1: Project Scaffolding & Package Structure

**Goal:** Transform the single-file script into a proper Python package.

### Steps:
1. Create the package directory structure:
   ```
   F1Telemetry/
   ├── f1_ers/
   │   ├── __init__.py
   │   ├── constants.py        # 2026 regulatory constants
   │   ├── data.py             # Session loading & telemetry preprocessing
   │   ├── model.py            # ICE baseline, ERS state classification, power flow
   │   ├── tests.py            # Test suite (TestResult + 5 validation tests)
   │   └── runner.py           # CLI entry point (run_all_tests)
   ├── pyproject.toml          # Project metadata & dependencies
   └── f1_ers_inference.py     # Keep as thin wrapper for backward compat
   ```

2. Create `pyproject.toml` with dependencies: `fastf1`, `pandas`, `numpy`, `scipy`
3. Move constants (lines 27–35) into `constants.py`
4. Move `load_session` and `get_car_data` into `data.py`
5. Move `build_ice_baseline`, `classify_harvest_state`, `compute_power_flow` into `model.py`
6. Move `TestResult` class and all 5 test functions into `tests.py`
7. Move `run_all_tests` into `runner.py` with a `__main__` guard
8. Update `f1_ers_inference.py` to import and delegate to `runner.run_all_tests`

---

## Phase 2: Model Performance Optimization

**Goal:** Replace the slow row-by-row iteration with vectorized NumPy/Pandas operations.

### Steps:
1. **Vectorize `classify_harvest_state`** — Replace `df.apply(classify_harvest_state, axis=1)` with vectorized conditions using `np.select`:
   ```python
   conditions = [
       df["Brake"] > 0,
       df["Throttle"] == 0,
       (df["Throttle"] > 98) & (~np.isnan(df["ICE_baseline"])) & (df["Accel"] > df["ICE_baseline"] * 1.05),
       (df["Throttle"] > 98),
       df["Throttle"] < 80,
   ]
   choices = ["braking_harvest", "coast_harvest", "deployment", "super_clipping", "partial_harvest"]
   df["ERS_state"] = np.select(conditions, choices, default="deployment")
   ```

2. **Vectorize `compute_power_flow`** — Replace the `for i, row in df.iterrows()` loop with vectorized array operations:
   - Compute `ke_rate` arrays for each state mask
   - Use `np.where` and mask-based assignment instead of per-row branching

3. **Fix deprecated pandas calls** — Replace `fillna(method="bfill")` / `fillna(method="ffill")` in `build_ice_baseline` with `bfill()` / `ffill()` (deprecated in pandas 2.x)

---

## Phase 3: Model Correctness Improvements

**Goal:** Address edge cases and improve physical accuracy.

### Steps:
1. **Handle gear 0 / neutral** — `build_ice_baseline` only iterates gears 1–8; add handling for samples where `nGear == 0` (pit lane, standing starts)

2. **Per-lap energy tracking** — Currently `SOC_J` uses a global `cumsum()` across all laps. Refactor to:
   - Detect lap boundaries using the `Distance` column reset points
   - Reset per-lap harvest/deploy accumulators at each lap boundary
   - Keep a running total SOC that carries across laps

3. **Improve safety car detection** — Current implementation uses `total_seconds()` on timedelta which can lose precision. Add fallback for sessions where `race_control_messages` uses different time formats.

4. **Fix double-computation in `test_teammate_correlation`** — The function manually recomputes telemetry preprocessing (lines 327–333) instead of calling `get_car_data` + `compute_power_flow`. Refactor to reuse existing functions.

---

## Phase 4: Test Suite Improvements

**Goal:** Make tests more robust and add new validation tests.

### Steps:
1. **Convert to pytest** — Replace custom `TestResult` class with standard pytest tests:
   - Create `tests/` directory with `test_energy.py`, `test_clipping.py`, etc.
   - Use pytest fixtures for session loading and data preparation
   - Keep the custom runner as an alternative for quick CLI validation

2. **Add new tests:**
   - **DRS correlation test** — When DRS is active, deployment power should be high
   - **Pit lane power test** — In pit lane (low speed, gear 0/1), model should show minimal deployment
   - **Lap-over-lap consistency test** — SOC profile shape should be similar across consecutive clean laps

3. **Parameterize tests** — Allow tests to run across multiple drivers and sessions via pytest parameterization

---

## Phase 5: Configuration & Usability

**Goal:** Make the model configurable and easier to use.

### Steps:
1. **Add CLI argument parsing** — Use `argparse` in `runner.py`:
   - `--year` (default: 2026)
   - `--round` (default: 1)
   - `--session` (default: "R")
   - `--driver` (optional, auto-selects top finishers if omitted)
   - `--output` (optional CSV/JSON export path)

2. **Add results export** — Write inference results to CSV for downstream analysis:
   - Full telemetry DataFrame with all computed columns
   - Summary statistics JSON

3. **Configure FastF1 cache path** — Make cache directory configurable via environment variable or CLI flag instead of hardcoded `"f1_cache"`

---

## Implementation Order & Dependencies

| Step | Phase | Description | Depends On |
|------|-------|-------------|------------|
| 1 | Phase 1 | Create package structure & move code | — |
| 2 | Phase 3.4 | Fix double-computation in teammate test | Phase 1 |
| 3 | Phase 2 | Vectorize classify + power flow | Phase 1 |
| 4 | Phase 3.1–3.3 | Model correctness fixes | Phase 2 |
| 5 | Phase 2.3 | Fix deprecated pandas calls | Phase 1 |
| 6 | Phase 4 | Test suite improvements | Phase 3 |
| 7 | Phase 5 | CLI args & export | Phase 1 |

---

## Key Risks & Mitigations

- **FastF1 API changes**: Pin `fastf1` version in `pyproject.toml`; 2026 season data may not be available yet — model should gracefully handle missing data
- **Vectorization correctness**: Verify vectorized output matches original row-by-row output on test data before removing old code
- **State classification edge cases**: The boundary between `partial_harvest` (throttle < 80) and `deployment` (throttle 80–98) is a gap in the current logic — samples with 80 ≤ throttle ≤ 98 fall through to default `deployment`, which should be made explicit
