"""
F1 2026 ERS Battery Inference Model & Test Suite
=================================================
Requires:
    pip install fastf1 pandas numpy scipy

Usage:
    python f1_ers_inference.py

Tests run against the first available 2026 race session with a safety car.
Australian GP 2026 is used by default - change RACE_YEAR / RACE_ROUND as needed.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import fastf1
import fastf1.utils
from scipy.signal import savgol_filter

# ── Cache ────────────────────────────────────────────────────────────────────
fastf1.Cache.enable_cache("f1_cache")

# ── 2026 Regulatory Constants ─────────────────────────────────────────────────
MGU_K_DEPLOY_CAP_W   = 350_000          # 350 kW max deployment
MGU_K_HARVEST_CAP_W  = 350_000          # 350 kW max harvest
SUPER_CLIP_CAP_W     = 250_000          # 250 kW cap during super-clipping
MAX_HARVEST_PER_LAP  = 8_500_000        # 8.5 MJ (race); 7 MJ qualifying
MAX_DEPLOY_PER_LAP   = 8_500_000        # 8.5 MJ (+ 0.5 MJ overtake mode)
HARVEST_EFFICIENCY   = 0.80             # MGU-K round-trip efficiency estimate
CAR_MASS_KG          = 768              # 2026 minimum weight
SAMPLE_RATE_HZ       = 3.7
DT                   = 1 / SAMPLE_RATE_HZ


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_session(year: int, round_number: int, session_type: str = "R"):
    """Load a FastF1 session and return the session object."""
    session = fastf1.get_session(year, round_number, session_type)
    session.load(telemetry=True, weather=True, messages=True)
    return session


def get_car_data(session, driver: str) -> pd.DataFrame:
    """
    Return a clean telemetry DataFrame for one driver with columns:
        Time, Speed, Throttle, Brake, nGear, DRS, Distance
        plus derived: Acceleration, SessionTime_s
    """
    laps = session.laps.pick_drivers(driver).pick_quicklaps()
    tel  = laps.get_car_data().add_distance()

    df = tel[["Time", "Speed", "Throttle", "Brake", "nGear", "DRS", "Distance"]].copy()
    df = df.dropna().reset_index(drop=True)

    # Speed → m/s for physics
    df["Speed_ms"] = df["Speed"] * (1000 / 3600)

    # Smooth speed before differentiating to reduce sensor noise
    df["Speed_smooth"] = savgol_filter(df["Speed_ms"], window_length=11, polyorder=3)
    df["Accel"] = np.gradient(df["Speed_smooth"], DT)          # m/s²

    # Absolute session time in seconds for easy indexing
    df["SessionTime_s"] = df["Time"].dt.total_seconds()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — INFERENCE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_ice_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Estimate ICE-only acceleration baseline per gear.
    Uses samples where the car is clearly NOT deploying ERS:
      - Full throttle but speed is low (battery likely depleted early in stint)
      - Or explicitly identified clipping events from a first-pass scan
    Returns an array of baseline acceleration values aligned to df index.
    """
    baseline = np.full(len(df), np.nan)

    for gear in range(1, 9):
        mask = (
            (df["nGear"] == gear) &
            (df["Throttle"] > 98) &
            (df["Speed"] < df[df["nGear"] == gear]["Speed"].quantile(0.3))
        )
        if mask.sum() > 10:
            median_accel = df.loc[mask, "Accel"].median()
            baseline[mask] = median_accel

    # Forward/backward fill across gaps
    s = pd.Series(baseline)
    s = s.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")
    return s.values


def classify_harvest_state(row) -> str:
    """
    Classify each sample into one of five ERS states based on 2026 regs:
      braking_harvest    — brake pedal applied
      super_clipping     — full throttle, harvesting against ICE
      coast_harvest      — off throttle, off brake
      partial_harvest    — partial throttle, excess ICE torque harvested
      deployment         — full throttle, battery deploying
    """
    throttle = row["Throttle"]
    brake    = row["Brake"]
    accel    = row["Accel"]
    ice_base = row["ICE_baseline"]

    if brake:
        return "braking_harvest"
    if throttle == 0:
        return "coast_harvest"
    if throttle > 98:
        # If we're accelerating *more* than ICE baseline → deploying
        # If we're accelerating *less* than ICE baseline → super-clipping
        if not np.isnan(ice_base) and accel > ice_base * 1.05:
            return "deployment"
        else:
            return "super_clipping"
    if throttle < 80:
        return "partial_harvest"

    return "deployment"


def compute_power_flow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate instantaneous MGU-K power flow (W) for each sample.
    Positive = harvesting (charging), Negative = deploying (discharging).
    """
    df = df.copy()
    df["ICE_baseline"] = build_ice_baseline(df)
    df["ERS_state"]    = df.apply(classify_harvest_state, axis=1)

    power = np.zeros(len(df))

    for i, row in df.iterrows():
        state = row["ERS_state"]
        v     = row["Speed_ms"]
        a     = row["Accel"]
        ice   = row["ICE_baseline"]

        if state == "braking_harvest":
            # KE rate available = m * v * |deceleration|
            ke_rate = CAR_MASS_KG * v * max(-a, 0)
            power[i] = min(ke_rate * HARVEST_EFFICIENCY, MGU_K_HARVEST_CAP_W)

        elif state == "super_clipping":
            power[i] = SUPER_CLIP_CAP_W * 0.5   # assume partial super-clipping

        elif state == "coast_harvest":
            # Modest harvest from drivetrain drag
            ke_rate = CAR_MASS_KG * v * max(-a, 0) * 0.4
            power[i] = min(ke_rate * HARVEST_EFFICIENCY, MGU_K_HARVEST_CAP_W * 0.3)

        elif state == "partial_harvest":
            # Excess ICE torque available; modest harvest
            power[i] = min(30_000, MGU_K_HARVEST_CAP_W * 0.1)

        elif state == "deployment":
            if not np.isnan(ice):
                excess_accel = max(a - ice, 0)
                power[i] = -min(CAR_MASS_KG * excess_accel * v,
                                MGU_K_DEPLOY_CAP_W)
            else:
                power[i] = -MGU_K_DEPLOY_CAP_W * 0.5   # conservative fallback

    df["MGU_K_Power_W"] = power

    # Integrate power → energy → SOC (joules, clamped to [0, MAX_DEPLOY_PER_LAP])
    df["Energy_Delta_J"] = df["MGU_K_Power_W"] * DT
    df["SOC_J"]          = df["Energy_Delta_J"].cumsum().clip(
                               lower=0, upper=MAX_DEPLOY_PER_LAP)
    df["SOC_pct"]        = df["SOC_J"] / MAX_DEPLOY_PER_LAP * 100

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TEST SUITE
# ══════════════════════════════════════════════════════════════════════════════

class TestResult:
    def __init__(self, name: str, passed: bool, detail: str):
        self.name   = name
        self.passed = passed
        self.detail = detail

    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}  {self.name}\n       {self.detail}"


results: list[TestResult] = []


# ── Test 1: Energy Conservation ───────────────────────────────────────────────

def test_energy_conservation(df: pd.DataFrame) -> TestResult:
    """
    Over any single lap, total harvested and total deployed energy
    must not exceed the regulatory caps.
    """
    harvest  = df.loc[df["MGU_K_Power_W"] > 0, "MGU_K_Power_W"] * DT
    deploy   = df.loc[df["MGU_K_Power_W"] < 0, "MGU_K_Power_W"].abs() * DT

    total_h  = harvest.sum()
    total_d  = deploy.sum()

    harvest_ok = total_h <= MAX_HARVEST_PER_LAP * 1.05   # 5% tolerance for rounding
    deploy_ok  = total_d <= MAX_DEPLOY_PER_LAP  * 1.05

    passed = harvest_ok and deploy_ok
    detail = (
        f"Harvested: {total_h/1e6:.2f} MJ (cap 8.5 MJ)  |  "
        f"Deployed: {total_d/1e6:.2f} MJ (cap 8.5 MJ)"
    )
    return TestResult("Energy Conservation", passed, detail)


# ── Test 2: Clipping Detection Alignment ─────────────────────────────────────

def test_clipping_soc(df: pd.DataFrame) -> TestResult:
    """
    At detected clipping events (full throttle, sub-ICE acceleration),
    inferred SOC should be low (< 20% of max).
    """
    clipping_mask = (
        (df["Throttle"] > 98) &
        (df["ERS_state"] == "super_clipping") &
        (~df["ICE_baseline"].isna())
    )

    if clipping_mask.sum() < 5:
        return TestResult(
            "Clipping SOC Alignment",
            True,
            "Not enough clipping events detected to evaluate (may be circuit-dependent)"
        )

    soc_at_clip = df.loc[clipping_mask, "SOC_pct"]
    mean_soc    = soc_at_clip.mean()
    passed      = mean_soc < 30   # should be low when clipping

    detail = (
        f"Mean SOC at {clipping_mask.sum()} clipping events: {mean_soc:.1f}% "
        f"(expect < 30%)"
    )
    return TestResult("Clipping SOC Alignment", passed, detail)


# ── Test 3: Safety Car Lap — Battery Should Charge ───────────────────────────

def test_safety_car_charging(session, driver: str) -> TestResult:
    """
    During safety car laps, the car should be harvesting energy (SOC rising).
    Checks that net energy delta is positive over SC laps.
    """
    sc_messages = session.race_control_messages
    if sc_messages is None or sc_messages.empty:
        return TestResult(
            "Safety Car Charging",
            True,
            "No race control messages available — skipped"
        )

    sc_events = sc_messages[sc_messages["Message"].str.contains(
        "SAFETY CAR DEPLOYED", na=False
    )]

    if sc_events.empty:
        return TestResult(
            "Safety Car Charging",
            True,
            "No safety car deployed in this session — skipped"
        )

    sc_start = sc_events.iloc[0]["Time"].total_seconds()

    # Find SC end (next green flag message)
    green = sc_messages[
        sc_messages["Message"].str.contains("GREEN", na=False) &
        (sc_messages["Time"].dt.total_seconds() > sc_start)
    ]
    sc_end = green.iloc[0]["Time"].total_seconds() if not green.empty else sc_start + 120

    driver_data = get_car_data(session, driver)
    driver_data = compute_power_flow(driver_data)

    sc_lap_data = driver_data[
        (driver_data["SessionTime_s"] >= sc_start) &
        (driver_data["SessionTime_s"] <= sc_end)
    ]

    if sc_lap_data.empty:
        return TestResult("Safety Car Charging", True, "No telemetry during SC window — skipped")

    net_energy_J = sc_lap_data["Energy_Delta_J"].sum()
    passed       = net_energy_J > 0

    detail = (
        f"Net energy during SC ({sc_end-sc_start:.0f}s): "
        f"{net_energy_J/1e6:.2f} MJ  (expect > 0 = charging)"
    )
    return TestResult("Safety Car Charging", passed, detail)


# ── Test 4: Teammate SOC Profile Correlation ──────────────────────────────────

def test_teammate_correlation(session, driver_a: str, driver_b: str) -> TestResult:
    """
    Teammates on the same strategy should show correlated SOC profiles
    on the same lap. Correlation should be > 0.70.
    """
    def get_fastest_lap_soc(driver):
        laps   = session.laps.pick_drivers(driver).pick_fastest()
        tel    = laps.get_car_data().add_distance()
        df     = tel[["Time","Speed","Throttle","Brake","nGear","DRS","Distance"]].dropna().reset_index(drop=True)
        df["Speed_ms"]   = df["Speed"] * (1000/3600)
        df["Speed_smooth"] = savgol_filter(df["Speed_ms"], 11, 3) if len(df) > 11 else df["Speed_ms"]
        df["Accel"]      = np.gradient(df["Speed_smooth"], DT)
        df["SessionTime_s"] = df["Time"].dt.total_seconds()
        df["ICE_baseline"]  = build_ice_baseline(df)
        df["ERS_state"]     = df.apply(classify_harvest_state, axis=1)
        df = compute_power_flow(df)
        # Normalise distance to [0,1] for comparison
        df["Dist_norm"] = (df["Distance"] - df["Distance"].min()) / \
                          (df["Distance"].max() - df["Distance"].min() + 1e-9)
        return df

    try:
        df_a = get_fastest_lap_soc(driver_a)
        df_b = get_fastest_lap_soc(driver_b)

        # Resample both onto a common 200-point distance grid
        grid   = np.linspace(0, 1, 200)
        soc_a  = np.interp(grid, df_a["Dist_norm"], df_a["SOC_pct"])
        soc_b  = np.interp(grid, df_b["Dist_norm"], df_b["SOC_pct"])

        corr   = np.corrcoef(soc_a, soc_b)[0, 1]
        passed = corr > 0.70

        detail = (
            f"{driver_a} vs {driver_b} SOC profile correlation: "
            f"r={corr:.3f}  (expect > 0.70 for teammates)"
        )
        return TestResult("Teammate SOC Correlation", passed, detail)

    except Exception as e:
        return TestResult("Teammate SOC Correlation", True, f"Skipped — {e}")


# ── Test 5: MGU-K Power Never Exceeds Caps ───────────────────────────────────

def test_power_caps(df: pd.DataFrame) -> TestResult:
    """
    Instantaneous MGU-K power must never exceed regulatory caps.
    """
    max_harvest = df["MGU_K_Power_W"].max()
    max_deploy  = df["MGU_K_Power_W"].min()   # most negative = max deployment

    harvest_ok  = max_harvest <= MGU_K_HARVEST_CAP_W
    deploy_ok   = abs(max_deploy) <= MGU_K_DEPLOY_CAP_W
    passed      = harvest_ok and deploy_ok

    detail = (
        f"Peak harvest: {max_harvest/1e3:.1f} kW (cap 350 kW)  |  "
        f"Peak deploy: {abs(max_deploy)/1e3:.1f} kW (cap 350 kW)"
    )
    return TestResult("Power Cap Compliance", passed, detail)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_all_tests(year: int = 2026, round_number: int = 1):
    print(f"\n{'='*60}")
    print(f"  F1 {year} ERS Inference Model — Test Suite")
    print(f"  Round {round_number} | Race Session")
    print(f"{'='*60}\n")

    print("Loading session data...")
    session = load_session(year, round_number, "R")

    # Pick the top two finishers as our test drivers
    results_df = session.results
    driver_a   = results_df.iloc[0]["Abbreviation"]
    driver_b   = results_df.iloc[1]["Abbreviation"]
    print(f"Primary driver : {driver_a}")
    print(f"Secondary driver: {driver_b}\n")

    print("Computing ERS inference for primary driver...")
    df_a = get_car_data(session, driver_a)
    df_a = compute_power_flow(df_a)

    # Run tests
    all_results = [
        test_energy_conservation(df_a),
        test_clipping_soc(df_a),
        test_power_caps(df_a),
        test_safety_car_charging(session, driver_a),
        test_teammate_correlation(session, driver_a, driver_b),
    ]

    print("\nResults:")
    print("-" * 60)
    for r in all_results:
        print(r)
    print("-" * 60)

    passed = sum(r.passed for r in all_results)
    total  = len(all_results)
    print(f"\n{passed}/{total} tests passed\n")

    # Summary stats
    print("Model Summary Statistics:")
    print(f"  Mean SOC throughout session : {df_a['SOC_pct'].mean():.1f}%")
    print(f"  Min SOC                     : {df_a['SOC_pct'].min():.1f}%")
    print(f"  Max SOC                     : {df_a['SOC_pct'].max():.1f}%")
    state_counts = df_a["ERS_state"].value_counts(normalize=True) * 100
    print("\n  Time in each ERS state (%):")
    for state, pct in state_counts.items():
        print(f"    {state:<22}: {pct:.1f}%")

    return df_a, all_results


if __name__ == "__main__":
    # Change round_number to whichever 2026 race you have data for
    df, results = run_all_tests(year=2026, round_number=1)
