"""
F1 2026 ERS Telemetry Dashboard
================================
Launch with:
    streamlit run dashboard.py

Requires:
    pip install streamlit plotly fastf1 pandas numpy scipy
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# Import the inference engine
from f1_ers_inference import (
    load_session,
    get_car_data,
    compute_power_flow,
    MGU_K_DEPLOY_CAP_W,
    MGU_K_HARVEST_CAP_W,
    MAX_DEPLOY_PER_LAP,
)

# ── Colour palette for ERS states ────────────────────────────────────────────
STATE_COLOURS = {
    "deployment":       "#ef4444",  # red
    "braking_harvest":  "#22c55e",  # green
    "super_clipping":   "#f59e0b",  # amber
    "coast_harvest":    "#3b82f6",  # blue
    "partial_harvest":  "#8b5cf6",  # purple
}

STATE_LABELS = {
    "deployment":       "Deploying",
    "braking_harvest":  "Braking Harvest",
    "super_clipping":   "Super-Clipping",
    "coast_harvest":    "Coast Harvest",
    "partial_harvest":  "Partial Harvest",
}


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 ERS Telemetry",
    page_icon="🏎️",
    layout="wide",
)

st.title("F1 2026 ERS Telemetry Dashboard")


# ── Sidebar controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session")
    year = st.number_input("Year", min_value=2023, max_value=2030, value=2026)
    round_number = st.number_input("Round", min_value=1, max_value=24, value=1)
    session_type = st.selectbox("Session", ["R", "Q", "FP1", "FP2", "FP3", "S", "SS"])


# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading session data...")
def load_data(year, round_number, session_type):
    session = load_session(year, round_number, session_type)
    drivers = list(session.results["Abbreviation"])
    event_name = session.event["EventName"]
    return session, drivers, event_name


@st.cache_data(show_spinner="Computing ERS inference...")
def compute_driver_data(_session, driver):
    df = get_car_data(_session, driver)
    df = compute_power_flow(df)
    return df


try:
    session, drivers, event_name = load_data(year, round_number, session_type)
except Exception as e:
    st.error(f"Failed to load session: {e}")
    st.stop()

st.caption(f"**{event_name}** — {year} Round {round_number} ({session_type})")

with st.sidebar:
    driver = st.selectbox("Driver", drivers)

try:
    df = compute_driver_data(session, driver)
except Exception as e:
    st.error(f"Failed to load telemetry for {driver}: {e}")
    st.stop()


# ── Downsample for plotting performance ──────────────────────────────────────
def downsample(df, max_points=4000):
    if len(df) <= max_points:
        return df
    step = len(df) // max_points
    return df.iloc[::step].reset_index(drop=True)


plot_df = downsample(df)
dist = plot_df["Distance"] / 1000  # km


# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Mean SOC", f"{df['SOC_pct'].mean():.1f}%")
col2.metric("Min SOC", f"{df['SOC_pct'].min():.1f}%")
col3.metric("Max SOC", f"{df['SOC_pct'].max():.1f}%")

total_harvest_mj = df.loc[df["MGU_K_Power_W"] > 0, "MGU_K_Power_W"].sum() * (1 / 3.7) / 1e6
total_deploy_mj = df.loc[df["MGU_K_Power_W"] < 0, "MGU_K_Power_W"].abs().sum() * (1 / 3.7) / 1e6
col4.metric("Total Harvested", f"{total_harvest_mj:.2f} MJ")
col5.metric("Total Deployed", f"{total_deploy_mj:.2f} MJ")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHART: Speed + Throttle/Brake + Power + SOC (stacked)
# ══════════════════════════════════════════════════════════════════════════════

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.30, 0.20, 0.25, 0.25],
    subplot_titles=("Speed (km/h)", "Throttle & Brake", "MGU-K Power (kW)", "Battery SOC (%)"),
)

# ── Row 1: Speed trace coloured by ERS state ────────────────────────────────
for state, colour in STATE_COLOURS.items():
    mask = plot_df["ERS_state"] == state
    if mask.any():
        fig.add_trace(
            go.Scattergl(
                x=dist[mask],
                y=plot_df.loc[mask, "Speed"],
                mode="markers",
                marker=dict(color=colour, size=2),
                name=STATE_LABELS.get(state, state),
                legendgroup=state,
            ),
            row=1, col=1,
        )

# ── Row 2: Throttle & Brake ─────────────────────────────────────────────────
fig.add_trace(
    go.Scattergl(
        x=dist, y=plot_df["Throttle"],
        mode="lines", line=dict(color="#22c55e", width=1),
        name="Throttle %", legendgroup="inputs", showlegend=True,
    ),
    row=2, col=1,
)
fig.add_trace(
    go.Scattergl(
        x=dist, y=plot_df["Brake"].astype(int) * 100,
        mode="lines", line=dict(color="#ef4444", width=1),
        name="Brake", legendgroup="inputs", showlegend=True,
    ),
    row=2, col=1,
)

# ── Row 3: MGU-K Power ──────────────────────────────────────────────────────
power_kw = plot_df["MGU_K_Power_W"] / 1000
colours = np.where(power_kw >= 0, "#22c55e", "#ef4444")

fig.add_trace(
    go.Bar(
        x=dist, y=power_kw,
        marker_color=colours.tolist(),
        name="MGU-K Power", showlegend=False,
    ),
    row=3, col=1,
)

# Regulatory cap lines
fig.add_hline(y=MGU_K_HARVEST_CAP_W / 1000, line_dash="dash", line_color="gray",
              annotation_text="Harvest cap", row=3, col=1)
fig.add_hline(y=-MGU_K_DEPLOY_CAP_W / 1000, line_dash="dash", line_color="gray",
              annotation_text="Deploy cap", row=3, col=1)

# ── Row 4: SOC ──────────────────────────────────────────────────────────────
fig.add_trace(
    go.Scattergl(
        x=dist, y=plot_df["SOC_pct"],
        mode="lines", line=dict(color="#f59e0b", width=2),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.15)",
        name="SOC %", showlegend=False,
    ),
    row=4, col=1,
)

fig.update_xaxes(title_text="Distance (km)", row=4, col=1)
fig.update_yaxes(title_text="km/h", row=1, col=1)
fig.update_yaxes(title_text="%", range=[0, 105], row=2, col=1)
fig.update_yaxes(title_text="kW", row=3, col=1)
fig.update_yaxes(title_text="%", range=[0, 105], row=4, col=1)

fig.update_layout(
    height=900,
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(l=60, r=20, t=80, b=40),
    hovermode="x unified",
    bargap=0,
)

st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM ROW: ERS State Distribution + Gear Usage
# ══════════════════════════════════════════════════════════════════════════════

left, right = st.columns(2)

# ── ERS State pie chart ──────────────────────────────────────────────────────
with left:
    state_counts = df["ERS_state"].value_counts()
    fig_pie = go.Figure(
        go.Pie(
            labels=[STATE_LABELS.get(s, s) for s in state_counts.index],
            values=state_counts.values,
            marker=dict(colors=[STATE_COLOURS.get(s, "#666") for s in state_counts.index]),
            hole=0.4,
            textinfo="label+percent",
        )
    )
    fig_pie.update_layout(
        title="Time in Each ERS State",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Gear histogram with mean power ──────────────────────────────────────────
with right:
    gear_stats = df.groupby("nGear").agg(
        count=("nGear", "size"),
        mean_power_kw=("MGU_K_Power_W", lambda x: x.mean() / 1000),
    ).reset_index()

    fig_gear = make_subplots(specs=[[{"secondary_y": True}]])
    fig_gear.add_trace(
        go.Bar(
            x=gear_stats["nGear"], y=gear_stats["count"],
            name="Samples",
            marker_color="#3b82f6",
        ),
        secondary_y=False,
    )
    fig_gear.add_trace(
        go.Scatter(
            x=gear_stats["nGear"], y=gear_stats["mean_power_kw"],
            name="Mean MGU-K (kW)",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )
    fig_gear.update_layout(
        title="Gear Distribution & Mean MGU-K Power",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig_gear.update_xaxes(title_text="Gear", dtick=1)
    fig_gear.update_yaxes(title_text="Sample Count", secondary_y=False)
    fig_gear.update_yaxes(title_text="Mean Power (kW)", secondary_y=True)

    st.plotly_chart(fig_gear, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# LAP-BY-LAP SOC OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("Lap-by-Lap SOC Overlay")

# Detect lap boundaries where Distance resets
dist_diff = df["Distance"].diff()
lap_starts = df.index[dist_diff < -100].tolist()
lap_starts = [0] + lap_starts

if len(lap_starts) > 1:
    fig_laps = go.Figure()
    num_laps = min(len(lap_starts) - 1, 30)  # cap at 30 laps for readability
    colorscale = np.linspace(0, 1, num_laps)

    for i in range(num_laps):
        start = lap_starts[i]
        end = lap_starts[i + 1] if i + 1 < len(lap_starts) else len(df)
        lap_data = df.iloc[start:end]

        # Normalise distance within lap to [0, 1]
        d = lap_data["Distance"].values
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-9)

        # Colour from blue (early) to red (late)
        r = int(255 * colorscale[i])
        b = int(255 * (1 - colorscale[i]))
        colour = f"rgb({r}, 80, {b})"

        fig_laps.add_trace(
            go.Scattergl(
                x=d_norm,
                y=lap_data["SOC_pct"].values,
                mode="lines",
                line=dict(color=colour, width=1),
                name=f"Lap {i+1}",
                opacity=0.7,
            )
        )

    fig_laps.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Normalised Lap Distance",
        yaxis_title="SOC (%)",
        yaxis_range=[0, 105],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_laps, use_container_width=True)
else:
    st.info("Not enough lap data to generate overlay.")


# ── Raw data expander ────────────────────────────────────────────────────────
with st.expander("Raw Telemetry Data"):
    display_cols = [
        "Distance", "Speed", "Throttle", "Brake", "nGear",
        "ERS_state", "MGU_K_Power_W", "SOC_pct",
    ]
    st.dataframe(df[display_cols], use_container_width=True, height=400)
