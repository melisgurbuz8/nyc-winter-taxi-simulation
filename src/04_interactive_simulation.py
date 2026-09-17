
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BASELINE_FILE = Path("data/processed/tlc_winter_baseline.csv")
DEFAULT_SEED = 780

# Triangular distributions: (low, most likely, high)
SCENARIOS = {
    "Normal Winter": {
        "snowfall": (0.0, 0.0, 0.0),
        "demand": (0.98, 1.00, 1.02),
        "supply": (0.97, 1.00, 1.00),
        "duration": (0.97, 1.00, 1.03),
    },
    "Light Snow": {
        "snowfall": (0.1, 0.5, 1.0),
        "demand": (1.00, 1.04, 1.08),
        "supply": (0.90, 0.97, 1.00),
        "duration": (0.95, 1.03, 1.12),
    },
    "Moderate Snow": {
        "snowfall": (1.0, 2.0, 3.0),
        "demand": (0.95, 1.05, 1.15),
        "supply": (0.75, 0.88, 0.97),
        "duration": (0.90, 1.08, 1.25),
    },
    "Heavy Snow": {
        "snowfall": (3.0, 4.5, 6.0),
        "demand": (0.80, 0.95, 1.10),
        "supply": (0.50, 0.72, 0.90),
        "duration": (0.80, 1.12, 1.40),
    },
    "Severe Snowstorm": {
        "snowfall": (6.0, 8.0, 12.0),
        "demand": (0.60, 0.85, 1.05),
        "supply": (0.25, 0.50, 0.90),
        "duration": (0.70, 1.15, 1.60),
    },
}


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

@st.cache_data
def load_baseline():
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            "Baseline file not found. Run src/01_build_baseline.py first."
        )
    return pd.read_csv(BASELINE_FILE)


def triangular_draw(rng, params, size):
    low, mode, high = params

    # np.random.triangular does not allow low == mode == high.
    if low == mode == high:
        return np.full(size, low, dtype=float)

    return rng.triangular(low, mode, high, size=size)


def simulate_one_setting(baseline_row, scenario_name, n_runs, seed):
    params = SCENARIOS[scenario_name]
    rng = np.random.default_rng(seed)

    snow = triangular_draw(rng, params["snowfall"], n_runs)
    demand_mult = triangular_draw(rng, params["demand"], n_runs)
    supply_mult = triangular_draw(rng, params["supply"], n_runs)
    duration_mult = triangular_draw(rng, params["duration"], n_runs)

    baseline_trips = float(baseline_row["baseline_trips"])
    baseline_capacity = float(baseline_row["baseline_capacity_proxy"])
    baseline_duration = float(baseline_row["baseline_duration"])

    expected_demand = np.clip(
        baseline_trips * demand_mult,
        0,
        None,
    )

    simulated_demand = rng.poisson(expected_demand)

    simulated_duration = baseline_duration * duration_mult

    simulated_capacity = np.floor(
        baseline_capacity
        * supply_mult
        / duration_mult
    )
    simulated_capacity = np.clip(
        simulated_capacity,
        0,
        None,
    ).astype(int)

    completed_trips = np.minimum(
        simulated_demand,
        simulated_capacity,
    )

    unmet_demand = simulated_demand - completed_trips

    unmet_demand_pct = np.divide(
        unmet_demand,
        simulated_demand,
        out=np.zeros_like(unmet_demand, dtype=float),
        where=simulated_demand > 0,
    )

    service_rate = np.divide(
        completed_trips,
        simulated_demand,
        out=np.ones_like(completed_trips, dtype=float),
        where=simulated_demand > 0,
    )

    capacity_utilization = np.divide(
        completed_trips,
        simulated_capacity,
        out=np.zeros_like(completed_trips, dtype=float),
        where=simulated_capacity > 0,
    )

    return pd.DataFrame(
        {
            "simulation_run": np.arange(1, n_runs + 1),
            "scenario": scenario_name,
            "snowfall_inches": snow,
            "demand_multiplier": demand_mult,
            "supply_multiplier": supply_mult,
            "duration_multiplier": duration_mult,
            "expected_demand": expected_demand,
            "simulated_demand": simulated_demand,
            "simulated_duration": simulated_duration,
            "simulated_capacity": simulated_capacity,
            "completed_trips": completed_trips,
            "unmet_demand": unmet_demand,
            "unmet_demand_pct": unmet_demand_pct,
            "service_rate": service_rate,
            "capacity_utilization": capacity_utilization,
        }
    )


def summarize_runs(sim):
    return {
        "avg_demand": sim["simulated_demand"].mean(),
        "avg_capacity": sim["simulated_capacity"].mean(),
        "avg_completed": sim["completed_trips"].mean(),
        "avg_unmet": sim["unmet_demand"].mean(),
        "avg_unmet_pct": sim["unmet_demand_pct"].mean(),
        "avg_service_rate": sim["service_rate"].mean(),
        "avg_duration": sim["simulated_duration"].mean(),
        "prob_shortage": (sim["unmet_demand"] > 0).mean(),
    }


def compare_scenarios(baseline_row, n_runs, seed):
    rows = []

    for i, scenario in enumerate(SCENARIOS):
        sim = simulate_one_setting(
            baseline_row,
            scenario,
            n_runs,
            seed + i,
        )
        s = summarize_runs(sim)

        rows.append(
            {
                "Scenario": scenario,
                "Avg Demand": s["avg_demand"],
                "Avg Capacity": s["avg_capacity"],
                "Avg Completed Trips": s["avg_completed"],
                "Avg Unmet Demand": s["avg_unmet"],
                "Unmet Demand %": s["avg_unmet_pct"],
                "Service Rate": s["avg_service_rate"],
                "Shortage Probability": s["prob_shortage"],
                "Avg Trip Duration": s["avg_duration"],
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------

st.set_page_config(
    page_title="NYC Taxi Winter Simulation",
    page_icon="🚕",
    layout="wide",
)

st.title("NYC Taxi Winter Stress Test")
st.caption(
    "Monte Carlo simulation built from the historical TLC winter baseline. "
    "Simulation outputs are synthetic scenarios, not observed taxi demand."
)

try:
    baseline = load_baseline()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------

st.sidebar.header("Simulation Controls")

zone_lookup = (
    baseline[["PULocationID", "pickup_zone", "borough"]]
    .drop_duplicates()
    .sort_values(["borough", "pickup_zone"])
)

zone_labels = {
    f"{row.pickup_zone} — {row.borough}": row.PULocationID
    for row in zone_lookup.itertuples()
}

selected_zone_label = st.sidebar.selectbox(
    "Pickup zone",
    options=list(zone_labels.keys()),
)

selected_zone_id = zone_labels[selected_zone_label]

selected_day_type = st.sidebar.radio(
    "Day type",
    options=["Weekday", "Weekend"],
)

selected_hour = st.sidebar.slider(
    "Hour of day",
    min_value=0,
    max_value=23,
    value=17,
    step=1,
)

selected_scenario = st.sidebar.selectbox(
    "Winter scenario",
    options=list(SCENARIOS.keys()),
    index=3,
)

n_runs = st.sidebar.slider(
    "Monte Carlo runs",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
)

seed = st.sidebar.number_input(
    "Random seed",
    min_value=1,
    max_value=999999,
    value=DEFAULT_SEED,
    step=1,
)


# ---------------------------------------------------------
# SELECT BASELINE ROW
# ---------------------------------------------------------

match = baseline[
    (baseline["PULocationID"] == selected_zone_id)
    & (baseline["day_type"] == selected_day_type)
    & (baseline["hour"] == selected_hour)
]

if len(match) != 1:
    st.error(
        "Could not find exactly one historical baseline row "
        "for this zone, hour, and day type."
    )
    st.stop()

baseline_row = match.iloc[0]


# ---------------------------------------------------------
# BASELINE DISPLAY
# ---------------------------------------------------------

st.subheader("Historical TLC baseline")

b1, b2, b3, b4 = st.columns(4)

b1.metric(
    "Typical trips / hour",
    f"{baseline_row['baseline_trips']:.0f}",
)

b2.metric(
    "Capacity proxy",
    f"{baseline_row['baseline_capacity_proxy']:.0f}",
)

b3.metric(
    "Median trip duration",
    f"{baseline_row['baseline_duration']:.1f} min",
)

b4.metric(
    "Median trip distance",
    f"{baseline_row['baseline_distance']:.2f} mi",
)

st.caption(
    f"{selected_zone_label} | "
    f"{selected_day_type} | "
    f"{selected_hour:02d}:00–{(selected_hour + 1) % 24:02d}:00"
)


# ---------------------------------------------------------
# RUN SIMULATION
# ---------------------------------------------------------

sim = simulate_one_setting(
    baseline_row=baseline_row,
    scenario_name=selected_scenario,
    n_runs=n_runs,
    seed=int(seed),
)

summary = summarize_runs(sim)

st.subheader(f"Simulation results: {selected_scenario}")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Average demand",
    f"{summary['avg_demand']:.1f}",
)

c2.metric(
    "Average capacity",
    f"{summary['avg_capacity']:.1f}",
)

c3.metric(
    "Average completed trips",
    f"{summary['avg_completed']:.1f}",
)

c4.metric(
    "Average unmet demand",
    f"{summary['avg_unmet']:.1f}",
)

c5, c6, c7, c8 = st.columns(4)

c5.metric(
    "Average service rate",
    f"{summary['avg_service_rate']:.1%}",
)

c6.metric(
    "Average unmet demand %",
    f"{summary['avg_unmet_pct']:.1%}",
)

c7.metric(
    "Probability of shortage",
    f"{summary['prob_shortage']:.1%}",
)

c8.metric(
    "Average trip duration",
    f"{summary['avg_duration']:.1f} min",
)


# ---------------------------------------------------------
# DISTRIBUTION OF UNMET DEMAND
# ---------------------------------------------------------

left, right = st.columns(2)

with left:
    st.markdown("#### Distribution of unmet demand")

    fig, ax = plt.subplots()
    ax.hist(
        sim["unmet_demand"],
        bins=30,
    )
    ax.set_xlabel("Unmet ride requests per hour")
    ax.set_ylabel("Simulation runs")
    ax.set_title(
        f"{selected_scenario}: {n_runs:,} Monte Carlo runs"
    )
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("#### Demand vs. service capacity")

    chart_data = (
        sim[["simulated_demand", "simulated_capacity"]]
        .sample(
            n=min(150, len(sim)),
            random_state=int(seed),
        )
        .reset_index(drop=True)
    )

    st.line_chart(
        chart_data,
        y=["simulated_demand", "simulated_capacity"],
    )


# ---------------------------------------------------------
# COMPARE ALL FIVE SCENARIOS
# ---------------------------------------------------------

st.subheader("Compare all winter scenarios")

comparison = compare_scenarios(
    baseline_row=baseline_row,
    n_runs=n_runs,
    seed=int(seed),
)

display_comparison = comparison.copy()

for col in [
    "Avg Demand",
    "Avg Capacity",
    "Avg Completed Trips",
    "Avg Unmet Demand",
    "Avg Trip Duration",
]:
    display_comparison[col] = display_comparison[col].round(1)

for col in [
    "Unmet Demand %",
    "Service Rate",
    "Shortage Probability",
]:
    display_comparison[col] = (
        display_comparison[col]
        .map(lambda x: f"{x:.1%}")
    )

st.dataframe(
    display_comparison,
    use_container_width=True,
    hide_index=True,
)

st.markdown("#### Average unmet demand by scenario")

scenario_chart = comparison.set_index("Scenario")[
    ["Avg Unmet Demand"]
]

st.bar_chart(scenario_chart)


# ---------------------------------------------------------
# RAW RUNS + DOWNLOAD
# ---------------------------------------------------------

with st.expander("View individual simulation runs"):
    st.dataframe(
        sim.head(200),
        use_container_width=True,
        hide_index=True,
    )

csv = sim.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download selected simulation runs as CSV",
    data=csv,
    file_name="selected_winter_simulation_runs.csv",
    mime="text/csv",
)

st.info(
    "Interpretation: the historical TLC data provides the baseline. "
    "Winter conditions are modeled using explicit probability distributions. "
    "Unmet demand is a simulated quantity because TLC records completed trips, "
    "not unsuccessful taxi requests."
)
