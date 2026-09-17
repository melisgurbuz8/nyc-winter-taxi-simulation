
from pathlib import Path
import time
import textwrap

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

BASELINE_FILE = Path("data/processed/tlc_winter_baseline.csv")
DEFAULT_SEED = 780

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
# HELPERS
# ---------------------------------------------------------

@st.cache_data
def load_baseline():
    return pd.read_csv(BASELINE_FILE)


def tri(rng, params):
    low, mode, high = params
    if low == mode == high:
        return float(low)
    return float(rng.triangular(low, mode, high))


def build_one_hour_simulation(baseline_row, scenario_name, seed):
    rng = np.random.default_rng(seed)
    params = SCENARIOS[scenario_name]

    snowfall = tri(rng, params["snowfall"])
    demand_multiplier = tri(rng, params["demand"])
    supply_multiplier = tri(rng, params["supply"])
    duration_multiplier = tri(rng, params["duration"])

    baseline_trips = float(baseline_row["baseline_trips"])
    baseline_capacity = float(baseline_row["baseline_capacity_proxy"])
    baseline_duration = float(baseline_row["baseline_duration"])

    expected_hourly_demand = max(
        0,
        baseline_trips * demand_multiplier,
    )

    hourly_capacity = max(
        0,
        int(
            np.floor(
                baseline_capacity
                * supply_multiplier
                / duration_multiplier
            )
        ),
    )

    # Passenger arrivals by minute.
    arrivals = rng.poisson(
        lam=expected_hourly_demand / 60,
        size=60,
    )

    # Potential service slots by minute.
    raw_service_slots = rng.poisson(
        lam=max(hourly_capacity / 60, 0),
        size=60,
    )

    # Do not allow more than the drawn hourly capacity in total.
    service_slots = []
    remaining_capacity = hourly_capacity

    for x in raw_service_slots:
        slots = int(min(x, remaining_capacity))
        service_slots.append(slots)
        remaining_capacity -= slots

    service_slots = np.array(service_slots, dtype=int)

    # If random minute allocation did not use all hourly capacity,
    # distribute the remaining capacity across random minutes.
    while remaining_capacity > 0:
        minute = int(rng.integers(0, 60))
        service_slots[minute] += 1
        remaining_capacity -= 1

    queue = 0
    cumulative_arrivals = 0
    cumulative_completed = 0

    rows = []

    for minute in range(60):
        new_riders = int(arrivals[minute])
        cumulative_arrivals += new_riders
        queue += new_riders

        possible_service = int(service_slots[minute])
        completed = min(queue, possible_service)

        queue -= completed
        cumulative_completed += completed

        service_rate = (
            cumulative_completed / cumulative_arrivals
            if cumulative_arrivals > 0
            else 1.0
        )

        rows.append(
            {
                "minute": minute + 1,
                "new_riders": new_riders,
                "service_slots": possible_service,
                "completed_this_minute": completed,
                "waiting_queue": queue,
                "cumulative_demand": cumulative_arrivals,
                "cumulative_completed": cumulative_completed,
                "cumulative_unmet": queue,
                "service_rate": service_rate,
            }
        )

    sim = pd.DataFrame(rows)

    metadata = {
        "snowfall": snowfall,
        "demand_multiplier": demand_multiplier,
        "supply_multiplier": supply_multiplier,
        "duration_multiplier": duration_multiplier,
        "expected_hourly_demand": expected_hourly_demand,
        "hourly_capacity": hourly_capacity,
        "baseline_duration": baseline_duration,
        "simulated_duration": baseline_duration * duration_multiplier,
    }

    return sim, metadata


def people_icons(n, max_icons=24):
    if n <= 0:
        return "—"
    icons = min(max_icons, max(1, int(np.ceil(n / 3))))
    text = "🧍" * icons
    if n > icons * 3:
        text += f" +{n - icons * 3}"
    return text


def taxi_icons(capacity_ratio, max_icons=12):
    n = int(round(max_icons * max(0.05, min(1.0, capacity_ratio))))
    return "🚕" * max(1, n)


def snow_icons(inches):
    if inches <= 0:
        return "☁️"
    if inches < 1:
        return "❄️"
    if inches < 3:
        return "❄️ ❄️"
    if inches < 6:
        return "❄️ ❄️ ❄️"
    return "🌨️ ❄️ ❄️ ❄️ ❄️"


def street_scene_html(
    minute,
    scenario,
    snowfall,
    waiting,
    completed,
    new_riders,
    capacity_ratio,
    service_rate,
):
    riders = people_icons(waiting)
    taxis = taxi_icons(capacity_ratio)
    weather = snow_icons(snowfall)

    return textwrap.dedent(f"""\
    <div style="
        border:1px solid #d9d9d9;
        border-radius:14px;
        padding:18px;
        background:#f7f7f7;
        margin-bottom:14px;
    ">
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:12px;
            flex-wrap:wrap;
            margin-bottom:12px;
        ">
            <div style="font-size:24px;font-weight:700;">
                Minute {minute} / 60
            </div>
            <div style="font-size:20px;">
                {weather} {scenario} · {snowfall:.1f}" snow
            </div>
        </div>

        <div style="
            background:#dfe7ec;
            border-radius:12px;
            padding:14px;
            margin-bottom:8px;
        ">
            <div style="font-size:13px;font-weight:700;margin-bottom:6px;">
                CURB — WAITING RIDERS
            </div>
            <div style="font-size:24px;min-height:38px;">
                {riders}
            </div>
            <div style="font-size:14px;margin-top:6px;">
                Waiting now: <b>{waiting}</b> · New riders this minute: <b>{new_riders}</b>
            </div>
        </div>

        <div style="
            background:#373737;
            border-radius:12px;
            padding:16px;
            color:white;
            overflow:hidden;
        ">
            <div style="font-size:13px;font-weight:700;margin-bottom:10px;">
                TAXI SERVICE LANE
            </div>
            <div style="
                white-space:nowrap;
                font-size:28px;
                letter-spacing:5px;
                animation: taxiMove 1.2s linear infinite alternate;
            ">
                {taxis}
            </div>
            <div style="
                border-top:2px dashed #f0f0f0;
                margin:12px 0;
            "></div>
            <div style="
                display:flex;
                justify-content:space-between;
                gap:10px;
                flex-wrap:wrap;
                font-size:14px;
            ">
                <span>Completed so far: <b>{completed}</b></span>
                <span>Service rate: <b>{service_rate:.1%}</b></span>
            </div>
        </div>

        <style>
            @keyframes taxiMove {{
                from {{ transform: translateX(0px); }}
                to {{ transform: translateX(24px); }}
            }}
        </style>
    </div>
    """).strip()


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------

st.set_page_config(
    page_title="NYC Taxi Visual Winter Simulation",
    page_icon="🚕",
    layout="wide",
)

st.title("NYC Taxi Winter Visual Simulation")
st.caption(
    "Watch one simulated taxi-service hour unfold minute by minute."
)

if not BASELINE_FILE.exists():
    st.error(
        "Missing data/processed/tlc_winter_baseline.csv. "
        "Run src/01_build_baseline.py first."
    )
    st.stop()

baseline = load_baseline()

st.sidebar.header("Simulation Controls")

zone_lookup = (
    baseline[["PULocationID", "pickup_zone", "borough"]]
    .drop_duplicates()
    .sort_values(["borough", "pickup_zone"])
)

zone_labels = {
    f"{r.pickup_zone} — {r.borough}": r.PULocationID
    for r in zone_lookup.itertuples()
}

zone_label = st.sidebar.selectbox(
    "Pickup zone",
    list(zone_labels.keys()),
)

zone_id = zone_labels[zone_label]

day_type = st.sidebar.radio(
    "Day type",
    ["Weekday", "Weekend"],
)

hour = st.sidebar.slider(
    "Hour",
    0,
    23,
    17,
)

scenario = st.sidebar.selectbox(
    "Winter scenario",
    list(SCENARIOS.keys()),
    index=3,
)

speed_label = st.sidebar.select_slider(
    "Animation speed",
    options=["Slow", "Medium", "Fast"],
    value="Medium",
)

speed_map = {
    "Slow": 0.35,
    "Medium": 0.16,
    "Fast": 0.06,
}

seed = st.sidebar.number_input(
    "Random seed",
    min_value=1,
    value=DEFAULT_SEED,
    step=1,
)

match = baseline[
    (baseline["PULocationID"] == zone_id)
    & (baseline["day_type"] == day_type)
    & (baseline["hour"] == hour)
]

if len(match) != 1:
    st.error("No unique baseline row found.")
    st.stop()

baseline_row = match.iloc[0]

st.markdown(
    f"### {zone_label} · {day_type} · "
    f"{hour:02d}:00–{(hour + 1) % 24:02d}:00"
)

base1, base2, base3, base4 = st.columns(4)

base1.metric(
    "Historical trips / hour",
    f"{baseline_row['baseline_trips']:.0f}",
)
base2.metric(
    "Historical capacity proxy",
    f"{baseline_row['baseline_capacity_proxy']:.0f}",
)
base3.metric(
    "Median trip duration",
    f"{baseline_row['baseline_duration']:.1f} min",
)
base4.metric(
    "Median trip distance",
    f"{baseline_row['baseline_distance']:.2f} mi",
)

run = st.button(
    "▶ Run Visual Simulation",
    type="primary",
    use_container_width=True,
)

if run:
    sim, meta = build_one_hour_simulation(
        baseline_row,
        scenario,
        int(seed),
    )

    st.markdown("### Live simulation")

    weather1, weather2, weather3, weather4 = st.columns(4)

    weather1.metric(
        "Snowfall drawn",
        f"{meta['snowfall']:.1f}\"",
    )
    weather2.metric(
        "Demand multiplier",
        f"{meta['demand_multiplier']:.2f}×",
    )
    weather3.metric(
        "Supply multiplier",
        f"{meta['supply_multiplier']:.2f}×",
    )
    weather4.metric(
        "Trip-time multiplier",
        f"{meta['duration_multiplier']:.2f}×",
    )

    scene = st.empty()
    metrics = st.empty()
    chart = st.empty()
    progress = st.progress(0)

    baseline_capacity = max(
        float(baseline_row["baseline_capacity_proxy"]),
        1,
    )

    capacity_ratio = min(
        1.0,
        meta["hourly_capacity"] / baseline_capacity,
    )

    for i, row in sim.iterrows():
        minute = int(row["minute"])

        scene.html(
            street_scene_html(
                minute=minute,
                scenario=scenario,
                snowfall=meta["snowfall"],
                waiting=int(row["waiting_queue"]),
                completed=int(row["cumulative_completed"]),
                new_riders=int(row["new_riders"]),
                capacity_ratio=capacity_ratio,
                service_rate=float(row["service_rate"]),
                        )
        )

        with metrics.container():
            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Passengers arrived",
                int(row["cumulative_demand"]),
            )
            m2.metric(
                "Completed trips",
                int(row["cumulative_completed"]),
            )
            m3.metric(
                "Waiting / unmet now",
                int(row["waiting_queue"]),
            )
            m4.metric(
                "Current service rate",
                f"{row['service_rate']:.1%}",
            )

        history = sim.iloc[: i + 1][
            [
                "minute",
                "cumulative_demand",
                "cumulative_completed",
                "waiting_queue",
            ]
        ].set_index("minute")

        chart.line_chart(
            history,
            height=300,
        )

        progress.progress(minute / 60)
        time.sleep(speed_map[speed_label])

    final = sim.iloc[-1]

    st.success("Simulation complete.")

    st.markdown("### End-of-hour result")

    f1, f2, f3, f4 = st.columns(4)

    f1.metric(
        "Total passenger demand",
        int(final["cumulative_demand"]),
    )
    f2.metric(
        "Completed taxi trips",
        int(final["cumulative_completed"]),
    )
    f3.metric(
        "Unmet demand",
        int(final["waiting_queue"]),
    )
    f4.metric(
        "Final service rate",
        f"{final['service_rate']:.1%}",
    )

    st.markdown(
        f"""
        This particular run simulated **{meta['snowfall']:.1f} inches of snow**.
        The model estimated an hourly service capacity of **{meta['hourly_capacity']} trips**
        compared with an expected passenger demand of approximately
        **{meta['expected_hourly_demand']:.1f} trips**.
        """
    )

    st.caption(
        "The rider queue is a simulated estimate. TLC records completed trips, "
        "not unsuccessful taxi requests."
    )

else:
    st.info(
        "Choose a zone, time, and storm scenario, then press "
        "'Run Visual Simulation' to watch the hour unfold."
    )
