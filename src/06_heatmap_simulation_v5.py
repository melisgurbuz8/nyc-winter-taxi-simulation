
from pathlib import Path
import io
import time
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st


# ---------------------------------------------------------
# FILES + MODEL SETTINGS
# ---------------------------------------------------------

BASELINE_FILE = Path("data/processed/tlc_winter_baseline.csv")
MAP_DIR = Path("data/map")
MAP_DIR.mkdir(parents=True, exist_ok=True)

TAXI_ZONE_ZIP = MAP_DIR / "taxi_zones.zip"
TAXI_ZONE_DIR = MAP_DIR / "taxi_zones"
TAXI_ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

DEFAULT_SEED = 780
DEFAULT_RUNS = 500

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

METRICS = {
    "Unmet demand %": "avg_unmet_pct",
    "Service rate": "avg_service_rate",
    "Average unmet demand": "avg_unmet_demand",
    "Average simulated demand": "avg_demand",
    "Average simulated capacity": "avg_capacity",
    "Average completed trips": "avg_completed",
}


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def load_baseline():
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            "Missing data/processed/tlc_winter_baseline.csv. "
            "Run src/01_build_baseline.py first."
        )
    return pd.read_csv(BASELINE_FILE)


def ensure_taxi_zone_files():
    shp_files = list(TAXI_ZONE_DIR.rglob("*.shp"))
    if shp_files:
        return shp_files[0]

    TAXI_ZONE_DIR.mkdir(parents=True, exist_ok=True)

    if not TAXI_ZONE_ZIP.exists():
        response = requests.get(TAXI_ZONE_URL, timeout=120)
        response.raise_for_status()
        TAXI_ZONE_ZIP.write_bytes(response.content)

    with zipfile.ZipFile(TAXI_ZONE_ZIP, "r") as zf:
        zf.extractall(TAXI_ZONE_DIR)

    shp_files = list(TAXI_ZONE_DIR.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(
            "Taxi-zone shapefile could not be found after extraction."
        )
    return shp_files[0]


@st.cache_data
def load_taxi_zones():
    shp_path = ensure_taxi_zone_files()
    zones = gpd.read_file(shp_path)

    # TLC shapefile is projected; PyDeck needs longitude/latitude.
    zones = zones.to_crs(epsg=4326)

    # Normalize ID field.
    possible_id_columns = [
        "LocationID",
        "locationid",
        "OBJECTID",
        "objectid",
    ]

    id_col = None
    for col in possible_id_columns:
        if col in zones.columns:
            id_col = col
            break

    if id_col is None:
        raise ValueError(
            f"Could not identify taxi-zone ID field. "
            f"Columns found: {zones.columns.tolist()}"
        )

    zones = zones.rename(columns={id_col: "PULocationID"})
    zones["PULocationID"] = zones["PULocationID"].astype(int)

    # Keep only NYC boroughs so the map stays NYC-focused.
    possible_borough_cols = ["borough", "Borough"]
    borough_col = None

    for col in possible_borough_cols:
        if col in zones.columns:
            borough_col = col
            break

    if borough_col is not None and borough_col != "borough":
        zones = zones.rename(columns={borough_col: "borough"})

    if "borough" in zones.columns:
        keep_boroughs = [
            "Manhattan",
            "Brooklyn",
            "Queens",
            "Bronx",
            "Staten Island",
        ]
        zones = zones[zones["borough"].isin(keep_boroughs)].copy()

    return zones


# ---------------------------------------------------------
# SIMULATION
# ---------------------------------------------------------

def triangular_draw(rng, params, size):
    low, mode, high = params
    if low == mode == high:
        return np.full(size, low, dtype=float)
    return rng.triangular(low, mode, high, size=size)


def simulate_zone(row, scenario_name, n_runs, seed):
    rng = np.random.default_rng(seed)
    params = SCENARIOS[scenario_name]

    demand_multiplier = triangular_draw(
        rng, params["demand"], n_runs
    )
    supply_multiplier = triangular_draw(
        rng, params["supply"], n_runs
    )
    duration_multiplier = triangular_draw(
        rng, params["duration"], n_runs
    )
    snowfall = triangular_draw(
        rng, params["snowfall"], n_runs
    )

    expected_demand = np.clip(
        float(row["baseline_trips"]) * demand_multiplier,
        0,
        None,
    )

    simulated_demand = rng.poisson(expected_demand)

    simulated_capacity = np.floor(
        float(row["baseline_capacity_proxy"])
        * supply_multiplier
        / duration_multiplier
    )
    simulated_capacity = np.clip(
        simulated_capacity,
        0,
        None,
    ).astype(int)

    completed = np.minimum(
        simulated_demand,
        simulated_capacity,
    )

    unmet = simulated_demand - completed

    unmet_pct = np.divide(
        unmet,
        simulated_demand,
        out=np.zeros_like(unmet, dtype=float),
        where=simulated_demand > 0,
    )

    service_rate = np.divide(
        completed,
        simulated_demand,
        out=np.ones_like(completed, dtype=float),
        where=simulated_demand > 0,
    )

    return {
        "PULocationID": int(row["PULocationID"]),
        "pickup_zone": row["pickup_zone"],
        "borough": row["borough"],
        "avg_snowfall": float(np.mean(snowfall)),
        "avg_demand": float(np.mean(simulated_demand)),
        "avg_capacity": float(np.mean(simulated_capacity)),
        "avg_completed": float(np.mean(completed)),
        "avg_unmet_demand": float(np.mean(unmet)),
        "avg_unmet_pct": float(np.mean(unmet_pct)),
        "avg_service_rate": float(np.mean(service_rate)),
        "shortage_probability": float(np.mean(unmet > 0)),
    }


@st.cache_data(show_spinner=False)
def simulate_city_hour(
    scenario_name,
    day_type,
    hour,
    n_runs,
    seed,
):
    baseline = load_baseline()

    subset = baseline[
        (baseline["day_type"] == day_type)
        & (baseline["hour"] == hour)
    ].copy()

    rows = []

    for idx, row in subset.iterrows():
        zone_seed = int(seed + int(row["PULocationID"]) * 97 + hour * 13)
        rows.append(
            simulate_zone(
                row=row,
                scenario_name=scenario_name,
                n_runs=n_runs,
                seed=zone_seed,
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# COLOR SCALE
# ---------------------------------------------------------

def normalize(values, scale_min=None, scale_max=None):
    values = np.asarray(values, dtype=float)

    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros(len(values))

    if scale_min is None:
        lo = np.nanpercentile(finite, 5)
    else:
        lo = float(scale_min)

    if scale_max is None:
        hi = np.nanpercentile(finite, 95)
    else:
        hi = float(scale_max)

    if hi <= lo:
        return np.zeros(len(values))

    return np.clip(
        (values - lo) / (hi - lo),
        0,
        1,
    )


def add_fill_colors(gdf, metric_col, scale_min=None, scale_max=None):
    out = gdf.copy()

    values = out[metric_col].astype(float).to_numpy()
    scaled = normalize(values, scale_min=scale_min, scale_max=scale_max)

    colors = []

    for value, s in zip(values, scaled):
        if not np.isfinite(value):
            # Unmodeled taxi zones.
            colors.append([215, 215, 215, 80])
            continue

        # Light blue -> amber -> dark red.
        if s < 0.5:
            t = s / 0.5
            start = np.array([222, 240, 255])
            end = np.array([247, 183, 77])
        else:
            t = (s - 0.5) / 0.5
            start = np.array([247, 183, 77])
            end = np.array([163, 20, 20])

        rgb = (
            start + t * (end - start)
        ).astype(int)

        colors.append(
            [int(rgb[0]), int(rgb[1]), int(rgb[2]), 190]
        )

    out["fill_color"] = colors
    return out


def prepare_map_data(
    city_results,
    metric_col,
    scale_min=None,
    scale_max=None,
):
    zones = load_taxi_zones()

    # Keep geometry for all zones so the full NYC taxi-zone footprint is visible.
    merged = zones.merge(
        city_results,
        on="PULocationID",
        how="left",
    )

    merged = add_fill_colors(
        merged,
        metric_col,
        scale_min=scale_min,
        scale_max=scale_max,
    )

    # Friendly tooltip values.
    merged["tooltip_zone"] = merged["pickup_zone"].fillna(
        merged.get("zone", pd.Series(index=merged.index, dtype=str))
    )

    if metric_col in ["avg_unmet_pct", "avg_service_rate", "shortage_probability"]:
        merged["tooltip_metric"] = merged[metric_col].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "Not modeled"
        )
    else:
        merged["tooltip_metric"] = merged[metric_col].map(
            lambda x: f"{x:.1f}" if pd.notna(x) else "Not modeled"
        )

    if "borough_y" in merged.columns and "borough_x" in merged.columns:
        merged["tooltip_borough"] = merged["borough_y"].fillna(
            merged["borough_x"]
        )
    elif "borough" in merged.columns:
        merged["tooltip_borough"] = merged["borough"]
    else:
        merged["tooltip_borough"] = ""

    return merged


# ---------------------------------------------------------
# MAP RENDERING
# ---------------------------------------------------------

def make_deck(
    city_results,
    metric_label,
    hour,
    scenario_name,
    scale_min=None,
    scale_max=None,
):
    metric_col = METRICS[metric_label]

    gdf = prepare_map_data(
        city_results,
        metric_col,
        scale_min=scale_min,
        scale_max=scale_max,
    )

    geojson = gdf.__geo_interface__

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        opacity=0.85,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[90, 90, 90, 100],
        line_width_min_pixels=0.5,
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=40.73,
        longitude=-73.94,
        zoom=10.55,
        pitch=0,
        bearing=0,
    )

    tooltip = {
        "html": (
            "<b>{tooltip_zone}</b><br/>"
            "{tooltip_borough}<br/>"
            f"{metric_label}: <b>{{tooltip_metric}}</b><br/>"
            "Demand: {avg_demand}<br/>"
            "Capacity: {avg_capacity}<br/>"
            "Unmet demand: {avg_unmet_demand}<br/>"
            "Service rate: {avg_service_rate}"
        ),
        "style": {
            "backgroundColor": "#202020",
            "color": "white",
            "fontSize": "13px",
        },
    }

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None,
    )


def format_hour(hour):
    return f"{hour:02d}:00–{(hour + 1) % 24:02d}:00"


# ---------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------

st.set_page_config(
    page_title="NYC Winter Taxi Heat Map",
    page_icon="🗺️",
    layout="wide",
)

st.title("NYC Taxi Winter Heat Map Simulation")
st.caption(
    "NYC-only simulated taxi stress by TLC pickup zone. "
    "Colored zones are the 20 busiest pickup zones in the historical winter baseline; "
    "gray zones are shown for geographic context but are not simulated."
)

try:
    baseline = load_baseline()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Simulation Controls")

scenario = st.sidebar.selectbox(
    "Winter scenario",
    list(SCENARIOS.keys()),
    index=3,
)

day_type = st.sidebar.radio(
    "Day type",
    ["Weekday", "Weekend"],
)

hour = st.sidebar.slider(
    "Hour",
    min_value=0,
    max_value=23,
    value=17,
    step=1,
)

metric_label = st.sidebar.selectbox(
    "Heat-map metric",
    list(METRICS.keys()),
    index=0,
)

n_runs = st.sidebar.slider(
    "Monte Carlo runs per zone",
    min_value=100,
    max_value=2000,
    value=DEFAULT_RUNS,
    step=100,
)

seed = st.sidebar.number_input(
    "Random seed",
    min_value=1,
    max_value=999999,
    value=DEFAULT_SEED,
    step=1,
)

animation_speed = st.sidebar.select_slider(
    "24-hour animation speed",
    options=["Slow", "Medium", "Fast"],
    value="Medium",
)

animation_delay = {
    "Slow": 2.0,
    "Medium": 1.0,
    "Fast": 0.4,
}[animation_speed]

st.sidebar.caption(
    "Approximate full-day runtime: "
    + {
        "Slow": "48 seconds",
        "Medium": "24 seconds",
        "Fast": "10 seconds",
    }[animation_speed]
)


# ---------------------------------------------------------
# CURRENT HOUR
# ---------------------------------------------------------

results = simulate_city_hour(
    scenario_name=scenario,
    day_type=day_type,
    hour=hour,
    n_runs=n_runs,
    seed=int(seed),
)

st.markdown(
    f"### {scenario} · {day_type} · {format_hour(hour)}"
)

m1, m2, m3, m4 = st.columns(4)

weighted_demand = results["avg_demand"].sum()
weighted_completed = results["avg_completed"].sum()
weighted_unmet = results["avg_unmet_demand"].sum()

city_service_rate = (
    weighted_completed / weighted_demand
    if weighted_demand > 0
    else 1.0
)

m1.metric(
    "Simulated demand",
    f"{weighted_demand:,.0f}",
)
m2.metric(
    "Completed trips",
    f"{weighted_completed:,.0f}",
)
m3.metric(
    "Unmet demand",
    f"{weighted_unmet:,.0f}",
)
m4.metric(
    "City service rate",
    f"{city_service_rate:.1%}",
)


# ---------------------------------------------------------
# MAP
# ---------------------------------------------------------

map_placeholder = st.empty()
title_placeholder = st.empty()

deck = make_deck(
    results,
    metric_label,
    hour,
    scenario,
)

map_placeholder.pydeck_chart(
    deck,
    use_container_width=True,
    height=620,
)

st.caption(
    "Heat scale is relative to the simulated values for the selected hour. "
    "Hover over a colored zone to inspect its simulated results."
)


# ---------------------------------------------------------
# 24-HOUR ANIMATION
# ---------------------------------------------------------

st.markdown("### Watch the city change over 24 hours")

animate = st.button(
    "▶ Play 24-hour heat-map animation",
    type="primary",
    use_container_width=True,
)

if animate:
    metric_col = METRICS[metric_label]

    # Precompute the full day first. This lets every frame use the SAME
    # color scale, so a darker/lighter zone actually means the metric changed.
    full_day = {}

    with st.spinner("Preparing the 24-hour simulation..."):
        for anim_hour in range(24):
            full_day[anim_hour] = simulate_city_hour(
                scenario_name=scenario,
                day_type=day_type,
                hour=anim_hour,
                n_runs=n_runs,
                seed=int(seed),
            )

    all_metric_values = pd.concat(
        [df[[metric_col]] for df in full_day.values()],
        ignore_index=True,
    )[metric_col]

    finite_values = all_metric_values[
        np.isfinite(all_metric_values)
    ]

    if len(finite_values) > 0:
        scale_min = float(
            np.nanpercentile(finite_values, 2)
        )
        scale_max = float(
            np.nanpercentile(finite_values, 98)
        )
    else:
        scale_min = 0.0
        scale_max = 1.0

    animation_status = st.empty()
    animation_metrics = st.empty()
    animation_progress = st.progress(0)

    for anim_hour in range(24):
        anim_results = full_day[anim_hour]

        anim_demand = anim_results["avg_demand"].sum()
        anim_completed = anim_results["avg_completed"].sum()
        anim_unmet = anim_results["avg_unmet_demand"].sum()

        anim_service_rate = (
            anim_completed / anim_demand
            if anim_demand > 0
            else 1.0
        )

        anim_deck = make_deck(
            anim_results,
            metric_label,
            anim_hour,
            scenario,
            scale_min=scale_min,
            scale_max=scale_max,
        )

        animation_status.markdown(
            f"## ⏱ Hour {anim_hour + 1} of 24: "
            f"{format_hour(anim_hour)}"
        )

        with animation_metrics.container():
            a1, a2, a3, a4 = st.columns(4)

            a1.metric(
                "Demand this hour",
                f"{anim_demand:,.0f}",
            )
            a2.metric(
                "Completed trips",
                f"{anim_completed:,.0f}",
            )
            a3.metric(
                "Unmet demand",
                f"{anim_unmet:,.0f}",
            )
            a4.metric(
                "Service rate",
                f"{anim_service_rate:.1%}",
            )

        title_placeholder.markdown(
            f"### Playing: {scenario} · {day_type} · "
            f"{format_hour(anim_hour)} · {metric_label}"
        )

        map_placeholder.pydeck_chart(
            anim_deck,
            use_container_width=True,
            height=620,
        )

        animation_progress.progress(
            (anim_hour + 1) / 24
        )

        time.sleep(animation_delay)

    animation_status.success(
        "24-hour simulation complete."
    )


# ---------------------------------------------------------
# ZONE DETAILS
# ---------------------------------------------------------

st.markdown("### Compare simulated pickup zones")

display = results.copy()

display["Unmet demand %"] = (
    display["avg_unmet_pct"] * 100
).round(1)

display["Service rate %"] = (
    display["avg_service_rate"] * 100
).round(1)

display["Shortage probability %"] = (
    display["shortage_probability"] * 100
).round(1)

display = display[
    [
        "pickup_zone",
        "borough",
        "avg_demand",
        "avg_capacity",
        "avg_completed",
        "avg_unmet_demand",
        "Unmet demand %",
        "Service rate %",
        "Shortage probability %",
    ]
].rename(
    columns={
        "pickup_zone": "Pickup zone",
        "borough": "Borough",
        "avg_demand": "Avg demand",
        "avg_capacity": "Avg capacity",
        "avg_completed": "Avg completed trips",
        "avg_unmet_demand": "Avg unmet demand",
    }
)

display = display.sort_values(
    "Unmet demand %",
    ascending=False,
)

for col in [
    "Avg demand",
    "Avg capacity",
    "Avg completed trips",
    "Avg unmet demand",
]:
    display[col] = display[col].round(1)

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# ONE-ZONE PROFILE
# ---------------------------------------------------------

st.markdown("### Inspect one pickup zone")

zone_options = (
    results[["PULocationID", "pickup_zone", "borough"]]
    .sort_values(["borough", "pickup_zone"])
)

zone_label_map = {
    f"{row.pickup_zone} — {row.borough}": row.PULocationID
    for row in zone_options.itertuples()
}

detail_label = st.selectbox(
    "Pickup zone",
    list(zone_label_map.keys()),
)

detail_id = zone_label_map[detail_label]

detail = results[
    results["PULocationID"] == detail_id
].iloc[0]

d1, d2, d3, d4 = st.columns(4)

d1.metric(
    "Avg demand",
    f"{detail['avg_demand']:.1f}",
)
d2.metric(
    "Avg capacity",
    f"{detail['avg_capacity']:.1f}",
)
d3.metric(
    "Unmet demand",
    f"{detail['avg_unmet_demand']:.1f}",
)
d4.metric(
    "Service rate",
    f"{detail['avg_service_rate']:.1%}",
)

st.info(
    "This map visualizes synthetic winter scenarios derived from the TLC historical baseline. "
    "Unmet demand is simulated because TLC trip records contain completed trips, "
    "not unsuccessful taxi requests."
)
