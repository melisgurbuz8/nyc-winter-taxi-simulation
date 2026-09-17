
from pathlib import Path
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# =========================================================
# CONFIG
# =========================================================

BASELINE_FILE = Path("data/processed/tlc_winter_baseline.csv")
MAP_DIR = Path("data/map")
MAP_DIR.mkdir(parents=True, exist_ok=True)

TAXI_ZONE_ZIP = MAP_DIR / "taxi_zones.zip"
TAXI_ZONE_DIR = MAP_DIR / "taxi_zones"
TAXI_ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

DEFAULT_SEED = 780
DEFAULT_RUNS = 500

NYC_BOROUGHS = [
    "Manhattan",
    "Brooklyn",
    "Queens",
    "Bronx",
    "Staten Island",
]

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
    "Unmet demand %": {
        "column": "avg_unmet_pct",
        "color_scale": "YlOrRd",
        "format": ".1%",
        "title": "Share of simulated ride demand that cannot be served",
    },
    "Service rate": {
        "column": "avg_service_rate",
        "color_scale": "RdYlGn",
        "format": ".1%",
        "title": "Share of simulated ride demand that is served",
    },
    "Average unmet rides": {
        "column": "avg_unmet_demand",
        "color_scale": "YlOrRd",
        "format": ".1f",
        "title": "Average simulated ride requests left unserved per hour",
    },
    "Simulated demand": {
        "column": "avg_demand",
        "color_scale": "Blues",
        "format": ".1f",
        "title": "Average simulated ride requests per hour",
    },
}


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_baseline():
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            "Missing data/processed/tlc_winter_baseline.csv. "
            "Run src/01_build_baseline.py first."
        )

    df = pd.read_csv(BASELINE_FILE)
    return df


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
def load_nyc_zones():
    shp_path = ensure_taxi_zone_files()
    zones = gpd.read_file(shp_path).to_crs(epsg=4326)

    # Normalize ID field
    id_col = None
    for col in ["LocationID", "locationid", "OBJECTID", "objectid"]:
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

    # Normalize borough field
    if "Borough" in zones.columns and "borough" not in zones.columns:
        zones = zones.rename(columns={"Borough": "borough"})

    if "borough" in zones.columns:
        zones = zones[
            zones["borough"].isin(NYC_BOROUGHS)
        ].copy()

    # Keep only NYC polygons and simplify very slightly for faster animation
    zones["geometry"] = zones["geometry"].simplify(
        tolerance=0.00015,
        preserve_topology=True,
    )

    # Representative points for labels
    points = zones.geometry.representative_point()
    zones["label_lon"] = points.x
    zones["label_lat"] = points.y

    return zones


# =========================================================
# SIMULATION
# =========================================================

def triangular_draw(rng, params, size):
    low, mode, high = params

    if low == mode == high:
        return np.full(size, low, dtype=float)

    return rng.triangular(
        low,
        mode,
        high,
        size=size,
    )


def simulate_zone(row, scenario_name, n_runs, seed):
    rng = np.random.default_rng(seed)
    params = SCENARIOS[scenario_name]

    demand_multiplier = triangular_draw(
        rng,
        params["demand"],
        n_runs,
    )

    supply_multiplier = triangular_draw(
        rng,
        params["supply"],
        n_runs,
    )

    duration_multiplier = triangular_draw(
        rng,
        params["duration"],
        n_runs,
    )

    snowfall = triangular_draw(
        rng,
        params["snowfall"],
        n_runs,
    )

    expected_demand = np.clip(
        float(row["baseline_trips"]) * demand_multiplier,
        0,
        None,
    )

    simulated_demand = rng.poisson(
        expected_demand
    )

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

    for _, row in subset.iterrows():
        zone_seed = int(
            seed
            + int(row["PULocationID"]) * 97
            + hour * 13
        )

        rows.append(
            simulate_zone(
                row=row,
                scenario_name=scenario_name,
                n_runs=n_runs,
                seed=zone_seed,
            )
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def simulate_full_day(
    scenario_name,
    day_type,
    n_runs,
    seed,
):
    frames = []

    for hour in range(24):
        hour_df = simulate_city_hour(
            scenario_name=scenario_name,
            day_type=day_type,
            hour=hour,
            n_runs=n_runs,
            seed=seed,
        ).copy()

        hour_df["hour"] = hour
        hour_df["hour_label"] = (
            f"{hour:02d}:00–{(hour + 1) % 24:02d}:00"
        )

        frames.append(hour_df)

    return pd.concat(
        frames,
        ignore_index=True,
    )


# =========================================================
# HELPERS
# =========================================================

def format_hour(hour):
    return f"{hour:02d}:00–{(hour + 1) % 24:02d}:00"


def city_kpis(results):
    demand = results["avg_demand"].sum()
    completed = results["avg_completed"].sum()
    unmet = results["avg_unmet_demand"].sum()

    service_rate = (
        completed / demand
        if demand > 0
        else 1.0
    )

    unmet_pct = (
        unmet / demand
        if demand > 0
        else 0.0
    )

    return {
        "demand": demand,
        "completed": completed,
        "unmet": unmet,
        "service_rate": service_rate,
        "unmet_pct": unmet_pct,
    }


def get_scale(metric_name, full_day):
    col = METRICS[metric_name]["column"]

    values = full_day[col].dropna()

    if len(values) == 0:
        return None

    if metric_name == "Unmet demand %":
        return [0, max(0.35, float(values.quantile(0.98)))]

    if metric_name == "Service rate":
        return [
            min(0.60, float(values.quantile(0.02))),
            1.0,
        ]

    return [
        0,
        max(
            1.0,
            float(values.quantile(0.98)),
        ),
    ]


def merge_results_with_zones(results):
    zones = load_nyc_zones()

    merged = zones.merge(
        results,
        on="PULocationID",
        how="left",
        suffixes=("_map", "_sim"),
    )

    if "borough_sim" in merged.columns:
        merged["display_borough"] = (
            merged["borough_sim"]
            .fillna(merged.get("borough_map"))
        )
    else:
        merged["display_borough"] = (
            merged.get("borough_map", "")
        )

    return merged


def make_snapshot_map(
    results,
    metric_name,
    full_day,
):
    metric = METRICS[metric_name]
    col = metric["column"]

    zones = load_nyc_zones().copy()
    modeled = zones.merge(
        results,
        on="PULocationID",
        how="inner",
        suffixes=("_map", "_sim"),
    )

    scale = get_scale(
        metric_name,
        full_day,
    )

    # Convert GeoDataFrames to GeoJSON once.
    all_geojson = zones.__geo_interface__
    modeled_geojson = modeled.__geo_interface__

    fig = go.Figure()

    # Layer 1: ALL NYC taxi zones in light gray.
    fig.add_trace(
        go.Choropleth(
            geojson=all_geojson,
            locations=zones["PULocationID"],
            featureidkey="properties.PULocationID",
            z=np.zeros(len(zones)),
            colorscale=[
                [0, "#eeeeee"],
                [1, "#eeeeee"],
            ],
            showscale=False,
            marker_line_color="#b8b8b8",
            marker_line_width=0.55,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "<i>Not modeled in top-20 baseline</i>"
                "<extra></extra>"
            ),
            customdata=np.column_stack(
                [
                    zones.get(
                        "zone",
                        pd.Series(
                            ["NYC taxi zone"] * len(zones),
                            index=zones.index,
                        ),
                    ).fillna("NYC taxi zone"),
                    zones.get(
                        "borough",
                        pd.Series(
                            ["NYC"] * len(zones),
                            index=zones.index,
                        ),
                    ).fillna("NYC"),
                ]
            ),
        )
    )

    # Layer 2: modeled top-20 zones colored by the selected metric.
    fig.add_trace(
        go.Choropleth(
            geojson=modeled_geojson,
            locations=modeled["PULocationID"],
            featureidkey="properties.PULocationID",
            z=modeled[col],
            colorscale=metric["color_scale"],
            zmin=scale[0] if scale else None,
            zmax=scale[1] if scale else None,
            colorbar=dict(
                title=metric_name,
                thickness=18,
                len=0.72,
            ),
            marker_line_color="#555555",
            marker_line_width=0.8,
            customdata=np.column_stack(
                [
                    modeled["pickup_zone"],
                    modeled["borough_sim"]
                    if "borough_sim" in modeled.columns
                    else modeled.get(
                        "borough_map",
                        pd.Series([""] * len(modeled)),
                    ),
                    modeled["avg_demand"],
                    modeled["avg_capacity"],
                    modeled["avg_unmet_demand"],
                    modeled["avg_unmet_pct"],
                    modeled["avg_service_rate"],
                ]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                f"{metric_name}: <b>%{{z:{metric['format']}}}</b><br>"
                "Demand: %{customdata[2]:.1f}<br>"
                "Capacity: %{customdata[3]:.1f}<br>"
                "Unmet rides: %{customdata[4]:.1f}<br>"
                "Unmet demand: %{customdata[5]:.1%}<br>"
                "Service rate: %{customdata[6]:.1%}"
                "<extra></extra>"
            ),
        )
    )

    # Label the five most stressed modeled zones.
    if metric_name == "Service rate":
        ranked = modeled.nsmallest(5, col)
    else:
        ranked = modeled.nlargest(5, col)

    fig.add_trace(
        go.Scattergeo(
            lon=ranked["label_lon"],
            lat=ranked["label_lat"],
            text=ranked["pickup_zone"],
            mode="text",
            textfont=dict(
                size=9,
                color="#222222",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Borough labels for geographic orientation.
    borough_labels = (
        zones.dissolve(by="borough")
        .reset_index()
    )
    borough_points = borough_labels.geometry.representative_point()

    fig.add_trace(
        go.Scattergeo(
            lon=borough_points.x,
            lat=borough_points.y,
            text=borough_labels["borough"],
            mode="text",
            textfont=dict(
                size=13,
                color="#666666",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=False,
        showsubunits=False,
        showcoastlines=False,
        showland=False,
        showlakes=False,
        bgcolor="white",
        projection_type="mercator",
    )

    fig.update_layout(
        height=690,
        margin=dict(
            l=0,
            r=0,
            t=5,
            b=0,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


def make_animated_map(
    full_day,
    metric_name,
):
    metric = METRICS[metric_name]
    col = metric["column"]

    zones = load_nyc_zones().copy()

    # Full-NYC static background.
    all_geojson = zones.__geo_interface__

    # Fixed daily color scale so colors are comparable across time.
    scale = get_scale(
        metric_name,
        full_day,
    )

    first_hour = full_day[
        full_day["hour"] == 0
    ].copy()

    first_modeled = zones.merge(
        first_hour,
        on="PULocationID",
        how="inner",
        suffixes=("_map", "_sim"),
    )

    fig = go.Figure()

    # Static NYC context layer.
    fig.add_trace(
        go.Choropleth(
            geojson=all_geojson,
            locations=zones["PULocationID"],
            featureidkey="properties.PULocationID",
            z=np.zeros(len(zones)),
            colorscale=[
                [0, "#eeeeee"],
                [1, "#eeeeee"],
            ],
            showscale=False,
            marker_line_color="#b8b8b8",
            marker_line_width=0.5,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "<i>NYC taxi zone</i>"
                "<extra></extra>"
            ),
            customdata=np.column_stack(
                [
                    zones.get(
                        "zone",
                        pd.Series(
                            ["NYC taxi zone"] * len(zones),
                            index=zones.index,
                        ),
                    ).fillna("NYC taxi zone"),
                    zones.get(
                        "borough",
                        pd.Series(
                            ["NYC"] * len(zones),
                            index=zones.index,
                        ),
                    ).fillna("NYC"),
                ]
            ),
        )
    )

    # Initial modeled overlay.
    fig.add_trace(
        go.Choropleth(
            geojson=first_modeled.__geo_interface__,
            locations=first_modeled["PULocationID"],
            featureidkey="properties.PULocationID",
            z=first_modeled[col],
            colorscale=metric["color_scale"],
            zmin=scale[0] if scale else None,
            zmax=scale[1] if scale else None,
            colorbar=dict(
                title=metric_name,
                thickness=18,
                len=0.72,
            ),
            marker_line_color="#555555",
            marker_line_width=0.8,
            customdata=np.column_stack(
                [
                    first_modeled["pickup_zone"],
                    first_modeled["borough_sim"]
                    if "borough_sim" in first_modeled.columns
                    else first_modeled.get(
                        "borough_map",
                        pd.Series([""] * len(first_modeled)),
                    ),
                    first_modeled["avg_demand"],
                    first_modeled["avg_unmet_demand"],
                    first_modeled["avg_unmet_pct"],
                    first_modeled["avg_service_rate"],
                ]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                f"{metric_name}: <b>%{{z:{metric['format']}}}</b><br>"
                "Demand: %{customdata[2]:.1f}<br>"
                "Unmet rides: %{customdata[3]:.1f}<br>"
                "Unmet demand: %{customdata[4]:.1%}<br>"
                "Service rate: %{customdata[5]:.1%}"
                "<extra></extra>"
            ),
        )
    )

    # Borough labels remain fixed for all animation frames.
    borough_labels = (
        zones.dissolve(by="borough")
        .reset_index()
    )
    borough_points = borough_labels.geometry.representative_point()

    fig.add_trace(
        go.Scattergeo(
            lon=borough_points.x,
            lat=borough_points.y,
            text=borough_labels["borough"],
            mode="text",
            textfont=dict(
                size=13,
                color="#666666",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    frames = []

    for hour in range(24):
        hour_df = full_day[
            full_day["hour"] == hour
        ].copy()

        modeled = zones.merge(
            hour_df,
            on="PULocationID",
            how="inner",
            suffixes=("_map", "_sim"),
        )

        frames.append(
            go.Frame(
                name=str(hour),
                data=[
                    go.Choropleth(
                        geojson=modeled.__geo_interface__,
                        locations=modeled["PULocationID"],
                        featureidkey="properties.PULocationID",
                        z=modeled[col],
                        colorscale=metric["color_scale"],
                        zmin=scale[0] if scale else None,
                        zmax=scale[1] if scale else None,
                        showscale=False,
                        marker_line_color="#555555",
                        marker_line_width=0.8,
                        customdata=np.column_stack(
                            [
                                modeled["pickup_zone"],
                                modeled["borough_sim"]
                                if "borough_sim" in modeled.columns
                                else modeled.get(
                                    "borough_map",
                                    pd.Series([""] * len(modeled)),
                                ),
                                modeled["avg_demand"],
                                modeled["avg_unmet_demand"],
                                modeled["avg_unmet_pct"],
                                modeled["avg_service_rate"],
                            ]
                        ),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "%{customdata[1]}<br>"
                            f"{metric_name}: <b>%{{z:{metric['format']}}}</b><br>"
                            "Demand: %{customdata[2]:.1f}<br>"
                            "Unmet rides: %{customdata[3]:.1f}<br>"
                            "Unmet demand: %{customdata[4]:.1%}<br>"
                            "Service rate: %{customdata[5]:.1%}"
                            "<extra></extra>"
                        ),
                    )
                ],
                traces=[1],
            )
        )

    fig.frames = frames

    steps = []

    for hour in range(24):
        steps.append(
            dict(
                method="animate",
                label=format_hour(hour),
                args=[
                    [str(hour)],
                    {
                        "mode": "immediate",
                        "frame": {
                            "duration": 700,
                            "redraw": True,
                        },
                        "transition": {
                            "duration": 200,
                        },
                    },
                ],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=-0.07,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {
                                    "duration": 850,
                                    "redraw": True,
                                },
                                "transition": {
                                    "duration": 250,
                                },
                            },
                        ],
                    ),
                    dict(
                        label="■ Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "transition": {
                                    "duration": 0,
                                },
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.14,
                y=-0.055,
                len=0.83,
                currentvalue=dict(
                    prefix="Time: ",
                    font=dict(size=13),
                ),
                steps=steps,
            )
        ],
        height=720,
        margin=dict(
            l=0,
            r=0,
            t=5,
            b=90,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=False,
        showsubunits=False,
        showcoastlines=False,
        showland=False,
        showlakes=False,
        bgcolor="white",
        projection_type="mercator",
    )

    return fig


def make_hourly_chart(
    full_day,
):
    hourly = (
        full_day.groupby(
            ["hour", "hour_label"],
            as_index=False,
        )
        .agg(
            demand=("avg_demand", "sum"),
            completed=("avg_completed", "sum"),
            unmet=("avg_unmet_demand", "sum"),
        )
    )

    hourly["service_rate"] = np.where(
        hourly["demand"] > 0,
        hourly["completed"] / hourly["demand"],
        1.0,
    )

    hourly["unmet_pct"] = np.where(
        hourly["demand"] > 0,
        hourly["unmet"] / hourly["demand"],
        0.0,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hourly["hour_label"],
            y=hourly["demand"],
            name="Demand",
            mode="lines+markers",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=hourly["hour_label"],
            y=hourly["completed"],
            name="Completed trips",
            mode="lines+markers",
        )
    )

    fig.add_trace(
        go.Bar(
            x=hourly["hour_label"],
            y=hourly["unmet"],
            name="Unmet demand",
            opacity=0.35,
        )
    )

    fig.update_layout(
        height=390,
        xaxis_title="Hour",
        yaxis_title="Simulated rides",
        hovermode="x unified",
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),
        legend=dict(
            orientation="h",
            y=1.12,
        ),
    )

    return fig


def make_top_zones_chart(
    results,
):
    top = (
        results.nlargest(
            7,
            "avg_unmet_pct",
        )
        .sort_values(
            "avg_unmet_pct",
            ascending=True,
        )
    )

    fig = px.bar(
        top,
        x="avg_unmet_pct",
        y="pickup_zone",
        orientation="h",
        text=top["avg_unmet_pct"].map(
            lambda x: f"{x:.1%}"
        ),
        labels={
            "avg_unmet_pct": "Unmet demand %",
            "pickup_zone": "",
        },
    )

    fig.update_layout(
        height=390,
        showlegend=False,
        xaxis_tickformat=".0%",
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),
    )

    return fig


def build_plain_english_summary(
    scenario,
    hour,
    kpis,
    results,
):
    worst = results.nlargest(
        1,
        "avg_unmet_pct",
    ).iloc[0]

    if kpis["unmet_pct"] < 0.03:
        severity = "little system-wide strain"
    elif kpis["unmet_pct"] < 0.10:
        severity = "noticeable service pressure"
    elif kpis["unmet_pct"] < 0.20:
        severity = "substantial service pressure"
    else:
        severity = "severe service disruption"

    return (
        f"At **{format_hour(hour)}** under **{scenario}**, "
        f"the modeled top-20 taxi zones experience **{severity}**. "
        f"About **{kpis['unmet_pct']:.1%}** of simulated ride demand "
        f"is left unserved overall. The most stressed modeled zone is "
        f"**{worst['pickup_zone']}**, where average unmet demand is "
        f"**{worst['avg_unmet_pct']:.1%}**."
    )


# =========================================================
# STREAMLIT APP
# =========================================================

st.set_page_config(
    page_title="NYC Winter Taxi Simulation",
    page_icon="🚕",
    layout="wide",
)

st.title("NYC Winter Taxi Stress Simulation")

st.caption(
    "A visual stress test of the 20 busiest NYC Yellow Taxi pickup zones "
    "using a historical TLC winter baseline and Monte Carlo winter-weather scenarios."
)

try:
    baseline = load_baseline()
    zones = load_nyc_zones()
except Exception as exc:
    st.error(str(exc))
    st.stop()


# =========================================================
# CONTROLS
# =========================================================

st.sidebar.header("Simulation controls")

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
    "Hour of day",
    min_value=0,
    max_value=23,
    value=17,
    step=1,
)

metric_name = st.sidebar.selectbox(
    "Map shows",
    list(METRICS.keys()),
    index=0,
)

n_runs = st.sidebar.select_slider(
    "Monte Carlo runs per zone",
    options=[100, 250, 500, 1000, 2000],
    value=500,
)

seed = st.sidebar.number_input(
    "Random seed",
    min_value=1,
    max_value=999999,
    value=DEFAULT_SEED,
    step=1,
)

st.sidebar.info(
    "Tip: Heavy Snow + Weekday + 17:00 usually makes the "
    "difference between demand and capacity easiest to see."
)


# =========================================================
# SIMULATE
# =========================================================

with st.spinner("Running NYC simulation..."):
    full_day = simulate_full_day(
        scenario_name=scenario,
        day_type=day_type,
        n_runs=n_runs,
        seed=int(seed),
    )

    current = full_day[
        full_day["hour"] == hour
    ].copy()

kpis = city_kpis(current)


# =========================================================
# KPI HEADER
# =========================================================

st.subheader(
    f"{scenario} · {day_type} · {format_hour(hour)}"
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Ride demand",
    f"{kpis['demand']:,.0f}",
)

k2.metric(
    "Trips completed",
    f"{kpis['completed']:,.0f}",
)

k3.metric(
    "Unmet rides",
    f"{kpis['unmet']:,.0f}",
)

k4.metric(
    "Service rate",
    f"{kpis['service_rate']:.1%}",
)

st.info(
    build_plain_english_summary(
        scenario=scenario,
        hour=hour,
        kpis=kpis,
        results=current,
    )
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📍 One-hour snapshot",
        "▶ Play the full day",
        "📊 Understand the pattern",
    ]
)


# =========================================================
# TAB 1: SNAPSHOT
# =========================================================

with tab1:
    st.markdown(
        f"#### NYC taxi-zone stress at {format_hour(hour)}"
    )

    st.caption(
        METRICS[metric_name]["title"]
    )

    snapshot_fig = make_snapshot_map(
        results=current,
        metric_name=metric_name,
        full_day=full_day,
    )

    st.plotly_chart(
        snapshot_fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    st.caption(
        "All NYC taxi zones are shown in light gray for geographic context. "
        "The 20 modeled high-volume pickup zones are colored by the selected simulation metric."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Most stressed modeled zones")
        st.plotly_chart(
            make_top_zones_chart(current),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with c2:
        st.markdown("#### What the colors mean")

        if metric_name == "Unmet demand %":
            st.write(
                "Darker red means a larger share of potential taxi rides "
                "cannot be served under the simulated conditions."
            )

        elif metric_name == "Service rate":
            st.write(
                "Green means most simulated demand is served. "
                "Red means the taxi system is unable to serve a larger share of demand."
            )

        elif metric_name == "Average unmet rides":
            st.write(
                "Darker zones have more simulated riders left without a completed taxi trip."
            )

        else:
            st.write(
                "Darker blue indicates greater simulated taxi demand."
            )

        worst = current.nlargest(
            5,
            "avg_unmet_pct",
        )[
            [
                "pickup_zone",
                "borough",
                "avg_demand",
                "avg_unmet_demand",
                "avg_unmet_pct",
                "avg_service_rate",
            ]
        ].copy()

        worst["avg_unmet_pct"] = (
            worst["avg_unmet_pct"] * 100
        ).round(1)

        worst["avg_service_rate"] = (
            worst["avg_service_rate"] * 100
        ).round(1)

        worst = worst.rename(
            columns={
                "pickup_zone": "Zone",
                "borough": "Borough",
                "avg_demand": "Demand",
                "avg_unmet_demand": "Unmet rides",
                "avg_unmet_pct": "Unmet %",
                "avg_service_rate": "Service %",
            }
        )

        st.dataframe(
            worst,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# TAB 2: TRUE PLAYABLE 24-HOUR ANIMATION
# =========================================================

with tab2:
    st.markdown(
        f"#### Watch {scenario.lower()} move through a {day_type.lower()} day"
    )

    st.write(
        "Use the **Play** button below the map or drag the time slider yourself. "
        "The full NYC taxi-zone map remains visible in light gray while the 20 modeled "
        "high-volume pickup zones change color as simulated system stress changes."
    )

    animated_fig = make_animated_map(
        full_day=full_day,
        metric_name=metric_name,
    )

    st.plotly_chart(
        animated_fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    st.caption(
        "The time slider runs from midnight through 11 PM. "
        "The color scale does not reset between hours, so a visible color change "
        "reflects a real change in the simulated metric."
    )


# =========================================================
# TAB 3: INTERPRETATION
# =========================================================

with tab3:
    left, right = st.columns(
        [1.35, 1]
    )

    with left:
        st.markdown("#### How taxi demand and service change through the day")

        st.plotly_chart(
            make_hourly_chart(full_day),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.write(
            "The gap between **Demand** and **Completed trips** is the simulated "
            "transportation pressure created when passenger demand exceeds available service."
        )

    with right:
        st.markdown("#### Peak system stress")

        hourly_summary = (
            full_day.groupby(
                ["hour", "hour_label"],
                as_index=False,
            )
            .agg(
                demand=("avg_demand", "sum"),
                completed=("avg_completed", "sum"),
                unmet=("avg_unmet_demand", "sum"),
            )
        )

        hourly_summary["unmet_pct"] = np.where(
            hourly_summary["demand"] > 0,
            hourly_summary["unmet"] / hourly_summary["demand"],
            0,
        )

        peak = hourly_summary.loc[
            hourly_summary["unmet_pct"].idxmax()
        ]

        st.metric(
            "Most stressed hour",
            peak["hour_label"],
        )

        st.metric(
            "Unmet demand at peak",
            f"{peak['unmet_pct']:.1%}",
        )

        st.metric(
            "Unserved rides at peak",
            f"{peak['unmet']:,.0f}",
        )

        st.markdown("#### How to interpret this simulation")

        st.write(
            "1. **Historical TLC data** provides the normal hourly pattern for each modeled zone."
        )

        st.write(
            "2. The selected winter scenario changes simulated passenger demand, taxi supply, "
            "and trip duration."
        )

        st.write(
            "3. When simulated demand exceeds available service capacity, the difference becomes "
            "**unmet demand**."
        )

        st.write(
            "4. The simulation does **not** claim these are observed failed taxi requests. "
            "They are scenario-based estimates created to stress-test the historical system."
        )


st.divider()

st.caption(
    "Model scope: the 20 busiest NYC Yellow Taxi pickup zones in the historical winter baseline. "
    "Synthetic scenario outputs are shown for analysis and are not observed rider-request data."
)
