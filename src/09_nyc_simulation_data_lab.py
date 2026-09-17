
from pathlib import Path
from datetime import datetime
import uuid
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# =========================================================
# CONFIG
# =========================================================

BASELINE_FILE = Path("data/processed/tlc_winter_baseline.csv")

RUN_DETAIL_FILE = Path(
    "data/processed/simulation_run_detail.csv"
)
RUN_SUMMARY_FILE = Path(
    "data/processed/simulation_run_summary.csv"
)

MAP_DIR = Path("data/map")
MAP_DIR.mkdir(parents=True, exist_ok=True)

TAXI_ZONE_ZIP = MAP_DIR / "taxi_zones.zip"
TAXI_ZONE_DIR = MAP_DIR / "taxi_zones"
TAXI_ZONE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/"
    "misc/taxi_zones.zip"
)

DEFAULT_BASE_SEED = 780

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
        "column": "unmet_demand_pct",
        "colorscale": "YlOrRd",
        "zmin": 0.0,
        "zmax": 0.40,
        "format": ".1%",
    },
    "Service rate": {
        "column": "service_rate",
        "colorscale": "RdYlGn",
        "zmin": 0.60,
        "zmax": 1.00,
        "format": ".1%",
    },
    "Unmet rides": {
        "column": "unmet_demand",
        "colorscale": "YlOrRd",
        "zmin": 0.0,
        "zmax": None,
        "format": ".0f",
    },
    "Simulated demand": {
        "column": "simulated_demand",
        "colorscale": "Blues",
        "zmin": 0.0,
        "zmax": None,
        "format": ".0f",
    },
}


# =========================================================
# LOAD INPUT DATA
# =========================================================

@st.cache_data
def load_baseline():
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            "Missing TLC baseline. Run "
            "src/01_build_baseline.py first."
        )

    return pd.read_csv(BASELINE_FILE)


def ensure_taxi_zone_files():
    shp_files = list(
        TAXI_ZONE_DIR.rglob("*.shp")
    )

    if shp_files:
        return shp_files[0]

    TAXI_ZONE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TAXI_ZONE_ZIP.exists():
        response = requests.get(
            TAXI_ZONE_URL,
            timeout=120,
        )
        response.raise_for_status()
        TAXI_ZONE_ZIP.write_bytes(
            response.content
        )

    with zipfile.ZipFile(
        TAXI_ZONE_ZIP,
        "r",
    ) as zf:
        zf.extractall(TAXI_ZONE_DIR)

    shp_files = list(
        TAXI_ZONE_DIR.rglob("*.shp")
    )

    if not shp_files:
        raise FileNotFoundError(
            "Taxi-zone shapefile could not "
            "be found after extraction."
        )

    return shp_files[0]


@st.cache_data
def load_nyc_zones():
    shp_path = ensure_taxi_zone_files()

    zones = (
        gpd.read_file(shp_path)
        .to_crs(epsg=4326)
    )

    id_col = None

    for col in [
        "LocationID",
        "locationid",
        "OBJECTID",
        "objectid",
    ]:
        if col in zones.columns:
            id_col = col
            break

    if id_col is None:
        raise ValueError(
            "Could not identify taxi-zone "
            "location ID field."
        )

    zones = zones.rename(
        columns={
            id_col: "PULocationID"
        }
    )

    zones["PULocationID"] = (
        zones["PULocationID"]
        .astype(int)
    )

    if (
        "Borough" in zones.columns
        and "borough" not in zones.columns
    ):
        zones = zones.rename(
            columns={
                "Borough": "borough"
            }
        )

    if (
        "zone" not in zones.columns
        and "Zone" in zones.columns
    ):
        zones = zones.rename(
            columns={
                "Zone": "zone"
            }
        )

    if "borough" in zones.columns:
        zones = zones[
            zones["borough"].isin(
                NYC_BOROUGHS
            )
        ].copy()

    zones["geometry"] = (
        zones["geometry"]
        .simplify(
            tolerance=0.00015,
            preserve_topology=True,
        )
    )

    return zones


# =========================================================
# ONE STOCHASTIC RUN
# =========================================================

def triangular_value(
    rng,
    params,
):
    low, mode, high = params

    if low == mode == high:
        return float(low)

    return float(
        rng.triangular(
            low,
            mode,
            high,
        )
    )


def simulate_one_zone_hour(
    row,
    scenario_name,
    rng,
):
    params = SCENARIOS[
        scenario_name
    ]

    snowfall = triangular_value(
        rng,
        params["snowfall"],
    )

    demand_multiplier = (
        triangular_value(
            rng,
            params["demand"],
        )
    )

    supply_multiplier = (
        triangular_value(
            rng,
            params["supply"],
        )
    )

    duration_multiplier = (
        triangular_value(
            rng,
            params["duration"],
        )
    )

    expected_demand = max(
        0.0,
        float(
            row["baseline_trips"]
        )
        * demand_multiplier,
    )

    simulated_demand = int(
        rng.poisson(
            expected_demand
        )
    )

    simulated_capacity = int(
        max(
            0,
            np.floor(
                float(
                    row[
                        "baseline_capacity_proxy"
                    ]
                )
                * supply_multiplier
                / duration_multiplier
            ),
        )
    )

    completed_trips = min(
        simulated_demand,
        simulated_capacity,
    )

    unmet_demand = (
        simulated_demand
        - completed_trips
    )

    if simulated_demand > 0:
        unmet_pct = (
            unmet_demand
            / simulated_demand
        )

        service_rate = (
            completed_trips
            / simulated_demand
        )
    else:
        unmet_pct = 0.0
        service_rate = 1.0

    simulated_duration = (
        float(
            row["baseline_duration"]
        )
        * duration_multiplier
    )

    return {
        "PULocationID": int(
            row["PULocationID"]
        ),
        "pickup_zone": row[
            "pickup_zone"
        ],
        "borough": row["borough"],
        "hour": int(row["hour"]),
        "day_type": row[
            "day_type"
        ],
        "scenario": scenario_name,
        "snowfall_inches": snowfall,
        "baseline_trips": float(
            row["baseline_trips"]
        ),
        "baseline_capacity_proxy": (
            float(
                row[
                    "baseline_capacity_proxy"
                ]
            )
        ),
        "baseline_duration": float(
            row["baseline_duration"]
        ),
        "demand_multiplier": (
            demand_multiplier
        ),
        "supply_multiplier": (
            supply_multiplier
        ),
        "duration_multiplier": (
            duration_multiplier
        ),
        "expected_demand": (
            expected_demand
        ),
        "simulated_demand": (
            simulated_demand
        ),
        "simulated_capacity": (
            simulated_capacity
        ),
        "completed_trips": (
            completed_trips
        ),
        "unmet_demand": (
            unmet_demand
        ),
        "unmet_demand_pct": (
            unmet_pct
        ),
        "service_rate": (
            service_rate
        ),
        "simulated_duration": (
            simulated_duration
        ),
    }


def next_run_number():
    if not RUN_SUMMARY_FILE.exists():
        return 1

    try:
        summary = pd.read_csv(
            RUN_SUMMARY_FILE
        )

        if summary.empty:
            return 1

        return (
            int(
                summary[
                    "run_number"
                ].max()
            )
            + 1
        )

    except Exception:
        return 1


def generate_run_id(
    run_number,
):
    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    short = (
        uuid.uuid4()
        .hex[:6]
        .upper()
    )

    return (
        f"RUN_{run_number:04d}_"
        f"{stamp}_{short}"
    )


def simulate_full_day_run(
    scenario_name,
    day_type,
    effective_seed,
    run_number,
):
    baseline = load_baseline()

    subset = baseline[
        baseline[
            "day_type"
        ].eq(day_type)
    ].copy()

    rng = np.random.default_rng(
        effective_seed
    )

    records = []

    for _, row in subset.iterrows():
        records.append(
            simulate_one_zone_hour(
                row=row,
                scenario_name=(
                    scenario_name
                ),
                rng=rng,
            )
        )

    run_df = pd.DataFrame(
        records
    )

    run_id = generate_run_id(
        run_number
    )

    run_time = (
        datetime.now()
        .astimezone()
        .isoformat(
            timespec="seconds"
        )
    )

    run_df.insert(
        0,
        "run_id",
        run_id,
    )

    run_df.insert(
        1,
        "run_number",
        run_number,
    )

    run_df.insert(
        2,
        "run_timestamp",
        run_time,
    )

    run_df.insert(
        3,
        "effective_seed",
        effective_seed,
    )

    return run_df


# =========================================================
# SAVE / LOAD RUNS
# =========================================================

def append_csv(
    df,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = path.exists()

    df.to_csv(
        path,
        mode="a",
        header=not exists,
        index=False,
    )


def summarize_run(
    run_df,
):
    total_demand = (
        run_df[
            "simulated_demand"
        ].sum()
    )

    total_completed = (
        run_df[
            "completed_trips"
        ].sum()
    )

    total_unmet = (
        run_df[
            "unmet_demand"
        ].sum()
    )

    service_rate = (
        total_completed
        / total_demand
        if total_demand > 0
        else 1.0
    )

    unmet_pct = (
        total_unmet
        / total_demand
        if total_demand > 0
        else 0.0
    )

    hourly = (
        run_df.groupby(
            "hour",
            as_index=False,
        )
        .agg(
            demand=(
                "simulated_demand",
                "sum",
            ),
            unmet=(
                "unmet_demand",
                "sum",
            ),
        )
    )

    hourly[
        "unmet_pct"
    ] = np.where(
        hourly["demand"] > 0,
        hourly["unmet"]
        / hourly["demand"],
        0.0,
    )

    peak = hourly.loc[
        hourly[
            "unmet_pct"
        ].idxmax()
    ]

    worst_zone = (
        run_df.groupby(
            [
                "PULocationID",
                "pickup_zone",
                "borough",
            ],
            as_index=False,
        )
        .agg(
            demand=(
                "simulated_demand",
                "sum",
            ),
            unmet=(
                "unmet_demand",
                "sum",
            ),
        )
    )

    worst_zone[
        "unmet_pct"
    ] = np.where(
        worst_zone[
            "demand"
        ] > 0,
        worst_zone[
            "unmet"
        ]
        / worst_zone[
            "demand"
        ],
        0.0,
    )

    worst = worst_zone.loc[
        worst_zone[
            "unmet_pct"
        ].idxmax()
    ]

    return pd.DataFrame(
        [
            {
                "run_id": (
                    run_df[
                        "run_id"
                    ].iloc[0]
                ),
                "run_number": (
                    int(
                        run_df[
                            "run_number"
                        ].iloc[0]
                    )
                ),
                "run_timestamp": (
                    run_df[
                        "run_timestamp"
                    ].iloc[0]
                ),
                "effective_seed": (
                    int(
                        run_df[
                            "effective_seed"
                        ].iloc[0]
                    )
                ),
                "scenario": (
                    run_df[
                        "scenario"
                    ].iloc[0]
                ),
                "day_type": (
                    run_df[
                        "day_type"
                    ].iloc[0]
                ),
                "detail_rows": (
                    len(run_df)
                ),
                "total_demand": (
                    int(
                        total_demand
                    )
                ),
                "total_completed": (
                    int(
                        total_completed
                    )
                ),
                "total_unmet": (
                    int(
                        total_unmet
                    )
                ),
                "service_rate": (
                    service_rate
                ),
                "unmet_demand_pct": (
                    unmet_pct
                ),
                "peak_stress_hour": (
                    int(
                        peak["hour"]
                    )
                ),
                "peak_hour_unmet_pct": (
                    float(
                        peak[
                            "unmet_pct"
                        ]
                    )
                ),
                "worst_zone": (
                    worst[
                        "pickup_zone"
                    ]
                ),
                "worst_zone_unmet_pct": (
                    float(
                        worst[
                            "unmet_pct"
                        ]
                    )
                ),
            }
        ]
    )


def save_run(
    run_df,
):
    summary = summarize_run(
        run_df
    )

    append_csv(
        run_df,
        RUN_DETAIL_FILE,
    )

    append_csv(
        summary,
        RUN_SUMMARY_FILE,
    )

    return summary


def load_saved_detail():
    if not RUN_DETAIL_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(
        RUN_DETAIL_FILE
    )


def load_saved_summary():
    if not RUN_SUMMARY_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(
        RUN_SUMMARY_FILE
    )


# =========================================================
# MAP HELPERS
# =========================================================

def format_hour(
    hour,
):
    return (
        f"{hour:02d}:00–"
        f"{(hour + 1) % 24:02d}:00"
    )


def metric_scale(
    run_df,
    metric_name,
):
    cfg = METRICS[
        metric_name
    ]

    col = cfg["column"]

    if cfg["zmax"] is not None:
        return (
            cfg["zmin"],
            cfg["zmax"],
        )

    values = run_df[
        col
    ].dropna()

    if values.empty:
        return (
            0.0,
            1.0,
        )

    return (
        cfg["zmin"],
        max(
            1.0,
            float(
                values.quantile(
                    0.98
                )
            ),
        ),
    )


def full_nyc_background_trace(
    zones,
):
    return go.Choropleth(
        geojson=(
            zones.__geo_interface__
        ),
        locations=(
            zones[
                "PULocationID"
            ]
        ),
        featureidkey=(
            "properties."
            "PULocationID"
        ),
        z=np.zeros(
            len(zones)
        ),
        colorscale=[
            [0, "#f2f2f2"],
            [1, "#f2f2f2"],
        ],
        showscale=False,
        marker_line_color=(
            "#c5c5c5"
        ),
        marker_line_width=0.5,
        customdata=np.column_stack(
            [
                zones.get(
                    "zone",
                    pd.Series(
                        [
                            "NYC taxi zone"
                        ]
                        * len(zones)
                    ),
                ),
                zones.get(
                    "borough",
                    pd.Series(
                        ["NYC"]
                        * len(zones)
                    ),
                ),
            ]
        ),
        hovertemplate=(
            "<b>%{customdata[0]}"
            "</b><br>"
            "%{customdata[1]}"
            "<br>"
            "<i>NYC taxi zone"
            "</i>"
            "<extra></extra>"
        ),
    )


def modeled_trace(
    merged,
    metric_name,
    run_df,
    show_scale=True,
):
    cfg = METRICS[
        metric_name
    ]

    col = cfg["column"]

    zmin, zmax = metric_scale(
        run_df,
        metric_name,
    )

    return go.Choropleth(
        geojson=(
            merged.__geo_interface__
        ),
        locations=(
            merged[
                "PULocationID"
            ]
        ),
        featureidkey=(
            "properties."
            "PULocationID"
        ),
        z=merged[col],
        colorscale=(
            cfg[
                "colorscale"
            ]
        ),
        zmin=zmin,
        zmax=zmax,
        showscale=show_scale,
        colorbar=(
            dict(
                title=metric_name,
                thickness=18,
                len=0.70,
            )
            if show_scale
            else None
        ),
        marker_line_color=(
            "#505050"
        ),
        marker_line_width=0.8,
        customdata=np.column_stack(
            [
                merged[
                    "pickup_zone"
                ],
                merged[
                    "borough_sim"
                ],
                merged[
                    "simulated_demand"
                ],
                merged[
                    "simulated_capacity"
                ],
                merged[
                    "unmet_demand"
                ],
                merged[
                    "unmet_demand_pct"
                ],
                merged[
                    "service_rate"
                ],
            ]
        ),
        hovertemplate=(
            "<b>%{customdata[0]}"
            "</b><br>"
            "%{customdata[1]}"
            "<br>"
            f"{metric_name}: "
            f"<b>%{{z:{cfg['format']}}}"
            "</b><br>"
            "Demand: "
            "%{customdata[2]:.0f}"
            "<br>"
            "Capacity: "
            "%{customdata[3]:.0f}"
            "<br>"
            "Unmet rides: "
            "%{customdata[4]:.0f}"
            "<br>"
            "Unmet demand: "
            "%{customdata[5]:.1%}"
            "<br>"
            "Service rate: "
            "%{customdata[6]:.1%}"
            "<extra></extra>"
        ),
    )


def borough_label_trace(
    zones,
):
    dissolved = (
        zones.dissolve(
            by="borough"
        )
        .reset_index()
    )

    points = (
        dissolved[
            "geometry"
        ].representative_point()
    )

    return go.Scattergeo(
        lon=points.x,
        lat=points.y,
        text=(
            dissolved[
                "borough"
            ]
        ),
        mode="text",
        textfont=dict(
            size=12,
            color="#666666",
        ),
        hoverinfo="skip",
        showlegend=False,
    )


def build_snapshot_map(
    run_df,
    hour,
    metric_name,
):
    zones = load_nyc_zones()

    hour_df = run_df[
        run_df["hour"].eq(
            hour
        )
    ].copy()

    merged = zones.merge(
        hour_df,
        on="PULocationID",
        how="inner",
        suffixes=(
            "_map",
            "_sim",
        ),
    )

    fig = go.Figure()

    fig.add_trace(
        full_nyc_background_trace(
            zones
        )
    )

    fig.add_trace(
        modeled_trace(
            merged=merged,
            metric_name=(
                metric_name
            ),
            run_df=run_df,
            show_scale=True,
        )
    )

    fig.add_trace(
        borough_label_trace(
            zones
        )
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type=(
            "mercator"
        ),
        bgcolor="white",
    )

    fig.update_layout(
        height=650,
        margin=dict(
            l=0,
            r=0,
            t=5,
            b=0,
        ),
        paper_bgcolor="white",
    )

    return fig


def build_animated_map(
    run_df,
    metric_name,
):
    zones = load_nyc_zones()

    first = run_df[
        run_df["hour"].eq(0)
    ].copy()

    merged_first = zones.merge(
        first,
        on="PULocationID",
        how="inner",
        suffixes=(
            "_map",
            "_sim",
        ),
    )

    fig = go.Figure()

    fig.add_trace(
        full_nyc_background_trace(
            zones
        )
    )

    fig.add_trace(
        modeled_trace(
            merged=merged_first,
            metric_name=(
                metric_name
            ),
            run_df=run_df,
            show_scale=True,
        )
    )

    fig.add_trace(
        borough_label_trace(
            zones
        )
    )

    frames = []

    for hour in range(24):
        hour_df = run_df[
            run_df[
                "hour"
            ].eq(hour)
        ].copy()

        merged = zones.merge(
            hour_df,
            on="PULocationID",
            how="inner",
            suffixes=(
                "_map",
                "_sim",
            ),
        )

        frames.append(
            go.Frame(
                name=str(hour),
                data=[
                    modeled_trace(
                        merged=merged,
                        metric_name=(
                            metric_name
                        ),
                        run_df=run_df,
                        show_scale=False,
                    )
                ],
                traces=[1],
            )
        )

    fig.frames = frames

    slider_steps = []

    for hour in range(24):
        slider_steps.append(
            dict(
                label=(
                    format_hour(
                        hour
                    )
                ),
                method="animate",
                args=[
                    [str(hour)],
                    {
                        "mode": (
                            "immediate"
                        ),
                        "frame": {
                            "duration": 600,
                            "redraw": True,
                        },
                        "transition": {
                            "duration": 150,
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
                y=-0.08,
                buttons=[
                    dict(
                        label=(
                            "▶ Play"
                        ),
                        method=(
                            "animate"
                        ),
                        args=[
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {
                                    "duration": 750,
                                    "redraw": True,
                                },
                                "transition": {
                                    "duration": 180,
                                },
                            },
                        ],
                    ),
                    dict(
                        label=(
                            "■ Pause"
                        ),
                        method=(
                            "animate"
                        ),
                        args=[
                            [None],
                            {
                                "mode": (
                                    "immediate"
                                ),
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
                x=0.16,
                y=-0.065,
                len=0.80,
                currentvalue=dict(
                    prefix="Time: ",
                ),
                steps=slider_steps,
            )
        ],
        height=720,
        margin=dict(
            l=0,
            r=0,
            t=5,
            b=95,
        ),
        paper_bgcolor="white",
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type=(
            "mercator"
        ),
        bgcolor="white",
    )

    return fig


# =========================================================
# CHART / SUMMARY HELPERS
# =========================================================

def run_totals(
    run_df,
):
    demand = int(
        run_df[
            "simulated_demand"
        ].sum()
    )

    completed = int(
        run_df[
            "completed_trips"
        ].sum()
    )

    unmet = int(
        run_df[
            "unmet_demand"
        ].sum()
    )

    return {
        "demand": demand,
        "completed": completed,
        "unmet": unmet,
        "service_rate": (
            completed / demand
            if demand > 0
            else 1.0
        ),
    }


def hour_totals(
    run_df,
    hour,
):
    df = run_df[
        run_df["hour"].eq(
            hour
        )
    ]

    demand = int(
        df[
            "simulated_demand"
        ].sum()
    )

    completed = int(
        df[
            "completed_trips"
        ].sum()
    )

    unmet = int(
        df[
            "unmet_demand"
        ].sum()
    )

    return {
        "demand": demand,
        "completed": completed,
        "unmet": unmet,
        "service_rate": (
            completed / demand
            if demand > 0
            else 1.0
        ),
    }


def hourly_chart(
    run_df,
):
    hourly = (
        run_df.groupby(
            "hour",
            as_index=False,
        )
        .agg(
            demand=(
                "simulated_demand",
                "sum",
            ),
            completed=(
                "completed_trips",
                "sum",
            ),
            unmet=(
                "unmet_demand",
                "sum",
            ),
        )
    )

    hourly["hour_label"] = (
        hourly["hour"]
        .map(format_hour)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hourly[
                "hour_label"
            ],
            y=hourly[
                "demand"
            ],
            name="Demand",
            mode="lines+markers",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=hourly[
                "hour_label"
            ],
            y=hourly[
                "completed"
            ],
            name="Completed trips",
            mode="lines+markers",
        )
    )

    fig.add_trace(
        go.Bar(
            x=hourly[
                "hour_label"
            ],
            y=hourly[
                "unmet"
            ],
            name="Unmet demand",
            opacity=0.35,
        )
    )

    fig.update_layout(
        height=390,
        hovermode="x unified",
        xaxis_title="Hour",
        yaxis_title="Rides",
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


def top_zone_table(
    run_df,
):
    grouped = (
        run_df.groupby(
            [
                "pickup_zone",
                "borough",
            ],
            as_index=False,
        )
        .agg(
            demand=(
                "simulated_demand",
                "sum",
            ),
            completed=(
                "completed_trips",
                "sum",
            ),
            unmet=(
                "unmet_demand",
                "sum",
            ),
        )
    )

    grouped[
        "unmet_pct"
    ] = np.where(
        grouped["demand"] > 0,
        grouped["unmet"]
        / grouped["demand"],
        0.0,
    )

    grouped[
        "service_rate"
    ] = np.where(
        grouped["demand"] > 0,
        grouped["completed"]
        / grouped["demand"],
        1.0,
    )

    grouped = grouped.sort_values(
        "unmet_pct",
        ascending=False,
    )

    display = grouped.head(
        10
    ).copy()

    display[
        "Unmet %"
    ] = (
        display[
            "unmet_pct"
        ]
        * 100
    ).round(1)

    display[
        "Service %"
    ] = (
        display[
            "service_rate"
        ]
        * 100
    ).round(1)

    return display[
        [
            "pickup_zone",
            "borough",
            "demand",
            "unmet",
            "Unmet %",
            "Service %",
        ]
    ].rename(
        columns={
            "pickup_zone": (
                "Pickup zone"
            ),
            "borough": "Borough",
            "demand": "Demand",
            "unmet": (
                "Unmet rides"
            ),
        }
    )


# =========================================================
# STREAMLIT APP
# =========================================================

st.set_page_config(
    page_title=(
        "NYC Taxi Simulation Lab"
    ),
    page_icon="🚕",
    layout="wide",
)

st.title(
    "NYC Winter Taxi Simulation Lab"
)

st.caption(
    "Each click creates and saves a new stochastic "
    "24-hour NYC taxi stress-test run."
)

try:
    baseline = load_baseline()
    zones = load_nyc_zones()
except Exception as exc:
    st.error(str(exc))
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "current_run" not in st.session_state:
    st.session_state[
        "current_run"
    ] = None

if (
    "current_summary"
    not in st.session_state
):
    st.session_state[
        "current_summary"
    ] = None


# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.header(
    "Experiment setup"
)

scenario = st.sidebar.selectbox(
    "Winter scenario",
    list(SCENARIOS.keys()),
    index=3,
)

day_type = st.sidebar.radio(
    "Day type",
    [
        "Weekday",
        "Weekend",
    ],
)

base_seed = st.sidebar.number_input(
    "Base seed",
    min_value=1,
    max_value=999999,
    value=DEFAULT_BASE_SEED,
    step=1,
)

batch_size = st.sidebar.selectbox(
    "How many runs to save now?",
    [
        1,
        5,
        10,
        25,
        50,
    ],
    index=0,
)

st.sidebar.caption(
    "Each run creates 480 rows: "
    "20 modeled zones × 24 hours."
)

run_button = st.sidebar.button(
    "▶ Run & save simulation",
    type="primary",
    use_container_width=True,
)


# =========================================================
# EXECUTE + SAVE RUN(S)
# =========================================================

if run_button:
    latest_run = None
    latest_summary = None

    start_number = (
        next_run_number()
    )

    progress = st.sidebar.progress(
        0
    )

    status = st.sidebar.empty()

    for i in range(batch_size):
        run_number = (
            start_number + i
        )

        effective_seed = (
            int(base_seed)
            + run_number * 100003
        )

        status.write(
            f"Running simulation "
            f"{i + 1} of "
            f"{batch_size}..."
        )

        run_df = (
            simulate_full_day_run(
                scenario_name=(
                    scenario
                ),
                day_type=(
                    day_type
                ),
                effective_seed=(
                    effective_seed
                ),
                run_number=(
                    run_number
                ),
            )
        )

        summary = save_run(
            run_df
        )

        latest_run = run_df
        latest_summary = summary

        progress.progress(
            (i + 1)
            / batch_size
        )

    st.session_state[
        "current_run"
    ] = latest_run

    st.session_state[
        "current_summary"
    ] = latest_summary

    status.success(
        f"Saved {batch_size} "
        f"simulation run"
        f"{'s' if batch_size > 1 else ''}."
    )

    st.rerun()


# =========================================================
# LOAD MOST RECENT SAVED RUN IF NEEDED
# =========================================================

if (
    st.session_state[
        "current_run"
    ]
    is None
):
    saved_detail = (
        load_saved_detail()
    )

    if not saved_detail.empty:
        latest_id = (
            saved_detail[
                "run_number"
            ].max()
        )

        latest_run = (
            saved_detail[
                saved_detail[
                    "run_number"
                ].eq(latest_id)
            ].copy()
        )

        st.session_state[
            "current_run"
        ] = latest_run


current_run = (
    st.session_state[
        "current_run"
    ]
)


# =========================================================
# EMPTY STATE
# =========================================================

if current_run is None:
    st.info(
        "Choose a scenario and press "
        "**Run & save simulation**. "
        "Your first run will appear here "
        "and will also be written to the "
        "project data folder."
    )

    st.stop()


# =========================================================
# CURRENT RUN INFO
# =========================================================

run_id = (
    current_run[
        "run_id"
    ].iloc[0]
)

run_number = int(
    current_run[
        "run_number"
    ].iloc[0]
)

run_scenario = (
    current_run[
        "scenario"
    ].iloc[0]
)

run_day_type = (
    current_run[
        "day_type"
    ].iloc[0]
)

run_seed = int(
    current_run[
        "effective_seed"
    ].iloc[0]
)

totals = run_totals(
    current_run
)

st.success(
    f"Viewing saved run "
    f"**#{run_number}** · "
    f"**{run_scenario}** · "
    f"**{run_day_type}** · "
    f"seed **{run_seed}**"
)

k1, k2, k3, k4 = st.columns(
    4
)

k1.metric(
    "24-hour demand",
    f"{totals['demand']:,}",
)

k2.metric(
    "Completed trips",
    f"{totals['completed']:,}",
)

k3.metric(
    "Unmet rides",
    f"{totals['unmet']:,}",
)

k4.metric(
    "Service rate",
    f"{totals['service_rate']:.1%}",
)


# =========================================================
# DISPLAY CONTROLS
# =========================================================

st.sidebar.divider()
st.sidebar.header(
    "View current run"
)

hour = st.sidebar.slider(
    "Hour",
    min_value=0,
    max_value=23,
    value=17,
    step=1,
)

metric_name = st.sidebar.selectbox(
    "Map metric",
    list(METRICS.keys()),
    index=0,
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📍 Hour snapshot",
        "▶ 24-hour animation",
        "📊 Run analysis",
        "🗂 Run history & data",
    ]
)


# =========================================================
# TAB 1
# =========================================================

with tab1:
    hour_kpi = hour_totals(
        current_run,
        hour,
    )

    st.subheader(
        f"{format_hour(hour)}"
    )

    h1, h2, h3, h4 = st.columns(
        4
    )

    h1.metric(
        "Demand",
        f"{hour_kpi['demand']:,}",
    )

    h2.metric(
        "Completed",
        f"{hour_kpi['completed']:,}",
    )

    h3.metric(
        "Unmet rides",
        f"{hour_kpi['unmet']:,}",
    )

    h4.metric(
        "Service rate",
        f"{hour_kpi['service_rate']:.1%}",
    )

    st.plotly_chart(
        build_snapshot_map(
            run_df=current_run,
            hour=hour,
            metric_name=metric_name,
        ),
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    hour_df = current_run[
        current_run["hour"].eq(
            hour
        )
    ]

    worst = (
        hour_df.nlargest(
            1,
            "unmet_demand_pct",
        )
        .iloc[0]
    )

    st.info(
        f"At {format_hour(hour)}, "
        f"the most stressed modeled zone "
        f"is **{worst['pickup_zone']}**, "
        f"with **{worst['unmet_demand_pct']:.1%}** "
        f"of simulated ride demand left unserved."
    )


# =========================================================
# TAB 2
# =========================================================

with tab2:
    st.subheader(
        "Watch this exact saved run "
        "change hour by hour"
    )

    st.write(
        "The gray NYC map stays fixed. "
        "Only the 20 modeled zones change color. "
        "Because this is one stochastic run—not "
        "a Monte Carlo average—the hour-to-hour "
        "variation should be visible."
    )

    st.plotly_chart(
        build_animated_map(
            run_df=current_run,
            metric_name=metric_name,
        ),
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )


# =========================================================
# TAB 3
# =========================================================

with tab3:
    left, right = st.columns(
        [1.3, 1]
    )

    with left:
        st.subheader(
            "Demand vs. completed trips"
        )

        st.plotly_chart(
            hourly_chart(
                current_run
            ),
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

    with right:
        st.subheader(
            "Most stressed zones"
        )

        st.dataframe(
            top_zone_table(
                current_run
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These values summarize the full "
            "24-hour simulated run."
        )


# =========================================================
# TAB 4: RUN HISTORY + EXPORT
# =========================================================

with tab4:
    summary_history = (
        load_saved_summary()
    )

    detail_history = (
        load_saved_detail()
    )

    st.subheader(
        "Saved simulation runs"
    )

    if summary_history.empty:
        st.info(
            "No saved runs yet."
        )

    else:
        st.metric(
            "Total saved runs",
            len(
                summary_history
            ),
        )

        st.metric(
            "Total detailed rows",
            len(
                detail_history
            ),
        )

        st.caption(
            "For reference, 25 saved runs "
            "produce 12,000 zone-hour rows."
        )

        display_summary = (
            summary_history.copy()
        )

        if (
            "service_rate"
            in display_summary.columns
        ):
            display_summary[
                "service_rate"
            ] = (
                display_summary[
                    "service_rate"
                ]
                * 100
            ).round(1)

        if (
            "unmet_demand_pct"
            in display_summary.columns
        ):
            display_summary[
                "unmet_demand_pct"
            ] = (
                display_summary[
                    "unmet_demand_pct"
                ]
                * 100
            ).round(1)

        if (
            "peak_hour_unmet_pct"
            in display_summary.columns
        ):
            display_summary[
                "peak_hour_unmet_pct"
            ] = (
                display_summary[
                    "peak_hour_unmet_pct"
                ]
                * 100
            ).round(1)

        if (
            "worst_zone_unmet_pct"
            in display_summary.columns
        ):
            display_summary[
                "worst_zone_unmet_pct"
            ] = (
                display_summary[
                    "worst_zone_unmet_pct"
                ]
                * 100
            ).round(1)

        st.dataframe(
            display_summary.sort_values(
                "run_number",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

        summary_csv = (
            summary_history
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )

        detail_csv = (
            detail_history
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )

        d1, d2 = st.columns(
            2
        )

        with d1:
            st.download_button(
                "Download run summary CSV",
                data=summary_csv,
                file_name=(
                    "simulation_run_summary.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        with d2:
            st.download_button(
                "Download detailed run data CSV",
                data=detail_csv,
                file_name=(
                    "simulation_run_detail.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown(
            "#### Files also saved automatically"
        )

        st.code(
            "data/processed/"
            "simulation_run_summary.csv\n"
            "data/processed/"
            "simulation_run_detail.csv"
        )


st.divider()

st.caption(
    "Each saved run is reproducible because "
    "its effective random seed is recorded. "
    "The simulation estimates unmet demand; "
    "TLC records completed trips, not failed ride requests."
)
