from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRIP_FILES = [
    RAW_DIR / "yellow_tripdata_2025-12.parquet",
    RAW_DIR / "yellow_tripdata_2026-01.parquet",
    RAW_DIR / "yellow_tripdata_2026-02.parquet",
]

ZONE_FILE = RAW_DIR / "taxi_zone_lookup.csv"

KEEP_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "trip_distance",
    "total_amount",
]

WINTER_START = pd.Timestamp("2025-12-01")
WINTER_END = pd.Timestamp("2026-03-01")
TOP_N_ZONES = 20


def load_trips() -> pd.DataFrame:
    frames = []

    for path in TRIP_FILES:
        print(f"Reading {path.name}...")
        df = pd.read_parquet(path, columns=KEEP_COLUMNS)
        frames.append(df)

    trips = pd.concat(frames, ignore_index=True)
    print(f"Raw combined rows: {len(trips):,}")
    return trips


def clean_trips(trips: pd.DataFrame) -> pd.DataFrame:
    trips = trips.copy()

    trips["tpep_pickup_datetime"] = pd.to_datetime(
        trips["tpep_pickup_datetime"], errors="coerce"
    )
    trips["tpep_dropoff_datetime"] = pd.to_datetime(
        trips["tpep_dropoff_datetime"], errors="coerce"
    )

    trips = trips.dropna(
        subset=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "PULocationID",
        ]
    )

    # Keep only the intended winter season.
    trips = trips[
        (trips["tpep_pickup_datetime"] >= WINTER_START)
        & (trips["tpep_pickup_datetime"] < WINTER_END)
    ].copy()

    trips["trip_duration_min"] = (
        trips["tpep_dropoff_datetime"] - trips["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Transparent cleaning rules to remove impossible/extreme records.
    trips = trips[
        trips["trip_duration_min"].between(1, 180)
        & trips["trip_distance"].between(0.1, 100)
        & trips["total_amount"].between(1, 500)
        & trips["PULocationID"].between(1, 265)
    ].copy()

    trips["PULocationID"] = trips["PULocationID"].astype(int)
    trips["date"] = trips["tpep_pickup_datetime"].dt.normalize()
    trips["hour"] = trips["tpep_pickup_datetime"].dt.hour
    trips["day_type"] = np.where(
        trips["tpep_pickup_datetime"].dt.dayofweek < 5,
        "Weekday",
        "Weekend",
    )

    print(f"Rows after cleaning: {len(trips):,}")
    return trips


def identify_top_zones(trips: pd.DataFrame) -> list[int]:
    top_zones = (
        trips["PULocationID"]
        .value_counts()
        .head(TOP_N_ZONES)
        .index
        .tolist()
    )
    print(f"Top {TOP_N_ZONES} pickup zones: {top_zones}")
    return top_zones


def build_hourly_grid(trips: pd.DataFrame, top_zones: list[int]) -> pd.DataFrame:
    trips_top = trips[trips["PULocationID"].isin(top_zones)].copy()

    hourly_counts = (
        trips_top.groupby(["date", "hour", "day_type", "PULocationID"])
        .size()
        .rename("trip_count")
        .reset_index()
    )

    # Build a complete date x hour x zone grid so zero-trip hours are represented.
    dates = pd.date_range(
        WINTER_START,
        WINTER_END - pd.Timedelta(days=1),
        freq="D",
    )

    full_grid = pd.MultiIndex.from_product(
        [dates, range(24), top_zones],
        names=["date", "hour", "PULocationID"],
    ).to_frame(index=False)

    full_grid["day_type"] = np.where(
        full_grid["date"].dt.dayofweek < 5,
        "Weekday",
        "Weekend",
    )

    full_grid = full_grid.merge(
        hourly_counts,
        on=["date", "hour", "day_type", "PULocationID"],
        how="left",
    )

    full_grid["trip_count"] = full_grid["trip_count"].fillna(0).astype(int)
    return full_grid


def build_baseline(
    trips: pd.DataFrame,
    hourly_grid: pd.DataFrame,
    top_zones: list[int],
) -> pd.DataFrame:
    trips_top = trips[trips["PULocationID"].isin(top_zones)].copy()

    # Demand and capacity-proxy metrics come from observed hourly counts.
    count_baseline = (
        hourly_grid.groupby(["PULocationID", "hour", "day_type"])
        .agg(
            baseline_trips=("trip_count", "median"),
            baseline_capacity_proxy=(
                "trip_count",
                lambda x: np.quantile(x, 0.95),
            ),
        )
        .reset_index()
    )

    # Trip characteristics come directly from historical completed trips.
    trip_baseline = (
        trips_top.groupby(["PULocationID", "hour", "day_type"])
        .agg(
            baseline_duration=("trip_duration_min", "median"),
            baseline_distance=("trip_distance", "median"),
            baseline_total_amount=("total_amount", "median"),
        )
        .reset_index()
    )

    baseline = count_baseline.merge(
        trip_baseline,
        on=["PULocationID", "hour", "day_type"],
        how="left",
    )

    baseline["baseline_capacity_proxy"] = np.maximum(
        baseline["baseline_capacity_proxy"],
        baseline["baseline_trips"],
    )

    zones = pd.read_csv(ZONE_FILE)
    zones = zones.rename(columns={"LocationID": "PULocationID"})

    baseline = baseline.merge(
        zones[["PULocationID", "Borough", "Zone"]],
        on="PULocationID",
        how="left",
    )

    baseline = baseline.rename(
        columns={
            "Zone": "pickup_zone",
            "Borough": "borough",
        }
    )

    baseline = baseline[
        [
            "PULocationID",
            "pickup_zone",
            "borough",
            "hour",
            "day_type",
            "baseline_trips",
            "baseline_capacity_proxy",
            "baseline_duration",
            "baseline_distance",
            "baseline_total_amount",
        ]
    ].sort_values(
        ["PULocationID", "day_type", "hour"]
    ).reset_index(drop=True)

    return baseline


def main():
    trips = load_trips()
    trips = clean_trips(trips)

    top_zones = identify_top_zones(trips)
    hourly_grid = build_hourly_grid(trips, top_zones)
    baseline = build_baseline(trips, hourly_grid, top_zones)

    baseline.to_csv(OUT_DIR / "tlc_winter_baseline.csv", index=False)

    print("\nBaseline created.")
    print(f"Rows: {len(baseline):,}")
    print(f"Expected rows: {TOP_N_ZONES * 24 * 2:,}")
    print("\nSample:")
    print(baseline.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
