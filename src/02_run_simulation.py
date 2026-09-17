from pathlib import Path
import numpy as np
import pandas as pd

IN_FILE = Path("data/processed/tlc_winter_baseline.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 780
N_RUNS = 100

# All distributions use triangular(low, most_likely, high).
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


def triangular_draw(rng, params, size):
    low, mode, high = params

    # np.random.triangular cannot use low == mode == high.
    if low == mode == high:
        return np.full(size, low, dtype=float)

    return rng.triangular(low, mode, high, size=size)


def simulate_scenario(
    baseline: pd.DataFrame,
    scenario_name: str,
    params: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    # Each baseline zone/hour/day-type gets N_RUNS simulation rows.
    sim = baseline.loc[
        baseline.index.repeat(N_RUNS)
    ].reset_index(drop=True)

    n = len(sim)

    sim["simulation_run"] = np.tile(
        np.arange(1, N_RUNS + 1),
        len(baseline),
    )

    sim["scenario"] = scenario_name

    sim["snowfall_inches"] = triangular_draw(
        rng, params["snowfall"], n
    )

    sim["demand_multiplier"] = triangular_draw(
        rng, params["demand"], n
    )

    sim["supply_multiplier"] = triangular_draw(
        rng, params["supply"], n
    )

    sim["duration_multiplier"] = triangular_draw(
        rng, params["duration"], n
    )

    # Passenger demand.
    sim["expected_demand"] = (
        sim["baseline_trips"]
        * sim["demand_multiplier"]
    ).clip(lower=0)

    sim["simulated_demand"] = rng.poisson(
        lam=sim["expected_demand"].to_numpy()
    )

    # Trip duration.
    sim["simulated_duration"] = (
        sim["baseline_duration"]
        * sim["duration_multiplier"]
    )

    # Service capacity proxy.
    sim["simulated_capacity"] = np.floor(
        sim["baseline_capacity_proxy"]
        * sim["supply_multiplier"]
        / sim["duration_multiplier"]
    ).clip(lower=0).astype(int)

    # Completed trips cannot exceed either passenger demand or service capacity.
    sim["completed_trips"] = np.minimum(
        sim["simulated_demand"],
        sim["simulated_capacity"],
    )

    sim["unmet_demand"] = (
        sim["simulated_demand"]
        - sim["completed_trips"]
    )

    sim["unmet_demand_pct"] = np.where(
        sim["simulated_demand"] > 0,
        sim["unmet_demand"] / sim["simulated_demand"],
        0.0,
    )

    sim["service_rate"] = np.where(
        sim["simulated_demand"] > 0,
        sim["completed_trips"] / sim["simulated_demand"],
        1.0,
    )

    sim["capacity_utilization"] = np.where(
        sim["simulated_capacity"] > 0,
        sim["completed_trips"] / sim["simulated_capacity"],
        0.0,
    )

    return sim


def main():
    baseline = pd.read_csv(IN_FILE)

    required = [
        "baseline_trips",
        "baseline_capacity_proxy",
        "baseline_duration",
    ]

    if baseline[required].isna().any().any():
        raise ValueError(
            "Baseline contains missing values in required simulation columns."
        )

    rng = np.random.default_rng(RANDOM_SEED)

    frames = []

    for scenario_name, params in SCENARIOS.items():
        print(f"Simulating: {scenario_name}")
        scenario_df = simulate_scenario(
            baseline,
            scenario_name,
            params,
            rng,
        )
        frames.append(scenario_df)

    simulation = pd.concat(frames, ignore_index=True)

    simulation.insert(
        0,
        "simulation_id",
        np.arange(1, len(simulation) + 1),
    )

    # Keep the most useful columns in a clear order.
    columns = [
        "simulation_id",
        "simulation_run",
        "scenario",
        "snowfall_inches",
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
        "demand_multiplier",
        "supply_multiplier",
        "duration_multiplier",
        "expected_demand",
        "simulated_demand",
        "simulated_duration",
        "simulated_capacity",
        "completed_trips",
        "unmet_demand",
        "unmet_demand_pct",
        "service_rate",
        "capacity_utilization",
    ]

    simulation = simulation[columns]

    csv_path = OUT_DIR / "winter_taxi_simulation.csv"
    parquet_path = OUT_DIR / "winter_taxi_simulation.parquet"

    simulation.to_csv(csv_path, index=False)
    simulation.to_parquet(parquet_path, index=False)

    print("\nSimulation complete.")
    print(f"Rows: {len(simulation):,}")
    print(f"Columns: {len(simulation.columns)}")
    print(f"CSV: {csv_path}")
    print(f"Parquet: {parquet_path}")


if __name__ == "__main__":
    main()
