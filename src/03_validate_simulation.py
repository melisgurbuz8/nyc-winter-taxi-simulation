from pathlib import Path
import pandas as pd

SIM_FILE = Path("data/processed/winter_taxi_simulation.parquet")
OUT_DIR = Path("data/processed")


def main():
    df = pd.read_parquet(SIM_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Logical validation tests.
    checks = {
        "No negative demand":
            (df["simulated_demand"] >= 0).all(),

        "No negative capacity":
            (df["simulated_capacity"] >= 0).all(),

        "Completed <= demand":
            (df["completed_trips"] <= df["simulated_demand"]).all(),

        "Completed <= capacity":
            (df["completed_trips"] <= df["simulated_capacity"]).all(),

        "Unmet demand identity":
            (
                df["unmet_demand"]
                == df["simulated_demand"] - df["completed_trips"]
            ).all(),

        "Service rate between 0 and 1":
            df["service_rate"].between(0, 1).all(),

        "Capacity utilization between 0 and 1":
            df["capacity_utilization"].between(0, 1).all(),
    }

    print("\nVALIDATION CHECKS")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise ValueError("At least one simulation validation check failed.")

    # Main scenario-level summary.
    summary = (
        df.groupby("scenario", as_index=False)
        .agg(
            avg_snowfall_inches=("snowfall_inches", "mean"),
            avg_simulated_demand=("simulated_demand", "mean"),
            avg_simulated_capacity=("simulated_capacity", "mean"),
            avg_completed_trips=("completed_trips", "mean"),
            avg_unmet_demand=("unmet_demand", "mean"),
            avg_unmet_demand_pct=("unmet_demand_pct", "mean"),
            avg_service_rate=("service_rate", "mean"),
            avg_capacity_utilization=("capacity_utilization", "mean"),
            avg_simulated_duration=("simulated_duration", "mean"),
        )
    )

    scenario_order = [
        "Normal Winter",
        "Light Snow",
        "Moderate Snow",
        "Heavy Snow",
        "Severe Snowstorm",
    ]

    summary["scenario"] = pd.Categorical(
        summary["scenario"],
        categories=scenario_order,
        ordered=True,
    )

    summary = summary.sort_values("scenario").reset_index(drop=True)

    summary.to_csv(
        OUT_DIR / "scenario_summary.csv",
        index=False,
    )

    print("\nSCENARIO SUMMARY")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
