# NYC Taxi Winter Simulation

This project builds a Monte Carlo simulation of NYC Yellow Taxi service under winter weather disruption.

## Historical winter
- December 2025
- January 2026
- February 2026

## Workflow
1. Download TLC Yellow Taxi trip files and Taxi Zone lookup.
2. Build a historical baseline for the 20 busiest pickup zones.
3. Simulate five winter scenarios across 100 Monte Carlo runs.
4. Validate the simulated dataset.

## Set up in VS Code

### 1. Open this folder in VS Code

### 2. Create a virtual environment

Mac/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install packages
```bash
pip install -r requirements.txt
```

### 4. Run the project
```bash
python src/00_download_data.py
python src/01_build_baseline.py
python src/02_run_simulation.py
python src/03_validate_simulation.py
```

## Main outputs

- `data/processed/tlc_winter_baseline.csv`
- `data/processed/winter_taxi_simulation.csv`
- `data/processed/winter_taxi_simulation.parquet`
- `data/processed/scenario_summary.csv`

## Reproducibility
The simulation uses random seed `780`.
