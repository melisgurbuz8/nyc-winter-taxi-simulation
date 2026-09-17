from pathlib import Path
import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "yellow_tripdata_2025-12.parquet":
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-12.parquet",
    "yellow_tripdata_2026-01.parquet":
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet",
    "yellow_tripdata_2026-02.parquet":
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-02.parquet",
    "taxi_zone_lookup.csv":
        "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
}


def download_file(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"Already exists: {destination}")
        return

    print(f"Downloading {destination.name}...")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()

        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"Saved {destination.name} ({size_mb:.1f} MB)")


def main():
    for filename, url in FILES.items():
        download_file(url, RAW_DIR / filename)

    print("\nAll TLC files are ready.")


if __name__ == "__main__":
    main()
