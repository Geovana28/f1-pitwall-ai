"""
F1 Real Telemetry & Race Data Ingestion Pipeline (2022 - 2025 Modern Ground Effect Era).
Author: Geovana Luiza Morais Moreira

Extracts ALL Grand Prix races, driver results, grid positions, and finish positions
from the Ergast / Jolpica F1 API across the entire 2022, 2023, 2024 and 2025 modern era.
"""

import os
import json
import time
import urllib.request
from typing import List, Dict, Any
import pandas as pd

API_BASE_URL = "https://api.jolpi.ca/ergast/f1"
OUTPUT_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(OUTPUT_DIR, "f1_real_races_2022_2025.csv")


def fetch_all_season_results(year: int) -> List[Dict[str, Any]]:
    """Paginate and fetch 100% of the race results for a given season."""
    offset = 0
    limit = 100
    all_season_races = {}

    print(f"[*] Downloading complete season {year}...")
    while True:
        url = f"{API_BASE_URL}/{year}/results.json?limit={limit}&offset={offset}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                mr_data = data.get("MRData", {})
                total = int(mr_data.get("total", 0))
                races = mr_data.get("RaceTable", {}).get("Races", [])
                
                if not races:
                    break

                for race in races:
                    round_num = race.get("round")
                    if round_num not in all_season_races:
                        all_season_races[round_num] = race
                    else:
                        all_season_races[round_num]["Results"].extend(race.get("Results", []))

                offset += limit
                if offset >= total:
                    break
                time.sleep(0.5)
        except Exception as e:
            print(f"    [!] Error at offset {offset} for {year}: {e}")
            break

    print(f"[OK] Season {year}: Successfully retrieved {len(all_season_races)} Grand Prix rounds.")
    return list(all_season_races.values())


def process_races_into_dataframe(all_seasons_races: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten nested Ergast race JSON into ML-ready tabular features."""
    rows = []

    for race in all_seasons_races:
        season = int(race.get("season", 0))
        round_no = int(race.get("round", 0))
        race_name = race.get("raceName", "")
        circuit_id = race.get("Circuit", {}).get("circuitId", "")
        country = race.get("Circuit", {}).get("Location", {}).get("country", "")
        date = race.get("date", "")

        results = race.get("Results", [])
        for res in results:
            driver = res.get("Driver", {})
            driver_id = driver.get("driverId", "")
            driver_name = f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip()
            driver_code = driver.get("code", driver_id[:3].upper())

            constructor = res.get("Constructor", {})
            team_id = constructor.get("constructorId", "")
            team_name = constructor.get("name", "")

            grid_position = int(res.get("grid", 0))
            finish_position_str = res.get("position", "20")
            try:
                finish_position = int(finish_position_str)
            except ValueError:
                finish_position = 20

            points = float(res.get("points", 0.0))
            laps_completed = int(res.get("laps", 0))
            status = res.get("status", "")
            is_dnf = 0 if ("Finished" in status or "Lap" in status) else 1

            # Fastest Lap Data
            fastest_lap_info = res.get("FastestLap", {})
            fastest_lap_rank = int(fastest_lap_info.get("rank", 20)) if fastest_lap_info.get("rank") else 20
            
            speed_trap = fastest_lap_info.get("AverageSpeed", {}).get("speed", 0.0)
            try:
                avg_speed_kmh = float(speed_trap)
            except (ValueError, TypeError):
                avg_speed_kmh = 0.0

            # Target 1: Posição no Pódio (Top 3)
            is_podium = 1 if finish_position in [1, 2, 3] else 0
            
            # Target 2: Pontuador (Top 10)
            is_points = 1 if finish_position <= 10 else 0

            # Feature: Ganhos ou Perdas de Posição
            positions_gained = (grid_position - finish_position) if grid_position > 0 else 0

            rows.append({
                "season": season,
                "round": round_no,
                "race_name": race_name,
                "circuit_id": circuit_id,
                "country": country,
                "date": date,
                "driver_id": driver_id,
                "driver_code": driver_code,
                "driver_name": driver_name,
                "team_id": team_id,
                "team_name": team_name,
                "grid_position": grid_position,
                "finish_position": finish_position,
                "is_podium": is_podium,
                "is_points": is_points,
                "points": points,
                "laps_completed": laps_completed,
                "positions_gained": positions_gained,
                "fastest_lap_rank": fastest_lap_rank,
                "avg_speed_kmh": avg_speed_kmh,
                "is_dnf": is_dnf,
                "status": status,
            })

    df = pd.DataFrame(rows)
    return df


def run_ingestion_pipeline():
    """Main execution across all ground effect seasons."""
    print("=" * 65)
    print(" F1 Modern Ground Effect Era Dataset Ingestion (2022 - 2025)")
    print("=" * 65)

    all_races = []
    for year in [2022, 2023, 2024, 2025]:
        races = fetch_all_season_results(year)
        all_races.extend(races)
        time.sleep(1)

    if not all_races:
        print("[!] No data retrieved.")
        return

    df = process_races_into_dataframe(all_races)
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8")
    print("\n" + "=" * 65)
    print(f"[SUCCESS] Ingested {len(df)} real driver race records across 4 seasons!")
    print(f"Saved to: {DATASET_PATH}")
    print("=" * 65)
    print("\nTop Constructors in Dataset:")
    print(df["team_name"].value_counts().head(8))


if __name__ == "__main__":
    run_ingestion_pipeline()
