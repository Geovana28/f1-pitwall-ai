"""
F1 2026 Season ML Inference & Grid Simulator (Interlagos GP - Kimi Antonelli Champion Era).
Author: Geovana Luiza Morais Moreira

Simulates the Sao Paulo GP at Interlagos in late 2026, where Kimi Antonelli has already secured
the World Drivers' Championship for Mercedes AMG Petronas, leading into round 21.
"""

import os
import joblib
import pandas as pd

PIPELINE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(PIPELINE_DIR, "f1_podium_predictor_model.joblib")


def simulate_interlagos_2026():
    """Simulate Interlagos 2026 (Round 21) with Champion Kimi Antonelli."""
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model file not found at {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)

    # 2026 Interlagos Starting Grid (Kimi Antonelli WDC Champion in Mercedes W16)
    grid_interlagos_2026 = [
        {"driver_name": "Kimi Antonelli", "driver_code": "ANT", "team_name": "Mercedes", "grid_position": 1, "team_season_point_share": 0.36, "avg_speed_kmh": 231.2, "fastest_lap_rank": 1},
        {"driver_name": "George Russell", "driver_code": "RUS", "team_name": "Mercedes", "grid_position": 2, "team_season_point_share": 0.36, "avg_speed_kmh": 230.1, "fastest_lap_rank": 2},
        {"driver_name": "Lewis Hamilton", "driver_code": "HAM", "team_name": "Ferrari", "grid_position": 3, "team_season_point_share": 0.22, "avg_speed_kmh": 229.4, "fastest_lap_rank": 3},
        {"driver_name": "Charles Leclerc", "driver_code": "LEC", "team_name": "Ferrari", "grid_position": 4, "team_season_point_share": 0.22, "avg_speed_kmh": 228.8, "fastest_lap_rank": 4},
        {"driver_name": "Lando Norris", "driver_code": "NOR", "team_name": "McLaren", "grid_position": 5, "team_season_point_share": 0.20, "avg_speed_kmh": 228.5, "fastest_lap_rank": 5},
        {"driver_name": "Max Verstappen", "driver_code": "VER", "team_name": "Red Bull", "grid_position": 6, "team_season_point_share": 0.15, "avg_speed_kmh": 228.0, "fastest_lap_rank": 6},
        {"driver_name": "Oscar Piastri", "driver_code": "PIA", "team_name": "McLaren", "grid_position": 7, "team_season_point_share": 0.20, "avg_speed_kmh": 227.6, "fastest_lap_rank": 7},
        {"driver_name": "Fernando Alonso", "driver_code": "ALO", "team_name": "Aston Martin", "grid_position": 8, "team_season_point_share": 0.05, "avg_speed_kmh": 224.2, "fastest_lap_rank": 8},
        {"driver_name": "Alexander Albon", "driver_code": "ALB", "team_name": "Williams", "grid_position": 9, "team_season_point_share": 0.03, "avg_speed_kmh": 223.5, "fastest_lap_rank": 9},
        {"driver_name": "Pierre Gasly", "driver_code": "GAS", "team_name": "Alpine F1 Team", "grid_position": 10, "team_season_point_share": 0.02, "avg_speed_kmh": 222.8, "fastest_lap_rank": 10},
    ]

    df_2026 = pd.DataFrame(grid_interlagos_2026)
    df_2026["circuit_id"] = "interlagos"
    df_2026["is_front_row"] = df_2026["grid_position"].apply(lambda g: 1 if 1 <= g <= 4 else 0)
    df_2026["is_pole_position"] = df_2026["grid_position"].apply(lambda g: 1 if g == 1 else 0)
    df_2026["qualifying_deficit"] = df_2026["grid_position"] - 1.0

    features = [
        "grid_position",
        "is_front_row",
        "is_pole_position",
        "team_season_point_share",
        "qualifying_deficit",
        "fastest_lap_rank",
        "avg_speed_kmh",
        "team_name",
        "circuit_id",
    ]

    podium_probs = model.predict_proba(df_2026[features])[:, 1]
    df_2026["podium_probability"] = (podium_probs * 100).round(1)

    df_sorted = df_2026.sort_values(by="podium_probability", ascending=False).reset_index(drop=True)

    print("=" * 75)
    print(" [*] FORMULA 1 - SAO PAULO GP 2026 (INTERLAGOS - R21)")
    print(" [*] STATUS: KIMI ANTONELLI (#12 MERCEDES-AMG) WORLD CHAMPION")
    print("=" * 75)
    print(f"{'Pos':<6} | {'Driver':<18} | {'Team':<16} | {'Grid':<5} | {'Podium Chance':<12}")
    print("-" * 75)

    for idx, row in df_sorted.iterrows():
        tag = "[P1] " if idx == 0 else ("[P2] " if idx == 1 else ("[P3] " if idx == 2 else "     "))
        champ_badge = " (WDC 2026)" if row["driver_code"] == "ANT" else ""
        print(f"{tag}{idx+1:<2} | {row['driver_name'] + champ_badge:<18} | {row['team_name']:<16} | P{row['grid_position']:<4} | {row['podium_probability']:>5.1f}%")

    print("=" * 75)


if __name__ == "__main__":
    simulate_interlagos_2026()
