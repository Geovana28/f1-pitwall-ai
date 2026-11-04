"""
F1 2026 Season ML Inference & Grid Simulator.
Author: Geovana Luiza Morais Moreira

Applies the Gradient Boosting model trained on 2022-2025 real FIA data to project
podium probabilities and finishing order for the upcoming 2026 Season Grid
(including Lewis Hamilton at Ferrari, Kimi Antonelli at Mercedes, and new aero/engine regulations).
"""

import os
import joblib
import pandas as pd

PIPELINE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(PIPELINE_DIR, "f1_podium_predictor_model.joblib")


def simulate_2026_grand_prix(circuit_id: str = "interlagos", race_name: str = "Sao Paulo Grand Prix (Interlagos)"):
    """Simulate a 2026 Grand Prix using the trained ML model."""
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model file not found at {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)

    # 2026 Official Grid Projection
    grid_2026 = [
        {"driver_name": "Lando Norris", "driver_code": "NOR", "team_name": "McLaren", "grid_position": 1, "team_season_point_share": 0.28, "avg_speed_kmh": 228.4, "fastest_lap_rank": 1},
        {"driver_name": "Lewis Hamilton", "driver_code": "HAM", "team_name": "Ferrari", "grid_position": 2, "team_season_point_share": 0.25, "avg_speed_kmh": 228.0, "fastest_lap_rank": 2},
        {"driver_name": "Charles Leclerc", "driver_code": "LEC", "team_name": "Ferrari", "grid_position": 3, "team_season_point_share": 0.25, "avg_speed_kmh": 227.6, "fastest_lap_rank": 3},
        {"driver_name": "Max Verstappen", "driver_code": "VER", "team_name": "Red Bull", "grid_position": 4, "team_season_point_share": 0.22, "avg_speed_kmh": 227.9, "fastest_lap_rank": 4},
        {"driver_name": "Kimi Antonelli", "driver_code": "ANT", "team_name": "Mercedes", "grid_position": 5, "team_season_point_share": 0.16, "avg_speed_kmh": 226.8, "fastest_lap_rank": 5},
        {"driver_name": "George Russell", "driver_code": "RUS", "team_name": "Mercedes", "grid_position": 6, "team_season_point_share": 0.16, "avg_speed_kmh": 226.5, "fastest_lap_rank": 6},
        {"driver_name": "Oscar Piastri", "driver_code": "PIA", "team_name": "McLaren", "grid_position": 7, "team_season_point_share": 0.28, "avg_speed_kmh": 227.0, "fastest_lap_rank": 7},
        {"driver_name": "Fernando Alonso", "driver_code": "ALO", "team_name": "Aston Martin", "grid_position": 8, "team_season_point_share": 0.08, "avg_speed_kmh": 224.2, "fastest_lap_rank": 8},
        {"driver_name": "Alexander Albon", "driver_code": "ALB", "team_name": "Williams", "grid_position": 9, "team_season_point_share": 0.04, "avg_speed_kmh": 223.5, "fastest_lap_rank": 9},
        {"driver_name": "Pierre Gasly", "driver_code": "GAS", "team_name": "Alpine F1 Team", "grid_position": 10, "team_season_point_share": 0.03, "avg_speed_kmh": 222.8, "fastest_lap_rank": 10},
    ]

    df_2026 = pd.DataFrame(grid_2026)
    df_2026["circuit_id"] = circuit_id
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

    # Predict probability of Podium (Top 3) using the trained 2022-2025 ML model
    podium_probs = model.predict_proba(df_2026[features])[:, 1]
    df_2026["podium_probability"] = (podium_probs * 100).round(1)

    df_sorted = df_2026.sort_values(by="podium_probability", ascending=False).reset_index(drop=True)

    print("=" * 70)
    print(f" [*] 2026 PREDICTED PODIUM PROBABILITY: {race_name.upper()}")
    print("=" * 70)
    print(f"{'Pos':<6} | {'Driver':<18} | {'Team':<16} | {'Grid':<5} | {'Podium Chance':<12}")
    print("-" * 70)

    for idx, row in df_sorted.iterrows():
        tag = "[P1] " if idx == 0 else ("[P2] " if idx == 1 else ("[P3] " if idx == 2 else "     "))
        print(f"{tag}{idx+1:<2} | {row['driver_name']:<18} | {row['team_name']:<16} | P{row['grid_position']:<4} | {row['podium_probability']:>5.1f}%")

    print("=" * 70)


if __name__ == "__main__":
    simulate_2026_grand_prix()
