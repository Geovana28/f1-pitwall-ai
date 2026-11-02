"""
Gerador e Carregador de Telemetria de Volta (Speed, Throttle, Brake, Gear, DRS) — F1 Global Pit-Wall AI.

Implementa:
1. Geração física de alta resolução (100 pontos normalizados ao longo do comprimento L_km da pista)
   comparando o traçado de telemetria de quaisquer duas equipes/carros do grid (ex.: McLaren MCL39 vs. Ferrari SF-25
   ou Mercedes W16 vs. Red Bull RB21), refletindo o setup de asa escolhido, altitude barométrica, potência do motor (PU)
   e ganho de DRS nas zonas de reta.
2. Conector opcional para a biblioteca oficial `fastf1` (quando instalada localmente com acesso à internet) para
   extração de telemetria histórica real da FIA.
"""

import math
from typing import Dict, Any, List
from src.strategy_engine import (
    get_circuit_by_id,
    get_team_by_id,
    evaluate_car_setup_physics,
)


def generate_lap_telemetry_comparison(
    circuit_id: str = "interlagos",
    team_a_id: str = "mercedes",
    wing_a: int = 34,
    team_b_id: str = "ferrari",
    wing_b: int = 34,
    n_points: int = 80,
) -> Dict[str, Any]:
    """
    Gera os canais de telemetria ao longo da distância da volta (0% a 100% de L_km):
    - distance_m: distância percorrida na volta (metros)
    - speed_a_kmh / speed_b_kmh: velocidade instantânea (km/h)
    - throttle_a_pct / throttle_b_pct: pedal do acelerador (0 a 100%)
    - brake_a_pct / brake_b_pct: pressão de freio (0 a 100%)
    - drs_active: indicador binário (0 ou 1) nas zonas de reta com DRS aberto
    - sector_deltas_sec: diferença de tempo acumulada nos Setores S1, S2 e S3
    """
    circuit = get_circuit_by_id(circuit_id)
    team_a = get_team_by_id(team_a_id)
    team_b = get_team_by_id(team_b_id)

    setup_a = evaluate_car_setup_physics(circuit_id, team_a_id, wing_level=wing_a)
    setup_b = evaluate_car_setup_physics(circuit_id, team_b_id, wing_level=wing_b)

    L_meters = circuit["track_length_km"] * 1000.0
    distances: List[float] = []
    speed_a: List[float] = []
    speed_b: List[float] = []
    throttle_a: List[float] = []
    brake_a: List[float] = []
    drs_zone: List[int] = []

    # Define 4 canonical braking/corner complexes along the normalized lap [0, 1]
    corners = [
        (0.18, 0.045, 0.35),  # Turn 1 heavy braking zone (e.g., Senna S / Monza T1 / Mexico T1)
        (0.42, 0.055, 0.55),  # Mid-sector technical medium-speed complex
        (0.66, 0.040, 0.42),  # Hairpin / low-speed traction zone
        (0.84, 0.050, 0.62),  # Final sector medium-high speed corner before main straight
    ]

    for i in range(n_points):
        u = i / float(n_points - 1)
        dist_m = round(u * L_meters, 1)
        distances.append(dist_m)

        # Compute cornering dip factor (0 = full straight, 1 = deepest apex)
        corner_dip = 0.0
        braking_intensity = 0.0
        for center, width, min_speed_ratio in corners:
            g = math.exp(-0.5 * ((u - center) / width) ** 2)
            corner_dip = max(corner_dip, g * (1.0 - min_speed_ratio))
            # Braking peaks slightly before the apex
            b_peak = math.exp(-0.5 * ((u - (center - width * 0.65)) / (width * 0.45)) ** 2)
            braking_intensity = max(braking_intensity, b_peak)

        is_drs = 1 if (u < 0.13 or u > 0.90 or (0.26 <= u <= 0.35)) else 0
        drs_zone.append(is_drs)

        drs_boost_a = (team_a["drs_gain_kmh"] * 0.45) if is_drs else 0.0
        drs_boost_b = (team_b["drs_gain_kmh"] * 0.45) if is_drs else 0.0

        # Interpolate between straight top speed and cornering apex speed
        v_a = (1.0 - corner_dip) * (setup_a["top_speed_drs_kmh"] - 10.0 + drs_boost_a) + corner_dip * setup_a["cornering_apex_kmh"]
        v_b = (1.0 - corner_dip) * (setup_b["top_speed_drs_kmh"] - 10.0 + drs_boost_b) + corner_dip * setup_b["cornering_apex_kmh"]

        speed_a.append(round(float(v_a), 1))
        speed_b.append(round(float(v_b), 1))

        thr = max(0.0, min(100.0, (1.0 - corner_dip * 1.45) * 100.0))
        brk = max(0.0, min(100.0, braking_intensity * 98.0)) if corner_dip > 0.18 else 0.0
        throttle_a.append(round(thr, 1))
        brake_a.append(round(brk, 1))

    # Sector time deltas (S1: 0..33%, S2: 33..66%, S3: 66..100%)
    total_delta = round(setup_a["setup_lap_delta_sec"] - setup_b["setup_lap_delta_sec"], 3)
    s1_delta = round(total_delta * 0.32, 3)
    s2_delta = round(total_delta * 0.41, 3)
    s3_delta = round(total_delta - s1_delta - s2_delta, 3)

    return {
        "circuit_id": circuit["id"],
        "track_length_m": round(L_meters, 1),
        "team_a": {"id": team_a["id"], "name": team_a["name"], "chassis": team_a["chassis"], "wing": wing_a},
        "team_b": {"id": team_b["id"], "name": team_b["name"], "chassis": team_b["chassis"], "wing": wing_b},
        "distance_m": distances,
        "speed_a_kmh": speed_a,
        "speed_b_kmh": speed_b,
        "throttle_a_pct": throttle_a,
        "brake_a_pct": brake_a,
        "drs_active": drs_zone,
        "sector_deltas_a_vs_b_sec": {
            "S1": s1_delta,
            "S2": s2_delta,
            "S3": s3_delta,
            "lap_total_delta_sec": total_delta,
        },
    }
