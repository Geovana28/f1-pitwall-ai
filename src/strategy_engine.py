"""
Motor Físico, Degradação de Pneus e Otimizador de Estratégia de Pit Stop — GP do México (Autódromo Hermanos Rodríguez)
Dedicado à análise de telemetria, física de altitude (2.240m) e estratégia de corrida (Kimi Antonelli #12).

Modelos implementados:
1. Física de Altitude: Densidade barométrica do ar rho(h) e redução percentual de arrasto/downforce em 2.240m.
2. Curva de Tempo de Volta (Lap Time Model):
   T(lap, compound, stint_age) = T_base + gamma_fuel * (L_total - lap) + alpha_c * stint_age + beta_c * max(0, stint_age - cliff_c)^2
3. Otimizador de Estratégia de Pit Stop (1-Stop vs 2-Stop) e cálculo de Undercut Delta.
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple


MEXICO_GP_CONSTANTS = {
    "circuit": "Autódromo Hermanos Rodríguez (Mexico City GP)",
    "total_laps": 71,
    "track_length_km": 4.304,
    "altitude_m": 2240.0,
    "pit_loss_sec": 21.8,
    "base_lap_sec": 77.60,  # ~1:17.600 qualy/race base pace
    "fuel_burn_effect_sec_per_lap": 0.055,  # ~1.5kg/lap * 0.036s/kg
}

TIRE_COMPOUNDS: Dict[str, Dict[str, float]] = {
    "SOFT": {
        "delta_base_sec": 0.00,
        "linear_deg_sec": 0.095,
        "cliff_lap": 18.0,
        "quad_deg_sec": 0.018,
    },
    "MEDIUM": {
        "delta_base_sec": 0.55,
        "linear_deg_sec": 0.052,
        "cliff_lap": 30.0,
        "quad_deg_sec": 0.011,
    },
    "HARD": {
        "delta_base_sec": 1.05,
        "linear_deg_sec": 0.028,
        "cliff_lap": 44.0,
        "quad_deg_sec": 0.007,
    },
}


def compute_altitude_aerodynamics(altitude_m: float = 2240.0, temp_c: float = 24.0) -> Dict[str, float]:
    """
    Calcula a densidade atmosférica barométrica (ISA) na altitude do circuito e a redução
    relativa de pressão aerodinâmica (arrasto e downforce) comparada ao nível do mar.
    """
    rho_0 = 1.225  # kg/m^3 at sea level (15°C)
    T_0 = 288.15   # K
    L = 0.0065     # K/m lapse rate
    g = 9.80665
    R = 287.058

    # Barometric formula for pressure ratio
    p_ratio = (1.0 - (L * max(0.0, altitude_m)) / T_0) ** (g / (R * L))
    T_kelvin = temp_c + 273.15
    rho = rho_0 * p_ratio * (T_0 / T_kelvin)
    aero_ratio = rho / rho_0

    return {
        "altitude_m": round(float(altitude_m), 1),
        "air_density_kg_m3": round(float(rho), 4),
        "sea_level_density_kg_m3": rho_0,
        "aero_force_retention_pct": round(float(aero_ratio * 100.0), 2),
        "drag_and_downforce_reduction_pct": round(float((1.0 - aero_ratio) * 100.0), 2),
    }


def predict_lap_time(
    lap_number: int,
    compound: str,
    stint_age: int,
    total_laps: int = 71,
    track_temp_c: float = 42.0,
) -> float:
    """
    Calcula o tempo de volta previsto (em segundos) combinando efeito de queima de combustível
    (carro fica mais leve a cada volta) e degradação não-linear do composto de pneu (com cliff térmico).
    """
    comp_key = compound.upper()
    if comp_key not in TIRE_COMPOUNDS:
        raise ValueError(f"Unknown tire compound: {compound}")

    c = TIRE_COMPOUNDS[comp_key]
    base = MEXICO_GP_CONSTANTS["base_lap_sec"]
    fuel_penalty = (total_laps - lap_number) * MEXICO_GP_CONSTANTS["fuel_burn_effect_sec_per_lap"]

    # Track temperature multiplier on degradation (reference 40°C)
    thermal_factor = 1.0 + max(-0.15, (track_temp_c - 40.0) * 0.012)
    linear_wear = c["linear_deg_sec"] * stint_age * thermal_factor
    over_cliff = max(0.0, float(stint_age) - c["cliff_lap"])
    cliff_wear = c["quad_deg_sec"] * (over_cliff ** 2) * thermal_factor

    lap_sec = base + c["delta_base_sec"] + fuel_penalty + linear_wear + cliff_wear
    return round(float(lap_sec), 3)


def simulate_race_strategy(
    stints: List[Tuple[str, int]],
    total_laps: int = 71,
    track_temp_c: float = 42.0,
    pit_loss_sec: float = 21.8,
) -> Dict[str, Any]:
    """
    Simula uma corrida completa dada uma lista de stints [(composto, num_voltas), ...].
    Exemplo: [("MEDIUM", 28), ("HARD", 43)] para 1-Stop nas 71 voltas do GP do México.
    """
    sum_laps = sum(n for _, n in stints)
    if sum_laps != total_laps:
        raise ValueError(f"Sum of stint laps ({sum_laps}) must equal total_laps ({total_laps}).")

    lap_times: List[float] = []
    compounds_per_lap: List[str] = []
    pit_laps: List[int] = []

    current_lap = 1
    for idx, (comp, stint_len) in enumerate(stints):
        for age in range(1, stint_len + 1):
            t = predict_lap_time(current_lap, comp, age, total_laps=total_laps, track_temp_c=track_temp_c)
            if idx > 0 and age == 1:
                t += pit_loss_sec
                pit_laps.append(current_lap - 1)
            lap_times.append(round(t, 3))
            compounds_per_lap.append(comp.upper())
            current_lap += 1

    total_time_sec = round(float(sum(lap_times)), 3)
    return {
        "strategy_label": " → ".join(f"{c}({n}L)" for c, n in stints),
        "num_stops": len(stints) - 1,
        "pit_laps": pit_laps,
        "total_race_time_sec": total_time_sec,
        "avg_lap_sec": round(total_time_sec / total_laps, 3),
        "lap_times": lap_times,
        "compounds_per_lap": compounds_per_lap,
    }


def find_optimal_one_stop_window(
    start_compound: str = "MEDIUM",
    end_compound: str = "HARD",
    total_laps: int = 71,
    track_temp_c: float = 42.0,
) -> Dict[str, Any]:
    """
    Encontra a volta exata de Pit Stop (entre a volta 15 e 55) que minimiza o tempo total de prova
    e calcula o ganho de Undercut de parar 1 volta antes do pneu entrar no cliff.
    """
    best_lap = 28
    best_time = float("inf")
    curve = []

    for pit_lap in range(15, 56):
        sim = simulate_race_strategy(
            [(start_compound, pit_lap), (end_compound, total_laps - pit_lap)],
            total_laps=total_laps,
            track_temp_c=track_temp_c,
        )
        t_total = sim["total_race_time_sec"]
        curve.append({"pit_lap": pit_lap, "total_time_sec": t_total})
        if t_total < best_time:
            best_time = t_total
            best_lap = pit_lap

    # Undercut delta: difference between fresh end_compound (age=1) vs worn start_compound at best_lap + 1
    worn_lap = predict_lap_time(best_lap + 1, start_compound, best_lap + 1, total_laps, track_temp_c)
    fresh_lap = predict_lap_time(best_lap + 1, end_compound, 1, total_laps, track_temp_c)
    undercut_gain_sec = round(worn_lap - fresh_lap, 3)

    return {
        "optimal_pit_lap": best_lap,
        "optimal_total_time_sec": best_time,
        "undercut_single_lap_gain_sec": undercut_gain_sec,
        "pit_window_optimal_range": [max(15, best_lap - 2), min(55, best_lap + 2)],
        "tradeoff_curve": curve,
    }
