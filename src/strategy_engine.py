"""
Motor Físico, Catálogo Global de 24 Circuitos, Setup das 10 Equipes e Otimizador de Estratégia de Pit Stop.

Modelos implementados:
1. Catálogo Global dos 24 Circuitos (data/circuits_catalog.json) + Detecção Automática do Próximo GP.
2. Catálogo das 10 Equipes e 20 Carros (data/teams_catalog.json) + Modelo Físico de Setup (Asa, Altura do Assoalho, ERS).
3. Física de Altitude: Densidade barométrica do ar rho(h, T) e redução percentual de arrasto/downforce.
4. Curva de Tempo de Volta (Lap Time Model):
   T(lap, compound, stint_age) = T_base_circuit + delta_team_setup + gamma_fuel * (L_total - lap)
                                 + alpha_c * stint_age + beta_c * max(0, stint_age - cliff_c)^2
5. Otimizador de Estratégia de Pit Stop (1-Stop vs 2-Stop) e cálculo de Undercut Delta.
"""

import json
import math
from datetime import date
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MEXICO_GP_CONSTANTS = {
    "circuit": "Autódromo Hermanos Rodríguez (Mexico City GP)",
    "total_laps": 71,
    "track_length_km": 4.304,
    "altitude_m": 2240.0,
    "pit_loss_sec": 21.8,
    "base_lap_sec": 77.60,
    "fuel_burn_effect_sec_per_lap": 0.055,
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


def load_circuits_catalog() -> List[Dict[str, Any]]:
    """Carrega o catálogo oficial dos 24 circuitos do calendário mundial da F1."""
    catalog_path = DATA_DIR / "circuits_catalog.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["circuits"]


def load_teams_catalog() -> List[Dict[str, Any]]:
    """Carrega o catálogo das 10 equipes do grid da F1 com parâmetros aerodinâmicos e mecânicos."""
    teams_path = DATA_DIR / "teams_catalog.json"
    with open(teams_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["teams"]


def get_circuit_by_id(circuit_id: str) -> Dict[str, Any]:
    """Busca um circuito pelo seu ID (ex.: 'interlagos', 'mexico', 'monaco', 'spa')."""
    for c in load_circuits_catalog():
        if c["id"] == circuit_id.lower():
            return c
    raise ValueError(f"Unknown circuit_id: {circuit_id}")


def get_team_by_id(team_id: str) -> Dict[str, Any]:
    """Busca uma equipe pelo seu ID (ex.: 'mercedes', 'mclaren', 'ferrari', 'red_bull')."""
    for t in load_teams_catalog():
        if t["id"] == team_id.lower():
            return t
    raise ValueError(f"Unknown team_id: {team_id}")


def get_next_upcoming_gp(reference_date_iso: Optional[str] = None) -> Dict[str, Any]:
    """
    Detecta automaticamente qual é o próximo GP do calendário a partir da data atual (ou reference_date_iso).
    Se todas as corridas do ano já passaram, retorna a etapa de abertura da temporada seguinte.
    """
    ref = date.fromisoformat(reference_date_iso) if reference_date_iso else date.today()
    circuits = sorted(load_circuits_catalog(), key=lambda x: x["race_date"])
    for c in circuits:
        race_dt = date.fromisoformat(c["race_date"])
        if race_dt >= ref:
            days_until = (race_dt - ref).days
            return {**c, "days_until_race": days_until}
    first = circuits[0]
    return {**first, "days_until_race": 0}


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


def evaluate_car_setup_physics(
    circuit_id: str,
    team_id: str,
    wing_level: int = 34,
    ride_height_mm: float = 32.0,
    ers_mode: str = "BALANCED",
) -> Dict[str, Any]:
    """
    Avalia o efeito físico do Setup do Carro (Ângulo de Asa 1..50, Altura do Assoalho em mm, Modo ERS)
    para qualquer equipe do grid em qualquer um dos 24 circuitos:
    - Velocidade máxima de reta (Top Speed km/h) ajustada por altitude, potência do motor, DRS e ângulo de asa.
    - Velocidade média de contorno de curva (Cornering Apex Speed km/h).
    - Penalidade ou bônus de tempo de volta (setup_lap_delta_sec) vs. asa ótima da pista.
    - Multiplicador de desgaste térmico do pneu (setup_tire_deg_multiplier): pouca asa gera sliding e superaquece o pneu.
    """
    circuit = get_circuit_by_id(circuit_id)
    team = get_team_by_id(team_id)
    aero = compute_altitude_aerodynamics(circuit["altitude_m"], temp_c=24.0)
    air_density_ratio = aero["aero_force_retention_pct"] / 100.0

    opt_wing = circuit["optimal_wing_level"]
    wing_diff = float(wing_level - opt_wing)

    # Deviation from optimal wing for this circuit causes quadratic drag/cornering loss
    wing_mismatch_penalty_sec = 0.0022 * (wing_diff ** 2)

    # Ride height effect: optimal ground-effect window is ~30mm; too low risks plank wear/bottoming, too high loses suction
    ride_diff = float(ride_height_mm - 30.0)
    ride_penalty_sec = 0.0035 * (ride_diff ** 2)

    # ERS deployment mode effect on lap time & battery/thermal stress
    ers_map = {
        "QUALY_PUSH": {"lap_delta": -0.35, "deg_mult": 1.06},
        "BALANCED": {"lap_delta": 0.00, "deg_mult": 1.00},
        "HARVEST_SAVE": {"lap_delta": 0.28, "deg_mult": 0.95},
    }
    ers_cfg = ers_map.get(ers_mode.upper(), ers_map["BALANCED"])

    # Team chassis & power unit adaptation to circuit profile
    aero_bonus_sec = (0.95 - team["aero_efficiency"]) * 2.8 * circuit["downforce_demand"]
    pu_bonus_sec = (0.96 - team["power_unit_rating"]) * 2.5 * circuit["power_sensitivity"]
    total_lap_delta_sec = (
        team["base_pace_offset_sec"]
        + aero_bonus_sec
        + pu_bonus_sec
        + wing_mismatch_penalty_sec
        + ride_penalty_sec
        + ers_cfg["lap_delta"]
    )

    # Top speed (km/h): higher power unit + lower air density + lower wing -> higher top speed
    effective_drag = (wing_level * 0.62) * air_density_ratio / team["aero_efficiency"]
    top_speed_kmh = 338.0 + (team["power_unit_rating"] - 0.90) * 110.0 + team["drs_gain_kmh"] - effective_drag
    cornering_index_kmh = 165.0 + (wing_level * 1.15 * air_density_ratio * team["aero_efficiency"]) - max(0.0, ride_diff * 0.8)

    # Running less wing than optimal causes sliding and increases thermal tire degradation
    undercar_sliding_factor = max(-0.06, -wing_diff * 0.0055)
    combined_deg_mult = (
        circuit["abrasiveness_factor"]
        * team["tire_deg_multiplier"]
        * (1.0 + undercar_sliding_factor)
        * ers_cfg["deg_mult"]
    )

    return {
        "circuit_id": circuit["id"],
        "team_id": team["id"],
        "wing_level": wing_level,
        "optimal_wing_level": opt_wing,
        "top_speed_drs_kmh": round(float(top_speed_kmh), 1),
        "cornering_apex_kmh": round(float(cornering_index_kmh), 1),
        "setup_lap_delta_sec": round(float(total_lap_delta_sec), 3),
        "combined_tire_deg_multiplier": round(float(combined_deg_mult), 3),
    }


def predict_lap_time(
    lap_number: int,
    compound: str,
    stint_age: int,
    total_laps: int = 71,
    track_temp_c: float = 42.0,
    base_lap_sec: float = MEXICO_GP_CONSTANTS["base_lap_sec"],
    fuel_burn_sec_per_lap: float = MEXICO_GP_CONSTANTS["fuel_burn_effect_sec_per_lap"],
    setup_lap_delta_sec: float = 0.0,
    deg_multiplier: float = 1.0,
) -> float:
    """
    Calcula o tempo de volta previsto (em segundos) combinando ritmo base da pista, diferencial de setup/equipe,
    efeito de queima de combustível e degradação não-linear do composto de pneu (com cliff térmico).
    """
    comp_key = compound.upper()
    if comp_key not in TIRE_COMPOUNDS:
        raise ValueError(f"Unknown tire compound: {compound}")

    c = TIRE_COMPOUNDS[comp_key]
    fuel_penalty = (total_laps - lap_number) * fuel_burn_sec_per_lap

    thermal_factor = (1.0 + max(-0.15, (track_temp_c - 40.0) * 0.012)) * deg_multiplier
    linear_wear = c["linear_deg_sec"] * stint_age * thermal_factor
    over_cliff = max(0.0, float(stint_age) - c["cliff_lap"])
    cliff_wear = c["quad_deg_sec"] * (over_cliff ** 2) * thermal_factor

    lap_sec = base_lap_sec + setup_lap_delta_sec + c["delta_base_sec"] + fuel_penalty + linear_wear + cliff_wear
    return round(float(lap_sec), 3)


def simulate_race_strategy(
    stints: List[Tuple[str, int]],
    total_laps: int = 71,
    track_temp_c: float = 42.0,
    pit_loss_sec: float = 21.8,
    base_lap_sec: float = MEXICO_GP_CONSTANTS["base_lap_sec"],
    fuel_burn_sec_per_lap: float = MEXICO_GP_CONSTANTS["fuel_burn_effect_sec_per_lap"],
    setup_lap_delta_sec: float = 0.0,
    deg_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """
    Simula uma corrida completa dada uma lista de stints [(composto, num_voltas), ...].
    Compatível com qualquer circuito do catálogo global e qualquer configuração de carro/equipe.
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
            t = predict_lap_time(
                current_lap,
                comp,
                age,
                total_laps=total_laps,
                track_temp_c=track_temp_c,
                base_lap_sec=base_lap_sec,
                fuel_burn_sec_per_lap=fuel_burn_sec_per_lap,
                setup_lap_delta_sec=setup_lap_delta_sec,
                deg_multiplier=deg_multiplier,
            )
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
    pit_loss_sec: float = 21.8,
    base_lap_sec: float = MEXICO_GP_CONSTANTS["base_lap_sec"],
    fuel_burn_sec_per_lap: float = MEXICO_GP_CONSTANTS["fuel_burn_effect_sec_per_lap"],
    setup_lap_delta_sec: float = 0.0,
    deg_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """
    Encontra a volta exata de Pit Stop que minimiza o tempo total de prova
    e calcula o ganho de Undercut de parar 1 volta antes do pneu entrar no cliff.
    """
    min_pit = max(10, int(total_laps * 0.20))
    max_pit = min(total_laps - 10, int(total_laps * 0.78))
    best_lap = min_pit
    best_time = float("inf")
    curve = []

    for pit_lap in range(min_pit, max_pit + 1):
        sim = simulate_race_strategy(
            [(start_compound, pit_lap), (end_compound, total_laps - pit_lap)],
            total_laps=total_laps,
            track_temp_c=track_temp_c,
            pit_loss_sec=pit_loss_sec,
            base_lap_sec=base_lap_sec,
            fuel_burn_sec_per_lap=fuel_burn_sec_per_lap,
            setup_lap_delta_sec=setup_lap_delta_sec,
            deg_multiplier=deg_multiplier,
        )
        t_total = sim["total_race_time_sec"]
        curve.append({"pit_lap": pit_lap, "total_time_sec": t_total})
        if t_total < best_time:
            best_time = t_total
            best_lap = pit_lap

    worn_lap = predict_lap_time(
        best_lap + 1,
        start_compound,
        best_lap + 1,
        total_laps,
        track_temp_c,
        base_lap_sec,
        fuel_burn_sec_per_lap,
        setup_lap_delta_sec,
        deg_multiplier,
    )
    fresh_lap = predict_lap_time(
        best_lap + 1,
        end_compound,
        1,
        total_laps,
        track_temp_c,
        base_lap_sec,
        fuel_burn_sec_per_lap,
        setup_lap_delta_sec,
        deg_multiplier,
    )
    undercut_gain_sec = round(worn_lap - fresh_lap, 3)

    return {
        "optimal_pit_lap": best_lap,
        "optimal_total_time_sec": best_time,
        "undercut_single_lap_gain_sec": undercut_gain_sec,
        "pit_window_optimal_range": [max(min_pit, best_lap - 2), min(max_pit, best_lap + 2)],
        "tradeoff_curve": curve,
    }
