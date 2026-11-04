"""
Simulador Estocástico de Corrida por Monte Carlo (Safety Car, VSC e Crossover de Chuva) — F1 Global Pit-Wall AI.

Modelos implementados:
1. Simulação de Monte Carlo (N corridas):
   - Sorteio de intervenções de Safety Car (processo de Bernoulli por corrida calibrado por safety_car_prob da pista,
     com economia de ~45% no tempo perdido de Pit Lane se a janela de parada coincidir com o Safety Car).
   - Variância estocástica de Pit Stop (distribuição Log-Normal calibrada pelo avg_pit_stop_sec da equipe,
     capturando a cauda direita de atrasos mecânicos na troca de pneus).
   - Ruído volta a volta de tráfego/vento (Gaussiano sigma = 0.18s).
2. Crossover de Pneu de Chuva (Slick vs. Intermediate vs. Full Wet):
   - Modela a lâmina d'água na pista w(t) em milímetros (0.0 mm a 5.0 mm) e determina a volta exata de Crossover
     em que o pneu Intermediate (Verde) passa a ser mais rápido que o Slick (Soft/Medium/Hard), e quando o Full Wet (Azul) supera o Intermediate.
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple
from src.strategy_engine import (
    get_circuit_by_id,
    get_team_by_id,
    evaluate_car_setup_physics,
    predict_lap_time,
)


WET_COMPOUNDS: Dict[str, Dict[str, float]] = {
    "INTERMEDIATE": {
        "dry_penalty_sec": 4.20,     # 4.2s slower than Soft on bone-dry track (overheats)
        "optimal_water_mm": 1.40,    # sweet spot water depth (mm)
        "aquaplaning_res": 2.80,     # handles up to ~2.8mm before severe aquaplaning
    },
    "WET": {
        "dry_penalty_sec": 8.50,     # 8.5s slower on dry track
        "optimal_water_mm": 3.50,    # heavy standing water sweet spot
        "aquaplaning_res": 6.00,
    },
}


def compute_wet_crossover_curve(
    circuit_id: str = "interlagos",
    slick_compound: str = "MEDIUM",
) -> Dict[str, Any]:
    """
    Calcula a curva de tempo de volta em função da lâmina de água na pista (0.0 mm a 4.5 mm)
    para três compostos: Slick (Seco), Intermediate (Chuva Leve/Moderada) e Full Wet (Chuva Forte),
    identificando os pontos exatos de Crossover (em mm de água) onde vale a pena chamar o carro ao box.
    """
    circuit = get_circuit_by_id(circuit_id)
    base_dry = predict_lap_time(
        lap_number=25,
        compound=slick_compound,
        stint_age=8,
        total_laps=circuit["total_laps"],
        track_temp_c=24.0,
        base_lap_sec=circuit["base_lap_sec"],
        fuel_burn_sec_per_lap=circuit["fuel_burn_sec_per_lap"],
    )

    water_levels = [round(x * 0.25, 2) for x in range(0, 19)]  # 0.00 to 4.50 mm
    slick_times: List[float] = []
    inter_times: List[float] = []
    wet_times: List[float] = []

    slick_to_inter_mm = None
    inter_to_wet_mm = None

    for w in water_levels:
        # Slick loses grip quadratically with water film depth (aquaplaning)
        t_slick = base_dry + 4.8 * (w ** 1.85) + (3.2 * w if w > 0.1 else 0.0)
        # Intermediate is slow on bone dry (0mm), optimal around 0.8-2.0mm, degrades above 2.6mm
        t_inter = base_dry + 4.2 * math.exp(-1.9 * w) + 0.85 * max(0.0, w - 0.6) ** 1.6 + 2.4 * max(0.0, w - 2.4) ** 2
        # Full Wet is very slow on dry, steady in heavy rain
        t_wet = base_dry + 8.5 * math.exp(-0.95 * w) + 0.45 * max(0.0, w - 1.2) ** 1.3

        slick_times.append(round(float(t_slick), 3))
        inter_times.append(round(float(t_inter), 3))
        wet_times.append(round(float(t_wet), 3))

        if slick_to_inter_mm is None and t_inter < t_slick:
            slick_to_inter_mm = w
        if inter_to_wet_mm is None and t_wet < t_inter:
            inter_to_wet_mm = w

    return {
        "circuit_id": circuit["id"],
        "slick_compound": slick_compound.upper(),
        "slick_to_inter_crossover_mm": slick_to_inter_mm if slick_to_inter_mm is not None else 0.75,
        "inter_to_wet_crossover_mm": inter_to_wet_mm if inter_to_wet_mm is not None else 2.75,
        "water_mm_grid": water_levels,
        "slick_lap_sec": slick_times,
        "intermediate_lap_sec": inter_times,
        "full_wet_lap_sec": wet_times,
    }


def run_monte_carlo_strategy_comparison(
    circuit_id: str = "interlagos",
    team_id: str = "mercedes",
    wing_level: int = 34,
    track_temp_c: float = 38.0,
    n_simulations: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executa N simulações estocásticas de Monte Carlo comparando 3 estratégias clássicas na pista escolhida:
    - Estratégia A (1-Stop Equilibrada): MEDIUM -> HARD
    - Estratégia B (1-Stop Agressiva): SOFT -> HARD
    - Estratégia C (2-Stops Ataque Total): SOFT -> MEDIUM -> MEDIUM

    Modela:
    - Sorteio de Safety Car na corrida com probabilidade p = circuit['safety_car_prob'].
      Se o Safety Car ocorre dentro de +/- 4 voltas da parada planejada, a perda de Pit Lane cai em 45% ("cheap stop").
    - Ruído log-normal no tempo de parada de box da equipe (capturando "slow stops" por falha de pistola pneumática).
    - Variabilidade volta a volta de tráfego (sigma = 0.18s).
    """
    rng = np.random.default_rng(seed)
    circuit = get_circuit_by_id(circuit_id)
    team = get_team_by_id(team_id)
    setup = evaluate_car_setup_physics(circuit_id, team_id, wing_level=wing_level)

    L = circuit["total_laps"]
    p_sc = circuit["safety_car_prob"]
    pit_loss_base = circuit["pit_loss_sec"] + (team["avg_pit_stop_sec"] - 2.25)

    # Define 3 candidate strategies scaled to the circuit's total laps
    p1_med = int(round(L * 0.40))
    p1_soft = int(round(L * 0.28))
    p2_a = int(round(L * 0.27))
    p2_b = int(round(L * 0.62))

    strategies = {
        "1-Stop (MEDIUM → HARD)": [("MEDIUM", p1_med), ("HARD", L - p1_med)],
        "1-Stop (SOFT → HARD)": [("SOFT", p1_soft), ("HARD", L - p1_soft)],
        "2-Stop (SOFT → MEDIUM → MEDIUM)": [("SOFT", p2_a), ("MEDIUM", p2_b - p2_a), ("MEDIUM", L - p2_b)],
    }

    # Pre-compute deterministic base race time for each strategy (without pit loss)
    det_base_times: Dict[str, float] = {}
    pit_laps_map: Dict[str, List[int]] = {}

    for name, stints in strategies.items():
        cur_lap = 1
        total_clean = 0.0
        p_laps: List[int] = []
        for idx, (comp, stint_len) in enumerate(stints):
            if idx > 0:
                p_laps.append(cur_lap - 1)
            for age in range(1, stint_len + 1):
                total_clean += predict_lap_time(
                    cur_lap,
                    comp,
                    age,
                    total_laps=L,
                    track_temp_c=track_temp_c,
                    base_lap_sec=circuit["base_lap_sec"],
                    fuel_burn_sec_per_lap=circuit["fuel_burn_sec_per_lap"],
                    setup_lap_delta_sec=setup["setup_lap_delta_sec"],
                    deg_multiplier=setup["combined_tire_deg_multiplier"],
                )
                cur_lap += 1
        det_base_times[name] = total_clean
        pit_laps_map[name] = p_laps

    # Vectorized Monte Carlo simulation across n_simulations
    sim_totals: Dict[str, np.ndarray] = {name: np.zeros(n_simulations) for name in strategies}

    has_sc = rng.random(n_simulations) < p_sc
    sc_lap = rng.integers(low=int(L * 0.18), high=int(L * 0.78), size=n_simulations)

    for name, p_laps in pit_laps_map.items():
        n_stops = len(p_laps)
        # Traffic & wind pace noise over L laps
        pace_noise = rng.normal(0.0, 0.18 * math.sqrt(L), size=n_simulations)
        # Pit stop execution jitter (log-normal heavy right tail for slow stops)
        pit_jitter = rng.lognormal(mean=0.0, sigma=0.22, size=(n_simulations, n_stops)) - 1.0
        total_pit_loss = np.zeros(n_simulations)

        for s_idx, p_lap in enumerate(p_laps):
            # If SC happens within +/- 4 laps of planned stop, driver boxes under SC and saves 42% of pit lane loss
            sc_window_hit = has_sc & (np.abs(sc_lap - p_lap) <= 4)
            discount = np.where(sc_window_hit, 0.58, 1.00)
            stop_loss = pit_loss_base * discount + pit_jitter[:, s_idx] * 0.9
            total_pit_loss += stop_loss

        # 2-stop has dirty-air overtake penalty (+2.4s average traffic loss on high-downforce tracks)
        traffic_penalty = (2.6 * circuit["downforce_demand"]) if n_stops >= 2 else 0.0
        sim_totals[name] = det_base_times[name] + total_pit_loss + pace_noise + traffic_penalty

    # Compute win probabilities (which strategy finishes fastest in each simulated race)
    names_list = list(strategies.keys())
    matrix = np.vstack([sim_totals[k] for k in names_list])  # shape (3, n_simulations)
    winners = np.argmin(matrix, axis=0)

    results_summary = []
    for idx, name in enumerate(names_list):
        arr = sim_totals[name]
        win_pct = round(float(np.mean(winners == idx) * 100.0), 1)
        results_summary.append({
            "strategy_name": name,
            "pit_laps": pit_laps_map[name],
            "win_probability_pct": win_pct,
            "mean_race_time_sec": round(float(np.mean(arr)), 2),
            "p10_optimistic_sec": round(float(np.percentile(arr, 10)), 2),
            "p90_risk_sec": round(float(np.percentile(arr, 90)), 2),
        })

    results_summary.sort(key=lambda x: x["win_probability_pct"], reverse=True)
    return {
        "circuit_id": circuit["id"],
        "team_id": team["id"],
        "n_simulations": n_simulations,
        "safety_car_prob_pct": round(p_sc * 100.0, 1),
        "recommended_strategy": results_summary[0]["strategy_name"],
        "strategies_ranking": results_summary,
    }

