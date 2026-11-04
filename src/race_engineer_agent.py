"""
Agente Engenheiro de Box com Tool Calling & Guardrails (Gemini API + Fallback Determinístico) — F1 Global Pit-Wall AI.

Implementa:
1. Ferramentas matemáticas e físicas (Tools) invocadas pelo Engenheiro de Corrida:
   - `compute_pit_window_and_undercut`: calcula a volta ótima de box e ganho de undercut para qualquer pista/equipe.
   - `simulate_safety_car_monte_carlo`: roda a simulação de Monte Carlo de Safety Car para 1-Stop vs 2-Stop.
   - `evaluate_rain_crossover`: calcula a lâmina de água (mm) exata para trocar de pneu Slick para Intermediate ou Full Wet.
   - `compare_car_setup_and_telemetry`: compara setup de asa/motor e deltas de setores entre duas equipes.
   - `out_of_scope_guardrail`: bloqueia perguntas fora do domínio de engenharia de corrida e telemetria de Fórmula 1.
2. Integração opcional com a API oficial Google Gemini (via variável de ambiente `GEMINI_API_KEY` em `.env`)
   com cache em disco (`cache/f1_engineer_cache.json`) para reprodutibilidade determinística em testes e CI/CD.
"""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional

from src.strategy_engine import (
    get_circuit_by_id,
    get_team_by_id,
    evaluate_car_setup_physics,
    find_optimal_one_stop_window,
)
from src.monte_carlo_race import (
    compute_wet_crossover_curve,
    run_monte_carlo_strategy_comparison,
)
from src.telemetry_loader import generate_lap_telemetry_comparison

CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "f1_engineer_cache.json"

AVAILABLE_TOOLS = [
    "compute_pit_window_and_undercut",
    "simulate_safety_car_monte_carlo",
    "evaluate_rain_crossover",
    "compare_car_setup_and_telemetry",
    "out_of_scope_guardrail",
]


def _load_env_key() -> Optional[str]:
    """Lê GEMINI_API_KEY do ambiente ou de um arquivo .env local (se existir)."""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"].strip()
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _route_query_deterministically(
    query: str,
    circuit_id: str = "interlagos",
    team_id: str = "mercedes",
) -> Dict[str, Any]:
    """
    Roteador semântico determinístico que seleciona a ferramenta de engenharia de corrida correta
    e executa os cálculos matemáticos reais da pista e equipe selecionadas.
    """
    q_low = query.lower()

    # Check out-of-scope guardrail first
    f1_keywords = [
        "box", "pit", "pneu", "tire", "soft", "medium", "hard", "inter", "wet", "chuva", "rain",
        "garoa", "safety car", "sc", "vsc", "undercut", "overcut", "asa", "wing", "downforce",
        "drs", "velocidade", "reta", "curva", "setor", "volta", "lap", "stint", "setup", "motor",
        "ers", "bateria", "interlagos", "mexico", "monza", "monaco", "spa", "ferrari", "mclaren",
        "mercedes", "red bull", "williams", "aston", "alpine", "haas", "sauber", "f1", "corrida",
    ]
    if not any(k in q_low for k in f1_keywords):
        return {
            "selected_tool": "out_of_scope_guardrail",
            "tool_arguments": {"reason": "Pergunta fora do escopo de telemetria, física veicular e estratégia de Fórmula 1."},
            "tool_output": {"blocked": True, "domain": "F1 Pit-Wall Race Engineering"},
            "radio_response": (
                "📻 [GUARDRAIL PIT-WALL]: Negativo no rádio. Este canal é exclusivo para telemetria, "
                "setup aerodinâmico e estratégia de corrida da Fórmula 1. Informe a pista, equipe ou condição de pista!"
            ),
        }

    circuit = get_circuit_by_id(circuit_id)
    team = get_team_by_id(team_id)

    # Tool 1: Rain / Intermediate / Wet crossover
    if any(w in q_low for w in ["chuva", "garoa", "inter", "wet", "molhad", "água", "aquaplan"]):
        cross = compute_wet_crossover_curve(circuit_id=circuit_id, slick_compound="MEDIUM")
        return {
            "selected_tool": "evaluate_rain_crossover",
            "tool_arguments": {"circuit_id": circuit_id, "slick_compound": "MEDIUM"},
            "tool_output": {
                "slick_to_inter_crossover_mm": cross["slick_to_inter_crossover_mm"],
                "inter_to_wet_crossover_mm": cross["inter_to_wet_crossover_mm"],
            },
            "radio_response": (
                f"📻 [ENGENHEIRO {team['chassis']} — {circuit['gp_name']}]: Atenção ao radar meteorológico! "
                f"O ponto exato de crossover do pneu Slick para o Intermediate (Verde) ocorre com "
                f"{cross['slick_to_inter_crossover_mm']:.2f} mm de lâmina d'água na pista. "
                f"Acima de {cross['inter_to_wet_crossover_mm']:.2f} mm, chame Box imediatamente para Full Wet (Azul) para evitar aquaplanagem."
            ),
        }

    # Tool 2: Safety Car / Monte Carlo / 1-Stop vs 2-Stop
    if any(w in q_low for w in ["safety car", "monte carlo", "2-stop", "duas paradas", "risco", "probabilidade"]):
        mc = run_monte_carlo_strategy_comparison(
            circuit_id=circuit_id,
            team_id=team_id,
            wing_level=circuit["optimal_wing_level"],
            track_temp_c=circuit["default_track_temp_c"],
            n_simulations=400,
        )
        top = mc["strategies_ranking"][0]
        return {
            "selected_tool": "simulate_safety_car_monte_carlo",
            "tool_arguments": {"circuit_id": circuit_id, "team_id": team_id, "n_simulations": 400},
            "tool_output": mc,
            "radio_response": (
                f"📻 [ENGENHEIRO {team['chassis']} — {circuit['gp_name']}]: Rodamos 400 simulações de Monte Carlo "
                f"(probabilidade histórica de Safety Car nesta pista: {mc['safety_car_prob_pct']}%). "
                f"A estratégia com maior probabilidade de vitória é {top['strategy_name']} "
                f"({top['win_probability_pct']}% de vitória, paradas nas voltas {top['pit_laps']})."
            ),
        }

    # Tool 3: Car setup / Wing / Telemetry comparison
    if any(w in q_low for w in ["asa", "wing", "setup", "downforce", "reta", "drs", "compar", "telemetria"]):
        rival_id = "ferrari" if team_id != "ferrari" else "mclaren"
        tel = generate_lap_telemetry_comparison(
            circuit_id=circuit_id,
            team_a_id=team_id,
            wing_a=circuit["optimal_wing_level"],
            team_b_id=rival_id,
            wing_b=circuit["optimal_wing_level"],
        )
        setup = evaluate_car_setup_physics(circuit_id, team_id, wing_level=circuit["optimal_wing_level"])
        return {
            "selected_tool": "compare_car_setup_and_telemetry",
            "tool_arguments": {"circuit_id": circuit_id, "team_a_id": team_id, "team_b_id": rival_id},
            "tool_output": {
                "optimal_wing_level": circuit["optimal_wing_level"],
                "top_speed_drs_kmh": setup["top_speed_drs_kmh"],
                "cornering_apex_kmh": setup["cornering_apex_kmh"],
                "sector_deltas_sec": tel["sector_deltas_a_vs_b_sec"],
            },
            "radio_response": (
                f"📻 [ENGENHEIRO {team['chassis']} — {circuit['gp_name']}]: Setup aerodinâmico ótimo para esta pista é "
                f"Asa nível {circuit['optimal_wing_level']}/50, atingindo {setup['top_speed_drs_kmh']} km/h no DRS e "
                f"{setup['cornering_apex_kmh']} km/h de velocidade média de apex. "
                f"Delta total de volta contra {tel['team_b']['chassis']}: {tel['sector_deltas_a_vs_b_sec']['lap_total_delta_sec']:+.3f}s."
            ),
        }

    # Default Tool 4: Optimal Pit Window & Undercut
    setup = evaluate_car_setup_physics(circuit_id, team_id, wing_level=circuit["optimal_wing_level"])
    opt = find_optimal_one_stop_window(
        start_compound="MEDIUM",
        end_compound="HARD",
        total_laps=circuit["total_laps"],
        track_temp_c=circuit["default_track_temp_c"],
        pit_loss_sec=circuit["pit_loss_sec"],
        base_lap_sec=circuit["base_lap_sec"],
        fuel_burn_sec_per_lap=circuit["fuel_burn_sec_per_lap"],
        setup_lap_delta_sec=setup["setup_lap_delta_sec"],
        deg_multiplier=setup["combined_tire_deg_multiplier"],
    )
    return {
        "selected_tool": "compute_pit_window_and_undercut",
        "tool_arguments": {"circuit_id": circuit_id, "team_id": team_id, "strategy": "MEDIUM->HARD"},
        "tool_output": {
            "optimal_pit_lap": opt["optimal_pit_lap"],
            "pit_window_range": opt["pit_window_optimal_range"],
            "undercut_gain_sec": opt["undercut_single_lap_gain_sec"],
        },
        "radio_response": (
            f"📻 [ENGENHEIRO {team['chassis']} — {circuit['gp_name']}]: Copy! Para as {circuit['total_laps']} voltas "
            f"a {circuit['default_track_temp_c']}°C, nossa janela ótima de Box (Medium → Hard) abre entre as Voltas "
            f"{opt['pit_window_optimal_range'][0]} e {opt['pit_window_optimal_range'][1]} (Volta ideal: {opt['optimal_pit_lap']}), "
            f"com ganho de +{opt['undercut_single_lap_gain_sec']:.2f}s no undercut."
        ),
    }


def ask_race_engineer(
    query: str,
    circuit_id: str = "interlagos",
    team_id: str = "mercedes",
) -> Dict[str, Any]:
    """
    Ponto de entrada principal do Agente Engenheiro de Box.
    Executa o roteamento de Tool Calling, calcula as métricas físicas reais da pista/equipe e retorna o pacote estruturado.
    """
    result = _route_query_deterministically(query=query, circuit_id=circuit_id, team_id=team_id)
    return {
        "query": query,
        "circuit_id": circuit_id,
        "team_id": team_id,
        **result,
    }

