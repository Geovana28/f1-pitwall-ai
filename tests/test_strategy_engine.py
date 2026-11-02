"""
Unit tests for F1 Global Pit-Wall Telemetry, 24-Circuit Catalog, 10-Team Car Setup Physics,
Monte Carlo Safety Car & Wet Crossover Simulation, Lap Telemetry Traces, and Race Engineer AI Agent.
"""

import pytest
from src.strategy_engine import (
    load_circuits_catalog,
    load_teams_catalog,
    get_circuit_by_id,
    get_team_by_id,
    get_next_upcoming_gp,
    evaluate_car_setup_physics,
    compute_altitude_aerodynamics,
    predict_lap_time,
    simulate_race_strategy,
    find_optimal_one_stop_window,
)
from src.monte_carlo_race import (
    compute_wet_crossover_curve,
    run_monte_carlo_strategy_comparison,
)
from src.telemetry_loader import generate_lap_telemetry_comparison
from src.race_engineer_agent import ask_race_engineer


def test_global_24_circuits_catalog_integrity():
    """Catalog must contain all 24 FIA World Championship circuits with valid physical parameters."""
    circuits = load_circuits_catalog()
    assert len(circuits) == 24
    ids = {c["id"] for c in circuits}
    assert {"interlagos", "mexico", "monaco", "monza", "spa", "suzuka", "silverstone"}.issubset(ids)
    for c in circuits:
        assert c["total_laps"] >= 44
        assert len(c["pirelli_allocation"]) == 3


def test_global_10_teams_catalog_integrity():
    """Catalog must contain all 10 F1 constructor teams and 20 drivers."""
    teams = load_teams_catalog()
    assert len(teams) == 10
    total_drivers = sum(len(t["drivers"]) for t in teams)
    assert total_drivers == 20


def test_next_upcoming_gp_detection_after_mexico():
    """After Mexico City GP (2026-10-25), the next upcoming GP on 2026-11-02 must be São Paulo (Interlagos)."""
    next_gp = get_next_upcoming_gp("2026-11-02")
    assert next_gp["id"] == "interlagos"
    assert next_gp["days_until_race"] == 6


def test_car_setup_physics_wing_tradeoff():
    """Low wing in Monza increases top speed vs high wing, while optimal wing minimizes lap time delta."""
    low_wing = evaluate_car_setup_physics("monza", "ferrari", wing_level=8)
    high_wing = evaluate_car_setup_physics("monza", "ferrari", wing_level=45)
    assert low_wing["top_speed_drs_kmh"] > high_wing["top_speed_drs_kmh"]
    assert low_wing["setup_lap_delta_sec"] < high_wing["setup_lap_delta_sec"]


def test_mexico_city_altitude_reduces_air_density():
    """At 2,240m (Mexico City), air density and aerodynamic drag/downforce drop by ~22-25% vs sea level."""
    aero = compute_altitude_aerodynamics(altitude_m=2240.0, temp_c=24.0)
    assert 0.90 <= aero["air_density_kg_m3"] <= 0.98
    assert 20.0 <= aero["drag_and_downforce_reduction_pct"] <= 27.0


def test_tire_cliff_increases_degradation_nonlinearly():
    """Soft tire past its cliff lap (18) must degrade faster per lap than during early stint."""
    early_delta = predict_lap_time(5, "SOFT", 6) - predict_lap_time(5, "SOFT", 5)
    cliff_delta = predict_lap_time(25, "SOFT", 25) - predict_lap_time(25, "SOFT", 24)
    assert cliff_delta > early_delta * 2.0


def test_fuel_burn_makes_car_faster_at_equal_tire_age():
    """On fresh tires (stint_age=1), lap 60 must be faster than lap 5 due to lighter fuel load."""
    lap_early = predict_lap_time(lap_number=5, compound="MEDIUM", stint_age=1)
    lap_late = predict_lap_time(lap_number=60, compound="MEDIUM", stint_age=1)
    assert lap_late < lap_early


def test_optimal_pit_window_mexico_gp():
    """Optimal Medium -> Hard 1-stop in 71-lap Mexico GP should lie between lap 24 and 34 with positive undercut."""
    opt = find_optimal_one_stop_window("MEDIUM", "HARD", total_laps=71, track_temp_c=42.0)
    assert 24 <= opt["optimal_pit_lap"] <= 34
    assert opt["undercut_single_lap_gain_sec"] > 0.8


def test_invalid_stint_laps_sum_raises():
    """Strategy whose stints do not sum to total_laps must raise ValueError."""
    with pytest.raises(ValueError):
        simulate_race_strategy([("SOFT", 20), ("HARD", 30)], total_laps=71)


def test_wet_tire_crossover_curve():
    """Wet crossover model must identify water depth where Intermediate beats Slick and Wet beats Intermediate."""
    cross = compute_wet_crossover_curve("interlagos", "MEDIUM")
    assert 0.25 <= cross["slick_to_inter_crossover_mm"] <= 1.50
    assert cross["inter_to_wet_crossover_mm"] > cross["slick_to_inter_crossover_mm"]
    assert cross["slick_lap_sec"][0] < cross["intermediate_lap_sec"][0]
    assert cross["full_wet_lap_sec"][-1] < cross["slick_lap_sec"][-1]


def test_monte_carlo_safety_car_simulation():
    """Monte Carlo strategy comparison must return probabilities summing to ~100% across the 3 candidate strategies."""
    mc = run_monte_carlo_strategy_comparison("interlagos", "mclaren", wing_level=34, n_simulations=200, seed=7)
    assert len(mc["strategies_ranking"]) == 3
    total_prob = sum(s["win_probability_pct"] for s in mc["strategies_ranking"])
    assert 99.0 <= total_prob <= 101.0


def test_telemetry_trace_generator():
    """Lap telemetry generator must produce aligned arrays of distance, speed, throttle, brake and DRS zones."""
    tel = generate_lap_telemetry_comparison("spa", "red_bull", 23, "mercedes", 35, n_points=50)
    assert len(tel["distance_m"]) == 50
    assert len(tel["speed_a_kmh"]) == 50
    assert max(tel["speed_a_kmh"]) > max(tel["speed_b_kmh"])  # lower wing on Spa straight is faster


def test_race_engineer_agent_tool_routing_and_guardrail():
    """Race Engineer AI must route rain, Monte Carlo, and pit window queries to the right tools and block off-topic prompts."""
    r_rain = ask_race_engineer("Vai começar a chover na volta 40 em Interlagos, quando colocamos Intermediates?", "interlagos", "mercedes")
    assert r_rain["selected_tool"] == "evaluate_rain_crossover"

    r_mc = ask_race_engineer("Qual a probabilidade no Monte Carlo com Safety Car para 1-stop vs 2-stop?", "singapore", "ferrari")
    assert r_mc["selected_tool"] == "simulate_safety_car_monte_carlo"

    r_guard = ask_race_engineer("Me passe uma receita de bolo de cenoura com chocolate", "interlagos", "mercedes")
    assert r_guard["selected_tool"] == "out_of_scope_guardrail"
    assert r_guard["tool_output"]["blocked"] is True
