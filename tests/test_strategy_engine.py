"""
Unit tests for F1 Global Pit-Wall Telemetry, 24-Circuit Catalog, 10-Team Car Setup Physics,
Barometric Altitude Aerodynamics, Tire Degradation & Pit Strategy Engine.
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
