"""
Unit tests for F1 Telemetry, Mexico GP Altitude Aerodynamics, Tire Degradation & Pit Strategy Engine.
"""

import pytest
from src.strategy_engine import (
    compute_altitude_aerodynamics,
    predict_lap_time,
    simulate_race_strategy,
    find_optimal_one_stop_window,
)


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
