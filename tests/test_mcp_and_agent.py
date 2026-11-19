"""
Test Suite for F1 Autonomous Race Engineer (MCP & LangGraph).
Validates MCP Tools JSON-RPC, StateGraph transitions, and Machine Learning integration.
"""

import pytest
import json
from mcp_server.server import F1RaceEngineerMCPServer
from agent.graph import F1RaceEngineerGraph


def test_mcp_server_lists_tools():
    """Verifies that MCP server correctly registers and exports official MCP tool schemas."""
    server = F1RaceEngineerMCPServer()
    tools = server.list_tools()
    tool_names = [t["name"] for t in tools]

    assert "get_live_telemetry" in tool_names
    assert "predict_podium_ml" in tool_names
    assert "simulate_pit_stop_strategy" in tool_names
    assert "validate_telemetry_physics" in tool_names


def test_mcp_call_predict_podium():
    """Verifies executing ML podium prediction via MCP protocol."""
    server = F1RaceEngineerMCPServer()
    args = {
        "driver": "Andrea Kimi Antonelli",
        "circuit": "Interlagos",
        "grid_position": 2,
        "tyre_compound": "MEDIUM",
        "current_lap": 22
    }
    res = server.call_tool("predict_podium_ml", args)
    assert res["isError"] is False
    
    data = json.loads(res["content"][0]["text"])
    assert "podium_probability" in data
    assert data["podium_probability"] >= 0.0 and data["podium_probability"] <= 1.0
    assert data["model_validation_accuracy"] == "92.07%"


def test_mcp_call_pit_strategy():
    """Verifies Monte Carlo strategy tool returns optimal window and compound."""
    server = F1RaceEngineerMCPServer()
    args = {
        "current_compound": "MEDIUM",
        "laps_on_tyre": 22,
        "weather_forecast": "Dry",
        "safety_car_prob": 0.20
    }
    res = server.call_tool("simulate_pit_stop_strategy", args)
    assert res["isError"] is False
    
    data = json.loads(res["content"][0]["text"])
    assert "recommended_action" in data
    assert "BOX" in data["recommended_action"]


def test_langgraph_agent_execution_trace():
    """Verifies full execution of the LangGraph StateGraph, node transitions and trace entries."""
    graph = F1RaceEngineerGraph()
    state = graph.run(
        user_query="Qual a chance do Antonelli ir pro pódio em Interlagos largando de P2?",
        driver="Andrea Kimi Antonelli",
        circuit="Interlagos",
        lap=22,
        grid=2
    )

    assert state.intent == "PODIUM_PREDICTION"
    assert len(state.execution_trace) >= 4  # Router, MCP Tools, Safety, Radio
    assert "RÁDIO BOX" in state.radio_response
    assert state.physics_approved is True
    assert "podium_probability" in state.mcp_tool_results.get("predict_podium_ml", {})


def test_langgraph_pit_call():
    """Verifies agent execution for pit stop radio inquiry."""
    graph = F1RaceEngineerGraph()
    state = graph.run(
        user_query="Bono, devemos parar agora no box ou estender o stint?",
        driver="Andrea Kimi Antonelli",
        circuit="Interlagos",
        lap=20
    )

    assert state.intent == "PIT_STRATEGY"
    assert state.tactical_code in ["BOX_NOW", "EXTEND"]
    assert len(state.radio_response) > 50
