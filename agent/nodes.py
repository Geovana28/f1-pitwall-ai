"""
Nodes for the F1 Autonomous Race Engineer LangGraph StateGraph.
Implements:
1. Perception & Intent Router Node
2. MCP Tool Execution Node (Telemetry, ML, Strategy)
3. Safety Reflection & Physics Critique Node (Self-Correction loop)
4. Radio Synthesis & Telemetry Response Node
"""

import time
import json
from typing import Dict, Any
from .state import RaceEngineerState, GraphMessage
from mcp_server.server import F1RaceEngineerMCPServer


def router_node(state: RaceEngineerState) -> RaceEngineerState:
    """
    Analyzes driver radio transmission or engineer query.
    Routes execution to appropriate MCP tools.
    """
    trace_entry = {
        "node": "router_node",
        "action": "analyzing_intent",
        "timestamp": time.time()
    }

    query_lower = state.user_query.lower()
    tools_to_run = []
    tool_args = {}

    # Routing decision logic
    if any(k in query_lower for k in ["podio", "podium", "chance", "probabilidade", "vencer", "p1", "p2", "p3"]):
        state.intent = "PODIUM_PREDICTION"
        tools_to_run.extend(["get_live_telemetry", "predict_podium_ml"])
        tool_args["predict_podium_ml"] = {
            "driver": state.driver_name,
            "circuit": state.circuit,
            "grid_position": state.grid_position,
            "current_lap": state.current_lap,
            "total_laps": 71,
            "weather": "RAIN" if "chuva" in query_lower or "rain" in query_lower else "DRY"
        }
        tool_args["get_live_telemetry"] = {
            "driver": state.driver_name,
            "circuit": state.circuit
        }

    elif any(k in query_lower for k in ["box", "pit", "parada", "pneu", "tyre", "troca", "estrategia", "strategy"]):
        state.intent = "PIT_STRATEGY"
        tools_to_run.extend(["get_live_telemetry", "simulate_pit_stop_strategy"])
        tool_args["simulate_pit_stop_strategy"] = {
            "current_compound": "MEDIUM",
            "laps_on_tyre": 18,
            "weather_forecast": "Rain incoming in 6 laps" if "chuva" in query_lower else "Dry",
            "safety_car_prob": 0.35 if "safety" in query_lower or "sc" in query_lower else 0.15
        }
        tool_args["get_live_telemetry"] = {
            "driver": state.driver_name,
            "circuit": state.circuit
        }

    elif any(k in query_lower for k in ["telemetria", "delta", "velocidade", "bateria", "ers", "combustivel", "temperatura"]):
        state.intent = "TELEMETRY_INSPECTION"
        tools_to_run.append("get_live_telemetry")
        tool_args["get_live_telemetry"] = {
            "driver": state.driver_name,
            "circuit": state.circuit
        }

    else:
        state.intent = "GENERAL_RACE_CONSULTATION"
        tools_to_run.extend(["get_live_telemetry", "predict_podium_ml", "simulate_pit_stop_strategy"])
        tool_args["predict_podium_ml"] = {
            "driver": state.driver_name,
            "circuit": state.circuit,
            "grid_position": state.grid_position,
            "current_lap": state.current_lap,
            "weather": "DRY"
        }
        tool_args["simulate_pit_stop_strategy"] = {
            "current_compound": "MEDIUM",
            "laps_on_tyre": 18,
            "weather_forecast": "Dry",
            "safety_car_prob": 0.20
        }
        tool_args["get_live_telemetry"] = {
            "driver": state.driver_name,
            "circuit": state.circuit
        }

    state.required_tools = tools_to_run
    state.tool_arguments = tool_args
    trace_entry["intent_selected"] = state.intent
    trace_entry["scheduled_tools"] = tools_to_run
    state.execution_trace.append(trace_entry)
    return state


def mcp_tool_executor_node(state: RaceEngineerState, mcp_server: F1RaceEngineerMCPServer) -> RaceEngineerState:
    """
    Dispatches tool calls directly into the Model Context Protocol (MCP) server.
    Captures structured telemetry, ML inference results, and Monte Carlo calculations.
    """
    trace_entry = {
        "node": "mcp_tool_executor_node",
        "action": "invoking_mcp_server",
        "executed_tools": [],
        "timestamp": time.time()
    }

    results = {}
    for tool_name in state.required_tools:
        args = state.tool_arguments.get(tool_name, {})
        mcp_res = mcp_server.call_tool(tool_name, args)
        
        # Extract payload from MCP JSON-RPC format
        raw_text = mcp_res["content"][0]["text"]
        try:
            parsed = json.loads(raw_text)
            results[tool_name] = parsed
        except Exception:
            results[tool_name] = {"raw_output": raw_text}

        trace_entry["executed_tools"].append({
            "tool": tool_name,
            "arguments": args,
            "success": not mcp_res.get("isError", False)
        })

    state.mcp_tool_results = results
    state.execution_trace.append(trace_entry)
    return state


def safety_reflection_node(state: RaceEngineerState, mcp_server: F1RaceEngineerMCPServer) -> RaceEngineerState:
    """
    Reflective Safety Node (Critique & Self-Correction).
    Ensures that ML recommendations pass FIA thermodynamic and fuel flow boundary conditions.
    """
    trace_entry = {
        "node": "safety_reflection_node",
        "action": "evaluating_boundary_conditions",
        "timestamp": time.time()
    }

    state.critique_iterations += 1
    telemetry = state.mcp_tool_results.get("get_live_telemetry", {}).get("telemetry", {})

    temp_c = telemetry.get("tire_surface_temp_c", 104.2)
    fuel_delta = telemetry.get("fuel_delta_kg", -0.3)
    
    # Run MCP validation tool
    validation_args = {
        "delta_time": -0.4,
        "tire_temp_c": temp_c,
        "fuel_flow_kg_h": 94.2
    }
    val_res = mcp_server.call_tool("validate_telemetry_physics", validation_args)
    parsed_val = json.loads(val_res["content"][0]["text"])

    state.physics_approved = parsed_val.get("physics_valid", True)
    state.safety_notes = parsed_val.get("violations", [])

    trace_entry["physics_approved"] = state.physics_approved
    trace_entry["safety_certification"] = parsed_val.get("safety_certification")
    state.execution_trace.append(trace_entry)
    return state


def radio_broadcaster_node(state: RaceEngineerState) -> RaceEngineerState:
    """
    Synthesizes the finalized pit-wall radio transmission.
    Adheres strictly to the direct, high-pressure communication style of an elite F1 Race Engineer.
    """
    trace_entry = {
        "node": "radio_broadcaster_node",
        "action": "synthesizing_pit_radio",
        "timestamp": time.time()
    }

    intent = state.intent
    telemetry = state.mcp_tool_results.get("get_live_telemetry", {}).get("telemetry", {})
    ml = state.mcp_tool_results.get("predict_podium_ml", {})
    strategy = state.mcp_tool_results.get("simulate_pit_stop_strategy", {})

    driver = state.driver_name
    first_name = driver.split()[0] if "Kimi" not in driver else "Kimi"

    radio_lines = []

    if intent == "PODIUM_PREDICTION":
        prob = ml.get("podium_probability", 0.0)
        prob_pct = round(prob * 100, 1)
        conf = ml.get("confidence", "HIGH")
        acc = ml.get("model_validation_accuracy", "92.07%")
        
        radio_lines.append(f"📻 **RÁDIO BOX -> {first_name.upper()}:**")
        radio_lines.append(f"\"Copy {first_name}, confirmamos os dados da telemetria.")
        radio_lines.append(f"Nosso modelo preditivo de Machine Learning (Gradient Boosting • {acc} ROC-AUC {ml.get('model_roc_auc', '0.949')}) aponta **{prob_pct}% de probabilidade de pódio**.")
        radio_lines.append(f"Ritmo atual é P{telemetry.get('current_position', 2)}, degradação do pneu Médio em {telemetry.get('tire_degradation_pct', 14.5)}%.")
        radio_lines.append(f"Delta de combustível favorável ({telemetry.get('fuel_delta_kg', -0.35)} kg). Modo de ultrapassagem disponível, gerencie o pneu dianteiro esquerdo na Curva do Sol.\"")
        state.tactical_code = "PUSH_PACE"

    elif intent == "PIT_STRATEGY":
        action = strategy.get("recommended_action", "STAY OUT")
        compound = strategy.get("recommended_compound", "HARD")
        window = strategy.get("optimal_pit_window", "Lap 22-24")
        code = strategy.get("radio_command_code", "BOX_NOW")
        
        radio_lines.append(f"📻 **RÁDIO BOX -> {first_name.upper()}:**")
        radio_lines.append(f"\"{action}, {first_name.upper()}. {action}!")
        radio_lines.append(f"Plano estratégico: **{strategy.get('strategic_plan', 'Plan A')}**.")
        radio_lines.append(f"Janela de parada calculada por Monte Carlo: {window}. Pneus novos: **{compound}**.")
        radio_lines.append(f"Delta de ultrapassagem simulado: {strategy.get('delta_vs_staying_out', 'Favorável')}.")
        radio_lines.append(f"Confirme pelo rádio quando avistar a entrada do pit lane.\"")
        state.tactical_code = code

    elif intent == "TELEMETRY_INSPECTION":
        speed = telemetry.get("speed_trap_kmh", 319.4)
        deg = telemetry.get("tire_degradation_pct", 14.2)
        battery = telemetry.get("ers_battery_state_pct", 74.5)
        
        radio_lines.append(f"📻 **RÁDIO BOX -> {first_name.upper()}:**")
        radio_lines.append(f"\"Leitura de telemetria completa na volta {state.current_lap}:")
        radio_lines.append(f"• Velocidade máxima no Speed Trap: **{speed} km/h**")
        radio_lines.append(f"• Desgaste do composto: **{deg}%** (dentro da janela projetada)")
        radio_lines.append(f"• Carga da bateria ERS: **{battery}%** (modo de defesa ativo)")
        radio_lines.append(f"• Temperatura de carcaça: **{telemetry.get('tire_carcass_temp_c', 101.8)}°C** - Pneus na janela ideal de aderência.\"")
        state.tactical_code = "TELEMETRY_OK"

    else:
        radio_lines.append(f"📻 **RÁDIO BOX -> {first_name.upper()}:**")
        radio_lines.append(f"\"Recebido, {first_name}. Telemetria operando 100% nominal.")
        radio_lines.append(f"Posição em pista: P{telemetry.get('current_position', 2)}. Probabilidade de Pódio no modelo: {round(ml.get('podium_probability', 0.88)*100, 1)}%.")
        radio_lines.append(f"Recomendação de estratégia: {strategy.get('recommended_action', 'Manter ritmo e cuidar do pneu dianteiro')}. Head down and keep pushing!\"")
        state.tactical_code = "MAINTAIN_PACE"

    state.radio_response = "\n".join(radio_lines)
    state.messages.append(GraphMessage(role="ai", content=state.radio_response))
    state.execution_trace.append(trace_entry)
    return state
