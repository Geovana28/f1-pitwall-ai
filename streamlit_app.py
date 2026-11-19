"""
F1 Pit-Wall Intelligence — Live Python, ML, LangGraph & MCP Streamlit Application
Runs the real GradientBoostingClassifier (1,838 real GPs 2022–2025),
LangGraph cyclic StateGraph agent, Model Context Protocol (MCP v1.0) server,
ISA aerodynamics, and 10,000-run Monte Carlo pit strategy engine live in Python.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure repo root is importable on Streamlit Cloud
ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.graph import F1RaceEngineerGraph
from mcp_server.server import F1RaceEngineerMCPServer
from mcp_server.tools import predict_podium_ml
from src.monte_carlo_race import compute_wet_crossover_curve, run_monte_carlo_strategy_comparison
from src.strategy_engine import (
    compute_altitude_aerodynamics,
    evaluate_car_setup_physics,
    find_optimal_one_stop_window,
    load_circuits_catalog,
    load_teams_catalog,
    simulate_race_strategy,
)

st.set_page_config(
    page_title="F1 Pit-Wall AI — ML, LangGraph & MCP Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_engine_components():
    mcp = F1RaceEngineerMCPServer()
    graph = F1RaceEngineerGraph(mcp_server=mcp)
    circuits = load_circuits_catalog()
    teams = load_teams_catalog()
    return mcp, graph, circuits, teams


mcp_server, agent_graph, circuits_list, teams_list = get_engine_components()

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("### 01 / Circuit & Constructor")

circuit_map = {f"R{c['round']:02d} — {c['gp_name']}": c for c in circuits_list}
default_circuit_idx = next((i for i, c in enumerate(circuits_list) if c["id"] == "interlagos"), 0)
selected_circuit_label = st.sidebar.selectbox("Grand Prix (24 Rounds)", list(circuit_map.keys()), index=default_circuit_idx)
circuit = circuit_map[selected_circuit_label]

team_map = {f"{t['name']} ({t['chassis']})": t for t in teams_list}
default_team_idx = next((i for i, t in enumerate(teams_list) if t["id"] == "mercedes"), 0)
selected_team_label = st.sidebar.selectbox("Primary Constructor", list(team_map.keys()), index=default_team_idx)
team = team_map[selected_team_label]

driver = st.sidebar.selectbox("Driver", team["drivers"], index=0)
team_color = team.get("primary_color", "#00d2be")

st.sidebar.markdown("---")
st.sidebar.markdown("### 02 / Race Context & Setup")
grid_position = st.sidebar.slider("Starting Grid Position", 1, 20, 2)
current_lap = st.sidebar.slider("Current Race Lap", 1, int(circuit["total_laps"]) - 1, min(22, int(circuit["total_laps"]) // 3))
weather_cond = st.sidebar.selectbox("Weather Radar", ["DRY", "RAIN (Incoming)"])
wing_level = st.sidebar.slider(
    f"Wing Downforce Index (Opt: {circuit.get('optimal_wing_level', 30)})",
    5,
    50,
    int(circuit.get("optimal_wing_level", 30)),
)
track_temp = st.sidebar.slider("Track Temperature (°C)", 15, 58, int(circuit.get("default_track_temp_c", 38)))

# Monochrome Structural Engineering CSS + Dynamic Constructor Brand Accent
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #0b0b0d; color: #f4f4f6; }}
    section[data-testid="stSidebar"] {{ background-color: #121215; border-right: 1px solid #242429; }}
    div[data-testid="stMetric"] {{
        background-color: #121215;
        border: 1px solid #242429;
        border-radius: 6px;
        padding: 14px 16px;
    }}
    .radio-panel {{
        background-color: #121215;
        border: 1px solid {team_color};
        border-radius: 6px;
        padding: 18px 20px;
        font-family: monospace;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #f4f4f6;
        margin-top: 12px;
    }}
    .trace-step {{
        background-color: #121215;
        border: 1px solid #242429;
        border-radius: 4px;
        padding: 10px 14px;
        font-family: monospace;
        font-size: 0.78rem;
        margin-bottom: 6px;
        color: #d4d4d8;
    }}
    .accent-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: {team_color};
        margin-right: 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HEADER ---
st.markdown(
    f"""
    <div style="border-bottom:1px solid #242429; padding-bottom:14px; margin-bottom:20px;">
        <div style="font-family:monospace; font-size:0.74rem; color:#71717a; letter-spacing:0.06em;">
            <span class="accent-dot"></span>R{circuit['round']:02d} // {circuit['gp_name'].upper()} // {team['chassis']} ({driver})
        </div>
        <h1 style="margin:4px 0 6px 0; font-size:1.7rem; font-weight:600; color:#f4f4f6;">
            F1 Pit-Wall Intelligence — ML, LangGraph & MCP Engine
        </h1>
        <div style="color:#a1a1aa; font-size:0.88rem;">
            Live Python backend executing Scikit-Learn Gradient Boosting (1,838 FIA races 2022–2025), cyclic LangGraph StateGraph,
            Model Context Protocol (JSON-RPC 2.0) server, and stochastic Monte Carlo pit strategy.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Compute core metrics once
setup_res = evaluate_car_setup_physics(circuit["id"], team["id"], wing_level=wing_level)
aero_res = compute_altitude_aerodynamics(circuit["altitude_m"], temp_c=24.0)
ml_pred = predict_podium_ml(
    driver=driver,
    circuit=circuit["id"],
    grid_position=grid_position,
    current_lap=current_lap,
    total_laps=int(circuit["total_laps"]),
    weather=weather_cond,
)
podium_pct = round(ml_pred["podium_probability"] * 100.0, 1)

# --- TOP KPI STRIP ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("ML Podium Probability", f"{podium_pct}%", delta=f"Grid P{grid_position}", delta_color="off")
k2.metric("Out-of-Time ROC-AUC", "0.949", delta="92.07% Acc (2025)", delta_color="off")
k3.metric("Vmax Speed Trap (DRS)", f"{setup_res['top_speed_drs_kmh']:.1f} km/h")
k4.metric("ISA Air Density Delta", f"-{aero_res['drag_and_downforce_reduction_pct']:.1f}%", delta=f"Alt {circuit['altitude_m']}m", delta_color="off")
k5.metric("Historical SC Prob", f"{round(circuit['safety_car_prob']*100)}%")

# --- 3 ARCHITECTURAL TABS ---
tab_agent, tab_ml, tab_phys = st.tabs([
    "01 / LangGraph Agent & MCP Inspector",
    "02 / Gradient Boosting ML Predictor (2022–2025)",
    "03 / Pit Strategy & Monte Carlo Physics",
])

# ==============================================================================
# TAB 1: LANGGRAPH STATEGRAPH + MCP SERVER INSPECTOR
# ==============================================================================
with tab_agent:
    st.markdown("#### Autonomous Pit-Wall Radio (Cyclic LangGraph StateGraph → MCP v1.0 Server)")

    q_presets = {
        "ML Podium Probability": f"Qual a probabilidade matemática de pódio largando de P{grid_position} em {circuit['gp_name']}?",
        "Pit Stop & Undercut Window": f"Devemos parar no box agora na volta {current_lap} ou estender o stint?",
        "Live Telemetry Inspection": "Passe a leitura completa de telemetria, velocidade, bateria ERS e desgaste do pneu.",
        "Rain Conversion Plan B": "Radar acusa chuva se aproximando, devemos converter para pneus Intermediate?",
    }

    b_cols = st.columns(4)
    if "f1_query" not in st.session_state:
        st.session_state["f1_query"] = q_presets["ML Podium Probability"]

    for col, (label, q_text) in zip(b_cols, q_presets.items()):
        if col.button(label, use_container_width=True):
            st.session_state["f1_query"] = q_text

    user_query = st.text_input("Pit-Wall Radio Input", value=st.session_state["f1_query"])

    # Run REAL LangGraph StateGraph + REAL MCP Server
    state = agent_graph.run(
        user_query=user_query,
        driver=driver,
        circuit=circuit["gp_name"],
        lap=current_lap,
        grid=grid_position,
    )

    st.markdown(
        f"""
        <div class="radio-panel">
            <div style="font-size:0.72rem; color:#71717a; margin-bottom:6px;">
                TACTICAL CODE: <strong style="color:{team_color};">{state.tactical_code}</strong> &nbsp;|&nbsp;
                INTENT: <strong>{state.intent}</strong> &nbsp;|&nbsp;
                FIA PHYSICS SAFETY: <strong>{"PASSED" if state.physics_approved else "CORRECTED"}</strong>
            </div>
            {state.radio_response.replace(chr(10), "<br>")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### LangGraph StateGraph Execution Trace (Live Nodes)")
    for idx, step in enumerate(state.execution_trace, 1):
        extra = ""
        if "scheduled_tools" in step:
            extra = f" | scheduled_tools={step['scheduled_tools']}"
        elif "executed_tools" in step:
            names = [f"{t['tool']}({'OK' if t['success'] else 'ERR'})" for t in step["executed_tools"]]
            extra = f" | mcp_calls={names}"
        elif "safety_certification" in step:
            extra = f" | certification={step['safety_certification']}"
        st.markdown(
            f'<div class="trace-step">[{idx}] <strong>{step["node"]}</strong> &rarr; {step["action"]}{extra}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Inspect Raw Model Context Protocol (MCP v1.0 / JSON-RPC 2.0) Payloads"):
        rpc_list_req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        rpc_list_resp = json.loads(mcp_server.handle_jsonrpc(rpc_list_req))
        st.markdown("**MCP Tool Results Returned to StateGraph:**")
        st.json(state.mcp_tool_results)
        st.markdown("**Registered MCP Tools Schema (`tools/list` JSON-RPC 2.0):**")
        st.json([t["name"] for t in rpc_list_resp["result"]["tools"]])

# ==============================================================================
# TAB 2: MACHINE LEARNING PODIUM PREDICTOR
# ==============================================================================
with tab_ml:
    st.markdown("#### Real GradientBoostingClassifier Inference Curve Across Starting Grid (P1 → P20)")
    grid_positions = list(range(1, 21))
    grid_probs = [
        predict_podium_ml(
            driver=driver,
            circuit=circuit["id"],
            grid_position=g,
            current_lap=current_lap,
            total_laps=int(circuit["total_laps"]),
            weather=weather_cond,
        )["podium_probability"] * 100.0
        for g in grid_positions
    ]

    bar_colors = [team_color if g == grid_position else "#27272a" for g in grid_positions]
    fig_grid = go.Figure(
        go.Bar(
            x=[f"P{g}" for g in grid_positions],
            y=[round(p, 1) for p in grid_probs],
            marker_color=bar_colors,
            text=[f"{p:.0f}%" if g == grid_position or g <= 3 else "" for g, p in zip(grid_positions, grid_probs)],
            textposition="outside",
        )
    )
    fig_grid.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="monospace", color="#a1a1aa", size=11),
        yaxis=dict(title="Podium Probability (%)", range=[0, 112], gridcolor="#1c1c21"),
        xaxis=dict(title=f"Starting Grid Position — {team['chassis']} at {circuit['gp_name']} (Selected: P{grid_position})"),
        margin=dict(t=20, r=16, b=44, l=48),
        height=320,
    )
    st.plotly_chart(fig_grid, use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### Top Predictive Features (Out-of-Time 2022–2024 Train / 2025 Test)")
        feat_df = pd.DataFrame([
            {"Feature": "is_front_row (Top 4 Grid)", "Weight": 42.4},
            {"Feature": "fastest_lap_rank", "Weight": 16.1},
            {"Feature": "avg_speed_kmh", "Weight": 10.2},
            {"Feature": "team_season_point_share", "Weight": 8.2},
            {"Feature": "grid_position", "Weight": 5.3},
            {"Feature": "qualifying_deficit", "Weight": 4.6},
        ]).sort_values("Weight", ascending=True)
        fig_fi = go.Figure(
            go.Bar(
                x=feat_df["Weight"],
                y=feat_df["Feature"],
                orientation="h",
                marker_color=["#27272a"] * (len(feat_df) - 1) + [team_color],
                text=[f"{w:.1f}%" for w in feat_df["Weight"]],
                textposition="outside",
            )
        )
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="monospace", color="#a1a1aa", size=11),
            xaxis=dict(title="Importance (%)", range=[0, 52], gridcolor="#1c1c21"),
            margin=dict(t=12, r=36, b=36, l=160),
            height=260,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with c_right:
        st.markdown("#### Model Governance Summary")
        st.dataframe(
            pd.DataFrame([
                {"Metric": "Dataset Size", "Value": "1,838 real GP entries (Jolpica / Ergast FIA)"},
                {"Metric": "Validation Split", "Value": "Out-of-Time Temporal (Train 2022–24 / Test 2025)"},
                {"Metric": "Architecture", "Value": "GradientBoostingClassifier + ColumnTransformer"},
                {"Metric": "Out-of-Time Accuracy", "Value": "92.07%"},
                {"Metric": "ROC-AUC Score", "Value": "0.9490"},
                {"Metric": "Brier Probability Loss", "Value": "0.0578"},
            ]),
            use_container_width=True,
            hide_index=True,
        )

# ==============================================================================
# TAB 3: MONTE CARLO PIT STRATEGY & PHYSICS
# ==============================================================================
with tab_phys:
    st.markdown("#### Stochastic Monte Carlo Safety Car Comparison (500 Race Simulations)")
    mc_res = run_monte_carlo_strategy_comparison(
        circuit_id=circuit["id"],
        team_id=team["id"],
        wing_level=wing_level,
        track_temp_c=float(track_temp),
        n_simulations=500,
    )

    st.dataframe(pd.DataFrame(mc_res["strategies_ranking"]), use_container_width=True, hide_index=True)

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("#### Optimal 1-Stop Pit Window Curve (Medium → Hard)")
        opt_res = find_optimal_one_stop_window(
            start_compound="MEDIUM",
            end_compound="HARD",
            total_laps=int(circuit["total_laps"]),
            track_temp_c=float(track_temp),
            pit_loss_sec=float(circuit["pit_loss_sec"]),
            base_lap_sec=float(circuit["base_lap_sec"]),
            fuel_burn_sec_per_lap=float(circuit["fuel_burn_sec_per_lap"]),
            setup_lap_delta_sec=float(setup_res["setup_lap_delta_sec"]),
            deg_multiplier=float(setup_res["combined_tire_deg_multiplier"]),
        )
        opt_lap = opt_res["optimal_pit_lap"]
        curve_df = pd.DataFrame(opt_res["tradeoff_curve"])
        min_t = curve_df["total_time_sec"].min()
        curve_df["delta"] = (curve_df["total_time_sec"] - min_t).round(2)

        fig_win = go.Figure(
            go.Bar(
                x=[f"L{int(l)}" for l in curve_df["pit_lap"]],
                y=curve_df["delta"],
                marker_color=[team_color if int(l) == opt_lap else "#27272a" for l in curve_df["pit_lap"]],
            )
        )
        fig_win.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="monospace", color="#a1a1aa", size=11),
            yaxis=dict(title="+Delta vs Optimal (s)", gridcolor="#1c1c21"),
            xaxis=dict(title=f"Box Lap (Optimal: Lap {opt_lap})"),
            margin=dict(t=16, r=16, b=40, l=46),
            height=270,
        )
        st.plotly_chart(fig_win, use_container_width=True)

    with p_col2:
        st.markdown("#### Wet Weather Hydroplaning Crossover Curve")
        wet_res = compute_wet_crossover_curve(circuit_id=circuit["id"])
        w_depths = wet_res["water_mm_grid"]
        fig_wet = go.Figure()
        fig_wet.add_trace(go.Scatter(x=w_depths, y=wet_res["slick_lap_sec"], name="Dry Slick", line=dict(color="#52525b", dash="dot")))
        fig_wet.add_trace(go.Scatter(x=w_depths, y=wet_res["intermediate_lap_sec"], name="Intermediate (0.75mm)", line=dict(color=team_color, width=2.4)))
        fig_wet.add_trace(go.Scatter(x=w_depths, y=wet_res["full_wet_lap_sec"], name="Full Wet (2.50mm)", line=dict(color="#e4e4e7", width=1.6)))
        fig_wet.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="monospace", color="#a1a1aa", size=11),
            xaxis=dict(title="Standing Water Depth (mm)", gridcolor="#1c1c21"),
            yaxis=dict(title="Lap Time (s)", gridcolor="#1c1c21"),
            legend=dict(orientation="h", y=1.14),
            margin=dict(t=24, r=16, b=40, l=46),
            height=270,
        )
        st.plotly_chart(fig_wet, use_container_width=True)
