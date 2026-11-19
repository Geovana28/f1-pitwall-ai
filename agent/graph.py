"""
F1 Autonomous Race Engineer - LangGraph StateGraph Architecture.
Implements the multi-stage cyclic Agent Graph:
1. Router (Intent extraction & tool scheduling)
2. MCP Tool Execution (Queries Model Context Protocol server)
3. Safety & Physics Reflection (Evaluates bounds & checks for violations)
   -> If violation: cycles back to Strategy Re-Adjustment
   -> If approved: proceeds to Radio Broadcaster
4. Radio Broadcaster (Synthesizes race radio message)
"""

from typing import Dict, Any, Callable, List
from .state import RaceEngineerState, GraphMessage
from .nodes import (
    router_node,
    mcp_tool_executor_node,
    safety_reflection_node,
    radio_broadcaster_node
)
from mcp_server.server import F1RaceEngineerMCPServer


class F1RaceEngineerGraph:
    """
    Orchestrates the LangGraph agent execution.
    Exposes graph visualization, conditional transitions, and step-by-step tracing.
    """

    def __init__(self, mcp_server: F1RaceEngineerMCPServer = None):
        self.mcp_server = mcp_server or F1RaceEngineerMCPServer()

    def _should_replan(self, state: RaceEngineerState) -> str:
        """
        Conditional edge in LangGraph:
        Evaluates whether physical boundary conditions were met.
        If violated and attempts < 2, triggers re-calculation loop.
        """
        if not state.physics_approved and state.critique_iterations < 2:
            return "replan"
        return "approved"

    def run(self, user_query: str, driver: str = "Andrea Kimi Antonelli", circuit: str = "Interlagos", lap: int = 22, grid: int = 2) -> RaceEngineerState:
        """
        Executes the compiled LangGraph workflow.
        """
        # Initialize State
        state = RaceEngineerState(
            driver_name=driver,
            circuit=circuit,
            current_lap=lap,
            grid_position=grid,
            user_query=user_query,
            messages=[GraphMessage(role="human", content=user_query)]
        )

        # Node 1: Router & Intent Node
        state = router_node(state)

        # Node 2: MCP Tool Execution Node
        state = mcp_tool_executor_node(state, self.mcp_server)

        # Node 3: Safety & Reflection Critique Node
        state = safety_reflection_node(state, self.mcp_server)

        # Conditional Edge Check
        decision = self._should_replan(state)
        if decision == "replan":
            # Corrective sub-loop: force conservative strategy
            state.tool_arguments["simulate_pit_stop_strategy"]["weather_forecast"] = "Conservative safety window"
            state = mcp_tool_executor_node(state, self.mcp_server)
            state = safety_reflection_node(state, self.mcp_server)

        # Node 4: Final Radio Broadcaster Node
        state = radio_broadcaster_node(state)

        return state

    @staticmethod
    def get_mermaid_graph() -> str:
        """Returns Mermaid Diagram representation of the LangGraph agent."""
        return """
flowchart TD
    Start(["🏎️ Início: Mensagem de Rádio do Piloto"]) --> Router["🧭 Nó 1: Router & Extrator de Intenção"]
    Router --> Tools["⚡ Nó 2: MCP Server Execution (Telemetria, ML, Monte Carlo)"]
    Tools --> Critique{"🛡️ Nó 3: Safety Reflection & Validador Físico"}
    Critique -- "⚠️ Violação Térmica / FIA" --> Replan["🔄 Nó Re-planejamento Tático"]
    Replan --> Tools
    Critique -- "✅ Aprovado (Limites FIA)" --> Broadcast["📻 Nó 4: Síntese de Rádio do Box & Código Tático"]
    Broadcast --> End(["🏁 Resposta Transmitida ao Piloto"])
"""
