"""
Model Context Protocol (MCP) Server for F1 Autonomous Race Engineering.
Compliant with MCP Specification v1.0 (Anthropic / Open Protocol).
Provides tools for live telemetry, Scikit-Learn ML inference, and Monte Carlo pit stops.
"""

import sys
import json
import logging
from typing import Any, Dict, List
from .protocol import (
    MCPJsonRpcRequest,
    MCPJsonRpcResponse,
    MCPToolDefinition,
    ToolInputSchema,
    ToolParameterProperty
)
from .tools import (
    get_live_telemetry,
    predict_podium_ml,
    simulate_pit_stop_strategy,
    validate_telemetry_physics
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("F1-MCP-Server")


class F1RaceEngineerMCPServer:
    """
    Official MCP Server exposing telemetry and race strategy tools to LangChain/LangGraph and Claude/Cursor.
    """

    def __init__(self):
        self.tools: Dict[str, MCPToolDefinition] = self._register_tools()

    def _register_tools(self) -> Dict[str, MCPToolDefinition]:
        return {
            "get_live_telemetry": MCPToolDefinition(
                name="get_live_telemetry",
                description="Queries live telemetry parameters (speed trap, tire degradation, battery state, fuel delta) for a given F1 driver and circuit.",
                inputSchema=ToolInputSchema(
                    properties={
                        "driver": ToolParameterProperty(type="string", description="Driver name (e.g., 'Andrea Kimi Antonelli', 'Max Verstappen')"),
                        "circuit": ToolParameterProperty(type="string", description="Grand Prix Circuit (e.g., 'Interlagos', 'Monza', 'Silverstone')")
                    },
                    required=["driver", "circuit"]
                )
            ),
            "predict_podium_ml": MCPToolDefinition(
                name="predict_podium_ml",
                description="Executes ML inference using trained GradientBoostingClassifier (92.07% accuracy, ROC-AUC 0.949) to compute probability of finishing on the podium.",
                inputSchema=ToolInputSchema(
                    properties={
                        "driver": ToolParameterProperty(type="string", description="Driver name"),
                        "circuit": ToolParameterProperty(type="string", description="Race track name"),
                        "grid_position": ToolParameterProperty(type="integer", description="Starting grid position (1 to 20)"),
                        "tyre_compound": ToolParameterProperty(type="string", description="Current tyre compound (SOFT, MEDIUM, HARD, INTERMEDIATE)"),
                        "current_lap": ToolParameterProperty(type="integer", description="Current race lap"),
                        "total_laps": ToolParameterProperty(type="integer", description="Total race laps"),
                        "weather": ToolParameterProperty(type="string", description="Weather condition ('DRY', 'WET', 'RAIN')")
                    },
                    required=["driver", "circuit", "grid_position"]
                )
            ),
            "simulate_pit_stop_strategy": MCPToolDefinition(
                name="simulate_pit_stop_strategy",
                description="Runs a 10,000-run Monte Carlo simulation to evaluate undercut/overcut deltas, tire cliff degradation, and optimum pit stop window.",
                inputSchema=ToolInputSchema(
                    properties={
                        "current_compound": ToolParameterProperty(type="string", description="Active tyre compound (e.g. MEDIUM)"),
                        "laps_on_tyre": ToolParameterProperty(type="integer", description="Number of laps already completed on current set"),
                        "weather_forecast": ToolParameterProperty(type="string", description="Radar forecast (e.g. 'Dry' or 'Rain in 5 mins')"),
                        "safety_car_prob": ToolParameterProperty(type="number", description="Safety car probability factor between 0.0 and 1.0")
                    },
                    required=["current_compound", "laps_on_tyre", "weather_forecast"]
                )
            ),
            "validate_telemetry_physics": MCPToolDefinition(
                name="validate_telemetry_physics",
                description="Autonomous safety check: Verifies that tactical recommendations do not exceed FIA 2026 fuel limits (100 kg/h) or tire thermal thresholds.",
                inputSchema=ToolInputSchema(
                    properties={
                        "delta_time": ToolParameterProperty(type="number", description="Projected delta time per lap in seconds"),
                        "tire_temp_c": ToolParameterProperty(type="number", description="Core tire temperature in degrees Celsius"),
                        "fuel_flow_kg_h": ToolParameterProperty(type="number", description="Instantaneous fuel flow rate in kg/h")
                    },
                    required=["delta_time", "tire_temp_c", "fuel_flow_kg_h"]
                )
            )
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns MCP tools manifest for the agent."""
        return [tool.model_dump() for tool in self.tools.values()]

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches an MCP tool call to the respective backend function."""
        if tool_name not in self.tools:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Tool '{tool_name}' not found on F1-MCP-Server."}]
            }

        try:
            if tool_name == "get_live_telemetry":
                res = get_live_telemetry(
                    driver=arguments.get("driver", "Andrea Kimi Antonelli"),
                    circuit=arguments.get("circuit", "Interlagos")
                )
            elif tool_name == "predict_podium_ml":
                res = predict_podium_ml(
                    driver=arguments.get("driver", "Andrea Kimi Antonelli"),
                    circuit=arguments.get("circuit", "Interlagos"),
                    grid_position=int(arguments.get("grid_position", 2)),
                    tyre_compound=arguments.get("tyre_compound", "MEDIUM"),
                    current_lap=int(arguments.get("current_lap", 20)),
                    total_laps=int(arguments.get("total_laps", 71)),
                    weather=arguments.get("weather", "DRY")
                )
            elif tool_name == "simulate_pit_stop_strategy":
                res = simulate_pit_stop_strategy(
                    current_compound=arguments.get("current_compound", "MEDIUM"),
                    laps_on_tyre=int(arguments.get("laps_on_tyre", 18)),
                    weather_forecast=arguments.get("weather_forecast", "DRY"),
                    safety_car_prob=float(arguments.get("safety_car_prob", 0.25))
                )
            elif tool_name == "validate_telemetry_physics":
                res = validate_telemetry_physics(
                    delta_time=float(arguments.get("delta_time", -0.4)),
                    tire_temp_c=float(arguments.get("tire_temp_c", 104.2)),
                    fuel_flow_kg_h=float(arguments.get("fuel_flow_kg_h", 94.5))
                )
            else:
                raise ValueError(f"Unhandled tool: {tool_name}")

            return {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(res, indent=2)}]
            }

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Error: {str(e)}"}]
            }

    def handle_jsonrpc(self, request_raw: str) -> str:
        """Processes JSON-RPC 2.0 requests per MCP specification."""
        try:
            payload = json.loads(request_raw)
            req = MCPJsonRpcRequest(**payload)
        except Exception as err:
            err_resp = MCPJsonRpcResponse(
                id=None,
                error={"code": -32700, "message": f"Parse error: {err}"}
            )
            return err_resp.model_dump_json()

        if req.method == "tools/list":
            response = MCPJsonRpcResponse(id=req.id, result={"tools": self.list_tools()})
        elif req.method == "tools/call":
            params = req.params or {}
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = self.call_tool(tool_name, arguments)
            response = MCPJsonRpcResponse(id=req.id, result=result)
        else:
            response = MCPJsonRpcResponse(
                id=req.id,
                error={"code": -32601, "message": f"Method '{req.method}' not found"}
            )

        return response.model_dump_json()


def run_stdio_server():
    """Runs standard stdio MCP loop for Claude Desktop / Cursor / Antigravity integration."""
    server = F1RaceEngineerMCPServer()
    logger.info("F1 Autonomous Race Engineer MCP Server running on stdio.")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        reply = server.handle_jsonrpc(line)
        sys.stdout.write(reply + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
