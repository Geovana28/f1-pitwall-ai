"""
State definitions for the F1 Autonomous Race Engineer LangGraph.
Follows the StateGraph pattern used in LangGraph / LangChain.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GraphMessage(BaseModel):
    role: str  # 'human', 'ai', 'tool', 'system'
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RaceEngineerState(BaseModel):
    """
    Central State passed between all nodes in the LangGraph StateGraph.
    """
    driver_name: str = "Andrea Kimi Antonelli"
    circuit: str = "Interlagos"
    current_lap: int = 22
    grid_position: int = 2
    user_query: str = ""
    messages: List[GraphMessage] = Field(default_factory=list)
    
    # Routing & Planning Node outputs
    intent: Optional[str] = None
    required_tools: List[str] = Field(default_factory=list)
    tool_arguments: Dict[str, Any] = Field(default_factory=dict)

    # Tool Execution Node outputs (via MCP Server)
    mcp_tool_results: Dict[str, Any] = Field(default_factory=dict)

    # Safety & Reflection Node outputs
    physics_approved: bool = True
    critique_iterations: int = 0
    safety_notes: List[str] = Field(default_factory=list)

    # Final Output Node
    radio_response: str = ""
    tactical_code: str = "COPY"
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
