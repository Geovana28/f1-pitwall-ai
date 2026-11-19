"""
Model Context Protocol (MCP) - Core Protocol Definitions
Implements the JSON-RPC 2.0 specification for the MCP v1.0 Standard.
Compatible with Claude Desktop, Cursor, Antigravity, and custom LLM host agents.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MCPJsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int]
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPJsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int]
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class ToolParameterProperty(BaseModel):
    type: str
    description: str
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None


class ToolInputSchema(BaseModel):
    type: str = "object"
    properties: Dict[str, ToolParameterProperty]
    required: List[str] = Field(default_factory=list)


class MCPToolDefinition(BaseModel):
    name: str
    description: str
    inputSchema: ToolInputSchema


class MCPToolCallResult(BaseModel):
    content: List[Dict[str, Any]]
    isError: bool = False
