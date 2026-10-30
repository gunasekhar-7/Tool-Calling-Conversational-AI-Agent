from agent.tools.orders import READ_ORDER_TOOLS
from agent.tools.registry import ToolSpec, to_langchain_tools, tool_map
from agent.tools.tickets import READ_TICKET_TOOLS
from agent.tools.tickets_write import WRITE_TICKET_TOOLS

ALL_TOOL_SPECS: list[ToolSpec] = [*READ_ORDER_TOOLS, *READ_TICKET_TOOLS, *WRITE_TICKET_TOOLS]

__all__ = [
    "ALL_TOOL_SPECS",
    "ToolSpec",
    "to_langchain_tools",
    "tool_map",
]
