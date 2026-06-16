class MCPclient:
    def __init__(self) -> None:
        pass

    async def connect(self) -> None: 
        """
        Connect to the MCP server.
        """
    async def get_available_tools(self) -> list[Any]:
        """
        Get a list of available tools from the MCP server.
        """
    async def execute_tool(self, tool_name: str, *args: Any) -> Any:
        """
        Execute a tool on the MCP server with the given arguments.
        """
    async def disconnect(self) -> None:
        """
        Disconnect from the MCP server. Clean up any resources if necessary.
        """