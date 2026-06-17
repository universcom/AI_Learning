class MCPclient:
    def __init__(self) -> None:
        pass

    async def connect(self) -> None: 
        """
        Connect to the MCP server.
        This will initialize your connection to the server. 
        The implementation will look different based on what you choose for your transport layer
        """
    async def get_available_tools(self) -> list[Any]:
        """
        Get a list of available tools from the MCP server.
        This will use the client’s server connection to retrieve the tools that the server makes available
        """
    async def execute_tool(self, tool_name: str, *args: Any) -> Any:
        """
        Execute a tool on the MCP server with the given arguments.
        This will use the client’s server connection to retrieve the tools that the server makes available
        """
    async def disconnect(self) -> None:
        """
        Disconnect from the MCP server. Clean up any resources if necessary.
        """