from mcp import ClientSession
from contextlib import AsyncExitStack
from mcp.client.stdio import StdioServerParameters, stdio_client
from typing import Any
import logging

class MCPClinet:
    def __init__(self, name: str, command: str, server_args: list[str], env_vars: dict[str, str]=None) -> None:
        self.name = name
        self.command = command
        self.server_args = server_args
        self.env_vars = env_vars or {}
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._connected: bool = False
    async def connect(self) -> None:
        if self._connected:
            raise RuntimeError(f"{self.name} is already connected.")
        
        server_parameters = StdioServerParameters(
            command=self.command,
            args=self.server_args,
            env=self.env_vars if self.env_vars else None
        )

        # Connect to stdio server, starting subprocess
        stdio_connection = await self._exit_stack.enter_async_context(stdio_client(server_parameters))
        self.read, self.write = stdio_connection

        # Start MCP client session
        self._session = await self._exit_stack.enter_async_context(ClientSession(read_stream=self.read, write_stream=self.write))

        # Initialize session
        await self._session.initialize()
        self._connected = True
    async def disconnect(self) -> None:
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._connected = False
            self._session = None
    async def get_available_tools(self) -> list[dict[str, Any]]:
         """"
         Fetches the list of available tools from the server and returns them as a list of dictionaries containing tool information.
         Each dictionary in the returned list contains the following
         - name: The name of the tool.
         - description: A brief description of what the tool does.
         - input_schema: The schema that defines the expected input for the tool, which can be used to understand how to interact with the tool and what kind of data it requires.
         - annotations: Any additional metadata or annotations associated with the tool, which can provide further context or information about the tool's functionality or usage.
         - model_config: Configuration details related to the model that the tool uses, which can include parameters, settings, or other relevant information that helps in understanding how the tool operates and how it can be utilized effectively.
         """
         if not self._connected:
            raise RuntimeError("Client not connected to a server")
         
         # Request the list of tools from the server using the MCP session. The result is expected to contain a list of tools, each with its own name, description, and input schema, annotations, and model configuration.
         tools_result = await self._session.list_tools()
         if tools_result.tools is None:
            logging.warning("No tools available from the server.")
            return []
         available_tools = [
             {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                    "annotations": tool.annotations,
                    "model_config": tool.model_config
             }
             for tool in tools_result.tools
         ]
         return available_tools

         