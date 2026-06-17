# Manages multiple async context managers and closes them together on exit
from contextlib import AsyncExitStack
# Generic type hint for values of any type 
from typing import Any
# MCP client session used to communicate with an MCP server
from mcp import ClientSession 

class MCPClient:
    def __init__(self, name: str, command: str, server_args: list[str], env_vars: dict[str, str]=None) -> None:
        self.name = name
        self.command = command
        self.server_args = server_args
        self.env_vars = env_vars or {}
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._connected: bool = False
        '''
        name -> just gives a human-readable name to the instantiated client.This is useful for logging, especially if you are maintaining connections to multiple servers
        command -> the command to execute on the server. This is the main instruction that tells the server what action to perform.
        server_args -> a list of additional arguments that may be required by the command. These could be parameters or options that modify the behavior of the command.
        env_vars -> a dictionary of environment variables that may be needed for the command execution. These variables can provide necessary context or configuration for the
        _session -> this will hold the active ClientSession once a connection is established. It starts as None and is set when the connect method is called.
        _exit_stack -> an AsyncExitStack is used to manage multiple asynchronous context managers. This allows for clean and efficient resource management, ensuring that all resources are properly released when the client is done.
        _connected -> a boolean flag to track whether the client is currently connected to the server. This can be useful for preventing multiple connections or for handling connection state in the application logic.
        '''