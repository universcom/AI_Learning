# Manages multiple async context managers and closes them together on exit
from contextlib import AsyncExitStack
# Generic type hint for values of any type 
from typing import Any
# MCP client session used to communicate with an MCP server
from mcp import ClientSession 
# Factory function to create a client session that communicates with the server using standard input and output streams.
from mcp.client.stdio import StdioServerParameters, stdio_client

class MCPClient:
    def __init__(self, name: str, command: str, server_args: list[str], env_vars: dict[str, str]=None) -> None:
        '''
        name -> just gives a human-readable name to the instantiated client.This is useful for logging, especially if you are maintaining connections to multiple servers
        command -> the command to execute on the server. This is the main instruction that tells the server what action to perform.
        server_args -> a list of additional arguments that may be required by the command. These could be parameters or options that modify the behavior of the command.
        env_vars -> a dictionary of environment variables that may be needed for the command execution. These variables can provide necessary context or configuration for the
        _session -> this will hold the active ClientSession once a connection is established. It starts as None and is set when the connect method is called.
        _exit_stack -> an AsyncExitStack is used to manage multiple asynchronous context managers. This allows for clean and efficient resource management, ensuring that all resources are properly released when the client is done.
        _connected -> a boolean flag to track whether the client is currently connected to the server. This can be useful for preventing multiple connections or for handling connection state in the application logic.
        '''
        self.name = name
        self.command = command
        self.server_args = server_args
        self.env_vars = env_vars or {}
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._connected: bool = False
    async def connect(self) -> None:
        """
        Connect to the server set in the constructor.
        server_parameters -> an instance of StdioServerParameters is created using the command, server_args, and env_vars provided in the constructor. This object encapsulates all the necessary information to start the server process and establish communication.
        stdio_connection -> the stdio_client function is called with the server_parameters to establish a connection to the server. This function likely starts the server process and sets up communication channels (standard input and output streams).
        read, write -> the stdio_connection returns two streams: one for reading from the server and one for writing to the server. These streams are stored in the instance variables self.read and self.write for later use in communication with the server.
        _session -> a ClientSession is created using the read and write streams. This session will manage the communication protocol with the server, allowing for sending requests and receiving responses in a structured manner.
        initialize -> the session is initialized, which may involve sending an initial handshake or setup messages to the server to establish the communication protocol. Once this is complete
        """
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
        """
        Clean up any resources and close the connection to the server. 
        This method ensures that all resources are properly released and that the client is no longer connected to the server. 
        It checks if there is an active exit stack (which manages the resources) and if so, it closes it asynchronously. 
        After closing the exit stack, it updates the connection status and clears the session information.
        """
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._connected = False
            self._session = None
        