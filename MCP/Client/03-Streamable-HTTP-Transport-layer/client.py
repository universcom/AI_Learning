# This code defines a class called MCPClient that is designed to connect to a server using the Streamable HTTP protocol.
from contextlib import AsyncExitStack # Manages multiple async context managers and closes them together on exit
from typing import Callable # Generic type hint for values of any type

from mcp import ClientSession # MCP client session used to communicate with an MCP server

from mcp.client.streamable_http import streamablehttp_client # Factory function to create a client session that communicates with the server using Streamable HTTP protocol. It takes a URL and optional headers to establish the connection and returns the necessary components for communication (read stream, write stream, and a callable to get the session ID).

class MCPClient:
    def __init__(self, name: str, server_url: str) -> None:
        """
        Initialize the MCPClient.

        name -> just gives a human-readable name to the instantiated client.This is useful for logging, especially if you are maintaining connections to multiple servers
        server_url -> the URL of the server to connect to. This is the address where the MCP server is running and will be used to establish a connection.
        _session -> this will hold the active ClientSession once a connection is established. It starts as None and is set when the connect method is called.
        exit_stack -> an AsyncExitStack is used to manage multiple asynchronous context managers. This allows for clean and efficient resource management, ensuring that all resources are properly released when the client is done.
        _connected -> a boolean flag to track whether the client is currently connected to the server. This can be useful for preventing multiple connections or for handling connection state in the application logic.
        _get_session_id -> a callable that returns a string, which is likely used to retrieve the session ID for the client session. This can be useful for logging, debugging, or managing multiple sessions.
        """
        self.name = name
        self.server_url = server_url
        self._session: ClientSession = None
        self.exit_stack = AsyncExitStack()
        self._connected: bool = False
        self._get_session_id: Callable[[], str] = None
    async def connect(self, headers:dict | None = None) -> None:
        """
        Connect to the server set in the constructor.
        headers -> an optional dictionary of headers that may be needed for the connection. These headers can provide necessary context or authentication information for establishing the connection with the server.
        streamable_connection -> the streamablehttp_client function is called with the server_url and headers to establish a connection to the Streamable HTTP server. This function likely sets up the necessary communication channels (such as WebSocket or HTTP streams) for interacting with the server.
        read, write, _get_session_id -> the streamable_connection returns three components: a read stream for receiving data from the server, a write stream for sending data to the server, and a callable to retrieve the session ID. These are stored in the instance variables for later use in communication with the server and for managing the session.
        _session -> a ClientSession is created using the read and write streams. This session will manage the communication protocol with the server, allowing for sending requests and receiving responses in a structured manner.
        initialize -> the session is initialized, which may involve sending an initial handshake or setup messages to the server to establish the communication protocol. Once this is complete, the client is marked as connected.
        """
        if self._connected:
            raise RuntimeError(f"{self.name} is already connected.")
        
        # Connect to Streamable HTTP server
        streamable_connection = await self._exit_stack.enter_async_context(streamablehttp_client(url=self.server_url, headers=headers))
        self.read, self.write, self._get_session_id = streamable_connection

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