from mcp import ClientSession
from contextlib import AsyncExitStack
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Resource , ResourceTemplate
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
    
    async def get_available_resources(self) -> list[Resource]:
        """
        Fetches the list of available resources from the server and returns them as a list of Resource objects. Each Resource object contains the following information:
        - uri: The unique URI identifying the resource.
        - name: The name of the resource.
        - description: A brief description of the resource.
        - mimeType: The MIME type of the resource's content (e.g. "text/plain", "application/json").
        - size: The size of the resource's content in bytes, if known.
        - annotations: Optional metadata providing additional context about the resource.
        - model_config: Pydantic model configuration for the object.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")
        
        resources_result = await self._session.list_resources()
        if resources_result.resources is None:
            logging.warning("No resources available from the server.")
            return []
        return resources_result.resources
    
    async def get_available_resource_templates(self) -> list[ResourceTemplate]:
        """
        Fetches the list of available resource templates from the server and returns them as a list of ResourceTemplate objects.
        A ResourceTemplate has almost the same set of properties as a Resource, except it has a uriTemplate instead of a uri:
        - uriTemplate: The URI template (RFC 6570) describing how to construct a resource URI, including the parameters the client must supply.
        - name: The name of the resource template.
        - description: A brief description of the resource template.
        - mimeType: The MIME type of the resources produced by this template, if known.
        - size: The size of the resource's content in bytes, if known.
        - annotations: Optional metadata providing additional context about the resource template.
        - model_config: Pydantic model configuration for the object.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        templates_result = await self._session.list_resource_templates()
        if templates_result.resourceTemplates is None:
            logging.warning("No resource templates available from the server.")
            return []
        return templates_result.resourceTemplates