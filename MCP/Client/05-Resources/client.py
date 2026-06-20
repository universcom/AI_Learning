from typing import Any

from mcp import ClientSession
from contextlib import AsyncExitStack
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Resource , ResourceTemplate, BlobResourceContents, TextResourceContents
import logging
import json

logger = logging.getLogger(__name__)

class MCPClient:
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
    
    async def use_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> list[str]:
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        tool_call_result = await self._session.call_tool(name=tool_name, arguments=arguments)
        logger.debug(f"Calling tool {tool_name} with arguments {arguments}")

        results = []
        if tool_call_result.content:
            for content in tool_call_result.content:
                match content.type:
                    case "text":
                        results.append(content.text)
                    case "image" | "audio":
                        results.append(content.data)
                    case "resource":
                        if isinstance(content.resource, TextResourceContents):
                            results.append(content.resource.text)
                        else:
                            results.append(content.resource.blob)
        else:
            logger.warning(f"No content in tool call result for tool {tool_name}")
        return results
    
    async def get_available_resources(self) -> list[Resource]:
        """
        Fetches the list of resources the server currently exposes.

        A resource is a concrete, directly-addressable piece of content (identified by a
        fixed URI) that the server makes available to the client, e.g. a file, a database
        row, or an API response.

        Each returned Resource object carries:
        - uri: The unique URI identifying the resource.
        - name: The name of the resource.
        - description: A brief description of the resource.
        - mimeType: The MIME type of the resource's content (e.g. "text/plain", "application/json").
        - size: The size of the resource's content in bytes, if known.
        - annotations: Optional metadata providing additional context about the resource.
        - model_config: Pydantic model configuration for the object.

        Returns:
            A list of Resource objects, or an empty list if the server exposes none.
        """
        # Guard against listing before a session has been established.
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        # Ask the server for its catalog of available resources.
        resources_result = await self._session.list_resources()

        # Treat a missing list as "no resources" rather than propagating None.
        if resources_result.resources is None:
            logging.warning("No resources available from the server.")
            return []
        return resources_result.resources

    async def get_available_resource_templates(self) -> list[ResourceTemplate]:
        """
        Fetches the list of resource templates the server currently exposes.

        A resource template describes a *family* of resources rather than a single one: it
        provides a parameterized URI template that the client fills in to construct the URI
        of a concrete resource (e.g. "file:///logs/{date}.log").

        A ResourceTemplate has almost the same set of properties as a Resource, except it has
        a uriTemplate instead of a uri:
        - uriTemplate: The URI template (RFC 6570) describing how to construct a resource URI, including the parameters the client must supply.
        - name: The name of the resource template.
        - description: A brief description of the resource template.
        - mimeType: The MIME type of the resources produced by this template, if known.
        - size: The size of the resource's content in bytes, if known.
        - annotations: Optional metadata providing additional context about the resource template.
        - model_config: Pydantic model configuration for the object.

        Returns:
            A list of ResourceTemplate objects, or an empty list if the server exposes none.
        """
        # Guard against listing before a session has been established.
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        # Ask the server for its catalog of available resource templates.
        templates_result = await self._session.list_resource_templates()

        # Treat a missing list as "no templates" rather than propagating None.
        if templates_result.resourceTemplates is None:
            logging.warning("No resource templates available from the server.")
            return []
        return templates_result.resourceTemplates
    
    async def get_resource(self, uri: str) -> list[BlobResourceContents | TextResourceContents]:
        """
        Reads the contents of a single resource identified by its URI and returns them as a list.
        A resource can hold multiple content items, each being one of two types:
        - TextResourceContents: For text-based resources, exposing the data through a `text` field.
        - BlobResourceContents: For binary resources, exposing the data (base64-encoded) through a `blob` field.

        Both content types also carry:
        - uri: The URI of the resource the content belongs to.
        - mimeType: The MIME type of the content, if known.

        Args:
            uri: The unique URI identifying the resource to read.

        Returns:
            A list of TextResourceContents / BlobResourceContents objects holding the resource's
            contents, or an empty list if the resource has no content.
        """
        # Guard against reading before a session has been established.
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        # Ask the server to read the resource at the given URI.
        resource_read_result = await self._session.read_resource(uri=uri)

        # An empty result is valid but worth flagging for debugging.
        if not resource_read_result.contents:
            logger.warning(f"No content read for resource URI {uri}")
        return resource_read_result.contents
    