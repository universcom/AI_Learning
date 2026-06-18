from mcp import ClientSession
from contextlib import AsyncExitStack
from mcp.client.stdio import StdioServerParameters, stdio_client
from typing import Any
import logging

from mcp.types import TextResourceContents, BlobResourceContents

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
    
    async def call_tool(self, tool_name: str, tool_input: dict[str, Any] | None = None) -> list[str]:
        """"
        Calls a specified tool on the server with the given input and returns the results as a list of strings. The method handles the communication with the server, processes the response based on the content type, and extracts the relevant information to be returned to the caller.
        tool_name -> the name of the tool to be called on the server. This should match the name of one of the available tools that can be fetched using the get_available_tools method.
        tool_input -> a dictionary containing the input parameters for the tool. This input should conform to the input schema defined for the tool, which can be obtained from the get_available_tools method. The input parameters will be sent to the server when calling the tool, and the server will use this input to execute the tool's functionality and generate a response.
        The method first checks if the client is connected to a server. If not, it raises a RuntimeError. Then, it uses the MCP session to call the specified tool on the server, passing the tool name and input arguments. The result from the server is expected to contain the output from the tool execution, which can be in various formats such as text, images, audio, or resources.
        The method processes the result content based on its type. If the content is of type "text", it extracts the text and adds it to the results list. If the content is of type "image" or "audio", it extracts the binary data and adds it to the results list. If the content is of type "resource", it checks if the resource is a text resource and extracts the text, or if it's a different type of resource, it extracts the binary data. The extracted data is then added to the results list, which is returned to the caller.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")
        
        # Call the specified tool on the server using the MCP session, passing the tool name and input arguments. The result is expected to contain the output from the tool execution, which can be in various formats such as text, images, audio, or resources.
        tool_call_result = await self._session.call_tool(name=tool_name, arguments=tool_input)
        results = []
        if tool_call_result.result is None:
            logging.warning(f"Tool {tool_name} did not return any result.")
            return []
        
        # Process the result content based on its type. The content can be of different types such as text, image, audio, or resource. Depending on the type, the appropriate data is extracted and added to the results list, which is then returned to the caller.
        for content in tool_call_result.result.content:
           match content.type:
               # Depending on the type of content returned by the tool execution, the method processes it accordingly. If the content is of type "text", it extracts the text and adds it to the results list. 
                case "text":
                    results.append(content.text)
                # If the content is of type "image" or "audio", it extracts the binary data as base64-encoded and adds it to the results list.
                case "image" | "audio":
                    results.append(content.data)
                # If the content is of type "resource", it checks if the resource is a text resource and extracts the text, or if it's a different type of resource, it extracts the binary data. The extracted data is then added to the results list.
                case "resource":
                    # If the content is of type "resource", it checks if the resource is a text resource and extracts the text
                    if isinstance(content.resource, TextResourceContents):
                        results.append(content.resource.text)
                    # If the content is of type "resource", it checks if the resource is a different type of resource (not a text resource), it extracts the binary data. The extracted data is then added to the results list.
                    elif isinstance(content.resource, BlobResourceContents):
                        results.append(content.resource.blob)
        return results
        

         