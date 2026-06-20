from mcp import ClientSession
from mcp.types import Prompt, PromptMessage
from mcp.client.stdio import StdioServerParameters, stdio_client
from contextlib import AsyncExitStack
import logging

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

    async def get_available_prompts(self) -> list[Prompt]:
        """
        Fetches the list of prompts the server currently exposes.

        A prompt is a reusable, server-defined message template that the client can
        request and render (often parameterized with arguments) to guide an LLM, e.g. a
        "summarize this text" or "review this code" template.

        Each returned Prompt object carries:
        - name: The unique name identifying the prompt.
        - description: A brief description of what the prompt does.
        - arguments: An optional list of arguments the prompt accepts, each with a name,
          description, and whether it is required.
        - model_config: Pydantic model configuration for the object.

        Returns:
            A list of Prompt objects, or an empty list if the server exposes none.
        """
        # Guard against listing before a session has been established.
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        prompt_result = await self._session.list_prompts()
        if not prompt_result.prompts:
            logger.warning("No prompts found on server")
        return prompt_result.prompts
    
    async def load_prompt(self, name: str, arguments: dict[str, str]) -> list[PromptMessage]:
        """
        Loads a single prompt from the server and renders it into concrete messages.

        Given a prompt's name and the arguments it expects, this asks the server to fill
        in its message template and return the resulting conversation messages, ready to
        be sent to an LLM.

        Args:
            name: The name of the prompt to load (as exposed by get_available_prompts).
            arguments: A mapping of argument names to values used to fill the template.

        Each returned PromptMessage object carries:
        - role: The role of the message, e.g. "user" or "assistant".
        - content: The rendered content of the message (text, image, audio, or an
          embedded resource).

        Returns:
            A list of PromptMessage objects, or an empty list if the prompt yields none.
        """
        # Guard against loading before a session has been established.
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        # Ask the server to render the named prompt with the supplied arguments.
        prompt_load_result = await self._session.get_prompt(name=name, arguments=arguments)

        if not prompt_load_result.messages:
            logger.warning(f"No prompt found for prompt {name}")
        else:
            logger.warning(f"Loaded prompt {name} with description {prompt_load_result.description}")
        return prompt_load_result.messages
    