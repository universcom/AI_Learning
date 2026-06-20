import os
import dotenv
import asyncio
import json
import logging
from typing import Any
from pathlib import Path
from anthropic import Anthropic
from client import MCPClient
from mcp.types import  TextResourceContents


logger = logging.getLogger(__name__)

# Load environment variables (e.g. LLM_API_KEY) from a local .env file.
dotenv.load_dotenv()

# Initialize the Anthropic client used to talk to the LLM.
LLM_API_KEY = os.environ["LLM_API_KEY"]
anthropic_client = Anthropic(api_key=LLM_API_KEY)

class Agent:
    """
    A conversational agent that wires an Anthropic LLM together with an MCP server.

    It exposes the server's tools to the LLM, discovers the server's resources,
    and lets the LLM pull in relevant resources as extra context before answering
    the user. The main entry point is `run`, which drives an interactive chat loop.
    """

    def __init__(self, mcp_client: MCPClient, anthropic_client: Anthropic) -> None:
        # MCP client used to reach the server (tools + resources).
        self.mcp_client = mcp_client
        # Anthropic client used for LLM completions.
        self.anthropic_client = anthropic_client
        # Map of resource name -> Resource, populated by `_refresh_resources`.
        self.available_resources = {}

    async def _select_resources(self, prompt: str) -> list[str]:
        """
        Use the LLM to decide which available resources are relevant to a prompt.

        The model is shown the user's question plus a name->description map of every
        known resource, and asked to return a JSON array of the resource names that
        would help answer it (possibly empty).

        Args:
            prompt: The user's question.

        Returns:
            A list of resource names that exist in `self.available_resources`. Returns
            an empty list when no resources are known, none are relevant, or the LLM
            call/parsing fails.
        """
        # Nothing to choose from if no resources have been discovered yet.
        if not self.available_resources:
            return []

        # Build a name -> description map to show the LLM what each resource offers.
        resource_descriptions = {
            name: resource.description or f"Resource: {name}"
            for name, resource in self.available_resources.items()
        }

        selection_prompt = f"""
            Given this user question: "{prompt}"

            And these available resources:
            {json.dumps(resource_descriptions, indent=2)}

            Which resources (if any) would be helpful to answer the user's question?
            Return a JSON array of resource names, or an empty array if no resources are needed.
            Only include resources that are directly relevant.

            Example: ["math-constants"] or []
            """
        
        try:
            # Ask the LLM which resources are relevant; keep the response short.
            response = await self.anthropic_client.messages.create(
                max_tokens=200,
                messages=[{"role": "user", "content": selection_prompt}],
                model="claude-sonnet-4-0",
            )
            response_text = response.content[0].text.strip()
            # Extract JSON from response (handle case where LLM adds explanation)
            if "[" in response_text and "]" in response_text:
                start = response_text.find("[")
                end = response_text.rfind("]") + 1
                json_part = response_text[start:end]
                selected_resources = json.loads(json_part)
                # Drop any hallucinated names that aren't actually available.
                return [r for r in selected_resources if r in self.available_resources]

        except Exception as e:
            # Resource selection is best-effort: on failure, fall through to no resources.
            logger.warning(f"Failed to select resources with LLM: {e}")

        return []

    async def _load_selected_resources(self, resource_names: list[str]) -> list[dict[str, Any]]:
        """
        Read the given resources from the MCP server and turn them into LLM context blocks.

        Each resource's contents are converted into Anthropic message content blocks:
        text contents become "text" blocks, and supported images become base64 "image"
        blocks. Unsupported MIME types are skipped with a warning.

        Args:
            resource_names: Names of resources to load (must exist in `self.available_resources`).

        Returns:
            A list of content-block dicts ready to be appended to a user message.
        """
        context_messages = []

        for resource_name in resource_names:
            if resource_name in self.available_resources:
                try:
                    # Resolve the resource's URI and read its contents from the server.
                    resource = self.available_resources[resource_name]
                    resource_contents = await self.mcp_client.get_resource(uri=resource.uri)
                    for content in resource_contents:
                        # Text resource -> plain text block, tagged with the resource name.
                        if isinstance(content, TextResourceContents):
                            context_messages.append(
                                {
                                    "type": "text",
                                    "text": f"[Resource: {resource_name}]\n{content.text}",
                                }
                            )
                        elif content.mimeType in [
                            "image/jpeg",
                            "image/png",
                            "image/gif",
                            "image/webp",
                        ]:  # Supported image -> base64 image block.
                            context_messages.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": content.mimeType,
                                        "data": content.blob,
                                    },
                                }
                            )
                        else:
                            # Any other MIME type can't be embedded as context; skip it.
                            print(
                                f"WARNING: Unable to process mimeType {content.mimeType} for resource {resource_name}"
                            )

                except Exception as e:
                    # Loading is best-effort: skip a resource that fails to load.
                    logger.warning(f"Failed to load resource '{resource_name}': {e}")

            # NOTE: this return is indented inside the for-loop, so only the first
            # resource is ever processed. Dedent it one level to load all resources.
            return context_messages

    async def _refresh_resources(self) -> None:
        """
        Re-discover the server's resources and rebuild the name->Resource lookup.

        Called once at startup and again whenever the user types 'refresh'.
        """
        available_resources = await self.mcp_client.get_available_resources()
        self.available_resources = {
            resource.name: resource for resource in available_resources
        }

    async def run(self) -> None:
        """
        Run the interactive chat loop.

        Connects to the MCP server, loads its tools and resources, then repeatedly:
        reads a user prompt, optionally pulls in relevant resources as context, and
        drives an LLM tool-use loop until the model produces a final text answer.
        Type 'goodbye' to quit or 'refresh' to reload resources. The server connection
        is always closed on exit.
        """
        try:
            # Establish the MCP session before doing anything else.
            await self.mcp_client.connect()
            print("Welcome to your AI Assistant. Type 'goodbye' to quit or 'refresh' to reload and redisplay available resources.")
            # Fetch the tool list (exposed to the LLM) and the initial resource catalog.
            available_tools = await self.mcp_client.get_available_tools()
            await self._refresh_resources()

            while True:
                prompt = input("You: ")

                # Exit command.
                if prompt.lower() == "goodbye":
                    print("AI Assistant: Goodbye!")
                    break

                # Re-discover resources on demand.
                if prompt.lower() == "refresh":
                    await self._refresh_resources()
                    print("Available resources refreshed:")
                    continue

                # Let the LLM pick relevant resources, then load them as context.
                selected_resource_names = await self._select_resources(prompt)

                context_messages = await self._load_selected_resources(selected_resource_names)

                # Build conversation with initial user message and any context
                user_content = [{"type": "text", "text": prompt}]
                if context_messages:
                    user_content.extend(context_messages)

                conversation_messages = [{"role": "user", "content": user_content}]

                # Tool use loop - continue until we get a final text response
                while True:
                    # Get LLM response
                    current_response = anthropic_client.messages.create(
                        max_tokens=4096,
                        messages=conversation_messages,
                        model="claude-sonnet-4-0",
                        tools=available_tools,
                        tool_choice={"type": "auto"},
                    )
                    # Add assistant message to conversation
                    conversation_messages.append(
                        {"role": "assistant", "content": current_response.content}
                    )
                     # Check if we need to use tools
                    if current_response.stop_reason == "tool_use":
                        # Extract tool use blocks
                        tool_use_blocks = [
                            block
                            for block in current_response.content
                            if block.type == "tool_use"
                        ]

                        print(f"Executing {len(tool_use_blocks)} tool(s)...")
                        # Execute all tools and collect results
                        tool_results = []
                        for tool_use in tool_use_blocks:
                            print(f"Using tool: {tool_use.name}")
                            tool_result = await self.mcp_client.use_tool(
                                tool_name=tool_use.name, arguments=tool_use.input
                            )
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use.id,
                                    "content": "\n".join(tool_result),
                                }
                            )

                        # Add tool results to conversation
                        conversation_messages.append(
                            {"role": "user", "content": tool_results}
                        )

                        # Continue loop to get next LLM response
                        continue
                    else:
                        # No tools needed, extract final text response
                        text_blocks = [
                            content.text
                            for content in current_response.content
                            if hasattr(content, "text") and content.text.strip()
                        ]

                        if text_blocks:
                            print(f"Assistant: {text_blocks[0]}")
                        else:
                            print("Assistant: [No text response available]")

                        # Exit the tool use loop
                        break

        finally:
            # Always tear down the server connection, even on error.
            await self.mcp_client.disconnect()

if __name__ == "__main__":
    # Configure an MCP client that launches the calculator server via `uv run`.
    mcp_client = MCPClient(
        name="calculator_server_connection",
        command="uv",
        server_args=[
            "--directory",
            str(Path(__file__).parent.parent.resolve()),
            "run",
            "calculator_server.py",
        ],
    )
    agent = Agent(mcp_client, anthropic_client)
    asyncio.run(agent.run())
