
from typing import Any

async def _load_selected_prompts(self, prompts: list[dict[str, Any]]) -> str:
        """
        Load the given server prompts and combine them into a single system-instruction string.

        Each entry names a prompt (and its arguments) to render via the MCP server. The
        rendered messages are flattened into plain text, tagged with the prompt name, and
        joined together so the result can be used as the system prompt for the LLM.

        Args:
            prompts: A list of {"name": str, "arguments": dict} entries describing which
                prompts to load and the arguments to render each with.

        Returns:
            The selected prompts concatenated into one instruction string, or an empty
            string when none could be loaded.
        """
        system_instructions = []

        for prompt in prompts:
            # Only load prompts the server actually exposes.
            if prompt["name"] in self.available_prompts:
                print(f"Using prompt: {prompt['name']}")
                try:
                    # Ask the server to render this prompt with its arguments.
                    prompt_content = await self.mcp_client.load_prompt(
                        name=prompt["name"], arguments=prompt["arguments"]
                    )

                    # Flatten the rendered messages into a single block of text.
                    prompt_text = ""
                    for message in prompt_content:
                        # Structured content exposes its text via a .text attribute.
                        if hasattr(message.content, "text"):
                            prompt_text += message.content.text + "\n"
                        # Some prompts return content as a plain string.
                        elif isinstance(message.content, str):
                            prompt_text += message.content + "\n"

                    # Tag the text with the prompt name so its origin stays clear.
                    if prompt_text.strip():
                        system_instructions.append(
                            f"[Prompt: {prompt['name']}]\n{prompt_text.strip()}"
                        )

                except Exception as e:
                    # Loading is best-effort: skip a prompt that fails to render.
                    print(f"Error loading prompt {prompt['name']}: {e}")

        # Separate each prompt's instructions with a blank line.
        return "\n\n".join(system_instructions)

async def _refresh(self) -> None:
        """
        Re-discover the server's prompts and rebuild the name->Prompt lookup.

        Called at startup and whenever the prompt catalog needs to be reloaded.
        """
        available_prompts = await self.mcp_client.get_available_prompts()
        self.available_prompts = {prompt.name: prompt for prompt in available_prompts}