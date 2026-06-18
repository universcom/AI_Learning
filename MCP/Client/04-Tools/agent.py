import os
import pathlib
import asyncio
import dotenv
from client import MCPClient
from anthropic import Anthropic

dotenv.load_dotenv()

LLM_API_KEY = os.environ["LLM_API_KEY"]
anthropic_client = Anthropic(api_key=LLM_API_KEY)


mcp_client = MCPClient(
        name="calculator_server_connection",
        command="uv",
        server_args=[
            "--directory",
            str(pathlib.Path(__file__).parent.resolve()),
            "run",
            "calculator_server.py",
        ],
    )
print("Welcome to your AI Assistant. Type 'goodbye' to quit.")

async def main():
    """"
    This function serves as the main entry point for the AI assistant application. It establishes a connection to the MCP server, retrieves the list of available tools, and manages the conversation loop with the user. The function handles user input, interacts with the LLM to generate responses, and manages tool usage when necessary. It ensures that resources are properly cleaned up by disconnecting from the MCP server when the application is finished.
        The main steps of the function include:
        1. Connecting to the MCP server using the MCPClient instance.
        2. Fetching the list of available tools from the server and displaying them to the user.
        3. Entering a conversation loop where it waits for user input, processes it, and generates responses using the LLM.
        4. If the LLM indicates that a tool needs to be used, it extracts the tool use information, calls the appropriate tool on the MCP server, and incorporates the results back into the conversation.
        5. Handling graceful shutdown by disconnecting from the MCP server when the user decides to exit the application.
    """
    try:
        await mcp_client.connect()
        available_tools = await mcp_client.get_available_tools()
        print(f"Available tools: {", ".join([tool['name'] for tool in available_tools])}")
        while True:
            prompt = input("Your message: ")
            if prompt.lower() == "goodbye":
                print("AI Assistant: Goodbye!")
                break
            # Build conversation starting with user message
            conversation_messages = [{"role": "user", "content": prompt}]
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
                    # Execute all tools and collect results
                    tool_results = []
                    for tool_use in tool_use_blocks:
                        print(f"Using tool: {tool_use.name}")
                        tool_result = await mcp_client.call_tool(
                            name=tool_use.name, arguments=tool_use.input
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

                    break
    finally:
        await mcp_client.disconnect()
    

if __name__ == "__main__":
    asyncio.run(main())