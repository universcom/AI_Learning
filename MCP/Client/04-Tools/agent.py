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
                """
                The current_response is obtained by calling the anthropic client's messages.create method, which generates a response based on the conversation messages, available tools, and tool choice settings. The response is then added to the conversation as an assistant message. If the response indicates that a tool needs to be used (stop_reason == "tool_use"), the code extracts the tool use blocks from the response, executes the specified tools using the MCP client, 
                and collects the results. These results are then added back into the conversation as user messages, allowing for an iterative process where the LLM can continue to generate responses based on the tool outputs until a final text response is produced.
                tool_chiose are: 
                - auto: the model decides whether to use a tool or not based on the conversation and the tools available. If the model determines that using a tool would be beneficial for generating a more accurate or relevant response, it will choose to use the tool. Otherwise, it will generate a response without utilizing any tools.
                - force: the model is required to use a tool for generating a response, regardless of whether it is necessary or not based on the conversation context. This setting forces the model to leverage the available tools for every response, which can be useful for testing or for scenarios where tool usage is desired in every interaction.
                - any: the model can choose to use any of the available tools regardless of the conversation context. This setting allows for more flexibility, as the model is not constrained by the conversation history when deciding whether to use a tool. It can choose to use a tool even if it may not be strictly necessary based on the conversation, which can lead to more creative or unexpected responses.
                - none: the model will not use any tools, regardless of the conversation context. This setting forces the model to generate responses solely based on its internal knowledge and understanding, without leveraging any external tools. This can be useful for testing the model's capabilities without any assistance or for situations where tool usage is not desired.
                """
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
                """
                If the message response has a stop_reason of tool_use, then we know the model is waiting for a response with the results of the tool or tools that it is requesting.
                We create tool use message blocks from the response contents, and then execute any tools in a loop. From the results, we build a dictionary with the key type set to "tool_use", 
                tool_use_id set to the ID from the model’s tool use message response, and content to the result of the tool call. All of those are then added to the conversation_messages list. Once we’ve made all of the tool calls, the results and the original user question are sent beck to the LLM for evaluation. Because there are potentially multiple tool calls that could be made before returning to the user, we wrap all of this in a while loop.
                """
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