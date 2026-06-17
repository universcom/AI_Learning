import os
import dotenv
import asyncio # for asynchronous programming
import pathlib # for handling file paths
from client import MCPClient  # MCP client class that connects to and communicates with an MCP server
from anthropic import Anthropic

dotenv.load_dotenv()

LLM_API_KEY = os.environ["LLM_API_KEY"]
anthropic_client = Anthropic(api_key=LLM_API_KEY)

mcp_client = MCPClient(
    name="calculator_server_connection", # a human-readable name for the client connection, useful for logging and debugging purposes.
    command="uv", # the command to execute on the server. In this case, "uv" is likely a command-line tool or script that will be run to start the server process.
    server_args=[
        "--directory",
        str(pathlib.Path(__file__).parent.resolve()),
        "run",
        "calculator_server.py",
    ],
)

print("Welcome to your AI Assistant. Type 'goodbye' to quit.")

async def main():
    await mcp_client.connect() # Establish a connection to the server using the connect method of the MCPClient instance. This will start the server process and set up communication channels.
    while True:
        prompt = input("You: ")
        if prompt.lower() == "goodbye":
            print("AI Assistant: Goodbye!")
            break
        message = anthropic_client.messages.create(
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="claude-sonnet-4-0",
        )
        for response in message.content:
            print(f"Assistant: {response.text}")
    await mcp_client.disconnect() # Clean up resources and close the connection to the server using the disconnect method of the MCPClient instance.
    
    
if __name__ == "__main__":
    asyncio.run(main()) # Run the main function using asyncio to handle asynchronous operations, such as connecting to the server and communicating with it.