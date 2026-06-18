
import pathlib
import asyncio
from client import MCPClient

async def main():
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
    await mcp_client.connect()
    # You can now use mcp_client to interact with the server, such as fetching available tools or sending requests.
    available_tools = await mcp_client.get_available_tools()
    print("Available tools from the server:")
    for tool in available_tools:
        print(f"- {tool['name']}: {tool['description']}")
    await mcp_client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())