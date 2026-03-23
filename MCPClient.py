"""
Simplified MCP Client for US Census Server
Gets US population by state using the Census MCP server
"""

import asyncio
import json
import subprocess
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os

import os
api_key = os.getenv("CENSUS_API_KEY")

async def simple_census_call():
    """
    Simple client that calls the US Census MCP server to get population by state.
    """
    
    # Server parameters - using current directory
    server_params = StdioServerParameters(
        command=sys.executable,  # Use current Python interpreter
        args=["us_census_server.py"],
        env={
            **os.environ,  # Pass all current environment variables
            "CENSUS_API_KEY": api_key  # Explicitly ensure the API key is passed
        }
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                
                # Initialize the session
                print("Initializing MCP session...")
                await session.initialize()
                print("Session initialized successfully!")
                
                # Skip listing tools for now to avoid compatibility issues
                print("Calling US Census tool for population by state...")
                
                # Tool arguments for getting population data
                tool_args = {
                    "dataset": "acs5",           # American Community Survey 5-year estimates
                    "year": 2022,                # Most recent available year
                    "variables": ["B01003_001E"], # Total population variable
                    "geography": "state"         # State-level data
                }
                
                # Call the tool
                result = await session.call_tool("us_census_tool", tool_args)
                
                # Print the result
                print("\n" + "="*60)
                print("CENSUS API RESPONSE:")
                print("="*60)
                
                for content in result:
                    if hasattr(content, 'text'):
                        print(content.text)
                    else:
                        print(str(content))
                
                print("="*60)
                print("Call completed successfully!")
                
    except Exception as e:
        print(f"Error during MCP call: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Additional debugging info
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    print("Simplified US Census MCP Client")
    print("=" * 40)

    # Make the actual MCP call
    asyncio.run(simple_census_call())