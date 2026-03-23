"""
US Census MCP Server

An MCP server that provides access to US Census Bureau data through their API.
Returns data as pandas DataFrames for easy analysis.
"""

import asyncio
import json
import os
import sys
import logging
from typing import Any, Dict, List, Optional
import httpx
import pandas as pd
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio


# Initialize the MCP server
server = Server("us-census-server")

logger = logging.getLogger(__name__)
def debug_print(message):
    """Print debug messages to stderr so they don't interfere with MCP protocol"""
    print(f"DEBUG: {message}", file=sys.stderr, flush=True)
    logger.debug(message)

## These are some variables and functions specific to accessing the US Census API
##
# Base URL for Census API
CENSUS_BASE_URL = "https://api.census.gov/data"

# Common datasets and their years
DATASETS = {
    "acs5": ("acs/acs5", "American Community Survey 5-Year"),
    "acs1": ("acs/acs1", "American Community Survey 1-Year"), 
    "decennial": ("dec/sf1", "Decennial Census"),
    "pep": ("pep/population", "Population Estimates Program"),
    "cbp": ("cbp", "County Business Patterns"),
    "eits": ("eits", "Economic Indicators Time Series")
}

# Geography types
GEOGRAPHY_TYPES = {
    "us": "United States",
    "state": "State",
    "county": "County", 
    "tract": "Census Tract",
    "block group": "Block Group",
    "place": "Place",
    "metropolitan statistical area/micropolitan statistical area": "Metro/Micro Area",
    "congressional district": "Congressional District",
    "zip code tabulation area": "ZIP Code Tabulation Area"
}

def get_api_key() -> str:
    """Get Census API key from environment variable."""
    api_key = os.getenv("CENSUS_API_KEY")
    if not api_key:
        raise ValueError("CENSUS_API_KEY environment variable is required")
    return api_key

async def fetch_census_data(
    dataset: str,
    year: int,
    variables: List[str],
    geography: str,
    geography_filter: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Fetch data from US Census API and return as DataFrame.
    
    Args:
        dataset: Census dataset (e.g., 'acs5', 'decennial')
        year: Data year
        variables: List of variable codes to retrieve
        geography: Geography level (e.g., 'state', 'county')
        geography_filter: Optional filters for geography (e.g., {'state': '06'})
    
    Returns:
        pandas.DataFrame: Census data
    """
    api_key = get_api_key()

    
    # Get the correct dataset path
    if dataset in DATASETS:
        dataset_path, dataset_name = DATASETS[dataset]
    else:
        dataset_path = dataset
        dataset_name = dataset
    
    # Build the API URL with correct format
    url = f"{CENSUS_BASE_URL}/{year}/{dataset_path}"
  
    # Prepare parameters
    params = {
        "get": ",".join(variables),
        "for": geography + ":*",
        "key": api_key
    }
    
    # Add geography filters if provided
    if geography_filter:
        for geo_type, geo_code in geography_filter.items():
            params["in"] = f"{geo_type}:{geo_code}"
    
    # Build debug URL manually
    param_strings = [f"{k}={v}" for k, v in params.items()]
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)          
            response.raise_for_status()
            
            data = response.json()
 
            
            # Convert to DataFrame
            if data and len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
                return df
            else:
                return pd.DataFrame() #No data case, return empty DataFrame
                
        except httpx.HTTPError as e:
            raise Exception(f"Failed to fetch Census data: {e}")
        except Exception as e:
            raise Exception("Unexpected error occurred while fetching Census data")
######
#### Enf of US Census API specific functions and variables


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="us_census_tool",
            description="Access US Census Bureau data and return as DataFrame",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": f"Census dataset. Options: {', '.join(DATASETS.keys())}",
                        "enum": list(DATASETS.keys())
                    },
                    "year": {
                        "type": "integer",
                        "description": "Data year (e.g., 2022, 2020, 2010)",
                        "minimum": 2000,
                        "maximum": 2030
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of Census variable codes (e.g., ['B01003_001E', 'B25077_001E'])"
                    },
                    "geography": {
                        "type": "string", 
                        "description": f"Geography level. Options: {', '.join(GEOGRAPHY_TYPES.keys())}",
                        "enum": list(GEOGRAPHY_TYPES.keys())
                    },
                    "geography_filter": {
                        "type": "object",
                        "description": "Optional geography filters (e.g., {'state': '06'} for California)",
                        "additionalProperties": {"type": "string"}
                    }
                },
                "required": ["dataset", "year", "variables", "geography"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls."""

    if name != "us_census_tool":
        raise ValueError(f"Unknown tool: {name}")
    
    try:
        # Extract arguments
        dataset = arguments["dataset"]
        year = arguments["year"]
        variables = arguments["variables"]
        geography = arguments["geography"]
        geography_filter = arguments.get("geography_filter")
        
        # Fetch data
        df = await fetch_census_data(
            dataset=dataset,
            year=year,
            variables=variables,
            geography=geography,
            geography_filter=geography_filter
        )
        
        # Format response
        if df.empty:
            result = "No data found for the specified parameters."
        else:
            # Create summary
            summary = f"""
US Census Data Retrieved Successfully
====================================
Dataset: {DATASETS.get(dataset, dataset)} ({year})
Geography: {GEOGRAPHY_TYPES.get(geography, geography)}
Variables: {', '.join(variables)}
Records: {len(df)}

Data Preview:
{df.head().to_string()}

Data Types:
{df.dtypes.to_string()}
"""
            
            # Also include the raw data as JSON for programmatic access
            data_json = df.to_json(orient='records', indent=2)            
            result = f"{summary}\n\nFull Data (JSON):\n{data_json}"
        
        return [types.TextContent(type="text", text=result)]
        
    except Exception as e:
        error_msg = f"Error fetching Census data: {str(e)}"
        import traceback
        return [types.TextContent(type="text", text=error_msg)]


async def main():
    """Main entry point for the server."""
    
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        debug_print("Created stdio streams, starting server...")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="us-census-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())