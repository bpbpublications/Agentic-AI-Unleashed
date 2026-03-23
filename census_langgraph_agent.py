import asyncio
import json
import os
import sys
import re
import argparse
from collections.abc import AsyncGenerator
from functools import reduce
from typing import TypedDict, Dict, List, Any, Optional

from acp_sdk.models import Message
from acp_sdk.models.models import MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class CensusAgentState(TypedDict):
    user_query: str
    intent: Dict[str, Any]
    mcp_query: str
    census_data: Dict[str, Any]
    final_response: str
    error: Optional[str]


class CensusDataAgent:
    def __init__(self, openai_api_key: str, mcp_server_path: str = "us_census_server.py"):
        """Initialize the Census Data Agent"""
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo-preview",
            temperature=0.1
        )
        self.mcp_server_path = mcp_server_path
        
        # State name mappings
        self.state_names = {
            "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", "06": "California",
            "08": "Colorado", "09": "Connecticut", "10": "Delaware", "11": "District of Columbia",
            "12": "Florida", "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
            "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana",
            "23": "Maine", "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
            "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada",
            "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York",
            "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma",
            "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island", "45": "South Carolina",
            "46": "South Dakota", "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont",
            "51": "Virginia", "53": "Washington", "54": "West Virginia", "55": "Wisconsin",
            "56": "Wyoming", "72": "Puerto Rico"
        }

    def understand_intent(self, state: CensusAgentState) -> Dict[str, Any]:
        """Step 1: Understand user intent and extract census data requirements"""
        user_query = state.get("user_query", "")
        
        intent_prompt = f"""
        Analyze the following user query about US census data and extract the intent:
        
        Query: "{user_query}"
        
        IMPORTANT RULES:
        - If the query asks about a specific state/location (contains "in [StateName]" or "of [StateName]"), 
          put the state name in "geographic_filter" and set analysis_type to "specific_location"
        - Examples: "What is the income in California?" → geographic_filter: "California", analysis_type: "specific_location"
        - Examples: "Population of Texas?" → geographic_filter: "Texas", analysis_type: "specific_location"
        
        Please identify:
        1. What demographic information is being asked for? (race, age, income, education, population, housing)
        2. What geographic level? (us, state, county, place, tract)
        3. What type of analysis? (highest, lowest, comparison, percentage, count, average, specific_location)
        4. Any specific demographic group mentioned?
        5. Any specific location mentioned?
        
        Respond in JSON format:
        {{
            "demographic_type": "race|age|income|education|population|housing",
            "geographic_level": "us|state|county|place|tract",
            "analysis_type": "highest|lowest|comparison|percentage|count|average|specific_location",
            "specific_demographic": "specific group if mentioned (e.g., 'Black or African American')",
            "geographic_filter": "ONLY the state name if asking about one specific state (e.g., 'California', 'Texas')",
            "metric_requested": "what exactly to measure"
        }}
        
        Examples:
        - "What is the population of California?" → {{"demographic_type": "population", "geographic_level": "state", "analysis_type": "specific_location", "geographic_filter": "California"}}
        - "Which state has the highest income?" → {{"demographic_type": "income", "geographic_level": "state", "analysis_type": "highest", "geographic_filter": null}}
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=intent_prompt)])
            intent_text = response.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', intent_text, re.DOTALL)
            if json_match:
                intent = json.loads(json_match.group())
            else:
                intent = {"error": "Could not parse intent"}
                
            return {
                **state,
                "intent": intent
            }
            
        except Exception as e:
            return {
                **state,
                "intent": {"error": str(e)},
                "error": f"Failed to understand intent: {str(e)}"
            }

    async def query_census_data(self, state: CensusAgentState) -> Dict[str, Any]:
        """Step 2: Query the MCP server for census data"""
        intent = state.get("intent", {})
        
        if "error" in intent:
            return {**state, "error": "Cannot query data due to intent parsing error"}
        
        try:
            # Build MCP query parameters based on intent
            mcp_query_params = self._build_mcp_query(intent)
            
            # Call the MCP server
            census_data = await self._call_mcp_server(mcp_query_params)
            
            return {
                **state,
                "mcp_query": json.dumps(mcp_query_params, indent=2),
                "census_data": census_data
            }
            
        except Exception as e:
            return {
                **state,
                "error": f"Failed to query census data: {str(e)}"
            }

    def process_and_answer(self, state: CensusAgentState) -> Dict[str, Any]:
        """Step 3: Process the census data and generate the final answer"""
        user_query = state.get("user_query", "")
        census_data = state.get("census_data", {})
        intent = state.get("intent", {})
        
        # Check if we have valid data
        if not census_data.get("success", False):
            error_msg = census_data.get("error", "No data returned from Census API")
            return {
                **state,
                "error": f"Census data query failed: {error_msg}"
            }
        
        data = census_data.get("data", [])
        
        # Handle case where data is a string (error message) instead of list
        if isinstance(data, str):
            return {
                **state,
                "error": f"Census API returned an error: {data}"
            }
        
        if not data or len(data) == 0:
            return {
                **state,
                "final_response": "No census data found matching your query criteria."
            }
        
        # Process the data based on the intent
        try:
            processed_answer = self._analyze_census_data(user_query, intent, data)
            return {
                **state,
                "final_response": processed_answer
            }
        except Exception as e:
            return {
                **state,
                "error": f"Error processing census data: {str(e)}"
            }

    def _build_mcp_query(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Build the appropriate query for the MCP server based on intent"""
        demographic = intent.get("demographic_type", "")
        geographic = intent.get("geographic_level", "state")
        specific_demo = intent.get("specific_demographic", "")
        
        # Default to 2022 ACS 5-year data
        dataset = "acs5"
        year = 2022
        
        # Map demographics to Census variable codes
        variables = self._get_census_variables(demographic, specific_demo)
        
        # Always query all states and filter in post-processing for simplicity
        query_params = {
            "dataset": dataset,
            "year": year,
            "variables": variables,
            "geography": geographic
        }
        
        return query_params

    def _get_census_variables(self, demographic_type: str, specific_demographic: str = "") -> List[str]:
        """Map demographic requests to Census Bureau variable codes"""
        
        if demographic_type == "race":
            return [
                "B01003_001E",  # Total population (always needed)
                "B02001_002E",  # White alone
                "B02001_003E",  # Black or African American alone
                "B02001_004E",  # American Indian and Alaska Native alone
                "B02001_005E",  # Asian alone
                "B02001_006E",  # Native Hawaiian and Other Pacific Islander alone
                "B02001_007E",  # Some other race alone
            ]
        elif demographic_type == "income":
            return ["B19013_001E"]  # Median household income only
        elif demographic_type == "population":
            return ["B01003_001E"]  # Total population only
        else:
            return ["B01003_001E"]  # Default to total population

    async def _call_mcp_server(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Call the MCP server with the constructed query parameters"""
        
        # Check for API key
        api_key = os.environ.get("CENSUS_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": "CENSUS_API_KEY environment variable not set!",
                "data": None
            }
        
        # Server parameters
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.mcp_server_path],
            env={
                **os.environ,
                "CENSUS_API_KEY": api_key
            }
        )
        
        try:
            # Use the same pattern as your working code
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    
                    # Initialize session
                    await session.initialize()
                    
                    # Call the tool
                    result = await session.call_tool("us_census_tool", query_params)
                    
                    # Extract the data from MCP response format
                    response_text = None
                    
                    if hasattr(result, 'content') and result.content:
                        for content_item in result.content:
                            if hasattr(content_item, 'text'):
                                response_text = content_item.text
                                break
                    
                    if not response_text:
                        return {
                            "success": False,
                            "error": "No text content found in server response",
                            "data": None
                        }
                    
                    # Check if it's an error
                    if response_text.startswith("Error"):
                        return {
                            "success": False,
                            "error": response_text,
                            "data": None
                        }
                    
                    # Extract JSON data from the response
                    try:
                        if "Full Data (JSON):" in response_text:
                            json_start = response_text.find("Full Data (JSON):") + len("Full Data (JSON):")
                            json_data = response_text[json_start:].strip()
                            
                            # Parse the JSON
                            data = json.loads(json_data)
                            
                            return {
                                "success": True,
                                "data": data,
                                "error": None,
                                "summary": response_text.split("Full Data (JSON):")[0].strip()
                            }
                        else:
                            return {
                                "success": True,
                                "data": response_text,
                                "error": None,
                                "summary": response_text
                            }
                    except json.JSONDecodeError as e:
                        return {
                            "success": False,
                            "error": f"Failed to parse JSON data: {e}",
                            "data": response_text
                        }
                        
        except Exception as e:
            return {
                "success": False,
                "error": f"Error during MCP call: {e}",
                "data": None
            }

    def _analyze_census_data(self, user_query: str, intent: Dict[str, Any], data: List[Dict]) -> str:
        """Analyze the census data and generate a specific answer"""
        try:
            demographic_type = intent.get("demographic_type", "")
            analysis_type = intent.get("analysis_type", "")
            specific_demographic = intent.get("specific_demographic", "")
            geographic_filter = intent.get("geographic_filter")
            
            # If there's a geographic filter (specific location requested), filter the data first
            if geographic_filter:
                data = self._filter_data_by_location(data, geographic_filter)
                if not data:
                    return f"No data found for {geographic_filter}. Please check the location name and try again."
            
            if demographic_type == "race" and "highest" in analysis_type.lower():
                return self._analyze_race_data_for_highest(data, specific_demographic)
            elif demographic_type == "income":
                return self._analyze_income_data(data, analysis_type, geographic_filter)
            elif demographic_type == "population":
                return self._analyze_population_data(data, analysis_type, geographic_filter)
            else:
                # Generic analysis
                return self._generic_data_analysis(data, user_query, intent)
                
        except Exception as e:
            return f"Error analyzing census data: {str(e)}"

    def _filter_data_by_location(self, data: List[Dict], location_name: str) -> List[Dict]:
        """Filter data for a specific location (state)"""
        # Find the state code for the requested location
        target_state_code = None
        for code, name in self.state_names.items():
            if name.lower() == location_name.lower():
                target_state_code = code
                break
        
        if not target_state_code:
            return []
        
        # Filter data for that state
        return [row for row in data if row.get("state") == target_state_code]

    def _analyze_race_data_for_highest(self, data: List[Dict], specific_demographic: str) -> str:
        """Analyze race data to find the state with highest proportion"""
        try:
            state_results = []
            
            for row in data:
                state_code = row.get("state", "")
                state_name = self.state_names.get(state_code, f"Unknown State ({state_code})")
                
                total_pop = int(row.get("B01003_001E", 0))
                
                if total_pop == 0:
                    continue
                
                # Get race-specific population
                race_pop = 0
                if "black" in specific_demographic.lower() or "african american" in specific_demographic.lower():
                    race_pop = int(row.get("B02001_003E", 0))  # Black or African American alone
                elif "white" in specific_demographic.lower():
                    race_pop = int(row.get("B02001_002E", 0))  # White alone
                elif "asian" in specific_demographic.lower():
                    race_pop = int(row.get("B02001_005E", 0))  # Asian alone    
                elif "native hawaiian" in specific_demographic.lower():
                    race_pop = int(row.get("B02001_006E", 0))  # Native Hawaiian and Other Pacific Islander alone   
                elif "american indian" in specific_demographic.lower():
                    race_pop = int(row.get("B02001_004E", 0))  # American Indian and Alaska Native alone
                elif "some other race" in specific_demographic.lower():
                    race_pop = int(row.get("B02001_007E", 0))  # Some other race alone

                if race_pop > 0:
                    percentage = (race_pop / total_pop) * 100
                    state_results.append({
                        "state": state_name,
                        "total_population": total_pop,
                        "demographic_population": race_pop,
                        "percentage": percentage
                    })
            
            # Sort by percentage (highest first)
            state_results.sort(key=lambda x: x["percentage"], reverse=True)
            
            if state_results:
                top_state = state_results[0]
                return f"{top_state['state']} has the highest proportion of {specific_demographic}. Of the {top_state['total_population']:,} residents, {top_state['demographic_population']:,} are {specific_demographic}, constituting {top_state['percentage']:.1f}% of the population."
            else:
                return "No data available for the specified demographic group."
                
        except Exception as e:
            return f"Error analyzing race data: {str(e)}"

    def _analyze_income_data(self, data: List[Dict], analysis_type: str, geographic_filter: str = None) -> str:
        """Analyze income data"""
        try:
            income_results = []
            
            for row in data:
                state_code = row.get("state", "")
                location_name = self.state_names.get(state_code, f"Unknown State ({state_code})")
                median_income = row.get("B19013_001E")
                
                if median_income and median_income != "-666666666":
                    income_results.append({
                        "location": location_name,
                        "median_income": int(median_income)
                    })
            
            # If a specific location was requested, return just that result
            if geographic_filter and income_results:
                location = income_results[0]
                return f"The median household income in {location['location']} is ${location['median_income']:,}."
            
            # Otherwise, handle comparison queries
            if "highest" in analysis_type.lower():
                income_results.sort(key=lambda x: x["median_income"], reverse=True)
                top_location = income_results[0]
                return f"{top_location['location']} has the highest median household income at ${top_location['median_income']:,}."
            elif "lowest" in analysis_type.lower():
                income_results.sort(key=lambda x: x["median_income"])
                bottom_location = income_results[0]
                return f"{bottom_location['location']} has the lowest median household income at ${bottom_location['median_income']:,}."
            else:
                avg_income = sum(r["median_income"] for r in income_results) / len(income_results)
                return f"Average median household income across all locations: ${avg_income:,.0f}"
                
        except Exception as e:
            return f"Error analyzing income data: {str(e)}"

    def _analyze_population_data(self, data: List[Dict], analysis_type: str, geographic_filter: str = None) -> str:
        """Analyze population data"""
        try:
            pop_results = []
            
            for row in data:
                state_code = row.get("state", "")
                location_name = self.state_names.get(state_code, f"Unknown State ({state_code})")
                total_pop = row.get("B01003_001E")
                
                if total_pop:
                    pop_results.append({
                        "location": location_name,
                        "population": int(total_pop)
                    })
            
            # If a specific location was requested, return just that result
            if geographic_filter and pop_results:
                location = pop_results[0]
                return f"The population of {location['location']} is {location['population']:,} residents."
            
            if "highest" in analysis_type.lower() or "largest" in analysis_type.lower():
                pop_results.sort(key=lambda x: x["population"], reverse=True)
                top_location = pop_results[0]
                return f"{top_location['location']} has the highest population with {top_location['population']:,} residents."
            elif "lowest" in analysis_type.lower() or "smallest" in analysis_type.lower():
                pop_results.sort(key=lambda x: x["population"])
                bottom_location = pop_results[0]
                return f"{bottom_location['location']} has the lowest population with {bottom_location['population']:,} residents."
            else:
                # total_pop = sum(r["population"] for r in pop_results)
                # return f"Total population across all locations: {total_pop:,}"
            
                #join the locations and their populations into a a list of strings, and concatenate into one big string
                pop_strings = [f"{r['location']}: {r['population']:,}" for r in pop_results]
                return "\n".join(pop_strings)

        except Exception as e:
            return f"Error analyzing population data: {str(e)}"

    def _generic_data_analysis(self, data: List[Dict], user_query: str, intent: Dict[str, Any]) -> str:
        """Generic data analysis when specific analysis isn't available"""
        try:
            if not data:
                return "No data available for your query."
            
            # Provide a summary of what was found
            summary = f"Found {len(data)} records matching your query. "
            
            # Show a sample of the data
            for sample_row in data:
                # sample_row = data[0]
                state_code = sample_row.get("state", "")
                location = self.state_names.get(state_code, "Unknown location")
                summary += f"\nSample data for {location}: "
                
                # Show relevant fields
                relevant_fields = []
                for key, value in sample_row.items():
                    if key != "state" and value and value != "-666666666":
                        relevant_fields.append(f"{key}: {value}")
                
                if relevant_fields:
                    # summary += ", ".join(relevant_fields[:3])  # Show first 2 fields
                    summary += ", ".join(relevant_fields[:])  # Show all fields
            
            return summary
            
        except Exception as e:
            return f"Error in generic analysis: {str(e)}"


# Initialize the Census Data Agent
census_agent = CensusDataAgent(
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    mcp_server_path="us_census_server.py"
)

# Create the LangGraph workflow
workflow = StateGraph(CensusAgentState)

# Add nodes (wrap methods in RunnableLambda)
workflow.add_node("understand_intent", RunnableLambda(census_agent.understand_intent))
workflow.add_node("query_census_data", RunnableLambda(census_agent.query_census_data))
workflow.add_node("process_and_answer", RunnableLambda(census_agent.process_and_answer))

# Connect nodes
workflow.set_entry_point("understand_intent")
workflow.add_edge("understand_intent", "query_census_data")
workflow.add_edge("query_census_data", "process_and_answer")
workflow.set_finish_point("process_and_answer")

# Compile the graph
graph = workflow.compile()

# Initialize the ACP Server (simplified like the working version)
server = Server()


@server.agent()
async def census_data_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """LangGraph Census Data Agent that answers questions about US Census data."""
    
    # Extract query from input messages
    query = reduce(lambda x, y: x + y, input)
    user_query = str(query).strip()
    
    if not user_query:
        yield MessagePart(content="Please provide a question about US Census data.")
        return
    
    # Check if required environment variables are set
    openai_key = os.getenv("OPENAI_API_KEY")
    census_key = os.getenv("CENSUS_API_KEY")
    
    if not openai_key:
        yield MessagePart(content="⚠️ Error: OPENAI_API_KEY environment variable not set. Please set it and try again.")
        return
    
    if not census_key:
        yield MessagePart(content="⚠️ Error: CENSUS_API_KEY environment variable not set. Please set it and try again.")
        return
    
    # Initialize the agent with API keys
    try:
        census_agent_instance = CensusDataAgent(
            openai_api_key=openai_key,
            mcp_server_path="us_census_server.py"
        )
    except Exception as e:
        yield MessagePart(content=f"❌ Error initializing Census agent: {str(e)}")
        return
    
    output = None
    try:
        # Create the LangGraph workflow on-demand
        workflow = StateGraph(CensusAgentState)
        workflow.add_node("understand_intent", RunnableLambda(census_agent_instance.understand_intent))
        workflow.add_node("query_census_data", RunnableLambda(census_agent_instance.query_census_data))
        workflow.add_node("process_and_answer", RunnableLambda(census_agent_instance.process_and_answer))
        
        workflow.set_entry_point("understand_intent")
        workflow.add_edge("understand_intent", "query_census_data")
        workflow.add_edge("query_census_data", "process_and_answer")
        workflow.set_finish_point("process_and_answer")
        
        graph = workflow.compile()
        
        # Stream through the graph execution
        async for event in graph.astream({"user_query": user_query}, stream_mode="updates"):
            for node_name, node_output in event.items():
                # Yield intermediate updates (like the working simple version)
                if node_name == "understand_intent" and "intent" in node_output:
                    intent = node_output["intent"]
                    if "error" not in intent:
                        yield {"update": {"intent_understood": f"Analyzing {intent.get('demographic_type', 'data')} query"}}
                elif node_name == "query_census_data" and "census_data" in node_output:
                    census_data = node_output["census_data"]
                    if census_data.get("success"):
                        data_count = len(census_data.get("data", []))
                        yield {"update": {"data_retrieved": f"Retrieved {data_count} records from Census API"}}
                elif node_name == "process_and_answer":
                    yield {"update": {"processing": "Analyzing census data and generating response..."}}
            output = event
        
        # Get the final response
        final_response = ""
        if output:
            for node_output in output.values():
                if "final_response" in node_output:
                    final_response = node_output["final_response"]
                    break
                elif "error" in node_output:
                    final_response = f"Error: {node_output['error']}"
                    break
        
        if not final_response:
            final_response = "I apologize, but I wasn't able to process your census data request."
        
        yield MessagePart(content=final_response)
        
    except Exception as e:
        yield MessagePart(content=f"An error occurred while processing your request: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ACP Server on a specified port")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()

    try:
        print("🏛️  Starting Census Data Agent with LangGraph and MCP...")
        print("Server will run on http://localhost:8000")
        print("Press Ctrl+C to stop")
        
        # Check for required environment variables
        openai_key = os.getenv("OPENAI_API_KEY")
        census_key = os.getenv("CENSUS_API_KEY")
        
        if not openai_key:
            print("⚠️  Warning: OPENAI_API_KEY not set")
        if not census_key:
            print("⚠️  Warning: CENSUS_API_KEY not set")
        
        # Run the server (same as working simple version)
        server.run(port=args.port)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down Census Data Agent")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)