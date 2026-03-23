from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool, tool
import requests
import argparse

@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a given city using Open-Meteo API.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        Weather information as a string
    """
    try:
        # First, get coordinates for the city using a geocoding service
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_response = requests.get(geocoding_url)
        geo_data = geo_response.json()
        
        if not geo_data.get('results'):
            return f"Could not find coordinates for {city}"
        
        lat = geo_data['results'][0]['latitude']
        lon = geo_data['results'][0]['longitude']
        country = geo_data['results'][0].get('country', 'Unknown')
        
        # Get weather data
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()
        
        current_weather = weather_data['current_weather']

        return current_weather

    except Exception as e:
        return f"Error getting weather for {city}: {str(e)}"



model = LiteLLMModel(
    model_id="openai/gpt-4o",  
    max_tokens=2048
)

server = Server()

@server.agent()
async def weather_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is a CodeAgent which answers questions about weather based on open-meteo."
    agent = CodeAgent(tools=[get_weather], model=model)

    prompt = input[0].parts[0].content
    response = agent.run(prompt)

    yield Message(parts=[MessagePart(content=str(response))])


@server.agent()
async def web_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is a CodeAgent which answers questions based on web resources."
    agent = CodeAgent(tools=[DuckDuckGoSearchTool(), VisitWebpageTool()], model=model)

    prompt = input[0].parts[0].content
    response = agent.run(prompt)

    yield Message(parts=[MessagePart(content=str(response))])




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ACP Server on a specified port")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    server.run(port=args.port)
