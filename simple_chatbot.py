from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import openai
from colorama import init, Fore, Style

init(autoreset=True)

class ConversationState(TypedDict):
    messages: List[dict] 

def user_input(state: ConversationState) -> ConversationState:
    text = input(f"{Fore.CYAN}You: {Style.RESET_ALL}")
    state["messages"].append({"role": "user", "content": text})
    return state

def bot_response(state: ConversationState) -> ConversationState:
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=state["messages"],
        temperature=0.7
    )
    reply = response.choices[0].message.content.strip()
    print(f"{Fore.GREEN}Assistant: {reply}{Style.RESET_ALL}")
    state["messages"].append({"role": "assistant", "content": reply})
    return state

def should_end(state):
    last_user_message = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), "")
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a classifier. If the user message indicates they want to end or exit the conversation (such as 'bye', 'quit', 'exit', 'goodbye', 'stop', etc), respond with 'True'. Otherwise, respond with 'False'."},
            {"role": "user", "content": last_user_message}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower() == "true"
    #return len(state["messages"]) > 6  # after 3 exchanges

graph = StateGraph(ConversationState)
graph.add_node("user_input", user_input)
graph.add_node("bot_response", bot_response)
graph.add_edge("user_input", "bot_response")
graph.add_conditional_edges("bot_response", should_end, {True: END, False: "user_input"})
graph.set_entry_point("user_input")

app = graph.compile()

initial_state = ConversationState(messages=[
    {"role": "system", "content": f"{Fore.YELLOW}You are a helpful assistant giving concise answers.{Style.RESET_ALL}"}
])  

# app.invoke(initial_state)

if __name__ == "__main__":
    print(f"{Fore.YELLOW}Chatbot started! Just let me know when you want to exit (type bye, for example).{Style.RESET_ALL}")
    app.invoke(initial_state)
    print(f"{Fore.YELLOW}Chatbot ended.{Style.RESET_ALL}")
