from openai import OpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from colorama import init, Fore, Style
import openai

init(autoreset=True)
client = OpenAI()

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer questions concisely and accurately, only answer the question asked. "
    "When asked for up-to-date facts, news, and current statistics, use your built-in web search capability. "
    "Cite your source or summarize findings when possible."
    "When asked for summaries, provide a concise summary using all the information in the working memory."
)

class ConversationState(TypedDict):
    messages: List[Dict[str, str]]
    scratchpad: str

def user_input(state: ConversationState) -> ConversationState:
    text = input(f"{Fore.CYAN}You: {Style.RESET_ALL}")
    state["messages"].append({"role": "user", "content": text})
    return state

def bot_response(state: ConversationState) -> ConversationState:
    # Build the message context with system prompt and (optionally) scratchpad
    msg = []
    msg.append({"role": "system", "content": SYSTEM_PROMPT})
    # Inject scratchpad working memory if not empty
    if state["scratchpad"].strip():
        msg.append({
            "role": "system",
            "content": f"Working memory (facts, observations, summaries so far):\n{state['scratchpad']}"
        })
    # Append the rest of the chat so far (excluding old system prompts)
    for m in state["messages"]:
        if m["role"] != "system":
            msg.append(m)

    response = client.chat.completions.create(
        model="gpt-4o-search-preview",
        web_search_options={},
        messages=msg
    )
    reply = response.choices[0].message.content.strip()
    print(f"{Fore.GREEN}Assistant: {reply}{Style.RESET_ALL}")
    state["messages"].append({"role": "assistant", "content": reply})

    # Add relevant info to scratchpad: e.g., every assistant response for now
    # (You can get fancier and only add extracted facts/web results)
    state["scratchpad"] += f"\n{reply}"
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

graph = StateGraph(ConversationState)
graph.add_node("user_input", user_input)
graph.add_node("bot_response", bot_response)
graph.add_edge("user_input", "bot_response")
graph.add_conditional_edges("bot_response", should_end, {True: END, False: "user_input"})
graph.set_entry_point("user_input")
app = graph.compile()

initial_state = ConversationState(
    messages=[{"role": "system", "content": SYSTEM_PROMPT}],
    scratchpad=""
)

if __name__ == "__main__":
    print(f"{Fore.YELLOW}Chatbot started! Just let me know when you want to exit (type bye, for example).{Style.RESET_ALL}")
    app.invoke(initial_state)
    print(f"{Fore.YELLOW}Chatbot ended.{Style.RESET_ALL}")
