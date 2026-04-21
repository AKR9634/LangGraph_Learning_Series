from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


# TOOLS

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Performs a basic arithmetic operation (add, subtract, multiply, divide)
    on two numbers and returns the result or an error.
    """

    try:
        if operation == "add":
            result = first_num + second_num

        elif operation == "subtract":
            result = first_num - second_num

        elif operation == "multiply":
            result = first_num * second_num

        elif operation == "divide":
            if second_num == 0:
                return {"error": "Cannot divide by zero"}
            result = first_num / second_num

        else:
            return {"error": f"Invalid operation '{operation}'"}

        return {"result": result}

    except Exception as e:
        return {"error": str(e)}
    
@tool
def get_weather_data(city: str) -> str:
    """
    This fucntion fetches the current weather data for a given city!!!
    """
    API_KEY = "058e7445cd6d419ff49d81465952392e"

    url = f"https://api.weatherstack.com/current?access_key={API_KEY}&query={city}"

    response = requests.get(url)

    return response.json()

tools = [search_tool, get_weather_data, calculator]
model_with_tools = model.bind_tools(tools)

# Defining the state class
class chat_state(TypedDict):

    messages : Annotated[List[BaseMessage], add_messages]


# Defining the Chat node
def Chat(state: chat_state) -> chat_state:

    result = model_with_tools.invoke(state['messages'])

    return {'messages': [result]}


# Creating ToolNode
tool_node = ToolNode(tools)

# Creating Graph

graph = StateGraph(chat_state)

graph.add_node('chat_node', Chat)
graph.add_node("tools", tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')


conn = sqlite3.connect('chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

workflow = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
