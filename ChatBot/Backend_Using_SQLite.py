from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


# Defining the state class
class chat_state(TypedDict):

    message : Annotated[List[BaseMessage], add_messages]


# Defining the Chat node
def Chat(state: chat_state) -> chat_state:

    result = model.invoke(state['message'])

    return {'message': [result]}


# Creating Graph

graph = StateGraph(chat_state)

graph.add_node('chat_node', Chat)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)


conn = sqlite3.connect('chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

workflow = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
