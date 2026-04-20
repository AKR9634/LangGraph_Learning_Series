from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

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

checkpointer = MemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

