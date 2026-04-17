from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


# Defining the State
class llm_state(TypedDict):

    topic: str
    outline: str
    blog: str


def generate_outline(state: llm_state) -> llm_state:

    prompt = PromptTemplate(
        template="Provide an outline on the following topic : \n {topic}",
        input_variables={'topic'}
    )

    chain = prompt | model

    state['outline'] = chain.invoke({'topic': state['topic']}).content

    return state


def generate_blog(state: llm_state) -> llm_state:

    prompt = PromptTemplate(
        template="Provide a professional blog on the topic: {topic} on the basis of given outline : \n {outline}",
        input_variables={'topic', 'outline'}
    )

    chain = prompt | model
    
    state['blog'] = chain.invoke({'topic': state['topic'], 'outline': state['outline']}).content

    return state


# Create a Graph workflow

graph = StateGraph(llm_state)

# Creating the nodes

graph.add_node('generate_outline', generate_outline)
graph.add_node('generate_blog', generate_blog)

# Creating the edges

graph.add_edge(START, 'generate_outline')
graph.add_edge('generate_outline', 'generate_blog')
graph.add_edge('generate_blog', END)

# Compile the Graph

workflow = graph.compile()



# Execute the workflow

result = workflow.invoke({'topic':'Pauri Garhwal'})

for key, value in result.items():
    print(f"{key} : {value}", "\n\n\n\n\n")