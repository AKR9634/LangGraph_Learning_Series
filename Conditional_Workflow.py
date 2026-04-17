from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import json

class sentiment(BaseModel):
    sentiment : Annotated[Literal['positive', 'negative'], Field(description="The Sentiment of the sentence!!!")]


class diagnosis(BaseModel):
    issue_type: Literal["service_failure","delayed_response","poor_quality","billing_problem","usability_issue"]
    tone: Literal["frustrated","angry","disappointed","sarcastic","complaining"]
    urgency: Literal["low","moderate","high","urgent","critical"]

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser1 = PydanticOutputParser(pydantic_object=sentiment)
parser2 = PydanticOutputParser(pydantic_object=diagnosis)



# Testing

# prompt = PromptTemplate(
#     template="Provide just the sentiment of the following sentence and no explantions needed... {text} \n {format_instruction}",
#     input_variables=['text'],
#     partial_variables={'format_instruction':parser1.get_format_instructions()}
# )

# chain = prompt | model | parser

# result = chain.invoke({'text':"This product is really very good!!!"})

# print(result.sentiment)

#------------------------------------------------------------------------------------------------------------------

# Creating State Model for the graph

class ReviewState(TypedDict):

    review: str
    sentiment: Literal['positive', 'negative']
    diagnosis: dict
    response: str



# Create functions for different nodes of the graph...

def classify_sentiment(state: ReviewState) -> ReviewState:
    
    prompt = PromptTemplate(
        template="Provide just the sentiment of the following sentence and no explantions needed... {text} \n {format_instruction}",
        input_variables=['text'],
        partial_variables={'format_instruction':parser1.get_format_instructions()}
    )

    chain = prompt | model | parser1

    result = chain.invoke({'text':state['review']})

    return {'sentiment':result.sentiment}


def positive_feedback(state: ReviewState) -> ReviewState:

    prompt = f"Give a positive and warm feedback to the user based on the review : {state['review']}"

    result = model.invoke(prompt).content

    return {'response':result}


def negative_feedback(state: ReviewState) -> ReviewState:

    diagnosis = state['diagnosis']
    prompt = f"""Give a sympathetic feedback to the user based on the negative review : {state['review']} and user's diagonosis
             issue_type: {diagnosis['issue_type']}
             tone: {diagnosis['tone']}
             urgency: {diagnosis['urgency']}
            """


    result = model.invoke(prompt).content

    return {'response':result}


def run_diagnosis(state: ReviewState) -> ReviewState:

    prompt = PromptTemplate(
        template="Diagnose this negative review {review} and return issue_type, tone and urgency.. No other explanation needed... \n {format_instruction}",
        input_variables=['review'],
        partial_variables={'format_instruction':parser2.get_format_instructions()}
    )

    chain = prompt | model | parser2

    result = chain.invoke({'review':state['review']})

    return {'diagnosis': result.model_dump()}


def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:

    if state['sentiment'] == 'positive':
        return "positive_response"
    else:
        return 'run_diagnosis'

# Creating the Graph and compiling it...

graph = StateGraph(ReviewState)

graph.add_node('classify_sentiment', classify_sentiment)
graph.add_node('positive_response', positive_feedback)
graph.add_node('negative_response', negative_feedback)
graph.add_node('run_diagnosis', run_diagnosis)

graph.add_edge(START, 'classify_sentiment')
graph.add_conditional_edges('classify_sentiment', check_sentiment)
graph.add_edge('positive_response', END)
graph.add_edge('run_diagnosis', 'negative_response')
graph.add_edge('negative_response', END)

workflow = graph.compile()


# Executing the workflow...

result = workflow.invoke({'review': "This is the worst product i bought!!!"})

for key, value in result.items():
    print(f"{key} : {value}", "\n\n\n\n\n")