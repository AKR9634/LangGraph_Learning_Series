from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class joke_eval(BaseModel):

    review : Annotated[str, Field(description="Review of the joke!!!")]
    score: Annotated[float, Field(description="Score of the joke!!!", ge=0, le=10)]

pyd_parser = PydanticOutputParser(pydantic_object=joke_eval)
str_parser = StrOutputParser()


#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
# TESTING

# prompt1 = PromptTemplate(
#     template="""
#                 Write a Twitter-style joke about {topic}.

#                 Requirements:

#                 Max 280 characters (ideally 1-2 lines)
#                 Strong hook in the first few words
#                 Include a clever twist, relatable observation, or absurd exaggeration
#                 Style: witty, slightly sarcastic, internet-native humor
#                 Can include modern references (memes, tech, daily life)
#                 Avoid clichés and obvious punchlines

#                 Optional:

#                 Add 1 subtle emoji (not required)
#                 Can mimic viral tweet tone or stand-up style

#                 Output only the tweet (no explanations).
#              """,
#     input_variables=['topic'],
#     )

# prompt2 = PromptTemplate(
#     template="""
#                 Review the following joke:

#                 {joke}

#                 Evaluate it based on:

#                 Humor (how funny it is)
#                 Originality
#                 Clarity
#                 Punchline effectiveness
#                 Relatability (especially for internet/Twitter audience)

#                 Give a single-line review summarizing the overall quality in a sharp, concise way.

#                 Then provide a score out of 10.

#                 Output format strictly:
#                 | Score: X/10

#                 Do not explain anything beyond this line.

#                 \n {instruction_format}
#              """,
#     input_variables=['joke'],    
#     partial_variables={'instruction_format':pyd_parser.get_format_instructions}
# )


# chain = prompt1 | model | str_parser | prompt2 | model | pyd_parser

# result = chain.invoke({'topic':'Food'})

# print(result)

#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------



# Creating the State Model!!!

class joke_state(TypedDict):

    topic: str
    joke: str
    review: str
    score: float
    joke_update: int


# Creating functions for the nodes!!!

def generate_joke(state: joke_state) -> joke_state:

    prompt1 = PromptTemplate(
    template="""
                Write a Twitter-style joke about {topic}.

                Requirements:

                Max 280 characters (ideally 1-2 lines)
                Strong hook in the first few words
                Include a clever twist, relatable observation, or absurd exaggeration
                Style: witty, slightly sarcastic, internet-native humor
                Can include modern references (memes, tech, daily life)
                Avoid clichés and obvious punchlines

                Optional:

                Add 1 subtle emoji (not required)
                Can mimic viral tweet tone or stand-up style

                Output only the tweet (no explanations).
             """,
        input_variables=['topic'],
    )

    chain = prompt1 | model | str_parser

    result = chain.invoke({'topic':state['topic']})

    joke_upd = state['joke_update'] + 1

    return {'joke': result, 'joke_update': joke_upd}


def generate_review_score(state: joke_state) -> joke_state:

    prompt2 = PromptTemplate(
    template="""
                Review the following joke:

                {joke}

                Evaluate it based on:

                Humor (how funny it is)
                Originality
                Clarity
                Punchline effectiveness
                Relatability (especially for internet/Twitter audience)

                Give a single-line review summarizing the overall quality in a sharp, concise way.

                Then provide a score out of 10.

                Output format strictly:
                | Score: X/10

                Do not explain anything beyond this line.

                \n {instruction_format}
             """,
        input_variables=['joke'],    
        partial_variables={'instruction_format':pyd_parser.get_format_instructions}
    )

    chain = prompt2 | model | pyd_parser

    result = chain.invoke({'joke':state['joke']})

    return {'review' : result.review, 'score' : result.score}


def regenerate_joke(state: joke_state) -> joke_state:

    prompt3 = PromptTemplate(
        template="""
                The following joke scored low and needs improvement:

                Original Joke:
                {joke}

                Review Feedback:
                {review}

                Your task:

                Rewrite this joke to make it significantly funnier and more engaging
                Improve the punchline, clarity, and originality
                Make it suitable for a Twitter-style audience (max 280 characters)
                Keep it concise with a strong hook and a sharp, clever twist
                Maintain the core idea but enhance execution

                Output only the improved joke. Do not provide explanations or multiple versions.
            """,
        input_variables=['joke', 'review'],    
    )

    chain = prompt3 | model | str_parser
    
    result = chain.invoke({'joke':state['joke'], 'review':state['review']})

    joke_upd = state['joke_update'] + 1

    return {'joke': result, 'joke_update': joke_upd}

def check_score(state: joke_state) -> Literal['Approved', 'Rejected']:

    if state['score'] < 8:
        return 'Rejected'
    else:
        return "Approved"
    
# Creating the Graph Model!!!

graph = StateGraph(joke_state)

graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_review_score', generate_review_score)
graph.add_node('regenerate_joke', regenerate_joke)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_review_score')
graph.add_conditional_edges('generate_review_score', check_score, {'Approved': END, 'Rejected': 'regenerate_joke'})
graph.add_edge('regenerate_joke', 'generate_review_score')

workflow = graph.compile()

result = workflow.invoke({'topic':'food', 'joke_update': 0})


for key, value in result.items():
    print(f"{key} : {value} \n\n\n")