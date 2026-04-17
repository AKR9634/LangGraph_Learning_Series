from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.prompts import PromptTemplate
import operator
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import json

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class essay(BaseModel):

    feedback : Annotated[str, Field(description="A one line feedback for the essay!!!")]
    score : Annotated[float, Field(description="The Score out of 10 for the essay", ge=0, le=10)]

parser = PydanticOutputParser(pydantic_object=essay)

prompt = PromptTemplate(
    template="""Evaluate the {criteria} quality of the following essay and provide a one line feedback and assign a score out of 10 \n {essay}
    
    Return Only in JSON format... No explanation needed...
    \n {format_instruction}...""",
    input_variables=['criteria', 'essay'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

text = """

Nestled in the serene lap of the Himalayas, Pauri Garhwal is one of the most picturesque districts in the Indian state of Uttarakhand. Known for its breathtaking landscapes, spiritual heritage, and rich cultural traditions, Pauri Garhwal represents the true essence of the Garhwal region. It is a place where nature, history, and spirituality blend harmoniously, offering both tranquility and inspiration.

Geography and Natural Beauty

Pauri Garhwal is characterized by lush green valleys, dense forests, and snow-capped peaks that create a mesmerizing panorama. Located at an elevation of around 1,800 meters above sea level, the town of Pauri offers spectacular views of the Himalayan ranges, including the majestic Chaukhamba peaks. The region enjoys a pleasant climate throughout the year, making it an ideal destination for nature lovers.

Nearby places such as Khirsu, known for its apple orchards and peaceful environment, add to the charm of the district. The forests of oak and deodar, along with diverse flora and fauna, make Pauri Garhwal a haven for eco-tourism and trekking enthusiasts.

Historical and Cultural Significance

Pauri Garhwal has a rich historical legacy that dates back to ancient times. It was once a part of the Garhwal Kingdom, ruled by the Parmar dynasty. The region played a significant role during the Indian independence movement, with many freedom fighters contributing to the struggle for independence.

Culturally, the people of Pauri Garhwal preserve their traditions through folk music, dance, and festivals. Celebrations such as Harela and Baisakhi reflect the agrarian lifestyle and deep connection of the locals with nature. The Garhwali language and customs continue to thrive, giving the region a unique identity.

Religious Importance

Pauri Garhwal is deeply rooted in spirituality, with numerous temples and sacred sites scattered across the district. Temples like Kandoliya Devta and Kyunkaleshwar Mahadev attract devotees and tourists alike. The region is also closely connected to the sacred pilgrimage circuit of Uttarakhand, often referred to as the “Land of Gods” or Devbhoomi.

Pilgrims traveling to revered sites such as Badrinath Temple and Kedarnath Temple often pass through parts of the Garhwal region, enhancing its spiritual significance.

Economy and Lifestyle

The economy of Pauri Garhwal is primarily based on agriculture, tourism, and small-scale industries. Farming remains the backbone of rural life, with crops like wheat, rice, and millets commonly grown on terraced fields. In recent years, tourism has emerged as a vital source of income, attracting visitors seeking peace and natural beauty.

Despite its scenic charm, the region faces challenges such as migration of youth to urban areas in search of better opportunities. However, efforts are being made to promote sustainable development and improve infrastructure.

Tourism and Attractions

Pauri Garhwal offers a variety of attractions for visitors. From serene viewpoints to spiritual centers, the district has something for everyone. Khirsu, Tarkeshwar Mahadev Temple, and the surrounding forests provide opportunities for relaxation and adventure alike. The calm and unspoiled environment makes it a perfect escape from the hustle and bustle of city life.

Conclusion

Pauri Garhwal stands as a symbol of natural beauty, cultural richness, and spiritual depth. Its majestic mountains, vibrant traditions, and peaceful environment make it a unique and cherished destination in Uttarakhand. As development continues, preserving its ecological balance and cultural heritage will be essential to maintaining its timeless charm.

"""

# Testing

# chain = prompt | model | parser

# result = chain.invoke({'criteria':'language', 'essay':text})

# print(result)

# print(type(result))

# language, depth analysis and clarity


# Creating the State Model for the Graph!!!

class UPSC_essay(TypedDict):

    essay: str

    language_feedback: str
    depth_feedback: str
    clarity_feedback: str

    overall_feedback: str
    individual_score: Annotated[list[float], operator.add]

    average_score: float


# Creating the functions for the nodes of the Graph!!!

def language_feedback(state: UPSC_essay) -> UPSC_essay:

    chain = prompt | model | parser

    result = chain.invoke({'criteria':'language', 'essay':text})

    return {'language_feedback':result.feedback, 'individual_score':[result.score]}

def depth_feedback(state: UPSC_essay) -> UPSC_essay:

    chain = prompt | model | parser

    result = chain.invoke({'criteria':'depth analysis', 'essay':text})

    return {'depth_feedback':result.feedback, 'individual_score':[result.score]}

def clarity_feedback(state: UPSC_essay) -> UPSC_essay:

    chain = prompt | model | parser

    result = chain.invoke({'criteria':'clarity', 'essay':text})

    return {'clarity_feedback':result.feedback, 'individual_score':[result.score]}

def overall_feedback(state: UPSC_essay) -> UPSC_essay:

    new_prompt = f"Provide a summarized feedback in one line using the following feedbacks: \n language_feedback: {state['language_feedback']}, \n depth feedback: {state['depth_feedback']}, \n clarity feedback: {state['clarity_feedback']}..."

    result = model.invoke(new_prompt).content

    return {'overall_feedback':result}

def calc_avg_score(state: UPSC_essay) -> UPSC_essay:

    avg_score = sum(state['individual_score'])/len(state['individual_score'])

    return {'average_score':avg_score}    


# Create and compile the graph!!!

graph = StateGraph(UPSC_essay)

graph.add_node('language_feedback', language_feedback)
graph.add_node('depth_feedback', depth_feedback)
graph.add_node('clarity_feedback', clarity_feedback)
graph.add_node('overall_feedback', overall_feedback)
graph.add_node('average_score', calc_avg_score)


graph.add_edge(START, 'language_feedback')
graph.add_edge(START, 'depth_feedback')
graph.add_edge(START, 'clarity_feedback')

graph.add_edge('language_feedback', 'overall_feedback')
graph.add_edge('depth_feedback', 'overall_feedback')
graph.add_edge('clarity_feedback', 'overall_feedback')

graph.add_edge('overall_feedback', 'average_score')

graph.add_edge('average_score', END)

workflow = graph.compile()


# Executing the workflow

result = workflow.invoke({'essay':text})

for key, value in result.items():
    print(f"{key} : {value}", "\n\n\n\n\n")