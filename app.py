import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from amadeus import Client, ResponseError
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import tool
from langchain.agents import AgentExecutor
from typing import Union, List, Dict
import os
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain.load.dump import dumps
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
amadeus = Client(client_id=os.getenv('AMADEUS_CLIENT_ID'), client_secret=os.getenv('AMADEUS_SECRET'))
mongo_uri = os.getenv('MONGO_URI')
client =MongoClient(mongo_uri)
embedding = OpenAIEmbeddings()
MEMORY_KEY ="chat_history"
chat_history = []
input1 =""

def generate_embeddings(text):
    return embedding.embed_query(text).data[0]


def response_store_into_MongoDB(response: Union[Dict, List[Dict]]) -> None:
    ""
    "Store response into MongoDB collection called flightData."
    ""
    # Create a connection to the MongoDB instance
    connection_string = mongo_uri
    # Access the 'flightData' collection in the database named 'skypulse'
    db = client['skypulse']
    collection = db['flightData']

    # Check if the response is a list and insert accordingly
    if isinstance(response, list):
        collection.insert_many(response)
    else:
        collection.insert_one(response)

@tool
def get_flight_offer(originLocationCode: str, destinationLocationCode: str, departureDate: str, adults: int) -> dict:
    """
    Fetch all possible flight offers given departure and arrival date and number of adults.
    """
    params = {
        'originLocationCode': originLocationCode,
        'destinationLocationCode': destinationLocationCode,
        'departureDate': departureDate,
        'adults': adults,
        'max': 3
    }
    try:
        response = amadeus.shopping.flight_offers_search.get(**params)
        if not response.data:
            return 0
        for v in response.data:
            v['params'] = params
        print(response.data)
        response_store_into_MongoDB(response.data)
        return response.data  # Assuming you want to return the data part of the response
    except ResponseError as error:
        print(error)
        return {}  # Return an empty dict or handle the error as appropriate

@tool
def get_3cheapest_flight_from_MongoDB(originLocationCode: str, destinationLocationCode: str, departureDate: str, email: str) -> list:
    """
    Fetch the cheapest flight from MongoDB collection called flightData.
    """
    # Create a connection to the MongoDB instance
    connection_string = mongo_uri
    # Access the 'flightData' collection in the database named 'skypulse'
    db = client['skypulse']
    collection = db['flightData']
    # Find 3 cheapest flights
    response = collection.find({
        'params.originLocationCode': originLocationCode,
        'params.destinationLocationCode': destinationLocationCode,
        'params.departureDate': departureDate
    }).sort("price.total", 1).limit(3)
    test = list(response)
    get_request_from_MongoDB(email, flights=test)
    return test

def get_request_from_MongoDB(email: str, hotels=[], poi=[], flights=[]) -> dict:
    """
    Update and retrieve user requests from MongoDB based on provided data.
    """
    # Create a connection to the MongoDB instance
    connection_string = mongo_uri
    # Access the 'requests' collection in the database named 'skypulse'
    db = client['skypulse']
    collection = db['requests']
    
    # Update POI data if provided
    if len(poi) > 0:
        request_update = collection.update_one(
            {"login": email},
            {"$set": {"poi": poi}}
        )
    
    # Update hotel data if provided
    if len(hotels) > 0:
        request_update = collection.update_one(
            {"login": email},
            {"$set": {"hotels": hotels}}
        )
    
    # Update flight data if provided
    if len(flights) > 0:
        request_update = collection.update_one(
            {"login": email},
            {"$set": {"flights": flights}}
        )
    
    # Retrieve and return the updated user request
    response = collection.find_one({"login": email})
    return response

@tool
def get_interest_activity_from_mongoDB(email: str) -> dict:
    """
    By getting the email from user, get his list of hobby. Then, fetch the interest activities.
    """
    db = client['skypulse']
    collection = db['users']
    userhobby = collection.find_one({"email": email}, {"_id": 0, "settings": 1})

    if userhobby is None:
        print(f"No user settings found for email: {email}")
        return {}  # Return an empty dict or handle as appropriate

    # Assuming settings is always a dictionary with lists as values
    try:
        text = ', '.join(item for sublist in userhobby['settings'].values() for item in sublist)
        userHobby_embeddings = generate_embeddings(text)
        hobbies = userhobby["settings"]["hobbies"]
        interests = userhobby["settings"]["interests"]
        foods = userhobby["settings"]["foods"]
        print(f"Hobbies, Interests, Foods: {hobbies}, {interests}, {foods}")

        collection1 = db['poi_embeddings']
        proposedPOI = list(collection1.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embeddings",
                    "queryVector": userHobby_embeddings,
                    "numCandidates": 10,
                    "limit": 3,
                    'filter': {'type': {'$ne': 'Hotel'}}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "embeddings": 0
                }
            }
        ]))

        print(f"Proposed POI: {proposedPOI}")
        get_request_from_MongoDB(email, poi=proposedPOI)
        return proposedPOI
    except Exception as e:
        print(f"Error processing user settings for email {email}: {e}")
        return {}

@tool
def get_interested_hotels_from_mongoDB(email: str) -> dict:
    """
    By getting the email from the user, get his list of past hotels. Then, fetch proposed hotels.
    """
    db = client['skypulse']
    collection = db['users']
    user_data = collection.find_one({"email": email})

    if not user_data or 'pastHotels' not in user_data or not user_data['pastHotels']:
        print(f"No past hotels found for email: {email}")
        return {}  # Return an empty dict if no past hotels are found

    pastHotels = user_data['pastHotels']
    if not pastHotels:
        print(f"No past hotels data available for email: {email}")
        return {}

    # Ensure pastHotels is a list of lists before attempting to concatenate
    if not all(isinstance(sublist, list) for sublist in pastHotels):
        print(f"Invalid past hotels data structure for email: {email}")
        return {}

    # Concatenate all text in the past hotels
    text = ', '.join(item for sublist in pastHotels for item in sublist)
    print(text)

    # Generate embeddings for the concatenated text
    pastHotels_embeddings = generate_embeddings(text)
    print(pastHotels_embeddings)

    # Access the 'poi_embeddings' collection
    collection1 = db['poi_embeddings']
    proposedHotels = list(collection1.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embeddings",
                "queryVector": pastHotels_embeddings,
                "numCandidates": 10,
                "limit": 3,
                'filter': {'type': {'$eq': 'Hotel'}}
            }
        },
        {
            "$project": {
                "_id": 0,
                "embeddings": 0
            }
        }
    ]))

    print(proposedHotels)

    # Update the MongoDB 'requests' collection with the proposed hotels
    get_request_from_MongoDB(email, hotels=proposedHotels)

    return proposedHotels

prompt_text = """
Overview: You are a top-tier travel planning assistant. Leverage historical data stored in MongoDB to generate your recommendations. When data is unavailable in MongoDB, retrieve new data from the Amadeus API.
Flight Information:
Use the tool get_3cheapest_flights_from_MongoDB to fetch the three cheapest flights.
If no results are found in MongoDB, use get_info_from_amadeus to obtain flight information.
Activities:
Use the tool get_interest_activity_from_MongoDB to retrieve interesting activities.
If MongoDB yields no results, use get_info_from_amadeus for activity information.
Hotels:
Use the tool get_interested_hotels_from_MongoDB to find hotels of interest.
If no hotels are found in MongoDB, fetch hotel information using get_info_from_amadeus.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_text),
    MessagesPlaceholder(variable_name=MEMORY_KEY),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tools = [get_flight_offer, get_3cheapest_flight_from_MongoDB, get_interest_activity_from_mongoDB, get_interested_hotels_from_mongoDB]
llm_with_tools = llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"]
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data['query']
    result = agent_executor.invoke({
        "input": query,
        "chat_history": chat_history
    })
    print("ICI LE RESULTAT")
    returnable_result = dumps(result, pretty=True)
    chat_history.extend([
        HumanMessage(content=input1),
        AIMessage(content=result["output"]),
    ])
    print(returnable_result)
    return returnable_result
