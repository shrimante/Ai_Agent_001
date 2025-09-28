#Step 1: Setup API keys for GROQ and OpenAI

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file if present
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TRAVILY_API_KEY = os.getenv("TRAVILY_API_KEY")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")

#Step 2: Initialize LLMs and Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


openai_llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPEN_API_KEY)
groq_llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, groq_api_key=GROQ_API_KEY)

search_tool = TavilySearch(max_results=2) # Uses TRAVILY_API_KEY from environment variables    

#step 3: Create the AI Agent
from langgraph.prebuilt import create_react_agent
system_prompt = "You are a travel assistant. Use the search tool to find information about travel destinations, attractions, and activities. Provide concise and accurate answers based on the search results."
agent = create_react_agent(
    model=groq_llm,
    tools=[search_tool],
    prompt=system_prompt
)


query = "What are the top 5 places to visit in Bangalore?"
state = {"messages": [{"role": "user", "content": query}]}
response = agent.invoke(state)
messages = response["messages"]
last_msg = messages[-1]
print("\n=== FINAL OUTPUT ===")
print(last_msg.content)