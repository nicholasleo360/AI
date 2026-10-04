from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
import requests

@tool
def multiple( a, b ) : 

    """ multiply two numbers"""
    return a*b

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash')
llm_with_tools = model.bind_tool(multiple)

print(llm_with_tools.invoke([HumanMessage(content = "What is the square of 50?")]))