from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv(override=True)  

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

chat_history = []

messages = [
    SystemMessage(
        content="You are a helpful assistant."
    )
]

while True : 
    user_input = input('you : ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit' :
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.text))
    print("AI : ", result.text)


print(result.text)