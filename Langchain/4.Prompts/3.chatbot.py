from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv(override=True)  

model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0)

chat_history = []

while True : 
    user_input = input('you : ')
    chat_history.append(user_input)
    if user_input == 'exit' :
        break

    result = model.invoke(chat_history)
    chat_history.append(result.text)
    print("AI : ", result.text)


print(chat_history)