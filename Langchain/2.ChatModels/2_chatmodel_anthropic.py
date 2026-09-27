from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
result = model.invoke("What is the capital of India?")
print(result.content)
