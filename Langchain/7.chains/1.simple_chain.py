from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# load_dotenv()

# prompt = PromptTemplate(
#     template="give 5 interesting facts on {topic}",
#     input_variables=['topic']
# )

# model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# parser = StrOutputParser()

# chain = prompt | model | parser

# result = chain.invoke({'topic': 'elon musk'})

# print(result)

# chain.get_graph().print_ascii()


load_dotenv()


prompt = PromptTemplate(

    template = "give me 5 interesting fact about a {topic}",
    input_variables = ['topic']
)

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash')

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({'topic' : 'cricket'}))

chain.get_graph().print_ascii()