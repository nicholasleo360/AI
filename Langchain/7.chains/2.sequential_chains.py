from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash")

prompt1 = PromptTemplate(

    template = "generate a detailed report on the {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    
    template = "give a 5 point summary of the {report}",
    input_variables = ['report']
)

parser = StrOuputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

chain.invoke("unemployement in india")

chain.get_graph().print_ascii()