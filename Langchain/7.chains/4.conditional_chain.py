from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI( model = 'gemini-3.5-flash')

class Feedback(BaseModel) : 

    Sentiment : Literal['positive', 'Negative'] = Field(description = "give the following feedback text into positive or negative \n {feedback} \n {format_instruction}")

parser = StrOuputParser()
parser2 = PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
    template = "classify the following sentiment into negative or positive {feedback}",
    input_variables = ['feedback'],
    partial_variables = { 'format_instruction' : parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template = 'write an appropriate response to this positive feedback {feedback}',
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = 'write an appropriate response to this negative feedback {feedback}',
    input_variables = ['feedback']
)

classifier_chain = prompt1  | model | parser2

result = classifier_chain.invoke({'feedback' : 'this is a wonderful smartPhone'})


branch_chain = RunnableBranch(
    (lambda x : x.Sentiment == 'positive', prompt2 | model | parser ),
    (lambda x : x.Sentiment == 'Negative', prompt3 | model | parser ),
    RunnableLambda(lambda x : 'could not find the sentiment')
)

chain = classifier_chain | branch_chain
print(chain.invoke('this is a wonderful phone'))

chain.get_graph().print_ascii()


