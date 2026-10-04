from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()


model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash')

prompt1 = PromptTemplate(
    template = "generate notes for the following the following {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "generate 5 question for the following {topic}",
    input_variables = ['topic']
)

prompt3 = PromptTemplate(
    template = "merge the following quiz and note into one single documents {notes} and {quiz}",
    input_variables = ['notes', 'quiz']
)

parser = StrOuputParser()

parallel_chain = RunnableParallel(
    'notes' = prompt1 | model | parser,
    'quiz' = prompt2 | model | parser     
)

merge_chain = prompt3 | model | parser

overall_chain = parallel_chain | merge_chain 

overall_chain.invoke({'topic' : 'AI'})

overall_chain.get_graph().print_ascii()

