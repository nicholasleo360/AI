
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

# 1. Setup model
llm = HuggingFaceEndpoint(  
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 2. Define Response Schemas (fields you want to extract)
schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic")
]

# 3. Create StructuredOutputParser from response schemas
parser = StructuredOutputParser.from_response_schemas(schema)

# 4. Define PromptTemplate with format instructions
template = PromptTemplate(
    template="Give 3 facts about {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 5. Create Chain: template -> model -> parser
chain = template | model | parser

# 6. Invoke Chain
result = chain.invoke({"topic": "black hole"})

print(result)
print("\nType of result:", type(result))
