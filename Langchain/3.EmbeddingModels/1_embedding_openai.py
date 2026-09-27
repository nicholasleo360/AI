from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
query_result = embeddings.embed_query("What is the capital of India?")
print(query_result[:5])  # print vector prefix
