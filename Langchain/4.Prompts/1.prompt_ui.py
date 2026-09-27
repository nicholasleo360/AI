from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv(override=True)

st.header("Research Tool Assistant")

user_input = st.text_input("Ask your Question")

if st.button("Summarize"):
    model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
    result = model.invoke(user_input)
    st.write(result.text)   