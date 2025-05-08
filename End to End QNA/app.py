import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import langchain
import streamlit as st
from langchain_core.output_parsers import StrOutputParser


import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_PROJECT']="Q&A CHATBOT"

prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant. Please response to the user queries"),
        ("user", "Question: {question}")
    ]
)


def generate_response(question,api_key,llm,temperature,max_tokens):
    genai.configure(api_key=api_key)
    llm=ChatGoogleGenerativeAI(model=llm)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

st.title("Q&A chatbot with genai")

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your genai Model:", type="password")

llm=st.sidebar.selectbox("Select an Genai Model", ["gemma-2b","gemma-7b","flan-t5-xl"])

temperature=st.sidebar.slider("Temporature:", min_value=0.0, max_value=1.0, value=0.7)
max_tokens=st.sidebar.slider("Max Tokens:", min_value=50, max_value=300, value=150)

st.write("Go ahed and ask question")
user_input=st.text_input("you:")

if user_input:
    response=generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")