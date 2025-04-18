from fastapi import FastAPI
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
load_dotenv()

gemini_api_key=os.getenv("GOOGLE_API_KEY")
gorq_api_key=os.getenv("GROQ_TESTING_API")
model=ChatGroq(model="Gemma2-9b-It",groq_api_key=gorq_api_key)

system_template="Translate the following in {language}:"

prompt_template=ChatPromptTemplate([
    ('system',system_template),
    ('user','{text}')
])

parser=StrOutputParser()


chain=prompt_template|model|parser

app=FastAPI(title="Langchain Server",
            version="1.0",
            description="A simple API server using Langchain runnable interfaces")

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)
