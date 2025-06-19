import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables from .env file (optional)
load_dotenv()

# Optional LangSmith tracking (can be removed if not used)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OPENAI"

# Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's queries."),
    ("user", "Question: {question}")
])

# Response generation function
def generate_response(question, api_key, engine, temperature, max_tokens):
    # Pass API key directly to ChatOpenAI
    llm = ChatOpenAI(
        model=engine,
        openai_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Streamlit UI
st.title("🤖 Enhanced Q&A Chatbot With OpenAI")

# Sidebar settings
st.sidebar.title("🔧 Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
engine = st.sidebar.selectbox("Select OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main input
st.write("Go ahead and ask any question!")
user_input = st.text_input("You:")

if user_input and api_key:
    try:
        response = generate_response(user_input, api_key, engine, temperature, max_tokens)
        st.write("**Assistant:**", response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
elif user_input:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
else:
    st.info("💬 Waiting for your question...")
