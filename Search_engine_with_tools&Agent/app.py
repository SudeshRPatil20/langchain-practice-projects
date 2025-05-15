import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os

load_dotenv()

#Arxiv and Wekipeadia initialize
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

weki_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
weki=WikipediaQueryRun(api_wrapper=weki_wrapper)

search=DuckDuckGoSearchRun(name="Search")


st.title("Langchain - Chat with Search")
""" 
In this example we are using StremlitCallbackHandler to Search with respect to actions and agents throught the llm.
try more agents as you can for practicing llms
"""

#settings for sidebar by stremalit
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter Groq Api Key: ", type="password")


if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi, I'm a chatbot who can search the web. How can i help you"}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])  

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)
    
    llm=ChatGroq(groq_api_key=api_key, model="gemma2-9b-it", streaming=True)
    tools=[search,arxiv,weki]
    
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)# Zero short is to search thing over internet and give reply  without geting past history and srtecturesd  is to search in structured w.r.t past history and also we are stetting parse errors true to handle that.
    
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)