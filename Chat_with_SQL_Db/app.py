import streamlit as st 
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os
load_dotenv()

st.set_page_config(page_title="Langchain: Chat with SQL DB", page_icon="🦜️")
st.title("🦜️ Langchain chat with SQL DB")

# INJECTION_WRAINING="""
# Sql agent is velnerable to prompt injection. Use a DB role with line.
# """
LOCALDB="USE_LOCALDB"
MYSQL="USE_MYSQL"

radio_opt=["Use SQLLite 3 Database- Student.db", "Connect to you MySQL Database"]

selected_opt=st.sidebar.radio(label="Choose the DB which you want to chat", options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri=MYSQL
    mysql_host=st.sidebar.text_input("Provide MySQL Host")
    mysql_user=st.sidebar.text_input("MySQL user")
    mysql_password=st.sidebar.text_input("MySql Password", type="password")
    mysql_db=st.sidebar.text_input("MySQL Database")
else:
    db_uri=LOCALDB
    
api_key=st.sidebar.text_input("Enter Groq Api Key: ", type="password")

if not api_key:
    st.error("Please Enter Groq Api Key ")
    st.stop()

#LLM model
llm = ChatGroq(api_key=api_key, model_name="gemma2-9b-it", streaming=True)

# as we now we did not want to catch the elements for full tile like if someone else want to use this app then they need to give its own api key so therefore we use chashe recond till total time limit
#we use decorator

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri==LOCALDB:
        dbfilepath=(Path(__file__).parent/"student.db").absolute()
        print(dbfilepath)
        return SQLDatabase(create_engine(f"sqlite:///{dbfilepath}"))
    elif db_uri==MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all mysql details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    
if db_uri==MYSQL:
    db=configure_db(
        db_uri=db_uri,
        mysql_host=mysql_host,
        mysql_user=mysql_user,
        mysql_password=mysql_password,
        mysql_db=mysql_db
    )
else:
    db=configure_db(db_uri)
    
    
#now we are integrating with sql toolkit and creating are web and interacting it with database

#toolkit

toolkit=SQLDatabaseToolkit(db=db, llm=llm)

#for creating an agent we use creaate sql agent that will creae an agent that will interact with sql query as well as llms that are are more important in SQLDB

agent=create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role":"assistant", "content":"How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
user_query=st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role":"user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        streamlit_callback=StreamlitCallbackHandler(st.container())
        response=agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)
