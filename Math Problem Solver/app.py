import streamlit as st 
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

## Set up the api for the stremlit
st.set_page_config(page_title="Text to Math Problem Solver And Data Search Assistant", page_icon="🦜️")
st.title("Text To Math Problem Solver Using Google Gemma")
# st.secrets["HF_TOKEN"]
# st.secrets["GROQ_TESTING_API"]
# st.secrets["GOOGLE_API_KEY"]
# st.secrets["LANGCHAIN_API_KEY"]
# st.secrets["LANGCHAIN_PROJECT"]
# st.secrets["HF_TOKEN2"]

groq_api_key=st.sidebar.text_input(label="Groq Api Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq Api Key to continue")
    st.stop()

llm=ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

#Insialize the tools
wekipedia_wrapper=WikipediaAPIWrapper()
wekipedia_tool=Tool(
    name="Wikipedia",
    func=wekipedia_wrapper.run,
    description="A tool for searching the Internet to find the various information on the topic mentioned"
)

##Initialize the Math tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering related question only for math problems and expresions."
)

prompt="""
Your a agen tasked for solving users mathemtical question. Logically arrived at the solution and provide a detailed explanation
and display it point wise for question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Combine all tools into chain
chain=LLMChain(llm=llm, prompt=prompt_template)

reasing_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering a logic-based and reasoning questions."
)

## Initialize the agent

assistant_agent=initialize_agent(
    tools=[wekipedia_tool, calculator, reasing_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a Math chatbot who can answer all your maths Problems"}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
    
## function to generate the response
# the below function is use if you are creating large project which contains frontend and backend both
# def generate_response(question):
#     response=assistant_agent.invoke({'input':question})
#     return response

## lets start the interaction
question=st.text_area("Enter your Question")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response...."):
            st.session_state.messages.append({"role":"user", "content":question})
            st.chat_message("user").write(question)
            
            st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            
            st.session_state.messages.append({'role':'assistant', "content":response})
            st.write('### Response:')
            st.success(response)
    else:
        st.warning("Please enter the question")

