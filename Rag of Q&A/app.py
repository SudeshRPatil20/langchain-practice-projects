import streamlit as st
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

#for loading the the file content or api we use
import os
st.secrets["GROQ_TESTING_API"]=="GROQ_TESTING_API"

os.environ['GROQ_TESTING_API']=os.getenv('GROQ_TESTING_API')
groq_api_key=os.getenv('GROQ_TESTING_API')

llm=ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}
    
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")  # till hear data ingestion part has being over
        st.session_state.docs=st.session_state.loader.load() #now the loading thing is happing
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


st.title("Rag Document and Q&A with groq and llama3")
user_prompt=st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")
    
import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriver=st.session_state.vectors.as_retriever()
    retriver_chain=create_retrieval_chain(retriver, document_chain)
    
    start=time.process_time()
    response=retriver_chain.invoke({'input':user_prompt})
    print(f"Response_time :{time.process_time()-start}")
    
    st.write(response['answer'])
    
    
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("---------------------")
            
