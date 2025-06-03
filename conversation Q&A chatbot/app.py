import os
from dotenv import load_dotenv
import streamlit as st 
# ❌ Removed: from langchain.chains import create_history_aware_retriever, create_retriever_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

# ✅ Added imports for chain composition
from langchain.chains.llm import LLMChain
from langchain.chains.retrieval import create_retrieval_chain

st.secrets["GROQ_TESTING_API"]="GROQ_TESTING_API"
st.secrets["GOOGLE_API_KEY"]="GOOGLE_API_KEY"
st.secrets["LANGCHAIN_API_KEY"]="LANGCHAIN_API_KEY"
st.secrets["HF_TOKEN"]="HF_TOKEN"
st.secrets["HF_TOKEN2"]="HF_TOKEN2"


load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversation Chat history with Q N A")
st.write("Upload pdf or chat with their content")

api_key = st.text_input("Enter Groq API", type="password")

if api_key:
    llm = ChatGroq(api_key=api_key, model="gemma2-9b-it")

    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose A PDF file", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        text_split = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        split = text_split.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=split, embedding=embedding)
        retriver = vectorstore.as_retriever()

        contextualize_system_prompts = (
            "Given a chat history and latest user question. "
            "Which might reference a context in a chat history. "
            "Formulate a standalone question which can be understood "
            "without the chat history. Do not answer the question, "
            "just reformulate if needed and otherwise return it as is."
        )

        contextualize_q_prompts = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_system_prompts),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # ✅ Replaced deprecated function with LCEL equivalent
        question_rewrite_chain = LLMChain(llm=llm, prompt=contextualize_q_prompts)
        history_aware_retriever = question_rewrite_chain | retriver

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # ✅ Replaced deprecated function with LCEL equivalent
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your Question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                },
            )

            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the Groq API")
