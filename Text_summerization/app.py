import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from urllib.parse import urlparse, parse_qs

def normalize_youtube_url(url):
    parsed = urlparse(url)
    if "youtube" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "v" in query:
            return f"https://www.youtube.com/watch?v={query['v'][0]}"
    elif "youtu.be" in parsed.netloc:
        video_id = parsed.path.lstrip("/")
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

# Streamlit UI
st.set_page_config(page_title="LangChain: Summarize", page_icon="🦜")
st.title("🦜 LangChain: Summarize YouTube or Website")

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")

generic_url = st.text_input("Enter URL")

if not groq_api_key:
    st.warning("Please enter your Groq API Key in the sidebar.")
    st.stop()

# ✅ Use a valid Groq model name
llm = ChatGroq(model="gemma-7b-it", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize"):
    if not validators.url(generic_url):
        st.error("Invalid URL")
    else:
        try:
            with st.spinner("Loading content..."):
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    yt_url = normalize_youtube_url(generic_url)
                    loader = YoutubeLoader.from_youtube_url(yt_url, add_video_info=True, language="en")
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=True,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                docs = loader.load()

                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

            st.subheader("Summary")
            st.write(output_summary)

        except Exception as e:
            st.error(f"Exception: {str(e)}")
