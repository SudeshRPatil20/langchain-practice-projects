import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from urllib.parse import urlparse, parse_qs

# Normalize YouTube URLs to standard format
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

# Streamlit app config
st.set_page_config(page_title="Langchain: Summarize Text From YT or Website", page_icon="🦜️")
st.title("🦜️ Langchain : Summarize text from YouTube or Website.")
st.subheader("Summarize URL")

# Sidebar for API key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# URL input
generic_url = st.text_input("Enter YouTube or Website URL", label_visibility="collapsed")

if not groq_api_key:
    st.error("Please enter Groq API Key")
    st.stop()

# Initialize LLM
llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)

# Prompt template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# On button click
if st.button("Summarize the content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or Website)")
    else:
        try:
            with st.spinner("Fetching and summarizing content..."):
                # Load content
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    yt_url = normalize_youtube_url(generic_url)
                    loader = YoutubeLoader.from_youtube_url(
                        yt_url,
                        add_video_info=True,
                        language="en"
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=True,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                docs = loader.load()

                # Summarization chain
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)  # FIXED: Use run() instead of invoke()

                st.subheader("Summary")
                st.write(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")
