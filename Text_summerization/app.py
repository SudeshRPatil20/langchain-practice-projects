import validators
import streamlit as st
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

# Streamlit config
st.set_page_config(page_title="Langchain: Summarize Text From YT or Website", page_icon="🦜️")
st.title("🦜️ Langchain: Summarize text from YouTube or website.")
st.subheader("Summarize URL content")

# Sidebar API key input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# URL input
generic_url = st.text_input("URL", label_visibility="collapsed")

if not groq_api_key:
    st.error("Please Enter Groq API Key")
    st.stop()

# LLM initialization
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Prompt template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Button click
if st.button("Summarize the content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT URL or website URL")
    else:
        try:
            with st.spinner("Loading content..."):
                # Load data
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
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/115.0.0.0 Safari/537.36"
                            )
                        }
                    )

                docs = loader.load()

                # Summarization chain
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.invoke({"input_documents": docs})

            st.success("Summary Generated!")
            st.write(output_summary["output_text"])

        except Exception as e:
            st.exception(f"Exception: {e}")
