import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from urllib.parse import urlparse, parse_qs
import yt_dlp

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

def fetch_youtube_transcript(url):
    """Try YoutubeLoader first, fallback to yt_dlp if transcript fails."""
    try:
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,
            language="en",
            download_audio=False
        )
        return loader.load()
    except Exception:
        # Fallback to yt_dlp to at least get video description
        ydl_opts = {"quiet": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            description = info.get("description", "")
            return [{"page_content": description}]

# Streamlit UI
st.set_page_config(page_title="Langchain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 Langchain: Summarize text from YouTube or Website")
st.subheader("Summarize URL")

# Sidebar
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")

generic_url = st.text_input("Enter YouTube or Website URL", placeholder="https://...")

if not groq_api_key:
    st.error("Please enter your Groq API Key")
    st.stop()

# LLM init
try:
    llm = ChatGroq(
        model="gemma2-9b-it",
        groq_api_key=groq_api_key,
        temperature=0,
        max_tokens=1024
    )
except Exception as e:
    st.error(f"LLM Init Error: {e}")
    st.stop()

prompt_template = """
Provide a concise summary (max 300 words) of the following content:

{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize Content"):
    if not generic_url.strip():
        st.error("Please provide a valid URL")
    elif not validators.url(generic_url):
        st.error("Invalid URL format")
    else:
        try:
            with st.spinner("Fetching and summarizing..."):
                # Load data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    yt_url = normalize_youtube_url(generic_url)
                    docs = fetch_youtube_transcript(yt_url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=True,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                if not docs:
                    st.error("No content could be fetched from the URL.")
                    st.stop()

                # Summarization
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.subheader("Summary:")
                st.write(output_summary)

        except Exception as e:
            st.error(f"Error: {e}")
