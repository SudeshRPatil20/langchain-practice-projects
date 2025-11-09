import os
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.schema import Document
from urllib.parse import urlparse, parse_qs
import yt_dlp

# ----------- Helper Functions -----------

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
    try:
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,
            language="en",
            download_audio=False
        )
        return loader.load()
    except Exception:
        ydl_opts = {"quiet": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            description = info.get("description", "")
            return [Document(page_content=description, metadata={"title": info.get("title", "")})]

# ----------- Streamlit UI -----------
st.set_page_config(page_title="AI Summarizer & Blog Generator", page_icon="🦜")
st.title("🦜 Langchain + Groq: Summarizer & AI Blog Generator")

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")

generic_url = st.text_input("Enter YouTube or Website URL", placeholder="https://...")

if not groq_api_key:
    st.error("Please enter your Groq API Key")
    st.stop()

# LLM Init
try:
    llm = ChatGroq(
        model="gemma2-9b-it",
        api_key=groq_api_key,
        temperature=0.4,
        max_tokens=1500
    )
except Exception as e:
    st.error(f"LLM Initialization Error: {e}")
    st.stop()

# ----------- Summarizer Section -----------
st.subheader("📌 Summarize a YouTube Video or Article")

prompt_template = """
Provide a clear, structured, easy-to-understand summary (max 300 words) of the following content:

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
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    yt_url = normalize_youtube_url(generic_url)
                    docs_raw = fetch_youtube_transcript(yt_url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=True,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs_raw = loader.load()

                docs = [Document(page_content=item.page_content) for item in docs_raw]

                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.subheader("✅ Summary:")
                st.write(output_summary)

        except Exception as e:
            st.error(f"Error: {e}")

# ----------- Blog Generator Section -----------
st.header("📝 AI Blog Generator")

blog_titles_input = st.text_area(
    "Enter Blog Titles & Short Notes (One per line)",
    placeholder="Example:\nWhat is REST API? | Explain fundamentals.\nReact vs Vue | Compare with use cases.\nMachine Learning Basics | Beginner friendly explanation."
)

if st.button("Generate Blogs"):
    if not blog_titles_input.strip():
        st.error("Please enter at least one blog idea.")
        st.stop()

    blog_entries = [line.strip() for line in blog_titles_input.split("\n") if line.strip()]
    os.makedirs("blog", exist_ok=True)

    with st.spinner("Generating blog articles..."):
        for entry in blog_entries:
            if "|" in entry:
                title, details = entry.split("|", 1)
            else:
                title = entry
                details = ""

            title = title.strip()
            details = details.strip()

            blog_prompt = f"""
Write a detailed technical blog article.

Title: {title}

Guidelines:
- Clear introduction
- Explain core concepts step-by-step
- Add real-world examples
- Add code examples where relevant
- End with a conclusion
Length: 700-1200 words

Additional context:
{details}
"""

            response = llm.predict(blog_prompt)

            filename = f"blog/{title.lower().replace(' ', '_').replace('/', '-')}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n{response}")

        st.success("✅ All blogs generated successfully! Check the /blog folder.")
