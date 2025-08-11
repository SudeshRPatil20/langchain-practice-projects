import validators,streamlit as st 
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

#stremlit app
st.set_page_config(page_title="Langchain: Summerize Text From YT or website", page_icon="🦜️")
st.title("🦜️ Langchain : Summerise text from youtube or website.")
st.subheader('Summarize URL')

### Groq api and url to summerize
with st.sidebar:
    groq_api_key=st.text_input("Groq Api Key", value="", type="password")
    



generic_url=st.text_input("URL",label_visibility="collapsed")

if not groq_api_key:
    st.error("Please Enter Groq Api Key ")
    st.stop()

llm=ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)


    

prompt_template=""" 
Provide a summary of following content in 300 words:
Content:{text}

"""


prompt=PromptTemplate(template=prompt_template, input_variables=["text"])


if st.button("Summerize the content form YT or Website"):
    ##validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can may be YT url or website URL")
    
    else:
        try:
            with st.spinner("Waiting..."):
                ## load the website or yt data
                
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    yt_url = normalize_youtube_url(generic_url)
                    loader = YoutubeLoader.from_youtube_url(
                        yt_url,
                        add_video_info=True,
                        language="en"
                    )
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url], ssl_verify=True,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"})
                docs= loader.load()
                
                ## Chain For Summerization
                chain=load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary=chain.invoke({"input_documents": docs})
                # Display result
                summary_text = output_summary.get("output_text", "").strip()
                if summary_text:
                    st.subheader("Summary:")
                    st.write(summary_text)
                else:
                    st.warning("No summary was generated.")
                
        except Exception as e:

            st.exception(f"Exception:{e}")



