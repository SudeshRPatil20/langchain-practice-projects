# app.py
import os
import re
import time
import streamlit as st
import google.generativeai as genai
from datetime import datetime
from pathlib import Path
import unicodedata
import hashlib

BLOG_DIR = Path("blog")
BLOG_DIR.mkdir(exist_ok=True)


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[-\s]+", "-", text)
    if not text:
        text = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return text


def prompt_template(title, details):
    return f"""
Write a full high-quality blog article titled: "{title}"

Requirements:
- Tone: professional but friendly
- Start with a short summary
- Add multiple sections with headings
- Use code examples where useful
- End with a conclusion and optional extra resources
- Insert bullet points where helpful
- If details provided: {details}

Format output in Markdown starting with:
# {title}
"""


def save_markdown(title, body):
    date = datetime.utcnow().strftime("%Y-%m-%d")
    slug = slugify(title)
    path = BLOG_DIR / f"{slug}.md"
    fm = f"---\ntitle: \"{title}\"\ndate: {date}\nauthor: AutoGen\ntags: programming, ai, tutorial\n---\n\n"
    path.write_text(fm + body, encoding="utf-8")
    return path


def call_gemini(api_key, prompt, model_name="gemini-flash-latest"):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text


st.set_page_config(page_title="AI Blog Generator", layout="centered")
st.title("📝 AI Blog Article Generator (Google Gemini API)")
st.write("Enter article titles below (one per line). Optionally add: `Title | notes`")

api_key = st.text_input("Enter Google Gemini API Key:", type="password")

titles_input = st.text_area("Titles", height=250, value=
"""Building REST APIs with FastAPI | include examples
Understanding Transformers in Deep Learning | diagrams + easy explanation
Optimizing SQL Queries | explain indexes and EXPLAIN plan
""")

temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.05)

generate = st.button("Generate Articles 🚀")

def parse_titles(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    parsed = []
    for line in lines:
        if "|" in line:
            t, d = map(str.strip, line.split("|", 1))
        else:
            t, d = line, ""
        parsed.append((t, d))
    return parsed


if generate:
    if not api_key:
        st.error("Please enter your Gemini API Key.")
    else:
        titles = parse_titles(titles_input)
        progress = st.progress(0)
        done = []

        for i, (title, details) in enumerate(titles, start=1):
            st.write(f"✍️ Generating: **{title}** ...")
            try:
                prompt = prompt_template(title, details)
                output = call_gemini(api_key, prompt)
                filepath = save_markdown(title, output)
                done.append(filepath)
                st.success(f"✅ Saved: {filepath}")
            except Exception as e:
                st.error(f"❌ Failed for {title}: {e}")
            progress.progress(i / len(titles))
            time.sleep(0.2)

        st.success("🎯 All done!")
        st.write("Generated files:")
        for f in done:
            st.write("-", f"`{f}`")


st.write("---")
st.header("📂 Existing Blog Files")
files = sorted(BLOG_DIR.glob("*.md"))
for f in files:
    st.write("-", f.name)

