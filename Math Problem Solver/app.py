# app.py
"""
Streamlit app: Blog article generator using LLMs (OpenAI or Google Gemini/Vertex AI).
- Paste titles (one per line). Optionally: "Title | details or tone".
- Click Generate -> saves markdown files in ./blog
- Has "Generate 10 sample articles" quick button.
"""

import os
import re
import time
import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import List
import unicodedata
import hashlib

# Optional LLM libs (comment/uncomment depending on which provider you use)
# For OpenAI (recommended default):
# pip install openai
try:
    import openai
except Exception:
    openai = None

# For Google Vertex AI (Gemini) -- optional example:
# pip install google-cloud-aiplatform
try:
    from google.cloud import aiplatform
except Exception:
    aiplatform = None

# --------------------------
# Helpers
# --------------------------
BLOG_DIR = Path("blog")
BLOG_DIR.mkdir(exist_ok=True)

def slugify(text: str) -> str:
    # Simple slugify: remove accents, non-alnum -> -, trim duplicates
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[-\s]+", "-", text)
    if not text:
        # fallback to hash
        text = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return text

def default_prompt_template(title: str, details: str) -> str:
    # Prompt that will be sent to the LLM. Editable in UI.
    return (
        f"You are an expert software engineer and technical writer. "
        f"Write a detailed technical article for developers titled: \"{title}\".\n\n"
        f"Constraints:\n"
        f"- Tone: professional and approachable.\n"
        f"- Include a 2-3 sentence summary, a clear structure with headings, code snippets where useful, "
        f"explanations, pros/cons or gotchas, and a short conclusion with further resources.\n"
        f"- Target audience: intermediate developers who know basics but want practical guidance.\n"
        f"- If additional details are provided: {details}\n\n"
        f"Output format:\n"
        f"# {title}\n\n"
        f"Summary:\n\n"
        f"Body (with headings and code blocks):\n\n"
        f"Conclusion:\n\n"
        f"Tags: (comma separated list)\n"
    )

def save_markdown(title: str, body: str, tags: List[str], author: str = "AutoGen", date: str = None):
    if date is None:
        date = datetime.utcnow().strftime("%Y-%m-%d")
    slug = slugify(title)
    filename = BLOG_DIR / f"{slug}.md"
    front_matter = f"---\ntitle: \"{title}\"\ndate: {date}\nauthor: {author}\ntags: {tags}\n---\n\n"
    filename.write_text(front_matter + body, encoding="utf-8")
    return filename

# --------------------------
# LLM wrappers
# --------------------------
def call_openai_completion(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 1200, temperature: float = 0.2):
    """Call OpenAI via openai library. Make sure OPENAI_API_KEY is set in env."""
    if openai is None:
        raise RuntimeError("openai library not installed. pip install openai")
    # You may also choose a model name available to you (gpt-4o-mini, gpt-4o, gpt-4, gpt-3.5-turbo, etc.)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable.")
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that writes high-quality technical articles."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
    )
    # Adapt depending on library version
    text = response.choices[0].message["content"] if hasattr(response.choices[0], "message") else response.choices[0].text
    return text

def call_gemini_vertex(prompt: str, model: str = "projects/PROJECT/locations/LOCATION/models/gemini-pro", max_tokens: int = 1024, temperature: float = 0.2):
    """
    Example Vertex AI (Google Gemini) call.
    NOTE: This function is illustrative. You must set GOOGLE_APPLICATION_CREDENTIALS and configure your project.
    Replace 'projects/PROJECT/locations/LOCATION/models/gemini-pro' with the full resource name of your model.
    """
    if aiplatform is None:
        raise RuntimeError("google-cloud-aiplatform not installed. pip install google-cloud-aiplatform")
    project = os.getenv("GOOGLE_PROJECT")
    location = os.getenv("GOOGLE_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("Set GOOGLE_PROJECT environment variable to your GCP project id.")
    # initialize aiplatform SDK (may be optional depending on auth)
    aiplatform.init(project=project, location=location)
    # The exact invocation depends on the version of Vertex AI SDK and available APIs.
    # This example uses the "Prediction" call for text generation — you may need to adapt to your model's API.
    endpoint = model  # user should pass the full model resource name
    # NOTE: For many setups you will use the Vertex AI TextGenerationModel class:
    try:
        TextGenModel = aiplatform.gapic.PredictionServiceClient  # placeholder
        # The exact code below is intentionally brief — adapt from Vertex AI docs.
        raise NotImplementedError("Adapt call_gemini_vertex to your Vertex AI client per your SDK version.")
    except Exception as e:
        raise RuntimeError("Please follow Vertex AI (Gemini) docs to call your model. Exception: " + str(e))

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="AI Blog Generator", layout="centered")

st.title("AI Blog Generator — Streamlit")
st.write("Paste a list of article titles (one per line). Optional details after a `|` — e.g. `Title | tone: casual, include code`.")

with st.expander("How it works (short)"):
    st.markdown(
        """
- Enter titles (one per line). Example: `Building a Fast REST API with FastAPI | include benchmarks`  
- Choose provider & fill API credentials.  
- Click Generate: the app will call the LLM and save each article as a Markdown file in `./blog`.  
- Use `Generate 10 sample articles` to quickly create a starter set.
"""
    )

col1, col2 = st.columns([3, 1])
with col1:
    titles_input = st.text_area("Article titles (one per line). Optional: `Title | notes`", height=220, value=(
        "Building Fast APIs with FastAPI | include code and deployment notes\n"
        "Practical Guide to Unit Testing in Python | examples with pytest\n"
        "Introduction to Transformer Architectures | explain intuition\n"
        "Docker for Developers | quickstart and best practices\n"
        "Efficient Data Processing with Pandas | performance tips\n"
        "Production ML Monitoring | metrics, drift detection\n"
        "Clean Architecture for Node.js | project skeleton\n"
        "Version Control Workflows | git branching strategies\n"
        "Optimizing SQL queries | indexes, EXPLAIN\n"
        "WebSockets vs HTTP/2 | when to use which"
    ))
with col2:
    st.write("Quick actions")
    if st.button("Generate 10 sample articles and save to ./blog"):
        # use the example titles above
        titles_input = st.session_state.get("titles_input", titles_input)
        st.success("Sample list loaded into titles area. Click 'Generate' to create them.")

provider = st.selectbox("LLM Provider", ["OpenAI (default)", "Google Gemini / Vertex AI (example)"])
model = st.text_input("Model name (provider-specific)", value="gpt-4o-mini" if provider.startswith("OpenAI") else "projects/PROJECT/locations/LOCATION/models/gemini-pro")
openai_key = st.text_input("OPENAI_API_KEY (leave blank to use environment variable)", type="password")
gcp_creds_path = st.text_input("Path to GOOGLE service account JSON (optional)", type="password")

temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.2, 0.05)
max_tokens = st.slider("Max output tokens (approx)", 200, 5000, 1200, 100)

prompt_template_area = st.text_area("Prompt template (editable)", value=default_prompt_template("{title}", "{details}"), height=200)

author_name = st.text_input("Author name for saved files", value="AutoGen")
tags_default = st.text_input("Default tags (comma-separated)", value="programming, tutorial")

generate_btn = st.button("Generate articles now")

# Helper to parse titles_input
def parse_lines(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parsed = []
    for line in lines:
        if "|" in line:
            title, details = map(str.strip, line.split("|", 1))
        else:
            title, details = line, ""
        parsed.append((title, details))
    return parsed

if generate_btn:
    # Save provided keys to env if user entered them (temporary for this run)
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if gcp_creds_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_creds_path

    titles = parse_lines(titles_input)
    if not titles:
        st.error("No titles provided.")
    else:
        results = []
        errors = []
        progress_bar = st.progress(0)
        total = len(titles)
        for i, (title, details) in enumerate(titles, start=1):
            prompt = prompt_template_area.replace("{title}", title).replace("{details}", details)
            st.info(f"Generating: {title}")
            try:
                if provider.startswith("OpenAI"):
                    # call OpenAI wrapper
                    generated = call_openai_completion(prompt=prompt, model=model, max_tokens=max_tokens, temperature=temperature)
                else:
                    # Provider = Google Gemini (Vertex AI) example
                    generated = call_gemini_vertex(prompt=prompt, model=model, max_tokens=max_tokens, temperature=temperature)
                # Try to auto-extract tags line (if model appended "Tags: ..." at end)
                tags_line = []
                # naive tag extraction:
                m = re.search(r"Tags[:\-]?\s*(.*)$", generated, re.IGNORECASE | re.MULTILINE)
                if m:
                    tags_line = [t.strip() for t in re.split(r"[,;]+", m.group(1)) if t.strip()]
                if not tags_line:
                    tags_line = [t.strip() for t in tags_default.split(",") if t.strip()]

                filepath = save_markdown(title=title, body=generated, tags=tags_line, author=author_name)
                results.append((title, filepath))
                st.success(f"Saved: {filepath}")
            except Exception as e:
                errors.append((title, str(e)))
                st.error(f"Failed to generate '{title}': {e}")
            progress_bar.progress(i / total)
            # polite pause to avoid rate limits (user can adjust/remove)
            time.sleep(0.5)

        st.write("### Generation complete")
        st.write(f"Saved {len(results)} files to `{BLOG_DIR}`")
        if results:
            for t, p in results:
                st.markdown(f"- **{t}** → `{p}`")
        if errors:
            st.write("### Errors")
            for t, err in errors:
                st.markdown(f"- **{t}**: {err}")

# Show existing blog files
st.markdown("---")
st.header("Current ./blog files")
md_files = sorted(BLOG_DIR.glob("*.md"), key=os.path.getmtime, reverse=True)
if not md_files:
    st.info("No markdown files in ./blog yet.")
else:
    for f in md_files[:50]:
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"- `{f.name}` — last modified {mtime} — size: {f.stat().st_size} bytes")
        if st.button(f"Preview: {f.name}", key=f"preview_{f.name}"):
            st.code(f.read_text(encoding="utf-8"), language="markdown")

st.markdown("---")
st.caption("This app stores generated articles as Markdown files in the local ./blog folder. Customize the prompt and provider in the UI.")
