# app.py
"""
Streamlit app: Public Profile Scraper + Structured Extraction + Local RAG (FAISS) + Google GenAI (Gemini)
Do NOT use this to bypass logins or scrape private pages. Use only on public pages you are authorized to access.
"""

import os
import time
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup

# Embeddings & vector DB
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Google GenAI (Gemini) SDK
# pip package name may vary; earlier examples used `google-genai` / `google.generativeai`.
# We'll attempt to import `google.generativeai` and fall back gracefully.
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# -----------------------
# UI Layout & Config
# -----------------------
st.set_page_config(page_title="Profile Scraper + RAG + Gemini", layout="wide")
st.title("Profile Scraper + RAG (FAISS) + Google GenAI (Gemini) — Public pages only")
st.markdown(
    """
**Important**: This app only fetches pages your HTTP client can already access (public pages).
**Do not** use it to bypass logins, automate account creation, or crawl private content.  
If you need authorized access to LinkedIn data, use LinkedIn's official API.
"""
)

# -----------------------
# Inputs
# -----------------------
st.sidebar.header("Scraper / Indexing")
urls_text = st.sidebar.text_area(
    "Profile URLs (one per line)",
    height=200,
    placeholder="https://www.linkedin.com/in/someone/\nhttps://example.com/person-profile",
)
user_agent = st.sidebar.text_input(
    "User-Agent header (optional)", value="ProfileRAG/1.0 (+https://example.com)"
)
delay = st.sidebar.number_input("Delay between requests (seconds)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Chunking / Embedding")
chunk_size = st.sidebar.number_input("Chunk size (chars)", min_value=200, max_value=4000, value=800, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=200, step=50)

st.sidebar.markdown("---")
st.sidebar.header("Google GenAI (optional)")
st.sidebar.info("Set environment var GOOGLE_API_KEY to enable calls to Google GenAI (Gemini).")
api_key = os.getenv("GOOGLE_API_KEY", "")
if api_key and GENAI_AVAILABLE:
    genai.configure(api_key=api_key)
elif api_key and not GENAI_AVAILABLE:
    st.sidebar.warning("GOOGLE_API_KEY found but `google.generativeai` SDK not installed/imported.")
else:
    st.sidebar.info("GOOGLE_API_KEY not set — LLM calls disabled.")

# -----------------------
# Helpers: scraping & extraction
# -----------------------
def fetch_page(url: str, ua: str, timeout=12) -> requests.Response:
    headers = {"User-Agent": ua} if ua else {}
    r = requests.get(url, timeout=timeout, headers=headers)
    return r

def extract_textual_content(soup: BeautifulSoup) -> str:
    # Remove unwanted tags and then collect text from main/article/paragraphs
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    pieces = []
    main = soup.find("main")
    if main:
        pieces.append(main.get_text(separator="\n", strip=True))
    article = soup.find("article")
    if article:
        pieces.append(article.get_text(separator="\n", strip=True))
    # fallback paragraphs
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    if paragraphs:
        pieces.append("\n".join(paragraphs))
    # title
    if soup.title and soup.title.string:
        pieces.append(soup.title.string.strip())
    return "\n\n".join([p for p in pieces if p]).strip()

def parse_json_ld_person(soup: BeautifulSoup) -> Dict[str, Any]:
    # JSON-LD often contains Person schema
    results = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
        except Exception:
            continue
        # data can be list or dict
        items = data if isinstance(data, list) else [data]
        for item in items:
            typ = item.get("@type") if isinstance(item, dict) else None
            if typ == "Person" or (isinstance(typ, list) and "Person" in typ):
                # extract common fields if present
                results["name"] = item.get("name") or results.get("name")
                results["headline"] = item.get("jobTitle") or results.get("headline")
                works = item.get("worksFor") or item.get("affiliation")
                if isinstance(works, dict):
                    results["company"] = works.get("name")
                results["description"] = item.get("description") or results.get("description")
                # location can be nested
                address = item.get("address")
                if isinstance(address, dict):
                    addr = ", ".join([v for k,v in address.items() if isinstance(v,str)])
                    results["location"] = addr
    return results

def extract_profile_fields(url: str, html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    data = {"url": url, "name": "", "headline": "", "location": "", "about": "", "current_role": "", "experience": [], "education": []}
    # JSON-LD fallback
    jsonld = parse_json_ld_person(soup)
    if jsonld:
        for k,v in jsonld.items():
            if k == "description":
                data["about"] = data["about"] or v
            else:
                data[k] = data.get(k) or v

    # Heuristics: LinkedIn-ish selectors & common markup
    # Name candidates
    name_selectors = [
        "h1", "div.ph5 h1", ".pv-text-details__left-panel h1", "li.inline.t-24.t-black.t-normal.break-words",
        "h1.text-heading-xlarge"
    ]
    for sel in name_selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            data["name"] = el.get_text(strip=True)
            break

    # Headline / summary candidates
    headline_selectors = [
        ".pv-text-details__left-panel .text-body-medium", ".text-body-medium.break-words",
        ".pv-top-card h2", ".profile-headline", "h2"
    ]
    for sel in headline_selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            data["headline"] = el.get_text(strip=True)
            break

    # Location
    loc_selectors = [
        ".pv-text-details__left-panel .text-body-small", ".text-body-small.inline.t-black--light.break-words",
        ".locality", ".location"
    ]
    for sel in loc_selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            data["location"] = el.get_text(strip=True)
            break

    # About / summary
    about_selectors = [
        "#about", ".pv-about__summary-text", ".about", ".summary", ".profile-summary"
    ]
    for sel in about_selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            data["about"] = el.get_text(separator=" ", strip=True)
            break

    # Experience – try to locate experience section and collect first few roles
    # LinkedIn experience uses id 'experience' or 'experience-section' or lists of experience items
    exp_root = soup.find(id="experience") or soup.find(id="experience-section") or soup.find("section", {"class": "experience"})
    if exp_root:
        items = exp_root.find_all("li")
        if not items:
            items = exp_root.select(".pv-entity__position-group-pager .pv-entity__summary-info")
        # collect up to 5 experience text blobs
        for li in items[:6]:
            txt = li.get_text(separator=" | ", strip=True)
            if txt:
                data["experience"].append(txt)
    else:
        # generic fallback: search for repeated role-like elements
        roles = soup.select(".experience-item, .work-experience, .job, .position")
        for r in roles[:6]:
            txt = r.get_text(separator=" | ", strip=True)
            if txt:
                data["experience"].append(txt)

    # education
    edu_root = soup.find(id="education") or soup.find(id="education-section") or soup.select_one(".education")
    if edu_root:
        edu_items = edu_root.find_all("li")
        for e in edu_items[:6]:
            txt = e.get_text(separator=" | ", strip=True)
            if txt:
                data["education"].append(txt)

    # If current_role empty, try to derive from experience first item
    if not data["current_role"] and data["experience"]:
        data["current_role"] = data["experience"][0]

    # As final fallback, use textual extraction for about if still empty
    if not data["about"]:
        data["about"] = extract_textual_content(soup)[:3000]  # keep reasonable length

    # Trim strings
    for k in ["name", "headline", "location", "about", "current_role"]:
        if isinstance(data.get(k), str):
            data[k] = data[k].strip()

    return data

# -----------------------
# Chunking + Indexing
# -----------------------
def chunk_text(text: str, size: int = 800, overlap: int = 200) -> List[str]:
    if not text:
        return []
    text = text.replace("\r", " ").replace("\n", " ")
    chunks = []
    start = 0
    L = len(text)
    stride = max(1, size - overlap)
    while start < L:
        end = min(start + size, L)
        c = text[start:end].strip()
        if c:
            chunks.append(c)
        if end == L:
            break
        start += stride
    return chunks

# -----------------------
# Session state init
# -----------------------
if "profiles" not in st.session_state:
    st.session_state.profiles = []  # list of extracted profile dicts
if "chunks" not in st.session_state:
    st.session_state.chunks = []  # list of {"text":..., "source":...}
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "emb_model" not in st.session_state:
    st.session_state.emb_model = None

# -----------------------
# Scrape & Index button
# -----------------------
if st.sidebar.button("Scrape URLs & Build Index"):
    raw_urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    if not raw_urls:
        st.sidebar.error("Please paste at least one URL (public pages only).")
    else:
        st.info(f"Fetching {len(raw_urls)} URL(s) — respecting delay and ToS.")
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})
        robots_texts = {}
        all_chunks = []
        extracted_profiles = []
        progress = st.progress(0)
        for i, url in enumerate(raw_urls, start=1):
            st.write(f"Fetching ({i}/{len(raw_urls)}): {url}")
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            if robots_url not in robots_texts:
                try:
                    rb = session.get(robots_url, timeout=6)
                    robots_texts[robots_url] = rb.text[:4000]
                except Exception:
                    robots_texts[robots_url] = "<unavailable>"
            try:
                resp = session.get(url, timeout=15)
            except Exception as e:
                st.warning(f"Failed to fetch {url}: {e}")
                progress.progress(i / len(raw_urls))
                time.sleep(delay)
                continue

            if resp.status_code != 200:
                st.warning(f"Status {resp.status_code} for {url} — skipped.")
                progress.progress(i / len(raw_urls))
                time.sleep(delay)
                continue

            html = resp.text
            # extract structured profile fields
            profile = extract_profile_fields(url, html)
            extracted_profiles.append(profile)

            # full textual content and chunk it
            soup = BeautifulSoup(html, "html.parser")
            full_text = extract_textual_content(soup)
            if full_text:
                chunks = chunk_text(full_text, size=int(chunk_size), overlap=int(chunk_overlap))
                for c in chunks:
                    all_chunks.append({"text": c, "source": url})
            progress.progress(i / len(raw_urls))
            time.sleep(delay)

        # Save into session
        st.session_state.profiles = extracted_profiles
        st.session_state.chunks = all_chunks

        if not all_chunks:
            st.warning("No textual chunks extracted; indexing skipped.")
        else:
            st.success(f"Extracted {len(extracted_profiles)} profiles and {len(all_chunks)} chunks. Building embeddings...")

            # load embedding model (once)
            if st.session_state.emb_model is None:
                st.session_state.emb_model = SentenceTransformer("all-MiniLM-L6-v2")
            model = st.session_state.emb_model
            texts = [c["text"] for c in all_chunks]
            # compute embeddings
            embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            # normalize embeddings and build FAISS index
            emb_dim = embeddings.shape[1]
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(emb_dim)
            index.add(embeddings)
            st.session_state.index = index
            st.session_state.embeddings = embeddings
            st.success("FAISS index built and stored in session state.")

        # show (cached) robots.txt of first domain for transparency
        if robots_texts:
            first_rb = next(iter(robots_texts.items()))
            st.text_area("robots.txt (sample)", value=first_rb[1], height=180)

# -----------------------
# Show extracted profiles and allow CSV download
# -----------------------
st.markdown("---")
st.header("Extracted Profiles")
if st.session_state.profiles:
    df_profiles = pd.DataFrame(st.session_state.profiles)
    st.dataframe(df_profiles)
    csv_bytes = df_profiles.to_csv(index=False).encode("utf-8")
    st.download_button("Download Profiles CSV", csv_bytes, file_name="profiles.csv", mime="text/csv")
else:
    st.info("No profiles extracted yet. Click 'Scrape URLs & Build Index' in the sidebar after entering URLs.")

# -----------------------
# RAG Q/A / Summarize
# -----------------------
st.markdown("---")
st.header("RAG Query / Profile Summaries")

query = st.text_input("Ask a question (will use RAG) or enter 'summarize: <url>' to summarize a profile", key="query_input")
top_k = st.number_input("Top K retrieved chunks", min_value=1, max_value=10, value=4, step=1)

if st.button("Retrieve & Generate Answer"):
    if not query:
        st.error("Enter a question or 'summarize: <url>'.")
    elif st.session_state.index is None or not st.session_state.chunks:
        st.error("No index available — scrape & build the index first.")
    else:
        # embed query
        if st.session_state.emb_model is None:
            st.session_state.emb_model = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = st.session_state.emb_model.encode([query], convert_to_numpy=True)
        # normalize and search
        faiss.normalize_L2(q_emb)
        D, I = st.session_state.index.search(q_emb, top_k)
        retrieved = []
        for idx in I[0]:
            if idx < len(st.session_state.chunks):
                retrieved.append(st.session_state.chunks[idx])

        st.write("### Retrieved passages (short preview)")
        for r in retrieved:
            st.write(f"- Source: {r['source']}")
            st.write(r['text'][:800] + ("..." if len(r['text'])>800 else ""))

        # Compose context for LLM
        context = "\n\n---\n\n".join([f"Source: {r['source']}\n\n{r['text']}" for r in retrieved])
        system_prompt = (
            "You are an assistant that answers questions only from the supplied context. "
            "If the information is not in the context, say you don't know. Provide concise answers and cite source URLs."
        )
        prompt = f"{system_prompt}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer succinctly and include source URLs."

        # Call Google GenAI if available and configured
        if api_key and GENAI_AVAILABLE:
            try:
                # NOTE: SDK usage may vary across versions. This is a typical call shape.
                resp = genai.generate_text(model="gemini-1.5", prompt=prompt, max_output_tokens=512, temperature=0.0)
                # response may be an object; extract textual content
                answer = getattr(resp, "text", None) or getattr(resp, "result", None) or str(resp)
                st.subheader("Generated answer (Gemini)")
                st.write(answer)
            except Exception as e:
                st.error(f"Gemini call failed: {e}")
                st.info("Check GOOGLE_API_KEY, installed genai SDK version, and model name. Falling back to simple concatenated context.")
                # fallback: show concatenated context
                st.subheader("Fallback: Concatenated context (no LLM)")
                st.write(context[:4000] + ("..." if len(context)>4000 else ""))
        else:
            st.warning("Google GenAI not configured or SDK not installed — LLM generation disabled.")
            st.subheader("Concatenated context (LLM disabled)")
            st.write(context[:4000] + ("..." if len(context)>4000 else ""))

# -----------------------
# Per-profile summarize button (single-click)
# -----------------------
st.markdown("---")
st.header("Summarize a specific profile (optional)")

profile_url = st.text_input("Enter profile URL to summarize")
if st.button("Summarize profile"):
    if not profile_url:
        st.error("Enter a profile URL")
    else:
        # find profile in session_state
        found = None
        for p in st.session_state.profiles:
            if p.get("url") == profile_url:
                found = p
                break
        if not found:
            st.error("Profile not found in extracted results. Make sure you scraped it first.")
        else:
            # create a summary prompt using extracted fields
            summary_prompt = (
                "You are a helpful assistant. Create a concise professional summary of the person using the fields provided. "
                "Include name, top current role, 2-sentence career summary, and 2 short bullets of notable experience/education if available. "
                "Do not invent facts — if something is missing, say 'not provided'.\n\n"
                f"DATA:\nName: {found.get('name')}\nHeadline: {found.get('headline')}\nLocation: {found.get('location')}\nAbout: {found.get('about')[:1200]}\n"
                f"Current role: {found.get('current_role')}\nExperience (first 3): {found.get('experience')[:3]}\nEducation (first 3): {found.get('education')[:3]}\n\n"
                "Write the summary now."
            )
            if api_key and GENAI_AVAILABLE:
                try:
                    resp = genai.generate_text(model="gemini-1.5", prompt=summary_prompt, max_output_tokens=300, temperature=0.0)
                    ans = getattr(resp, "text", None) or getattr(resp, "result", None) or str(resp)
                    st.subheader("Profile Summary (Gemini)")
                    st.write(ans)
                except Exception as e:
                    st.error(f"Gemini call failed: {e}")
                    st.subheader("Fallback summary from fields")
                    st.write(f"Name: {found.get('name')}\nHeadline: {found.get('headline')}\nAbout: {found.get('about')[:800]}")
            else:
                st.warning("Gemini not configured or SDK unavailable. Showing fields as summary.")
                st.write(f"Name: {found.get('name')}\nHeadline: {found.get('headline')}\nAbout: {found.get('about')[:1200]}")

# -----------------------
# Debugging / developer utilities
# -----------------------
st.markdown("---")
st.header("Debug / Developer tools")
if st.checkbox("Show raw indexed chunk count and sample"):
    st.write("Chunks indexed:", len(st.session_state.chunks))
    if st.session_state.chunks:
        st.write(st.session_state.chunks[:3])

if st.checkbox("Show raw profiles JSON"):
    st.json(st.session_state.profiles)

st.caption("This tool is educational/demo-grade. For production, use: robust HTML extraction, handling JS-rendered pages (ethically), authorized APIs, rate-limiting, retries, and legal review.")
