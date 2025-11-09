# app.py
"""
Basic safe project:
- Scrape public profile pages (requests + BeautifulSoup)
- Extract basic structured fields (name, headline, location, about, experience)
- Save results to CSV (download button)
- Build local embeddings (sentence-transformers) + FAISS
- RAG: retrieve top chunks and optionally call Google GenAI if GOOGLE_API_KEY set
Important: Do NOT use this to bypass login or create fake accounts.
"""

import os, time, json
from urllib.parse import urlparse
from typing import List, Dict

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd

# embeddings + index
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# optional Google GenAI (Gemini) SDK import guard
try:
    import google.generativeai as genai
    GENAI_OK = True
except Exception:
    GENAI_OK = False

st.set_page_config(page_title="Basic Profile Scraper + RAG", layout="wide")
st.title("Basic Profile Scraper + RAG (Public pages only)")

st.markdown(
    "**Legal note:** This app fetches only publicly accessible pages. Do not attempt to bypass logins or create fake accounts. Use LinkedIn API for authorized access."
)

# -------------------------
# Inputs
# -------------------------
urls_input = st.text_area("Paste public profile URLs (one per line)", height=180,
                          placeholder="https://example.com/profile1\nhttps://example.com/profile2")
user_agent = st.text_input("User-Agent (optional)", value="BasicProfileScraper/1.0")
delay = st.number_input("Delay between requests (s)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=4000, value=800)
chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=200)

st.sidebar.header("Optional: Google GenAI (Gemini)")
api_key = os.getenv("GOOGLE_API_KEY", "")
if api_key and GENAI_OK:
    genai.configure(api_key=api_key)
    st.sidebar.success("Google GenAI available")
elif api_key and not GENAI_OK:
    st.sidebar.warning("GOOGLE_API_KEY set but google.generativeai SDK not installed/imported")
else:
    st.sidebar.info("Set GOOGLE_API_KEY in environment to enable Gemini calls (optional)")

# -------------------------
# Utility functions
# -------------------------
def fetch_html(url: str, ua: str, timeout=12) -> str:
    headers = {"User-Agent": ua} if ua else {}
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.text

def clean_text_from_soup(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    main = soup.find("main")
    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs) if paragraphs else soup.get_text(separator="\n", strip=True)
    return text

def extract_profile(url: str, html: str) -> Dict:
    soup = BeautifulSoup(html, "html.parser")
    # Basic heuristics
    profile = {"url": url, "name": "", "headline": "", "location": "", "about": "", "experience": []}
    # name heuristics
    name_sel = ["h1", ".profile-name", ".pv-top-card--list li", "h1.text-heading-xlarge"]
    for s in name_sel:
        el = soup.select_one(s)
        if el and el.get_text(strip=True):
            profile["name"] = el.get_text(strip=True)
            break
    # headline
    hd_sel = [".headline", ".pv-text-details__left-panel .text-body-medium", "h2"]
    for s in hd_sel:
        el = soup.select_one(s)
        if el and el.get_text(strip=True):
            profile["headline"] = el.get_text(strip=True)
            break
    # location
    loc_sel = [".location", ".pv-text-details__left-panel .text-body-small"]
    for s in loc_sel:
        el = soup.select_one(s)
        if el and el.get_text(strip=True):
            profile["location"] = el.get_text(strip=True)
            break
    # about
    about_sel = ["#about", ".about", ".profile-summary", ".pv-about__summary-text"]
    for s in about_sel:
        el = soup.select_one(s)
        if el and el.get_text(strip=True):
            profile["about"] = el.get_text(separator=" ", strip=True)
            break
    # experience heuristics (collect a few items)
    exp_section = soup.find(id="experience") or soup.find("section", {"class": "experience"})
    if exp_section:
        items = exp_section.find_all("li")
        for li in items[:5]:
            txt = li.get_text(separator=" | ", strip=True)
            if txt:
                profile["experience"].append(txt)
    else:
        # generic fallback find repeated job-like blocks
        for sel in [".experience-item", ".work-experience", ".job", ".position"]:
            els = soup.select(sel)
            for el in els[:5]:
                txt = el.get_text(separator=" | ", strip=True)
                if txt:
                    profile["experience"].append(txt)
    # fallback: fill about with page text if empty
    if not profile["about"]:
        profile["about"] = clean_text_from_soup(soup)[:3000]
    return profile

def chunk_text(text: str, size=800, overlap=200):
    if not text:
        return []
    text = text.replace("\n", " ")
    chunks = []
    stride = max(1, size - overlap)
    i = 0
    L = len(text)
    while i < L:
        chunks.append(text[i: i+size].strip())
        i += stride
    return chunks

# session caches
if "profiles" not in st.session_state:
    st.session_state["profiles"] = []
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "index" not in st.session_state:
    st.session_state["index"] = None
if "embs" not in st.session_state:
    st.session_state["embs"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None

# -------------------------
# Scrape & index
# -------------------------
if st.button("Scrape URLs & build index"):
    st.session_state["profiles"] = []
    st.session_state["chunks"] = []
    raw_urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
    if not raw_urls:
        st.error("Paste at least one URL")
    else:
        session = requests.Session()
        if user_agent:
            session.headers.update({"User-Agent": user_agent})
        progress = st.progress(0)
        for i, url in enumerate(raw_urls, start=1):
            st.write(f"Fetching ({i}/{len(raw_urls)}): {url}")
            try:
                html = fetch_html(url, user_agent)
            except Exception as e:
                st.warning(f"Failed: {e}")
                progress.progress(i/len(raw_urls))
                time.sleep(delay)
                continue
            prof = extract_profile(url, html)
            st.session_state["profiles"].append(prof)
            # chunk page text
            chunks = chunk_text(clean_text_from_soup(BeautifulSoup(html, "html.parser")), size=chunk_size, overlap=chunk_overlap)
            for c in chunks:
                st.session_state["chunks"].append({"text": c, "source": url})
            progress.progress(i/len(raw_urls))
            time.sleep(delay)
        # build embeddings & faiss
        chunks = st.session_state["chunks"]
        if not chunks:
            st.warning("No chunks to index (pages may be JS heavy / blocked).")
        else:
            texts = [c["text"] for c in chunks]
            if st.session_state["model"] is None:
                st.session_state["model"] = SentenceTransformer("all-MiniLM-L6-v2")
            model = st.session_state["model"]
            st.info("Encoding embeddings (this may take a few seconds)...")
            embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            faiss.normalize_L2(embs)
            dim = embs.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embs)
            st.session_state["index"] = index
            st.session_state["embs"] = embs
            st.success(f"Indexed {len(texts)} chunks")

# -------------------------
# Show profiles and download
# -------------------------
st.header("Extracted profiles")
if st.session_state["profiles"]:
    df = pd.DataFrame(st.session_state["profiles"])
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download profiles CSV", csv, file_name="profiles.csv", mime="text/csv")
else:
    st.info("No profiles yet. Scrape public URLs first.")

# -------------------------
# RAG Query
# -------------------------
st.markdown("---")
st.header("RAG Query (local)")

query = st.text_input("Enter a question (will retrieve top-K chunks)")
top_k = st.number_input("Top K", min_value=1, max_value=10, value=3)
if st.button("Retrieve & (optionally) call Gemini"):
    if st.session_state["index"] is None:
        st.error("Index not built. Run scraping step first.")
    elif not query:
        st.error("Enter a query")
    else:
        # embed query
        if st.session_state["model"] is None:
            st.session_state["model"] = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = st.session_state["model"].encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = st.session_state["index"].search(q_emb, top_k)
        retrieved = []
        for idx in I[0]:
            if idx < len(st.session_state["chunks"]):
                retrieved.append(st.session_state["chunks"][idx])
        st.write("Retrieved snippets:")
        for r in retrieved:
            st.write(f"- {r['source']} — {r['text'][:300]}...")
        # compose context
        context = "\n\n---\n\n".join([f"Source: {r['source']}\n{r['text']}" for r in retrieved])
        st.text_area("RAG context (preview)", context[:4000], height=200)
        # call Gemini if available
        if api_key and GENAI_OK:
            try:
                prompt = f"You are a helpful assistant. Use the context below to answer the question. If not found, say 'I don't know'.\n\nCONTEXT:\n{context}\n\nQUESTION: {query}"
                resp = genai.generate_text(model="gemini-1.5", prompt=prompt, max_output_tokens=512, temperature=0.0)
                ans = getattr(resp, "text", None) or str(resp)
                st.subheader("Answer (Gemini)")
                st.write(ans)
            except Exception as e:
                st.error(f"Gemini call failed: {e}")
                st.subheader("Context only (no LLM)")
                st.write(context[:3000])
        else:
            st.info("Gemini not configured or SDK missing; showing context only.")
            st.write(context[:3000])

# -------------------------
# Small debug toggles
# -------------------------
st.markdown("---")
if st.checkbox("Show raw chunks (debug)"):
    st.write(st.session_state["chunks"][:10])
if st.checkbox("Show raw profiles JSON"):
    st.json(st.session_state["profiles"])
