# app.py
"""
Streamlit app with a dedicated LinkedIn profiles input + scraping.
Only fetches public pages. Will not bypass login.
"""

import os
import time
import json
from urllib.parse import urlparse
from typing import List, Dict, Any

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd

# embeddings + index
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# optional Google GenAI (Gemini)
try:
    import google.generativeai as genai
    GENAI_OK = True
except Exception:
    GENAI_OK = False

st.set_page_config(page_title="Profile Scraper + LinkedIn Section", layout="wide")
st.title("Profile Scraper — Add LinkedIn profiles (public only)")

st.markdown(
    """
**Important:** This tool only scrapes pages your HTTP client can already access.  
**Do not** use it to bypass login, create fake/test accounts, or break LinkedIn's Terms of Service.
"""
)

# -------------------------
# Controls
# -------------------------
st.sidebar.header("Scraper settings")
user_agent = st.sidebar.text_input("User-Agent header", value="ProfileRAG/1.0 (+https://example.com)")
delay = st.sidebar.number_input("Delay between requests (seconds)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Chunking / Embedding")
chunk_size = st.sidebar.number_input("Chunk size (chars)", min_value=200, max_value=4000, value=800)
chunk_overlap = st.sidebar.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=200)

st.sidebar.markdown("---")
st.sidebar.header("Google GenAI (optional)")
api_key = os.getenv("GOOGLE_API_KEY", "")
if api_key and GENAI_OK:
    genai.configure(api_key=api_key)
    st.sidebar.success("Google GenAI configured")
elif api_key and not GENAI_OK:
    st.sidebar.warning("GOOGLE_API_KEY set but google.generativeai SDK not available")
else:
    st.sidebar.info("Set GOOGLE_API_KEY to enable Gemini (optional)")

# -------------------------
# Utility functions
# -------------------------
def fetch_html(url: str, ua: str, timeout: int = 12) -> requests.Response:
    headers = {"User-Agent": ua} if ua else {}
    resp = requests.get(url, headers=headers, timeout=timeout)
    return resp

def looks_like_login_page(html_text: str, resp_url: str) -> bool:
    t = html_text.lower()
    checks = [
        "sign in", "sign in to linkedin", "please sign in", "join linkedin", "login to linkedin",
        "/uas/login", "login?trk", "authwall", "login.linkedin.com", "signin"
    ]
    # also check redirected url
    if any(x in resp_url.lower() for x in ["login", "auth", "signin"]):
        return True
    for c in checks:
        if c in t:
            return True
    return False

def clean_text_from_soup(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    main = soup.find("main")
    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        paragraphs = soup.find_all("p")
        if paragraphs:
            text = "\n".join(p.get_text(strip=True) for p in paragraphs)
        else:
            text = soup.get_text(separator="\n", strip=True)
    return text

def parse_json_ld_person(soup: BeautifulSoup) -> Dict[str, Any]:
    results = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            typ = item.get("@type")
            if typ == "Person" or (isinstance(typ, list) and "Person" in typ):
                results["name"] = item.get("name") or results.get("name")
                results["headline"] = item.get("jobTitle") or results.get("headline")
                results["description"] = item.get("description") or results.get("description")
                works = item.get("worksFor") or item.get("affiliation")
                if isinstance(works, dict):
                    results["company"] = works.get("name") or results.get("company")
                address = item.get("address")
                if isinstance(address, dict):
                    results["location"] = ", ".join([v for k,v in address.items() if isinstance(v, str)])
    return results

def extract_profile_fields(url: str, html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    data = {"url": url, "name": "", "headline": "", "location": "", "about": "", "current_role": "", "experience": [], "status": "ok"}
    # JSON-LD
    jsonld = parse_json_ld_person(soup)
    if jsonld:
        data["name"] = data["name"] or jsonld.get("name", "")
        data["headline"] = data["headline"] or jsonld.get("headline", "")
        data["about"] = data["about"] or jsonld.get("description", "")
        data["location"] = data["location"] or jsonld.get("location", "")

    # Heuristics for LinkedIn-like / profile-like pages
    name_selectors = ["h1", ".pv-text-details__left-panel h1", "h1.text-heading-xlarge", ".profile-name"]
    for sel in name_selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            data["name"] = data["name"] or el.get_text(strip=True)
            break

    headline_selectors = [".text-body-medium.break-words", ".pv-text-details__left-panel .text-body-medium", ".profile-headline", "h2"]
    for sel in headline_selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            data["headline"] = data["headline"] or el.get_text(strip=True)
            break

    loc_selectors = [".pv-text-details__left-panel .text-body-small", ".locality", ".location"]
    for sel in loc_selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            data["location"] = data["location"] or el.get_text(strip=True)
            break

    about_selectors = ["#about", ".pv-about__summary-text", ".about", ".summary", ".profile-summary"]
    for sel in about_selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            data["about"] = data["about"] or el.get_text(separator=" ", strip=True)
            break

    # Experience: try several fallbacks
    exp_root = soup.find(id="experience") or soup.find(id="experience-section") or soup.find("section", {"class": "experience"})
    if exp_root:
        items = exp_root.find_all("li")
        for li in items[:6]:
            txt = li.get_text(separator=" | ", strip=True)
            if txt:
                data["experience"].append(txt)
    else:
        roles = soup.select(".experience-item, .pv-entity__position-group-pager li, .work-experience, .job, .position")
        for r in roles[:6]:
            txt = r.get_text(separator=" | ", strip=True)
            if txt:
                data["experience"].append(txt)

    if not data["about"]:
        data["about"] = clean_text_from_soup(soup)[:3000]

    if not data["current_role"] and data["experience"]:
        data["current_role"] = data["experience"][0]

    # final trimming
    for k in ["name", "headline", "location", "about", "current_role"]:
        if isinstance(data.get(k), str):
            data[k] = data[k].strip()

    return data

def chunk_text(text: str, size: int = 800, overlap: int = 200) -> List[str]:
    if not text:
        return []
    text = text.replace("\n", " ")
    chunks = []
    stride = max(1, size - overlap)
    start = 0
    L = len(text)
    while start < L:
        end = min(start + size, L)
        chunks.append(text[start:end].strip())
        if end == L:
            break
        start += stride
    return chunks

# -------------------------
# Session caches
# -------------------------
if "profiles" not in st.session_state:
    st.session_state["profiles"] = []
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "index" not in st.session_state:
    st.session_state["index"] = None
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = None
if "emb_model" not in st.session_state:
    st.session_state["emb_model"] = None

# -------------------------
# Main UI - LinkedIn section
# -------------------------
st.header("LinkedIn profiles (paste public LinkedIn profile URLs here)")
st.markdown(
    "Paste LinkedIn profile URLs (one per line). **If profiles require login in your browser they will likely be marked `requires_login`.**"
)
linkedin_urls_text = st.text_area("LinkedIn profile URLs (one per line)", height=180, placeholder="https://www.linkedin.com/in/someone/\nhttps://www.linkedin.com/in/another/")

if st.button("Scrape LinkedIn profiles"):
    raw = [u.strip() for u in linkedin_urls_text.splitlines() if u.strip()]
    if not raw:
        st.error("Paste at least one LinkedIn URL.")
    else:
        session = requests.Session()
        if user_agent:
            session.headers.update({"User-Agent": user_agent})
        progress = st.progress(0)
        new_profiles = []
        new_chunks = []
        for i, url in enumerate(raw, start=1):
            st.write(f"Fetching ({i}/{len(raw)}): {url}")
            try:
                resp = session.get(url, timeout=15)
            except Exception as e:
                st.warning(f"Failed to fetch {url}: {e}")
                profile = {"url": url, "status": "fetch_error", "error": str(e)}
                new_profiles.append(profile)
                progress.progress(i/len(raw))
                time.sleep(delay)
                continue

            if resp.status_code != 200:
                st.warning(f"Status {resp.status_code} for {url}")
                profile = {"url": url, "status": f"status_{resp.status_code}"}
                new_profiles.append(profile)
                progress.progress(i/len(raw))
                time.sleep(delay)
                continue

            html = resp.text
            # detect login/protected page
            if looks_like_login_page(html, resp.url):
                st.warning(f"Page appears to require login / redirect for {url} — marked requires_login.")
                profile = {"url": url, "status": "requires_login"}
                new_profiles.append(profile)
                progress.progress(i/len(raw))
                time.sleep(delay)
                continue

            # extract fields
            prof = extract_profile_fields(url, html)
            prof["status"] = "ok"
            new_profiles.append(prof)

            # chunk textual content for index
            soup = BeautifulSoup(html, "html.parser")
            page_text = clean_text_from_soup(soup)
            chunks = chunk_text(page_text, size=int(chunk_size), overlap=int(chunk_overlap))
            for c in chunks:
                new_chunks.append({"text": c, "source": url})

            progress.progress(i/len(raw))
            time.sleep(delay)

        # append to session lists
        st.session_state["profiles"].extend(new_profiles)
        st.session_state["chunks"].extend(new_chunks)

        st.success(f"Processed {len(raw)} LinkedIn URLs. Profiles added: {len(new_profiles)}; chunks added: {len(new_chunks)}")

        # build/update index if there are chunks
        if new_chunks:
            all_texts = [c["text"] for c in st.session_state["chunks"]]
            if st.session_state["emb_model"] is None:
                st.session_state["emb_model"] = SentenceTransformer("all-MiniLM-L6-v2")
            model = st.session_state["emb_model"]
            st.info("Computing embeddings and updating FAISS index...")
            embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            st.session_state["index"] = index
            st.session_state["embeddings"] = embeddings
            st.success("Index updated with LinkedIn chunks.")

# -------------------------
# Show results & CSV
# -------------------------
st.markdown("---")
st.header("All extracted profiles (including LinkedIn results)")
if st.session_state["profiles"]:
    df = pd.DataFrame(st.session_state["profiles"])
    st.dataframe(df)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download profiles CSV", csv_bytes, file_name="profiles.csv", mime="text/csv")
else:
    st.info("No profiles extracted yet. Use the LinkedIn section above or the generic scraper.")

# -------------------------
# Quick RAG query (reuse index)
# -------------------------
st.markdown("---")
st.header("Quick RAG Query (uses built index)")
query = st.text_input("Enter a question to retrieve from indexed chunks (or empty to skip)")
top_k = st.number_input("Top K", min_value=1, max_value=10, value=3)
if st.button("Retrieve (and optionally call Gemini)"):
    if st.session_state["index"] is None or not st.session_state["chunks"]:
        st.error("No index available. Scrape pages first (LinkedIn or other).")
    elif not query:
        st.error("Enter a query")
    else:
        if st.session_state["emb_model"] is None:
            st.session_state["emb_model"] = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = st.session_state["emb_model"].encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = st.session_state["index"].search(q_emb, top_k)
        retrieved = []
        for idx in I[0]:
            if idx < len(st.session_state["chunks"]):
                retrieved.append(st.session_state["chunks"][idx])
        st.write("Retrieved snippets:")
        for r in retrieved:
            st.write(f"- {r['source']} — {r['text'][:300]}...")
        context = "\n\n---\n\n".join([f"Source: {r['source']}\n{r['text']}" for r in retrieved])
        st.text_area("RAG context preview", context[:4000], height=220)
        if api_key and GENAI_OK:
            try:
                prompt = f"You are an assistant constrained to the following context. Answer only from it. \n\nCONTEXT:\n{context}\n\nQUESTION: {query}"
                resp = genai.generate_text(model="gemini-1.5", prompt=prompt, max_output_tokens=512, temperature=0.0)
                answer = getattr(resp, "text", None) or str(resp)
                st.subheader("Gemini Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Gemini call failed: {e}")
                st.subheader("Context (no LLM)")
                st.write(context[:4000])
        else:
            st.info("Gemini not configured or SDK unavailable; showing context only.")
            st.write(context[:4000])

# -------------------------
# Debug toggles
# -------------------------
st.markdown("---")
if st.checkbox("Show raw profiles JSON (debug)"):
    st.json(st.session_state["profiles"])
if st.checkbox("Show indexed chunk count (debug)"):
    st.write("Chunks:", len(st.session_state["chunks"]))

st.caption("Reminder: Respect site ToS. If a LinkedIn page requires login in your browser this app will not extract profile details.")
