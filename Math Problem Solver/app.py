import os
import time
import json
from urllib.parse import urlparse
from typing import List, Dict, Any

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

try:
    import google.generativeai as genai
    GENAI_OK = True
except:
    GENAI_OK = False

st.set_page_config(page_title="Profile Scraper", layout="wide")
st.title("Profile Scraper")

user_agent = st.sidebar.text_input("User-Agent", value="ProfileRAG/1.0")
delay = st.sidebar.number_input("Delay (sec)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=4000, value=800)
chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=1000, value=200)

api_key = os.getenv("GOOGLE_API_KEY", "")
if api_key and GENAI_OK:
    genai.configure(api_key=api_key)

def fetch_html(url, ua, timeout=12):
    headers = {"User-Agent": ua} if ua else {}
    return requests.get(url, headers=headers, timeout=timeout)

def looks_like_login_page(html, resp_url):
    t = html.lower()
    checks = ["sign in","join linkedin","login","signin","authwall"]
    if any(x in resp_url.lower() for x in ["login","auth","signin"]):
        return True
    for c in checks:
        if c in t:
            return True
    return False

def clean_text_from_soup(soup):
    for tag in soup(["script","style","noscript","iframe","svg"]):
        tag.decompose()
    main = soup.find("main")
    if main:
        text = main.get_text("\n", strip=True)
    else:
        paragraphs = soup.find_all("p")
        if paragraphs:
            text = "\n".join(p.get_text(strip=True) for p in paragraphs)
        else:
            text = soup.get_text("\n", strip=True)
    return text

def parse_json_ld_person(soup):
    results = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
        except:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            typ = item.get("@type")
            if typ == "Person" or (isinstance(typ, list) and "Person" in typ):
                results["name"] = item.get("name")
                results["headline"] = item.get("jobTitle")
                results["description"] = item.get("description")
                works = item.get("worksFor") or item.get("affiliation")
                if isinstance(works, dict):
                    results["company"] = works.get("name")
                address = item.get("address")
                if isinstance(address, dict):
                    results["location"] = ", ".join([v for v in address.values() if isinstance(v, str)])
    return results

def extract_profile_fields(url, html):
    soup = BeautifulSoup(html, "html.parser")
    data = {"url": url, "name": "", "headline": "", "location": "", "about": "", "current_role": "", "experience": [], "status": "ok"}
    jsonld = parse_json_ld_person(soup)
    if jsonld:
        data["name"] = data["name"] or jsonld.get("name","")
        data["headline"] = data["headline"] or jsonld.get("headline","")
        data["about"] = data["about"] or jsonld.get("description","")
        data["location"] = data["location"] or jsonld.get("location","")

    name_selectors = ["h1",".profile-name"]
    for sel in name_selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            data["name"] = data["name"] or el.get_text(strip=True)
            break

    headline_selectors = ["h2",".profile-headline"]
    for sel in headline_selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            data["headline"] = data["headline"] or el.get_text(strip=True)
            break

    loc_selectors = [".location",".locality"]
    for sel in loc_selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            data["location"] = data["location"] or el.get_text(strip=True)
            break

    about_selectors = [".about",".summary"]
    for sel in about_selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            data["about"] = data["about"] or el.get_text(" ", strip=True)
            break

    exp_root = soup.find(id="experience")
    if exp_root:
        items = exp_root.find_all("li")
        for li in items[:6]:
            txt = li.get_text(" | ", strip=True)
            if txt:
                data["experience"].append(txt)

    if not data["about"]:
        data["about"] = clean_text_from_soup(soup)[:3000]

    if not data["current_role"] and data["experience"]:
        data["current_role"] = data["experience"][0]

    return data

def chunk_text(text, size=800, overlap=200):
    if not text:
        return []
    text = text.replace("\n"," ")
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

st.header("Enter Profile URLs")
linkedin_urls_text = st.text_area("URLs (one per line)", height=180)

if st.button("Scrape"):
    raw = [u.strip() for u in linkedin_urls_text.splitlines() if u.strip()]
    session = requests.Session()
    if user_agent:
        session.headers.update({"User-Agent": user_agent})
    progress = st.progress(0)
    new_profiles = []
    new_chunks = []
    for i, url in enumerate(raw, start=1):
        try:
            resp = session.get(url, timeout=15)
        except Exception as e:
            profile = {"url": url, "status": "fetch_error", "error": str(e)}
            new_profiles.append(profile)
            progress.progress(i/len(raw))
            time.sleep(delay)
            continue
        if resp.status_code != 200:
            profile = {"url": url, "status": f"status_{resp.status_code}"}
            new_profiles.append(profile)
            progress.progress(i/len(raw))
            time.sleep(delay)
            continue
        html = resp.text
        if looks_like_login_page(html, resp.url):
            profile = {"url": url, "status": "requires_login"}
            new_profiles.append(profile)
            progress.progress(i/len(raw))
            time.sleep(delay)
            continue
        prof = extract_profile_fields(url, html)
        prof["status"] = "ok"
        new_profiles.append(prof)
        soup = BeautifulSoup(html, "html.parser")
        page_text = clean_text_from_soup(soup)
        chunks = chunk_text(page_text, size=int(chunk_size), overlap=int(chunk_overlap))
        for c in chunks:
            new_chunks.append({"text": c, "source": url})
        progress.progress(i/len(raw))
        time.sleep(delay)

    st.session_state["profiles"].extend(new_profiles)
    st.session_state["chunks"].extend(new_chunks)

    if new_chunks:
        all_texts = [c["text"] for c in st.session_state["chunks"]]
        if st.session_state["emb_model"] is None:
            st.session_state["emb_model"] = SentenceTransformer("all-MiniLM-L6-v2")
        model = st.session_state["emb_model"]
        embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        st.session_state["index"] = index
        st.session_state["embeddings"] = embeddings

st.header("Profiles")
if st.session_state["profiles"]:
    df = pd.DataFrame(st.session_state["profiles"])
    st.dataframe(df)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, file_name="profiles.csv", mime="text/csv")

st.header("Ask")
query = st.text_input("Question")
top_k = st.number_input("Top K", min_value=1, max_value=10, value=3)

if st.button("Search"):
    if st.session_state["index"] is None or not st.session_state["chunks"]:
        st.write("No index yet")
    elif not query:
        st.write("Enter question")
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
        context = "\n\n".join([f"{r['source']}\n{r['text']}" for r in retrieved])
        st.text_area("Result", context[:4000], height=220)
        if api_key and GENAI_OK:
            try:
                prompt = f"Use only this context:\n{context}\n\nQuestion: {query}"
                resp = genai.generate_text(model="gemini-1.5", prompt=prompt, max_output_tokens=512, temperature=0.0)
                answer = getattr(resp, "text", None) or str(resp)
                st.write(answer)
            except:
                st.write(context[:4000])
