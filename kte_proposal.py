"""
kte_rag.py
==========

End‑to‑end Retrieval‑Augmented Generation (RAG) + HyDE system that extracts the
fields of the KTE reporting form from a medical research article.

*  Ingests PDF or DOCX.
*  Performs section‑aware chunking.
*  Stores embeddings in a local Chroma DB.
*  Uses HyDE to expand user questions for better retrieval.
*  Generates each template field with the user‑selected LLM
   (local Ollama model OR OpenAI GPT model).

Author: <you>
"""

# ---------------------------------------------------------------------#
# 0.‑‑ Packages and CONFIG                                             #
# ---------------------------------------------------------------------#
import os, re, tempfile, json
from pathlib import Path
from typing import List, Dict

from pydantic.v1 import BaseModel, Field, validator

# LangChain core
from langchain_community.document_loaders import (
    PyMuPDFLoader, Docx2txtLoader
)
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
)

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_community.vectorstores import Chroma

from langchain.prompts import PromptTemplate


from langchain.output_parsers import PydanticOutputParser

# UI (optional)
import streamlit as st

LLM_LIST = {
    "gemma2": {"type": "ollama", "name": "gemma2:9b"},
    "llama3-med": {"type": "ollama", "name": "llama3-med42"},
    "mistral":    {"type": "ollama", "name": "mistral:instruct"},
    "gpt-4o":     {"type": "openai", "name": "gpt-4o-mini"}
}



# ---------------------------------------------------------------------#
# Model registry (single source of truth)                              #
# ---------------------------------------------------------------------#
# Keys are what users select in the UI; "name" is the underlying model id.
# Add/remove entries here to change the available models everywhere.
LLM_REG = {
    # ---- OpenAI (cloud) ----
    "gpt-4o-mini": {"backend": "OpenAI", "name": "gpt-4o-mini"},
    "gpt-4o":      {"backend": "OpenAI", "name": "gpt-4o"},

    # ---- Ollama (local) ----
    "gemma2":                 {"backend": "Ollama", "name": "gemma2:9b"},
    "llama3.1:8b-instruct":   {"backend": "Ollama", "name": "llama3.1:8b-instruct"},
    "mistral:7b-instruct":    {"backend": "Ollama", "name": "mistral:7b-instruct"},
}

# Back-compat alias (some modules may still import LLM_LIST)
LLM_LIST = LLM_REG

EMBED_MODEL = "nomic-embed-text"

# ---------------------------------------------------------------------#
# 1.‑‑ Pydantic schema describing the 12 fields of the KTE form       #
# ---------------------------------------------------------------------#
class KTEForm(BaseModel):
    title: str               = Field(max_length=120)    # ≈20 words
    key_message: str         = Field(max_length=480)    # ≈80 words
    plain_summary: str       = Field(max_length=1500)   # ≈240 words
    importance: str          = Field(max_length=900)    # ≈150 words
    nontech_results: str     = Field(max_length=450)    # ≈70 words
    applications: str        = Field(max_length=480)    # ≈80 words
    impacts: List[str]       
    limitations: str
    audience: List[str]
    references: List[str]

    # Optional word‑count guardrail  – regenerate if violated
    @validator('*', pre=True)
    def _check_len(cls, v, values, field):
        if isinstance(v, str) and len(v.split()) > (field.field_info.max_length or 9999) / 6:
            raise ValueError(f"{field.name} too long")
        return v


# ---- Length clamps mirroring KTEForm ----
_STR_LIMITS = {
    "title": 120,
    "key_message": 480,
    "plain_summary": 1500,
    "importance": 900,
    "nontech_results": 450,
    "applications": 480,
    # "limitations": no hard cap in schema; leave as-is
}

_LIST_LIMITS = {
    "impacts": {"max_items": 5, "item_len": 120},
    "audience": {"max_items": 6, "item_len": 40},
    "references": {"max_items": 5, "item_len": 200},
}

def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip()

def clamp_results(d: dict) -> dict:
    out = dict(d)
    # strings
    for k, n in _STR_LIMITS.items():
        if isinstance(out.get(k), str):
            out[k] = _clip(out[k], n)
    # lists (normalize text -> list if needed)
    for k, cfg in _LIST_LIMITS.items():
        v = out.get(k, [])
        if isinstance(v, str):
            items = [x.strip("•- \t") for x in v.split("\n") if x.strip()]
        else:
            items = list(v) if isinstance(v, list) else []
        items = items[: cfg["max_items"]]
        items = [_clip(str(it), cfg["item_len"]) for it in items]
        out[k] = items
    return out

# ---------------------------------------------------------------------#
# 2.‑‑ Ingestion utilities                                             #
# ---------------------------------------------------------------------#
def load_document(path: str):
    """Return a list[Document] from PDF or DOCX."""
    if path.lower().endswith(".pdf"):
        return PyMuPDFLoader(path).load()
    if path.lower().endswith(".docx"):
        return Docx2txtLoader(path).load()
    raise ValueError("Only PDF or DOCX accepted")

HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("##", "abstract"), ("##", "introduction"),
        ("##", "methods"),  ("##", "results"),
        ("##", "discussion"), ("##", "conclusion"),
        ("#",  "title")
    ]
)

def split_sections(raw_docs):
    """Chunk docs at section boundaries when headings exist,
    otherwise fall back to recursive splitting."""
    docs = []
    for d in raw_docs:
        if re.search(r"\n[A-Z][A-Za-z ]{1,40}\n", d.page_content):
            docs.extend(HEADER_SPLITTER.split_text(d.page_content))
        else:
            docs.append(d)
    # fine‑grained splitter (prevents >4k‑token chunks)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80
    )
    return splitter.split_documents(docs)

# ---------------------------------------------------------------------#
# 3.‑‑ Embeddings, Vector store, HyDE retriever                        #
# ---------------------------------------------------------------------#
def get_embeddings():
    """
    Choose embeddings based on env var KTE_EMBED_MODEL.
    - If it looks like an OpenAI model (e.g., 'text-embedding-3-small'), use OpenAIEmbeddings
    - Else try OllamaEmbeddings with the given/local name
    - Fallback to SentenceTransformer if neither works
    """
    import os
    from langchain_community.embeddings import OllamaEmbeddings, SentenceTransformerEmbeddings

    name = os.getenv("KTE_EMBED_MODEL", EMBED_MODEL).strip().lower()

    # OpenAI path (cloud-friendly). Requires OPENAI_API_KEY.
    # e.g., set KTE_EMBED_MODEL=text-embedding-3-small
    if name.startswith("text-embedding-") or name in {"openai", "openai-emb"}:
        from langchain_openai import OpenAIEmbeddings
        model = name if name.startswith("text-embedding-") else "text-embedding-3-small"
        return OpenAIEmbeddings(model=model)  # reads OPENAI_API_KEY

    # Local default via Ollama
    try:
        return OllamaEmbeddings(model=os.getenv("KTE_EMBED_MODEL", EMBED_MODEL))
    except Exception:
        # Small, portable fallback (no API key needed)
        return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")



def build_vectordb(chunks):
    emb = get_embeddings()
    return Chroma.from_documents(chunks, emb)


def make_hyde_search(llm, vectordb, k=4):
    def search(query: str):
        hypo   = llm.invoke(f"Write a concise answer to this question:\n{query}").content
        vec  = vectordb._embedding_function.embed_query(hypo)
        return vectordb.similarity_search_by_vector(vec, k=k)
    return search
# ---------------------------------------------------------------------#
# 4.‑‑ Prompt templates for each form field                            #
# ---------------------------------------------------------------------#
BASE_TEMPLATE = """You are MedKTE‑Bot.
The user needs the <{field_name}> field for a knowledge‑translation form.

## Instructions
{extra_instructions}

## Context
{context}

Write ONLY the <{field_name}>. Cite paragraph IDs like [P12].
"""

FIELD_PROMPTS: Dict[str, str] = {
    "title":           "the official article title and study setting",
    "key_message":     "the most important single finding and its practical implication",
    "plain_summary":   "short overview: population, setting, time window, methods, key results, take-home",
    "importance":      "why the study matters to clinicians/policymakers",
    "nontech_results": "core numbers: sample size, ICU admission %, mortality %, hospital stay length, differences between strata/time periods",
    "applications":    "practical uses of the findings for care pathways, triage, vaccination planning, resource allocation",
    "impacts":         "policy and clinical impact statements",
    "limitations":     "study limitations and caveats",
    "audience":        "stakeholder groups likely to use these findings",
    "references":      "key references cited in the paper",
}

def get_prompt(field_name: str) -> PromptTemplate:
    template = BASE_TEMPLATE.format(
        field_name=field_name,
        extra_instructions=FIELD_PROMPTS[field_name],
        context="{context}"     # keep placeholder!
    )
    return PromptTemplate.from_template(template)

# ---------------------------------------------------------------------#
# 5.‑‑ LLM selector                                                    #
# ---------------------------------------------------------------------#

def choose_llm(model_key: str):
    """Return an LLM client from the registry. Falls back to key as raw name."""
    cfg = LLM_REG.get(model_key, {"backend": "Ollama", "name": model_key})
    if cfg["backend"] == "OpenAI":
        # Keep signature consistent with your working env
        return ChatOpenAI(model_name=cfg["name"], temperature=0)
    # Ollama (local)
    return ChatOllama(model=cfg["name"])

# ---------------------------------------------------------------------#
# 6.‑‑ Driver function                                                 #
# ---------------------------------------------------------------------#




def generate_kte(path: str, model_key: str = "llama3-med", use_hyde: bool | None = None) -> KTEForm:
    # Resolve HyDE flag: CLI/Streamlit may pass it; otherwise read env (default True)
    if use_hyde is None:
        env = os.getenv("KTE_USE_HYDE", "1")
        use_hyde = env.strip().lower() not in {"0", "false", "no"}

    raw_docs = load_document(path)
    chunks   = split_sections(raw_docs)
    vectordb = build_vectordb(chunks)
    llm      = choose_llm(model_key)

    # Choose retrieval based on the HyDE flag
    def plain_search(q: str):
        return vectordb.similarity_search(q, k=4)

    try:
        search = make_hyde_search(llm, vectordb, k=4) if use_hyde else plain_search
    except NameError:
        # if make_hyde_search isn’t defined, just fall back gracefully
        search = plain_search

    results = {}
    for field in FIELD_PROMPTS.keys():
        docs = search(FIELD_PROMPTS.get(field, f"context for {field}"))
        context = "\n\n".join(d.page_content for d in docs)
        answer = llm.invoke(get_prompt(field).format(context=context)).content

        if field in {"impacts", "audience"}:
            answer = [x.strip("•– ").strip() for x in re.split(r"\n|,", answer) if x.strip()]
        if field == "references":
            answer = [x.strip() for x in re.split(r";|\n", answer) if x.strip()]
        results[field] = answer

    results = clamp_results(results)
    return KTEForm(**results)


# ---------------------------------------------------------------------#
# 7.‑‑ Simple CLI                                                      #
# ---------------------------------------------------------------------#
def _cli():
    import argparse, textwrap
    ap = argparse.ArgumentParser(
        description="Extract KTE messages from a medical article using RAG+HyDE"
    )
    ap.add_argument("file", type=str, help="PDF or DOCX article")
    ap.add_argument("--model", default="llama3-med",
                    choices=LLM_LIST.keys(),
                    help="Which LLM backend to use")
    args = ap.parse_args()

    kte = generate_kte(args.file, args.model)
    print(textwrap.indent(kte.json(indent=2, ensure_ascii=False), "  "))

# ---------------------------------------------------------------------#
# 8.‑‑ Streamlit UI (optional)                                         #
# ---------------------------------------------------------------------#
def _streamlit():
    st.title("KTE Extractor (RAG + HyDE)")
    model = st.sidebar.selectbox("Model", list(LLM_LIST.keys()))
    uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf","docx"])
    if uploaded and st.button("Generate KTE"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        kte_obj = generate_kte(tmp_path, model)
        st.json(json.loads(kte_obj.json()))

# ---------------------------------------------------------------------#
if __name__ == "__main__":
    # If launched via `streamlit run`, Streamlit sets this flag
    if os.getenv("STREAMLIT_SERVER_RUNNING"):
        _streamlit()
    else:
        _cli()
 