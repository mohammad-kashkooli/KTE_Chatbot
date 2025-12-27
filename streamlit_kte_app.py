"""
Streamlit UI for KTE (Knowledge Translation) study runs
------------------------------------------------------

What this app does
- Upload a manuscript (PDF/DOCX)
- Choose backend/model and toggle HyDE on/off
- Generate KTE fields using your pipeline (tries to call `generate_kte` from `kte_proposal.py`)
- Preview the result (JSON)
- Render and download a Persian KTE form PDF similar to Tarjoman-e-Danesh
- (Optional) Upload the human gold (e.g., tarjoman_danesh.pdf) and get ROUGE-L & BERTScore per-field

How to run
1) Install deps (adjust as needed):
   pip install streamlit pdfminer.six jinja2 weasyprint pandas rouge-score bert-score python-docx arabic-reshaper python-bidi reportlab
   # For OpenAI backend (optional): pip install openai
   # For Ollama backend (optional): pip install ollama
2) Put a Persian-capable font TTF next to this file (e.g., fonts/Vazirmatn-Regular.ttf or fonts/NotoNaskhArabic-Regular.ttf)
3) streamlit run streamlit_kte_app.py

Notes
- The app tries to import `generate_kte` from your existing `kte_proposal.py`.
  It will pass (path, model_name, use_hyde) if available; otherwise it falls back to (path, model_name).
- If your generator returns a Pydantic model, dict, or JSON string—it's handled.
- PDF rendering uses WeasyPrint with an in-code Jinja2 HTML template.
- Gold PDF parsing is heuristic: it looks for standard Persian headings to split sections.

Author: you
"""
from __future__ import annotations
import os
import io
import re
import json
import base64
import math
import tempfile
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd

# Optional deps used at runtime if present
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None

try:
    from bert_score import score as bert_score
except Exception:
    bert_score = None

from jinja2 import Template

# RAG helpers from your generator module
from kte_proposal import load_document, split_sections, build_vectordb  # reuse same pipeline
from kte_proposal import assign_pids, format_context, validate_citations, sanitize_citations

# -----------------------------
# Field mappings & UI constants
# -----------------------------
EN_TO_FA = {
    "title": "عنوان پیام پژوهشی",
    "key_message": "پیام کلیدی",
    "plain_summary": "متن پیام پژوهشی",
    "importance": "اهمیت موضوع",
    "nontech_results": "مهم‌ترین نتایج به زبان غیرتخصصی",
    "applications": "موارد کاربرد نتایج طرح",
    "impacts": "تأثیرات و کاربردها",
    "limitations": "محدودیت‌های شواهد",
    "audience": "مخاطبین طرح",
    "references": "منابع و مراجع",
}

FA_HEADINGS = list(EN_TO_FA.values())

HTML_TEMPLATE = r"""
<!doctype html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8" />
  <style>
    @font-face {
      font-family: 'FAFont';
      src: url('[[FONT_PATH]]') format('truetype');
    }
    body { font-family: FAFont, sans-serif; margin: 32px; }
    h1 { font-size: 22pt; margin-bottom: 6px; }
    h2 { font-size: 16pt; margin: 18px 0 6px; }
    p, li { font-size: 12pt; line-height: 1.8; text-align: justify; }
    .section { page-break-inside: avoid; }
    .key { font-weight: bold; display: block; margin-bottom: 6px; }
    .box { border: 1px solid #999; border-radius: 6px; padding: 10px 14px; }
    .meta { font-size: 10pt; color: #666; margin-bottom: 14px; }
  </style>
</head>
<body>
  <h1>ترجمان دانش - برگه پیام پژوهشی</h1>
  <div class="meta">خروجی تولیدشده به‌صورت خودکار توسط سامانه KTE</div>

  {% for key, label in labels.items() %}
  <div class="section">
    <h2>{{ label }}</h2>
    <div class="box">
      {% if key in bullets %}
        <ul>
          {% for item in data.get(key, []) %}<li>{{ item }}</li>{% endfor %}
        </ul>
      {% else %}
        <p>{{ data.get(key, '') }}</p>
      {% endif %}
    </div>
  </div>
  {% endfor %}
</body>
</html>
"""

BULLET_FIELDS = {"impacts", "applications", "audience"}

# -----------------------------
# Utilities
# -----------------------------

def _try_import_generate():
    try:
        from kte_proposal import generate_kte  # type: ignore
        return generate_kte
    except Exception as e:
        return None


def _safe_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert various return types (pydantic model, dict, json string) to dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # pydantic v1
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()
        except Exception:
            pass
    # pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # JSON string
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return {"raw": obj}
    # Fallback to repr
    return {"raw": repr(obj)}



def _apply_pdf_image_settings_from_state() -> None:
    """
    UI-controlled PDF image handling.
    We keep compatibility with the generator by passing settings via env vars.
    """
    mode = (st.session_state.get("pdf_img_mode") or os.getenv("KTE_PDF_IMAGE_MODE", "none")).strip().lower()
    if mode not in {"none", "auto", "ocr", "vision"}:
        mode = "none"

    def _f(name: str, env_name: str, default: float) -> float:
        v = st.session_state.get(name, None)
        if v is None:
            try:
                return float(os.getenv(env_name, str(default)))
            except Exception:
                return default
        try:
            return float(v)
        except Exception:
            return default

    def _i(name: str, env_name: str, default: int) -> int:
        v = st.session_state.get(name, None)
        if v is None:
            try:
                return int(os.getenv(env_name, str(default)))
            except Exception:
                return default
        try:
            return int(v)
        except Exception:
            return default

    
    min_area = max(0.0, min(1.0, _f("pdf_img_min_area", "KTE_PDF_IMAGE_MIN_AREA", 0.08)))
    max_regions = max(1, _i("pdf_img_max_regions", "KTE_PDF_IMAGE_MAX_REGIONS", 4))

    os.environ["KTE_PDF_IMAGE_MODE"] = mode
    os.environ["KTE_PDF_IMAGE_MIN_AREA"] = str(min_area)
    os.environ["KTE_PDF_IMAGE_MAX_REGIONS"] = str(max_regions)
    
    
    # Vision backend/model (used when mode is vision/auto)
    vb = (st.session_state.get("vision_backend") or os.getenv("KTE_VISION_BACKEND", "openai")).strip().lower()
    vm = (st.session_state.get("vision_model") or os.getenv("KTE_VISION_MODEL", "")).strip()
    if not vm:
        vm = "gpt-4o" if vb == "openai" else "gemma3:4b"
    os.environ["KTE_VISION_BACKEND"] = vb
    os.environ["KTE_VISION_MODEL"] = vm


def _apply_advanced_rag_settings_from_state() -> None:
    """Apply chunking / RAG / context settings (UI -> env vars) so kte_proposal.py can read them."""
    def _i(key: str, default: int) -> int:
        # Prefer session_state, otherwise env var (KTE_*) if already set
        try:
            raw = st.session_state.get(key, os.getenv(key.upper(), default))
            return int(str(raw).strip())
        except Exception:
            return int(default)

    os.environ["KTE_CHUNK_SIZE"] = str(_i("kte_chunk_size", 800))
    os.environ["KTE_CHUNK_OVERLAP"] = str(_i("kte_chunk_overlap", 80))
    os.environ["KTE_RAG_K"] = str(_i("kte_rag_k", 6))
    os.environ["KTE_RAG_PER_DOC_CHARS"] = str(_i("kte_rag_per_doc_chars", 1200))
    os.environ["KTE_RAG_MAX_CHARS"] = str(_i("kte_rag_max_chars", 6000))
    os.environ["KTE_OLLAMA_NUM_CTX"] = str(_i("kte_ollama_num_ctx", 4096))
    os.environ["KTE_MAX_OUTPUT_TOKENS"] = str(_i("kte_max_output_tokens", 1024))


def generate_kte_wrapper(file_path: str, backend: str, model_name: str, use_hyde: bool) -> Dict[str, Any]:
    """Call user's generator if available; otherwise raise a helpful error."""
    # Pass settings to user's code via env vars, if they choose to read them
    os.environ["KTE_BACKEND"] = backend
    os.environ["KTE_USE_HYDE"] = "1" if use_hyde else "0"
    
    
    # UI-controlled PDF image handling (none|auto|ocr|vision)
    _apply_pdf_image_settings_from_state()
    _apply_advanced_rag_settings_from_state()
    
    # Optional embedding model override from the UI
    if embed_model_override := st.session_state.get("embed_model_override", None):
        os.environ["KTE_EMBED_MODEL"] = embed_model_override

    # Optional embedding fallback model from the UI
    if embed_fallback_model := st.session_state.get("embed_fallback_model", None):
        os.environ["KTE_EMBED_FALLBACK_MODEL"] = embed_fallback_model

    # Normalize to registry key if the user typed underlying name
    model_key = model_name
    try:
        from kte_proposal import LLM_REG  # type: ignore
        if model_name in LLM_REG:
            model_key = model_name
        else:
            for k, cfg in LLM_REG.items():
                if str(cfg.get("name")) == model_name:
                    model_key = k
                    break
    except Exception:
        pass
    

    gen = _try_import_generate()
    if gen is None:
        raise RuntimeError(
            "Couldn't import generate_kte from kte_proposal.py. "
            "Please expose a function named generate_kte(path, model_name, use_hyde:bool=False) or (path, model_name)."
        )

    try:
        out = gen(file_path, model_key, use_hyde=use_hyde)  # preferred signature (key + toggle)
    except TypeError:
        out = gen(file_path, model_key)  # legacy signature accepts only the model key

    return _safe_to_dict(out)


# ----- Persian PDF rendering via WeasyPrint -----

def render_tarjoman_pdf(kte: Dict[str, Any], font_path: Optional[str] = None) -> bytes:
    """Render a Persian PDF using WeasyPrint. Returns PDF bytes."""
    try:
        from weasyprint import HTML
    except Exception as e:
        raise RuntimeError("WeasyPrint not installed. Please: pip install weasyprint") from e

    # Normalize to expected keys
    data = {}
    for en_key, fa_label in EN_TO_FA.items():
        val = kte.get(en_key)
        # allow Persian-keyed input too (e.g., if your pipeline already returns FA keys)
        if val is None:
            val = kte.get(fa_label)
        # tolerate bullet lists as \n-delimited strings
        if en_key in BULLET_FIELDS and isinstance(val, str):
            data[en_key] = [x.strip("•- \t") for x in val.split("\n") if x.strip()]
        else:
            data[en_key] = val if val is not None else ""

    # Try to locate a Persian-capable font
    fp = font_path or _find_font_path()
    font_src = fp if fp else ""

    html = Template(HTML_TEMPLATE.replace("[[FONT_PATH]]", font_src)).render(
        data=data,
        labels=EN_TO_FA,
        bullets=BULLET_FIELDS,
    )

    pdf_buf = io.BytesIO()
    HTML(string=html, base_url=os.getcwd()).write_pdf(pdf_buf)
    return pdf_buf.getvalue()


def _find_font_path() -> Optional[str]:
    """Best-effort discovery of a Persian-capable TTF in ./fonts"""
    candidates = [
        "fonts/Vazirmatn-Regular.ttf",
        "fonts/NotoNaskhArabic-Regular.ttf",
        "Vazirmatn-Regular.ttf",
        "NotoNaskhArabic-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


# ----- Gold parsing & metrics -----

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    if pdf_extract_text is None:
        raise RuntimeError("pdfminer.six not installed. Please: pip install pdfminer.six")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        path = tmp.name
    try:
        return pdf_extract_text(path)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def parse_gold_fields_fa(text: str) -> Dict[str, str]:
    """Heuristically split a Persian KTE PDF text into sections keyed by EN_TO_FA keys."""
    # Create a pattern like: (عنوان پیام پژوهشی|پیام کلیدی|...)
    headings_escaped = [re.escape(h) for h in FA_HEADINGS]
    splitter = re.compile(r"(?m)^(%s)\s*$" % "|".join(headings_escaped))

    parts = splitter.split(text)
    # parts looks like [pre, heading1, body1, heading2, body2, ...]
    out_fa = {h: "" for h in FA_HEADINGS}
    if len(parts) < 3:
        # fallback: try colon-separated
        for h in FA_HEADINGS:
            m = re.search(re.escape(h) + r"\s*[:\u061B]?\s*(.+?)(?=\n\s*\S+\s*[:\u061B]|\Z)", text, re.S)
            if m:
                out_fa[h] = m.group(1).strip()
        return {en: out_fa[fa] for en, fa in EN_TO_FA.items()}

    # Walk pairwise over headings and bodies
    it = iter(parts[1:])
    for heading, body in zip(it, it):
        body = body.strip()
        out_fa[heading] = body

    # Map FA heading text back to EN keys
    return {en: out_fa.get(fa, "").strip() for en, fa in EN_TO_FA.items()}


def compute_metrics_per_field(cand: Dict[str, str], gold: Dict[str, str]) -> pd.DataFrame:
    rows = []
    # Rouge
    r_scorer = None
    if rouge_scorer is not None:
        r_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # BERTScore
    has_bert = bert_score is not None

    for key, label in EN_TO_FA.items():
        c = (cand.get(key) or "").strip()
        g = (gold.get(key) or "").strip()
        rL = None
        if r_scorer and c and g:
            r = r_scorer.score(g, c)["rougeL"]
            rL = r.fmeasure
        bP = bR = bF = None
        if has_bert and c and g:
            try:
                P, R, F1 = bert_score([c], [g], lang="fa")
                bP, bR, bF = float(P.mean()), float(R.mean()), float(F1.mean())
            except Exception:
                pass
        rows.append({
            "field": label,
            "rougeL_f": rL,
            "bertscore_P": bP,
            "bertscore_R": bR,
            "bertscore_F1": bF,
            "cand_len": len(c),
            "gold_len": len(g),
        })
    df = pd.DataFrame(rows)
    return df



# -----------------------------
# Chat helpers (editor mode)
# -----------------------------

# Optional: clamp via your generator module if available
try:
    from kte_proposal import clamp_results  # reuses your field limits
except Exception:
    def clamp_results(d):  # minimal fallback
        return d

ALLOWED_FIELDS = [
    "title","key_message","plain_summary","importance",
    "nontech_results","applications","impacts","limitations",
    "audience","references"
]

def _resolve_model_name_and_backend(model_key: str, chosen_backend: str) -> tuple[str,str]:
    backend = chosen_backend
    name = model_key
    try:
        from kte_proposal import LLM_LIST  # your registry
        if model_key in LLM_LIST:
            cfg = LLM_LIST[model_key]
            # Use the registry’s backend field
            reg_backend = str(cfg.get("backend", "")).lower()
            if reg_backend == "openai":
                backend = "OpenAI"
            elif reg_backend == "ollama":
                backend = "Ollama"
            name = str(cfg.get("name", model_key))
    except Exception:
        pass
    return name, backend

def _ollama_resp_text(resp) -> str:
    """Handle both dict-style and pydantic-object responses from ollama-python."""
    try:
        if hasattr(resp, "message") and hasattr(resp.message, "content"):
            return (resp.message.content or "").strip()
    except Exception:
        pass
    try:
        if isinstance(resp, dict):
            return ((resp.get("message") or {}).get("content") or "").strip()
    except Exception:
        pass
    return ""

def _chat_llm(messages: list[dict], backend: str, model_key: str) -> str:
    """Call OpenAI or Ollama based on the UI/backend selection."""
    model_name, eff_backend = _resolve_model_name_and_backend(model_key, backend)

    if eff_backend == "OpenAI":
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=int(st.session_state.get("kte_max_output_tokens", os.getenv("KTE_MAX_OUTPUT_TOKENS", "1024")))
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[OpenAI error] {e}"

    # Ollama (local)
    try:
        import ollama

        resp = ollama.chat(
            model=model_name,
            messages=messages,
            options={
                "num_ctx": int(st.session_state.get("kte_ollama_num_ctx", os.getenv("KTE_OLLAMA_NUM_CTX", "4096"))),
                "num_predict": int(st.session_state.get("kte_max_output_tokens", os.getenv("KTE_MAX_OUTPUT_TOKENS", "1024"))),
            },
        )
        return _ollama_resp_text(resp)
    
    except Exception as e:
        return f"[Ollama error] {e}"

def _extract_json(text: str) -> dict | None:
    """Best-effort to parse a JSON object from the model output."""
    import json, re
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try fenced code blocks
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try first {...} span
    m = re.search(r"(\{.*\})", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None

def _apply_updates(kte: dict, updates: dict) -> tuple[dict, list[str]]:
    """Apply model-suggested updates to allowed fields, with clamping."""
    changed = []
    new_kte = dict(kte)
    for k, v in (updates or {}).items():
        if k not in ALLOWED_FIELDS:
            continue
        # Normalize simple list-like fields
        if k in {"impacts","audience","references"} and isinstance(v, str):
            v = [x.strip("•- \t") for x in v.split("\n") if x.strip()]
        new_kte[k] = v
        changed.append(k)
    # Reuse your clamps to respect field length limits
    new_kte = clamp_results(new_kte)
    return new_kte, changed


def _is_title_like_query(q: str) -> bool:
    t = (q or "").strip().lower()
    # English + Persian (expand as needed)
    keys = [
        "title", "paper title", "article title",
        "authors", "author list",
        "doi", "journal", "year",
        "عنوان", "عنوان مقاله", "نام مقاله", "نویسندگان", "doi"
    ]
    return any(k in t for k in keys)

def _front_page_docs():
    # Uses your already-built chunks; safest + fastest
    chunks = st.session_state.get("article_chunks") or []
    front = [d for d in chunks if d.metadata.get("page") in (0, 1)]
    return front[:10] if front else chunks[:10]

def _dedupe_docs(docs):
    out, seen = [], set()
    for d in docs or []:
        pid = (d.metadata or {}).get("pid") or d.page_content[:80]
        if pid in seen:
            continue
        seen.add(pid)
        out.append(d)
    return out

def show_chat_editor():
    """Render the chat UI if a KTE dict is present in session state."""
    if "kte" not in st.session_state:
        return

    st.subheader("Refine via Chat ✍️")


    # Show chat history
    if "chat_messages" not in st.session_state:
        mode = st.session_state.get("chat_mode", "Q&A")
        if mode == "Edit KTE (JSON patch)":
            sys_content = (
                "You are KTE-Editor. You revise fields of the KTE form.\n"
                "Answer briefly, then output a JSON patch like:\n"
                '{"reply":"...", "updates":{"title":"...","nontech_results":"..."}}\n'
                "Only include fields you change. Keep within length limits.\n"
                "If you cite sources, cite ONLY IDs that appear in CONTEXT (e.g., [P0007]). "
                "Never invent citations."
            )
        else:
            # Q&A mode — just answer; do NOT output JSON unless explicitly asked
            sys_content = (
                "You are a helpful research assistant. Answer the user's question succinctly "
                "using the provided CONTEXT and CURRENT_KTE_JSON when helpful. "
                "If you cite sources, cite ONLY IDs that appear in CONTEXT (e.g., [P0007]). "
                "Never invent citations. Do NOT output JSON unless explicitly asked; just answer."
            )
        st.session_state.chat_messages = [{"role": "system", "content": sys_content}]


    # Only render non-system messages
    for m in st.session_state.chat_messages:
        if m.get("role") == "system":
            continue
        with st.chat_message(m["role"]):
            st.write(m["content"])


    prompt = st.chat_input("Ask to fix a field (e.g., “Title should be …”, “Focus more on ICU results…”)")
    if not prompt:
        return
    
    with st.chat_message("user"):
        st.write(prompt)


    # ----- Conversational RAG: retrieve context for this user turn -----
    rag_context = ""
    docs = []
    hypothetical = None
    qry_used = prompt
    
    if st.session_state.get("use_rag_in_chat") and st.session_state.get("retriever"):
        try:
            # Build the query string for retrieval
            qry = prompt

            # Optional HyDE expansion (per-turn, only if enabled)
            
            is_title_like = _is_title_like_query(prompt)
            use_hyde_this_turn = bool(st.session_state.get("use_hyde_in_chat")) and not is_title_like

            if use_hyde_this_turn:
                # Use the current chat backend/model to draft a hypothetical answer
                # (This is the “HyDE” doc we’ll embed + search with)
                hyde_sys = {"role": "system", "content": (
                    "Write ONE concise, keyword-rich paragraph that could plausibly appear in the paper and answer the user's question.\n"
                    "Rules:\n"
                    "- Do NOT ask the user for the paper/text.\n"
                    "- Do NOT say you lack access / need more context.\n"
                    "- Do NOT use placeholders.\n"
                    "- Include concrete domain terms/entities likely to appear in the manuscript."
                )}
                hyde_user = {"role": "user", "content": prompt}

                # Reuse your existing chat LLM helper so it works with either OpenAI or Ollama
                effective_backend  = st.session_state.get("chat_backend") or st.session_state.get("backend", "OpenAI")
                effective_modelkey = st.session_state.get("chat_model_key") or st.session_state.get("model_key", "gpt-4o")
                hypothetical = _chat_llm([hyde_sys, hyde_user], backend=effective_backend, model_key=effective_modelkey)

                # Use the hypothetical text for retrieval (you can also combine with the original query if you prefer)
                qry = f"{prompt}\n\n{hypothetical}"

            # Retrieve with (possibly) expanded query
            qry_used = qry
            
            desired_k = int(st.session_state.get("kte_rag_k", os.getenv("KTE_RAG_K", "6")))
            raw_k = max(desired_k * 3, desired_k)  # over-fetch to survive dedupe
            raw_docs = []
            try:
                vs = getattr(st.session_state.retriever, "vectorstore", None)
                if vs is not None and hasattr(vs, "similarity_search"):
                    raw_docs = vs.similarity_search(qry_used, k=raw_k)
                else:
                    # Fallback: temporarily bump retriever k
                    try:
                        st.session_state.retriever.search_kwargs["k"] = raw_k
                    except Exception:
                        pass
                    raw_docs = st.session_state.retriever.get_relevant_documents(qry_used)
            except Exception:
                raw_docs = st.session_state.retriever.get_relevant_documents(qry_used)

            docs = _dedupe_docs(raw_docs)[:desired_k]
            
            # Title-like questions: pin front matter so we don't retrieve license/footer pages
            if _is_title_like_query(prompt):
                pinned = _front_page_docs()
                # Optional: drop common license/footer noise when user asks for title
                docs = [d for d in docs if "creativecommons" not in (d.page_content or "").lower()]
                docs = _dedupe_docs(pinned + docs)
            
            
            rag_context = format_context(
                docs,
                per_doc_chars=int(st.session_state.get("kte_rag_per_doc_chars", os.getenv("KTE_RAG_PER_DOC_CHARS", "1200"))),
                max_chars=int(st.session_state.get("kte_rag_max_chars", os.getenv("KTE_RAG_MAX_CHARS", "6000"))),
            )  # cap to keep tokens in check
        except Exception as e:
            rag_context = f"(retrieval failed: {e})"
            
    # ---- Debug expander (essential) ----
    try:
        retrieved_view = []
        for d in (docs or []):
            meta = d.metadata or {}
            retrieved_view.append({
                "pid": meta.get("pid"),
                "page": (meta.get("page") + 1) if isinstance(meta.get("page"), int) else meta.get("page"),
                "snippet": (d.page_content or "")[:250],
            })
        debug_payload = {
            "use_rag": bool(st.session_state.get("use_rag_in_chat")),
            "use_hyde": bool(st.session_state.get("use_hyde_in_chat")),
            "k_requested": int(st.session_state.get("kte_rag_k", os.getenv("KTE_RAG_K", "6"))),
            "k_returned": len(docs or []),
            "k_unique": len({((d.metadata or {}).get("pid") or "") for d in (docs or [])}),
            "prompt": prompt,
            "qry_used": qry_used,
            "hyde": hypothetical,
            "retrieved": retrieved_view,
        }
        st.session_state.last_rag_debug = debug_payload
        with st.expander("Debug: retrieval context & citations", expanded=False):
            st.json(debug_payload)
            st.text_area("RAG context (what the model sees)", rag_context or "", height=260)
    except Exception:
        pass
    # Append user message
    st.session_state.chat_messages.append({"role":"user","content":prompt})

    # Build the chat context with current KTE JSON
    import json
    kte_json = json.dumps(st.session_state.kte, ensure_ascii=False, indent=2)
    sys = st.session_state.chat_messages[0]  # the system message we seeded
    
    mode = st.session_state.get("chat_mode", "Q&A")

    if mode == "Edit KTE (JSON patch)":
        user = {
            "role": "user",
            "content": (
                "You are KTE-Editor. Use ONLY the provided CONTEXT when changing fields. "
                "If a fact is not in context, write 'گزارش نشده'.\n\n"
                "If you cite sources, cite ONLY IDs that appear in CONTEXT (e.g., [P0007]). "
                "Never invent citations.\n\n"
                f"CONTEXT:\n{rag_context or '(no context)'}\n\n"
                f"CURRENT_KTE_JSON:\n{kte_json}\n\n"
                f"USER_REQUEST:\n{prompt}\n\n"
                "Respond succinctly, then output a JSON object:\n"
                '{"reply":"what you changed","updates":{"field":"new text"}} '
                "Only include fields you actually update."
            ),
        }
    else:
        # Q&A mode: just answer the question – no JSON contract
        user = {
            "role": "user",
            "content": (
                "Answer the following question using the CONTEXT. "
                "If the answer is not present, say 'نامشخص/گزارش نشده'. "
                "Keep it concise and helpful.\n\n"
                "If you cite sources, cite ONLY IDs that appear in CONTEXT (e.g., [P0007]). "
                "Never invent citations.\n\n"
                f"CONTEXT:\n{rag_context or '(no context)'}\n\n"
                f"CURRENT_KTE_JSON:\n{kte_json}\n\n"
                f"QUESTION:\n{prompt}"
            ),
        }


    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            
            # Prefer chat overrides; otherwise fall back to generation settings
            backend  = st.session_state.get("chat_backend") or st.session_state.get("backend", "OpenAI")
            modelkey = st.session_state.get("chat_model_key") or st.session_state.get("model_key", "gpt-4o")
            
            
            if backend == "OpenAI" and not os.getenv("OPENAI_API_KEY"):
                st.warning("OPENAI_API_KEY not set. Add it in Secrets or environment to use chat.")
                return


            # NEW (include prior messages, which already live in session_state)
            out = _chat_llm(st.session_state.chat_messages + [user], backend=backend, model_key=modelkey)
            
            # If the model produced citations, ensure they're REAL (only from retrieved CONTEXT).
            # Repair once automatically if invalid citations appear.
            try:
                ok, cited, invalid, valid = validate_citations(out, docs)
                out = sanitize_citations(out, valid)
                if (not ok) and cited:
                    allowed = ", ".join(sorted(valid)) if valid else "(none)"
                    repair_user = dict(user)
                    repair_user["content"] = (
                        "Your previous answer contained citations that do not exist in the provided CONTEXT.\n"
                        f"Only cite from: {allowed}\n"
                        "Never invent citations. If not present, say 'نامشخص/گزارش نشده'.\n\n"
                        + user["content"]
                    )
                    out = _chat_llm(st.session_state.chat_messages + [repair_user], backend=backend, model_key=modelkey)
                    
                    try:
                        _, _, _, valid2 = validate_citations(out, docs)
                        out = sanitize_citations(out, valid2)
                    except Exception:
                        pass
            except Exception:
                pass

        
        # Q&A vs Edit mode
        mode = st.session_state.get("chat_mode", "Q&A")

        if mode == "Edit KTE (JSON patch)":
            # Parse & apply JSON updates
            payload = _extract_json(out)
            reply = out
            updates = {}
            if isinstance(payload, dict):
                reply = payload.get("reply", out)
                updates = payload.get("updates", {})

            st.write(reply)

            if updates:
                new_kte, changed = _apply_updates(st.session_state.kte, updates)
                st.session_state.kte = new_kte
                if changed:
                    st.success("Applied updates: " + ", ".join(changed))
                    # Try to re-render PDF if WeasyPrint available
                    try:
                        pdf_bytes = render_tarjoman_pdf(new_kte, font_path=st.session_state.get("font_hint"))
                        st.download_button(
                            label="⬇️ Download UPDATED KTE PDF",
                            data=pdf_bytes,
                            file_name="tarjoman_updated.pdf",
                            mime="application/pdf",
                        )
                    except Exception as e:
                        st.warning(f"PDF re-render failed: {e}")
        else:
            # Q&A mode: just show the assistant’s answer as-is (no JSON parsing)
            st.write(out)
            reply = out


    # Persist assistant message into history
    st.session_state.chat_messages.append({"role":"assistant","content":reply})




# ---------- Always-on KTE panel + simple HTML fallback ----------
import html

def simple_html_from_kte(kte: dict) -> str:
    """Pure-Python HTML (no WeasyPrint) so users can Print→Save as PDF."""
    def esc(x): return html.escape(str(x or ""))
    blocks = []
    def add(h, body):
        blocks.append(f"<h3 style='margin:12px 0'>{esc(h)}</h3><div style='white-space:pre-wrap'>{esc(body)}</div>")
    add("Title", kte.get("title",""))
    add("Key Message", kte.get("key_message",""))
    add("Plain Summary", kte.get("plain_summary",""))
    add("Importance", kte.get("importance",""))
    add("Non-technical Results", kte.get("nontech_results",""))
    add("Applications", "\n".join(kte.get("applications",[])) if isinstance(kte.get("applications"), list) else kte.get("applications",""))
    add("Impacts", "\n".join(kte.get("impacts",[])) if isinstance(kte.get("impacts"), list) else kte.get("impacts",""))
    add("Limitations", kte.get("limitations",""))
    add("Audience", "\n".join(kte.get("audience",[])) if isinstance(kte.get("audience"), list) else kte.get("audience",""))
    add("References", "\n".join(kte.get("references",[])) if isinstance(kte.get("references"), list) else kte.get("references",""))
    css = "<style>body{font-family:system-ui,Segoe UI,Arial;direction:rtl;text-align:right;padding:24px;}</style>"
    return f"<!doctype html><meta charset='utf-8'>{css}<h2>Tarjoman-e Danesh</h2>{''.join(blocks)}"

def display_kte_panel():
    """Always show the latest KTE + export buttons, regardless of `run`."""
    if "kte" not in st.session_state:
        return
    kte = st.session_state.kte
    st.subheader("Generated KTE (JSON)")
    st.json(kte)

    # Try PDF first (WeasyPrint). If missing, offer HTML/JSON fallbacks.
    try:
        pdf_bytes = render_tarjoman_pdf(kte, font_path=st.session_state.get("font_hint"))
        st.download_button("⬇️ Download KTE PDF", data=pdf_bytes,
                           file_name="tarjoman_danesh.pdf", mime="application/pdf")
    except Exception:
        st.info("PDF export unavailable on this machine. Download HTML or JSON below, "
                "then use your browser’s **Print → Save as PDF**.")
        html_str = simple_html_from_kte(kte)
        st.download_button("⬇️ Download KTE HTML", data=html_str,
                           file_name="tarjoman_danesh.html", mime="text/html")
        st.download_button("⬇️ Download KTE JSON",
                           data=json.dumps(kte, ensure_ascii=False, indent=2),
                           file_name="tarjoman_danesh.json", mime="application/json")



# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="KTE Study Runner", layout="wide")
st.title("KTE Study Runner 🧪")

with st.sidebar:
    st.header("Generation Settings")

# Try to import model registry from user's kte_proposal.py so we can
# present the *keys* that generate_kte() expects (e.g., "gemma2").
def _available_models():
    try:
        from kte_proposal import LLM_REG  # type: ignore
        return LLM_REG
    except Exception:
        return None

LLM_REG = _available_models()

backend = st.selectbox("Backend", ["OpenAI", "Ollama"], index=1)

if LLM_REG:
    # generate_kte() expects registry KEYS (e.g., "gemma2"), not raw names like "gemma2:9b"
    model_key = st.selectbox(
        "Model (key)",
        list(LLM_REG.keys()),
        index=list(LLM_REG.keys()).index("gemma2") if "gemma2" in LLM_REG else 0
    )
    
else:
    model_key = st.text_input("Model (key or raw name)", value=("gpt-4o" if backend=="OpenAI" else "llama3-med"))

use_hyde = st.checkbox("Use HyDE (Hypothetical Document Expansion)", value=True)
font_hint = st.text_input("Persian font TTF path (optional)", value="fonts/Vazirmatn-Regular.ttf")
embed_model_override = st.text_input("Embedding model (optional)", value="")
embed_fallback_model = st.text_input("Embedding fallback model (optional)", value=os.getenv("KTE_EMBED_FALLBACK_MODEL",""))



# ---- PDF image handling controls (optional) ----
with st.expander("PDF image handling (optional)", expanded=False):
    st.caption(
        "If your PDF has tables/figures as images, you can extract them into the text pipeline.\n"
        "- none: ignore images\n"
        "- ocr: use Tesseract (local) if available\n"
        "- vision: use a vision model (OpenAI or Ollama) to read images\n"
        "- auto: try vision if available, otherwise OCR"
    )

    _mode_choices = ["none", "auto", "ocr", "vision"]
    _mode_default = (os.getenv("KTE_PDF_IMAGE_MODE", "none") or "none").strip().lower()
    if _mode_default not in set(_mode_choices):
        _mode_default = "none"

    st.selectbox(
        "Image mode",
        _mode_choices,
        index=_mode_choices.index(_mode_default),
        key="pdf_img_mode",
        help="Controls how image regions in the PDF are handled during ingestion."
    )

    try:
        _min_default = float(os.getenv("KTE_PDF_IMAGE_MIN_AREA", "0.08"))
    except Exception:
        _min_default = 0.08
    st.slider(
        "Min image area (fraction of page)",
        min_value=0.01, max_value=0.30,
        value=float(_min_default),
        step=0.01,
        key="pdf_img_min_area",
        help="Only image regions larger than this fraction of the page are extracted."
    )

    try:
        _maxr_default = int(os.getenv("KTE_PDF_IMAGE_MAX_REGIONS", "4"))
    except Exception:
        _maxr_default = 4
    st.number_input(
        "Max image regions per page",
        min_value=1, max_value=10,
        value=int(_maxr_default),
        step=1,
        key="pdf_img_max_regions",
        help="Caps how many large image regions per page are processed."
    )
    
    # --- NEW: Vision backend selector (UI-controlled) ---
    if st.session_state.get("pdf_img_mode") in {"vision", "auto"}:
        st.selectbox(
            "Vision backend",
            ["openai", "ollama"],
            index=0 if (os.getenv("KTE_VISION_BACKEND", "openai").strip().lower() != "ollama") else 1,
            key="vision_backend",
            help="Where image-reading runs. Ollama requires a vision-capable model (e.g., gemma3:4b)."
        )
        vb = (st.session_state.get("vision_backend") or "openai").strip().lower()
        default_vm = "gpt-4o" if vb == "openai" else "gemma3:4b"
        st.text_input(
            "Vision model",
            value=(os.getenv("KTE_VISION_MODEL", "") or default_vm),
            key="vision_model",
            help="OpenAI example: gpt-4o | Ollama example: gemma3:4b"
        )

# Apply immediately so BOTH generation and the chat retriever ingestion use the same settings.
_apply_pdf_image_settings_from_state()



# ---- Advanced chunking / retrieval / context controls (optional) ----
with st.expander("Advanced: chunking & RAG size", expanded=False):
    st.caption("These settings affect BOTH the KTE extraction and the Q&A chat retriever.")
    st.slider("Chunk size (chars)", 200, 2500, int(os.getenv("KTE_CHUNK_SIZE", "800")), 50, key="kte_chunk_size")
    st.slider("Chunk overlap (chars)", 0, 400, int(os.getenv("KTE_CHUNK_OVERLAP", "80")), 10, key="kte_chunk_overlap")
    st.slider("Top-K retrieved chunks", 1, 12, int(os.getenv("KTE_RAG_K", "6")), 1, key="kte_rag_k")
    st.slider("Per-chunk context cap (chars)", 200, 4000, int(os.getenv("KTE_RAG_PER_DOC_CHARS", "1200")), 100, key="kte_rag_per_doc_chars")
    st.slider("Total context cap (chars)", 1000, 20000, int(os.getenv("KTE_RAG_MAX_CHARS", "6000")), 500, key="kte_rag_max_chars")
    st.slider("Ollama context window num_ctx", 512, 16384, int(os.getenv("KTE_OLLAMA_NUM_CTX", "4096")), 256, key="kte_ollama_num_ctx")
    st.slider("Max output tokens (num_predict / max_tokens)", 128, 4096, int(os.getenv("KTE_MAX_OUTPUT_TOKENS", "1024")), 64, key="kte_max_output_tokens")

_apply_advanced_rag_settings_from_state()


# If a retriever already exists (from a previous "Generate & Evaluate"),
# update its k immediately so chat respects the slider without requiring a re-run.
if st.session_state.get("retriever"):
    try:
        st.session_state.retriever.search_kwargs["k"] = int(
            st.session_state.get("kte_rag_k", os.getenv("KTE_RAG_K", "6"))
        )
    except Exception:
        pass

use_rag_in_chat = st.checkbox(
    "Use RAG in chat", value=True,
    help="Retrieve relevant snippets from the uploaded article for each chat turn."
)

use_hyde_in_chat = st.checkbox("Use HyDE in chat (generate hypothetical answer before searching)", value=False,
                               help="Adds one LLM call per message to create a short hypothetical answer, then searches by that text to improve recall.")
st.session_state.use_hyde_in_chat = use_hyde_in_chat


# ---- Chat model overrides (optional) ----
st.markdown("### Chat model (optional)")

# If you have a registry from kte_proposal.py we reuse its keys
if LLM_REG:
    chat_model_choice = st.selectbox(
        "Chat model",
        ["Same as generation"] + list(LLM_REG.keys()),
        index=0,
        help="Pick a different model JUST for chat. Leave as 'Same as generation' to reuse the generator model."
    )
    
else:
    chat_model_choice = st.selectbox(
        "Chat model",
        ["Same as generation", "gpt-4o", "gemma2", "mistral"],
        index=0
    )

chat_backend_choice = st.radio(
    "Chat backend",
    ["Same as generation", "OpenAI", "Ollama"],
    index=0
)

# Persist overrides (None means: use generation settings)
st.session_state.chat_model_key = (
    None if chat_model_choice == "Same as generation" else chat_model_choice
)
st.session_state.chat_backend = (
    None if chat_backend_choice == "Same as generation" else chat_backend_choice
)

# --- Chat mode (Q&A vs Edit JSON) ---
chat_mode = st.radio(
    "Chat mode",
    ["Q&A", "Edit KTE (JSON patch)"],
    index=0,
    help="Q&A answers questions without modifying JSON. "
         "Edit KTE expects a JSON patch (updates) to modify fields."
)
st.session_state.chat_mode = chat_mode

# Detect mode change and reseed system prompt
prev_mode = st.session_state.get("prev_chat_mode")
if prev_mode is None:
    st.session_state.prev_chat_mode = chat_mode
elif chat_mode != prev_mode:
    st.session_state.prev_chat_mode = chat_mode
    # Re-seed the system message for the new mode
    if chat_mode == "Edit KTE (JSON patch)":
        sys_content = (
            "You are KTE-Editor. You revise fields of the KTE form.\n"
            "Answer briefly, then output a JSON patch like:\n"
            '{"reply":"...", "updates":{"title":"...","nontech_results":"..."}}'
        )
    else:
        sys_content = (
            "You are a helpful research assistant. Answer the user's question succinctly "
            "using the provided CONTEXT and CURRENT_KTE_JSON when helpful. "
            "Do NOT output JSON unless explicitly asked; just answer."
        )
    # Start a fresh chat with the new instruction
    st.session_state.chat_messages = [{"role": "system", "content": sys_content}]
    st.rerun()  # immediately refresh with the new mode


st.session_state.use_rag_in_chat = use_rag_in_chat

st.caption("Tip: If your generator is from kte_proposal.py, choose a model *key* like 'gemma2' from the dropdown. Set an embedding override like 'nomic-embed-text:latest' if needed. Put a Persian-capable font file at the given path for best RTL rendering.")

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload manuscript (PDF or DOCX)", type=["pdf", "docx"])

with col2:
    gold = st.file_uploader("Upload human gold (optional, PDF)", type=["pdf"]) 

# keep embed model in session for wrapper access
if embed_model_override:
    st.session_state["embed_model_override"] = embed_model_override

# keep embed fallback model in session for wrapper access
if embed_fallback_model:
    st.session_state["embed_fallback_model"] = embed_fallback_model

run = st.button("Generate & Evaluate")

if run:
    if not uploaded:
        st.warning("Please upload a manuscript first.")
        st.stop()

    with st.spinner("Generating KTE…"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp.flush()
            doc_path = tmp.name
        try:
            # 🔐 Guard: ensure OPENAI_API_KEY is set when using OpenAI backend
            if backend == "OpenAI" and not os.getenv("OPENAI_API_KEY"):
                st.warning("Set OPENAI_API_KEY in your environment or Streamlit **Secrets** and try again.")
                st.stop()
                
            kte = generate_kte_wrapper(doc_path, backend=backend, model_name=model_key, use_hyde=use_hyde)
            
            
            # ---- Build a retriever for chat from this very manuscript (only if enabled) ----
            if use_rag_in_chat:
                try:
                    # Use the same ingestion settings for the chat retriever, too
                    _apply_pdf_image_settings_from_state()
                    raw_docs = load_document(doc_path)
                    chunks   = assign_pids(split_sections(raw_docs))
                    vectordb = build_vectordb(chunks)
                    
                    st.session_state.retriever = vectordb.as_retriever(search_kwargs={"k": int(st.session_state.get("kte_rag_k", os.getenv("KTE_RAG_K", "6")))})
                    st.session_state.article_chunks = chunks
                except Exception as e:
                    st.session_state.retriever = None
                    st.warning(f"RAG retriever not available: {e}")
            else:
                # Explicitly clear any old retriever when the toggle is off
                st.session_state.retriever = None
                st.session_state.article_chunks = None
                
                
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()
        finally:
            try:
                os.remove(doc_path)
            except Exception:
                pass

    
    # persist for subsequent reruns (e.g., after chat)
    st.session_state.kte = kte
    st.session_state.font_hint = font_hint


    # Prepare normalized (string) view for metrics
    cand_text = {}
    for en_key in EN_TO_FA.keys():
        v = kte.get(en_key) or kte.get(EN_TO_FA[en_key]) or ""
        if isinstance(v, list):
            v = "\n".join(v)
        cand_text[en_key] = str(v)
    
    # NEW: stash result & settings for the chat editor
    st.session_state.kte = kte
    st.session_state.backend = backend
    st.session_state.model_key = model_key
    st.session_state.font_hint = font_hint
    
    # Render Persian PDF
    with st.spinner("Rendering Persian PDF…"):
        try:
            pdf_bytes = render_tarjoman_pdf(kte, font_path=font_hint if font_hint else None)
            st.success("Rendered KTE PDF.")
            st.download_button(
                label="⬇️ Download KTE PDF",
                data=pdf_bytes,
                file_name="tarjoman_output.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"PDF render failed: {e}")

    # --- Chat editor appears after first output ---
    show_chat_editor()

    # If gold provided, compute metrics
    if gold is not None:
        st.subheader("Metrics vs. Human Gold")
        if pdf_extract_text is None:
            st.info("Install pdfminer.six for gold parsing: pip install pdfminer.six")
        else:
            try:
                gold_text = extract_text_from_pdf_bytes(gold.read())
                gold_fields = parse_gold_fields_fa(gold_text)
                df = compute_metrics_per_field(cand_text, gold_fields)
                st.dataframe(df, use_container_width=True)

                # Download CSV
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download metrics (CSV)", data=csv, file_name="kte_metrics.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Gold parsing or metrics failed: {e}")


# Always show the current KTE (even after chat-based reruns)
display_kte_panel()

st.markdown("---")
st.caption("Tip: If your generator already outputs Persian field names, the app automatically maps them.")




if "kte" in st.session_state and not run:
    show_chat_editor()

