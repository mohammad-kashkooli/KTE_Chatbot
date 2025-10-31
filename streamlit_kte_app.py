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


def generate_kte_wrapper(file_path: str, backend: str, model_name: str, use_hyde: bool) -> Dict[str, Any]:
    """Call user's generator if available; otherwise raise a helpful error."""
    # Pass settings to user's code via env vars, if they choose to read them
    os.environ["KTE_BACKEND"] = backend
    os.environ["KTE_USE_HYDE"] = "1" if use_hyde else "0"
    # Optional embedding model override from the UI
    if embed_model_override := st.session_state.get("embed_model_override", None):
        os.environ["KTE_EMBED_MODEL"] = embed_model_override

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
    """Map your UI model_key to the underlying model name. Fall back to (model_key, chosen_backend)."""
    backend = chosen_backend
    name = model_key
    try:
        from kte_proposal import LLM_LIST  # your registry
        if model_key in LLM_LIST:
            cfg = LLM_LIST[model_key]
            backend = "OpenAI" if cfg.get("type") == "openai" else "Ollama"
            name = str(cfg.get("name", model_key))
    except Exception:
        pass
    return name, backend

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
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[OpenAI error] {e}"

    # Ollama (local)
    try:
        import ollama
        resp = ollama.chat(model=model_name, messages=messages)
        return (resp.get("message") or {}).get("content", "")
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
                "Only include fields you change. Keep within length limits and use [P##] citations when relevant."
            )
        else:
            # Q&A mode — just answer; do NOT output JSON unless explicitly asked
            sys_content = (
                "You are a helpful research assistant. Answer the user's question succinctly "
                "using the provided CONTEXT and CURRENT_KTE_JSON when helpful. "
                "Do NOT output JSON unless explicitly asked; just answer."
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
    if st.session_state.get("use_rag_in_chat") and st.session_state.get("retriever"):
        try:
            # Build the query string for retrieval
            qry = prompt

            # Optional HyDE expansion (per-turn, only if enabled)
            if st.session_state.get("use_hyde_in_chat"):
                # Use the current chat backend/model to draft a hypothetical answer
                # (This is the “HyDE” doc we’ll embed + search with)
                hyde_sys = {"role": "system", "content": (
                    "Write a concise, neutral paragraph that could plausibly appear in the manuscript and answer the user's question."
                )}
                hyde_user = {"role": "user", "content": prompt}

                # Reuse your existing chat LLM helper so it works with either OpenAI or Ollama
                effective_backend  = st.session_state.get("chat_backend") or st.session_state.get("backend", "OpenAI")
                effective_modelkey = st.session_state.get("chat_model_key") or st.session_state.get("model_key", "gpt-4o")
                hypothetical = _chat_llm([hyde_sys, hyde_user], backend=effective_backend, model_key=effective_modelkey)

                # Use the hypothetical text for retrieval (you can also combine with the original query if you prefer)
                qry = f"{prompt}\n\n{hypothetical}"

            # Retrieve with (possibly) expanded query
            docs = st.session_state.retriever.get_relevant_documents(qry)

            rag_context = "\n\n---\n".join(
                d.page_content.strip()[:1200] for d in docs if d and getattr(d, "page_content", None)
            )[:4000]  # cap to keep tokens in check
        except Exception as e:
            rag_context = f"(retrieval failed: {e})"

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
                    raw_docs = load_document(doc_path)
                    chunks   = split_sections(raw_docs)
                    vectordb = build_vectordb(chunks)
                    st.session_state.retriever = vectordb.as_retriever(search_kwargs={"k": 6})
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

