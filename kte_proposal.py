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
from typing import Any, Tuple

from pydantic.v1 import BaseModel, Field, validator

# LangChain core
from langchain_community.document_loaders import (
    PyMuPDFLoader, Docx2txtLoader
)


try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document  # fallback for older LangChain

from langchain.text_splitter import (
    MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
)

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_community.vectorstores import Chroma

try:
    # Chroma requires scalar metadata; this removes dict/list/etc.
    from langchain_community.vectorstores.utils import filter_complex_metadata
except Exception:
    filter_complex_metadata = None


def _env_int(name: str, default: int) -> int:
    """Read an int env var safely."""
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    """Read a float env var safely."""
    try:
        return float(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default

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
    "gpt-4.1-mini": {"backend": "OpenAI", "name": "gpt-4.1-mini"},
    "gpt-4o-mini":  {"backend": "OpenAI", "name": "gpt-4o-mini"},
    "gpt-4o":       {"backend": "OpenAI", "name": "gpt-4o"},

    # ---- Ollama (local) ----
    "gemma3:4b":              {"backend": "Ollama", "name": "gemma3:4b"},
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
        """Soft length guardrails.

        Older versions raised a ValidationError (crashing the whole generation) when a field
        exceeded an approximate word budget. That makes the app brittle because any model
        can occasionally be verbose.

        Instead of raising, we truncate to both:
        - an approximate max word count (max_length/6), AND
        - the Field max_length in characters (if set).
        """
        if not isinstance(v, str):
            return v
        s = v.strip()
        max_len = field.field_info.max_length or None
        if max_len:
            # Approximate word budget used previously
            max_words = max(1, int(max_len / 6))
            words = s.split()
            if len(words) > max_words:
                s = " ".join(words[:max_words]).strip()
            # Hard char clamp (pydantic also enforces max_length for strings)
            if len(s) > max_len:
                s2 = s[:max_len].rstrip()
                # avoid cutting in the middle of a word if possible
                if " " in s2:
                    s2 = s2.rsplit(" ", 1)[0].strip()
                s = s2 if s2 else s[:max_len]
        return s


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


_CITE_TOKEN_RE = re.compile(r"\[P\d{4}\]")

_CITE_FRAGMENT_RE = re.compile(r"\[P\d{0,4}$")


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    # Primary cut
    clipped = s[:n].rstrip()

    # If we cut THROUGH a citation token, keep the full token instead of dropping it.
    # Example: "... blah [P000" -> include "[P0007]" fully.
    for m in _CITE_TOKEN_RE.finditer(s):
        if m.start() < n < m.end():
            clipped = (s[:m.start()] + m.group(0)).rstrip()
            break

    # Avoid leaving a broken citation fragment at the end (e.g., "[P00")
    m2 = _CITE_FRAGMENT_RE.search(clipped)
    if m2 and "]" not in clipped[m2.start():]:
        clipped = clipped[:m2.start()].rstrip()
    # Avoid dangling '[' without a matching ']'
    if clipped.count("[") > clipped.count("]"):
        clipped = clipped.rsplit("[", 1)[0].rstrip()
    return clipped

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



def _try_import_fitz():
    try:
        import fitz  # PyMuPDF
        return fitz
    except Exception:
        return None

def _page_has_images(page) -> bool:
    # Fast check: embedded images list
    try:
        imgs = page.get_images(full=True)
        return bool(imgs)
    except Exception:
        return False

def _detect_image_regions(page, min_area_ratio: float = 0.08, max_regions: int = 4):
    """
    Return (regions, debug) where regions is list of Rects likely containing figures/tables.
    Uses PyMuPDF "rawdict" image blocks when available; falls back to whole-page for scans.
    """
    fitz = _try_import_fitz()
    if not fitz:
        return [], []

    rect = page.rect
    page_area = float(rect.get_area()) if hasattr(rect, "get_area") else float(rect.width * rect.height)

    regions: list[Tuple[float, Any]] = []
    debug: list[dict] = []

    # 1) Best: image blocks with bounding boxes
    try:
        raw = page.get_text("rawdict")
        for b in raw.get("blocks", []):
            if b.get("type") == 1 and "bbox" in b:  # 1 = image block
                x0, y0, x1, y1 = b["bbox"]
                r = fitz.Rect(x0, y0, x1, y1)
                area = float(r.get_area())
                ratio = (area / page_area) if page_area else 0.0
                regions.append((ratio, r))
                debug.append({"kind": "block", "bbox": [x0, y0, x1, y1], "area_ratio": round(ratio, 4)})
    except Exception:
        pass

    # 2) Fallback: if page seems scanned/image-only, treat the whole page as one region
    try:
        txt = (page.get_text("text") or "").strip()
    except Exception:
        txt = ""
    if (len(txt) < 40) and _page_has_images(page):
        regions = [(1.0, rect)]
        debug = [{"kind": "scan_page", "bbox": [rect.x0, rect.y0, rect.x1, rect.y1], "area_ratio": 1.0}]

    regions.sort(key=lambda x: x[0], reverse=True)
    keep = [(ratio, r) for (ratio, r) in regions if ratio >= float(min_area_ratio)]
    keep = keep[: int(max_regions)]
    keep_rects = [r for _, r in keep]

    # Filter debug to only kept rects (best-effort)
    kept_debug = debug[: len(keep_rects)] if debug else []
    return keep_rects, kept_debug

def _render_region_png(page, rect=None, dpi: int = 220) -> bytes:
    fitz = _try_import_fitz()
    if not fitz:
        return b""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect)
    return pix.tobytes("png")

def _ocr_png_bytes(png_bytes: bytes) -> str:
    
    """Local OCR (recommended default). Requires:
    pip install pytesseract pillow
    And Tesseract installed on the OS.
    """
    try:
        import io
        from PIL import Image
        import pytesseract
        img = Image.open(io.BytesIO(png_bytes))
        txt = pytesseract.image_to_string(img)
        return (txt or "").strip()
    except Exception:
        return ""

def _ollama_resp_text(resp) -> str:
    """Handle both dict-style and pydantic-object responses from ollama-python."""
    try:
        # Newer ollama-python: response.message.content
        if hasattr(resp, "message") and hasattr(resp.message, "content"):
            return (resp.message.content or "").strip()
    except Exception:
        pass
    try:
        # Dict-style fallback
        if isinstance(resp, dict):
            return ((resp.get("message") or {}).get("content") or "").strip()
    except Exception:
        pass
    return ""

def _vision_extract_png_bytes(png_bytes: bytes) -> str:
    
    """Optional: extract text from a figure/table image region.
    Backend is UI-controlled via env:
      KTE_VISION_BACKEND = openai | ollama   (default: openai)
      KTE_VISION_MODEL   = gpt-4o | gemma3:4b | ... (default depends on backend)
     """
    try:
        backend = os.getenv("KTE_VISION_BACKEND", "openai").strip().lower()
        prompt = (
            "Extract all readable text from this figure/table. "
            "If it is a table, output a clean TSV (tab-separated). "
            "If it is a plot, describe axes and key numeric values mentioned."
        )
        if backend == "ollama":
            # Ollama multimodal: pass image bytes via the "images" key
            # (Requires a vision-capable model like gemma3:4b and Ollama installed.)
            import ollama
            model = os.getenv("KTE_VISION_MODEL", "gemma3:4b")
            resp = ollama.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [png_bytes],
                }],
            )
            return _ollama_resp_text(resp)

        # Default: OpenAI-compatible vision
        import base64
        from openai import OpenAI
        client = OpenAI()
        model = os.getenv("KTE_VISION_MODEL", "gpt-4o")
        b64 = base64.b64encode(png_bytes).decode("utf-8")

        # Try Responses API first (newer); fall back to chat.completions
        if hasattr(client, "responses"):
            resp = client.responses.create(
                model=model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}
                    ],
                }],
            )
            return (getattr(resp, "output_text", "") or "").strip()
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ],
            }],
            temperature=0,
        )
        return ((resp.choices[0].message.content) or "").strip()
    except Exception:
        return ""

def load_pdf_hybrid(path: str) -> list[Document]:
    """
    Hybrid loader:
    - Extract normal text
    - Detect big image regions (figures/tables/scans)
    - OCR (or vision) only those regions and append as 'FIGURE_TEXT' blocks
    Controlled by env:
      KTE_PDF_IMAGE_MODE = none | ocr | vision | auto   (default: none)
      KTE_PDF_IMAGE_MIN_AREA = 0.08  (default)
      KTE_PDF_IMAGE_MAX_REGIONS = 4  (default)
    """
    fitz = _try_import_fitz()
    if not fitz:
        # If fitz missing, fallback to old loader
        return PyMuPDFLoader(path).load()

    mode = os.getenv("KTE_PDF_IMAGE_MODE", "none").strip().lower()
    min_area = float(os.getenv("KTE_PDF_IMAGE_MIN_AREA", "0.08"))
    max_regions = int(os.getenv("KTE_PDF_IMAGE_MAX_REGIONS", "4"))
    vision_backend = os.getenv("KTE_VISION_BACKEND", "openai").strip().lower()

    doc = fitz.open(path)
    out_docs: list[Document] = []
    for i in range(doc.page_count):
        page = doc[i]
        try:
            text = (page.get_text("text") or "").strip()
        except Exception:
            text = ""

        regions, regions_dbg = _detect_image_regions(page, min_area_ratio=min_area, max_regions=max_regions)

        figure_blocks = []
        did_any = False
        if mode in {"ocr", "vision", "auto"} and regions:
            # auto: OCR only when page is likely scanned (low text) OR large regions exist
            do_extract = (mode != "auto") or (len(text) < 200) or (len(regions) > 0)
            if do_extract:
                for ridx, rect in enumerate(regions, start=1):
                    png = _render_region_png(page, rect=rect, dpi=220)
                    if not png:
                        continue

                    extracted = ""
                    if mode == "vision":
                        extracted = _vision_extract_png_bytes(png)
                    elif mode == "auto":
                        # If vision is configured (OpenAI key OR Ollama backend), try vision first, then OCR fallback.
                        can_vision = (vision_backend == "ollama") or bool(os.getenv("OPENAI_API_KEY"))
                        if can_vision:
                            extracted = _vision_extract_png_bytes(png)
                        if not extracted:
                            extracted = _ocr_png_bytes(png)
                    else:
                        extracted = _ocr_png_bytes(png)
                        
                    extracted = (extracted or "").strip()
                    if extracted:
                        did_any = True
                        # keep figure text from exploding your chunk sizes
                        extracted = extracted[:1500]
                        figure_blocks.append(
                            f"\n\n[FIGURE_TEXT page={i+1} region={ridx}]\n{extracted}\n"
                        )

        combined = text + ("".join(figure_blocks) if figure_blocks else "")
        out_docs.append(Document(
            page_content=combined,
            metadata={
                "source": path,
                "page": i,
                "has_images": bool(regions),
                "img_mode": mode,
                "img_vision_backend": vision_backend,
                "img_extracted": bool(did_any),
                # Chroma metadata must be scalar types (no list/dict).
                # Keep region debug as JSON string so it’s still available for UI/debug.
                "img_regions_json": json.dumps(regions_dbg, ensure_ascii=False),
                "img_regions_n": int(len(regions_dbg) if regions_dbg else 0),
            }
        ))
    doc.close()
    return out_docs

def load_document(path: str):
    """Return a list[Document] from PDF or DOCX."""
    if path.lower().endswith(".pdf"):

        # Hybrid image-aware path (optional)
        mode = os.getenv("KTE_PDF_IMAGE_MODE", "none").strip().lower()
        if mode in {"ocr", "vision", "auto"}:
            return load_pdf_hybrid(path)
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
            split_docs = HEADER_SPLITTER.split_text(d.page_content)
            base_meta = dict(d.metadata or {})
            for sd in split_docs:
                sd.metadata = {**base_meta, **dict(sd.metadata or {})}
            docs.extend(split_docs)
        else:
            docs.append(d)
    # fine‑grained splitter (prevents >4k‑token chunks)
    
    chunk_size = _env_int("KTE_CHUNK_SIZE", 800)
    chunk_overlap = _env_int("KTE_CHUNK_OVERLAP", 80)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)



# ---------------------------------------------------------------------#
# 2.5 -- Stable paragraph IDs + citeable context formatting             #
# ---------------------------------------------------------------------#
def assign_pids(chunks):
    """
    Attach stable, citeable IDs to each chunk for the current document,
    e.g. P0001, P0002, ... stored in Document.metadata["pid"].
    """
    for i, d in enumerate(chunks, start=1):
        d.metadata = dict(d.metadata or {})
        d.metadata["pid"] = f"P{i:04d}"
    return chunks

def format_context(docs, per_doc_chars: int = 1200, max_chars: int = 6000) -> str:
    """
    Render retrieved docs into citeable blocks:
      [P0007] page=3
      <text...>
    """
    blocks = []
    for d in (docs or []):
        if not d or not getattr(d, "page_content", None):
            continue
        meta = d.metadata or {}
        pid = meta.get("pid", "P????")
        page = meta.get("page")
        page_str = f" page={page+1}" if isinstance(page, int) else ""
        txt = (d.page_content or "").strip()
        txt = txt[:per_doc_chars]
        blocks.append(f"[{pid}]{page_str}\n{txt}")
    ctx = "\n\n---\n".join(blocks)
    return ctx[:max_chars]

def validate_citations(text: str, docs) -> tuple[bool, set[str], set[str], set[str]]:
    """
    Returns: (ok, cited_tokens, invalid_tokens, valid_tokens)
    Where tokens look like "[P0007]".
    """
    cited = set(_CITE_TOKEN_RE.findall(text or ""))
    valid = {f"[{(d.metadata or {}).get('pid')}]" for d in (docs or []) if (d and (d.metadata or {}).get("pid"))}
    invalid = cited - valid
    return (len(invalid) == 0), cited, invalid, valid

def sanitize_citations(text: str, valid_tokens: set[str]) -> str:
    """Remove any citation tokens not present in valid_tokens (hard guarantee)."""
    text = (text or "").strip()
    if not text:
        return text
    if not valid_tokens:
        return _CITE_TOKEN_RE.sub("", text).strip()
    return re.sub(
        r"\[P\d{4}\]",
        lambda m: m.group(0) if m.group(0) in valid_tokens else "",
        text,
    ).strip()

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
    # Import defensively (LangChain packaging differs by version)
    try:
        from langchain_ollama import OllamaEmbeddings  # preferred in newer installs
    except Exception:
        from langchain_community.embeddings import OllamaEmbeddings  # older/alt installs
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    raw = os.getenv("KTE_EMBED_MODEL", EMBED_MODEL).strip()
    name = raw.lower()
    
    # Convenience aliases: allow HF-style names in the UI, but still use Ollama when possible
    if name in {"baai/bge-m3", "flagembedding/bge-m3"}:
        raw = "bge-m3"
        name = "bge-m3"


    # Common typo alias
    if name in {"gbe-m3", "bge_m3"}:
        raw = "bge-m3"
        name = "bge-m3"

    # Explicit HuggingFace/Sentence-Transformers selection via prefix:
    # e.g., hf:intfloat/multilingual-e5-small
    if name.startswith("hf:"):
        hf_name = raw.split(":", 1)[1].strip()
        return SentenceTransformerEmbeddings(model_name=hf_name)

    # OpenAI path (cloud-friendly). Requires OPENAI_API_KEY.
    # e.g., set KTE_EMBED_MODEL=text-embedding-3-small
    if name.startswith("text-embedding-") or name in {"openai", "openai-emb"}:
        from langchain_openai import OpenAIEmbeddings
        model = name if name.startswith("text-embedding-") else "text-embedding-3-small"
        return OpenAIEmbeddings(model=model)  # reads OPENAI_API_KEY

    # Local default via Ollama
    try:
        return OllamaEmbeddings(model=raw)
    except Exception:
        
        # If it looks like a HF repo name, try loading it locally; else fall back.
        if "/" in raw:
            try:
                return SentenceTransformerEmbeddings(model_name=raw)
            except Exception:
                pass
        return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")



def _sanitize_for_embedding(text: str) -> str:
    """Sanitize text before sending it to an embedding model.

    Why this exists:
    - Some Ollama embedding endpoints have been observed to return NaNs for certain
      pathological inputs (control chars, lots of formatting marks, extremely long "words").
    - PDF extractors sometimes produce weird Unicode (directionality marks, zero-width chars)
      and very long tokens (no spaces) that can stress tokenizers.

    This function:
    - Normalizes Unicode (NFKC)
    - Removes control + format characters (Cc/Cf)
    - Normalizes whitespace
    - Breaks extremely long tokens into smaller pieces
    """
    import re
    import unicodedata

    if text is None:
        return ""

    # Unicode normalize
    try:
        text = unicodedata.normalize("NFKC", str(text))
    except Exception:
        text = str(text)

    # Replace NUL explicitly
    text = text.replace("\x00", " ")

    # Remove control chars + format chars (directionality/zero-width)
    cleaned_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Cc", "Cf"):
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(ch)
    text = "".join(cleaned_chars)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    # Break extremely long tokens (common in some PDF extracts)
    toks = text.split(" ")
    out = []
    for t in toks:
        if len(t) <= 120:
            out.append(t)
            continue
        for i in range(0, len(t), 60):
            out.append(t[i:i+60])
    text = " ".join(out).strip()

    # Guard: skip strings that end up with no letters/digits (punctuation-only)
    if sum(1 for ch in text if ch.isalnum()) < 8:
        return ""

    return text



def _is_finite_vector(vec) -> bool:
    try:
        for v in vec:
            # v!=v detects NaN
            if v != v or v == float("inf") or v == float("-inf"):
                return False
        return True
    except Exception:
        return False


def _try_embed_with_backoff(emb, text: str, max_chars: int):
    """
    Ollama's embed endpoint can occasionally return NaNs (server 500) for very long
    or pathological inputs. Retry with progressively shorter text.
    Returns (vector, used_text).
    """
    text = _sanitize_for_embedding(text)
    if not text:
        raise ValueError("Empty text after sanitization")

    # Conservative backoff sizes (chars). Final fallback is small but usually safe.
    sizes = [max_chars, max_chars // 2, max(800, max_chars // 4), 800]
    last_err = None
    for n in sizes:
        t = text if len(text) <= n else text[:n]
        t = t.strip()
        if not t:
            continue
        try:
            vec = emb.embed_query(t)  # single-item request (isolates failures)
            if not _is_finite_vector(vec):
                raise ValueError("Non-finite values in embedding vector")
            return vec, t
        except Exception as e:
            last_err = e
            continue
    # re-raise last error for visibility
    raise last_err if last_err else RuntimeError("Embedding failed")


def build_vectordb(chunks, _failover_done: bool = False):
    """
    Build a Chroma vector DB from chunks, but robust to Ollama embedding NaN/500 issues.
    - Sanitizes text
    - Truncates overly long texts (env: KTE_EMBED_MAX_CHARS, default 6000)
    - Embeds one-by-one with backoff retries; skips chunks that still fail
    """
    import os, json, uuid, math
    from langchain_community.vectorstores import Chroma

    emb = get_embeddings()

    # --- Probe embedding early and fallback if the selected embedding model/server is broken ---
    # This avoids the "No valid chunks to embed" crash when *every* embedding call fails (e.g., NaN/500 from Ollama).
    try:
        _try_embed_with_backoff(emb, "embedding probe", max_chars=128)
    except Exception as e:
        fallback_model = os.getenv("KTE_EMBED_FALLBACK_MODEL", "nomic-embed-text").strip() or None
        selected_model = os.getenv("KTE_EMBED_MODEL", EMBED_MODEL).strip()
        if fallback_model and fallback_model != selected_model:
            if os.getenv("KTE_DEBUG", "0") == "1":
                print(f"[build_vectordb] embedding probe failed for '{selected_model}': {e}")
                print(f"[build_vectordb] falling back to '{fallback_model}'")
            os.environ["KTE_EMBED_MODEL"] = fallback_model
            emb = get_embeddings()
            # If the fallback also fails, raise the original error for visibility
            _try_embed_with_backoff(emb, "embedding probe", max_chars=128)
        else:
            raise


    # Ensure metadata is Chroma-compatible (str/int/float/bool only).
    if filter_complex_metadata is not None:
        try:
            chunks = filter_complex_metadata(chunks)
        except Exception:
            pass
    else:
        # Fallback: serialize any complex metadata values to strings
        cleaned = []
        for d in chunks:
            meta = {}
            for k, v in (d.metadata or {}).items():
                if v is None or isinstance(v, (str, int, float, bool)):
                    # Avoid NaN/Inf inside metadata too (Chroma/JSON hates it)
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        meta[k] = str(v)
                    else:
                        meta[k] = v
                else:
                    meta[k] = json.dumps(v, ensure_ascii=False, default=str)
            d.metadata = meta
            cleaned.append(d)
        chunks = cleaned

    max_chars = int(os.getenv("KTE_EMBED_MAX_CHARS", "6000"))
    skip_policy = os.getenv("KTE_EMBED_SKIP_POLICY", "skip").strip().lower()  # skip | placeholder

    texts = []
    metadatas = []
    ids = []
    embeddings = []
    skipped = 0
    seen_ids = set()
    embed_errors = []

    for d in (chunks or []):
        txt = _sanitize_for_embedding(getattr(d, "page_content", "") or "")
        if not txt:
            skipped += 1
            continue

        # Stable ID: prefer pid, else generate
        pid = (getattr(d, "metadata", None) or {}).get("pid") or f"X{uuid.uuid4().hex[:10]}"
        pid = str(pid)
        if pid in seen_ids:
            # ensure unique IDs for Chroma
            pid = f"{pid}_{uuid.uuid4().hex[:6]}"
        seen_ids.add(pid)

        try:
            vec, used_txt = _try_embed_with_backoff(emb, txt, max_chars=max_chars)
            embeddings.append(vec)
            texts.append(used_txt)  # store the same text we embedded (ensures consistency)
            metadatas.append(d.metadata or {})
            ids.append(pid)
        except Exception as e:
            skipped += 1
            if len(embed_errors) < 8:
                page = (d.metadata or {}).get("page") if hasattr(d, "metadata") else None
                embed_errors.append((pid, page, len(txt), str(e)))
            if skip_policy == "placeholder":
                # Embed a tiny safe placeholder so indexing continues
                try:
                    vec, _ = _try_embed_with_backoff(emb, "placeholder", max_chars=64)
                    embeddings.append(vec)
                    texts.append("placeholder")
                    md = dict(d.metadata or {})
                    md["_embed_error"] = str(e)
                    metadatas.append(md)
                    ids.append(pid)
                except Exception:
                    # even placeholder failed; truly skip
                    continue
            else:
                # skip
                continue
    # --- If most embeddings failed, switch to fallback model and retry ---
    model_name = os.getenv("KTE_EMBED_MODEL", EMBED_MODEL).strip()
    fallback_model = os.getenv("KTE_EMBED_FALLBACK_MODEL", "").strip()
    total_chunks = len(chunks or [])
    embedded_n = len(texts)
    attempted_n = embedded_n + skipped
    success_rate = (embedded_n / attempted_n) if attempted_n else 0.0

    try:
        min_embedded = int(os.getenv("KTE_EMBED_MIN_EMBEDDED", "10"))
    except Exception:
        min_embedded = 10
    try:
        min_success_rate = float(os.getenv("KTE_EMBED_MIN_SUCCESS_RATE", "0.25"))
    except Exception:
        min_success_rate = 0.25

    # Require at least ~20% of chunks, but never more than min_embedded.
    desired_min = min(min_embedded, max(1, int(0.2 * total_chunks))) if total_chunks else 1

    if (not _failover_done) and fallback_model and (fallback_model != model_name):
        if (embedded_n < desired_min) or (success_rate < min_success_rate):
            if os.getenv("KTE_DEBUG", "0") == "1":
                print(
                    f"[build_vectordb] low embedding yield with '{model_name}': "
                    f"embedded={embedded_n}/{attempted_n} (rate={success_rate:.2f}); "
                    f"retrying with fallback '{fallback_model}'"
                )
            os.environ["KTE_EMBED_MODEL"] = fallback_model
            return build_vectordb(chunks, _failover_done=True)


    if not texts:
        model_name = os.getenv("KTE_EMBED_MODEL", EMBED_MODEL).strip()
        hint = (
            "Check that the model name is correct (e.g., bge-m3, nomic-embed-text) and that Ollama is running. "
            "You can set KTE_DEBUG=1 for verbose logs. "
            "You can also set KTE_EMBED_FALLBACK_MODEL (e.g., nomic-embed-text or hf:BAAI/bge-m3) for failover."
        )

        # IMPORTANT: Some setups return NaN/500 for many real-document inputs even though a tiny probe works.
        # If we embedded ZERO chunks, do a full failover retry with the fallback model.
        fallback_model = os.getenv("KTE_EMBED_FALLBACK_MODEL", "nomic-embed-text").strip()

        if (not _failover_done) and fallback_model and (fallback_model != model_name):
            if os.getenv("KTE_DEBUG", "0") == "1":
                print(f"[build_vectordb] embedded=0 with '{model_name}'. Retrying with fallback '{fallback_model}' ...")
            os.environ["KTE_EMBED_MODEL"] = fallback_model
            return build_vectordb(chunks, _failover_done=True)

        if embed_errors:
            sample = "\n".join(
                [f"{pid} page={page} chars={n_chars} err={err}" for (pid, page, n_chars, err) in embed_errors[:8]]
            )
            raise RuntimeError(
                "No valid chunks to embed (all were empty or embedding failed). "
                f"embed_model={model_name}. "
                + hint
                + "\nSample embedding failures:\n"
                + sample
            )
        raise RuntimeError(
            "No valid chunks to embed (all were empty or embedding failed). "
            f"embed_model={model_name}. "
            + hint
        )

    # Create DB (embedding function needed later for similarity_search/querying)
    vectordb = Chroma(collection_name="kte_docs", embedding_function=emb)

    # Add precomputed embeddings directly to avoid batch embed failures
    vectordb._collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    if os.getenv("KTE_DEBUG", "0") == "1":
        print(f"[build_vectordb] embedded={len(texts)} skipped={skipped} max_chars={max_chars}")
        if embed_errors:
            print("[build_vectordb] sample embed failures (pid, page, n_chars, err):")
            for it in embed_errors:
                print("  -", it)

    return vectordb


def make_hyde_search(llm, vectordb, k=4):
    """HyDE retriever with guardrails.

    If HyDE generation or embedding fails (including Ollama NaN/500), we fall back to
    normal similarity_search on the original query.
    """
    hyde_max_chars = _env_int("KTE_HYDE_MAX_CHARS", 2000)

    def search(query: str):
        def _hyde_bad(t: str) -> bool:
            tl = (t or '').strip().lower()
            if len(tl) < 40:
                return True
            bad = [
                "please provide", "provide the paper", "provide the text", "provide the document", "upload the",
                "send the pdf", "share the pdf", "share the paper",
                "i don't have access", "i do not have access", "i can't access", "cannot access",
                "need more context", "need the text", "need the paper", "need the document",
                "as an ai", "as a language model",
                "لطفا", "ارسال کنید", "متن را", "سند را", "فایل را", "در اختیار", "ندارم",
            ]
            return any(b in tl for b in bad)

        hyde_prompt = (
            "You are generating a Hypothetical Document ONLY for retrieval (HyDE). You are NOT answering the user.\n"
            "Never ask for the paper/PDF/text and never mention that you lack access.\n"
            "If the question is underspecified, make reasonable assumptions and STILL write plausible manuscript text.\n"
            "Do NOT use placeholders like [AUTHOR], [YEAR], 'TBD'.\n\n"
            "OUTPUT FORMAT (STRICT):\n"
            "Line 1: 12-20 semicolon-separated keywords/phrases likely to appear in the manuscript.\n"
            "Line 2: (blank)\n"
            "Line 3: ONE paragraph (3-6 sentences) in academic style that could plausibly appear in the paper.\n"
            "No other lines.\n\n"
            f"QUESTION:\n{query}\n"
        )

        try:
            hypo = (llm.invoke(hyde_prompt).content or "").strip()
            if _hyde_bad(hypo):
                hyde_prompt_retry = (
                "Your previous output violated the rules (e.g., asked for the document or refused).\n"
                "Rewrite it and FOLLOW THE STRICT OUTPUT FORMAT EXACTLY.\n\n"
                + hyde_prompt
                )
                hypo2 = (llm.invoke(hyde_prompt_retry).content or "").strip()
                if not _hyde_bad(hypo2):
                    hypo = hypo2
                else:
                    hypo = ""

            hypo = _sanitize_for_embedding(hypo)
            if not hypo:
                return vectordb.similarity_search(query, k=k)
            if len(hypo) > hyde_max_chars:
                hypo = hypo[:hyde_max_chars].strip()

            # Embedding HyDE text can occasionally trigger Ollama NaN/500.
            try:
                vec = vectordb._embedding_function.embed_query(hypo)
                return vectordb.similarity_search_by_vector(vec, k=k)
            except Exception:
                return vectordb.similarity_search(query, k=k)
        except Exception:
            return vectordb.similarity_search(query, k=k)

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

Write ONLY the content for "{field_name}" (do NOT include XML/angle-bracket tags).
Citations:
- Put citations immediately after the sentence they support (inline), not only at the end.
- You MAY cite sources using ONLY the IDs that appear in the Context blocks (e.g., [P0007]).
- Never invent citations. If a specific fact is not present in Context, write 'گزارش نشده'.
-Terminology rule: do NOT expand abbreviations unless the expansion appears verbatim in Context.
"""

FIELD_PROMPTS: Dict[str, str] = {
    "title": "the paper's full title (NOT the journal name like 'Scientific Reports'); include study setting only if explicitly stated near the title/abstract",
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
    """Return an LLM client from the registry. Falls back to key as raw name.

    Notes:
    - Some Ollama backends/models can error at temperature=0 (greedy) in rare cases.
      We therefore use a small default temperature and allow override via KTE_TEMPERATURE.
    """
    cfg = LLM_REG.get(model_key, {"backend": "Ollama", "name": model_key})

    if cfg["backend"] == "OpenAI":
        # Keep deterministic by default for form filling
        return ChatOpenAI(model_name=cfg["name"], temperature=0)

    # Ollama (local)
    temp = _env_float("KTE_TEMPERATURE", 0.2)
    if temp < 0.01:
        temp = 0.01

    return ChatOllama(
        model=cfg["name"],
        num_ctx=_env_int("KTE_OLLAMA_NUM_CTX", 4096),
        num_predict=_env_int("KTE_MAX_OUTPUT_TOKENS", 1024),
        temperature=temp,
    )

# ---------------------------------------------------------------------#
# 6.‑‑ Driver function                                                 #
# ---------------------------------------------------------------------#




def generate_kte(path: str, model_key: str = "llama3-med", use_hyde: bool | None = None) -> KTEForm:
    # Resolve HyDE flag: CLI/Streamlit may pass it; otherwise read env (default True)
    if use_hyde is None:
        env = os.getenv("KTE_USE_HYDE", "1")
        use_hyde = env.strip().lower() not in {"0", "false", "no"}

    raw_docs = load_document(path)
    chunks   = assign_pids(split_sections(raw_docs))
    vectordb = build_vectordb(chunks)
    llm      = choose_llm(model_key)
    
    # RAG controls (tunable via env vars / Streamlit UI)
    rag_k = _env_int("KTE_RAG_K", 4)
    rag_per_doc_chars = _env_int("KTE_RAG_PER_DOC_CHARS", 1200)
    rag_max_chars = _env_int("KTE_RAG_MAX_CHARS", 6000)


    # Choose retrieval based on the HyDE flag
    def plain_search(q: str):
        return vectordb.similarity_search(q, k=rag_k)

    try:
        search = make_hyde_search(llm, vectordb, k=rag_k) if use_hyde else plain_search
    except NameError:
        # if make_hyde_search isn’t defined, just fall back gracefully
        search = plain_search

    results = {}
    for field in FIELD_PROMPTS.keys():
        
        # Title is almost always on page 0/1; avoid HyDE drift + license/footer retrieval.
        if field == "title":
            front = [d for d in chunks if d.metadata.get("page") in (0, 1)]
            docs = (front[:10] if front else chunks[:10])
        else:
            docs = search(FIELD_PROMPTS.get(field, f"context for {field}"))
            
        context = format_context(docs, per_doc_chars=rag_per_doc_chars, max_chars=rag_max_chars)
        prompt_txt = get_prompt(field).format(context=context)
        answer = llm.invoke(prompt_txt).content

        # Verify citations are real (only from retrieved Context IDs). Repair once if needed.
        ok, cited, invalid, valid = validate_citations(answer, docs)
        if not ok and valid:
            allowed = ", ".join(sorted(valid))
            repair_txt = (
                "Your answer contained citations that do not exist in the provided CONTEXT.\n"
                f"Only cite from: {allowed}\n"
                "Never invent citations. If a fact is not present in CONTEXT, write 'گزارش نشده'.\n\n"
                + prompt_txt
            )
            answer = llm.invoke(repair_txt).content
            
        # Hard guarantee: never output invented citations
        if isinstance(answer, str):
            answer = sanitize_citations(answer, valid)


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
 
