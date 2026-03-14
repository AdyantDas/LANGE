import os
import json
import io
import hashlib
import fitz  # PyMuPDF
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from docx import Document
from newspaper import Article
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# ─────────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────────
load_dotenv()
app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory cache keyed by (url_hash, lang, task, tone)
_url_cache: dict = {}

# ─────────────────────────────────────────────
# 2. REQUEST MODELS
# ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    text: str
    target_lang: str
    task: str
    tone: str = "Neutral"

class URLRequest(BaseModel):
    url: str
    target_lang: str
    task: str
    tone: str = "Neutral"

# ─────────────────────────────────────────────
# 3. SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a professional multilingual editor and media analyst.
Return ONLY a valid JSON object — no prose, no markdown fences.

Schema:
{
  "translation": "<translated or summarized text in target language>",
  "bias_score": <integer 0-100, overall opinionation>,
  "obj_label": "<word for Objective in target language>",
  "opi_label": "<word for Opinionated in target language>",
  "bias_categories": {
    "political": <0-100>,
    "emotional": <0-100>,
    "cultural":  <0-100>
  },
  "debiased": "<neutralized rewrite of the ORIGINAL text in the same language as the original>",
  "reading_level": {
    "source": "<e.g. Grade 8 / Academic / Simple>",
    "translated": "<reading level of the translation>"
  },
  "back_translation_score": <0-100, similarity estimate if result were translated back>,
  "confidence": "<High | Medium | Low based on input length>"
}

Rules:
- bias_score 0 = fully objective, 100 = highly opinionated
- debiased must be in the ORIGINAL input language, not the target language
- back_translation_score: 90+ = excellent, 70-89 = good, <70 = significant drift
- confidence: High if input > 200 words, Medium if 50-200 words, Low if < 50 words
- ALWAYS write the translation field in the target language native script
- NEVER transliterate Korean, Japanese, Chinese, Arabic into Latin characters
"""

# ─────────────────────────────────────────────
# 4. CORE LLM CALLER  — Bug 1 fixed
# ─────────────────────────────────────────────
def call_groq(content: str, target_lang: str, task_type: str, tone: str = "Neutral") -> dict:
    # FIX: removed the stray `result = json.loads(raw)` and `return result`
    # that were accidentally placed inside the else branch
    if task_type == "translate":
        instruction = (
            f"Translate this text into {target_lang} using a {tone} tone. "
            f"You MUST write the translation field entirely in {target_lang} native script. "
            f"Do NOT transliterate or use Latin characters."
        )
    else:
        instruction = (
            f"Summarize this text into {target_lang} using a {tone} tone. "
            f"You MUST write the translation field entirely in {target_lang} native script."
        )

    prompt = (
        f"{instruction}. Also analyze the text for bias and produce a debiased rewrite.\n\n"
        f"TEXT:\n{content}"
    )

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        raw = completion.choices[0].message.content

        # Robust parsing — strip accidental markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)

        # Safe defaults for any missing keys
        result.setdefault("translation", "")
        result.setdefault("bias_score", 50)
        result.setdefault("obj_label", "Objective")
        result.setdefault("opi_label", "Opinionated")
        result.setdefault("bias_categories", {"political": 50, "emotional": 50, "cultural": 50})
        result.setdefault("debiased", "")
        result.setdefault("reading_level", {"source": "Unknown", "translated": "Unknown"})
        result.setdefault("back_translation_score", 75)
        result.setdefault("confidence", "Medium")
        return result

    except Exception as e:
        print(f"[Groq error] {e}")
        return {
            "translation": "Error processing request.",
            "bias_score": 50,
            "obj_label": "Objective",
            "opi_label": "Opinionated",
            "bias_categories": {"political": 50, "emotional": 50, "cultural": 50},
            "debiased": "",
            "reading_level": {"source": "Unknown", "translated": "Unknown"},
            "back_translation_score": 0,
            "confidence": "Low",
        }

# ─────────────────────────────────────────────
# 5. ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/ask")
async def ask_groq(request: QueryRequest):
    """Plain text — translate or summarize."""
    result = call_groq(request.text, request.target_lang, request.task, request.tone)
    return JSONResponse(content=result, media_type="application/json; charset=utf-8")


@app.post("/ask-url")
async def ask_groq_url(request: URLRequest):
    """Scrape a URL then process. Results cached per (url, lang, task, tone)."""
    cache_key = hashlib.md5(
        f"{request.url}|{request.target_lang}|{request.task}|{request.tone}".encode()
    ).hexdigest()

    if cache_key in _url_cache:
        print(f"[Cache HIT] {request.url}")
        return JSONResponse(content=_url_cache[cache_key], media_type="application/json; charset=utf-8")

    # FIX Bug 2: scraping isolated in its own try/except
    # so call_groq and caching are never accidentally skipped
    try:
        article = Article(request.url)
        article.download()
        article.parse()
        text = article.text
        if not text.strip():
            raise ValueError("Article body was empty — site may block scrapers")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scrape failed: {str(e)}")

    # FIX Bug 3: use `text` variable, not `article.text`
    result = call_groq(text, request.target_lang, request.task, request.tone)
    _url_cache[cache_key] = result
    return JSONResponse(content=result, media_type="application/json; charset=utf-8")


@app.post("/ask-file")
async def ask_groq_file(
    target_lang: str = Form(...),
    task: str = Form(...),
    tone: str = Form("Neutral"),
    file: UploadFile = File(...),
):
    """Extract text from PDF / DOCX / TXT then process."""
    try:
        filename  = file.filename.lower()
        content   = await file.read()
        extracted = ""

        if filename.endswith(".pdf"):
            with fitz.open(stream=io.BytesIO(content), filetype="pdf") as doc:
                extracted = "".join(page.get_text() for page in doc)
        elif filename.endswith(".docx"):
            doc = Document(io.BytesIO(content))
            extracted = "\n".join(p.text for p in doc.paragraphs)
        else:
            extracted = content.decode("utf-8", errors="replace")

        if not extracted.strip():
            raise ValueError("No text found in file")

        result = call_groq(extracted, target_lang, task, tone)
        return JSONResponse(content=result, media_type="application/json; charset=utf-8")

    except Exception as e:
        print(f"[File error] {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}
