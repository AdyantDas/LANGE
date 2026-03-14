# 🌐 LANGE — AI News Editor

Translate and summarize news articles across languages with real-time bias detection.

## Features
- 🌍 Translate to 14 languages with dialect support
- 📊 Multi-category bias analysis (Political, Emotional, Cultural)
- ✍️ Debiased rewrite suggestions
- 🔄 Cross-language comparison mode
- 📎 Supports text, URLs, PDF, DOCX

## Setup
1. Clone the repo
2. Create a `.env` file with `GROQ_API_KEY=your_key`
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `uvicorn program:app --reload`
5. Open `index.html` in your browser

## Tech Stack
- Frontend: React, Tailwind CSS
- Backend: FastAPI, Groq (LLaMA 3.3 70B)
