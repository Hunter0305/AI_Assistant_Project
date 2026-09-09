# AI Customer Service Assistant
## Nullclass Gen-AI Internship Project

> **Extended from the original Nullclass customer-service chatbot training project**  
> Foundation preserved · Six capabilities added · One unified application

---

## 🎯 Project Overview

This project extends the training **Nullclass Customer Service Chatbot** (a LangChain + Google PaLM + FAISS FAQ system) into a comprehensive AI assistant with six additional capabilities required by the internship curriculum.

The original chatbot answered learner questions about Nullclass courses using a CSV FAQ dataset. This project preserves that foundation and progressively enhances it with:

1. **Sentiment-Aware Responses** — detects customer emotion and adapts tone
2. **Medical Q&A** — specialized health information using MedQuAD dataset
3. **Dynamic Knowledge Base** — add new documents without rebuilding from scratch
4. **Scientific Research Assistant** — semantic arXiv paper retrieval and explanation
5. **Multimodal Reasoning** — image + text understanding with Gemini Vision
6. **Multilingual Chat** — English, Hindi, Marathi, Spanish with context preservation

---

## 🏛️ Training Project Foundation

| Original Component | Status | Change |
|---|---|---|
| `langchain_helper.py` | Modernized → `src/core/chatbot.py` | Updated deprecated APIs |
| `GooglePalm` LLM | Replaced → `gemini-1.5-flash` | PaLM deprecated; Gemini is successor |
| `HuggingFaceInstructEmbeddings` | Replaced → `all-MiniLM-L6-v2` | Lighter, maintained, CPU-compatible |
| `langchain==0.0.339` | Updated → `langchain>=0.2.0` | Modern API compatibility |
| `dataset.csv` (Nullclass FAQ) | Preserved | Core knowledge base unchanged |
| FAISS vector store | Preserved | Same technology, extended usage |
| Streamlit UI | Extended | Multi-mode sidebar navigation |

---

## ✨ Features

- 🏢 **Customer Support** — RAG over Nullclass FAQ with sentiment-aware responses
- 🏥 **Medical Q&A** — MedQuAD dataset, entity recognition, safety disclaimers
- 🔬 **Research Assistant** — arXiv CS paper search, summarization, visualizations
- 🖼️ **Multimodal** — Image upload + analysis with Gemini Vision, follow-up context
- 🌍 **Multilingual** — Auto-detect Hindi/Marathi/Spanish/English, language switching
- 📚 **Knowledge Base Management** — Dynamic ingestion of text/PDF/URLs, scheduler

---

## 🏗️ Architecture

```
internship_project/
├── app.py                         # Main Streamlit entry point
├── requirements.txt
│
├── data/
│   └── customer_support/dataset.csv  # Original training dataset
│
├── src/
│   ├── core/          # chatbot.py (modernized foundation), conversation.py, config.py
│   ├── retrieval/     # embeddings.py, vector_store.py, retriever.py
│   ├── sentiment/     # analyzer.py (VADER + TextBlob)
│   ├── medical/       # ingestion.py, retrieval.py, entities.py
│   ├── knowledge/     # ingestion.py, updater.py, scheduler.py
│   ├── research/      # arxiv_loader.py, retrieval.py, summarizer.py, visualization.py
│   ├── multimodal/    # image_processor.py, reasoning.py, validator.py
│   └── multilingual/  # detector.py, processor.py, context.py
│
├── tests/             # pytest test suite for all 6 tasks
└── scripts/           # build_indexes.py, update_knowledge.py
```

---

## 🛠️ Technology Stack

| Component | Technology | Notes |
|---|---|---|
| LLM | Gemini 1.5 Flash | Free tier, replaces deprecated PaLM |
| Embeddings | `all-MiniLM-L6-v2` | ~80MB, CPU-only, sentence-transformers |
| Vector DB | FAISS (CPU) | Same as training project |
| Framework | LangChain >= 0.2 | Modernized from 0.0.339 |
| UI | Streamlit >= 1.35 | Single unified app |
| Sentiment | VADER + TextBlob | Open-source, no GPU |
| Medical NER | Custom patterns + spaCy | CPU-capable en_core_web_sm |
| Language Detection | langdetect | 55 languages, open-source |
| Translation | deep-translator | Google Translate wrapper, free |
| Image Analysis | Gemini 1.5 Flash Vision | Same API key as LLM |
| Visualizations | Plotly | Interactive charts |

---

## 📋 Project Structure

See Architecture section above. Original training project preserved at:  
`customer_service_chatbot_LLM/` (untouched reference)

---

## ⚙️ Installation

### 1. Prerequisites
- Python 3.9+
- pip

### 2. Navigate to project
```bash
cd internship_project
```

### 3. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Install spaCy model (for medical NER)
```bash
python -m spacy download en_core_web_sm
```

### 6. Download TextBlob corpora (for sentiment)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

Edit `.env`:
```
GOOGLE_API_KEY=your_key_from_aistudio.google.com
```

Get a free API key at: https://aistudio.google.com/

**NEVER commit your `.env` file. It is in `.gitignore`.**

---

## 📊 Dataset Setup

### Customer Support (automatic)
The Nullclass FAQ dataset is included at `data/customer_support/dataset.csv`.

### Medical Q&A (automatic download)
MedQuAD data is downloaded automatically from GitHub on first use.  
Or pre-build the index:
```bash
python scripts/build_indexes.py --medical
```

### Research (automatic download)
arXiv papers are fetched via the arXiv API on first use.  
Or pre-build:
```bash
python scripts/build_indexes.py --research
```

### Build all indexes at once
```bash
python scripts/build_indexes.py --all
```

---

## 🚀 Running the Application

```bash
cd internship_project
streamlit run app.py
```

The app opens at `http://localhost:8501`

**First-time setup:**
1. Click "Build Customer Support Index" in the sidebar
2. Switch to Medical Q&A → click "Build Medical Index"
3. Switch to Research → click "Build Research Index"

---

## 🔄 Knowledge Base Update Process

### Manual (via UI)
1. Navigate to "📚 Knowledge Base" mode
2. Use "Add Content" tab to paste text, upload files, or add URLs
3. Query the knowledge base immediately after to verify

### Manual (via CLI)
```bash
# Add a text file
python scripts/update_knowledge.py --file /path/to/document.pdf

# Add a URL
python scripts/update_knowledge.py --url https://example.com/article

# Show statistics
python scripts/update_knowledge.py --stats
```

### Automatic (inbox watcher)
1. Place `.txt`, `.md`, or `.pdf` files in `data/knowledge_inbox/`
2. Click "▶️ Start Scheduler" in the Knowledge Base management panel
3. Files are automatically ingested every N minutes (configurable)

---

## ⚕️ Medical Safety Limitations

- This assistant provides **educational information only**
- It will **never** claim to diagnose conditions
- Emergency symptoms automatically trigger an emergency services alert
- Personal diagnosis requests are redirected to professional consultation
- Answers grounded only in MedQuAD — no invented medical facts

---

## 🖼️ Multimodal Functionality

- Upload PNG, JPG/JPEG, WEBP images via the UI
- Ask questions about the image in natural language
- Follow-up questions retain image context across turns
- The system decides automatically whether to use image analysis, retrieval, or text-only reasoning

---

## 🌍 Multilingual Functionality

- Supported: **English, Hindi, Marathi, Spanish**
- Language is automatically detected per message
- Mixed-language inputs (e.g., "मेरा password reset नहीं हो रहा") are handled
- Language can switch turn-by-turn within a conversation
- Conversation context is preserved across language switches

---

## 🔬 Research Assistant

- Searches arXiv CS papers (cs.AI, cs.LG, cs.CL, cs.IR, cs.CV)
- Returns paper metadata: title, authors, categories, publication date, PDF link
- Generates concept explanations citing retrieved papers
- Provides paper summarization with structured sections
- Visualizations: category distribution, keyword frequency, publication timeline, concept map

---

## 🧪 Testing

```bash
cd internship_project
python -m pytest tests/ -v
```

Run individual task tests:
```bash
python -m pytest tests/test_sentiment.py -v     # Task 1
python -m pytest tests/test_medical.py -v       # Task 2
python -m pytest tests/test_knowledge.py -v     # Task 3
python -m pytest tests/test_research.py -v      # Task 4
python -m pytest tests/test_multimodal.py -v    # Task 5
python -m pytest tests/test_multilingual.py -v  # Task 6
```

---

## 📝 Originality Statement

This project was developed as an original extension of the Nullclass training chatbot provided as the course foundation. The implementation:

- Preserves the original `customer_service_chatbot_LLM/` code untouched
- Modernizes deprecated APIs (PaLM → Gemini, LangChain 0.0.339 → 0.2+)
- Implements all six tasks as original code written specifically for this application
- Does not copy complete implementations from external repositories
- Uses official documentation and API references for all integrations

---

## 🔮 Future Improvements

1. **Fine-tuning**: Domain-specific fine-tuned embedding model for better retrieval
2. **Voice input**: Speech-to-text for multilingual voice queries
3. **Real-time web search**: Live internet retrieval for up-to-date information
4. **Multi-user sessions**: Database-backed conversation persistence
5. **Advanced medical NER**: Fine-tuned BioNER model for higher entity accuracy
6. **Offline LLM**: Local Ollama model as API-free fallback
