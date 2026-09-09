# Internship Report
## Data Science / Generative AI Virtual Internship
### Nullclass — AI Customer Service Assistant Project

---

## 1. Introduction

This report documents the work undertaken during the Nullclass Data Science / Generative AI virtual internship. The internship required extending an existing training customer-service chatbot with six advanced AI capabilities, creating a production-quality, integrated AI assistant.

The final application is a single Streamlit-based platform that preserves the original Nullclass FAQ chatbot as its foundation while adding sentiment analysis, medical Q&A, dynamic knowledge management, scientific research assistance, multimodal image reasoning, and multilingual conversation capabilities.

---

## 2. Background

**Organization**: Nullclass  
**Role**: AI/ML Intern — Generative AI Track  
**Duration**: [Insert internship duration]  
**Project Title**: Extended AI Customer Service Assistant

The internship is grounded in the course curriculum, which covers LLMs, LangChain, FAISS vector stores, Streamlit development, and the Google Gemini API. The training project is a minimal FAQ chatbot using Google PaLM and LangChain.

The internship extends this training project rather than creating an unrelated standalone application, demonstrating the ability to progressively enhance an existing codebase with advanced capabilities.

---

## 3. Learning Objectives

1. Modernize deprecated LLM APIs (PaLM → Gemini) while preserving application logic
2. Implement sentiment analysis using ensemble NLP methods
3. Build domain-specific RAG pipelines using external datasets (MedQuAD, arXiv)
4. Design dynamic knowledge base systems with deduplication and metadata tracking
5. Implement multimodal AI reasoning with image + text orchestration
6. Build multilingual conversation handling with cross-language context preservation
7. Write modular, testable Python code following software engineering best practices
8. Create production-ready documentation and test suites

---

## 4. Activities and Tasks

### Week 1 — Foundation Analysis and Modernization
- Analyzed the existing `customer_service_chatbot_LLM` training project
- Identified deprecated dependencies: `langchain==0.0.339`, `GooglePalm`, `HuggingFaceInstructEmbeddings`
- Designed the extended project architecture (8-module structure)
- Modernized the core chatbot to use `gemini-1.5-flash` and `all-MiniLM-L6-v2`
- Set up the project directory structure within the same repository

### Week 2 — Sentiment Analysis (Task 1) and Medical Q&A (Task 2)
- Implemented dual-model sentiment analysis (VADER + TextBlob ensemble)
- Integrated sentiment into prompt generation to adapt LLM tone
- Designed and implemented the MedQuAD download pipeline (GitHub XML parsing)
- Built the medical RAG chain with safety disclaimers and emergency detection
- Implemented medical entity recognition using keyword pattern matching + spaCy

### Week 3 — Dynamic Knowledge Base (Task 3) and Research Assistant (Task 4)
- Built the multi-source ingestion pipeline (text, PDF, URL)
- Implemented SHA-256 deduplication to prevent re-ingestion
- Created the background scheduler using Python threading.Timer
- Fetched arXiv CS papers using the official arxiv Python API
- Built the research RAG chain with paper citation instructions
- Implemented research visualizations (4 chart types using Plotly)
- Added paper summarization with structured section generation

### Week 4 — Multimodal (Task 5) and Multilingual (Task 6)
- Implemented Gemini Vision image analysis with structured description generation
- Built the multimodal orchestration layer with intent classification
- Implemented response validation to prevent unsupported claims
- Integrated langdetect for automatic language detection
- Implemented deep-translator for translate-to-English → process → translate-back pipeline
- Built the MultilingualContext class for cross-turn language state management
- Added Devanagari script disambiguation for Hindi vs Marathi detection

### Week 5 — Integration Testing and Documentation
- Wrote comprehensive test suites for all 6 tasks using pytest
- Created the unified Streamlit UI with dark theme and 6-mode navigation
- Wrote README, REQUIREMENTS_CHECKLIST, INTERNSHIP_REPORT, DAILY_PROGRESS_LOG
- Fixed integration issues and verified end-to-end functionality

---

## 5. Skills and Competencies

### Technical Skills Developed

| Skill | Application |
|---|---|
| LangChain (modern) | RAG chains, retrievers, prompt templates across 3 domains |
| FAISS vector stores | Create, load, append without rebuild (multi-collection) |
| Google Gemini API | Text generation, vision analysis, multimodal reasoning |
| Sentence Transformers | Embedding generation for semantic retrieval |
| NLP (VADER, TextBlob, spaCy) | Sentiment analysis, entity extraction |
| Data pipeline design | MedQuAD XML parsing, arXiv API fetching, deduplication |
| Streamlit development | Multi-mode UI, session state management, dynamic components |
| Multilingual NLP | Language detection, translation, cross-lingual context |
| Python threading | Background scheduler for periodic knowledge updates |
| pytest | Unit testing across 6 functional domains |
| API integration | Google Gemini, arXiv, GitHub (for MedQuAD) |

### Soft Skills Developed
- Technical documentation and requirement traceability
- Modular software architecture design
- Incremental enhancement of existing codebases
- Balancing feature completeness with practical hardware constraints

---

## 6. Feedback and Evidence

> [PLACEHOLDER: Insert mentor feedback here after internship review sessions]

> [PLACEHOLDER: Insert screenshots of working application — one per mode]

> [PLACEHOLDER: Insert test run output from `python -m pytest tests/ -v`]

> [PLACEHOLDER: Insert any communication records or review feedback from supervisors]

---

## 7. Challenges and Solutions

### Challenge 1: Deprecated Training Project APIs
**Problem**: The training project used `langchain==0.0.339` with `GooglePalm` (fully deprecated) and `HuggingFaceInstructEmbeddings` (no longer maintained).  
**Solution**: Upgraded to `langchain>=0.2.0`, replaced `GooglePalm` with `ChatGoogleGenerativeAI` (Gemini 1.5 Flash), and `HuggingFaceInstructEmbeddings` with `HuggingFaceEmbeddings` using `all-MiniLM-L6-v2`. Preserved all original logic while updating the API surface.

### Challenge 2: MedQuAD Dataset Access
**Problem**: MedQuAD is distributed as XML files on GitHub. Downloading and parsing the entire dataset at startup would be slow and unreliable.  
**Solution**: Implemented a selective download strategy: fetch 25 XML files per clinical folder, parse to JSON, and cache locally. Subsequent runs use the cache. Added graceful fallback for network failures.

### Challenge 3: Hindi vs Marathi Disambiguation
**Problem**: Both Hindi and Marathi use Devanagari script. Generic language detectors often confuse them.  
**Solution**: Implemented a custom disambiguation function that checks for language-specific marker words (e.g., "आहे", "आणि" for Marathi vs "है", "और" for Hindi) before falling back to langdetect.

### Challenge 4: Preserving Existing Knowledge on KB Updates
**Problem**: Naive re-ingestion would rebuild the FAISS index and lose previously added documents.  
**Solution**: Implemented `append_to_index()` using FAISS's `merge_from()` method, which merges new vectors into the existing index without destroying prior entries. SHA-256 hashing prevents duplicate processing.

### Challenge 5: Multimodal Context Across Turns
**Problem**: Follow-up questions about an image ("explain the important part") need the image context from the previous turn.  
**Solution**: Stored the Gemini Vision image description in Streamlit session state. The `_detect_intent()` function checks for `has_image_context` and routes to `_handle_image_followup()` when appropriate.

---

## 8. Outcomes and Impact

### Deliverables Completed
1. ✅ Unified Streamlit application with 6 operational modes
2. ✅ Customer support chatbot (modernized foundation) with sentiment-aware responses
3. ✅ Medical Q&A system with MedQuAD retrieval and safety layer
4. ✅ Dynamic knowledge base with deduplication and scheduling
5. ✅ Research assistant with arXiv retrieval and Plotly visualizations
6. ✅ Multimodal assistant with Gemini Vision and context tracking
7. ✅ Multilingual support for English/Hindi/Marathi/Spanish
8. ✅ Comprehensive test suite (6 test files, 40+ test cases)
9. ✅ Complete documentation (README, REQUIREMENTS_CHECKLIST, this report)

### Performance Observations
> [PLACEHOLDER: Insert measured response times, test pass rates, and any quantitative metrics after actual runs]

### Skills Impact
The internship provided hands-on experience building a full-stack AI application from an existing foundation — a realistic industrial scenario where engineers extend inherited codebases rather than starting from scratch.

---

## 9. Conclusion

The internship successfully delivered a comprehensive AI customer service assistant that extends the Nullclass training chatbot with six advanced capabilities. The project demonstrates:

- The ability to work within constraints of an existing codebase
- Practical application of RAG, embeddings, and vector search across multiple domains
- Understanding of multimodal, multilingual, and sentiment-aware AI systems
- Software engineering discipline: modular architecture, testing, documentation

The final application is suitable as a portfolio piece demonstrating practical Generative AI engineering skills at an internship level.

> [PLACEHOLDER: Insert supervisor sign-off or completion confirmation]
