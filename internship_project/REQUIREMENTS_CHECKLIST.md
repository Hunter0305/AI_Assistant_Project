# Requirements Traceability Checklist

> Maps every internship requirement to its exact implementation.
> Items marked `[x]` are implemented and tested.
> Items marked `[ ]` are pending or partially implemented.

---

## Task 1 — Sentiment Analysis

- [x] **Automatic sentiment detection** — `src/sentiment/analyzer.py::analyze_sentiment()` runs on every user message
- [x] **Detects positive / neutral / negative** — VADER + TextBlob ensemble returns one of three labels
- [x] **Confidence score** — `analyze_sentiment()` returns `(label, float)` where float is 0–1 confidence
- [x] **Stored in conversation state** — `ConversationState.add_user_message()` stores `sentiment` and `sentiment_score` on each `Message`
- [x] **Adapts chatbot response** — `answer_customer_query()` in `chatbot.py` selects one of three prompt prefixes based on sentiment label
- [x] **Positive tone** — warm/enthusiastic prefix: "The customer seems happy..."
- [x] **Negative tone** — empathetic prefix: "Be especially empathetic, acknowledge their frustration..."
- [x] **Neutral tone** — professional prefix: "Provide a direct, professional response"
- [x] **UI sentiment indicator** — colored badge in sidebar showing current sentiment with emoji + percentage
- [x] **Model documented** — VADER (vaderSentiment) + TextBlob, selected for CPU-only operation, no GPU needed
- [x] **Test: clearly positive** — `tests/test_sentiment.py::test_clearly_positive_message`
- [x] **Test: clearly negative** — `tests/test_sentiment.py::test_clearly_negative_message`
- [x] **Test: neutral question** — `tests/test_sentiment.py::test_neutral_question`
- [x] **Test: ambiguous/mixed** — `tests/test_sentiment.py::test_ambiguous_mixed_sentiment`

---

## Task 2 — Medical Q&A Chatbot

- [x] **MedQuAD dataset** — `src/medical/ingestion.py` downloads from `github.com/abachaa/MedQuAD`
- [x] **Reproducible download** — `download_medquad()` fetches XML files from public GitHub API, caches locally
- [x] **Not hard-coded** — downloads real MedQuAD XML files (800+ QA pairs from 7 clinical folders)
- [x] **Preprocessing** — `_parse_xml_to_qa()` parses XML, cleans whitespace, filters short answers
- [x] **Converts to documents** — `qa_pairs_to_documents()` creates LangChain Documents
- [x] **Generates embeddings** — shared `get_embeddings()` (all-MiniLM-L6-v2) via `vector_store.create_index()`
- [x] **Vector database** — FAISS at `faiss_indexes/medical/`
- [x] **Semantic retrieval** — `get_retriever(MEDICAL_INDEX, k=4)` does cosine similarity search
- [x] **Generates answers with retrieved context** — `answer_medical_query()` uses RetrievalQA chain
- [x] **Medical entity recognition** — `src/medical/entities.py` detects symptoms, diseases, treatments, medications using keyword patterns + optional spaCy
- [x] **Displays retrieved sources** — source Q&A pairs shown in expandable panel
- [x] **Medical Q&A mode in Streamlit** — dedicated mode in sidebar navigation
- [x] **Safety layer: not a doctor** — prompt explicitly states "You are NOT a doctor"
- [x] **Emergency detection** — `_is_emergency_query()` flags chest pain, stroke, overdose etc.
- [x] **Diagnosis refusal** — `_is_dangerous_query()` catches "do I have X?" requests
- [x] **Recommends professionals** — every answer ends with healthcare professional recommendation
- [x] **Insufficient info handling** — checks for "don't have sufficient information" phrases, sets `insufficient_info` flag
- [x] **Disclaimer always shown** — `MEDICAL_DISCLAIMER` displayed on every response
- [x] **Test: symptom question** — `tests/test_medical.py::test_symptom_detection`
- [x] **Test: disease question** — `tests/test_medical.py::test_disease_detection`
- [x] **Test: treatment question** — `tests/test_medical.py::test_treatment_detection`
- [x] **Test: emergency detection** — `tests/test_medical.py::test_emergency_detection_chest_pain`
- [x] **Test: diagnosis refusal** — `tests/test_medical.py::test_dangerous_diagnosis_request`

---

## Task 3 — Dynamic Knowledge Base Expansion

- [x] **Reusable ingestion pipeline** — `src/knowledge/ingestion.py` with `ingest_text()`, `ingest_file()`, `ingest_url()`
- [x] **Fetch/read content** — supports text, PDF (`pypdf`), and URL (urllib)
- [x] **Clean content** — `_clean_text()` normalizes whitespace
- [x] **Split into chunks** — `RecursiveCharacterTextSplitter` (800 chars, 100 overlap)
- [x] **Generate embeddings** — shared `get_embeddings()` singleton
- [x] **Update vector database** — `append_to_index()` in `vector_store.py` merges new FAISS with existing
- [x] **Information available to retrieval** — `answer_from_knowledge_base()` queries updated index
- [x] **Local documents** — `ingest_file()` supports .txt, .md
- [x] **PDF support** — `ingest_file()` supports .pdf via pypdf
- [x] **URL support** — `ingest_url()` fetches and strips HTML
- [x] **Metadata** — source name, type, ingestion timestamp, URL stored per document
- [x] **Duplicate prevention** — SHA-256 content hash comparison via `_is_duplicate()`
- [x] **Periodic update mechanism** — `src/knowledge/scheduler.py` uses threading.Timer, watches directory
- [x] **Manual update button** — "Scan Now" button in Knowledge Base panel
- [x] **Scheduled auto-update** — "Start Scheduler" button triggers background file watcher
- [x] **Retrieval of new content** — `answer_from_knowledge_base()` queries KNOWLEDGE_INDEX
- [x] **Preserves existing knowledge** — `append_to_index()` merges (not replaces) FAISS stores
- [x] **Source registry** — `KNOWLEDGE_METADATA_FILE` JSON tracks all ingested documents
- [x] **Test: text ingestion** — `tests/test_knowledge.py::test_ingest_text_returns_documents`
- [x] **Test: duplicate prevention** — `tests/test_knowledge.py::test_duplicate_ingestion_skipped`
- [x] **Test: metadata** — `tests/test_knowledge.py::test_ingest_text_document_has_metadata`

---

## Task 4 — Scientific Research Chatbot

- [x] **arXiv dataset (reproducible)** — `src/research/arxiv_loader.py` uses `arxiv` Python package (official API)
- [x] **CS domain subset** — fetches cs.AI, cs.LG, cs.CL, cs.IR, cs.CV categories
- [x] **Not loading entire dataset** — fetches max 500 papers, configurable via `ARXIV_MAX_PAPERS`
- [x] **Paper metadata extracted** — title, authors, abstract, categories, published date, arxiv_id, pdf_url
- [x] **Embeddings created** — `papers_to_documents()` → FAISS index
- [x] **Papers in vector database** — `faiss_indexes/research/`
- [x] **Semantic paper retrieval** — `search_papers()` with min_score threshold
- [x] **Paper search in Streamlit** — Research Assistant mode with chat input
- [x] **Returns papers with metadata** — expandable panel shows title, authors, date, PDF link
- [x] **Paper summarization** — `src/research/summarizer.py::summarize_paper()` generates 6-section summaries
- [x] **Information extraction** — `extract_key_concepts()` extracts technical terms from abstracts
- [x] **LLM for explanation** — Gemini 1.5 Flash (LLM, not just API — used for generation)
- [x] **Explains complex concepts** — `answer_research_query()` uses retrieved papers to explain topics
- [x] **Cites retrieved papers** — prompt instructs LLM to cite paper titles in response
- [x] **Follow-up question support** — `get_history_text()` passed as history to research prompt
- [x] **Concept visualization** — `src/research/visualization.py` with 4 chart types
- [x] **Category distribution chart** — real data from cached papers
- [x] **Keyword frequency chart** — extracted from paper titles/abstracts
- [x] **Publication timeline** — papers by year line chart
- [x] **Retrieved concept map** — circular scatter of retrieved papers
- [x] **Distinguishes retrieved vs generated** — prompt explicitly says "According to retrieved papers:" vs "General knowledge:"
- [x] **Insufficient evidence safeguard** — checks evidence_available flag, warns user if limited
- [x] **Test: paper-to-document conversion** — `tests/test_research.py::test_papers_to_documents_count`
- [x] **Test: metadata presence** — `tests/test_research.py::test_documents_have_metadata`
- [x] **Test: visualization with real data** — `tests/test_research.py::test_charts_use_actual_data`

---

## Task 5 — Multimodal AI Assistant

- [x] **Image upload via Streamlit** — `st.file_uploader()` in Multimodal mode
- [x] **PNG support** — `get_mime_type()` returns `image/png`
- [x] **JPG/JPEG support** — `get_mime_type()` returns `image/jpeg`
- [x] **WEBP support** — `get_mime_type()` returns `image/webp`
- [x] **Analyzes visual content** — `analyze_image()` uses Gemini 1.5 Flash vision
- [x] **Extracts relevant information** — 5-point structured image description
- [x] **Combines image + text** — `process_multimodal_request()` passes both to Gemini
- [x] **Contextual reasoning** — `_detect_intent()` classifies intent before processing
- [x] **Multi-turn image context** — `image_context` stored in session state for follow-ups
- [x] **Follow-up across turns** — `_handle_image_followup()` uses stored image description
- [x] **Orchestration layer** — `reasoning.py` decides path: image_analysis / image_followup / text_only / clarification
- [x] **Ambiguity handling** — empty text without image triggers `needs_clarification` response
- [x] **Response validation** — `src/multimodal/validator.py::validate_response()` checks grounding
- [x] **Validation: empty check** — fails empty responses
- [x] **Validation: hallucination check** — flags "I imagine", "I assume" indicators
- [x] **Validation: non-answer check** — flags "as an AI, I cannot view"
- [x] **Intelligent decision-making** — `_detect_intent()` function selects processing path
- [x] **Modular image processing** — `image_processor.py` is standalone module
- [x] **Image preview in UI** — `st.image()` displays uploaded image
- [x] **Processing path shown** — UI shows which processing path was used (transparency)
- [x] **Test: intent detection (image)** — `tests/test_multimodal.py::test_image_upload_triggers_image_analysis`
- [x] **Test: intent detection (followup)** — `tests/test_multimodal.py::test_followup_without_image_uses_context`
- [x] **Test: response validation** — `tests/test_multimodal.py::test_valid_response_passes`
- [x] **Test: MIME type detection** — `tests/test_multimodal.py::TestMimeTypeDetection`

---

## Task 6 — Multilingual Chatbot

- [x] **Automatic language detection** — `detect_language()` uses langdetect (open-source)
- [x] **English support** — base language, processed natively
- [x] **Hindi support** — detected via Devanagari script + langdetect
- [x] **Marathi support** — disambiguated from Hindi via Marathi-specific word markers
- [x] **Spanish support** — langdetect identifies correctly
- [x] **Conversation context preserved** — `MultilingualContext.get_english_history()` maintains English-normalized history
- [x] **Language switching in same conversation** — `language_switched()` detects and reports changes
- [x] **Intent preservation** — English-translated query processed by same RAG pipeline
- [x] **Entity preservation** — `known_entities` list maintained across turns
- [x] **Mixed-language input handling** — `normalize_mixed_language_query()` normalizes to English
- [x] **Ambiguous query handling** — pronoun references resolved via `resolve_pronoun_reference()`
- [x] **Responses in appropriate language** — `translate_from_english()` translates response back
- [x] **Consistent answers across languages** — same English RAG pipeline, different response language
- [x] **Open-source models** — langdetect (Apache 2.0), deep-translator (MIT)
- [x] **Conversation-level language layer** — `MultilingualContext` class persists across turns (not per-message translation)
- [x] **Language indicator in UI** — flag emoji + language name in sidebar
- [x] **Language switch notification** — "🔄 Language switched: English → Hindi" shown in chat
- [x] **Test: English detection** — `tests/test_multilingual.py::test_english_detected`
- [x] **Test: Hindi detection** — `tests/test_multilingual.py::test_hindi_devanagari_detected`
- [x] **Test: Spanish detection** — `tests/test_multilingual.py::test_spanish_detected`
- [x] **Test: mixed language** — `tests/test_multilingual.py::test_mixed_language_returns_valid_code`
- [x] **Test: language switching** — `tests/test_multilingual.py::test_language_switch_detection`
- [x] **Test: context preservation** — `tests/test_multilingual.py::test_english_history_preserved_across_languages`

---

## Foundation — Training Chatbot

- [x] **Original training chatbot preserved** — `customer_service_chatbot_LLM/` untouched
- [x] **Modernized foundation works** — `src/core/chatbot.py` is functional equivalent
- [x] **RAG pipeline works** — FAISS + retriever + LLM chain
- [x] **Streamlit works** — `app.py` runs with `streamlit run app.py`
- [x] **Dataset preserved** — `data/customer_support/dataset.csv` same content as original
- [x] **No API keys hard-coded** — all keys from `.env`
- [x] **`.env` ignored** — in `.gitignore`
- [x] **`requirements.txt` complete** — all dependencies listed
- [x] **README exists** — `README.md`
- [x] **Internship report exists** — `INTERNSHIP_REPORT.md`
- [x] **Requirements checklist exists** — this file
- [x] **Test cases exist** — `tests/` directory with 6 test files
- [x] **No fake claims** — all metrics and placeholders clearly marked
- [x] **No copied external code** — original implementations throughout
