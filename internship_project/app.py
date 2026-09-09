"""
app.py — Main Streamlit application for the AI Customer Service Assistant.
Foundation: Nullclass customer-service chatbot (training project)
Extended with: Tasks 1–6 of the internship requirements.

Run: streamlit run app.py
"""
import os
import sys
import logging

# Add src to path so modules can be imported cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Customer Service Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(124, 58, 237, 0.3);
}

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.mode-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    backdrop-filter: blur(10px);
}

.sentiment-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 0.5rem;
}

.source-card {
    background: rgba(124, 58, 237, 0.1);
    border-left: 3px solid #7c3aed;
    padding: 0.5rem 0.8rem;
    border-radius: 0 8px 8px 0;
    margin: 0.3rem 0;
    font-size: 0.85rem;
}

.warning-card {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 8px;
    padding: 0.8rem;
    margin: 0.5rem 0;
}

.info-card {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 8px;
    padding: 0.8rem;
    margin: 0.5rem 0;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #5b21b6);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6, #7c3aed);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(124, 58, 237, 0.4);
    border-radius: 8px;
    color: #e0e0e0;
}

.lang-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    background: rgba(52, 211, 153, 0.2);
    border: 1px solid rgba(52, 211, 153, 0.5);
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #34d399;
}
</style>
""", unsafe_allow_html=True)

# ── Session state initialization ───────────────────────────────────────────────
def init_session_state():
    if "conversation" not in st.session_state:
        from src.core.conversation import ConversationState
        st.session_state.conversation = ConversationState()

    if "mode" not in st.session_state:
        st.session_state.mode = "customer_support"

    if "ml_context" not in st.session_state:
        from src.multilingual.context import MultilingualContext
        st.session_state.ml_context = MultilingualContext()

    if "kb_scheduler" not in st.session_state:
        st.session_state.kb_scheduler = None

    if "last_sentiment" not in st.session_state:
        st.session_state.last_sentiment = "neutral"
        st.session_state.last_sentiment_score = 0.5

    if "last_lang" not in st.session_state:
        st.session_state.last_lang = "en"

    if "uploaded_image_bytes" not in st.session_state:
        st.session_state.uploaded_image_bytes = None
        st.session_state.uploaded_image_name = None
        st.session_state.image_context = None

    if "research_papers" not in st.session_state:
        st.session_state.research_papers = []


init_session_state()

# ── Config validation ──────────────────────────────────────────────────────────
from src.core.config import validate_config
config_issues = validate_config()
if config_issues:
    st.error(
        f"⚠️ **Configuration Issues Detected:**\n"
        + "\n".join(f"- **{k}**: {v}" for k, v in config_issues.items())
        + "\n\nPlease create a `.env` file with the required API keys. "
        "See `.env.example` for reference."
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-title">🤖 AI Assistant</p>', unsafe_allow_html=True)
    st.caption("Nullclass Internship Project — Extended Customer Service AI")
    st.divider()

    # Mode selection
    st.subheader("📋 Select Mode")
    mode = st.radio(
        "Active Mode",
        options=[
            "customer_support",
            "medical_qa",
            "research_assistant",
            "multimodal",
            "multilingual",
            "knowledge_base",
        ],
        format_func=lambda x: {
            "customer_support": "🏢 Customer Support",
            "medical_qa": "🏥 Medical Q&A",
            "research_assistant": "🔬 Research Assistant",
            "multimodal": "🖼️ Multimodal Assistant",
            "multilingual": "🌍 Multilingual Chat",
            "knowledge_base": "📚 Knowledge Base",
        }[x],
        label_visibility="collapsed",
    )
    st.session_state.mode = mode
    st.session_state.conversation.mode = mode

    st.divider()

    # Sentiment indicator
    st.subheader("😊 Sentiment Detector")
    sentiment_colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
    sentiment_emojis = {"positive": "😊", "neutral": "😐", "negative": "😟"}
    sent = st.session_state.last_sentiment
    st.markdown(
        f'<div style="background:{sentiment_colors.get(sent,"#95a5a6")}22;'
        f'border:1px solid {sentiment_colors.get(sent,"#95a5a6")};'
        f'border-radius:8px;padding:0.5rem;text-align:center;font-weight:600;">'
        f'{sentiment_emojis.get(sent,"😐")} {sent.capitalize()} '
        f'({int(st.session_state.last_sentiment_score*100)}%)</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Language indicator
    st.subheader("🌍 Language")
    from src.multilingual.detector import get_language_name, get_language_flag
    lang_code = st.session_state.last_lang
    st.markdown(
        f'<span class="lang-badge">{get_language_flag(lang_code)} '
        f'{get_language_name(lang_code)}</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Index building controls
    st.subheader("⚙️ Index Controls")

    if st.button("🏗️ Build Customer Support Index", key="build_cs"):
        with st.spinner("Building customer support index..."):
            from src.core.chatbot import build_customer_support_index
            ok = build_customer_support_index()
            if ok:
                st.success("✅ Customer support index built!")
            else:
                st.error("❌ Failed. Check data/customer_support/dataset.csv")

    if mode == "medical_qa":
        if st.button("🏥 Build Medical Index", key="build_med"):
            with st.spinner("Downloading MedQuAD and building index (this may take a few minutes)..."):
                from src.medical.ingestion import build_medical_index
                ok = build_medical_index()
                if ok:
                    st.success("✅ Medical index built!")
                else:
                    st.error("❌ Failed to build medical index.")

    if mode == "research_assistant":
        if st.button("🔬 Build Research Index", key="build_res"):
            with st.spinner("Fetching arXiv papers and building index..."):
                from src.research.arxiv_loader import build_research_index
                ok = build_research_index()
                if ok:
                    st.success("✅ Research index built!")
                else:
                    st.error("❌ Failed to build research index.")

    st.divider()
    if st.button("🗑️ Clear Conversation", key="clear_chat"):
        st.session_state.conversation.clear()
        st.session_state.ml_context = __import__(
            "src.multilingual.context", fromlist=["MultilingualContext"]
        ).MultilingualContext()
        st.session_state.image_context = None
        st.session_state.uploaded_image_bytes = None
        st.session_state.research_papers = []
        st.rerun()

# ── Main content area ──────────────────────────────────────────────────────────
mode_titles = {
    "customer_support": "🏢 Customer Support Chat",
    "medical_qa": "🏥 Medical Q&A Assistant",
    "research_assistant": "🔬 Scientific Research Assistant",
    "multimodal": "🖼️ Multimodal AI Assistant",
    "multilingual": "🌍 Multilingual Chat",
    "knowledge_base": "📚 Knowledge Base Management",
}

st.markdown(f'<h1 class="main-title">{mode_titles.get(mode, "AI Assistant")}</h1>', unsafe_allow_html=True)

# ── Display chat history ───────────────────────────────────────────────────────
def display_chat_history():
    """Render existing conversation messages."""
    for msg in st.session_state.conversation.messages:
        with st.chat_message(msg.role):
            st.write(msg.content)
            if msg.role == "user" and msg.sentiment:
                sent_color = sentiment_colors.get(msg.sentiment, "#95a5a6")
                st.markdown(
                    f'<small style="color:{sent_color};">'
                    f'{sentiment_emojis.get(msg.sentiment,"")}'
                    f' {msg.sentiment} ({int((msg.sentiment_score or 0)*100)}%)</small>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# MODE: CUSTOMER SUPPORT (Foundation — original training project)
# ══════════════════════════════════════════════════════════════════════════════
if mode == "customer_support":
    st.caption("Powered by Nullclass FAQ dataset · RAG · Gemini 1.5 Flash · Sentiment-aware responses")

    display_chat_history()

    user_input = st.chat_input("Ask a question about Nullclass courses or services...")

    if user_input:
        # Task 1: Sentiment analysis
        from src.sentiment.analyzer import analyze_sentiment
        sentiment_label, sentiment_score = analyze_sentiment(user_input)
        st.session_state.last_sentiment = sentiment_label
        st.session_state.last_sentiment_score = sentiment_score

        # Add to conversation state
        st.session_state.conversation.add_user_message(
            user_input, sentiment=sentiment_label, sentiment_score=sentiment_score
        )

        with st.chat_message("user"):
            st.write(user_input)
            sent_color = sentiment_colors.get(sentiment_label, "#95a5a6")
            st.markdown(
                f'<small style="color:{sent_color};">'
                f'{sentiment_emojis.get(sentiment_label, "")} '
                f'{sentiment_label} ({int(sentiment_score * 100)}%)</small>',
                unsafe_allow_html=True,
            )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                from src.core.chatbot import answer_customer_query
                history = st.session_state.conversation.get_history_text(max_turns=4)
                result = answer_customer_query(
                    question=user_input,
                    sentiment=sentiment_label,
                    history=history,
                )
            answer = result.get("result", "I'm sorry, I couldn't find an answer.")
            st.write(answer)

            if result.get("sources"):
                with st.expander("📄 Sources", expanded=False):
                    for src in result["sources"]:
                        st.markdown(f'<div class="source-card">🔗 {src}</div>', unsafe_allow_html=True)

        st.session_state.conversation.add_assistant_message(answer)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MODE: MEDICAL Q&A (Task 2)
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "medical_qa":
    st.caption("Powered by MedQuAD dataset · Medical RAG · Safety layer active")
    st.markdown(
        '<div class="warning-card">⚠️ <strong>Medical Disclaimer:</strong> '
        'This assistant provides educational health information only. '
        'It is NOT a substitute for professional medical advice. '
        'Always consult a qualified healthcare professional.</div>',
        unsafe_allow_html=True,
    )

    display_chat_history()

    user_input = st.chat_input("Ask a medical question (symptoms, diseases, treatments)...")

    if user_input:
        from src.sentiment.analyzer import analyze_sentiment
        sentiment_label, sentiment_score = analyze_sentiment(user_input)
        st.session_state.last_sentiment = sentiment_label
        st.session_state.last_sentiment_score = sentiment_score

        st.session_state.conversation.add_user_message(
            user_input, sentiment=sentiment_label, sentiment_score=sentiment_score
        )

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching medical knowledge base..."):
                from src.medical.retrieval import answer_medical_query
                history = st.session_state.conversation.get_history_text(max_turns=4)
                result = answer_medical_query(user_input, history=history)

            if result.get("is_emergency"):
                st.error(result["result"])
            else:
                st.write(result["result"])

            # Medical entities display
            entities = result.get("entities", {})
            entity_display = []
            if entities.get("symptoms"):
                entity_display.append(f"🤒 **Symptoms detected**: {', '.join(entities['symptoms'])}")
            if entities.get("diseases"):
                entity_display.append(f"🏥 **Conditions**: {', '.join(entities['diseases'])}")
            if entities.get("treatments"):
                entity_display.append(f"💊 **Treatments**: {', '.join(entities['treatments'])}")
            if entities.get("medications"):
                entity_display.append(f"💉 **Medications**: {', '.join(entities['medications'])}")

            if entity_display:
                with st.expander("🔍 Detected Medical Entities"):
                    for e in entity_display:
                        st.markdown(e)

            # Sources
            if result.get("sources"):
                with st.expander("📚 Knowledge Base Sources"):
                    for src in result["sources"]:
                        st.markdown(f'<div class="source-card">{src}</div>', unsafe_allow_html=True)

            st.markdown(
                f'<div class="info-card" style="font-size:0.8rem;">'
                f'{result.get("disclaimer", "")}</div>',
                unsafe_allow_html=True,
            )

        st.session_state.conversation.add_assistant_message(result["result"])
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MODE: RESEARCH ASSISTANT (Task 4)
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "research_assistant":
    st.caption("Powered by arXiv CS papers · Semantic paper retrieval · Gemini explanations")

    col1, col2 = st.columns([2, 1])

    with col1:
        display_chat_history()
        user_input = st.chat_input("Ask about AI/ML concepts, search for papers, request explanations...")

        if user_input:
            from src.sentiment.analyzer import analyze_sentiment
            sentiment_label, sentiment_score = analyze_sentiment(user_input)
            st.session_state.last_sentiment = sentiment_label
            st.session_state.last_sentiment_score = sentiment_score

            st.session_state.conversation.add_user_message(user_input)

            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Searching research papers..."):
                    from src.research.retrieval import answer_research_query, search_papers
                    history = st.session_state.conversation.get_history_text(max_turns=4)
                    result = answer_research_query(user_input, history=history)
                    papers = result.get("papers", [])
                    st.session_state.research_papers = papers

                st.write(result["result"])

                if not result.get("evidence_available"):
                    st.warning("⚠️ Limited research evidence found. Consider building the research index first.")

                if papers:
                    with st.expander(f"📄 {len(papers)} Retrieved Papers"):
                        for p in papers:
                            st.markdown(f"**{p.get('title', 'Unknown')}**")
                            st.caption(f"Authors: {p.get('authors', '')[:80]}")
                            st.caption(f"Published: {p.get('published', '')[:10]} | {p.get('categories', '')}")
                            if p.get("pdf_url"):
                                st.markdown(f"[📄 View PDF]({p['pdf_url']})")
                            st.divider()

            st.session_state.conversation.add_assistant_message(result["result"])
            st.rerun()

    with col2:
        st.subheader("📊 Research Analytics")

        from src.research.arxiv_loader import get_cached_papers, get_paper_categories_distribution
        cached_papers = get_cached_papers()

        if cached_papers:
            st.metric("Papers in Index", len(cached_papers))
            tab_viz1, tab_viz2, tab_viz3 = st.tabs(["Categories", "Keywords", "Timeline"])

            from src.research.visualization import (
                plot_category_distribution,
                plot_keyword_frequency,
                plot_publication_timeline,
                plot_retrieved_concept_map,
            )

            with tab_viz1:
                fig = plot_category_distribution(cached_papers)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            with tab_viz2:
                fig = plot_keyword_frequency(cached_papers[:100])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            with tab_viz3:
                fig = plot_publication_timeline(cached_papers)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            if st.session_state.research_papers:
                st.subheader("🗺️ Retrieved Concept Map")
                fig = plot_retrieved_concept_map(st.session_state.research_papers)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            # Paper summarizer
            st.subheader("📝 Summarize a Paper")
            if st.session_state.research_papers:
                paper_titles = [p.get("title", "Unknown")[:60] for p in st.session_state.research_papers]
                selected_idx = st.selectbox("Select paper to summarize", range(len(paper_titles)),
                                            format_func=lambda i: paper_titles[i])
                if st.button("Generate Summary", key="summarize_paper"):
                    with st.spinner("Summarizing..."):
                        from src.research.summarizer import summarize_paper
                        summary = summarize_paper(st.session_state.research_papers[selected_idx])
                    st.markdown(summary)
        else:
            st.info("Build the research index first to see analytics.")


# ══════════════════════════════════════════════════════════════════════════════
# MODE: MULTIMODAL (Task 5)
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "multimodal":
    st.caption("Upload images + ask questions · Gemini Vision · Context-aware reasoning")

    # Image upload area
    uploaded_file = st.file_uploader(
        "📎 Upload an image (PNG, JPG, JPEG, WEBP)",
        type=["png", "jpg", "jpeg", "webp"],
        key="image_uploader",
    )

    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
        st.session_state.uploaded_image_bytes = img_bytes
        st.session_state.uploaded_image_name = uploaded_file.name

        col_img1, col_img2 = st.columns([1, 2])
        with col_img1:
            st.image(img_bytes, caption=f"📎 {uploaded_file.name}", use_column_width=True)
        with col_img2:
            if st.session_state.image_context:
                st.markdown("**📋 Previous Analysis Available**")
                st.caption(st.session_state.image_context[:200] + "...")
            else:
                st.info("💡 Ask a question about this image, or submit without text to get a general analysis.")

    elif st.session_state.uploaded_image_bytes:
        st.info("📎 Using previously uploaded image (clear conversation to remove it)")
        st.image(st.session_state.uploaded_image_bytes, width=200)

    st.divider()
    display_chat_history()

    user_input = st.chat_input("Ask about the image, or type a question...")

    if user_input or (uploaded_file is not None and not user_input):
        user_text = user_input or "Please analyze and describe this image."

        from src.sentiment.analyzer import analyze_sentiment
        sentiment_label, sentiment_score = analyze_sentiment(user_text)
        st.session_state.last_sentiment = sentiment_label
        st.session_state.last_sentiment_score = sentiment_score

        st.session_state.conversation.add_user_message(user_text)

        with st.chat_message("user"):
            st.write(user_text)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                from src.multimodal.reasoning import process_multimodal_request
                result = process_multimodal_request(
                    user_text=user_text,
                    image_bytes=st.session_state.uploaded_image_bytes if uploaded_file else None,
                    image_filename=st.session_state.uploaded_image_name or "image.jpg",
                    image_context=st.session_state.image_context,
                    conversation_history=st.session_state.conversation.get_history_text(max_turns=4),
                )

            st.write(result["response"])

            # Update image context for follow-ups
            if result.get("image_description"):
                st.session_state.image_context = result["image_description"]

            # Show processing path for transparency
            path = result.get("processing_path", "")
            path_labels = {
                "image_analysis": "🖼️ Image analyzed",
                "image_followup": "🔄 Follow-up on image",
                "text_only": "💬 Text reasoning",
                "clarification": "❓ Clarification",
            }
            if path in path_labels:
                st.caption(f"Processing: {path_labels[path]}")

            if not result.get("valid", True):
                notes = result.get("validation_notes", "")
                if notes:
                    st.caption(f"⚠️ Validation note: {notes}")

        # Clear the uploaded file reference after processing
        st.session_state.uploaded_image_bytes = None if not st.session_state.image_context else st.session_state.uploaded_image_bytes
        st.session_state.conversation.add_assistant_message(result["response"])
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MODE: MULTILINGUAL (Task 6)
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "multilingual":
    st.caption("Supports English · Hindi · Marathi · Spanish · Mixed-language inputs")

    lang_cols = st.columns(4)
    lang_examples = {
        "🇬🇧 English": "How do I reset my password?",
        "🇮🇳 Hindi": "मेरा पासवर्ड कैसे रीसेट करूं?",
        "🇮🇳 Marathi": "माझा पासवर्ड कसा रीसेट करायचा?",
        "🇪🇸 Spanish": "¿Cómo restablezco mi contraseña?",
    }
    for (lang_name, example), col in zip(lang_examples.items(), lang_cols):
        with col:
            st.markdown(f"**{lang_name}**")
            st.caption(example)

    st.divider()

    # Language switching stats
    ml_stats = st.session_state.ml_context.get_stats()
    if ml_stats["total_turns"] > 0:
        st.markdown(
            f'<div class="info-card">'
            f'🌐 Languages used: {", ".join(ml_stats["languages_used"])} · '
            f'Turns: {ml_stats["total_turns"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    display_chat_history()

    user_input = st.chat_input("Type in any supported language or mix languages freely...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                from src.multilingual.context import process_multilingual_query
                from src.sentiment.analyzer import analyze_sentiment

                # Sentiment on original text
                sentiment_label, sentiment_score = analyze_sentiment(user_input)
                st.session_state.last_sentiment = sentiment_label
                st.session_state.last_sentiment_score = sentiment_score

                def _chat_handler(english_query: str, history: str) -> str:
                    from src.core.chatbot import answer_customer_query
                    result = answer_customer_query(
                        question=english_query,
                        sentiment=sentiment_label,
                        history=history,
                    )
                    return result.get("result", "I don't know.")

                ml_result = process_multilingual_query(
                    user_text=user_input,
                    ml_context=st.session_state.ml_context,
                    chat_handler=_chat_handler,
                )

            st.session_state.last_lang = ml_result["detected_lang"]
            st.write(ml_result["response"])

            # Language info display
            flag = get_language_flag(ml_result["detected_lang"])
            lang_name = ml_result["lang_name"]
            st.markdown(
                f'<small>{flag} Detected: <strong>{lang_name}</strong> '
                f'(confidence: {int(ml_result.get("confidence", 0.8)*100)}%)</small>',
                unsafe_allow_html=True,
            )

            if ml_result.get("language_switch_note"):
                st.markdown(f'<small>{ml_result["language_switch_note"]}</small>', unsafe_allow_html=True)

            if ml_result.get("was_translated"):
                st.caption(f"🔄 English query: \"{ml_result['english_query'][:80]}\"")

        st.session_state.conversation.add_user_message(
            user_input,
            sentiment=sentiment_label,
            sentiment_score=sentiment_score,
            language=ml_result["detected_lang"],
        )
        st.session_state.conversation.add_assistant_message(ml_result["response"])
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MODE: KNOWLEDGE BASE MANAGEMENT (Task 3)
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "knowledge_base":
    st.caption("Dynamically expand the knowledge base · Text, PDF, or URL sources")

    from src.knowledge.updater import get_knowledge_base_stats, update_from_text, update_from_url, update_from_uploaded_bytes, answer_from_knowledge_base
    from src.knowledge.scheduler import KnowledgeScheduler

    # Stats panel
    stats = get_knowledge_base_stats()
    metric_cols = st.columns(3)
    metric_cols[0].metric("📄 Total Sources", stats["total_sources"])
    metric_cols[1].metric("🧩 Total Chunks", stats["total_chunks"])
    metric_cols[2].metric("✅ Index Active", "Yes" if stats["index_exists"] else "No")

    st.divider()

    tab_add, tab_query, tab_scheduler, tab_sources = st.tabs([
        "➕ Add Content",
        "🔍 Query Knowledge Base",
        "⏱️ Scheduler",
        "📋 Source Registry",
    ])

    with tab_add:
        st.subheader("Add New Content to Knowledge Base")
        add_type = st.radio("Source type", ["📝 Text", "📎 File Upload", "🌐 URL"])

        if add_type == "📝 Text":
            doc_name = st.text_input("Document name", value="custom_doc")
            doc_text = st.text_area(
                "Paste content here",
                height=200,
                placeholder="Paste any text content to add to the knowledge base...",
            )
            if st.button("➕ Add Text", key="add_text"):
                if doc_text.strip():
                    with st.spinner("Processing and indexing..."):
                        result = update_from_text(doc_text, name=doc_name)
                    if result["success"]:
                        st.success(result["message"])
                    else:
                        st.error(result["message"])
                else:
                    st.warning("Please enter some text content.")

        elif add_type == "📎 File Upload":
            uploaded_doc = st.file_uploader(
                "Upload a document",
                type=["txt", "md", "pdf"],
                key="kb_file_uploader",
            )
            if uploaded_doc and st.button("➕ Add File", key="add_file"):
                with st.spinner("Processing file..."):
                    result = update_from_uploaded_bytes(
                        uploaded_doc.read(), uploaded_doc.name
                    )
                if result["success"]:
                    st.success(result["message"])
                else:
                    st.error(result["message"])

        elif add_type == "🌐 URL":
            url_input = st.text_input("Enter URL", placeholder="https://example.com/article")
            url_name = st.text_input("Source name (optional)", value="")
            if st.button("➕ Fetch & Add URL", key="add_url"):
                if url_input.strip():
                    with st.spinner("Fetching and indexing URL content..."):
                        result = update_from_url(url_input, name=url_name or None)
                    if result["success"]:
                        st.success(result["message"])
                    else:
                        st.error(result["message"])
                else:
                    st.warning("Please enter a URL.")

    with tab_query:
        st.subheader("Query the Knowledge Base")
        st.caption("Test that newly added content is retrievable:")

        kb_question = st.text_input("Ask a question", key="kb_query_input")
        if kb_question and st.button("🔍 Search", key="kb_search"):
            with st.spinner("Searching..."):
                result = answer_from_knowledge_base(kb_question)
            st.write("**Answer:**", result["result"])
            if result.get("sources"):
                st.caption("Sources: " + ", ".join(result["sources"]))

    with tab_scheduler:
        st.subheader("Periodic Update Scheduler")
        st.markdown("""
        The scheduler watches a directory for new files and automatically ingests them.

        **How it works:**
        1. Place `.txt`, `.md`, or `.pdf` files in the `data/knowledge_inbox/` folder
        2. The scheduler checks every N minutes
        3. New files are automatically ingested into the knowledge base
        """)

        interval = st.slider("Check interval (minutes)", min_value=1, max_value=60, value=15)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            if st.button("▶️ Start Scheduler", key="start_scheduler"):
                if st.session_state.kb_scheduler is None:
                    watch_dir = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "data", "knowledge_inbox"
                    )
                    scheduler = KnowledgeScheduler(watch_dir=watch_dir, interval_minutes=interval)
                    scheduler.start()
                    st.session_state.kb_scheduler = scheduler
                    st.success(f"✅ Scheduler started. Watching `data/knowledge_inbox/` every {interval} min.")
                else:
                    st.info("Scheduler is already running.")

        with col_s2:
            if st.button("⏹️ Stop Scheduler", key="stop_scheduler"):
                if st.session_state.kb_scheduler:
                    st.session_state.kb_scheduler.stop()
                    st.session_state.kb_scheduler = None
                    st.success("Scheduler stopped.")

        with col_s3:
            if st.button("🔄 Scan Now", key="scan_now"):
                if st.session_state.kb_scheduler:
                    logs = st.session_state.kb_scheduler.run_now()
                    if logs:
                        for entry in logs[:5]:
                            st.write(f"✅ {entry['file']}: {entry['chunks_added']} chunks added")
                    else:
                        st.info("No new files found in watch directory.")
                else:
                    st.warning("Start the scheduler first.")

        if st.session_state.kb_scheduler:
            sched_status = st.session_state.kb_scheduler.get_status()
            st.info(
                f"⏱️ **Scheduler running** · "
                f"Watching: `{sched_status['watch_dir']}` · "
                f"Interval: {sched_status['interval_minutes']} min · "
                f"Files processed: {sched_status['processed_files']}"
            )

    with tab_sources:
        st.subheader("Ingested Source Registry")
        if stats["sources"]:
            for src in stats["sources"]:
                with st.expander(f"📄 {src.get('name', 'Unknown')} ({src.get('type', '')})"):
                    st.write(f"**Ingested:** {src.get('ingested_at', 'Unknown')}")
                    st.write(f"**Chunks:** {src.get('chunks', 0)}")
                    st.write(f"**Type:** {src.get('type', 'Unknown')}")
                    if src.get("url"):
                        st.write(f"**Source:** {src['url']}")
        else:
            st.info("No documents in the knowledge base yet. Add content using the 'Add Content' tab.")
