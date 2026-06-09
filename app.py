
import streamlit as st
from utils.pdf_loader import load_pdf
from utils.chunker import create_chunks
from utils.embeddings import get_embedding
from utils.search import search_chunks
from utils.gemini_helper import ask_gemini

# ------------------------
# Page Config
# ------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)
st.markdown("""
<style>

.stTextInput input {
    border-radius: 12px;
}

[data-testid="stMetric"] {
    background: #1e293b;
    padding: 15px;
    border-radius: 12px;
}

.stChatMessage {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------
# Session State
# ------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------
# Cached Embeddings
# ------------------------

@st.cache_resource
def build_embeddings(chunks):
    return [
        get_embedding(chunk)
        for chunk in chunks
    ]

# ------------------------
# UI
# ------------------------

st.title("🤖 AI Document Assistant")

uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)

# ------------------------
# Process PDFs
# ------------------------

if uploaded_files:

    text = ""

    for uploaded_file in uploaded_files:
        text += load_pdf(uploaded_file)
        text += "\n"

    chunks = create_chunks(text)

    chunk_embeddings = build_embeddings(
        chunks
    )

    # Sidebar Dashboard

    with st.sidebar:

        st.title("📊 Dashboard")

        st.metric(
            "Documents",
            len(uploaded_files)
        )

        st.metric(
            "Chunks",
            len(chunks)
        )

        st.markdown("---")

        st.markdown("### Suggested Questions")

        st.markdown("""
        - How many casual leaves are provided?
        - What is the work from home policy?
        - What is the maternity leave duration?
        - What are employee benefits?
        """)

    st.success("PDFs Loaded Successfully")

    st.write(
        f"📄 Total Chunks Created: {len(chunks)}"
    )

    question = st.text_input(
        "💬 Ask a question about the uploaded documents"
    )

    if question:

        import time

        start_time = time.time()

        question_embedding = get_embedding(
            question
        )

        results = search_chunks(
            question_embedding,
            chunk_embeddings,
            chunks
        )
         # Create Context

        context = "\n".join(
            [chunk for score, chunk in results]
        )

        with st.spinner(
            "🤖 Analyzing documents..."
        ):

            answer = ask_gemini(
                question,
                context
            )

        end_time = time.time()

        # Chat Style UI

        st.chat_message("user").write(
            question
        )

        st.chat_message("assistant").write(
            answer
        )

        st.caption(
            f"⏱️ Response generated in {end_time - start_time:.2f} seconds"
        )

        # Save Chat History

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )

        # Update Sidebar

        with st.sidebar:

            st.metric(
                "Sources Found",
                len(results)
            )

        # Sources

        st.subheader("📚 Sources")

        for i, (score, chunk) in enumerate(
            results,
            start=1
        ):

            with st.expander(
                f"📄 Source {i}"
            ):

                st.caption(
                    f"Relevance Score: {score:.4f}"
                )

                st.write(chunk)

# ------------------------
# Chat History
# ------------------------

if st.session_state.chat_history:

    st.markdown("---")

    st.subheader(
        "🕒 Previous Conversations"
    )

    for chat in reversed(
        st.session_state.chat_history
    ):

        with st.expander(
            f"💬 {chat['question']}"
        ):

            st.write(
                chat["answer"]
            )