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

    st.success("PDFs Loaded Successfully")

    st.write(
        f"Total Chunks Created: {len(chunks)}"
    )

    question = st.text_input(
        "Ask a question about the uploaded documents"
    )

    if question:

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

        # Gemini Answer

        answer = ask_gemini(
            question,
            context
        )

        # Display Answer

        st.subheader("AI Answer")

        st.success(answer)

        # Save Chat History

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )

        # Sources

        st.subheader("Sources")

        for i, (score, chunk) in enumerate(
            results,
            start=1
        ):

            with st.expander(
                f"Source {i} | Similarity {score:.4f}"
            ):
                st.write(chunk)

# ------------------------
# Chat History
# ------------------------

if st.session_state.chat_history:

    st.subheader("Chat History")

    for chat in reversed(
        st.session_state.chat_history
    ):

        with st.expander(
            chat["question"]
        ):
            st.write(
                chat["answer"]
            )