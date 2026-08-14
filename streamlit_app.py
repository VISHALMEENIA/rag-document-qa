import os
import tempfile

import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Q&A")
st.write("Upload a PDF and ask questions about it.")


# ============================================================
# SESSION STATE
# ============================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embeddings():

    with st.spinner("Loading embedding model..."):

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return embeddings


# ============================================================
# LOAD LANGUAGE MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_name = "google/flan-t5-small"

    with st.spinner("Loading AI model..."):

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

    return tokenizer, model


# ============================================================
# PROCESS PDF FUNCTION
# ============================================================

def process_pdf(uploaded_file):

    pdf_path = None

    try:

        # ----------------------------------------------------
        # Save PDF temporarily
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            pdf_path = temp_file.name


        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        with st.spinner("📖 Reading PDF..."):

            loader = PyPDFLoader(pdf_path)

            documents = loader.load()

        st.success(
            f"📄 PDF loaded successfully! "
            f"Pages: {len(documents)}"
        )


        # ----------------------------------------------------
        # Split PDF
        # ----------------------------------------------------

        with st.spinner("✂️ Splitting document..."):

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.split_documents(
                documents
            )

        st.info(
            f"🧩 Number of chunks: {len(chunks)}"
        )


        # ----------------------------------------------------
        # Load embeddings
        # ----------------------------------------------------

        embeddings = load_embeddings()


        # ----------------------------------------------------
        # Create FAISS vector database
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Creating FAISS vector database..."
        ):

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

        st.success(
            "✅ FAISS vector database created successfully!"
        )

        return vectorstore


    finally:

        # ----------------------------------------------------
        # Delete temporary PDF
        # ----------------------------------------------------

        if pdf_path is not None:

            try:
                os.remove(pdf_path)

            except Exception:
                pass


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload your PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS UPLOADED PDF
# ============================================================

if uploaded_file is not None:

    # Check whether a new PDF was uploaded
    if (
        st.session_state.uploaded_file_name
        != uploaded_file.name
    ):

        st.session_state.vectorstore = None
        st.session_state.pdf_processed = False

        st.session_state.uploaded_file_name = (
            uploaded_file.name
        )


    st.success(
        f"📎 Uploaded: {uploaded_file.name}"
    )


    # --------------------------------------------------------
    # Process only once
    # --------------------------------------------------------

    if not st.session_state.pdf_processed:

        try:

            vectorstore = process_pdf(
                uploaded_file
            )

            st.session_state.vectorstore = (
                vectorstore
            )

            st.session_state.pdf_processed = True

        except Exception as e:

            st.error(
                "❌ Error while processing PDF"
            )

            st.exception(e)


    # ========================================================
    # QUESTION SECTION
    # ========================================================

    if st.session_state.vectorstore is not None:

        st.divider()

        st.subheader(
            "💬 Ask a question about the PDF"
        )


        question = st.text_input(
            "Enter your question:",
            placeholder="Example: What is Data Science?"
        )


        ask_button = st.button(
            "🔍 Ask Question",
            type="primary"
        )


        # ====================================================
        # ASK QUESTION
        # ====================================================

        if ask_button:

            if not question.strip():

                st.warning(
                    "⚠️ Please enter a question."
                )

            else:

                try:

                    # ----------------------------------------
                    # Similarity Search
                    # ----------------------------------------

                    with st.spinner(
                        "🔎 Searching document..."
                    ):

                        results = (
                            st.session_state
                            .vectorstore
                            .similarity_search(
                                question,
                                k=3
                            )
                        )


                    if not results:

                        st.warning(
                            "No relevant information "
                            "was found in the PDF."
                        )

                    else:

                        # ------------------------------------
                        # Create context
                        # ------------------------------------

                        context = "\n\n".join(
                            doc.page_content
                            for doc in results
                        )


                        # ------------------------------------
                        # Prompt
                        # ------------------------------------

                        prompt = f"""
Answer the question using ONLY the information
provided in the context.

If the answer is not present in the context,
say: "The answer is not available in the document."

Context:
{context}

Question:
{question}

Answer:
"""


                        # ------------------------------------
                        # Load model
                        # ------------------------------------

                        tokenizer, model = load_model()


                        # ------------------------------------
                        # Tokenize
                        # ------------------------------------

                        with st.spinner(
                            "🤖 Generating answer..."
                        ):

                            inputs = tokenizer(
                                prompt,
                                return_tensors="pt",
                                truncation=True,
                                max_length=512
                            )


                            # --------------------------------
                            # Generate
                            # --------------------------------

                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=150,
                                num_beams=4,
                                early_stopping=True
                            )


                            answer = tokenizer.decode(
                                outputs[0],
                                skip_special_tokens=True
                            )


                        # ------------------------------------
                        # Display Answer
                        # ------------------------------------

                        st.subheader(
                            "💡 Answer"
                        )

                        st.write(answer)


                        # ------------------------------------
                        # Sources
                        # ------------------------------------

                        st.divider()

                        with st.expander(
                            "📖 View retrieved sources"
                        ):

                            for i, doc in enumerate(
                                results,
                                start=1
                            ):

                                page_number = (
                                    doc.metadata.get(
                                        "page",
                                        "Unknown"
                                    )
                                )

                                # PDF pages are zero-indexed
                                if isinstance(
                                    page_number,
                                    int
                                ):

                                    page_number += 1


                                st.markdown(
                                    f"### Source {i} "
                                    f"— Page {page_number}"
                                )

                                st.write(
                                    doc.page_content
                                )

                                st.divider()


                except Exception as e:

                    st.error(
                        "❌ Error while answering question."
                    )

                    st.exception(e)


else:

    st.info(
        "👆 Upload a PDF to get started."
    )
