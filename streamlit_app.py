import streamlit as st
import tempfile
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚"
)

st.title("📚 RAG Document Q&A")
st.write("Upload a PDF and ask questions about it.")


# --------------------------------
# LOAD LANGUAGE MODEL
# --------------------------------

@st.cache_resource
def load_model():

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


# --------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


#tokenizer, model = load_model()
#embeddings = load_embeddings()


# --------------------------------
# PDF UPLOAD
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# --------------------------------
# PROCESS PDF
# --------------------------------

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        pdf_path = temp_file.name


    # Load PDF
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    st.write(f"📄 Pages: {len(documents)}")


    # Split document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    st.write(f"✂️ Chunks: {len(chunks)}")


    # Create FAISS database
    with st.spinner("Creating document index..."):

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    st.success("✅ Document processed successfully!")


    # --------------------------------
    # QUESTION
    # --------------------------------

    question = st.text_input(
        "Ask a question about the PDF:"
    )


    if st.button("Ask"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("Searching document..."):

                # Search
                results = vectorstore.similarity_search(
                    question,
                    k=3
                )


                # Combine retrieved chunks
                context = "\n\n".join(
                    doc.page_content
                    for doc in results
                )


                # --------------------------------
                # PROMPT
                # --------------------------------

                prompt = f"""
Context:
{context}

Question:
{question}

Answer the question using only the information
from the context.

Give a clear and complete answer.
"""


                # --------------------------------
                # GENERATE ANSWER
                # --------------------------------

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )


                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    num_beams=4
                )


                answer = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )


            # --------------------------------
            # DISPLAY ANSWER
            # --------------------------------

            st.subheader("💡 Answer")

            st.write(answer)


            # --------------------------------
            # SHOW SOURCES
            # --------------------------------

            with st.expander("📖 View retrieved sources"):

                for i, doc in enumerate(results, 1):

                    st.write(
                        f"**Source {i} — Page "
                        f"{doc.metadata.get('page', 'Unknown')}**"
                    )

                    st.write(doc.page_content)


    # Remove temporary file
    try:
        os.remove(pdf_path)
    except:
        pass

else:

    st.info("👆 Upload a PDF to get started.")
