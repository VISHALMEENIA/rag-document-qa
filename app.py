from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NEW IMPORTS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PDF_PATH = "documents/UNIT 1.pdf"

# Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print("PDF loaded successfully!")
print("Number of pages:", len(documents))


# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

print("\nFirst chunk:\n")
print(chunks[0].page_content)


# --------------------------------
# CREATE EMBEDDINGS
# --------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating embeddings...")


# --------------------------------
# CREATE FAISS VECTOR DATABASE
# --------------------------------

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("FAISS vector database created successfully!")


# --------------------------------
# SAVE FAISS DATABASE
# --------------------------------

vectorstore.save_local("faiss_index")

print("FAISS index saved successfully!")