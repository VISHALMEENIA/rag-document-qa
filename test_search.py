from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the saved FAISS database
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Ask a question
question = "What is Data Science?"

# Search the FAISS database
results = vectorstore.similarity_search(question, k=3)

print("\nSearch Results:\n")

for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(doc.page_content)
    print()