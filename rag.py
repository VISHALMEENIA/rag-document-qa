from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading FAISS database...")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Loading language model...")

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Everything loaded successfully!")

print("\n================================")
print("     RAG DOCUMENT CHATBOT")
print("================================")
print("Type 'exit' to stop.\n")


while True:

    question = input("Enter your question: ")

    if question.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    if not question.strip():
        continue

    # Search FAISS
    results = vectorstore.similarity_search(
        question,
        k=3
    )

    # Use the most relevant chunk
    context = results[0].page_content

    # Create prompt
    prompt = f"""
Context:
{context}

Question:
{question}

Answer the question using only the information from the context.
Give a clear and complete answer.
"""

    print("Generating answer...")

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Generate answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=4,
        early_stopping=True
    )

    # Decode
    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\nAnswer:")
    print(answer)
    print("\n" + "-" * 50 + "\n")