
# ============================================================
# STABLE RAG CHATBOT FOR GOOGLE COLAB (2026)
# ============================================================
#
# Uses:
# - pypdf
# - sentence-transformers
# - FAISS
# - transformers
#
# ============================================================


# ============================================================
# INSTALL PACKAGES
# ============================================================

# Run this FIRST in Colab

!pip install -q \
transformers \
sentence-transformers \
faiss-cpu \
pypdf \
accelerate

# ============================================================
# IMPORTS
# ============================================================

from google.colab import drive

from pypdf import PdfReader

from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
import pickle

from transformers import pipeline

import os

# ============================================================
# MOUNT GOOGLE DRIVE
# ============================================================

drive.mount('/content/drive')

# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        reader = PdfReader(pdf)

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

    return text

# ============================================================
# TEXT CHUNKING
# ============================================================

def get_text_chunks(
    text,
    chunk_size=1000,
    overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2'
)

print("Embedding model loaded.")

# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vector_db(chunks):

    print("Creating embeddings...")

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print("FAISS index created.")

    return index, embeddings

# ============================================================
# SAVE VECTOR DATABASE
# ============================================================

def save_vector_db(index, chunks, path):

    faiss.write_index(index, f"{path}/faiss.index")

    with open(f"{path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Vector DB saved.")

# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

def load_vector_db(path):

    index = faiss.read_index(f"{path}/faiss.index")

    with open(f"{path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    print("Vector DB loaded.")

    return index, chunks

# ============================================================
# RETRIEVE TOP CHUNKS
# ============================================================

def retrieve(query, index, chunks, k=3):

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(query_embedding).astype('float32')

    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []

    for idx in indices[0]:

        retrieved_chunks.append(chunks[idx])

    return retrieved_chunks

# ============================================================
# LOAD LOCAL LLM
# ============================================================

print("Loading local LLM...")

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=256
)

print("LLM loaded.")

# ============================================================
# GENERATE RESPONSE
# ============================================================

def ask_llm(prompt):

    response = pipe(
        prompt,
        do_sample=True,
        temperature=0.5
    )

    return response[0]["generated_text"]

# ============================================================
# PDF PATH
# ============================================================

path_to_pdf = [
    '/content/drive/MyDrive/book.pdf'
]

# Change this filename to your PDF

# ============================================================
# EXTRACT TEXT
# ============================================================

print("Extracting text from PDF...")

raw_text = get_pdf_text(['/content/drive/MyDrive/Bernard S. An Introduction To Enterprise Architecture 3ed 2012.pdf'])

print("Text extraction complete.")

# ============================================================
# CREATE CHUNKS
# ============================================================

print("Creating text chunks...")

chunks = get_text_chunks(raw_text)

print(f"Total chunks: {len(chunks)}")

# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

index, embeddings = create_vector_db(chunks)

# ============================================================
# SAVE DATABASE
# ============================================================

SAVE_PATH = "/content/drive/MyDrive/rag_db"

os.makedirs(SAVE_PATH, exist_ok=True)

save_vector_db(index, chunks, SAVE_PATH)

# ============================================================
# CHAT LOOP
# ============================================================

print("\nRAG chatbot ready.")
print("Type 'exit' to quit.\n")

while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":

        print("Goodbye.")
        break

    try:

        # Retrieve relevant chunks
        retrieved_chunks = retrieve(
            query,
            index,
            chunks
        )

        # Build context
        context = "\n\n".join(retrieved_chunks)

        # Prompt
        prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

        # Generate answer
        answer = ask_llm(prompt)

        print("\nANSWER:\n")

        print(answer)

        print("\n" + "=" * 60)

    except Exception as e:

        print("\nError occurred:")

        print(e)
