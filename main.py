import os
import numpy as np
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import faiss
import uuid

# -----------------------------
# CONFIGURATION
# -----------------------------
PDF_PATH = "/Users/ahir/disease_assitance_service/data/book.pdf"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store"

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print("📘 Step 1: Loading PDF...")

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"❌ File '{PDF_PATH}' not found in current directory!")

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages from PDF.")

    # -----------------------------
    print("✂️ Step 2: Splitting into text chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = splitter.split_documents(docs)
    print(f"✅ Split into {len(texts)} chunks.")

    # -----------------------------
    print("🧠 Step 3: Creating embeddings...")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectors = [embedding_model.embed_query(chunk.page_content) for chunk in tqdm(texts)]

    print(f"✅ Created {len(vectors)} embeddings, each of dimension {len(vectors[0])}.")

    # -----------------------------
    print("💾 Step 4: Building FAISS index...")

    emb_array = np.ascontiguousarray(np.stack(vectors).astype("float32"))
    embedding_dim = emb_array.shape[1]

    index = faiss.IndexFlatL2(int(embedding_dim))
    index.add(emb_array)
    print(f"✅ Added {index.ntotal} vectors to FAISS index.")

    # -----------------------------
    print("🗂️ Step 5: Building document store...")

    documents = [Document(page_content=t.page_content) for t in texts[:len(emb_array)]]
    index_to_docstore_id = [str(uuid.uuid4()) for _ in range(len(documents))]
    docstore = InMemoryDocstore(dict(zip(index_to_docstore_id, documents)))

    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    # -----------------------------
    print("💾 Step 6: Saving vector store locally...")
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"✅ Vector store created and saved in '{VECTOR_STORE_PATH}/'.")

    print("\n🎉 All steps completed successfully!")

# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    main()
