# query.py — Offline RAG using TinyLlama (fast CPU model)
# Works with LangChain 1.x+ and no OpenAI key

import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


VECTOR_STORE_PATH = "vector_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  


device = "cpu"
torch.backends.mps.enabled = False


print("💾 Loading vector store...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})


print(f"🧠 Loading local model ({LLM_MODEL})... this may take ~20 seconds ⏳")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    dtype=torch.float32,
    device_map={"": device}
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,
    do_sample=True,
    top_p=0.9,
)
llm = HuggingFacePipeline(pipeline=pipe)


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful medical assistant. Use the provided context to answer accurately.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
)

def combine_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | combine_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)


print("\n✅ Offline assistant ready! Type 'exit' to quit.\n")

while True:
    q = input("🧠 Question: ")
    if q.lower() in ["exit", "quit", "q"]:
        print("👋 Exiting. Have a great day!")
        break

    answer = rag_chain.invoke(q)
    print(f"\n💬 Answer: {answer}\n")
