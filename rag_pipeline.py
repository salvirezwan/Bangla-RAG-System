# rag_pipeline.py
from embed_store import VectorStore
from query_engine import ask_gemini

# Load preprocessed text (NO OCR)
with open("HSC26_Bangla_1st_Paper_cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()

store = VectorStore()
chunks = store.chunk_text(text)
store.embed_chunks()

def rerank_by_keyword(chunks, query):
    q_terms = set(query.split())
    processed_chunks = [str(c.text) if hasattr(c, 'text') else str(c) for c in chunks]
    return sorted(processed_chunks, key=lambda c: -len(q_terms & set(c.split())))

def rag_answer(query, return_chunks=False):
    top_chunks = store.search(query, top_k=7)
    top_chunks = rerank_by_keyword(top_chunks, query)[:5]
    context_text = "\n\n".join(top_chunks)

    prompt = f"""
তুমি একজন বিশেষজ্ঞ বাংলা সাহিত্য সহকারী। তুমি নিচের প্রাসঙ্গিক তথ্যের উপর ভিত্তি করে সেরা সম্ভাব্য উত্তর দেবে। 
তুমি এমনভাবে উত্তর দেবে যেন এটা একজন এইচএসসি পরীক্ষার্থীর জন্য বোঝা সহজ হয়। 
যদি নির্দিষ্ট তথ্য না পাওয়া যায়, তবে তোমার অনুমান বলবে, কিন্তু সৎ থাকবে।

প্রাসঙ্গিক তথ্য:
{context_text}

প্রশ্ন:
{query}

উত্তর (বাংলায় লিখো):
"""

    answer = ask_gemini(prompt).strip()
    return (answer, top_chunks) if return_chunks else answer



