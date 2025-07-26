from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.text_chunks = []

    def chunk_text(self, text, max_len=500):
        paragraphs = text.split('\n\n')  # split by paragraphs
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if len(current_chunk) + len(para) + 2 <= max_len:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If paragraph is longer than max_len, split by sentence-ending danda or period
                if len(para) > max_len:
                    # Split by danda or period, but split at the earliest delimiter found
                    sentences = []
                    temp = para
                    while len(temp) > 0:
                        danda_pos = temp.find('।')
                        period_pos = temp.find('.')
                        # Find earliest delimiter position >=0
                        candidates = [pos for pos in [danda_pos, period_pos] if pos != -1]
                        if not candidates:
                            # No delimiter found, take all
                            sentences.append(temp)
                            temp = ""
                        else:
                            split_pos = min(candidates)
                            sentences.append(temp[:split_pos+1])
                            temp = temp[split_pos+1:].strip()

                    sent_chunk = ""
                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent_chunk) + len(sent) + 1 <= max_len:
                            sent_chunk += sent + " "
                        else:
                            chunks.append(sent_chunk.strip())
                            sent_chunk = sent + " "
                    if sent_chunk:
                        chunks.append(sent_chunk.strip())
                    current_chunk = ""
                else:
                    current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        self.text_chunks = chunks
        return self.text_chunks

    def embed_chunks(self):
        if not self.text_chunks:
            raise ValueError("No text chunks found. Call chunk_text() first.")
        embeddings = self.model.encode(self.text_chunks, convert_to_numpy=True)
        # Normalize embeddings for cosine similarity with IndexFlatIP
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query, top_k=5):
        if self.index is None:
            raise ValueError("Index is not built. Call embed_chunks() first.")
        query_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec, top_k)
        return [self.text_chunks[i] for i in I[0]]











