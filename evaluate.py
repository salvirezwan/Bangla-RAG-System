from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load once at module level for efficiency
eval_embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def compute_cosine_sim(text1, text2):
    emb1 = eval_embed_model.encode([text1], convert_to_numpy=True)
    emb2 = eval_embed_model.encode([text2], convert_to_numpy=True)
    return cosine_similarity(emb1, emb2)[0][0]

def evaluate_with_results(results):
    groundedness_scores = []
    relevance_scores = []

    lines = []
    for i, (query, expected, answer, chunks) in enumerate(results, 1):
        context_text = "\n\n".join(chunks)

        groundedness = compute_cosine_sim(answer, context_text)  # How well answer matches context
        relevance = compute_cosine_sim(query, context_text)      # How relevant context is to query

        groundedness_scores.append(groundedness)
        relevance_scores.append(relevance)

        block = (
            f"\n=== Evaluation for Test Case {i} ===\n"
            f"Query: {query}\n"
            f"Answer: {answer}\n"
            f"Groundedness (Answer vs Context): {groundedness:.4f}\n"
            f"Relevance (Query vs Context): {relevance:.4f}\n"
        )
        print(block)
        lines.append(block)

    avg_groundedness = np.mean(groundedness_scores) if groundedness_scores else 0
    avg_relevance = np.mean(relevance_scores) if relevance_scores else 0

    summary = (
        "\n=== Overall Evaluation Summary ===\n"
        f"Average Groundedness: {avg_groundedness:.4f}\n"
        f"Average Relevance: {avg_relevance:.4f}\n"
    )
    print(summary)
    lines.append(summary)

    # Write all evaluation output to file
    with open("evaluation.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)

