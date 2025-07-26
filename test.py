from rag_pipeline import rag_answer
import evaluate  # your evaluation module

test_cases = [
    ("অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "শুম্ভুনাথ"),
    ("কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "মামাকে"),
    ("বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "১৫ বছর"),
]

if __name__ == "__main__":
    results = []

    for i, (query, expected) in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i} ===")
        print("🔎 Question:", query)

        try:
            answer, chunks = rag_answer(query, return_chunks=True)
            results.append((query, expected, answer, chunks))
        except Exception as e:
            print(f"❌ Error during rag_answer: {e}")
            continue

        print("📚 Retrieved Chunks:")
        for idx, chunk in enumerate(chunks, 1):
            print(f"Chunk {idx}:\n{chunk}\n")

        print("✅ Answer:\n", answer)
        print("🎯 Expected:", expected)

        with open(f"output_test{i}.txt", "w", encoding="utf-8") as f:
            f.write(f"Question: {query}\n\n")
            f.write(f"Answer: {answer}\n\n")
            f.write(f"Expected: {expected}\n\n")
            f.write("Chunks:\n")
            f.write("\n---\n".join(chunks))

    print("\n\n=== Running Evaluation ===")
    evaluate.evaluate_with_results(results)

