from rag.retriever import retrieve, STRATEGIES


query = "क्या आप पात्र में लाल प्याज उगा सकते हैं?"

for strategy in STRATEGIES:
    print("\n" + "=" * 70)
    print(f"STRATEGY: {strategy}")
    print("=" * 70)

    results = retrieve(
        query,
        strategy_name=strategy,
        k=5
    )

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Score: {result['score']:.4f}")
        print(f"Query ID: {result['query_id']}")
        print(f"Selected: {result['is_selected']}")
        print(f"Language: {result['language']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Chunk: {result['chunk_text'][:250]}")

        if (
            result["is_selected"] == 1
            and result["query_id"] == 78076
        ):
            print(">>> CORRECT PASSAGE FOUND <<<")