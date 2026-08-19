import sys
import io
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from rag.retriever import retrieve
import time

queries = [
    ("Hindi", "क्या आप पात्र में लाल प्याज उगा सकते हैं?", 78076),
    ("Bengali", "আপনি কি পাত্রে লাল পেঁয়াজ জন্মাতে পারেন?", 78076),
    ("English", "Can you grow red onions in containers?", 78076),
]

strategy = "no_chunk_bge"
print(f"Testing Strategy: {strategy}")
print("=" * 70)

for lang, query_text, expected_qid in queries:
    print(f"\n--- Language: {lang} | Query: '{query_text}' ---")
    t0 = time.perf_counter()
    results = retrieve(query_text, strategy_name=strategy, k=5)
    t1 = time.perf_counter()
    print(f"Latency: {(t1 - t0) * 1000:.1f}ms")
    
    found = False
    for rank, res in enumerate(results, start=1):
        is_match = (res["is_selected"] == 1 and float(res["query_id"]) == float(expected_qid))
        mark = " <<< [CORRECT TARGET MATCH]" if is_match else ""
        print(f"  Rank {rank} | Score: {res['score']:.4f} | Lang: {res['language']} | QID: {res['query_id']}{mark}")
        if is_match:
            found = True
            print(f"         Snippet: {res['chunk_text'][:120]}...")
            
    if not found:
        print("  ❌ Target match NOT in top 5!")