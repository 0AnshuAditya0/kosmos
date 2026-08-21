from rag.fast_path import retrieve_with_evidence, extract_verified_sentence

FALLBACK_STRATEGIES = ["no_chunk", "sentence_aware", "fixed_size", "fixed_size_overlap"]

CANDIDATES = [
    "कॉर्पोरेशन क्या है?",
    "टेस्ला कॉइल क्या है?",
    "ब्लिस्टर पैक क्या है?",
    "स्नेयर ड्रम क्या है?",
    "क्या आप पात्र में लाल प्याज उगा सकते हैं?",
    "फ्लुइड फोर्स क्या है?",
    "आज मौसम कैसा है?",
    "आपका पसंदीदा रंग क्या है?",
]


def check(question):
    for strategy in FALLBACK_STRATEGIES:
        chunks = retrieve_with_evidence(question, strategy, k=5)
        result = extract_verified_sentence(question, chunks)
        if result.get("verified"):
            return True, strategy, result["answer"][:70]
    return False, None, None


if __name__ == "__main__":
    print(f"{'VERIFIED':<10} {'STRATEGY':<20} QUESTION -> ANSWER PREVIEW")
    print("-" * 90)
    for q in CANDIDATES:
        verified, strategy, preview = check(q)
        status = "YES" if verified else "NO"
        print(f"{status:<10} {strategy or '-':<20} {q}")
        if preview:
            print(f"{'':10} {'':20} -> {preview}")