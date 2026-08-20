import time

from stt.sarvam_stt import transcribe
from rag.fast_path import retrieve_with_evidence, extract_verified_sentence

from guardrails.checks import unsafe_input_check


def _with_retry(function, max_retries=2):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return function()
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                raise
    raise last_error

CASCADE = [
    ("no_chunk", False),
    ("sentence_aware", False),
    ("no_chunk_bge", True),
]


def _fast_answer_from_question(question: str, k: int = 5) -> dict:
    start = time.perf_counter()
    result = {
        "answer": None, "sources": [], "status": "error",
        "timing": {"generation_ms": 0}, "question": question,
        "language": None, "mode": "extractive",
    }
    try:
        if not unsafe_input_check(question):
            result["status"] = "unsafe_input"
            return result

        retrieval_start = time.perf_counter()

        chunks, extraction, tier_used = [], {"verified": False}, None
        for strategy, is_bge in CASCADE:
            chunks = retrieve_with_evidence(question, strategy, k=k, use_bge=is_bge)
            extraction = extract_verified_sentence(question, chunks, use_bge=is_bge)
            if extraction.get("verified"):
                tier_used = strategy
                break

        result["timing"]["retrieval_ms"] = round(
            (time.perf_counter() - retrieval_start) * 1000, 2
        )
        result["sources"] = chunks if extraction.get("verified") else []
        result["tier"] = tier_used

        if not extraction.get("verified"):
            result["status"] = "off_topic"
            return result

        result["answer"] = extraction["answer"]
        result["source_chunk_id"] = extraction["source_chunk_id"]
        result["confidence"] = extraction["confidence"]
        result["status"] = "success"
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        result["timing"]["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return result


def run(audio_bytes: bytes, k: int = 5) -> dict:
    start = time.perf_counter()
    result = {"answer": None, "sources": [], "status": "error", "timing": {}}
    try:
        stt_start = time.perf_counter()
        transcription = _with_retry(lambda: transcribe(audio_bytes))
        question = transcription["text"]
        language = transcription.get("language")
        stt_ms = round((time.perf_counter() - stt_start) * 1000, 2)

        fast_result = _fast_answer_from_question(question, k=k)
        result.update(fast_result)
        result["language"] = language
        result["timing"]["stt_ms"] = stt_ms
        result["timing"]["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    return result


def run_from_text(question: str, k: int = 5) -> dict:
    result = _fast_answer_from_question(question, k=k)
    result["timing"]["stt_ms"] = 0
    return result


def run_extractive_from_text(question: str, k: int = 5) -> dict:
    return run_from_text(question, k=k)