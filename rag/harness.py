import time

from stt.sarvam_stt import transcribe
from rag.generator import generate
from rag.retriever import retrieve
from rag.fast_path import retrieve_with_evidence, extract_verified_sentence

from guardrails.checks import (
    unsafe_input_check,
    off_topic_check,
    groundedness_check,
)


BEST_STRATEGY = "no_chunk"


def _with_retry(function, max_retries=2):
    """
    Retry a failing external call up to max_retries times.
    """

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return function()
        except Exception as exc:
            last_error = exc

            if attempt == max_retries:
                raise

    raise last_error


FALLBACK_STRATEGIES = ["no_chunk", "sentence_aware", "fixed_size", "fixed_size_overlap"]

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

        chunks, extraction = [], {"verified": False}
        for strategy in FALLBACK_STRATEGIES:
            chunks = retrieve_with_evidence(question, strategy, k=k)
            extraction = extract_verified_sentence(question, chunks)
            if extraction.get("verified"):
                break

        result["timing"]["retrieval_ms"] = round(
            (time.perf_counter() - retrieval_start) * 1000, 2
        )
        result["sources"] = chunks if extraction.get("verified") else []

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

    result = {
        "answer": None,
        "sources": [],
        "status": "error",
        "timing": {},
    }

    try:
        # -------------------------
        # 1. Speech to text
        # -------------------------
        stt_start = time.perf_counter()

        transcription = _with_retry(
            lambda: transcribe(audio_bytes)
        )

        question = transcription["text"]
        language = transcription.get("language")

        result["timing"]["stt_ms"] = round(
            (time.perf_counter() - stt_start) * 1000,
            2,
        )
        stt_ms = result["timing"]["stt_ms"]

        result["question"] = question
        result["language"] = language

        fast_result = _fast_answer_from_question(question, k=k)
        result.update(fast_result)
        result["language"] = language
        result["timing"]["stt_ms"] = stt_ms
        result["timing"]["total_ms"] = round(
            (time.perf_counter() - start) * 1000, 2
        )

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    result["timing"]["total_ms"] = round(
        (time.perf_counter() - start) * 1000,
        2,
    )

    return result

def run_from_text(question: str, k: int = 5) -> dict:
    """
    Same pipeline as run(), but for text input instead of audio.
    Skips the STT stage entirely - used by the /ask-text fallback route
    for when a user types instead of speaks (or when a mic isn't available).
    """
    result = _fast_answer_from_question(question, k=k)
    result["timing"]["stt_ms"] = 0
    return result


def run_extractive_from_text(question: str, k: int = 5) -> dict:
    """
    LLM-free pipeline: hybrid (dense + BM25) retrieval, then extractive
    sentence selection - no Groq call, no network round trip beyond
    retrieval itself. Much lower and more predictable latency than the
    generative path, at the cost of less fluent answers (verbatim spans,
    not composed sentences).
    """
    result = _fast_answer_from_question(question, k=k)
    result["timing"]["stt_ms"] = 0
    return result
