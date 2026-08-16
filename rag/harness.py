import time

from stt.sarvam_stt import transcribe
from rag.retriever import retrieve
from rag.generator import generate

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

        result["question"] = question
        result["language"] = language

        # -------------------------
        # 2. Unsafe input check
        # -------------------------
        if not unsafe_input_check(question):
            result["status"] = "unsafe_input"
            result["timing"]["total_ms"] = round(
                (time.perf_counter() - start) * 1000,
                2,
            )
            return result

        # -------------------------
        # 3. Retrieval
        # -------------------------
        retrieval_start = time.perf_counter()

        chunks = retrieve(
            question,
            BEST_STRATEGY,
            k=k,
        )

        result["timing"]["retrieval_ms"] = round(
            (time.perf_counter() - retrieval_start) * 1000,
            2,
        )

        # -------------------------
        # 4. Off-topic check
        # -------------------------
        if not off_topic_check(chunks):
            result["status"] = "off_topic"
            result["sources"] = chunks

            result["timing"]["total_ms"] = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            return result

        # -------------------------
        # 5. Generation
        # -------------------------
        generation_start = time.perf_counter()

        answer = _with_retry(
            lambda: generate(question, chunks)
        )

        result["timing"]["generation_ms"] = round(
            (time.perf_counter() - generation_start) * 1000,
            2,
        )

        # -------------------------
        # 6. Groundedness check
        # -------------------------
        if not groundedness_check(answer, chunks):
            result["status"] = "ungrounded"
            result["sources"] = chunks

            result["timing"]["total_ms"] = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            return result

        # -------------------------
        # 7. Success
        # -------------------------
        result["answer"] = answer
        result["sources"] = chunks
        result["status"] = "success"

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
    start = time.perf_counter()

    result = {
        "answer": None,
        "sources": [],
        "status": "error",
        "timing": {"stt_ms": 0},
        "question": question,
        "language": None,
    }

    try:
        if not unsafe_input_check(question):
            result["status"] = "unsafe_input"
            result["timing"]["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
            return result

        retrieval_start = time.perf_counter()
        chunks = retrieve(question, BEST_STRATEGY, k=k)
        result["timing"]["retrieval_ms"] = round((time.perf_counter() - retrieval_start) * 1000, 2)

        if not off_topic_check(chunks):
            result["status"] = "off_topic"
            result["sources"] = chunks
            result["timing"]["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
            return result

        generation_start = time.perf_counter()
        answer = _with_retry(lambda: generate(question, chunks))
        result["timing"]["generation_ms"] = round((time.perf_counter() - generation_start) * 1000, 2)

        if not groundedness_check(answer, chunks):
            result["status"] = "ungrounded"
            result["sources"] = chunks
            result["timing"]["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
            return result

        result["answer"] = answer
        result["sources"] = chunks
        result["status"] = "success"

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    result["timing"]["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return result


def run_extractive_from_text(question: str, k: int = 5) -> dict:
    """
    LLM-free pipeline: hybrid (dense + BM25) retrieval, then extractive
    sentence selection - no Groq call, no network round trip beyond
    retrieval itself. Much lower and more predictable latency than the
    generative path, at the cost of less fluent answers (verbatim spans,
    not composed sentences).
    """
    from rag.hybrid_retriever import hybrid_retrieve
    from rag.extractive import extract_answer

    start = time.perf_counter()

    result = {
        "answer": None,
        "sources": [],
        "status": "error",
        "timing": {"stt_ms": 0, "generation_ms": 0},
        "question": question,
        "language": None,
        "mode": "extractive",
    }

    try:
        if not unsafe_input_check(question):
            result["status"] = "unsafe_input"
            result["timing"]["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
            return result

        retrieval_start = time.perf_counter()
        chunks = hybrid_retrieve(question, BEST_STRATEGY, k=k)
        result["timing"]["retrieval_ms"] = round((time.perf_counter() - retrieval_start) * 1000, 2)

        if not off_topic_check(chunks):
            result["status"] = "off_topic"
            result["sources"] = chunks
            result["timing"]["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
            return result

        extraction = extract_answer(question, chunks)

        if not extraction["answer"] or extraction["score"] < 0.3:
            result["status"] = "ungrounded"
            result["sources"] = chunks
            result["timing"]["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
            return result

        result["answer"] = extraction["answer"]
        result["sources"] = chunks
        result["status"] = "success"

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    result["timing"]["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return result