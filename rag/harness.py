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