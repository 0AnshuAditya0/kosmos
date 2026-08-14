import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = """
You are a retrieval-augmented assistant.

Answer ONLY using the provided context.

Rules:
1. Do not use outside knowledge.
2. If the context does not contain enough information to answer,
   say "I don't know."
3. Do not invent facts.
4. Keep the answer concise and directly answer the question.
"""


class Generator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

        self.client = Groq(api_key=api_key)

    def generate(self, question: str, chunks: list[dict]) -> str:
        if not question:
            raise ValueError("question cannot be empty")

        context_parts = []

        for i, chunk in enumerate(chunks, start=1):
            text = chunk.get("chunk_text", "")

            if text:
                context_parts.append(
                    f"[Source {i}]\n{text}"
                )

        context = "\n\n".join(context_parts)

        if not context:
            return "I don't know."

        user_prompt = f"""
Context:

{context}

Question:
{question}

Answer using ONLY the context above.
"""

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        return response.choices[0].message.content.strip()


_default_generator = None


def generate(question: str, chunks: list[dict]) -> str:
    global _default_generator

    if _default_generator is None:
        _default_generator = Generator()

    return _default_generator.generate(question, chunks)