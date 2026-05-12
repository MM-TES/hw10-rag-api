"""LLM-as-judge via Anthropic Haiku 4.5."""
from __future__ import annotations

import asyncio
import json
import os

from anthropic import APIError, APIStatusError, AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

JUDGE_PROMPT = """You are evaluating a RAG system's answer quality.

QUESTION: {question}
EXPECTED ANSWER (ground truth keywords): {keywords}
ACTUAL ANSWER: {actual}

For "out_of_scope" category, expected behavior is REFUSAL like "I don't know based on the provided document".
If correctly refused — give 5/5/5. If hallucinated — give 1/1/1.

Score 1-5 on three dimensions (1=terrible, 5=excellent):
- faithfulness: stays grounded in retrieved context, no hallucination
- relevance: actually answers the question
- completeness: covers the expected facts

Return ONLY valid JSON, no markdown, no preamble:
{{"faithfulness": N, "relevance": N, "completeness": N, "rationale": "one short sentence"}}"""

REFUSAL_MARKERS = (
    "i don't know",
    "not in the",
    "cannot answer",
    "provided document",
    "based on the provided",
    "no information",
)


def _is_refusal(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Drop opening fence + optional language tag
        first_nl = text.find("\n")
        if first_nl >= 0:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


async def judge_answer(question, expected_keywords, actual, category="factual"):
    if category == "out_of_scope":
        if _is_refusal(actual):
            return {
                "faithfulness": 5,
                "relevance": 5,
                "completeness": 5,
                "rationale": "Correctly refused",
            }
        # else fall through to LLM judge — it will penalize hallucination

    keywords_str = ", ".join(expected_keywords) if expected_keywords else "(none)"
    prompt = JUDGE_PROMPT.format(
        question=question,
        keywords=keywords_str,
        actual=(actual or "")[:2000],
    )

    last_err: str | None = None
    for attempt in range(2):
        try:
            msg = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = _strip_code_fence(msg.content[0].text)
            try:
                parsed = json.loads(text)
                # Coerce fields to ints
                return {
                    "faithfulness": int(parsed.get("faithfulness", 0)),
                    "relevance": int(parsed.get("relevance", 0)),
                    "completeness": int(parsed.get("completeness", 0)),
                    "rationale": str(parsed.get("rationale", ""))[:240],
                }
            except (json.JSONDecodeError, ValueError):
                return {
                    "faithfulness": 0,
                    "relevance": 0,
                    "completeness": 0,
                    "rationale": f"Parse error: {text[:200]}",
                }
        except APIStatusError as e:
            last_err = f"http_{e.status_code}: {str(e)[:120]}"
            if e.status_code == 429 and attempt == 0:
                await asyncio.sleep(30)
                continue
            break
        except APIError as e:
            last_err = f"api_error: {str(e)[:120]}"
            break

    return {
        "faithfulness": 0,
        "relevance": 0,
        "completeness": 0,
        "rationale": f"judge_unavailable: {last_err}",
    }


if __name__ == "__main__":
    async def test():
        r = await judge_answer(
            question="What is the first factor?",
            expected_keywords=["codebase", "version control"],
            actual="The first factor is Codebase - one codebase tracked in version control.",
            category="factual",
        )
        print(json.dumps(r, indent=2))

    asyncio.run(test())
