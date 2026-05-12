"""LLM-as-judge — dual judge: Anthropic Haiku 4.5 + OpenAI gpt-4o-mini (via OpenRouter).

Returns three score sets per call:
  - haiku        (Anthropic native)
  - openai_mini  (OpenRouter -> openai/gpt-4o-mini)
  - mean         (per-metric arithmetic mean, used as the headline number)

Rationale: two different families reduces single-judge self-bias and gives us a
direct inter-rater agreement signal (Spearman / Kendall in EXP-09).
"""
from __future__ import annotations

import asyncio
import json
import os

from anthropic import APIError, APIStatusError, AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

_anthropic = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

HAIKU_MODEL = "claude-haiku-4-5-20251001"
OPENAI_JUDGE_MODEL = "openai/gpt-4o-mini"  # via OpenRouter

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
        first_nl = text.find("\n")
        if first_nl >= 0:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


def _parse_judge_json(text: str) -> dict:
    text = _strip_code_fence(text)
    try:
        parsed = json.loads(text)
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


def _refusal_shortcut() -> dict:
    return {
        "faithfulness": 5,
        "relevance": 5,
        "completeness": 5,
        "rationale": "Correctly refused",
    }


# --- Haiku judge (Anthropic native) -----------------------------------------


async def judge_haiku(question, expected_keywords, actual, category="factual") -> dict:
    if category == "out_of_scope" and _is_refusal(actual or ""):
        return _refusal_shortcut()

    keywords_str = ", ".join(expected_keywords) if expected_keywords else "(none)"
    prompt = JUDGE_PROMPT.format(
        question=question,
        keywords=keywords_str,
        actual=(actual or "")[:2000],
    )

    last_err: str | None = None
    for attempt in range(2):
        try:
            msg = await _anthropic.messages.create(
                model=HAIKU_MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_judge_json(msg.content[0].text)
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
        "rationale": f"judge_unavailable_haiku: {last_err}",
    }


# --- OpenAI gpt-4o-mini judge (via OpenRouter) ------------------------------


async def judge_openai_mini(question, expected_keywords, actual, category="factual") -> dict:
    if category == "out_of_scope" and _is_refusal(actual or ""):
        return _refusal_shortcut()

    # Lazy import to avoid cycle / heavy import on judge_haiku-only paths
    from experiments.common import get_openrouter

    client = get_openrouter()
    keywords_str = ", ".join(expected_keywords) if expected_keywords else "(none)"
    prompt = JUDGE_PROMPT.format(
        question=question,
        keywords=keywords_str,
        actual=(actual or "")[:2000],
    )

    last_err: str | None = None
    for attempt in range(2):
        try:
            resp = await client.chat.completions.create(
                model=OPENAI_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
            return _parse_judge_json(text)
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            if attempt == 0 and "429" in last_err:
                await asyncio.sleep(15)
                continue
            break

    return {
        "faithfulness": 0,
        "relevance": 0,
        "completeness": 0,
        "rationale": f"judge_unavailable_4omini: {last_err}",
    }


# --- Dual judge -------------------------------------------------------------


async def dual_judge(question, expected_keywords, actual, category="factual") -> dict:
    """Run both judges in parallel; return {haiku, openai_mini, mean}.

    `mean` is the headline number used by aggregate CSVs and plots; the raw
    per-judge scores are recorded for inter-rater agreement (EXP-09).
    """
    haiku_task = judge_haiku(question, expected_keywords, actual, category)
    openai_task = judge_openai_mini(question, expected_keywords, actual, category)
    haiku, openai_mini = await asyncio.gather(haiku_task, openai_task)

    def _mean(a: int, b: int) -> float:
        if a <= 0 and b <= 0:
            return 0.0
        if a <= 0:
            return float(b)
        if b <= 0:
            return float(a)
        return (a + b) / 2.0

    mean = {
        "faithfulness": _mean(haiku["faithfulness"], openai_mini["faithfulness"]),
        "relevance": _mean(haiku["relevance"], openai_mini["relevance"]),
        "completeness": _mean(haiku["completeness"], openai_mini["completeness"]),
    }
    return {"haiku": haiku, "openai_mini": openai_mini, "mean": mean}


# Back-compat alias for any caller still using the old name
judge_answer = judge_haiku


if __name__ == "__main__":
    async def test():
        r = await dual_judge(
            question="What is the first factor?",
            expected_keywords=["codebase", "version control"],
            actual="The first factor is Codebase - one codebase tracked in version control.",
            category="factual",
        )
        print(json.dumps(r, indent=2))

    asyncio.run(test())
