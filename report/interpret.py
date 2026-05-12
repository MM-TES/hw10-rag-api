"""Claude Opus interpretations for Phase 2 experiments (Ukrainian)."""
from __future__ import annotations

import os

from anthropic import APIError, APIStatusError, AsyncAnthropic
from dotenv import load_dotenv

from experiments.budget_guard import BudgetExceededError, guard
from experiments.common import write_log

load_dotenv()

_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL_OPUS = "claude-opus-4-7"
PER_OPUS_BUDGET_USD = 0.15  # very generous; only enforced if request_logs sees prod spend

INTERPRET_PROMPT = """Ти — старший інженер AI/ML-інфраструктури, який ревʼює результати експериментів з Production-Ready RAG API.

Напиши технічну інтерпретацію 3-5 параграфами **українською** для експерименту нижче.

Охопи:
1. Що показують дані — кількісні спостереження (цифри з CSV)
2. Чому — корінна причина на основі теорії RAG/LLM
3. Trade-off — що виграли, що програли
4. Інженерне рішення — яку конфігурацію виставляти в проді?
5. Caveats — що цей експеримент НЕ доводить

Стиль: технічний, без води, з цифрами. Без хеджування.

---

EXPERIMENT: {exp_id} — {exp_title}

HYPOTHESIS: {hypothesis}

CSV DATA:
{csv_data}

KEY METRICS NOTE: {key_metrics}
"""

EXEC_PROMPT = """Ти — старший інженер AI/ML-інфраструктури. Напиши **одно-параграфний executive summary** українською мовою на 4-6 речень для Phase 2 R&D-звіту проекту HW10 Production-Ready RAG API.

Контекст: проведено 8 експериментів (chunking sweep, top-K sweep, cache threshold, model comparison, load test, fallback observed, prompt-injection suite, cost projection). Сервіс задеплоєно на Fly.io shared-cpu-1x.

Ключові цифри з усіх 8 експериментів:
{summary}

Пиши конкретно з цифрами. Без шаблонних фраз типу «в результаті дослідження». Жодних емодзі.
"""

CROSS_PROMPT = """Ти — старший інженер AI/ML-інфраструктури. Виведи 3-4 крос-експериментальних інсайти українською мовою для Phase 2 R&D-звіту.

Контекст: 8 експериментів виконано. Цифри (зведення):
{summary}

Шукай взаємозв'язки між експериментами (напр. чи кореляція top_k та cost у EXP-02 пояснює вибір моделі в EXP-04? чи знайдений threshold з EXP-03 підтверджується кеш-хіт-рейтом у EXP-06?).

Формат: маркований список з 3-4 пунктів, кожен 2-4 речення.
"""


async def _opus(prompt: str, max_tokens: int = 2048) -> str:
    try:
        await guard.check(projected_usd=PER_OPUS_BUDGET_USD)
    except BudgetExceededError as e:
        write_log(f"[interpret] budget skip Opus: {e}")
        return "(Opus interpretation skipped — budget cap reached.)"
    try:
        msg = await _client.messages.create(
            model=MODEL_OPUS,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except APIStatusError as e:
        write_log(f"[interpret] Opus status error: {e.status_code}")
        return f"(Opus call failed: HTTP {e.status_code} — {str(e)[:160]})"
    except APIError as e:
        write_log(f"[interpret] Opus api error: {type(e).__name__}: {e}")
        return f"(Opus call failed: {type(e).__name__})"


async def interpret(exp_id: str, exp_title: str, hypothesis: str, csv_data: str, key_metrics: str) -> str:
    prompt = INTERPRET_PROMPT.format(
        exp_id=exp_id,
        exp_title=exp_title,
        hypothesis=hypothesis,
        csv_data=csv_data[:4000],
        key_metrics=key_metrics,
    )
    return await _opus(prompt, max_tokens=1800)


async def executive_summary(summary_text: str) -> str:
    return await _opus(EXEC_PROMPT.format(summary=summary_text[:4000]), max_tokens=900)


async def cross_insights(summary_text: str) -> str:
    return await _opus(CROSS_PROMPT.format(summary=summary_text[:4000]), max_tokens=1400)
