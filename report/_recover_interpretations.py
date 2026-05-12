"""Extract Opus interpretation text from current HW10_Report.html and save to cache.

Lets us re-render the report without paying for Opus again.
"""
from __future__ import annotations

import json
from pathlib import Path

from bs4 import BeautifulSoup

HTML = Path("report/output/HW10_Report.html")
CACHE = Path("report/output/_interpretations.json")


def _div_to_markdown(div) -> str:
    """Reconstruct markdown-like text from a <div class="interpretation"> by joining inner <p> blocks with blank lines.
    Also collapse `<br>` to `\n` for the executive/cross divs (rendered earlier with replace('\n', '<br>'))."""
    # Strategy: walk children, accumulate text, treat <br> as newline and <p> blocks as paragraph separators.
    lines: list[str] = []
    current: list[str] = []

    def flush():
        if current:
            lines.append(" ".join(current).strip())
            current.clear()

    for el in div.children:
        if getattr(el, "name", None) == "p":
            flush()
            # walk paragraph contents, treating <br> as newlines
            paragraph_lines: list[str] = []
            buf: list[str] = []
            for sub in el.children:
                if getattr(sub, "name", None) == "br":
                    paragraph_lines.append("".join(buf).strip())
                    buf = []
                else:
                    text = sub.get_text() if hasattr(sub, "get_text") else str(sub)
                    buf.append(text)
            if buf:
                paragraph_lines.append("".join(buf).strip())
            lines.append("\n".join(paragraph_lines).strip())
            lines.append("")  # blank line between paragraphs
        elif getattr(el, "name", None) == "br":
            current.append("\n")
        else:
            text = el.get_text() if hasattr(el, "get_text") else str(el)
            current.append(text)
    flush()
    return "\n".join(lines).strip()


def main() -> None:
    html = HTML.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    # Per-experiment interpretations are inside <div class="interpretation"> that
    # appears under each <section> with <h2>EXP-NN — …</h2>
    interpretations: dict[str, str] = {}
    for section in soup.find_all("section"):
        h2 = section.find("h2")
        if not h2:
            continue
        h2_text = h2.get_text(strip=True).upper()
        div = section.find("div", class_="interpretation") or section.find("div", class_="abstract")
        if not div:
            continue
        first = h2_text.split()[0] if h2_text.split() else ""
        if first.startswith("EXP"):
            # h2 like "EXP01 — Chunking..." → first token "EXP01" → "exp01"
            exp_id = first.lower()
            md = _div_to_markdown(div)
            interpretations[exp_id] = md
        elif "EXECUTIVE" in h2_text:
            interpretations["executive"] = _div_to_markdown(div)
        elif "CROSS-EXPERIMENT" in h2_text:
            interpretations["cross"] = _div_to_markdown(div)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(interpretations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(interpretations)} interpretations to {CACHE}")
    for k, v in interpretations.items():
        print(f"  {k}: {len(v)} chars, head={v[:60]!r}")


if __name__ == "__main__":
    main()
