"""Extract Opus interpretation text from current HW10_Report.html and save to cache.

Lets us re-render the report without paying for Opus again.
"""
from __future__ import annotations

import json
from pathlib import Path

from bs4 import BeautifulSoup

HTML = Path("report/output/HW10_Report.html")
CACHE = Path("report/output/_interpretations.json")


HEADER_PREFIX = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### "}


def _ul_to_md(ul) -> str:
    items = []
    for li in ul.find_all("li", recursive=False):
        text = li.get_text(" ", strip=True)
        items.append(f"- {text}")
    return "\n".join(items)


def _div_to_markdown(div) -> str:
    """Reconstruct markdown-like text from an interpretation block.

    Walks block-level children (h1..h4, p, ul, ol, pre, br) and emits a
    markdown-ish string suitable for re-rendering via python-markdown.
    """
    blocks: list[str] = []
    for el in div.children:
        name = getattr(el, "name", None)
        if name in HEADER_PREFIX:
            blocks.append(HEADER_PREFIX[name] + el.get_text(" ", strip=True))
        elif name == "p":
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
            blocks.append("\n".join(paragraph_lines).strip())
        elif name in ("ul", "ol"):
            blocks.append(_ul_to_md(el))
        elif name in ("pre", "code"):
            txt = el.get_text("\n", strip=True)
            blocks.append("```\n" + txt + "\n```")
        elif name is None:
            # bare text or whitespace
            s = str(el).strip()
            if s:
                blocks.append(s)
    return "\n\n".join(b for b in blocks if b).strip()


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
