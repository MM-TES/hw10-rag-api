"""Download The Twelve-Factor App methodology → data/source.md.

Fetches the heroku/12factor GitHub repo as zip, unpacks `content/*.md` files,
and concatenates them in section order with headers preserved.
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import httpx

ZIP_URL = "https://github.com/heroku/12factor/archive/refs/heads/master.zip"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "source.md"


def _section_order_key(name: str) -> tuple[int, str]:
    """Sort by leading integer prefix (NN_title.md), then by name."""
    m = re.match(r"^(\d+)", name)
    return (int(m.group(1)) if m else 99, name)


def main() -> int:
    print(f"[download_doc] fetching {ZIP_URL} ...")
    r = httpx.get(ZIP_URL, follow_redirects=True, timeout=60.0)
    r.raise_for_status()
    print(f"[download_doc] got {len(r.content)} bytes")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sections: list[tuple[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        md_names = [
            n for n in zf.namelist()
            if "/content/en/" in n and n.endswith(".md") and not n.endswith("/")
        ]
        md_names.sort(key=lambda n: _section_order_key(Path(n).name))
        for name in md_names:
            with zf.open(name) as f:
                text = f.read().decode("utf-8", errors="replace").strip()
            sections.append((Path(name).name, text))

    if not sections:
        print("[download_doc] ERROR: no content/*.md files found in zip", file=sys.stderr)
        return 1

    parts: list[str] = ["# The Twelve-Factor App\n", "https://12factor.net\n"]
    for fname, body in sections:
        title = re.sub(r"^\d+_", "", Path(fname).stem).replace("_", " ").title()
        parts.append(f"\n\n## {title}\n\n{body}\n")
    OUT_PATH.write_text("".join(parts), encoding="utf-8")

    print(f"[download_doc] wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes, {len(sections)} sections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
