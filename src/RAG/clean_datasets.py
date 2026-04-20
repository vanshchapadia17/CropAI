"""One-shot cleaner for rice_dataset/ and sugarcane_dataset/ scraped txt files.

The scraper dumped soup.get_text(separator='\n') which leaves:
- Boilerplate navigation repeated in every file
- Sentences fragmented across lines (species names, dates, etc.)
- "Top of page" jump-links interleaved with content

This script merges continuation lines into coherent paragraphs and strips
navigation, producing text that chunks cleanly for the FAISS index.

Run once: python -m src.RAG.clean_datasets
Then delete artifacts/rag_index.faiss and artifacts/rag_docs.pkl so the
RAG pipeline rebuilds the index on the next /chat call.
"""

from pathlib import Path
from typing import List, Tuple

RAG_DIR = Path(__file__).parent

# Exact-match navigation lines to drop (case-sensitive — scrape is stable).
BOILERPLATE = {
    "Expert System for Sugarcane",
    "Expert System For Sugarcane",
    "Expert System for Paddy",
    "Expert System For Paddy",
    "Home",
    "|",
    "About Us",
    "Contact Us",
    "Categories",
    "Botany & Climate",
    "Season & Varieties",
    "Nursery Management",
    "Cultivation Practices",
    "Sustainable Sugarcane Initiative",
    "Sustainable Rice Initiative",
    "Irrigation Management",
    "Nutrient Management",
    "Crop Protection",
    "Farm Implements",
    "Post Harvest Technology",
    "Marketing",
    "Institutions & Schemes",
    "FAQ's",
    "FAQs",
    "Related Links",
    "Contract All",
    "Expand All",
    "Back to Top",
}

SECTION_BREAKS = {"Top of page", "TOP", "Back"}
SENTENCE_END = (".", "?", "!", ":", ";", ")", "\"", "'")


def _is_paragraph_end(line: str) -> bool:
    """A line ends a paragraph if it ends with sentence-ending punctuation."""
    stripped = line.rstrip()
    return bool(stripped) and stripped.endswith(SENTENCE_END)


def clean_file(path: Path) -> Tuple[int, int]:
    """Return (lines_before, lines_after) for reporting."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()
    lines_before = len(lines)

    # Preserve the Source: header if present
    source_line = None
    if lines and lines[0].startswith("Source:"):
        source_line = lines[0]
        lines = lines[1:]

    # 1) Strip boilerplate + collapse blanks
    filtered: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if filtered and filtered[-1] != "":
                filtered.append("")
            continue
        if stripped in BOILERPLATE:
            continue
        filtered.append(stripped)

    # 2) Merge continuation lines into paragraphs.
    # A new paragraph starts after a sentence-ending line, a blank line,
    # or a section-break marker like "Top of page".
    paragraphs: List[str] = []
    buffer = ""

    def flush():
        nonlocal buffer
        if buffer.strip():
            paragraphs.append(buffer.strip())
        buffer = ""

    for line in filtered:
        if not line:
            flush()
            continue
        if line in SECTION_BREAKS:
            flush()
            continue
        if not buffer:
            buffer = line
        elif _is_paragraph_end(buffer):
            flush()
            buffer = line
        else:
            # Continuation — join with a space so split species/terms re-unite
            buffer = f"{buffer} {line}"
    flush()

    # 3) Rebuild file: Source on first line, blank, then paragraphs separated
    # by blank lines (chunks nicely under RecursiveCharacterTextSplitter).
    out: List[str] = []
    if source_line:
        out.extend([source_line, ""])
    for para in paragraphs:
        out.append(para)
        out.append("")
    while out and out[-1] == "":
        out.pop()

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return lines_before, len(out)


def main() -> None:
    total_before = 0
    total_after = 0
    for folder in ("rice_dataset", "sugarcane_dataset"):
        target = RAG_DIR / folder
        if not target.exists():
            print(f"[skip] {target} does not exist")
            continue
        for txt in sorted(target.glob("*.txt")):
            before, after = clean_file(txt)
            total_before += before
            total_after += after
            print(f"{folder}/{txt.name}: {before} ->{after} lines")
    print(f"\nTotal: {total_before} ->{total_after} lines")


if __name__ == "__main__":
    main()
