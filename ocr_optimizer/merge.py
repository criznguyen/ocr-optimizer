"""Smart merge algorithm for multi-pass OCR results."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .ocr_client import OCRResult


@dataclass
class MergedResult:
    text: str
    confidence: float
    lines: list[MergedLine] = field(default_factory=list)
    source_results: list[OCRResult] = field(default_factory=list)


@dataclass
class MergedLine:
    text: str
    confidence: float
    source: str  # which pass/region produced this line
    alternatives: list[str] = field(default_factory=list)


def clean_text(text: str) -> str:
    """Remove non-OCR artifacts from model output."""
    text = re.sub(r"</?doc[^>]*>", "", text)
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("The image shows", "This image", "I can see")):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _line_similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two lines."""
    if not a.strip() or not b.strip():
        return 0.0
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()


def _find_best_match(line: str, candidates: list[str], threshold: float) -> tuple[int, float]:
    """Find the best matching line index in candidates. Returns (index, similarity)."""
    best_idx = -1
    best_sim = 0.0
    for i, cand in enumerate(candidates):
        sim = _line_similarity(line, cand)
        if sim > best_sim and sim >= threshold:
            best_sim = sim
            best_idx = i
    return best_idx, best_sim


def _pick_best_line(candidates: list[tuple[str, str]]) -> tuple[str, float, str]:
    """Pick the best version of a line from multiple candidates.

    candidates: list of (text, source_name)
    Returns: (best_text, confidence, source_name)
    """
    non_empty = [(t, s) for t, s in candidates if t.strip()]
    if not non_empty:
        return "", 0.0, ""
    if len(non_empty) == 1:
        return non_empty[0][0], 0.5, non_empty[0][1]

    # Count how many candidates agree (high similarity)
    texts = [t for t, _ in non_empty]
    # Use the longest as default pick (more detail)
    longest_idx = max(range(len(texts)), key=lambda i: len(texts[i].strip()))
    best = non_empty[longest_idx]

    # Calculate confidence based on agreement
    agreement_count = 0
    for i, t in enumerate(texts):
        if i == longest_idx:
            continue
        sim = _line_similarity(t, best[0])
        if sim > 0.8:
            agreement_count += 1

    confidence = (agreement_count + 1) / len(non_empty)
    return best[0], confidence, best[1]


def merge_results(
    results: list[OCRResult],
    similarity_threshold: float = 0.6,
    min_line_length: int = 10,
) -> MergedResult:
    """Merge multiple OCR results using line alignment and voting.

    Strategy:
    1. Use the full-image result as primary reference
    2. Align crop results against reference
    3. For each line, pick the best version from all sources
    4. Recover lines that crops captured but full-image missed
    """
    if not results:
        return MergedResult(text="", confidence=0.0)

    # Clean all outputs
    cleaned = [(clean_text(r.text), r) for r in results if r.text.strip()]
    if not cleaned:
        return MergedResult(text="", confidence=0.0)

    if len(cleaned) == 1:
        text = cleaned[0][0]
        return MergedResult(
            text=text,
            confidence=1.0,
            source_results=results,
            lines=[MergedLine(text=l, confidence=1.0, source=cleaned[0][1].region_name)
                   for l in text.split("\n")],
        )

    # Find the full-image result (primary reference)
    # Heuristic: it's usually the first result, or the longest one
    full_results = [(t, r) for t, r in cleaned if "full" in r.region_name or not r.region_name]
    crop_results = [(t, r) for t, r in cleaned if r.region_name and "full" not in r.region_name]

    if full_results:
        ref_text, ref_result = max(full_results, key=lambda x: len(x[0]))
    else:
        ref_text, ref_result = max(cleaned, key=lambda x: len(x[0]))

    ref_lines = ref_text.split("\n")

    # Collect all crop lines with their source names
    all_crop_lines: list[tuple[str, str]] = []  # (line_text, source_name)
    for text, r in crop_results:
        for line in text.split("\n"):
            all_crop_lines.append((line, r.region_name))

    # For additional full passes (multi-pass same image)
    other_full_lines: list[tuple[str, str]] = []
    for text, r in full_results:
        if r is not ref_result:
            for line in text.split("\n"):
                other_full_lines.append((line, r.region_name))

    # Phase 1: For each reference line, find best match from all sources
    merged_lines: list[MergedLine] = []
    used_crop_indices: set[int] = set()

    for ref_line in ref_lines:
        if not ref_line.strip():
            merged_lines.append(MergedLine(text=ref_line, confidence=1.0, source=ref_result.region_name))
            continue

        candidates: list[tuple[str, str]] = [(ref_line, ref_result.region_name)]

        # Find matching lines from other full passes
        for line, src in other_full_lines:
            sim = _line_similarity(ref_line, line)
            if sim >= similarity_threshold:
                candidates.append((line, src))

        # Find matching lines from crops
        for ci, (cline, csrc) in enumerate(all_crop_lines):
            if ci in used_crop_indices:
                continue
            sim = _line_similarity(ref_line, cline)
            if sim >= similarity_threshold:
                candidates.append((cline, csrc))
                used_crop_indices.add(ci)

        best_text, confidence, source = _pick_best_line(candidates)
        alts = [t for t, _ in candidates if t != best_text]
        merged_lines.append(MergedLine(
            text=best_text, confidence=confidence, source=source, alternatives=alts,
        ))

    # Phase 2: Recover lines from crops that were missed in full-image
    merged_texts = [ml.text for ml in merged_lines]
    recovered: list[MergedLine] = []

    for ci, (cline, csrc) in enumerate(all_crop_lines):
        if ci in used_crop_indices:
            continue
        cline_stripped = cline.strip()
        if len(cline_stripped) < min_line_length:
            continue

        # Check if this line already exists in merged output (fuzzy match or substring)
        _, best_sim = _find_best_match(cline, merged_texts, 0.5)
        if best_sim >= 0.5:
            continue
        # Also skip if it's a substring of any merged line (partial crop artifact)
        is_substring = any(cline_stripped in m for m in merged_texts if m.strip())
        if is_substring:
            continue

        # This is a genuinely new line — recover it
        recovered.append(MergedLine(
            text=cline, confidence=0.3, source=f"{csrc} (recovered)",
        ))

    # Insert recovered lines at appropriate positions
    # Try to find insertion point based on neighboring lines in the crop
    if recovered:
        merged_lines.extend(recovered)

    # Build final text
    final_text = "\n".join(ml.text for ml in merged_lines)
    avg_confidence = sum(ml.confidence for ml in merged_lines) / len(merged_lines) if merged_lines else 0

    return MergedResult(
        text=final_text,
        confidence=avg_confidence,
        lines=merged_lines,
        source_results=results,
    )
