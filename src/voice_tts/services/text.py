"""Text validation and sentence-aware chunking."""

import re


class TextValidationError(ValueError):
    """Raised when synthesis text cannot be safely processed."""


_SENTENCE_ENDINGS = frozenset(".!?。！？")


def validate_and_chunk_text(text: str, max_chars: int) -> list[str]:
    """Return normalized, whole-sentence chunks no longer than max_chars."""
    normalized = " ".join(text.split())
    if not normalized:
        raise TextValidationError("텍스트를 입력해 주세요.")
    if max_chars < 1:
        raise ValueError("최대 문장 길이는 1자 이상이어야 합니다.")

    sentences = _split_sentences(normalized)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            raise TextValidationError("한 문장이 너무 깁니다. 문장을 나눠 주세요.")
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_sentences(text: str) -> list[str]:
    parts = re.findall(r"[^.!?。！？]+[.!?。！？]*", text)
    return [part.strip() for part in parts if part.strip()]
