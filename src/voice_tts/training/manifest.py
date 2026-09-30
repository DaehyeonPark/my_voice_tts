"""Rules for deciding which recorded speech may enter a training manifest."""

from __future__ import annotations

from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import unicodedata


AUTO_APPROVE_SIMILARITY = 0.92


@dataclass(frozen=True)
class ExcludedRange:
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if self.start_seconds >= self.end_seconds:
            raise ValueError("제외 구간의 끝은 시작보다 뒤여야 합니다.")


GOLF_EXCLUSION = ExcludedRange(289.0, 311.0)


@dataclass(frozen=True)
class CandidateSegment:
    start_seconds: float
    end_seconds: float
    expected_text: str
    observed_text: str

    def __post_init__(self) -> None:
        if self.start_seconds >= self.end_seconds:
            raise ValueError("발화 구간의 끝은 시작보다 뒤여야 합니다.")


@dataclass(frozen=True)
class ReviewRecord:
    segment: CandidateSegment
    similarity: float
    auto_approved: bool
    approved: bool
    reason: str | None

    def with_approval(self, approved: bool) -> "ReviewRecord":
        if self.reason == "excluded_golf_discussion":
            return replace(self, approved=False)
        return replace(self, approved=approved)


def overlaps_exclusion(segment: CandidateSegment, excluded: ExcludedRange) -> bool:
    """Return whether two half-open time ranges overlap."""

    return segment.start_seconds < excluded.end_seconds and segment.end_seconds > excluded.start_seconds


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def normalized_similarity(expected_text: str, observed_text: str) -> float:
    expected = normalize_text(expected_text)
    observed = normalize_text(observed_text)
    if not expected or not observed:
        return 0.0
    return SequenceMatcher(a=expected, b=observed, autojunk=False).ratio()


def build_review_record(segment: CandidateSegment) -> ReviewRecord:
    if overlaps_exclusion(segment, GOLF_EXCLUSION):
        return ReviewRecord(segment, 0.0, False, False, "excluded_golf_discussion")

    similarity = normalized_similarity(segment.expected_text, segment.observed_text)
    if similarity < AUTO_APPROVE_SIMILARITY:
        return ReviewRecord(segment, similarity, False, False, "transcript_mismatch")
    return ReviewRecord(segment, similarity, True, False, None)


def approved_records_to_jsonl(
    records: list[ReviewRecord], *, audio_path: str, reference_audio_path: str
) -> list[dict[str, str]]:
    """Convert only explicitly approved, non-excluded records to Qwen JSON rows."""

    return [
        {"audio": audio_path, "text": record.segment.expected_text, "ref_audio": reference_audio_path}
        for record in records
        if record.approved and record.reason is None
    ]
