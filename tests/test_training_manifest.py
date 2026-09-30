from voice_tts.training.manifest import (
    AUTO_APPROVE_SIMILARITY,
    GOLF_EXCLUSION,
    CandidateSegment,
    ExcludedRange,
    approved_records_to_jsonl,
    build_review_record,
    normalized_similarity,
    overlaps_exclusion,
)


def test_exclusion_rejects_segments_that_touch_either_golf_boundary() -> None:
    assert GOLF_EXCLUSION == ExcludedRange(289.0, 311.0)
    assert overlaps_exclusion(CandidateSegment(288.5, 290.5, "대본", "대본"), GOLF_EXCLUSION)
    assert overlaps_exclusion(CandidateSegment(310.5, 312.0, "대본", "대본"), GOLF_EXCLUSION)


def test_low_similarity_segment_is_held_for_human_review() -> None:
    segment = CandidateSegment(100.0, 104.0, "오늘 날씨가 좋습니다.", "골프 이야기를 하겠습니다.")

    review = build_review_record(segment)

    assert normalized_similarity(segment.expected_text, segment.observed_text) < AUTO_APPROVE_SIMILARITY
    assert review.auto_approved is False
    assert review.approved is False
    assert review.reason == "transcript_mismatch"


def test_only_approved_records_become_qwen_training_rows() -> None:
    accepted = build_review_record(CandidateSegment(12.0, 16.0, "안녕하세요.", "안녕하세요."))
    accepted = accepted.with_approval(True)
    rejected = build_review_record(CandidateSegment(290.0, 294.0, "제외합니다.", "제외합니다."))

    rows = approved_records_to_jsonl(
        [accepted, rejected],
        audio_path="clips/0001.wav",
        reference_audio_path="reference.wav",
    )

    assert rows == [{"audio": "clips/0001.wav", "text": "안녕하세요.", "ref_audio": "reference.wav"}]
