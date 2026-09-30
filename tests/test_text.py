import pytest

from voice_tts.services.text import TextValidationError, validate_and_chunk_text


def test_splits_at_korean_sentence_boundaries() -> None:
    result = validate_and_chunk_text("첫 문장입니다. 둘째 문장입니다.", 12)

    assert result == ["첫 문장입니다.", "둘째 문장입니다."]


def test_rejects_blank_and_single_oversized_sentence() -> None:
    with pytest.raises(TextValidationError, match="텍스트를 입력"):
        validate_and_chunk_text("  ", 20)

    with pytest.raises(TextValidationError, match="한 문장이 너무 깁니다"):
        validate_and_chunk_text("가" * 21, 20)
