from training_windows.prepare_dataset import (
    PreparedSegment,
    WordTiming,
    build_ffmpeg_command,
    build_review_entry,
    group_word_timings,
)
from training_windows.review_manifest import promote_approved_entries


def test_preparation_builds_24khz_mono_ffmpeg_command_and_keeps_vad_groups_within_limits() -> None:
    command = build_ffmpeg_command("voice.m4a", "processed.wav")
    words = [WordTiming(float(index), float(index + 1), "말") for index in range(24)]

    segments = group_word_timings(words)

    assert command == [
        "ffmpeg", "-y", "-i", "voice.m4a", "-ar", "24000", "-ac", "1", "-sample_fmt", "s16", "processed.wav",
    ]
    assert segments
    assert all(2.0 <= segment.end_seconds - segment.start_seconds <= 12.0 for segment in segments)


def test_review_entry_for_golf_overlap_is_never_auto_approved() -> None:
    entry = build_review_entry(
        PreparedSegment(288.5, 290.5, "원본 대본"),
        expected_text="원본 대본",
        audio_path="clips/001.wav",
    )

    assert entry["reason"] == "excluded_golf_discussion"
    assert entry["auto_approved"] is False
    assert entry["approved"] is False


def test_promotion_uses_only_approved_entries_and_every_fifth_for_validation() -> None:
    entries = [
        {
            "audio": f"clips/{index}.wav",
            "start_seconds": float(index * 10),
            "end_seconds": float(index * 10 + 4),
            "expected_text": f"문장 {index}",
            "approved": True,
            "reason": None,
        }
        for index in range(1, 6)
    ]
    entries.append(
        {
            "audio": "clips/golf.wav",
            "start_seconds": 290.0,
            "end_seconds": 294.0,
            "expected_text": "제외 문장",
            "approved": True,
            "reason": None,
        }
    )

    train_rows, validation_rows = promote_approved_entries(entries, reference_audio="reference.wav")

    assert [row["text"] for row in train_rows] == ["문장 1", "문장 2", "문장 3", "문장 4"]
    assert validation_rows == [{"audio": "clips/5.wav", "text": "문장 5", "ref_audio": "reference.wav"}]
