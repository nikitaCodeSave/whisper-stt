"""Sprint 1: Tests for SpeakerHints and validate_speaker_hints()."""

import pytest

from stt.core.speaker_hints import SpeakerHints, validate_speaker_hints


class TestSpeakerHintsDefaults:
    def test_default_num_speakers_is_none(self) -> None:
        hints = SpeakerHints()
        assert hints.num_speakers is None

    def test_default_min_speakers(self) -> None:
        hints = SpeakerHints()
        assert hints.min_speakers == 1

    def test_default_max_speakers(self) -> None:
        hints = SpeakerHints()
        assert hints.max_speakers == 8


class TestSpeakerHintsCustomValues:
    def test_custom_num_speakers(self) -> None:
        hints = SpeakerHints(num_speakers=3)
        assert hints.num_speakers == 3

    def test_custom_min_max(self) -> None:
        hints = SpeakerHints(min_speakers=2, max_speakers=5)
        assert hints.min_speakers == 2
        assert hints.max_speakers == 5


class TestValidateSpeakerHints:
    def test_num_speakers_sets_min_max(self) -> None:
        hints = SpeakerHints(num_speakers=3)
        validated = validate_speaker_hints(hints)
        assert validated.min_speakers == 3
        assert validated.max_speakers == 3
        assert validated.num_speakers == 3

    def test_min_greater_than_max_raises(self) -> None:
        hints = SpeakerHints(min_speakers=5, max_speakers=2)
        with pytest.raises(ValueError):
            validate_speaker_hints(hints)

    def test_defaults_pass_validation(self) -> None:
        hints = SpeakerHints()
        validated = validate_speaker_hints(hints)
        assert validated.min_speakers == 1
        assert validated.max_speakers == 8
        assert validated.num_speakers is None

    def test_min_equals_max_is_valid(self) -> None:
        hints = SpeakerHints(min_speakers=3, max_speakers=3)
        validated = validate_speaker_hints(hints)
        assert validated.min_speakers == 3
        assert validated.max_speakers == 3
