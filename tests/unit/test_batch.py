"""Tests for stt.core.batch discover_audio_files — Sprint 6 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from stt.core.batch import discover_audio_files


class TestDiscoverFlat:
    def test_finds_audio_files_in_flat_dir(self, tmp_path: Path) -> None:
        (tmp_path / "a.mp3").write_bytes(b"\x00" * 10)
        (tmp_path / "b.wav").write_bytes(b"\x00" * 10)
        (tmp_path / "c.txt").write_text("not audio")

        files = discover_audio_files(tmp_path)
        names = {f.name for f in files}
        assert "a.mp3" in names
        assert "b.wav" in names
        assert "c.txt" not in names

    def test_finds_all_supported_extensions(
        self, tmp_path: Path,
    ) -> None:
        for ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"):
            (tmp_path / f"test{ext}").write_bytes(b"\x00" * 10)

        files = discover_audio_files(tmp_path)
        assert len(files) == 6


class TestDiscoverRecursive:
    def test_finds_files_in_subdirs(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "a.mp3").write_bytes(b"\x00" * 10)
        (sub / "b.wav").write_bytes(b"\x00" * 10)

        files = discover_audio_files(tmp_path, recursive=True)
        names = {f.name for f in files}
        assert "a.mp3" in names
        assert "b.wav" in names

    def test_non_recursive_skips_subdirs(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "a.mp3").write_bytes(b"\x00" * 10)
        (sub / "b.wav").write_bytes(b"\x00" * 10)

        files = discover_audio_files(tmp_path, recursive=False)
        names = {f.name for f in files}
        assert "a.mp3" in names
        assert "b.wav" not in names


class TestDiscoverPattern:
    def test_pattern_filters_by_extension(self, tmp_path: Path) -> None:
        (tmp_path / "a.mp3").write_bytes(b"\x00" * 10)
        (tmp_path / "b.wav").write_bytes(b"\x00" * 10)

        files = discover_audio_files(tmp_path, pattern="*.mp3")
        assert len(files) == 1
        assert files[0].name == "a.mp3"


class TestDiscoverEmpty:
    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        files = discover_audio_files(tmp_path)
        assert files == []


class TestDiscoverNonexistent:
    def test_nonexistent_dir_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            discover_audio_files(Path("/nonexistent/dir"))
