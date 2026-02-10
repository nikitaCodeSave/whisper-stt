"""Tests for SttConfig, load_config(), resolve_config(), build_pipeline_config()."""

from __future__ import annotations

from pathlib import Path

import pytest

from stt.config import SttConfig, build_pipeline_config, load_config, resolve_config


class TestSttConfigDefaults:
    def test_default_model(self) -> None:
        cfg = SttConfig()
        assert cfg.model == "large-v3"

    def test_default_language(self) -> None:
        cfg = SttConfig()
        assert cfg.language == "ru"

    def test_default_format(self) -> None:
        cfg = SttConfig()
        assert cfg.format == "json"

    def test_default_device(self) -> None:
        cfg = SttConfig()
        assert cfg.device == "cuda"

    def test_default_compute_type(self) -> None:
        cfg = SttConfig()
        assert cfg.compute_type == "float16"

    def test_default_diarization_enabled(self) -> None:
        cfg = SttConfig()
        assert cfg.diarization_enabled is True

    def test_default_max_speakers(self) -> None:
        cfg = SttConfig()
        assert cfg.max_speakers == 8

    def test_default_output_dir(self) -> None:
        cfg = SttConfig()
        assert cfg.output_dir == "."

    def test_default_model_dir(self) -> None:
        cfg = SttConfig()
        assert cfg.model_dir == "models"


class TestLoadConfigFromYaml:
    def test_load_config_from_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "model: small\n"
            "language: en\n"
            "format: txt\n"
            "device: cpu\n"
            "compute_type: int8\n"
            "diarization:\n"
            "  enabled: false\n"
            "  max_speakers: 4\n"
            "output_dir: /tmp/out\n"
            "model_dir: /tmp/models\n"
        )
        cfg = load_config(config_file)
        assert cfg.model == "small"
        assert cfg.language == "en"
        assert cfg.format == "txt"
        assert cfg.device == "cpu"
        assert cfg.compute_type == "int8"
        assert cfg.diarization_enabled is False
        assert cfg.max_speakers == 4
        assert cfg.output_dir == "/tmp/out"
        assert cfg.model_dir == "/tmp/models"

    def test_load_config_partial_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: medium\nlanguage: en\n")
        cfg = load_config(config_file)
        assert cfg.model == "medium"
        assert cfg.language == "en"
        assert cfg.format == "json"  # default
        assert cfg.device == "cuda"  # default

    def test_load_config_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        cfg = load_config(missing)
        assert cfg.model == "large-v3"
        assert cfg.language == "ru"

    def test_load_config_from_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_file = tmp_path / "env_config.yaml"
        config_file.write_text("model: tiny\nlanguage: en\n")
        monkeypatch.setenv("STT_CONFIG", str(config_file))
        cfg = load_config()
        assert cfg.model == "tiny"
        assert cfg.language == "en"


class TestLoadConfigCwdFallback:
    def test_cwd_config_yaml_used_as_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("STT_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: base\nlanguage: de\n")
        cfg = load_config()
        assert cfg.model == "base"
        assert cfg.language == "de"

    def test_env_var_takes_priority_over_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cwd_config = tmp_path / "config.yaml"
        cwd_config.write_text("model: base\n")
        env_config = tmp_path / "env.yaml"
        env_config.write_text("model: tiny\n")
        monkeypatch.setenv("STT_CONFIG", str(env_config))
        cfg = load_config()
        assert cfg.model == "tiny"

    def test_no_cwd_config_returns_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("STT_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.model == "large-v3"
        assert cfg.model_dir == "models"


class TestLoadConfigModelDirEnv:
    def test_stt_model_dir_env_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("STT_CONFIG", raising=False)
        monkeypatch.setenv("STT_MODEL_DIR", "/opt/models")
        cfg = load_config()
        assert cfg.model_dir == "/opt/models"

    def test_stt_model_dir_env_overrides_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_dir: /yaml/models\n")
        monkeypatch.setenv("STT_MODEL_DIR", "/env/models")
        cfg = load_config(config_file)
        assert cfg.model_dir == "/env/models"

    def test_no_stt_model_dir_env_keeps_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("STT_MODEL_DIR", raising=False)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_dir: /yaml/models\n")
        cfg = load_config(config_file)
        assert cfg.model_dir == "/yaml/models"


class TestSttConfigMinSpeakers:
    def test_default_min_speakers(self) -> None:
        cfg = SttConfig()
        assert cfg.min_speakers == 1

    def test_min_speakers_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "diarization:\n"
            "  min_speakers: 2\n"
            "  max_speakers: 5\n"
        )
        cfg = load_config(config_file)
        assert cfg.min_speakers == 2
        assert cfg.max_speakers == 5


class TestSttConfigWithOverrides:
    def test_with_overrides_returns_new_config(self) -> None:
        cfg = SttConfig()
        new_cfg = cfg.with_overrides(model="small")
        assert new_cfg is not cfg
        assert new_cfg.model == "small"
        assert cfg.model == "large-v3"  # original unchanged

    def test_with_overrides_multiple_fields(self) -> None:
        cfg = SttConfig()
        new_cfg = cfg.with_overrides(model="medium", language="en", device="cpu")
        assert new_cfg.model == "medium"
        assert new_cfg.language == "en"
        assert new_cfg.device == "cpu"
        assert new_cfg.format == "json"  # unchanged default

    def test_with_overrides_preserves_non_overridden(self) -> None:
        cfg = SttConfig()
        new_cfg = cfg.with_overrides(model="small")
        assert new_cfg.language == "ru"
        assert new_cfg.format == "json"
        assert new_cfg.device == "cuda"
        assert new_cfg.compute_type == "float16"
        assert new_cfg.diarization_enabled is True
        assert new_cfg.max_speakers == 8


class TestSttConfigHfToken:
    def test_default_hf_token_is_none(self) -> None:
        cfg = SttConfig()
        assert cfg.hf_token is None

    def test_hf_token_from_env(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("STT_CONFIG", raising=False)
        monkeypatch.setenv("HF_TOKEN", "hf_test_abc")
        cfg = load_config()
        assert cfg.hf_token == "hf_test_abc"

    def test_no_hf_token_env_keeps_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("STT_CONFIG", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        cfg = load_config()
        assert cfg.hf_token is None


class TestResolveConfig:
    def test_cli_overrides_config(self) -> None:
        cfg = SttConfig(model="large-v3", language="ru")
        resolved = resolve_config(cfg, model="small", language="en")
        assert resolved.model == "small"
        assert resolved.language == "en"

    def test_none_does_not_override(self) -> None:
        cfg = SttConfig(model="medium", language="de")
        resolved = resolve_config(cfg, model=None, language=None)
        assert resolved.model == "medium"
        assert resolved.language == "de"

    def test_no_diarize_disables_diarization(self) -> None:
        cfg = SttConfig(diarization_enabled=True)
        resolved = resolve_config(cfg, no_diarize=True)
        assert resolved.diarization_enabled is False

    def test_no_diarize_false_keeps_enabled(self) -> None:
        cfg = SttConfig(diarization_enabled=True)
        resolved = resolve_config(cfg, no_diarize=False)
        assert resolved.diarization_enabled is True

    def test_all_fields_override(self) -> None:
        cfg = SttConfig()
        resolved = resolve_config(
            cfg,
            model="tiny",
            language="en",
            format="srt",
            output_dir="/out",
            device="cpu",
            compute_type="int8",
            model_dir="/models",
            min_speakers=2,
            max_speakers=4,
        )
        assert resolved.model == "tiny"
        assert resolved.language == "en"
        assert resolved.format == "srt"
        assert resolved.output_dir == "/out"
        assert resolved.device == "cpu"
        assert resolved.compute_type == "int8"
        assert resolved.model_dir == "/models"
        assert resolved.min_speakers == 2
        assert resolved.max_speakers == 4

    def test_preserves_hf_token(self) -> None:
        cfg = SttConfig(hf_token="hf_abc")
        resolved = resolve_config(cfg, model="small")
        assert resolved.hf_token == "hf_abc"

    def test_returns_same_object_when_no_overrides(self) -> None:
        cfg = SttConfig()
        resolved = resolve_config(cfg)
        assert resolved is cfg


class TestBuildPipelineConfig:
    def test_maps_fields_correctly(self) -> None:
        cfg = SttConfig(
            model="small",
            device="cpu",
            compute_type="int8",
            language="en",
            diarization_enabled=False,
            min_speakers=2,
            max_speakers=5,
            format="txt",
            output_dir="/output",
            model_dir="/models",
            hf_token="hf_test",
        )
        pc = build_pipeline_config(cfg, num_speakers=3)
        assert pc.model_size == "small"
        assert pc.device == "cpu"
        assert pc.compute_type == "int8"
        assert pc.language == "en"
        assert pc.diarization_enabled is False
        assert pc.num_speakers == 3
        assert pc.min_speakers == 2
        assert pc.max_speakers == 5
        assert pc.formats == "txt"
        assert pc.output_dir == "/output"
        assert pc.model_dir == "/models"
        assert pc.hf_token == "hf_test"

    def test_num_speakers_default_none(self) -> None:
        cfg = SttConfig()
        pc = build_pipeline_config(cfg)
        assert pc.num_speakers is None

    def test_hf_token_none_by_default(self) -> None:
        cfg = SttConfig()
        pc = build_pipeline_config(cfg)
        assert pc.hf_token is None


class TestSttConfigBatchParams:
    def test_default_batch_size_is_8(self) -> None:
        cfg = SttConfig()
        assert cfg.batch_size == 8

    def test_default_vad_filter(self) -> None:
        cfg = SttConfig()
        assert cfg.vad_filter is True

    def test_default_condition_on_previous_text(self) -> None:
        cfg = SttConfig()
        assert cfg.condition_on_previous_text is False

    def test_default_hallucination_silence_threshold(self) -> None:
        cfg = SttConfig()
        assert cfg.hallucination_silence_threshold == 2.0

    def test_default_use_subprocess(self) -> None:
        cfg = SttConfig()
        assert cfg.use_subprocess is False

    def test_default_use_batched_is_false(self) -> None:
        cfg = SttConfig()
        assert cfg.use_batched is False

    def test_yaml_whisper_section_loaded(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "whisper:\n"
            "  batch_size: 16\n"
            "  vad_filter: false\n"
            "  condition_on_previous_text: true\n"
            "  hallucination_silence_threshold: 3.0\n"
        )
        cfg = load_config(config_file)
        assert cfg.batch_size == 16
        assert cfg.vad_filter is False
        assert cfg.condition_on_previous_text is True
        assert cfg.hallucination_silence_threshold == 3.0

    def test_build_pipeline_config_threads_batch_params(self) -> None:
        cfg = SttConfig(batch_size=16, vad_filter=False, hallucination_silence_threshold=3.0)
        pc = build_pipeline_config(cfg)
        assert pc.batch_size == 16
        assert pc.vad_filter is False
        assert pc.hallucination_silence_threshold == 3.0

    def test_yaml_whisper_use_batched_loaded(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "whisper:\n"
            "  use_batched: true\n"
        )
        cfg = load_config(config_file)
        assert cfg.use_batched is True

    def test_build_pipeline_config_threads_use_batched(self) -> None:
        cfg = SttConfig(use_batched=True)
        pc = build_pipeline_config(cfg)
        assert pc.use_batched is True
