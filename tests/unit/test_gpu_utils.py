"""Tests for stt.core.gpu_utils."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from stt.core.gpu_utils import cleanup_gpu_memory, configure_cuda_allocator, log_gpu_memory


class TestLogGpuMemory:
    @patch("stt.core.gpu_utils.torch")
    def test_log_gpu_memory_with_cuda(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 100
        mock_torch.cuda.memory_reserved.return_value = 1024 * 1024 * 200
        mock_torch.cuda.max_memory_allocated.return_value = 1024 * 1024 * 150
        log_gpu_memory("test_label")
        # Should not raise

    @patch("stt.core.gpu_utils.torch")
    def test_log_gpu_memory_no_cuda(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = False
        log_gpu_memory("test_label")
        # Should not raise


class TestCleanupGpuMemory:
    @patch("stt.core.gpu_utils.torch")
    @patch("stt.core.gpu_utils.gc")
    def test_cleanup_calls_gc_and_cache(
        self, mock_gc: MagicMock, mock_torch: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0
        mock_torch.cuda.max_memory_allocated.return_value = 0
        cleanup_gpu_memory("test")
        mock_gc.collect.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("stt.core.gpu_utils.torch")
    @patch("stt.core.gpu_utils.gc")
    def test_cleanup_no_cuda_skips_cache(
        self, mock_gc: MagicMock, mock_torch: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = False
        cleanup_gpu_memory("test")
        mock_gc.collect.assert_called_once()
        mock_torch.cuda.empty_cache.assert_not_called()


class TestConfigureCudaAllocator:
    def test_configure_cuda_allocator_sets_env(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        configure_cuda_allocator()
        val = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        assert "max_split_size_mb:128" in val
        assert "expandable_segments:True" in val

    def test_configure_preserves_existing_env(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(
            "PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6",
        )
        configure_cuda_allocator()
        val = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        assert "garbage_collection_threshold:0.6" in val
        assert "max_split_size_mb" in val
