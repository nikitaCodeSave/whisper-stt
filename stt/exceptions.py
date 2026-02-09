"""Custom exceptions for the STT service."""


class AudioValidationError(Exception):
    """Raised when audio file validation fails."""


class GpuError(Exception):
    """Raised when GPU is unavailable or fails."""


class ModelError(Exception):
    """Raised when model loading or inference fails."""
