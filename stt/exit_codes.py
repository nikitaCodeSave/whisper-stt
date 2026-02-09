"""Exit codes for the STT CLI."""

from enum import IntEnum


class ExitCode(IntEnum):
    SUCCESS = 0
    ERROR_GENERAL = 1
    ERROR_ARGS = 2
    ERROR_FILE = 3
    ERROR_MODEL = 4
    ERROR_GPU = 5
    PARTIAL_SUCCESS = 10
