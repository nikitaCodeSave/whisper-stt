"""Sprint 1: Tests for ExitCode IntEnum."""

from enum import IntEnum

from stt.exit_codes import ExitCode


class TestExitCodeIsIntEnum:
    def test_exit_code_is_intenum(self) -> None:
        assert issubclass(ExitCode, IntEnum)


class TestExitCodeValues:
    def test_success(self) -> None:
        assert ExitCode.SUCCESS == 0

    def test_error_general(self) -> None:
        assert ExitCode.ERROR_GENERAL == 1

    def test_error_args(self) -> None:
        assert ExitCode.ERROR_ARGS == 2

    def test_error_file(self) -> None:
        assert ExitCode.ERROR_FILE == 3

    def test_error_model(self) -> None:
        assert ExitCode.ERROR_MODEL == 4

    def test_error_gpu(self) -> None:
        assert ExitCode.ERROR_GPU == 5

    def test_partial_success(self) -> None:
        assert ExitCode.PARTIAL_SUCCESS == 10


class TestExitCodeUsableAsInt:
    def test_can_use_as_int(self) -> None:
        assert int(ExitCode.SUCCESS) == 0
        assert int(ExitCode.ERROR_GENERAL) == 1

    def test_all_members_count(self) -> None:
        assert len(ExitCode) == 8

    def test_error_oom_value(self) -> None:
        assert ExitCode.ERROR_OOM == 6
