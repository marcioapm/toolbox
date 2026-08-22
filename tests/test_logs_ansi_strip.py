"""Tests for toolbox.agent_run._strip_ansi_bytes / _LOGS_CSI_RE (`agent-run
logs --plain`'s ANSI stripper)."""
from __future__ import annotations

from toolbox.agent_run import _strip_ansi_bytes


class TestLogsCsiPrivateParameterBytes:
    """ECMA-48 5.4c reserves 0x3C-0x3F (``<=>?``) as CSI private-parameter
    bytes. _LOGS_CSI_RE must match all four, not just ``?`` -- a narrower
    class lets the terminator spill through as literal text: ``\\x1b[>4;1m``
    (xterm's modifyOtherKeys query reply) previously left ``>4;1m`` in
    plain-text output."""

    def test_greater_than_parameter_byte(self):
        assert _strip_ansi_bytes(b"\x1b[>4;1m") == b""

    def test_equals_parameter_byte(self):
        assert _strip_ansi_bytes(b"\x1b[=3h") == b""

    def test_less_than_parameter_byte(self):
        assert _strip_ansi_bytes(b"\x1b[<0c") == b""

    def test_question_mark_parameter_byte_still_matches(self):
        assert _strip_ansi_bytes(b"\x1b[?25l") == b""

    def test_surrounding_text_survives(self):
        assert _strip_ansi_bytes(b"before\x1b[>4;1mafter") == b"beforeafter"
