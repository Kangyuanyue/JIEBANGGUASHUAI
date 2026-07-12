"""Unit tests for competition metrics (no model weights required)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from metrics_cer import (  # noqa: E402
    cer_single,
    is_correct_rejection,
    is_rejection_sample,
)


def test_cer_exact():
    assert cer_single("打开空调", "打开空调") == 0.0


def test_cer_substitution():
    assert cer_single("打开空调", "关闭空调") > 0.0


def test_rejection():
    assert is_rejection_sample("")
    assert is_rejection_sample(None)
    assert is_correct_rejection("", "")


def run_all():
    test_cer_exact()
    test_cer_substitution()
    test_rejection()
    print("metrics tests OK")


if __name__ == "__main__":
    run_all()
