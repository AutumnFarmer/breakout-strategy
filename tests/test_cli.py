from __future__ import annotations

import pytest

from a_breakout_screener.cli import main


def test_cli_doctor_skip_network_imports_cleanly(tmp_path) -> None:
    assert main(["--config", str(tmp_path / "missing.toml"), "doctor", "--skip-network"]) == 0


def test_backtest_first_signal_help_shows_executable_options(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["backtest-first-signal", "--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--lot-size" in output
    assert "--max-capital-per-trade" in output
    assert "--max-buys-per-day" in output
    assert "--slippage-bps" in output
    assert "--fee-bps" in output
