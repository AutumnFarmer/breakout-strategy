from __future__ import annotations

from a_breakout_screener.cli import main


def test_cli_doctor_skip_network_imports_cleanly(tmp_path) -> None:
    assert main(["--config", str(tmp_path / "missing.toml"), "doctor", "--skip-network"]) == 0
