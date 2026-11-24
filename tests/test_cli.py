from ml_deploy_ops.cli import main


def test_cli_register_list_eval(tmp_path, capsys):
    state = str(tmp_path / "reg.json")
    assert main(["--state", state, "register", "m", "1.0.0"]) == 0
    assert main(["--state", state, "list"]) == 0
    assert "m:1.0.0" in capsys.readouterr().out
    assert main(["--state", state, "eval", "m", "1.0.0", "--metric", "accuracy=0.95"]) == 0
    assert main(["--state", state, "eval", "m", "1.0.0", "--metric", "accuracy=0.1"]) == 2


def test_cli_bench(capsys):
    assert main(["bench", "--models", "5", "--requests", "20"]) == 0
    assert "requests_simulated" in capsys.readouterr().out


def test_cli_demo(capsys):
    assert main(["demo"]) == 0
    assert "promoted" in capsys.readouterr().out
