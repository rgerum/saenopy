"""Execute the real script-runner branch without constructing the Qt GUI."""
import ast
from pathlib import Path
import sys

import pytest


def test_script_can_assign_sys_and_exit_successfully(tmp_path, monkeypatch):
    launcher = Path(__file__).parents[1] / "saenopy/gui/gui_master.py"
    tree = ast.parse(launcher.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.If)
                and "__name__" in ast.unparse(node.test))
    runner = next(node for node in main.body if isinstance(node, ast.If)
                  and "sys.argv[1]" in ast.unparse(node.test))
    script = tmp_path / "analysis.py"
    script.write_text("result = 42\nsys = 1\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["saenopy.exe", "run", str(script)])
    namespace = {"sys": sys}
    with pytest.raises(SystemExit) as stopped:
        exec(compile(ast.Module(body=[runner], type_ignores=[]), str(launcher), "exec"), namespace)
    assert namespace["result"] == 42
    assert stopped.value.code == 0
