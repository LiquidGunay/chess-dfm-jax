import ast
from pathlib import Path


def test_trusted_mounts_are_bound_before_train_torch_imports():
    source = Path("research/infra/modal_hh002_audit.py").read_text()
    tree = ast.parse(source)
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    environment = next(node.lineno for node in ast.walk(run)
                       if isinstance(node, ast.Call) and ast.unparse(node.func) == "os.environ.update")
    imports = [node.lineno for node in ast.walk(run) if isinstance(node, ast.ImportFrom)
               and node.module in {"research.train_torch", "research.interpretability.models",
                                   "research.interpretability.parameter_diff"}]
    assert environment < min(imports)
    assert "resolve(strict=True)" in source
