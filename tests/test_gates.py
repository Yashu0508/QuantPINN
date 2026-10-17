import pytest
from src.validation.gates import REQUIRED_CHECKS, solver_gate, require_solver_gate

def test_solver_gate_requires_complete_true_booleans():
    report = {"status": "accepted", "checks": dict.fromkeys(REQUIRED_CHECKS, True)}
    assert solver_gate(report)["passed"]
    for invalid in (False, "false", 1, None):
        report["checks"]["delta_mae"] = invalid
        assert not solver_gate(report)["passed"]

def test_partial_checks_cannot_promote_checkpoint():
    assert not solver_gate({"status": "accepted", "checks": {"price": True}})["passed"]
    assert not solver_gate({"status": "accepted", "checks": {}})["passed"]

def test_project_solver_gate_fails_closed():
    with pytest.raises(RuntimeError, match="research-only"):
        require_solver_gate()
