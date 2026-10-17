"""Require complete boolean checks and checkpoint identity for solver readiness."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_CHECKS = frozenset({
    "price_over_strike_mae", "price_over_strike_max_error", "delta_mae",
    "strike_scaled_gamma_mae", "equivalent_horizon_price_over_strike_max_error",
})

def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_solver_report(path=None):
    report = json.loads(Path(path or ROOT / "reports/solver_validation.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("checks"), dict):
        raise ValueError("solver report and checks must be objects")
    return report

def solver_gate(report):
    checks = report.get("checks", {})
    passed = (
        isinstance(checks, dict) and REQUIRED_CHECKS.issubset(checks)
        and all(value is True for value in checks.values())
        and report.get("status") == "accepted"
    )
    return {"passed": passed, "status": "accepted" if passed else "failed",
            "checks": checks, "interpretation": str(report.get("interpretation", ""))}

def require_solver_gate(path=None, checkpoint=None):
    report = load_solver_report(path)
    summary = solver_gate(report)
    if not summary["passed"]:
        raise RuntimeError("analytical solver gate has not passed; model remains research-only")
    if checkpoint is not None and report.get("checkpoint_sha256") != checkpoint_sha256(checkpoint):
        raise RuntimeError("solver report does not match checkpoint; model remains research-only")
    return summary
