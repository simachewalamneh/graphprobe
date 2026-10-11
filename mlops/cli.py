"""Command line for the whole pipeline.  Examples:
  python -m mlops.cli snapshot                 # export Cora snapshot (needs torch_geometric once)
  python -m mlops.cli experiments              # >=3 tracked MLflow runs -> register -> gated promotion
  python -m mlops.cli gate-demo                # first model / improvement / blocked regression / tie / rollback
  python -m mlops.cli gate-check               # CI: candidate vs approved, exit 1 if the gate fails
  python -m mlops.cli status | promote v2 | rollback
  python -m mlops.cli unseen-demo | drift-demo | consistency-demo | failure-demo
  python -m mlops.cli serve                    # FastAPI on :8000"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from mlops.bundle import load_bundle
from mlops.data import SNAPSHOT, build_cora_snapshot, load_snapshot, make_training_graph, validation_set
from mlops.registry import GateConfig, Registry
from mlops.train import tracked_run, train_bundle

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "mlops" / "registry"
RESULTS = ROOT / "results" / "mlops"

EXPERIMENTS = [  # (run name, params)
    ("gcn_h16_baseline", dict(hidden=16, lr=0.01, dropout=0.5, wd=5e-4, epochs=200, seed=0)),
    ("gcn_h64_wide", dict(hidden=64, lr=0.01, dropout=0.5, wd=5e-4, epochs=200, seed=0)),
    ("gcn_h32_lowlr_highdrop", dict(hidden=32, lr=0.005, dropout=0.8, wd=5e-4, epochs=200, seed=0)),
]
CI_PARAMS = dict(hidden=64, lr=0.01, dropout=0.5, wd=5e-4, epochs=200, seed=0)


def _snap():
    return load_snapshot("cora")


def _val(snap):
    return validation_set(make_training_graph(snap, 150, 0))


def _print_decision(name, d):
    print(f"  {name}: {'PROMOTED' if d.allowed else 'BLOCKED '} rule={d.rule} | {d.reason}")


def cmd_snapshot(a):
    print("wrote", build_cora_snapshot())


def cmd_experiments(a):
    snap, reg = _snap(), Registry(a.registry)
    val = _val(snap)
    for name, params in EXPERIMENTS:
        v, m, _ = tracked_run(snap, name, params, reg)
        print(f"{name} -> {v}  val_macro_f1={m['val_macro_f1']:.4f} unseen_acc={m['unseen_accuracy']:.4f}")
        _print_decision(v, reg.promote(v, val, GateConfig()))
    print("approved:", reg.approved)


def cmd_gate_demo(a):
    snap = _snap()
    val = _val(snap)
    demo = ROOT / "mlops" / "registry_demo"
    if demo.exists():
        shutil.rmtree(demo)
    reg, cfg = Registry(demo), GateConfig()
    print(f"gate: metric={cfg.metric} min_improvement={cfg.min_improvement} tie_policy={cfg.tie_policy} floor={cfg.first_model_floor}")
    steps = [("1) first model (floor rule), weak but above floor", "weak_h16_60epochs", dict(hidden=16, lr=0.01, epochs=60)),
             ("2) better candidate -> promoted", "gcn_h64_wide", EXPERIMENTS[1][1]),
             ("3) regression (undertrained) -> blocked", "undertrained_3_epochs", dict(hidden=16, epochs=3)),
             ("4) exact tie (same config again) -> blocked by tie_policy", "gcn_h64_wide_again", EXPERIMENTS[1][1])]
    for title, name, params in steps:
        v, m, _ = tracked_run(snap, name, params, reg, experiment="graphprobe-gate-demo")
        print(title); _print_decision(v, reg.promote(v, val, cfg)); print("   approved ->", reg.approved)
    print("5) missing metric -> blocked")
    from mlops.registry import evaluate_gate
    _print_decision("candidate", evaluate_gate({"val_macro_f1": None}, {"val_macro_f1": 0.8}, cfg))
    print("6) rollback after v2 turns out to be problematic in production")
    print("   approved before:", reg.approved, "-> after rollback:", reg.rollback())


def cmd_gate_check(a):
    """CI gate: train a fresh candidate and compare with the approved model on the SAME validation data.
    Non-mutating; exit code 1 when the gate would block."""
    snap, reg = _snap(), Registry(a.registry)
    with tempfile.TemporaryDirectory() as tmp:
        m, _, val, _ = train_bundle(snap, CI_PARAMS, Path(tmp) / "b")
        cand = load_bundle(Path(tmp) / "b")
        cfg = GateConfig(min_improvement=-a.tolerance, tie_policy="promote")  # CI = "no regression beyond tolerance"
        d = reg.check(cand, val, cfg)
    print(json.dumps({"approved": reg.approved, "candidate_val_macro_f1": d.candidate, "approved_val_macro_f1": d.approved,
                      "delta": d.delta, "rule": d.rule, "reason": d.reason, "passed": d.allowed}, indent=2))
    sys.exit(0 if d.allowed else 1)


def cmd_status(a):
    st = Registry(a.registry)._load()
    print("approved:", st["approved"], "| history:", st["history"])
    for v, info in st["versions"].items():
        print(f"  {v}: {info['status']:<11} val_macro_f1={info['metrics']['val_macro_f1']:.4f} run={info.get('run_name')}")


def cmd_promote(a):
    reg = Registry(a.registry)
    _print_decision(a.version, reg.promote(a.version, _val(_snap()), GateConfig()))


def cmd_rollback(a):
    print("approved ->", Registry(a.registry).rollback())


def cmd_serve(a):
    import uvicorn
    uvicorn.run("mlops.service:app", host="0.0.0.0", port=a.port)


def _demo(name):
    def run(a):
        from mlops import demos
        getattr(demos, name)(Registry(a.registry), _snap(), RESULTS)
    return run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mlops.cli")
    ap.add_argument("--registry", default=str(REGISTRY))
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("snapshot", cmd_snapshot), ("experiments", cmd_experiments), ("gate-demo", cmd_gate_demo),
                     ("status", cmd_status), ("rollback", cmd_rollback), ("serve", cmd_serve),
                     ("unseen-demo", _demo("unseen_demo")), ("drift-demo", _demo("drift_demo")),
                     ("consistency-demo", _demo("consistency_demo")), ("failure-demo", _demo("failure_demo"))]:
        sp.add_parser(name).set_defaults(fn=fn)
    p = sp.add_parser("gate-check"); p.add_argument("--tolerance", type=float, default=0.01); p.set_defaults(fn=cmd_gate_check)
    p = sp.add_parser("promote"); p.add_argument("version"); p.set_defaults(fn=cmd_promote)
    sp.choices["serve"].add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
