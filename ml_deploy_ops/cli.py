"""CLI for ml-deploy-ops. State is held in a JSON registry file (--state)."""
from __future__ import annotations

import argparse
import json
import sys

from ml_deploy_ops.bench import run_simulation
from ml_deploy_ops.evaluation import EvalGate
from ml_deploy_ops.service import ControlPlane


def _parse_kv(pairs, cast=float):
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[k] = cast(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ml-deploy-ops", description=__doc__)
    p.add_argument("--state", default="registry.json", help="registry JSON file")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register"); r.add_argument("name"); r.add_argument("version")
    r.add_argument("--artifact", default=""); r.add_argument("--framework", default="unknown")

    sub.add_parser("list")

    e = sub.add_parser("eval"); e.add_argument("name"); e.add_argument("version")
    e.add_argument("--dataset", default="holdout"); e.add_argument("--metric", action="append", default=[])
    e.add_argument("--threshold", action="append", default=[], help="metric=value minimum, repeatable")

    d = sub.add_parser("deploy"); d.add_argument("name"); d.add_argument("version")
    d.add_argument("--traffic", type=float, default=5.0)

    pr = sub.add_parser("promote"); pr.add_argument("deployment_id")
    rb = sub.add_parser("rollback"); rb.add_argument("deployment_id")
    sub.add_parser("deployments")

    m = sub.add_parser("metrics"); m.add_argument("name"); m.add_argument("version")

    dm = sub.add_parser("demo")
    bn = sub.add_parser("bench"); bn.add_argument("--models", type=int, default=20)
    bn.add_argument("--requests", type=int, default=200)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "bench":
        print(json.dumps(run_simulation(models=args.models, requests_per_model=args.requests), indent=2))
        return 0

    cp = ControlPlane(registry_path=args.state)
    try:
        if args.cmd == "register":
            m = cp.register_model(args.name, args.version, artifact_uri=args.artifact, framework=args.framework)
            print(f"registered {m.key} stage={m.stage.value}")
        elif args.cmd == "list":
            for m in cp.registry.list():
                print(f"{m.key:30s} stage={m.stage.value:12s} framework={m.framework} metrics={m.metrics}")
            if len(cp.registry) == 0:
                print("(no models registered)")
        elif args.cmd == "eval":
            thresholds = {k: ("min", v) for k, v in _parse_kv(args.threshold).items()} or None
            cp_gate = ControlPlane(registry_path=args.state, gate=EvalGate(thresholds=thresholds))
            # Reuse the same registry file; eval through the gated plane.
            verdict = cp_gate.run_eval(args.name, args.version, args.dataset, _parse_kv(args.metric))
            print(f"eval {'PASSED' if verdict.passed else 'FAILED'} for {args.name}:{args.version}")
            for r in verdict.reasons:
                print(f"  - {r}")
            return 0 if verdict.passed else 2
        elif args.cmd == "deploy":
            dep = cp.deploy_canary(args.name, args.version, traffic_percent=args.traffic)
            print(f"canary deployment {dep.id} model={dep.model_key} traffic={dep.traffic_percent}%")
            print("note: deployments are in-process in this build; only the registry persists across CLI calls")
        elif args.cmd == "promote":
            dep = cp.canary.promote(args.deployment_id); print(f"promoted {dep.model_key} to production")
        elif args.cmd == "rollback":
            dep = cp.canary.rollback(args.deployment_id); print(f"rolled back {dep.model_key}")
        elif args.cmd == "deployments":
            for d in cp.canary.list():
                print(f"{d.id} {d.model_key} stage={d.stage.value} traffic={d.traffic_percent}% active={d.active}")
            print("(deployments are in-process; a fresh CLI process lists none — see README limitations)")
        elif args.cmd == "metrics":
            s = cp.metrics.summary(f"{args.name}:{args.version}")
            print(f"{s.model_key}: count={s.count} (metrics are in-process; see `demo`/`bench` for populated metrics)")
        elif args.cmd == "demo":
            cp2 = ControlPlane()
            cp2.register_model("sentiment", "1.0.0", framework="synthetic")
            print(cp2.run_eval("sentiment", "1.0.0", "holdout", {"accuracy": 0.93}).reasons)
            dep = cp2.deploy_canary("sentiment", "1.0.0", traffic_percent=10.0)
            for i in range(50):
                cp2.record_request("sentiment", "1.0.0", latency_ms=40 + i % 7, ok=True)
            print(cp2.metrics.summary("sentiment:1.0.0"))
            cp2.canary.promote(dep.id)
            print(f"promoted; production={cp2.registry.production('sentiment').key}")
    except (KeyError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
