"""Beginner entry points: doctor, point, scan, and compare.

All default scans call the compiled physics drivers. Unit-test fakes are never
available as a physics backend from the command line.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from .backend import PhysicsBackend, read_json, save_json
from .baseline import (ScanSettings, run_scan, target_decision, TARGET_POLICIES,
                       TARGET_POLICY_NOTE)
from .domain import score
from .learning import Surrogates


def parser():
    root = argparse.ArgumentParser(prog="mssm-mls-logomega", description="Model-4 MSSM baseline: raw inputs, log-omega-only labels")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("doctor", "point", "scan"):
        p = sub.add_parser(name)
        p.add_argument("--backend", type=Path, default=Path("configs/backend.local.json"))
        p.add_argument("--allow-version-substitution", action="store_true",
                       help="Explicitly allow a run labelled with different physics versions")
        if name != "doctor":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--target-policy", choices=TARGET_POLICIES,
                           help="Default paper-higgs requires Eq.(7) plus HB=1; eq7-only is diagnostic")
        if name == "point":
            p.add_argument("--M1", type=float, required=True)
            p.add_argument("--mu", type=float, required=True)
            p.add_argument("--tan-beta", type=float, required=True)
        if name == "scan":
            p.add_argument("--settings", type=Path, default=Path("configs/baseline.json"))
            p.add_argument("--method", choices=("mls", "random"), default="mls")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--cache", type=Path, help="Exact-input SQLite truth cache with physics fingerprint")
            p.add_argument("--reuse-run", type=Path, action="append", default=[],
                           help="Import completed truth from a previous run; requires --cache")
            for flag in ("budget", "initial", "epochs"):
                p.add_argument("--" + flag, type=int)
    p = sub.add_parser("compare")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    return root


def compare_runs(paths, output):
    """Compare true accepted targets only under the same explicit policy.

    Older manifests without a policy used Eq.(7) alone; never relabel them as
    passing HiggsBounds. Their files are read without alteration.
    """
    loaded = []
    for path in paths:
        manifest = read_json(path / "manifest.json")
        summary = read_json(path / "summary.json")
        records = [json.loads(line) for line in (path / "records.jsonl").read_text().splitlines()]
        policy = manifest.get("settings", {}).get("target_policy", "eq7-only")
        loaded.append((path, manifest, summary, records, policy))
    policies = {entry[4] for entry in loaded}
    if len(policies) != 1:
        raise ValueError("Cannot compare different target policies; legacy runs are eq7-only diagnostics")
    policy = next(iter(policies))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    rows = []
    for path, manifest, summary, records, policy in loaded:
        name = "random" if manifest["method"] == "random" else manifest.get("algorithm", "baseline") + " MLS"
        guard = manifest.get("settings", {}).get("max_consecutive_tool_errors", 5)
        suffix = f" (error guard {guard})" if guard != 5 else ""
        variant = manifest.get("variant", "legacy-unspecified")
        label = f"{name}{suffix}, seed {manifest['seed']} [{policy}; {variant}]"
        ax.plot(np.arange(1, len(records) + 1), np.cumsum([r["target"] for r in records]), label=label)
        rows.append({"run": str(path.resolve()), "label": label, **summary, "target_policy": policy})
    ylabel = "Cumulative true Eq.(7) + HB-allowed targets" if policy == "paper-higgs" else "Cumulative Eq.(7)-only diagnostic targets"
    ax.set(xlabel="Logical truth evaluations (including cache reuse)", ylabel=ylabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    save_json(output / "comparison.json", rows)
    return rows


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "compare":
            print(json.dumps(compare_runs(args.runs, args.output), ensure_ascii=False, indent=2))
            return
        root = args.output / "physics" if args.command != "doctor" else Path("runs/doctor-unused")
        backend = PhysicsBackend(args.backend, root, allow_version_substitution=args.allow_version_substitution)
        readiness = backend.doctor()
        if args.command == "doctor":
            print(json.dumps(readiness, ensure_ascii=False, indent=2))
            if not readiness["ready"]:
                raise SystemExit(2)
            return
        if not readiness["ready"]:
            raise RuntimeError("Physics binaries are not ready. Build them, then run doctor.")
        if args.command == "point":
            args.output.mkdir(parents=True, exist_ok=False)
            result = backend.evaluate({"M1": args.M1, "mu": args.mu, "tan_beta": args.tan_beta}, 1)
            eq7_target = False
            if result["status"] == "ok":
                o = result["observables"]
                weights, hits = score([[o[k] for k in ("mh", "mH", "chi2_hs", "omega")]])
                eq7_target = bool(hits[0])
                result["log_score"] = float(weights[0]) if np.isfinite(weights[0]) else None
            result.update(target_decision(result, eq7_target, args.target_policy or "paper-higgs"))
            result["eq7_target"] = result["target_eq7"]  # retain the original point-CLI field
            result["target_policy_note"] = TARGET_POLICY_NOTE
            save_json(args.output / "point.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result["status"] == "tool_error":
                raise SystemExit(2)
            return
        config = read_json(args.settings)
        evaluator = backend
        if args.reuse_run and args.cache is None:
            raise ValueError("--reuse-run requires --cache")
        if args.cache is not None:
            from .execution import CachedBackend
            evaluator = CachedBackend(backend, args.cache, workers=1)
            for previous in args.reuse_run:
                evaluator.import_run(previous)
        for name in ("budget", "initial", "epochs"):
            if getattr(args, name) is not None:
                config[name] = getattr(args, name)
        if args.target_policy is not None:
            config["target_policy"] = args.target_policy
        try:
            settings = ScanSettings(**config)
            model = Surrogates(epochs=settings.epochs, learning_rate=settings.learning_rate,
                               batch_size=settings.batch_size, seed=args.seed) if args.method == "mls" else None
            result = run_scan(settings, evaluator, args.output, seed=args.seed, method=args.method, models=model)
            if model is not None and model.fit_count:
                torch.save(model.export(), args.output / "surrogates.pt")
        finally:
            if (args.output / "manifest.json").is_file():
                save_json(args.output / "physics_manifest.json", readiness | {"configuration": backend.config})
                if args.cache is not None:
                    save_json(args.output / "execution.json", evaluator.report())
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
