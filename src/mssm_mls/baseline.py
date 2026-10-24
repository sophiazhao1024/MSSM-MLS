"""Cumulative MLS with genuine rejection sampling and extra random exploration.

This reconstructs the disclosed algorithm. It cannot recreate undisclosed
physics cards, software options or random seeds from the lost original code.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .domain import (BOUNDS, FIXED_PARAMETERS, PARAMETER_NAMES, SOURCE,
                     EXPECTED_TOOL_VERSIONS, parameter_dict, sample, score,
                     validate_observables)
from .learning import RECONSTRUCTION_ASSUMPTIONS, Surrogates


EVIDENCE_NOTES = {
    "paper": {
        "source": SOURCE,
        "disclosed": ["four separate regressors for mh, mH, chi2_hs, ln(omega)",
                      "one physical/unphysical classifier; four hidden layers of 60",
                      "Adam; up to 2000 epochs; cumulative truth training",
                      "classifier before regression; likelihood-guided sampling"],
        "scaling_statement": "input scaling and output mean/std scaling are not stated; absence of reporting is not proof of absence",
    },
    "user": "this variant explicitly uses unscaled physical inputs and only natural-log omega labels; no mean/std, min-max or [-1,1] mapping",
    "xBIT": "framework reference only, not recovered Model-4 author code; its multi-output, dropout, holdout and likelihood-training branches are not adopted",
    "unreported": "fixed 2000 epochs, lr=0.001, batch=256, fresh initialization, threshold=0.5 and detailed physics settings remain reconstruction choices",
}


TARGET_POLICIES = ("paper-higgs", "eq7-only")
TARGET_POLICY_NOTE = (
    "paper-higgs requires true Eq.(7) cuts and HiggsBounds result 1; "
    "eq7-only is diagnostic. HB-excluded points remain physical training data. "
    "Proposals learn only the four Eq.(7) observables, with no extra HB network, "
    "so recommendations can spend evaluations on HB-excluded points."
)


def validate_target_policy(policy):
    if policy not in TARGET_POLICIES:
        raise ValueError(f"target_policy must be one of {TARGET_POLICIES}")


def target_decision(answer, target_eq7, policy):
    """Separate numerical physics validity, Eq.(7), and Higgs-data acceptance.

    HB result 0 means experimentally excluded, not an unphysical spectrum.
    Only the explicit integer 1 passes HB; absent/invalid/not-applicable HB
    does not establish Higgs consistency. A legacy provenance copy is usable
    when the original result has no top-level HB diagnostic.
    """
    validate_target_policy(policy)
    diagnostics = answer.get("diagnostics", {})
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    if "hb_result" in diagnostics:
        raw = diagnostics["hb_result"]
    else:
        provenance = answer.get("provenance", {})
        higgs = provenance.get("higgs", {}) if isinstance(provenance, dict) else {}
        raw = higgs.get("hb_result") if isinstance(higgs, dict) else None
    hb_result = raw if type(raw) is int and raw in (-1, 0, 1) else None
    eq7 = answer.get("status") == "ok" and bool(target_eq7)
    hb_allowed = hb_result == 1
    return {"target_policy": policy, "target_eq7": eq7,
            "hb_result": hb_result, "hb_allowed": hb_allowed,
            "target": eq7 and (policy == "eq7-only" or hb_allowed)}


def target_summary(records, policy):
    """Report both definitions against the same attempted-evaluation budget."""
    n, eq7, accepted = len(records), sum(r["target_eq7"] for r in records), sum(r["target"] for r in records)
    return {"target_policy": policy, "targets_eq7": eq7, "targets": accepted,
            "target_eq7_fraction_of_attempts": eq7 / max(n, 1),
            "target_fraction_of_attempts": accepted / max(n, 1),
            "eq7_hb_excluded": sum(r["target_eq7"] and r["hb_result"] == 0 for r in records),
            "eq7_hb_unresolved": sum(r["target_eq7"] and r["hb_result"] not in (0, 1) for r in records)}


@dataclass(frozen=True)
class ScanSettings:
    budget: int = 20000
    initial: int = 1000
    guided: int = 100
    random_extra: int = 5
    epochs: int = 2000
    learning_rate: float = 0.001
    batch_size: int = 256
    classifier_threshold: float = 0.5
    candidate_chunk: int = 65536
    max_proposals: int = 20000000
    max_consecutive_tool_errors: int = 5
    target_policy: str = "paper-higgs"

    def __post_init__(self):
        validate_target_policy(self.target_policy)
        counts = ("budget", "initial", "guided", "random_extra", "epochs", "batch_size",
                  "candidate_chunk", "max_proposals", "max_consecutive_tool_errors")
        if any(type(getattr(self, n)) is not int or getattr(self, n) < 1 for n in counts):
            raise ValueError("All count settings must be positive integers")
        if self.budget < self.initial or self.epochs > 2000:
            raise ValueError("Require budget >= initial and epochs <= 2000")
        if self.max_proposals < self.guided:
            raise ValueError("max_proposals must be at least guided")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 < self.classifier_threshold < 1:
            raise ValueError("classifier_threshold must lie in (0,1)")


class ProposalExhausted(RuntimeError):
    """The finite proposal cap was reached; NO random fallback is inserted."""

    def __init__(self, diagnostics):
        self.diagnostics = dict(diagnostics)
        super().__init__(f"Rejection proposal cap reached: {diagnostics['accepted']} / "
                         f"{diagnostics['requested']} guided points accepted from "
                         f"{diagnostics['proposals']} proposals")


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def rejection_sample(models, count, settings, rng, *, iteration=0):
    """Independent uniform proposals accepted with probability Lhat / M.

    For admissible predictions mh>0, Eq.(7) has 0 <= Lhat < 1, so the fixed
    envelope M=1 is valid. Log-space acceptance avoids exponential underflow.
    We never divide by a finite pool's observed maximum and never select top-k.
    Chunking is only vectorization: retain the first ``count`` accepted draws.
    The sampled density is proportional to classifier_gate * Lhat, NOT a
    posterior distribution certified by the real HEP calculation.
    """
    if type(count) is not int or count < 0:
        raise ValueError("count must be non-negative")
    accepted = []
    diagnostics = {"iteration": iteration, "requested": count, "accepted": 0,
                   "proposals": 0, "classifier_survivors": 0,
                   "positive_weight": 0, "envelope": 1.0}
    while len(accepted) < count and diagnostics["proposals"] < settings.max_proposals:
        size = min(settings.candidate_chunk,
                   settings.max_proposals - diagnostics["proposals"])
        candidates = sample(size, rng)
        predictions, probabilities = models.predict(candidates, settings.classifier_threshold)
        predictions, probabilities = np.asarray(predictions), np.asarray(probabilities)
        if predictions.shape != (size, 4) or probabilities.shape != (size,):
            raise ValueError("Surrogate returned incorrectly shaped predictions")
        log_weight, _ = score(predictions)
        gate = np.isfinite(probabilities) & (probabilities >= settings.classifier_threshold)
        log_weight[~gate] = -np.inf
        finite = np.isfinite(log_weight)
        if np.any(log_weight[finite] > 0):
            raise RuntimeError("Eq.(7) exceeded fixed rejection envelope M=1")
        # random() is [0,1); log(0)=-inf is legitimate. Requiring a finite
        # log_weight prevents a zero-weight row from passing -inf <= -inf.
        with np.errstate(divide="ignore"):
            log_uniform = np.log(rng.random(size))
        selected = np.flatnonzero(finite & (log_uniform <= log_weight))
        for index in selected[:count - len(accepted)]:
            accepted.append({"x": candidates[index], "origin": "guided",
                             "predicted": predictions[index].tolist()})
        diagnostics["proposals"] += size
        diagnostics["classifier_survivors"] += int(gate.sum())
        diagnostics["positive_weight"] += int(finite.sum())
        diagnostics["accepted"] = len(accepted)
    if len(accepted) != count:
        raise ProposalExhausted(diagnostics)
    return accepted, diagnostics


def run_scan(settings, evaluator, output_dir=None, *, seed=0, method="mls", models=None,
             checkpoint_path=None, resume_from=None, physics_fingerprint=None,
             observer=None, stop_after_iteration=None):
    """Evaluate truth, train cumulative models, reject-sample, then repeat.

    evaluator.evaluate(params, eval_id), or evaluator(params, eval_id), returns
    {status: ok|unphysical|tool_error, observables: {mh,mH,chi2_hs,omega}, ...}.
    Each attempt, including errors, uses one expensive-call budget unit.

    Returns {records, history, summary, manifest}. Optional output_dir must be
    empty/new. On an exception, partial files remain and exc.scan_result holds
    the same structure. ``models`` exists for deterministic contract tests.

    Every COMPLETE iteration is 100 guided + 5 independent random points by
    default. At the 20,000-attempt limit the terminal iteration is 95+5 because
    20,000-1,000 is not divisible by 105. This explicit truncation is our
    convention, not an additional setting disclosed by the original paper.

    Optional checkpoints contain complete-iteration truth, the three NumPy RNG
    states and the fresh-fit counter, not a mid-training optimizer snapshot.
    Resume requires identical settings, code/dependencies and an explicit
    caller-supplied physics fingerprint. It writes into a NEW output directory.
    Mid-iteration journals remain diagnostic; only the last complete boundary
    is resumable. ``observer(event)`` is read-only monitoring: exceptions are
    recorded, never used to change sampling or training. ``stop_after_iteration``
    requests a clean boundary pause (0 means after the initial sample).
    """
    if method not in {"mls", "random"} or type(seed) is not int or seed < 0:
        raise ValueError("method must be mls/random and seed must be non-negative")
    checkpointing = checkpoint_path is not None or resume_from is not None
    if checkpointing and (not isinstance(physics_fingerprint, str) or not physics_fingerprint):
        raise ValueError("Checkpoint/resume requires a nonempty physics_fingerprint")
    if observer is not None and not callable(observer):
        raise TypeError("observer must be callable")
    if stop_after_iteration is not None and (
            type(stop_after_iteration) is not int or stop_after_iteration < 0):
        raise ValueError("stop_after_iteration must be a non-negative integer or None")
    checkpoint_file = None if checkpoint_path is None else Path(checkpoint_path)
    if checkpoint_file is not None:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    evaluate = evaluator.evaluate if hasattr(evaluator, "evaluate") else evaluator
    if not callable(evaluate):
        raise TypeError("evaluator must be callable or expose evaluate")
    out = None if output_dir is None else Path(output_dir)
    if out is not None:
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise FileExistsError("Use a new or empty output directory")
        out.mkdir(parents=True, exist_ok=True)
    streams = np.random.SeedSequence(seed).spawn(3)
    initial_rng, proposal_rng, explore_rng = [np.random.default_rng(s) for s in streams]
    if models is None and method == "mls":
        models = Surrogates(epochs=settings.epochs, learning_rate=settings.learning_rate,
                            batch_size=settings.batch_size, seed=seed)
    # Every baseline fit creates new networks and Adam optimizers. The only
    # model state needed by the NEXT fit is its deterministic seed counter.
    # Injected test models must obey the same fresh-fit contract to resume.
    if checkpointing and method == "mls" and (
            type(getattr(models, "fit_count", None)) is not int or models.fit_count < 0):
        raise ValueError("Resumable models must expose their fresh-fit integer fit_count")
    assumptions = dict(RECONSTRUCTION_ASSUMPTIONS)
    assumptions.update({
        "proposal": "uniform box, iid with replacement; no deduplication",
        "rejection_envelope": "fixed M=1 after positive physical-domain guard",
        "proposal_cap": "stop on exhaustion; never substitute random fallback",
        "theta_zero": "0; strict direct decimal endpoint comparisons, no epsilon",
        "final_batch": "strict attempt budget; preserve up to 5 random slots",
        "technical_failure": "counts attempted calls, excluded from supervised labels",
        "seed_scope": "reproducible in same deterministic CPU/software environment",
        "target_policy": TARGET_POLICY_NOTE,
    })
    manifest = {"source": SOURCE, "algorithm": "baseline", "method": method, "seed": seed,
                "variant": "logomega-only",
                "input_scaling": "none", "output_scaling": "logomega only",
                "evidence": EVIDENCE_NOTES,
                "settings": asdict(settings), "status": "running",
                "parameter_names": list(PARAMETER_NAMES), "bounds": BOUNDS.tolist(),
                "fixed_parameters": dict(FIXED_PARAMETERS),
                "expected_tool_versions": EXPECTED_TOOL_VERSIONS,
                "numpy": np.__version__, "torch": torch.__version__,
                "reconstruction_assumptions": assumptions}
    records, history, summary = [], [], {}
    result = {"records": records, "history": history, "summary": summary, "manifest": manifest}
    started, consecutive_errors = time.perf_counter(), 0
    iteration, previous_seconds = 0, 0.
    identity = None
    if checkpointing:
        code_files = ("baseline.py", "learning.py", "domain.py")
        identity = {
            "settings": asdict(settings), "seed": seed, "method": method,
            "physics_fingerprint": physics_fingerprint,
            "numpy": np.__version__, "torch": torch.__version__,
            "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in code_files},
        }
    if resume_from is not None:
        envelope = json.loads(Path(resume_from).read_text(encoding="utf-8"))
        saved = envelope["payload"]
        canonical = json.dumps(saved, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest() != envelope.get("sha256"):
            raise ValueError("Checkpoint checksum mismatch")
        if saved.get("schema") != 1 or saved.get("kind") != "complete_iteration":
            raise ValueError("Not a supported complete-iteration checkpoint")
        if saved.get("identity") != identity:
            raise ValueError("Checkpoint settings, seed, method, physics or implementation mismatch")
        iteration = saved["completed_iteration"]
        if type(iteration) is not int or iteration < 0:
            raise ValueError("Invalid checkpoint iteration")
        expected = min(settings.initial + iteration * (settings.guided + settings.random_extra),
                       settings.budget)
        records.extend(saved["records"])
        history.extend(saved["history"])
        if (len(records) != expected or len(history) != iteration
                or [r.get("eval_id") for r in records] != list(range(1, expected + 1))
                or [h.get("iteration") for h in history] != list(range(1, iteration + 1))):
            raise ValueError("Checkpoint is not a complete, contiguous iteration boundary")
        if saved["fit_count"] != (iteration if method == "mls" else None):
            raise ValueError("Checkpoint fresh-fit counter disagrees with complete iterations")
        if method == "mls":
            models.fit_count = saved["fit_count"]
        for rng, key in zip((initial_rng, proposal_rng, explore_rng), ("initial", "proposal", "explore")):
            rng.bit_generator.state = saved["rng_states"][key]
        consecutive_errors = saved["consecutive_tool_errors"]
        if type(consecutive_errors) is not int or not 0 <= consecutive_errors < settings.max_consecutive_tool_errors:
            raise ValueError("Invalid checkpoint technical-error streak")
        previous_seconds = float(saved["elapsed_seconds"])
        if not np.isfinite(previous_seconds) or previous_seconds < 0:
            raise ValueError("Invalid checkpoint elapsed time")
        manifest["resumed_from"] = str(Path(resume_from).resolve())
        manifest["resumed_attempts"] = len(records)
    if checkpointing:
        manifest["checkpoint_policy"] = "complete iteration only; fresh next fit; no mid-epoch resume"
        manifest["physics_fingerprint"] = physics_fingerprint

    def notify(event, **details):
        if observer is None:
            return
        payload = {"event": event, "iteration": iteration, "attempts": len(records),
                   "elapsed_seconds": previous_seconds + time.perf_counter() - started, **details}
        try:
            # No live data/model/RNG objects are exposed to an observer.
            observer(payload)
        except Exception as exc:
            manifest.setdefault("observer_errors", []).append(f"{type(exc).__name__}: {exc}")

    def save_boundary():
        if checkpoint_file is None:
            return
        payload = {"schema": 1, "kind": "complete_iteration", "identity": identity,
                   "completed_iteration": iteration, "records": records, "history": history,
                   "fit_count": models.fit_count if method == "mls" else None,
                   "rng_states": {key: rng.bit_generator.state for key, rng in zip(
                       ("initial", "proposal", "explore"), (initial_rng, proposal_rng, explore_rng))},
                   "consecutive_tool_errors": consecutive_errors,
                   "elapsed_seconds": previous_seconds + time.perf_counter() - started}
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        write_json(checkpoint_file, {"sha256": hashlib.sha256(canonical).hexdigest(), "payload": payload})
        notify("checkpoint", path=str(checkpoint_file), completed_iteration=iteration)

    if out is not None:
        write_json(out / "manifest.json", manifest)
        if resume_from is not None:
            with (out / "records.jsonl").open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            write_json(out / "training_history.json", history)

    def evaluate_batch(batch, iteration):
        nonlocal consecutive_errors
        for item in batch:
            params = parameter_dict(item["x"])
            eval_id = len(records) + 1
            try:
                answer = evaluate(params, eval_id)
                if not isinstance(answer, dict) or answer.get("status") not in {
                        "ok", "unphysical", "tool_error"}:
                    raise ValueError("Backend returned an invalid result/status")
                observations, vector, log_score, target = None, None, None, False
                if answer["status"] == "ok":
                    vector = validate_observables(answer.get("observables"))
                    observations = dict(zip(("mh", "mH", "chi2_hs", "omega"), map(float, vector)))
                    weights, hits = score(vector.reshape(1, 4))
                    log_score = float(weights[0]) if np.isfinite(weights[0]) else None
                    target = bool(hits[0])
            except Exception as exc:
                answer = {"status": "tool_error", "reason": f"{type(exc).__name__}: {exc}"}
                observations, vector, log_score, target = None, None, None, False
            record = {"eval_id": eval_id, "x": list(map(float, item["x"])),
                      "parameters": params, "status": answer["status"],
                      "observables": observations, "y": None if vector is None else vector.tolist(),
                      "log_score": log_score,
                      **target_decision(answer, target, settings.target_policy),
                      "origin": item["origin"], "iteration": iteration,
                      "predicted": item.get("predicted"),
                      "reason": answer.get("reason", ""),
                      "provenance": answer.get("provenance", {}),
                      "diagnostics": answer.get("diagnostics", {})}
            records.append(record)
            if out is not None:
                with (out / "records.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            notify("point", eval_id=eval_id, status=record["status"], target=record["target"])
            consecutive_errors = consecutive_errors + 1 if record["status"] == "tool_error" else 0
            if consecutive_errors >= settings.max_consecutive_tool_errors:
                raise RuntimeError("Repeated technical failures; inspect the physics driver logs")

    try:
        notify("resumed" if resume_from is not None else "started")
        if resume_from is None:
            initial = [{"x": x, "origin": "initial", "predicted": None}
                       for x in sample(settings.initial, initial_rng)]
            notify("evaluation_start", batch_size=len(initial))
            evaluate_batch(initial, 0)
            notify("evaluation_complete", batch_size=len(initial))
            save_boundary()
        while (len(records) < settings.budget
               and (stop_after_iteration is None or iteration < stop_after_iteration)):
            iteration += 1
            size = min(settings.guided + settings.random_extra, settings.budget - len(records))
            n_random = min(settings.random_extra, size)
            n_guided = size - n_random
            entry = {"iteration": iteration, "trained_through": len(records),
                     "batch_size": size, "guided_count": n_guided if method == "mls" else 0,
                     "random_count": n_random if method == "mls" else size}
            history.append(entry)
            if method == "mls":
                phase_start = time.perf_counter() if observer is not None else None
                notify("fit_start", trained_through=len(records))
                entry["fit"] = models.fit(records)
                notify("fit_complete", seconds=None if phase_start is None else time.perf_counter() - phase_start)
                phase_start = time.perf_counter() if observer is not None else None
                notify("selection_start", guided_count=n_guided)
                try:
                    batch, entry["selection"] = rejection_sample(
                        models, n_guided, settings, proposal_rng, iteration=iteration)
                except ProposalExhausted as exc:
                    entry["selection"] = exc.diagnostics
                    raise
                notify("selection_complete", seconds=None if phase_start is None else time.perf_counter() - phase_start,
                       proposals=entry["selection"]["proposals"])
                # These bypass both the classifier and the regression-based weight.
                batch += [{"x": x, "origin": "explore", "predicted": None}
                          for x in sample(n_random, explore_rng)]
            else:
                batch = [{"x": x, "origin": "random", "predicted": None}
                         for x in sample(size, explore_rng)]
            phase_start = time.perf_counter() if observer is not None else None
            notify("evaluation_start", batch_size=len(batch))
            evaluate_batch(batch, iteration)
            notify("evaluation_complete", batch_size=len(batch),
                   seconds=None if phase_start is None else time.perf_counter() - phase_start)
            if out is not None:
                write_json(out / "training_history.json", history)
            save_boundary()
        manifest["status"] = "completed" if len(records) == settings.budget else "paused"
    except Exception as exc:
        manifest["status"] = "proposal_exhausted" if isinstance(exc, ProposalExhausted) else "aborted"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        exc.scan_result = result
        raise
    finally:
        if manifest["status"] == "running":
            manifest["status"] = "interrupted"
        summary.update({"status": manifest["status"], "attempts": len(records),
                        **{s: sum(r["status"] == s for r in records)
                           for s in ("ok", "unphysical", "tool_error")},
                        **target_summary(records, settings.target_policy),
                        "wall_seconds": previous_seconds + time.perf_counter() - started,
                        "note": TARGET_POLICY_NOTE + " No posterior/evidence claim."})
        if out is not None:
            write_json(out / "manifest.json", manifest)
            write_json(out / "summary.json", summary)
            write_json(out / "training_history.json", history)
        notify("finished", status=manifest["status"])
    return result
