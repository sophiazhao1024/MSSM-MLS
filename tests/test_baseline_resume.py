"""Boundary restart contracts using explicit software doubles, not SUSY truth."""
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from mssm_mls.baseline import ScanSettings, run_scan


class FreshModels:
    """Only fit_count persists between fresh fits, as in baseline Surrogates."""
    def __init__(self):
        self.fit_count = 0

    def fit(self, records):
        self.fit_count += 1
        self.mass = 1. + self.fit_count + np.mean([r["x"][0] for r in records]) / 20.
        return {"fit": self.fit_count, "training_count": len(records), "mass": self.mass}

    def predict(self, x, threshold):
        values = np.tile([self.mass, 126., 40., .1186], (len(x), 1))
        values[:, 0] += x[:, 0] / 10.
        probability = np.where(x[:, 2] < 3., .1, .9)
        values[probability < threshold] = np.nan
        return values, probability


def truth(parameters, eval_id):
    if eval_id % 11 == 0:
        return {"status": "tool_error", "reason": "explicit test fixture failure"}
    if parameters["M1"] < 4.:
        return {"status": "unphysical", "reason": "explicit test fixture rejection"}
    return {"status": "ok", "observables": {
        "mh": parameters["M1"], "mH": 126., "chi2_hs": 40., "omega": .1186},
        "diagnostics": {"hb_result": int(parameters["mu"] > 8000.)},
        "provenance": {"fixture": True}}


def settings():
    return ScanSettings(initial=7, budget=20, guided=3, random_extra=2,
                        epochs=2, candidate_chunk=8, max_proposals=10000)


@pytest.mark.parametrize("method", ["mls", "random"])
@pytest.mark.parametrize("boundary", [0, 1, 2])
def test_continuous_and_boundary_resume_are_pointwise_identical(tmp_path, method, boundary):
    config = settings()
    whole = run_scan(config, truth, models=FreshModels(), method=method, seed=17)
    checkpoint = tmp_path / "checkpoint.json"
    first = run_scan(config, truth, tmp_path / "first", models=FreshModels(), method=method,
                     seed=17, checkpoint_path=checkpoint, physics_fingerprint="fixture-oracle-1",
                     stop_after_iteration=boundary)
    assert first["summary"]["status"] == "paused"
    evaluated = []
    def observed_truth(params, eval_id):
        evaluated.append(eval_id)
        return truth(params, eval_id)
    events = []
    resumed = run_scan(config, observed_truth, tmp_path / "resumed", models=FreshModels(),
                       method=method, seed=17, resume_from=checkpoint,
                       checkpoint_path=tmp_path / "next.json",
                       physics_fingerprint="fixture-oracle-1", observer=events.append)
    assert resumed["records"] == whole["records"]
    assert resumed["history"] == whole["history"]
    assert evaluated == list(range(len(first["records"]) + 1, config.budget + 1))
    assert resumed["summary"]["status"] == "completed"
    assert len((tmp_path / "resumed" / "records.jsonl").read_text().splitlines()) == config.budget
    assert events[0]["event"] == "resumed"
    saved = json.loads((tmp_path / "next.json").read_text())["payload"]
    assert saved["completed_iteration"] == 3
    assert saved["fit_count"] == (3 if method == "mls" else None)
    assert set(saved["rng_states"]) == {"initial", "proposal", "explore"}


def test_observation_and_checkpointing_do_not_change_numerical_path(tmp_path):
    config = settings()
    plain = run_scan(config, truth, models=FreshModels(), seed=4)
    events = []
    monitored = run_scan(config, truth, models=FreshModels(), seed=4,
                         checkpoint_path=tmp_path / "checkpoint.json",
                         physics_fingerprint="fixture", observer=events.append)
    assert plain["records"] == monitored["records"]
    assert plain["history"] == monitored["history"]
    assert [event["eval_id"] for event in events if event["event"] == "point"] == list(range(1, 21))
    assert sum(event["event"] == "fit_start" for event in events) == 3
    assert sum(event["event"] == "fit_complete" for event in events) == 3
    assert "checkpoint_policy" not in plain["manifest"]
    assert "physics_fingerprint" not in plain["manifest"]


def test_observer_failure_never_changes_the_scan():
    def broken_observer(event):
        event["attempts"] = -999  # detached snapshot, no live record access
        raise OSError("fixture monitor unavailable")
    plain = run_scan(settings(), truth, models=FreshModels(), seed=4)
    monitored = run_scan(settings(), truth, models=FreshModels(), seed=4, observer=broken_observer)
    assert plain["records"] == monitored["records"]
    assert plain["history"] == monitored["history"]
    assert monitored["manifest"]["observer_errors"]
    assert monitored["summary"]["status"] == "completed"


@pytest.mark.parametrize("change", ["seed", "method", "settings", "physics"])
def test_resume_rejects_mismatched_identity(tmp_path, change):
    checkpoint = tmp_path / "checkpoint.json"
    config = settings()
    run_scan(config, truth, models=FreshModels(), seed=4, stop_after_iteration=1,
             checkpoint_path=checkpoint, physics_fingerprint="fixture-1")
    kwargs = {"seed": 4, "method": "mls", "physics_fingerprint": "fixture-1"}
    if change == "seed":
        kwargs["seed"] = 5
    elif change == "method":
        kwargs["method"] = "random"
    elif change == "settings":
        config = replace(config, budget=25)
    else:
        kwargs["physics_fingerprint"] = "fixture-2"
    with pytest.raises(ValueError, match="mismatch"):
        run_scan(config, truth, models=FreshModels(), resume_from=checkpoint, **kwargs)


def rewrite_envelope(path, envelope):
    canonical = json.dumps(envelope["payload"], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    envelope["sha256"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(envelope))


def test_checkpoint_integrity_and_complete_boundary_are_validated(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    run_scan(settings(), truth, models=FreshModels(), stop_after_iteration=1,
             checkpoint_path=checkpoint, physics_fingerprint="fixture")
    pristine = checkpoint.read_text()
    envelope = json.loads(pristine)
    envelope["payload"]["records"][0]["x"][0] += 1
    checkpoint.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="checksum"):
        run_scan(settings(), truth, models=FreshModels(), resume_from=checkpoint,
                 physics_fingerprint="fixture")
    envelope = json.loads(pristine)
    envelope["payload"]["records"].pop()
    rewrite_envelope(checkpoint, envelope)
    with pytest.raises(ValueError, match="contiguous"):
        run_scan(settings(), truth, models=FreshModels(), resume_from=checkpoint,
                 physics_fingerprint="fixture")


def test_partial_failure_does_not_replace_last_complete_checkpoint(tmp_path):
    config = replace(settings(), max_consecutive_tool_errors=2)
    def failing(params, eval_id):
        if eval_id in (13, 14):
            return {"status": "tool_error", "reason": "fixture"}
        return truth(params, eval_id)
    checkpoint = tmp_path / "checkpoint.json"
    with pytest.raises(RuntimeError, match="Repeated") as caught:
        run_scan(config, failing, tmp_path / "partial", models=FreshModels(),
                 checkpoint_path=checkpoint, physics_fingerprint="fixture")
    saved = json.loads(checkpoint.read_text())["payload"]
    assert saved["completed_iteration"] == 1
    assert len(saved["records"]) == 12
    assert len(caught.value.scan_result["records"]) == 14
    assert len((tmp_path / "partial" / "records.jsonl").read_text().splitlines()) == 14
    assert caught.value.scan_result["summary"]["status"] == "aborted"


def test_error_streak_is_preserved_across_boundary(tmp_path):
    config = replace(settings(), max_consecutive_tool_errors=2)
    def fail_across_boundary(params, eval_id):
        if eval_id in (12, 13):
            return {"status": "tool_error", "reason": "fixture"}
        return truth(params, eval_id) if eval_id != 11 else truth(params, 1)
    checkpoint = tmp_path / "checkpoint.json"
    run_scan(config, fail_across_boundary, models=FreshModels(), stop_after_iteration=1,
             checkpoint_path=checkpoint, physics_fingerprint="fixture")
    assert json.loads(checkpoint.read_text())["payload"]["consecutive_tool_errors"] == 1
    with pytest.raises(RuntimeError, match="Repeated") as resumed:
        run_scan(config, fail_across_boundary, models=FreshModels(), resume_from=checkpoint,
                 physics_fingerprint="fixture")
    with pytest.raises(RuntimeError, match="Repeated") as continuous:
        run_scan(config, fail_across_boundary, models=FreshModels())
    assert resumed.value.scan_result["records"] == continuous.value.scan_result["records"]
    assert len(resumed.value.scan_result["records"]) == 13


def test_fingerprint_is_required_for_checkpoint_or_resume(tmp_path):
    with pytest.raises(ValueError, match="physics_fingerprint"):
        run_scan(settings(), truth, checkpoint_path=tmp_path / "checkpoint.json")
