"""Cache/concurrency contracts with explicit test doubles; no MSSM claims."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import sys

import pytest

from mssm_mls.execution import CachedBackend
from mssm_mls.backend import PhysicsBackend
from mssm_mls.cards import write_fh_card


PARAMS = {"M1": 10., "mu": 8000., "tan_beta": 5.}
OBS = {"mh": 5., "mH": 125., "chi2_hs": 90., "omega": .12}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TruthDouble:
    """Retain per-point artifacts just like the real subprocess backend."""
    def __init__(self, root, statuses=()):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_path = self.root / "config.json"
        self.config_path.write_text('{"setting":1}')
        self.paths = {name: self.root / name for name in ("fh_driver", "omega_driver")}
        for name, path in self.paths.items():
            path.write_text(name)
        self.adapter_paths = {name: self.root / name for name in ("cards.py", "domain.py", "backend.py")}
        for name, path in self.adapter_paths.items():
            path.write_text(name + " initial implementation")
        self.statuses = list(statuses)
        self.calls = []
        self.lock = threading.Lock()
        self.before_evaluate = None

    def doctor(self):
        return {"ready": True, "config_sha256": sha(self.config_path),
                "adapter_sha256": {k: sha(v) for k, v in self.adapter_paths.items()},
                "checks": {name: {"sha256": sha(path)} for name, path in self.paths.items()}}

    def evaluate(self, parameters, eval_id):
        with self.lock:
            self.calls.append((dict(parameters), eval_id))
            status = self.statuses.pop(0) if self.statuses else "ok"
        if self.before_evaluate:
            self.before_evaluate(parameters, eval_id)
        point = self.root / "physics" / f"point_{eval_id:06d}"
        point.mkdir(parents=True, exist_ok=False)
        (point / "request.json").write_text(json.dumps(parameters))
        oracle = self.doctor()
        result = {"status": status, "observables": dict(OBS) if status == "ok" else {},
                  "reason": "explicit test double", "seconds": 1.25,
                  "provenance": {"config_sha256": oracle["config_sha256"],
                      "adapter_sha256": oracle["adapter_sha256"],
                      "binary_sha256": {k: v["sha256"] for k, v in oracle["checks"].items()},
                      "input_parameters": dict(parameters),
                      "request_sha256": sha(point / "request.json"),
                      "workdir": str(point.resolve())}}
        (point / "evaluation.json").write_text(json.dumps(result))
        return result


def test_exact_float_key_distinguishes_adjacent_values_and_order(tmp_path):
    cache = CachedBackend(TruthDouble(tmp_path / "source"), tmp_path / "truth.sqlite")
    adjacent = PARAMS | {"M1": math.nextafter(PARAMS["M1"], math.inf)}
    assert round(adjacent["M1"], 12) == round(PARAMS["M1"], 12)
    assert cache.key(PARAMS) != cache.key(adjacent)
    assert cache.key(PARAMS) == cache.key(dict(reversed(list(PARAMS.items()))))


@pytest.mark.parametrize("change", ["config", "fh_driver", "omega_driver"])
def test_fingerprint_changes_for_config_or_either_binary(tmp_path, change):
    backend = TruthDouble(tmp_path / "source")
    first = CachedBackend(backend, tmp_path / "truth.sqlite")
    key = first.key(PARAMS)
    first.evaluate(PARAMS, 0)
    path = backend.config_path if change == "config" else backend.paths[change]
    path.write_text(path.read_text() + "\nchanged")
    second = CachedBackend(backend, tmp_path / "truth.sqlite")
    assert second.key(PARAMS) != key
    second.evaluate(PARAMS, 1)
    assert second.report()["cache_hits"] == 0
    assert len(backend.calls) == 2


def test_tool_failure_is_retried_and_never_cached(tmp_path):
    backend = TruthDouble(tmp_path / "source", ["tool_error", "ok"])
    cache = CachedBackend(backend, tmp_path / "truth.sqlite")
    assert cache.evaluate(PARAMS, 0)["status"] == "tool_error"
    assert cache.evaluate(PARAMS, 1)["status"] == "ok"
    hit = cache.evaluate(PARAMS, 2)
    assert hit["provenance"]["reuse"]["cache_hit"] is True
    assert hit["seconds"] == 0. and len(backend.calls) == 2
    assert cache.report()["external_calls"] == 2


def test_explicit_physical_rejection_can_be_reused(tmp_path):
    backend = TruthDouble(tmp_path / "source", ["unphysical"])
    cache = CachedBackend(backend, tmp_path / "truth.sqlite")
    first = cache.evaluate(PARAMS, 0)
    second = cache.evaluate(PARAMS, 1)
    assert first["status"] == second["status"] == "unphysical"
    assert second["observables"] == {} and len(backend.calls) == 1


def test_parallel_results_preserve_order_and_point_isolation(tmp_path):
    backend = TruthDouble(tmp_path / "source")
    barrier = threading.Barrier(3, timeout=5)
    completed = []
    def overlap(parameters, eval_id):
        barrier.wait()
        time.sleep((3 - eval_id) * .02)
        completed.append(eval_id)
    backend.before_evaluate = overlap
    cache = CachedBackend(backend, tmp_path / "truth.sqlite", workers=3)
    points = [PARAMS | {"M1": 10. + i} for i in range(3)]
    answers = cache.evaluate_many(points, [0, 1, 2])
    assert completed != [0, 1, 2]  # actual out-of-order completion occurred
    dirs = [Path(a["provenance"]["workdir"]) for a in answers]
    assert [d.name for d in dirs] == ["point_000000", "point_000001", "point_000002"]
    assert len(set(dirs)) == 3
    assert [json.loads((d / "request.json").read_text()) for d in dirs] == points
    assert cache.report()["external_calls"] == 3


@pytest.mark.parametrize("ids", [[1, 1], [1]])
def test_bad_parallel_ids_fail_before_any_external_call(tmp_path, ids):
    backend = TruthDouble(tmp_path / "source")
    cache = CachedBackend(backend, tmp_path / "truth.sqlite", workers=2)
    with pytest.raises(ValueError, match="IDs"):
        cache.evaluate_many([PARAMS, PARAMS], ids)
    assert backend.calls == []


@pytest.mark.parametrize("change", ["config", "omega_driver"])
def test_mid_run_oracle_change_is_rejected_before_mixing_results(tmp_path, change):
    backend = TruthDouble(tmp_path / "source")
    cache = CachedBackend(backend, tmp_path / "truth.sqlite")
    def mutate(parameters, eval_id):
        path = backend.config_path if change == "config" else backend.paths[change]
        path.write_text(path.read_text() + " changed during call")
    backend.before_evaluate = mutate
    with pytest.raises(RuntimeError, match="changed|fingerprint|oracle|config"):
        cache.evaluate(PARAMS, 0)


def test_import_accepts_retained_truth_and_ignores_tool_error(tmp_path):
    backend = TruthDouble(tmp_path / "source", ["ok", "tool_error"])
    backend.evaluate(PARAMS, 0)
    failed_params = PARAMS | {"M1": 11.}
    backend.evaluate(failed_params, 1)
    cache = CachedBackend(backend, tmp_path / "truth.sqlite")
    cache.import_run(backend.root)
    result = cache.evaluate(PARAMS, 10)
    assert result["provenance"]["reuse"]["cache_hit"]
    cache.evaluate(failed_params, 11)
    assert len(backend.calls) == 3  # failed input had to call truth again


def test_import_rejects_other_oracle_fingerprint(tmp_path):
    backend = TruthDouble(tmp_path / "source")
    backend.evaluate(PARAMS, 0)
    backend.paths["omega_driver"].write_text("replacement binary")
    cache = CachedBackend(backend, tmp_path / "truth.sqlite")
    assert cache.import_run(backend.root) == 0
    cache.evaluate(PARAMS, 1)
    assert cache.report()["cache_hits"] == 0


@pytest.mark.parametrize("corruption", ["predicted_observables", "different_parameters", "missing_artifact"])
def test_journal_import_requires_matching_retained_truth(tmp_path, corruption):
    backend = TruthDouble(tmp_path / "source")
    truth = backend.evaluate(PARAMS, 0)
    # The import directory intentionally contains only a journal. A cache-hit
    # journal must be verified against its original retained truth artifacts.
    journal_root = tmp_path / "journal"
    journal_root.mkdir()
    row = deepcopy(truth) | {"parameters": dict(PARAMS)}
    if corruption == "predicted_observables":
        row["observables"]["omega"] = .13  # plausible NN prediction is not truth
    elif corruption == "different_parameters":
        row["parameters"]["M1"] = math.nextafter(10., math.inf)
    else:
        row["provenance"]["workdir"] = str(tmp_path / "does-not-exist")
    (journal_root / "records.jsonl").write_text(json.dumps(row) + "\n")
    cache = CachedBackend(backend, tmp_path / "truth.sqlite")
    if corruption == "missing_artifact":
        assert cache.import_run(journal_root) == 0
    else:
        with pytest.raises(ValueError, match="disagree"):
            cache.import_run(journal_root)
    cache.evaluate(row["parameters"], 1)
    assert cache.report()["cache_hits"] == 0


def test_journal_can_import_a_verified_cache_hit_record(tmp_path):
    backend = TruthDouble(tmp_path / "source")
    truth = backend.evaluate(PARAMS, 0)
    # Earlier real baseline artifacts have no embedded request hashes. The
    # journal can still verify them by comparing both retained artifact files.
    truth["provenance"].pop("input_parameters")
    truth["provenance"].pop("request_sha256")
    (Path(truth["provenance"]["workdir"]) / "evaluation.json").write_text(json.dumps(truth))
    journal_root = tmp_path / "journal"
    journal_root.mkdir()
    row = deepcopy(truth) | {"parameters": PARAMS, "seconds": 0.}
    row["provenance"]["reuse"] = {"cache_hit": True}
    (journal_root / "records.jsonl").write_text(json.dumps(row) + "\n")
    cache = CachedBackend(backend, tmp_path / "truth.sqlite")
    assert cache.import_run(journal_root) == 1
    assert cache.evaluate(PARAMS, 1)["provenance"]["reuse"]["cache_hit"]


def test_changed_binary_is_rejected_even_on_cache_hit(tmp_path):
    backend = TruthDouble(tmp_path / "source")
    cache = CachedBackend(backend, tmp_path / "truth.sqlite")
    cache.evaluate(PARAMS, 0)
    backend.paths["omega_driver"].write_text("changed after cache fill")
    with pytest.raises(RuntimeError, match="changed|fingerprint|oracle|binar"):
        cache.evaluate(PARAMS, 1)


def test_binary_changed_after_result_fingerprint_is_detected(tmp_path):
    backend = TruthDouble(tmp_path / "source")
    cache = CachedBackend(backend, tmp_path / "truth.sqlite")
    original_evaluate = backend.evaluate
    def change_after_snapshot(parameters, eval_id):
        result = original_evaluate(parameters, eval_id)
        backend.paths["fh_driver"].write_text("replaced during subprocess execution")
        return result  # carries the old pre-subprocess provenance
    backend.evaluate = change_after_snapshot
    with pytest.raises(RuntimeError, match="changed|fingerprint|oracle|binar"):
        cache.evaluate(PARAMS, 0)


def test_real_backend_pool_gives_subprocesses_separate_mo_work(tmp_path):
    # Execute tiny stand-ins to exercise the real per-point subprocess and
    # copytree paths. These executables do not calculate physics.
    fh = tmp_path / "fh"
    fh.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nimport json,sys,time\n"
                  "point=Path(sys.argv[3]).parent\n"
                  "time.sleep((3-int(point.name.split('_')[-1]))*.01)\n"
                  "Path(sys.argv[3]).write_text(json.dumps({'status':'ok','mh':5.,'mH':125.,'chi2_hs':90.}))\n"
                  "Path(sys.argv[4]).write_text('test-double SLHA')\n")
    mo = tmp_path / "mo"
    mo.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nimport json,os,sys\n"
                  "work=Path(os.environ['MO_WORK_DIR'])\n"
                  "assert (work/'template-marker').read_text()=='seed'\n"
                  "with (work/'owner').open('x') as f: f.write(str(Path(sys.argv[2]).parent))\n"
                  "Path(sys.argv[2]).write_text(json.dumps({'status':'ok','omega_h2':.12,'source_version':'4.3.2'}))\n")
    fh.chmod(0o755)
    mo.chmod(0o755)
    template = tmp_path / "template"
    template.mkdir()
    (template / "template-marker").write_text("seed")
    config = {"fh_driver": str(fh), "omega_driver": str(mo), "mo_work_template": str(template),
              "versions": {"FeynHiggs":"2.13.0", "HiggsBounds":"4.3.1",
                           "HiggsSignals":"1.4.0", "micrOMEGAs":"4.3.2"}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    backend = PhysicsBackend(config_path, tmp_path / "physics")
    cache = CachedBackend(backend, tmp_path / "cache.sqlite", workers=3)
    results = cache.evaluate_many([PARAMS | {"M1": 10.+i} for i in range(3)], [0,1,2])
    assert [r["status"] for r in results] == ["ok"] * 3
    for index, result in enumerate(results):
        point = Path(result["provenance"]["workdir"])
        assert point.name == f"point_{index:06d}"
        assert (point / "work/owner").read_text() == str(point)
    assert not (template / "owner").exists()


@pytest.mark.parametrize("filename", ["cards.py", "domain.py", "backend.py"])
def test_adapter_change_invalidates_cache_identity(tmp_path, filename):
    backend = TruthDouble(tmp_path / "source")
    first = CachedBackend(backend, tmp_path / "truth.sqlite")
    first.evaluate(PARAMS, 0)
    backend.adapter_paths[filename].write_text("new physics input implementation")
    second = CachedBackend(backend, tmp_path / "truth.sqlite")
    assert second.fingerprint != first.fingerprint
    assert second.import_run(backend.root) == 0
    second.evaluate(PARAMS, 1)
    assert second.report()["cache_hits"] == 0


def test_adapter_change_during_active_run_is_rejected(tmp_path):
    backend = TruthDouble(tmp_path / "source")
    cache = CachedBackend(backend, tmp_path / "truth.sqlite")
    cache.evaluate(PARAMS, 0)
    backend.adapter_paths["cards.py"].write_text("changed fixed MSSM mapping")
    with pytest.raises(RuntimeError, match="adapter|changed|oracle"):
        cache.evaluate(PARAMS, 1)


def test_parameter_card_preserves_adjacent_binary64_inputs(tmp_path):
    nearby = PARAMS | {"M1": math.nextafter(10., math.inf)}
    path = tmp_path / "feynhiggs.in"
    write_fh_card(path, nearby)
    rows = dict(line.split() for line in path.read_text().splitlines() if not line.startswith("#"))
    assert float(rows["Re(M_1)"]) == nearby["M1"]
    assert float(rows["Re(M_1)"]) != PARAMS["M1"]
