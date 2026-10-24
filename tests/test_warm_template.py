"""Integrity contracts using explicit file/process-result doubles, not physics.

Actual independent SLHA fresh/warm numerical equivalence is recorded separately
in mssm-reproduction-work/mo_warm_check/report.json.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from mssm_mls.backend import (MODEL_FILES, PhysicsBackend, WARM_MANIFEST_NAME,
                              read_json, save_json, sha256)
from mssm_mls.domain import EXPECTED_TOOL_VERSIONS
from mssm_mls.execution import CachedBackend

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/warm_template.py"
SPEC = importlib.util.spec_from_file_location("warm_template_script", SCRIPT)
WARM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WARM)


@pytest.fixture
def oracle_files(tmp_path):
    """A tiny pretend installation; no generated library is ever executed."""
    root = tmp_path / "source"
    required = (
        "MSSM/omega_driver", "fh_driver", "lib/micromegas.a", "MSSM/lib/aLib.a",
        "CalcHEP_src/lib/sqme_aux.so", "CalcHEP_src/bin/s_calchep", "CalcHEP_src/sbin/newProcess",
        "CalcHEP_src/FlagsForMake", "CalcHEP_src/FlagsForSh", "sources/omega.c",
        "CalcHEP_src/c_source/dynamicME/dynamic_cs.c", "CalcHEP_src/c_source/dynamicME/kin4.c")
    for name in required:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Software test double: " + name)
    fh, mo = root / "fh_driver", root / "MSSM/omega_driver"
    fh.chmod(0o755)
    mo.chmod(0o755)
    template = root / "MSSM/work"
    for name in MODEL_FILES:
        path = template / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Fixed test model " + name)
    for name in ("tmp", "results", "so_generated"):
        (template / name).mkdir()
    (template / "work_aux.a").write_bytes(b"test archive")
    config = tmp_path / "backend.json"
    save_json(config, {"fh_driver": str(fh), "omega_driver": str(mo),
                       "mo_work_template": str(template), "versions": EXPECTED_TOOL_VERSIONS,
                       "options": {"fh_flags": "4002420111"}})
    backend = PhysicsBackend(config, tmp_path / "points")
    doctor = backend.doctor()
    assert doctor["ready"]
    point = tmp_path / "points/point_000000"
    point.mkdir(parents=True)
    import shutil
    shutil.copytree(template, point / "work")
    (point / "work/so_generated/test_process.so").write_bytes(b"never executed")
    (point / "work/LOCK").write_text("transient lock")
    (point / "work/process.log").write_text("transient log")
    request = {"M1": 10., "mu": 8000., "tan_beta": 9.7}
    save_json(point / "request.json", request)
    (point / "spectrum.slha").write_text("BLOCK MASS\n 25 40.0\n")
    fh_result = {"status": "ok", "mh": 40., "mH": 125., "chi2_hs": 100.}
    mo_result = {"status": "ok", "source_version": "4.3.2", "omega_h2": .1186,
                 "settings": backend.mo}
    save_json(point / "higgs.json", fh_result)
    save_json(point / "omega.json", mo_result)
    result = {"status": "ok", "observables": {"mh": 40., "mH": 125., "chi2_hs": 100., "omega": .1186},
              "provenance": {"workdir": str(point), "config_sha256": doctor["config_sha256"],
                             "binary_sha256": dict(backend.binary_hashes), "versions": backend.versions,
                             "input_parameters": request, "request_sha256": sha256(point / "request.json"),
                             "adapter_sha256": doctor["adapter_sha256"], "higgs": fh_result,
                             "micromegas": mo_result,
                             "commands": [{"stage": "higgs", "argv": [str(fh)], "returncode": 0},
                                          {"stage": "omega", "argv": [str(mo)], "returncode": 0}]}}
    save_json(point / "evaluation.json", result)
    return {"config": config, "point": point, "root": root, "fh": fh, "mo": mo,
            "cold": template, "output": tmp_path / "warm", "newconfig": tmp_path / "warm.json"}


def create(files):
    return WARM.create_warm_template(files["config"], files["point"], files["output"], files["newconfig"])


def doctor(files):
    return PhysicsBackend(files["newconfig"], files["output"].parent / "newpoints").doctor()


def test_template_copies_code_and_provenance_without_point_observables(oracle_files):
    f = oracle_files
    source_hashes = {str(path): sha256(path) for path in f["point"].rglob("*") if path.is_file()}
    answer = create(f)
    assert answer["validation"]["valid"]
    assert doctor(f)["ready"]
    manifest = read_json(f["output"] / WARM_MANIFEST_NAME)
    assert manifest["model_sha256"].keys() == set(MODEL_FILES)
    assert manifest["omega_driver_sha256"] == sha256(f["mo"])
    assert "so_generated/test_process.so" in manifest["template_files_sha256"]
    assert "work_aux.a" in manifest["template_files_sha256"]  # must not drop required archives
    assert set(manifest["excluded_source_entries"]) == {"LOCK", "process.log"}
    assert not (f["output"] / "LOCK").exists()
    assert not (f["output"] / "omega.json").exists()
    assert "omega_h2" not in json.dumps(manifest)
    for name in MODEL_FILES:
        assert (f["output"] / name).stat().st_mtime_ns == (f["point"] / "work" / name).stat().st_mtime_ns
    assert source_hashes == {str(path): sha256(path) for path in f["point"].rglob("*") if path.is_file()}


@pytest.mark.parametrize("which", ["fh", "mo", "config"])
def test_creation_rejects_another_binary_or_configuration(oracle_files, which):
    f = oracle_files
    with f[which].open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="different"):
        create(f)
    assert not f["output"].exists()
    assert not f["newconfig"].exists()


@pytest.mark.parametrize("what", ["status", "omega", "request", "returncode", "models"])
def test_creation_rejects_inconsistent_or_failed_source_point(oracle_files, what):
    f = oracle_files
    evaluation = read_json(f["point"] / "evaluation.json")
    if what == "status":
        evaluation["status"] = "tool_error"
    elif what == "omega":
        evaluation["observables"]["omega"] = .2
    elif what == "request":
        save_json(f["point"] / "request.json", {"M1": 11., "mu": 8000., "tan_beta": 9.7})
    elif what == "returncode":
        evaluation["provenance"]["commands"][1]["returncode"] = 1
    elif what == "models":
        (f["point"] / "work" / MODEL_FILES[0]).write_text("different model")
    save_json(f["point"] / "evaluation.json", evaluation)
    with pytest.raises(ValueError):
        create(f)
    assert not f["output"].exists()


@pytest.mark.parametrize("kind", ["model", "library", "newfile", "missingfile", "dependency", "driver", "manifest", "missinghash", "settings"])
def test_doctor_refuses_changed_or_incomplete_warm_template(oracle_files, kind):
    f = oracle_files
    create(f)
    if kind == "model":
        (f["output"] / MODEL_FILES[0]).write_text("corrupt model")
    elif kind == "library":
        (f["output"] / "so_generated/test_process.so").write_bytes(b"corrupt library")
    elif kind == "newfile":
        (f["output"] / "unrecorded").write_text("added")
    elif kind == "missingfile":
        (f["output"] / "work_aux.a").unlink()
    elif kind == "dependency":
        (f["root"] / "CalcHEP_src/lib/sqme_aux.so").write_bytes(b"new build")
    elif kind == "driver":
        f["mo"].write_text("different executable")
    elif kind == "manifest":
        (f["output"] / WARM_MANIFEST_NAME).unlink()
    else:
        config = read_json(f["newconfig"])
        if kind == "missinghash":
            del config["warm_template_manifest_sha256"]
        elif kind == "settings":
            config["top_mass"] = 173.
        save_json(f["newconfig"], config)
    result = doctor(f)
    assert not result["ready"]
    assert not result["checks"]["warm_template"]["valid"]


@pytest.mark.parametrize("already_exists", ["output", "newconfig"])
def test_never_overwrites_outputs(oracle_files, already_exists):
    f = oracle_files
    target = f[already_exists]
    if already_exists == "output":
        target.mkdir()
        target = target / "precious.txt"
    target.write_text("keep me")
    with pytest.raises(FileExistsError):
        create(f)
    assert target.read_text() == "keep me"


def test_no_shared_mutable_symlinks(oracle_files):
    f = oracle_files
    (f["point"] / "work/alias").symlink_to(f["cold"] / "work_aux.a")
    with pytest.raises(ValueError, match="symbolic link"):
        create(f)
    assert not f["output"].exists()


def test_copy_failure_removes_only_its_new_output(oracle_files, monkeypatch):
    f = oracle_files
    original = WARM.shutil.copytree
    def failed_copy(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("simulated copy failure")
    monkeypatch.setattr(WARM.shutil, "copytree", failed_copy)
    with pytest.raises(OSError, match="simulated"):
        create(f)
    assert not f["output"].exists()
    assert not f["newconfig"].exists()
    assert (f["point"] / "evaluation.json").is_file()


@pytest.mark.parametrize("when", ["before_cached_hit", "during_evaluation"])
def test_cache_refuses_warm_template_drift(oracle_files, monkeypatch, when):
    f = oracle_files
    create(f)
    backend = PhysicsBackend(f["newconfig"], f["output"].parent / "newpoints")
    cache = CachedBackend(backend, f["output"].parent / "truth.sqlite")
    params = read_json(f["point"] / "request.json")
    result = read_json(f["point"] / "evaluation.json")
    # This is a software double. Its provenance is bound to the new warm
    # configuration solely to exercise storage validation; no physics runs.
    result["provenance"]["config_sha256"] = cache.config_hash
    result["provenance"]["adapter_sha256"] = cache.adapter_hash
    def drift():
        (f["output"] / "so_generated/test_process.so").write_bytes(b"changed")
    if when == "before_cached_hit":
        assert cache._store(cache.key(params), result)
        drift()
        monkeypatch.setattr(backend, "evaluate", lambda *_: pytest.fail("must not evaluate"))
    else:
        def evaluate(*_):
            drift()
            return result
        monkeypatch.setattr(backend, "evaluate", evaluate)
    with pytest.raises(RuntimeError, match="no longer ready"):
        cache.evaluate(params, 0)
    assert cache.counts["cache_hits"] == 0
    if when == "during_evaluation":
        with cache._connect() as db:
            assert db.execute("SELECT COUNT(*) FROM truth").fetchone()[0] == 0


def test_model4_backend_refuses_inconsistent_charged_higgs_flag(oracle_files):
    f = oracle_files
    config = read_json(f["config"])
    config["options"]["fh_flags"] = "4002020110"
    save_json(f["config"], config)
    with pytest.raises(ValueError, match="MHp input.*tlCplxApprox=0"):
        PhysicsBackend(f["config"], f["output"])
