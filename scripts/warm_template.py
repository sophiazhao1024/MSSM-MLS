#!/usr/bin/env python3
"""Make an immutable, audited CalcHEP template from one completed true point.

Only generated matrix-element code is reused. Each subsequent point gets an
independent copy, imports its own SLHA parameters and calls darkOmega again.
This utility never changes the source point, physics sources or executables.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from mssm_mls.backend import (  # noqa: E402
    MODEL_FILES, WARM_MANIFEST_NAME, PhysicsBackend, mo_dependency_hashes,
    read_json, sha256, template_file_hashes, warm_physics_identity,
)
from mssm_mls.domain import PARAMETER_NAMES, parameter_dict, validate_observables  # noqa: E402


def _excluded(relative: Path) -> bool:
    """Drop known transient coordination/log files, never required libraries."""
    return any(part in {WARM_MANIFEST_NAME, "LOCK", "lock_"}
               or part.endswith(".log")
               or re.fullmatch(r"_.+_[0-9]+_", part)
               for part in relative.parts)


def _work_inventory(work: Path):
    """Reject aliases; record retained content and an explicit exclusion list."""
    retained, excluded = {}, []
    for path in sorted(work.rglob("*")):
        relative = path.relative_to(work)
        if path.is_symlink():
            raise ValueError(f"Source work directory contains a symbolic link: {relative}")
        if _excluded(relative):
            excluded.append(relative.as_posix())
        elif path.is_file():
            if path.name in {"omega.json", "evaluation.json", "request.json", "spectrum.slha"}:
                raise ValueError(f"Unexpected point result inside CalcHEP work: {relative}")
            retained[relative.as_posix()] = sha256(path)
    return retained, excluded


def _write_exclusive(path: Path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # Exclusive creation prevents overwrite; remove only the file we created
    # if the write itself fails (for example because storage is exhausted).
    stream = path.open("x", encoding="utf-8")
    try:
        with stream:
            stream.write(text)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _check_source(backend: PhysicsBackend, point: Path, doctor):
    """Cross-check saved truth, process result and the exact configured oracle."""
    evaluation = read_json(point / "evaluation.json")
    if evaluation.get("status") != "ok":
        raise ValueError("Source point must have a successful true evaluation")
    validate_observables(evaluation.get("observables"))
    provenance = evaluation["provenance"]
    if Path(provenance["workdir"]).resolve() != point:
        raise ValueError("Source point provenance workdir differs from --point-dir")
    if provenance.get("config_sha256") != doctor["config_sha256"]:
        raise ValueError("Source point was evaluated with a different configuration")
    if provenance.get("binary_sha256") != backend.binary_hashes:
        raise ValueError("Source point was evaluated with different physics executables")
    if provenance.get("versions") != backend.versions:
        raise ValueError("Source point tool versions differ")
    request = read_json(point / "request.json")
    parameters = parameter_dict([request[k] for k in PARAMETER_NAMES])
    if request != parameters:
        raise ValueError("Source request is not the exact model parameter mapping")
    if "request_sha256" in provenance and provenance["request_sha256"] != sha256(point / "request.json"):
        raise ValueError("Source request hash differs from its provenance")
    if "input_parameters" in provenance and provenance["input_parameters"] != parameters:
        raise ValueError("Source input parameters differ from its request")
    mo = read_json(point / "omega.json")
    if (mo.get("status") != "ok" or mo.get("source_version") != "4.3.2"
        or mo.get("settings") != backend.mo
        or mo.get("omega_h2") != evaluation["observables"]["omega"]
        or mo != provenance.get("micromegas")):
        raise ValueError("Source micrOMEGAs result/version/settings do not match saved truth")
    fh = read_json(point / "higgs.json")
    if (fh.get("status") != "ok" or fh != provenance.get("higgs")
        or any(fh.get(name) != evaluation["observables"][name] for name in ("mh", "mH", "chi2_hs"))):
        raise ValueError("Source Higgs result does not match saved truth")
    spectrum = point / "spectrum.slha"
    if not spectrum.is_file() or spectrum.stat().st_size == 0:
        raise ValueError("Source point has no SLHA spectrum")
    stages = {entry.get("stage"): entry for entry in provenance.get("commands", [])}
    for name, executable in (("higgs", backend.fh_driver), ("omega", backend.omega_driver)):
        entry = stages.get(name, {})
        argv = entry.get("argv", [])
        if entry.get("returncode") != 0 or not argv or Path(argv[0]).resolve() != executable:
            raise ValueError(f"Source point lacks a successful recorded {name} process")
    source_models = {name: sha256(point / "work" / name) for name in MODEL_FILES}
    current_models = {name: sha256(backend.mo_work_template / name) for name in MODEL_FILES}
    if source_models != current_models:
        raise ValueError("Source point model files differ from the configured model template")
    return provenance, parameters


def create_warm_template(config_path, point_dir, output, output_config):
    """Create new files or fail; never overwrite a template, config or point."""
    config_path, point, output, output_config = map(
        lambda value: Path(value).expanduser().resolve(),
        (config_path, point_dir, output, output_config))
    for path in (point, output, output_config):
        if any(character.isspace() for character in str(path)):
            raise ValueError("Use paths without whitespace for legacy CalcHEP")
    if not point.is_dir():
        raise ValueError("--point-dir must be a completed point directory")
    if output.exists() or output_config.exists():
        raise FileExistsError("The output template and output configuration must both be new")
    if output.is_relative_to(point) or point.is_relative_to(output):
        raise ValueError("Output template must be outside the source point directory")
    if output_config.is_relative_to(output) or output_config.is_relative_to(point):
        raise ValueError("Output configuration must be outside the template and source point")
    if not output.parent.is_dir() or not output_config.parent.is_dir():
        raise ValueError("Create output parent directories first")

    backend = PhysicsBackend(config_path, output.parent / "unused-warm-validation")
    doctor = backend.doctor()
    if not doctor["ready"]:
        raise ValueError("Backend doctor is not ready: " + json.dumps(doctor["checks"]))
    if backend.versions.get("micrOMEGAs") != "4.3.2":
        raise ValueError("Warm template creation currently supports micrOMEGAs 4.3.2 only")
    provenance, parameters = _check_source(backend, point, doctor)
    work = point / "work"
    retained, excluded = _work_inventory(work)
    dependencies = mo_dependency_hashes(backend.omega_driver)
    artifact_names = ("evaluation.json", "request.json", "higgs.json", "omega.json", "spectrum.slha")
    artifact_hashes = {name: sha256(point / name) for name in artifact_names}
    created_output = created_config = False
    try:
        # Reserve our directory; copy2 (copytree default) preserves model mtimes,
        # which the legacy CalcHEP loader checks before reusing process code.
        output.mkdir()
        created_output = True
        def ignore(directory, names):
            return [name for name in names if _excluded((Path(directory) / name).relative_to(work))]
        shutil.copytree(work, output, dirs_exist_ok=True, ignore=ignore)
        copied = template_file_hashes(output)
        if copied != retained or _work_inventory(work)[0] != retained:
            raise ValueError("Source CalcHEP workspace changed while it was copied")
        if artifact_hashes != {name: sha256(point / name) for name in artifact_names}:
            raise ValueError("Source point artifacts changed while the template was copied")
        if (doctor["config_sha256"] != sha256(config_path)
            or backend.binary_hashes != {"fh_driver": sha256(backend.fh_driver), "omega_driver": sha256(backend.omega_driver)}
            or dependencies != mo_dependency_hashes(backend.omega_driver)):
            raise ValueError("Physics configuration/build changed while the template was copied")
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "micromegas_version": "4.3.2",
            "reuse_scope": "Generated matrix-element code only; import new SLHA and recompute darkOmega for every point",
            "source_point": str(point),
            "source_artifacts_sha256": artifact_hashes,
            "source_provenance": {key: provenance[key] for key in (
                "workdir", "versions", "config_sha256", "binary_sha256", "adapter_sha256",
                "request_sha256", "input_parameters", "paper_version_match") if key in provenance},
            "source_input_parameters": parameters,
            "source_config": str(config_path),
            "source_adapter_note": "Recorded if present; no observables are reused. Configuration and both executable hashes must match.",
            "omega_driver_sha256": sha256(backend.omega_driver),
            "physics_config_sha256": warm_physics_identity(backend.config, backend.fh_driver, backend.omega_driver),
            "model_sha256": {name: copied[name] for name in MODEL_FILES},
            "template_files_sha256": copied,
            "build_dependencies_sha256": dependencies,
            "dependency_hash_scope": "Current retained build libraries, generators and relevant sources; old point logs did not record historical dependency hashes",
            "excluded_source_entries": excluded,
            "exclusion_rule": "Only previous warm manifests, LOCK/lock_, *.log and generated _name_PID_ temporary directories; no result data is copied",
        }
        manifest_path = output / WARM_MANIFEST_NAME
        _write_exclusive(manifest_path, manifest)
        config = dict(backend.config)
        config.update(fh_driver=str(backend.fh_driver), omega_driver=str(backend.omega_driver),
                      mo_work_template=str(output), warm_template_manifest=str(manifest_path),
                      warm_template_manifest_sha256=sha256(manifest_path))
        _write_exclusive(output_config, config)
        created_config = True
        final = PhysicsBackend(output_config, output.parent / "unused-warm-validation").doctor()
        if not final["ready"]:
            raise ValueError("New template failed validation: " + json.dumps(final["checks"]))
        return {"template": str(output), "configuration": str(output_config),
                "manifest": str(manifest_path), "validation": final["checks"]["warm_template"]}
    except Exception:
        if created_config:
            output_config.unlink()
        if created_output:
            shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, help="Exact backend config that produced the point")
    parser.add_argument("--point-dir", required=True, help="Completed successful physics/point_N directory")
    parser.add_argument("--output", required=True, help="New immutable template directory")
    parser.add_argument("--output-config", required=True, help="New backend configuration file")
    args = parser.parse_args()
    try:
        result = create_warm_template(args.backend, args.point_dir, args.output, args.output_config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"warm_template: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
