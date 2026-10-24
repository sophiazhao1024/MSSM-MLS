"""Actual FeynHiggs/HiggsBounds/HiggsSignals -> SLHA -> micrOMEGAs chain.

No approximate or synthetic physics fallback exists here. The C and Fortran
drivers are provided in physics/. Every point has its input, spectrum and logs.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from .cards import DEFAULT_OPTIONS, write_fh_card, write_options
from .domain import EXPECTED_TOOL_VERSIONS, PARAMETER_NAMES, parameter_dict, validate_observables


def read_json(path):
    """Reject NaN/Infinity, including nonstandard JSON emitted by broken tools."""
    def reject(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_sha256():
    """Include hard-coded physics mapping as well as external program hashes."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("backend.py", "cards.py", "domain.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


WARM_MANIFEST_NAME = ".warm-template.json"
MODEL_FILES = tuple(f"models/{name}1.mdl" for name in ("vars", "func", "prtcls", "lgrng", "extlib"))


def template_file_hashes(directory):
    """Inventory every retained template file, excluding only its manifest.

    A warm template is immutable. Point-local copies may generate more code,
    but those writes must never reach this shared source directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("Warm template directory is missing")
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Warm template must not contain symbolic links: {path}")
        if path.is_file() and path.relative_to(directory).as_posix() != WARM_MANIFEST_NAME:
            hashes[path.relative_to(directory).as_posix()] = sha256(path)
    for relative in MODEL_FILES:
        if relative not in hashes:
            raise ValueError(f"Warm template is missing model file {relative}")
    for name in ("tmp", "results", "so_generated"):
        if not (directory / name).is_dir():
            raise ValueError(f"Warm template is missing directory {name}")
    if not any(name.startswith("so_generated/") and name.endswith(".so") for name in hashes):
        raise ValueError("Warm template contains no compiled process library")
    return hashes


def mo_dependency_hashes(omega_driver):
    """Bind generated code to its inspected micrOMEGAs/CalcHEP build.

    The supplied executable lives at SOURCE_ROOT/MSSM/omega_driver. Generated
    process libraries depend on this retained installation; they are not a
    portable binary bundle and must be regenerated after build/model changes.
    """
    root = Path(omega_driver).resolve().parent.parent
    required = [
        root / "lib/micromegas.a", root / "MSSM/lib/aLib.a",
        root / "CalcHEP_src/lib/sqme_aux.so", root / "CalcHEP_src/bin/s_calchep",
        root / "CalcHEP_src/sbin/newProcess", root / "CalcHEP_src/FlagsForMake",
        root / "CalcHEP_src/FlagsForSh", root / "sources/omega.c",
        root / "CalcHEP_src/c_source/dynamicME/dynamic_cs.c",
        root / "CalcHEP_src/c_source/dynamicME/kin4.c",
    ]
    for path in required:
        if not path.is_file():
            raise ValueError(f"Missing micrOMEGAs build dependency: {path}")
    paths = set(required)
    # Include static/shared libraries and the executable generation helpers.
    for relative in ("CalcHEP_src/lib", "CalcHEP_src/bin", "CalcHEP_src/sbin"):
        for path in (root / relative).iterdir():
            if path.is_file():
                paths.add(path)
    return {str(path.resolve()): sha256(path) for path in sorted(paths)}


def warm_physics_identity(config, fh_driver, omega_driver):
    """Configuration identity with only template-location fields removed."""
    clean = {key: value for key, value in config.items() if key not in {
        "mo_work_template", "warm_template_manifest", "warm_template_manifest_sha256"}}
    clean["fh_driver"] = str(Path(fh_driver).resolve())
    clean["omega_driver"] = str(Path(omega_driver).resolve())
    return hashlib.sha256(json.dumps(clean, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_warm_template(config, config_path, template, fh_driver, omega_driver):
    """Return an audit result; a failed warm check makes doctor.ready false."""
    try:
        reference = config.get("warm_template_manifest")
        if not isinstance(reference, str) or not reference:
            raise ValueError("Warm template manifest reference is missing")
        manifest_path = Path(reference).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = Path(config_path).parent / manifest_path
        manifest_path = manifest_path.resolve()
        if manifest_path != Path(template).resolve() / WARM_MANIFEST_NAME:
            raise ValueError("Warm manifest must be stored inside its template")
        if config.get("warm_template_manifest_sha256") != sha256(manifest_path):
            raise ValueError("Warm template manifest hash is missing or changed")
        manifest = read_json(manifest_path)
        if manifest.get("schema_version") != 1 or manifest.get("micromegas_version") != "4.3.2":
            raise ValueError("Unsupported warm template schema or micrOMEGAs version")
        if manifest.get("omega_driver_sha256") != sha256(omega_driver):
            raise ValueError("Warm template was created by a different omega driver")
        identity = warm_physics_identity(config, fh_driver, omega_driver)
        if manifest.get("physics_config_sha256") != identity:
            raise ValueError("Warm template physics configuration differs")
        actual = template_file_hashes(template)
        if manifest.get("template_files_sha256") != actual:
            raise ValueError("Warm template files are missing, added or modified")
        if manifest.get("model_sha256") != {name: actual[name] for name in MODEL_FILES}:
            raise ValueError("Warm template model hashes are missing or inconsistent")
        if manifest.get("build_dependencies_sha256") != mo_dependency_hashes(omega_driver):
            raise ValueError("Warm template CalcHEP libraries/tools/sources differ")
        return {"valid": True, "path": str(manifest_path), "sha256": sha256(manifest_path),
                "compiled_process_libraries": sum(name.startswith("so_generated/") and name.endswith(".so") for name in actual)}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return {"valid": False, "error": str(exc)}


class PhysicsBackend:
    """Truth evaluator consumed by both baseline and optimized scanners.

    Paths in the configuration are relative to that configuration file.
    Work directories are created lazily, so a scanner can own the output root.
    """
    def __init__(self, config_path, work_root, *, allow_version_substitution=False):
        self.config_path = Path(config_path).resolve()
        self.config = read_json(self.config_path)
        self.work_root = Path(work_root).resolve()
        if any(character.isspace() for character in str(self.work_root)):
            raise ValueError("Use a run directory without whitespace: legacy CalcHEP generates shell/make paths")
        base = self.config_path.parent
        def resolve(name):
            value = Path(self.config[name]).expanduser()
            return (base / value).resolve() if not value.is_absolute() else value.resolve()
        self.fh_driver, self.omega_driver = resolve("fh_driver"), resolve("omega_driver")
        self.mo_work_template = resolve("mo_work_template")
        for path in (self.fh_driver, self.omega_driver, self.mo_work_template):
            if any(character.isspace() for character in str(path)):
                raise ValueError("Legacy physics tool paths must not contain whitespace")
        self.versions = self.config["versions"]
        self.version_differences = {k: {"paper": v, "actual": self.versions.get(k)}
                                    for k, v in EXPECTED_TOOL_VERSIONS.items()
                                    if self.versions.get(k) != v}
        if self.version_differences and not allow_version_substitution:
            raise ValueError("Physics versions differ from paper: " + json.dumps(self.version_differences)
                             + ". Use --allow-version-substitution only for a labelled substitute run.")
        flags = self.config.get("options", {}).get("fh_flags", DEFAULT_OPTIONS["fh_flags"])
        if isinstance(flags, str) and len(flags) == 10 and flags[9] == "0":
            raise ValueError("Model-4 uses MHp input: tlCplxApprox=0 (tenth FeynHiggs flag) is inconsistent; use a justified nonzero setting")
        self.timeout = float(self.config.get("timeout_seconds", 300))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.mo = {"fast": 1, "beps": 1e-5, "vw": 0, "vz": 0} | self.config.get("micromegas", {})
        if set(self.mo) != {"fast", "beps", "vw", "vz"}:
            raise ValueError("Unknown micrOMEGAs setting")
        if type(self.mo["fast"]) is not int or self.mo["fast"] not in {-1, 0, 1}:
            raise ValueError("fast must be -1, 0 or 1")
        if self.versions.get("micrOMEGAs") == "4.3.2" and self.mo["fast"] == -1:
            raise ValueError("fast=-1 was added after micrOMEGAs 4.3.2; historical profile supports 0/1")
        if any(type(self.mo[k]) is not int or self.mo[k] not in {0, 1, 2} for k in ("vw", "vz")):
            raise ValueError("vw and vz must be 0, 1 or 2")
        if not isinstance(self.mo["beps"], (int, float)) or not 0 < self.mo["beps"] < 1:
            raise ValueError("beps must be in (0,1)")
        self.environment = os.environ.copy()
        self.environment.update({str(k): str(v) for k, v in self.config.get("environment", {}).items()})
        self.binary_hashes = {}

    def doctor(self):
        """Check files and provenance; this is NOT a successful physics run."""
        checks = {}
        for name, path in (("fh_driver", self.fh_driver), ("omega_driver", self.omega_driver)):
            checks[name] = {"path": str(path), "executable": path.is_file() and os.access(path, os.X_OK)}
            if checks[name]["executable"]:
                checks[name]["sha256"] = sha256(path)
                self.binary_hashes[name] = checks[name]["sha256"]
        checks["mo_work_template"] = {"path": str(self.mo_work_template), "exists": self.mo_work_template.is_dir()}
        warm_valid = True
        if (self.config.get("warm_template_manifest") is not None
            or (self.mo_work_template / WARM_MANIFEST_NAME).exists()):
            checks["warm_template"] = verify_warm_template(
                self.config, self.config_path, self.mo_work_template, self.fh_driver, self.omega_driver)
            warm_valid = checks["warm_template"]["valid"]
        return {"ready": all(checks[k]["executable"] for k in ("fh_driver", "omega_driver"))
                         and checks["mo_work_template"]["exists"] and warm_valid,
                "checks": checks, "versions": self.versions,
                "paper_version_match": not self.version_differences,
                "version_differences": self.version_differences,
                "adapter_sha256": adapter_sha256(),
                "config_sha256": sha256(self.config_path)}

    def _run(self, argv, directory, stage, provenance):
        """No shell interpolation; child-process streams go to per-point logs."""
        started = time.perf_counter()
        with (directory / f"{stage}.stdout.log").open("wb") as out, (directory / f"{stage}.stderr.log").open("wb") as err:
            environment = self.environment.copy()
            if stage == "omega":
                environment["MO_WORK_DIR"] = str(directory / "work")
            completed = subprocess.run([str(v) for v in argv], cwd=directory,
                                       env=environment, stdout=out, stderr=err,
                                       timeout=self.timeout, check=False)
        provenance.setdefault("commands", []).append({"stage": stage, "argv": list(map(str, argv)),
            "returncode": completed.returncode, "seconds": time.perf_counter() - started})
        return completed.returncode

    def evaluate(self, parameters, eval_id):
        """Return ok/unphysical/tool_error, four observable values and provenance.

        ok means calculable physical outputs, even if all experimental cuts fail.
        Only an explicit verified physical rejection can yield unphysical.
        Missing files, timeouts, bad numbers and unexplained errors are tool_error.
        """
        started = time.perf_counter()
        result = {"status": "tool_error", "observables": {}, "reason": "", "provenance": {
            "versions": self.versions, "paper_version_match": not self.version_differences,
            "version_differences": self.version_differences, "config_sha256": sha256(self.config_path),
            "adapter_sha256": adapter_sha256()}}
        directory = None
        try:
            if type(eval_id) is not int or eval_id < 0:
                raise ValueError("eval_id must be a nonnegative integer")
            clean = parameter_dict([parameters[k] for k in PARAMETER_NAMES])
            ready = self.doctor()
            if not ready["ready"]:
                raise FileNotFoundError("Physics toolchain is not ready; run doctor")
            result["provenance"]["binary_sha256"] = dict(self.binary_hashes)
            result["provenance"]["input_parameters"] = clean
            target = self.work_root / f"point_{eval_id:06d}"
            target.mkdir(parents=True, exist_ok=False)
            directory = target  # do not overwrite logs of an existing point
            result["provenance"]["workdir"] = str(directory)
            save_json(directory / "request.json", clean)
            result["provenance"]["request_sha256"] = sha256(directory / "request.json")
            write_fh_card(directory / "feynhiggs.in", clean, top_mass=self.config.get("top_mass", 172.))
            options = write_options(directory / "options.nml", self.config.get("options", {}))
            result["provenance"]["physics_options"] = options
            result["provenance"]["top_mass_GeV"] = self.config.get("top_mass", 172.)
            fh_json, spectrum = directory / "higgs.json", directory / "spectrum.slha"
            code = self._run([self.fh_driver, directory / "feynhiggs.in", directory / "options.nml",
                              fh_json, spectrum], directory, "higgs", result["provenance"])
            fh = read_json(fh_json)
            result["provenance"]["higgs"] = fh
            if code != 0 or fh.get("status") != "ok":
                if code == 0 and fh.get("status") == "unphysical":
                    result.update(status="unphysical", reason=fh.get("reason") or fh.get("physics_reason") or "Explicit Higgs-sector rejection")
                else:
                    raise RuntimeError(f"Higgs driver failure (exit {code}): {fh}")
            else:
                if not spectrum.is_file() or spectrum.stat().st_size == 0:
                    raise ValueError("Higgs driver produced no SLHA spectrum")
                # CalcHEP dynamically generates process libraries. Every point
                # gets its own mutable workspace, never shared by concurrent calls.
                shutil.copytree(self.mo_work_template, directory / "work")
                mo_json = directory / "omega.json"
                args = [self.omega_driver, spectrum, mo_json] + [str(self.mo[k]) for k in ("fast", "beps", "vw", "vz")]
                code = self._run(args, directory, "omega", result["provenance"])
                mo = read_json(mo_json)
                result["provenance"]["micromegas"] = mo
                if mo.get("source_version") != self.versions["micrOMEGAs"]:
                    raise ValueError("micrOMEGAs binary version disagrees with backend configuration")
                if code == 0 and mo.get("status") == "unphysical":
                    result.update(status="unphysical", reason=mo.get("reason", "Explicit dark-matter rejection"))
                elif code != 0 or mo.get("status") != "ok":
                    raise RuntimeError(f"micrOMEGAs failure (exit {code}): {mo}")
                else:
                    values = {"mh": fh["mh"], "mH": fh["mH"], "chi2_hs": fh["chi2_hs"], "omega": mo["omega_h2"]}
                    validate_observables(values)
                    result.update(status="ok", observables=values)
                    # HiggsBounds is reported as a diagnostic. Eq.(7) does not
                    # contain an extra HB veto; adding one changes the target.
                    result["diagnostics"] = {k: v for k, v in fh.items() if k not in {"status", "mh", "mH", "chi2_hs"}}
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
            result.update(status="tool_error", observables={}, reason=f"{type(exc).__name__}: {exc}")
        result["seconds"] = time.perf_counter() - started
        if directory is not None:
            save_json(directory / "evaluation.json", result)
        return result
