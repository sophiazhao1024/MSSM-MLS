"""Exact-input truth reuse and isolated concurrent external evaluations.

Cache keys include binary hashes, the complete backend configuration hash and
float.hex() inputs. No rounded point and no failed calculation is cached.
Restart means deterministic replay into a NEW run directory with cached truth;
it does not claim instruction-level optimizer or process-state restoration.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from .backend import read_json, sha256
from .domain import PARAMETER_NAMES, parameter_dict, validate_observables


class CachedBackend:
    def __init__(self, backend, cache_path, *, workers=1):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.backend, self.workers = backend, workers
        self.cache_path = Path(cache_path).resolve()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.oracle = backend.doctor()
        if not self.oracle["ready"]:
            raise ValueError("Cannot fingerprint an unavailable physics backend")
        self.config_hash = self.oracle["config_sha256"]
        self.adapter_hash = self.oracle.get("adapter_sha256")
        self.binary_hashes = {k: self.oracle["checks"][k]["sha256"] for k in ("fh_driver", "omega_driver")}
        identity = {"config": self.config_hash, "binaries": self.binary_hashes, "adapter": self.adapter_hash}
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.counts = {"requests": 0, "cache_hits": 0, "external_calls": 0, "imported": 0}
        self.lock = threading.Lock()
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS truth (key TEXT PRIMARY KEY, answer TEXT NOT NULL)")

    def _connect(self):
        return sqlite3.connect(self.cache_path, timeout=60)

    def key(self, parameters):
        clean = parameter_dict([parameters[k] for k in PARAMETER_NAMES])
        content = {"oracle": self.fingerprint, "x": [clean[k].hex() for k in PARAMETER_NAMES]}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def _compatible(self, result):
        provenance = result.get("provenance", {})
        if provenance.get("config_sha256") != self.config_hash or provenance.get("binary_sha256") != self.binary_hashes:
            return False
        if provenance.get("adapter_sha256") != self.adapter_hash:
            return False
        if result.get("status") == "ok":
            validate_observables(result.get("observables"))
            return True
        return result.get("status") == "unphysical"

    def _store(self, key, result):
        if not self._compatible(result):
            return False
        serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO truth VALUES (?,?)", (key, serialized))
        return True

    def import_run(self, run_directory):
        """Import completed point artifacts, including a partially finished run.

        Each stored request is checked against its result provenance. Stale
        versions/settings/binaries and all technical failures are excluded.
        No NN-predicted value is accepted as a true label.
        """
        imported = 0
        run_directory = Path(run_directory)
        for result_file in sorted((run_directory / "physics").glob("point_*/evaluation.json")):
            request_file = result_file.with_name("request.json")
            if request_file.exists():
                result = read_json(result_file)
                params = read_json(request_file)
                provenance = result.get("provenance", {})
                # Orphan artifacts need an embedded request identity. Earlier
                # journals can still be imported below through a double check.
                if (provenance.get("input_parameters") == params
                    and provenance.get("request_sha256") == sha256(request_file)
                    and self._store(self.key(params), result)):
                    imported += 1
        # Cached points may exist only in the run's journal, whose provenance
        # still points to the original retained external-tool artifacts.
        journal = run_directory / "records.jsonl"
        if journal.exists():
            for line in journal.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                provenance = row.get("provenance", {})
                if not provenance.get("workdir"):
                    continue
                original = Path(provenance["workdir"])
                if not (original / "evaluation.json").is_file() or not (original / "request.json").is_file():
                    continue
                actual_params = read_json(original / "request.json")
                actual_result = read_json(original / "evaluation.json")
                if self.key(row["parameters"]) != self.key(actual_params):
                    raise ValueError("Journal parameters disagree with original physics request")
                if any(row.get(k) != actual_result.get(k) for k in ("status", "observables")):
                    raise ValueError("Journal observations disagree with original physics result")
                if not self._compatible({k: row.get(k) for k in ("status", "observables", "provenance")}):
                    continue
                if self._store(self.key(actual_params), actual_result):
                    imported += 1
        self.counts["imported"] += imported  # accepted records, may include duplicate keys
        return imported

    def evaluate(self, parameters, eval_id):
        if sha256(self.backend.config_path) != self.config_hash:
            raise RuntimeError("Physics configuration changed during this run")
        current = self.backend.doctor()
        if not current.get("ready", False):
            raise RuntimeError("Physics backend is no longer ready; template/build validation failed")
        if current.get("adapter_sha256") != self.adapter_hash:
            raise RuntimeError("Physics adapter changed during this run")
        if any(current["checks"][k].get("sha256") != digest for k, digest in self.binary_hashes.items()):
            raise RuntimeError("Physics binaries changed during this run")
        key = self.key(parameters)
        with self.lock:
            self.counts["requests"] += 1
        with self._connect() as db:
            hit = db.execute("SELECT answer FROM truth WHERE key=?", (key,)).fetchone()
        if hit is not None:
            result = json.loads(hit[0])
            result.setdefault("provenance", {})["reuse"] = {
                "cache_hit": True, "oracle_fingerprint": self.fingerprint,
                "source_seconds": result.get("seconds"), "key": key}
            result["seconds"] = 0.
            with self.lock:
                self.counts["cache_hits"] += 1
            return result
        result = self.backend.evaluate(parameters, eval_id)
        with self.lock:
            self.counts["external_calls"] += 1
        if sha256(self.backend.config_path) != self.config_hash or result.get("provenance", {}).get("config_sha256") != self.config_hash:
            raise RuntimeError("Physics configuration changed during this run")
        current = self.backend.doctor()
        if not current.get("ready", False):
            raise RuntimeError("Physics backend is no longer ready after evaluation; refusing cache write")
        if current.get("adapter_sha256") != self.adapter_hash or result.get("provenance", {}).get("adapter_sha256") != self.adapter_hash:
            raise RuntimeError("Physics adapter changed during this run")
        if any(current["checks"][k].get("sha256") != digest for k, digest in self.binary_hashes.items()):
            raise RuntimeError("Physics binaries changed during this run")
        if result.get("provenance", {}).get("binary_sha256") != self.binary_hashes:
            raise RuntimeError("Physics binaries changed during this run; refusing mixed-oracle cache")
        self._store(key, result)
        return result

    def evaluate_many(self, parameters, eval_ids):
        """Preserve requested order even when independent subprocesses overlap."""
        if len(parameters) != len(eval_ids) or len(set(eval_ids)) != len(eval_ids):
            raise ValueError("Batch IDs must be distinct and match the parameter list")
        if self.workers == 1:
            return [self.evaluate(p, i) for p, i in zip(parameters, eval_ids)]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self.evaluate, parameters, eval_ids))

    def report(self):
        return {**self.counts, "workers": self.workers, "oracle_fingerprint": self.fingerprint,
                "cache": str(self.cache_path),
                "timing_caution": "Cache reuse changes wall time. Compare scan yield by logical requests; do not claim an uncached speedup."}
