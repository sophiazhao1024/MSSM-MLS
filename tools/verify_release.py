#!/usr/bin/env python3
"""Verify this source release and its public saved-truth fixture.

Standard library only. Does not train, run physics, or certify scientific
accuracy. RELEASE_MANIFEST.json describes the packaged snapshot; intentional
source edits should be reviewed and released with a new manifest.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    failures = []
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path) != expected:
            failures.append(f"changed: {relative}")
    provenance = json.loads((ROOT / "validation/data_provenance.json").read_text())
    data = ROOT / "validation/initial1000_records.jsonl"
    if sha256(data) != provenance["public_file_sha256"]:
        failures.append("public fixture hash differs from its provenance")
    records = [json.loads(line) for line in data.read_text().splitlines()]
    projection = [{k: row.get(k) for k in provenance["projection_fields"]}
                  for row in records]
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != provenance["learning_projection_sha256"]:
        failures.append("learning inputs/labels differ from the source projection")
    statuses = dict(Counter(row["status"] for row in records))
    if len(records) != 1000 or statuses != {"ok": 70, "unphysical": 825, "tool_error": 105}:
        failures.append("unexpected fixture size or status distribution")
    if [r["eval_id"] for r in records] != list(range(1, 1001)):
        failures.append("fixture evaluation order changed")
    if len({tuple(r["x"]) for r in records}) != 1000:
        failures.append("fixture contains duplicate parameter points")
    for row in records:
        if row["status"] == "ok":
            values = [row["observables"][k] for k in ("mh", "mH", "chi2_hs", "omega")]
            if row["y"] != values or values[3] <= 0:
                failures.append(f"invalid physical labels at eval_id={row['eval_id']}")
    result = {"release_integrity": not failures,
              "files_checked": len(manifest["files"]),
              "record_count": len(records), "status_counts": statuses,
              "new_physics_calls": 0, "new_training": False,
              "full_scan_completed": False, "failures": failures}
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
