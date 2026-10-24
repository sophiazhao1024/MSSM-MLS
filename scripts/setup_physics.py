#!/usr/bin/env python3
"""Download checksum-pinned sources and build the supplied real physics drivers.

Run from the unpacked project: python scripts/setup_physics.py
Requires C/C++/Fortran compilers and make. Does not install system packages or
accept software licences. Read third-party licences in each downloaded source.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
        return result.hexdigest()


def fetch_source(name, item, vendor, downloads):
    """Preserve archives, check bytes, safely extract to a new directory."""
    archive = downloads / (name + "-" + item["version"] + ".tar.gz")
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        print(f"Downloading {name} {item['version']} ({item['provenance']})", flush=True)
        request = urllib.request.Request(item["url"], headers={"User-Agent": "MSSM-reproduction/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(archive)
    actual = digest(archive)
    if actual != item["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {archive}; expected {item['sha256']}, found {actual}")
    destination = vendor / item["directory"]
    marker = destination / ".source-provenance.json"
    if destination.exists():
        if not marker.exists() or json.loads(marker.read_text()).get("sha256") != actual:
            raise RuntimeError(f"Refusing to replace unverified existing source directory: {destination}")
        return destination
    with tempfile.TemporaryDirectory(prefix="extract-", dir=vendor) as temporary:
        extracted = Path(temporary)
        with tarfile.open(archive, "r:gz") as tar:
            # data_filter rejects path traversal and links outside destination.
            if not hasattr(tarfile, "data_filter"):
                raise RuntimeError("This Python needs a security-patched tarfile with data_filter; use Python 3.12+")
            tar.extractall(extracted, filter="data")
        # The official HB/HS tarballs contain macOS AppleDouble metadata
        # siblings (._HiggsBounds-...). They are not source directories.
        roots = [p for p in extracted.iterdir() if not p.name.startswith("._") and p.name != ".DS_Store"]
        if len(roots) != 1 or not roots[0].is_dir():
            raise RuntimeError(f"Unexpected archive layout for {name}")
        shutil.move(str(roots[0]), destination)
    marker.write_text(json.dumps(item, indent=2) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, default=PROJECT / "vendor")
    parser.add_argument("--toolchain-bin", type=Path)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    vendor = args.vendor.resolve()
    for path in (PROJECT, vendor, args.toolchain_bin):
        if path is not None and any(character.isspace() for character in str(path.resolve())):
            raise RuntimeError("Legacy CalcHEP/make does not reliably support whitespace in paths. Use a path without spaces.")
    downloads = vendor / "archives"
    downloads.mkdir(parents=True, exist_ok=True)
    lock = json.loads((PROJECT / "configs/sources.lock.json").read_text())
    paths = {name: fetch_source(name, item, vendor, downloads) for name, item in lock.items()}
    if args.download_only:
        return
    environment = os.environ.copy()
    toolchain = str(args.toolchain_bin.resolve()) if args.toolchain_bin else ""
    if toolchain:
        environment["PATH"] = toolchain + os.pathsep + environment["PATH"]
    if sys.platform == "darwin" and Path("/Library/Developer/CommandLineTools").exists():
        environment.setdefault("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
        environment.setdefault("SDKROOT", subprocess.check_output(["xcrun", "--show-sdk-path"], env=environment, text=True).strip())
    for compiler in ("gfortran", "cc", "c++", "make"):
        if not shutil.which(compiler, path=environment["PATH"]):
            raise RuntimeError(f"Missing {compiler}. See README installation steps.")
    environment["REPRO_PYTHON"] = sys.executable
    subprocess.run(["bash", str(PROJECT / "physics/build_fh_hs.sh"), str(vendor), toolchain],
                   env=environment, check=True)
    subprocess.run(["sh", str(PROJECT / "scripts/build_mo.sh"), str(paths["micrOMEGAs"]), toolchain, "4.3.2"],
                   env=environment, check=True)
    config = json.loads((PROJECT / "configs/backend.example.json").read_text())
    config.update(fh_driver=str(PROJECT / "physics/bin/fh_hs_driver"),
                  omega_driver=str(paths["micrOMEGAs"] / "MSSM/omega_driver"),
                  mo_work_template=str(paths["micrOMEGAs"] / "MSSM/work"),
                  versions={k: v["version"] for k, v in lock.items()}, archive_provenance=lock)
    config["environment"] = {k: environment[k] for k in ("PATH", "DEVELOPER_DIR", "SDKROOT") if k in environment}
    target = PROJECT / "configs/backend.local.json"
    if target.exists():
        target = PROJECT / "configs/backend.built.json"
    target.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Built real physics tools. Backend configuration: {target}")
    print("Run mssm-mls-logomega doctor --backend " + str(target))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc))
