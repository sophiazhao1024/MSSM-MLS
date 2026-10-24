# MSSM MLS 

A source distribution reconstructing the published **Model-4 MSSM Machine Learning Scan** in *Exploring Supersymmetry with Machine Learning*, Nuclear Physics B 943 (2019) 114613, [doi:10.1016/j.nuclphysb.2019.114613](https://doi.org/10.1016/j.nuclphysb.2019.114613).
It preserves raw physical inputs and learns `mh`, `mH`, `chi2_hs`, and `ln(omega)` with four independent networks. The Python distribution and command are named `mssm-mls-logomega`.


Release 0.1.1 passed **124 software tests** on macOS arm64 with Python 3.12.14. The public sanitized fixture was replayed through the full 2,000-epoch training and 20-million-proposal sampler; its fit and selection reports matched the original experiment. The frozen-weight check also passed. No new physics calculation or external-library build was performed for this packaging release. Exact scope is recorded in `docs/release_validation.json`; preservation of the 17 numerical/physics files is recorded in `docs/core_preservation.json`.

## What is implemented

- Parameters: `M1` in [2, 20] GeV, `mu` in [7000, 9000] GeV, and `tan_beta` in [2, 10]; fixed MSSM parameters are in `domain.py`.
- Four independent 3 → 60 → 60 → 60 → 60 → 1 regressors and one physical/unphysical classifier; cumulative real labels; Adam; classifier before regression.
- Only the abundance label is transformed: `omega → ln(omega)` for training and `exp` after prediction.
- Eq. (7) weight: `exp(-0.15*mh)` when `123 < mH < 129`, `chi2_hs < 112.7273`, and `0.08209 < omega < 0.15511`; zero otherwise. Weights use physical observable values.
- 1,000 initial random evaluations; normally 100 recommended plus 5 random evaluations per iteration. The strict 20,000 budget ends with a 95 + 5 final batch.
- Real FeynHiggs 2.13.0 → HiggsBounds 4.3.1 / HiggsSignals 1.4.0 → SLHA → micrOMEGAs 4.3.2 drivers, plus a random-scan baseline.


## Install the Python package

Use a complete source checkout or unpacked release ZIP. Physics scripts/configuration are repository files and are not supplied by installing only a Python wheel. Use a path without spaces, Python **3.12**, and a separate virtual environment; do not install another distribution providing `mssm_mls` into that environment.

```bash
cd mssm-mls-historical-reconstruction
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
mssm-mls-logomega --help
python tools/verify_release.py
```

`requirements-tested.txt` records exact versions from the verified environment; it is not a guarantee that identical wheels exist on every platform. To request those versions, install that file in the new environment before the editable install. Software tests use explicit artificial fixtures and do not certify MSSM results. The included CI workflow, if run by GitHub, checks software behavior; providing a workflow does not mean a GitHub run has already passed.

## Inspect the archived failure without physics software

```bash
python scripts/check_frozen_model.py --output runs/frozen-check.json
```

This reads the included model weights with `weights_only=True`, verifies the frozen model's diagnostic claim, and does not train or call physics programs. The archived truth JSONL is a documented sanitized derivative: machine-specific paths were removed/replaced while numerical and status fields were preserved. Original and public-file hashes are recorded in its provenance. It is not byte-identical to the private original journal and is not silently injected into normal scans.

To repeat the complete training and 20-million-proposal first-round replay, which is substantially slower:

```bash
python scripts/validate_saved_first_round.py --project . \
  --records validation/initial1000_records.jsonl \
  --output runs/offline-first-round.json
```

An accurately reproduced `proposal_exhausted` result is the expected finding for that saved experiment, not a successful scan.

## Build the real physics backend

Install working C, C++, Fortran compilers and `make` first. macOS also requires the Command Line Tools and usable SDK. Linux build branches are provided, but a successful build on your Linux system is not asserted. Native Windows is not the intended build environment; use a suitable Linux environment instead.

```bash
python scripts/setup_physics.py
# If compilers are outside PATH:
# python scripts/setup_physics.py --toolchain-bin /path/to/compiler/bin
mssm-mls-logomega doctor --backend configs/backend.local.json
```


## Run real calculations

Each output directory must be new. These commands use the unchanged baseline settings unless explicitly stated.

```bash
# A previously validated coordinate; verify the result on your own build.
mssm-mls-logomega point --backend configs/backend.local.json \
  --M1 7.3 --mu 7600 --tan-beta 7.725 --output runs/first-point

# Ten real random evaluations as an installation check, not an efficiency study.
mssm-mls-logomega scan --backend configs/backend.local.json \
  --settings configs/baseline.json --method random --seed 0 \
  --budget 10 --initial 10 --output runs/random-check-s0

# Attempt the initial 1000 plus one 100+5 MLS batch; completion is not promised.
mssm-mls-logomega scan --backend configs/backend.local.json \
  --settings configs/baseline.json --method mls --seed 0 \
  --budget 1105 --output runs/mls-attempt-s0

# Full-budget commands; this release has not completed this MLS experiment.
mssm-mls-logomega scan --backend configs/backend.local.json \
  --settings configs/baseline.json --method random --seed 0 \
  --budget 20000 --output runs/random-20000-s0
mssm-mls-logomega scan --backend configs/backend.local.json \
  --settings configs/baseline.json --method mls --seed 0 \
  --budget 20000 --output runs/mls-20000-s0
```

`ok` means an evaluable physical result, not experimental acceptance. `unphysical` is an identified physical failure; `tool_error` is a technical/numerical/program failure excluded from training. Every attempted truth evaluation consumes the logical budget, including errors. Five consecutive technical errors abort the default run. A failed recommendation batch stops with diagnostics; no random substitute is inserted.

`records.jsonl`, `training_history.json`, `summary.json`, `manifest.json`, `physics_manifest.json`, and the per-point `physics/` directories record results and provenance. Default final acceptance, called `paper-higgs`, requires true Eq. (7) cuts **and** `HB_result == 1`; `eq7-only` is a separately labelled diagnostic policy. This HB integration is a disclosed reconstruction choice, not recovered original code. Both counts are retained. Predictions are not counted as true targets.

```bash
mssm-mls-logomega compare runs/random-20000-s0 runs/mls-20000-s0 \
  --output runs/comparison-s0
```

An incomplete MLS run remains incomplete in the comparison. Compare matching policies, physics settings, seeds and budgets. Optional `--cache runs/truth.sqlite` enables exact-parameter, physics-fingerprinted truth reuse; it does not provide new scientific observations or establish a wall-clock speedup. See the Chinese guide for details.


