# Sources and reconstruction evidence

This repository implements a published method from its description. It is not recovered original author code. The distinction between explicit paper statements, inherited choices, release constraints and unknown historical details is part of the scientific record.

## Primary paper

Jie Ren, Lei Wu, Jin Min Yang and Jun Zhao, *Exploring supersymmetry with machine learning*, Nuclear Physics B **943** (2019) 114613. [DOI](https://doi.org/10.1016/j.nuclphysb.2019.114613) · [SCOAP3 full text](https://scoap3-prod-backend.s3.cern.ch/media/files/46801/10.1016/j.nuclphysb.2019.114613.pdf).

Page numbers below refer to the journal/PDF pages printed 1–12. Model-4 is on pages 8–10; Eq.(6) defines the scan subspace and Eq.(7) defines its likelihood weight. General MLS is described on pages 2–4. Model-1 on pages 4–5 supplies some training choices that Model-4 does not separately restate. The paper's source-code footnote appears in the Model-3 discussion; it does not make this reconstruction the authors' released Model-4 code.

## Evidence matrix

| Topic | Evidence class and primary location | Implementation and boundary |
|---|---|---|
| Three scan parameters and fixed MSSM quantities | Explicit Model-4, Eq.(6), p.8 | `src/mssm_mls/domain.py:10–23`; masses in GeV, mu and right-handed squark mass symbols are distinct. |
| Four targets mh, mH, χ²_HS, ln(Ωh²) | Explicit Model-4, p.9 | `learning.py:101–138`; four independent scalar models, not a shared multi-output network. |
| Physical/unphysical classifier; 4 hidden layers × 60 | Explicit Model-4, pp.9–10 | `learning.py:34–44` and `:135–142`; classification threshold itself is not stated by the paper. |
| Sigmoid output and BCE classifier loss | Explicit Model-4, pp.9–10 | `learning.py:47–58` and `:145–163`; numerically combined BCE-with-logits in training and explicit sigmoid at prediction. |
| Adam; up to 2000 epochs | Explicit Model-4, p.10 | `learning.py:54–67`; default runs exactly 2000, a reconstruction of an upper limit without a recovered early-stop rule. |
| Cumulative truth data without holdout or regularization | General MLS, pp.3–4 | `learning.py:101–142`; all eligible accumulated rows participate. Missing/failed-result rules are reconstructed. |
| Classifier before regressors | Explicit Model-4, p.10 | `learning.py:145–163`; rejected predictions are NaN placeholders, not extra true negative records. |
| 1000 initial; 100 recommended plus 5 random | Explicit Model-4, p.10 | `configs/baseline.json`; strict final-budget truncation and ordering of failure recovery remain implementation choices. |
| Likelihood-guided rejection sampling | General MLS, p.3, and Model-4 Eq.(7), p.9 | `baseline.py:132–176`; uniform candidates, M=1 envelope, chunk size and cap are declared choices. |
| ReLU, linear regression output, MSE, constant lr=0.001 | Explicit for Model-1, pp.4–5; inherited here | `learning.py:14–27`, `:34–67`; not mislabeled as separately recovered Model-4 settings. |
| Only natural-log abundance; no other input/output scaling | Release constraint, consistent with the explicit target list but not a complete historical disclosure | `learning.py:97–116` and `:160–164`; paper does not report extra scaling. No additional transformation is inserted. |
| Batch 256, fresh models/Adam, CPU float32, p≥0.5, seeds | Historical details not uniquely specified | `learning.py:14–27`, `:82–95`, `:122–142`; see `reconstruction_assumptions.txt`. |
| FeynHiggs 2.13.0; HB 4.3.1; HS 1.4.0; MO 4.3.2 | Explicit physics versions, Model-4 pp.8–9 | `configs/sources.lock.json`; detailed cards, switches and source archive provenance remain separate issues. |
| True target count including HB | Paper discusses Higgs consistency and HB; Eq.(7) does not display a separate HB factor | `baseline.py:34–81`; `paper-higgs` requires HB_result=1, while `eq7-only` is a labeled diagnostic. Original integration is unknown. |
| Approximately 80% at 20000 collected samples | Paper Model-4 Fig.8 discussion, p.10 | A published result, not this repository's result. Current default first-round replay fails; see `known_failure.md`. |

The line references above use paths relative to `src/mssm_mls/` where a shorter filename is shown, and describe this release's source snapshot. Function names remain the reliable navigation points if later documentation-only maintenance shifts line numbers.

## Historical physics configuration

The exact defaults are in [`configs/backend.example.json`](../configs/backend.example.json), [`cards.py`](../src/mssm_mls/cards.py) and [`reconstruction_assumptions.txt`](reconstruction_assumptions.txt). They include native FeynHiggs `MHp=155`, `MA0=-1`, `MT=172`, flags `4002420111`, HS dataset `latestresults-1.4.0`, Gaussian mass PDF `2`, peak-centered mode `1`, neutral-Higgs mass uncertainties `[0,2,0]` GeV, zero HB mass uncertainties and HB group `LandH`. micrOMEGAs uses `fast=1`, `Beps=1e-5`, `VW=VZ=0` with the external SLHA spectrum.

These detailed settings are reconstructions from historical package examples/API behavior. They are not all specified in the article. Keeping the advertised version numbers does not establish numerical identity with the authors' original calculations. First/second-generation trilinears, SM defaults and top mass must also be disclosed.

[`physics/fh_hs_driver.F`](../physics/fh_hs_driver.F) adapts HiggsSignals 1.4.0's `example_programs/HBandHSwithFH.F` (Tim Stefaniak, 06/03/2013). The original example provenance and GPLv3-or-later notice are retained. The downloaded HiggsSignals README describes those terms; the full license accompanies the driver in `physics/LICENSE-HiggsSignals.txt`. Adaptations include the FH 2.13 API, explicit input/options/output paths, SLHA output, physical-state identification and structured error reporting. [`THIRD_PARTY_NOTICES.txt`](../THIRD_PARTY_NOTICES.txt) distinguishes these derived parts from the independently implemented Python scanner.

The build scripts disclose compatibility changes for current compilers, including BOZ integer conversion and Fortran STOP spacing. They do not establish that a modern executable is bitwise identical to a binary used in 2019.

## Source archives

The exact URLs and SHA256 hashes are stored in [`configs/sources.lock.json`](../configs/sources.lock.json). This machine-readable lock is the source of truth for downloaded bytes; version names alone are insufficient.

| Package | Pinned source | Provenance limitation |
|---|---|---|
| FeynHiggs 2.13.0 | [Historical official archive](https://wwwth.mpp.mpg.de/members/heinemey/feynhiggs/newversion/FeynHiggs-2.13.0.tar.gz) | Build compatibility edits are separately disclosed. |
| HiggsBounds 4.3.1 | [Historical official archive](https://higgsbounds.hepforge.org/downloads/HiggsBounds-4.3.1.tar.gz) | Checksums pin a particular archive, not undisclosed original run options. |
| HiggsSignals 1.4.0 | [Historical official archive](https://higgsbounds.hepforge.org/downloads/HiggsSignals-1.4.0.tar.gz) | Same-number updates existed; the lock pins the inspected archive. |
| micrOMEGAs 4.3.2 | [HEPcodes historical mirror commit](https://github.com/HEPcodes/micrOMEGAs/tree/671f3705527658618b51e721805a4def7f75d028) | This is a third-party mirror. The original official tarball checksum was unavailable; byte-identical official-archive authentication is not claimed. |

The source release does not bundle these libraries' binaries, generated CalcHEP working files or full archives. The setup script downloads the locked sources. Preserve each upstream package's notices and scientific citation requirements after installation.

The newly authored reconstruction source is released under GPL-3.0-or-later. The derived HiggsSignals driver retains its existing attribution and GPLv3-or-later terms. This repository's license does not replace the licenses of separately downloaded physics packages; see the root `LICENSE` and `THIRD_PARTY_NOTICES.txt` for scope.

## What xBIT contributes and what it does not establish

[xBIT's author repository](https://github.com/fstaub/xBIT) is a framework reference. The evidence here comes from an inspected local source snapshot; no commit identity for that snapshot was available. Links to the author's repository identify the project, not a verified statement that its current default branch is byte-identical to the inspected files. xBIT source files are not copied into this release.

| Inspected xBIT location | Behavior | Relation to this reconstruction |
|---|---|---|
| `package/scans/MLS.py:65–74`; `package/ml.py:85–89` | Raw parameter inputs become FloatTensor data. | Consistent with raw-input use here; cannot prove the paper authors' undisclosed preprocessing. |
| `package/ml.py:142–164` | Per-column natural log only when max/min>100. | Paper-specific labels fix log on omega here. For the actual 70 first-fit physical rows, both rules give identity, identity, identity, log. |
| `package/ml.py:34–47` | One multi-output predictor with Dropout(0.1). | Not used: Model-4 explicitly has four regressors and the general method states no regularization. |
| `package/ml.py:49–58` | Two hidden layers of 50 and two sigmoid outputs for classification. | Not used: the paper specifies four layers of 60 and a sigmoid output neuron. |
| `package/ml.py:85–89,119–137` | Approximate 80/20 split and early stopping/checkpoint recovery. | Not used: contrary to the paper's explicit cumulative all-data description. |
| `package/scans/MLS.py:67–69` | Optional direct training on ln(L+epsilon). | Not used: this Model-4 learns physical observables. |
| `package/scanning.py:103–110`; `package/aux.py:33–38` | Generic Gaussian or function named `logL` score according to transformation flags. | Not Eq.(7). `logL` is `1/[1+(v-m)^2/(2s)^2]`, not a logarithm of a likelihood. |

Consequently, an xBIT run cannot be treated as a direct validation of this paper-specific scanner. Adopting its dropout, holdout, different classifier or soft score would create another implementation, not recover an otherwise missing author setting. The inspected xBIT README calls it open source, but no LICENSE/COPYING file was found in that local snapshot; this repository does not assert or assign a license to that external project.

## Interpretation of the included validation

The original local evaluation logs are not included wholesale because they contain installation-specific metadata and third-party generated working files. The published learning fixture and its provenance preserve the numerical observations needed for an offline replay. Original-source hashes and sanitized-publication hashes are different and must not be interchanged.

`validation/first_round/report.json` records the default first-fit diagnostics; neighboring CSV files expose truth/prediction and training history. `frozen_first_fit.pt` identifies the fitted functions; `independent_check.json` records an independent whole-box affine and rounding-bound audit. These establish a known failure for a particular fit, not a general impossibility theorem for unscaled learning. They do not stand in for a newly completed physical scan.

Source attribution is distinct from measured validation: tests check software contracts, a real physical point checks a selected toolchain calculation, and a saved-data replay checks training/proposal behavior. The repository's scientific claims must stay within those boundaries.
