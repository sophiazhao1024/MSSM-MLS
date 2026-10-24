# Known failure of the default first recommendation round

The default historical reconstruction has a reproducible failure when trained on the supplied saved first 1000 true evaluations: **it accepts zero recommended points from 20,000,000 neural-network candidates**. This is a limitation of this release, not a successful reproduction of the paper's approximately 80% efficiency. The default configuration is preserved so that the failure can be inspected without hidden tuning.

The immediate obstruction has been identified more strongly than a finite random search: **the frozen χ² and ln(Ωh²) regressors cannot predict inside their target windows anywhere in the entire three-dimensional scan box**. That statement concerns these particular trained weights, input box and numerical implementation. It does not say that all networks trained without standardization must fail.

## What was actually run

The experiment reused the first 1000 records of an earlier real-physics scan. The published fixture removes machine-specific metadata and retains its numerical learning inputs and labels; its provenance distinguishes original and publication-file hashes. No new external physics evaluations were made for this replay.

| Saved-record category | Count | Role in training |
|---|---:|---|
| `ok` | 70 | All four regressors; classifier positive class |
| `unphysical` | 825 | Classifier negative class |
| `tool_error` | 105 | Excluded from both kinds of training |
| Total | 1000 | Saved true-evaluation records |

Settings were seed 0, 2000 epochs, learning rate 0.001, batch size 256, four independent 4×60 regressors and a 4×60 classifier, raw inputs and only natural-log omega labels. Each regressor sees 70 rows, hence one optimizer step per epoch and 2000 steps total. The classifier sees 895 rows, hence four steps per epoch and 8000 total. Epoch count is not the same as gradient-update count.

An instrumented training replay produced tensors bitwise equal to the ordinary fit in the audited environment. All training losses stayed finite. The published model is the final default fit, not a replacement selected from an epoch sweep. Exact matching on another platform is not guaranteed merely by using the same random seed; dependency versions and floating-point kernels also matter.

## Full candidate replay

| Stage | Count |
|---|---:|
| Cheap candidates proposed | 20,000,000 |
| Classifier survivors at p ≥ 0.5 | 1,805,893 |
| Finite, physically admissible survivor predictions | 1,805,893 |
| Survivors passing 123 < mH < 129 | 526,298 |
| Survivors passing χ² < 112.7273 | 0 |
| Survivors passing 0.08209 < Ωh² < 0.15511 | 0 |
| Positive Eq.(7) weight | 0 |
| Guided points accepted | 0 of 100 |

Prediction ranges among classifier survivors were:

| Observable | Minimum | Maximum |
|---|---:|---:|
| mh, GeV | 7.15291977 | 63.74360275 |
| mH, GeV | 116.56064606 | 145.37298584 |
| χ² | 175.38018799 | 218.63969421 |
| Ωh² | 83.37916806 | 293.57879570 |

The original selection diagnostics were reproduced. `ProposalExhausted` is the expected outcome. The five extra random evaluations are scheduled after successful guided selection, so this failure prevents the next data update. The implementation does not silently substitute random points for the missing recommendations.

These 20 million points are cheap surrogate proposals, not 20 million true physical evaluations. **Zero accepted proposals is not a measured physical scan efficiency of 0/20,000,000.** Nor is this a completed 1105-point or 20000-point MLS run.

## Why more candidates cannot help this frozen fit

For the final mH, χ² and ln(Ωh²) networks, interval analysis proves that every ReLU unit remains either active or inactive throughout the whole input box. A ReLU is `max(0,z)`; when its sign never changes, it either passes its input or returns zero. A composition of linear layers with fixed ReLU gates is an affine function: a weighted sum plus a constant.

The resulting χ² and ln(Ωh²) functions, rounded here for readability, are:

```text
chi2_hat = 0.029846361357460466 * M1
         + 0.024954618082378997 * mu
         + 0.022100664469056595 * tan_beta
         + 0.41066439803502836

ln_omega_hat = 0.014383463383658717 * M1
             + 0.0005819545610854155 * mu
             + 0.016372614075626055 * tan_beta
             + 0.16122336051961295
```

All coefficients in these two functions are positive. On M1∈[2,20], mu∈[7000,9000], tan_beta∈[2,10], their minima therefore occur at `(2,7000,2)`:

| Frozen function | Ideal-arithmetic global minimum | Required upper bound |
|---|---:|---:|
| χ² | 175.19688502634105 | 112.7273 |
| ln(Ωh²) | 4.296417443036091 | ln(0.15511), approximately −1.86362 |
| Ωh² after exp | approximately 73.43623238 | 0.15511 |

Either violated inequality is independently sufficient to make Eq.(7) zero. The conclusion therefore holds regardless of the classifier decision, random candidate stream or a larger candidate count, **while these weights remain fixed**. Merely being affine is not itself a proof of failure; the certified output ranges outside the target windows are decisive.

An independent certificate recomputed exact dyadic-weight arithmetic and propagated a conservative float32 rounding bound through all layers. Under ordinary nearest-rounded IEEE float32 multiplication/addition or FMA, without overflow, underflow or reduced-precision transforms, the ReLU signs remain fixed. Its enclosing lower bounds are χ² ≥ 175.0723911 and ln(Ωh²) ≥ 4.1900876. Both remain far outside the windows. This is not a boundary-rounding issue.

The proof describes the final network. It does not prove that gates never switched during training, that every neuron is inactive, that the outputs are constant, or that every earlier epoch also had an empty target region.

## The regressors have not fitted the training values adequately

For comparison, a trivial predictor can return the training-label mean for every input. This comparison uses each regressor's own original training-label space; it does not standardize or alter the model.

| Label | Final network training MSE | Constant-mean training MSE |
|---|---:|---:|
| mh | 417.6708984 | 627.4418122 |
| mH | 20.2079792 | 1.4511498 |
| χ² | 1328.4642334 | 1254.8514971 |
| ln(Ωh²) | 6.0670023 | 6.0466574 |

Three regressors do not outperform this constant predictor even on their training rows. These are training diagnostics, not held-out generalization estimates. The mH error is approximately 13.93 times the constant predictor's error. The mean log abundance is approximately 4.853185, corresponding to an abundance near 128.15; the narrow target is around log abundance −2.

Among the 70 true physical rows, 60 pass the mH window, four pass the χ² cut and one passes the abundance window. No row simultaneously passes χ² and abundance, so no row passes all Eq.(7) cuts. Absence of a joint target in the initial batch does not make continuous observable learning impossible, but it shows how sparse the relevant evidence is.

Two concrete training rows illustrate the missed information:

- Evaluation 1 has true χ²=94.3478175, but predicted χ²=191.6193085. Its true abundance is 1.1367939, so this is not a full target point.
- Evaluation 969 has true abundance 0.08724954 inside the window, but predicted abundance approximately 115.59834. Its true χ²=186.33433 is outside the χ² window.

The smallest recorded training MSE for mH, χ² and ln(Ωh²) occurs at epoch 2000. This fact alone does not establish what target regions existed at earlier checkpoints. No early checkpoint was selected to replace the default final model.

## Classifier errors are real but not the sole obstruction

At threshold 0.5, 42 of the 70 physical training rows pass and 28 are rejected: training positive-class recall is 60%. This is not a separate test-set score.

A previously saved, independently calculated allowed point `(M1,mu,tan_beta)=(7.3,7600,7.725)` was not among those 70 training rows. Its true observables are approximately `(16.0891054,125.2553516,93.8537747,0.13569349)`, and HiggsBounds allows it. The classifier assigns probability 0.08179148 and rejects it. A diagnostic direct call to the regressors, bypassing this gate without changing production selection, gives approximately `(15.5880661,126.6434021,190.4543762,123.4154771)`. The regressors still miss both decisive windows. Removing the classifier would not fix this frozen model.

## Mechanism, possible contributing factors and unknown causes

**Established:** the default trained regressors have no predicted target region; the mean-predictor comparison demonstrates inadequate training fit; the classifier also misses some physical points; the guided-first control flow prevents the five random extras from updating the dataset after this failure.

**Observed, potentially contributing:** the raw mu input dominates typical first-layer weighted input contributions. For the abundance regressor, initial mean absolute contributions from `(M1,mu,tan_beta)` were approximately `(3.087,2005.231,1.471)` and final contributions `(3.355,1992.872,1.683)`. The initial training labels contain little evidence near the joint target. Finite optimization depends on initialization, batching and the other disclosed reconstruction choices.

**Not established:** a unique causal attribution to input scaling or output standardization. The previous standardized reconstruction differs in both input mapping and output treatment, so this comparison is not a one-factor experiment. Large raw gradients also do not prove gradient explosion or imply proportional Adam updates. These findings do not imply that all unscaled networks fail or that restoring standardization is necessary. No tuning experiment was used to support this release.

The original cards, exact batch size, initialization, stopping rule before 2000 epochs, and zero-recommendation recovery behavior remain unknown. Matching an observed failure mechanism does not recover those missing historical choices.

## Paper and xBIT comparison

The paper specifies four separate observable regressors, the log-abundance target, a classifier and Eq.(7). The local xBIT implementation examined for this reconstruction is a different program: it uses raw inputs, conditionally logs each output column when max/min>100, and has its own multi-output predictor, dropout, holdout/early-stopping and scoring logic.

For these actual 70 rows, xBIT's max/min ratios are approximately `(14.29,1.045,2.813,11873)`, so its column transformation would be `(identity,identity,identity,log)` too. A different log-column decision therefore does not explain this first fit. It does not follow that the two programs have identical training or recommendation behavior.

xBIT's function named `logL` is a rational soft score, `1/[1+(v-m)^2/(2s)^2]`, not the logarithm of a likelihood and not Eq.(7)'s hard windows. Its generic selection cannot be substituted and still described as an unchanged Model-4 reconstruction. Nor does the inspected xBIT loop establish a universal random recovery when all scores are zero. See [sources.md](sources.md).

## Checks that did not identify an implementation mistake

The audit checked observable order, mh versus mH, the log column, exactly one exp restoration, classifier label direction, N×1 target shape, Eq.(7) signs and physical units, finite/domain guards, candidate random streams and zero-weight rejection when the uniform variate is exactly zero. It did not identify a defect in these contracts. Log-space acceptance excludes exponential underflow as the reason for zero positive weights. HiggsBounds does not enter the surrogate weight, so it cannot directly explain the zero-weight count. This was the first fit, so loss of previously trained weights is not an explanation.

Physics settings still influence which initial truth rows exist. The existing known allowed true point shows that this reconstructed physical setup is not globally devoid of targets. The paper does not uniquely specify all settings, and this investigation does not establish pointwise agreement with the authors' original physical calculations.

## Evidence and reproducibility

The release includes the saved-data replay report and diagnostic artifacts under `validation/first_round/`: `report.json`, `training_predictions.csv`, `training_history.csv`, `frozen_first_fit.pt` and `independent_check.json`. The independent certificate records its numerical scope and weight hash. The diagnostic report records code/data hashes and counts. Use the documented saved-data replay entry point to retrain and repeat selection without a physics installation.

The frozen weight-file SHA256 is `144fc555304e75affda15ab9f47e464d013e1d7971a9bd24b7a45ee3ab3f343b`. Original audited core hashes are:

```text
learning.py f5ae7df2df69434957d1b1735ee60e800bf178f2a304ab91201f089aa6364767
domain.py   852ce9e9eb563dbd1b63f4f33881eb9b5961a3c443d3d596252a882ee204aad3
baseline.py 4aaa98d2a93be08470824a0416e6f2e357252ff356b147889b85499d30735f1a
```

Core weights and scientific settings were not changed to make a demonstration succeed. New results should be reported under their own manifest and target policy. Unit tests, a valid real physical point and a repeatable failed first round establish different facts; none establishes the paper's cumulative efficiency or full coverage.
