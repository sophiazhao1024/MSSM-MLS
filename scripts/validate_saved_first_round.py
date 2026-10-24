"""Replay learning/recommendation from saved truth; this is NOT a new scan.

No physics outcome is fabricated and no outcome is added to a scan journal.
The public derivative of 1000 saved records preserves the original learning
inputs and labels; operational metadata was removed. The first
recommendation uses the new package's actual default training and proposal cap.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--project', type=Path, required=True)
p.add_argument('--records', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.project.resolve() / 'src'))
import numpy as np
from mssm_mls.baseline import ScanSettings, ProposalExhausted, rejection_sample
from mssm_mls.learning import Surrogates
from mssm_mls.domain import sample, score

settings = ScanSettings(**json.loads((a.project / 'configs/baseline.json').read_text()))
records = []
with a.records.open() as f:
    for line in f:
        if len(records) == settings.initial:
            break
        records.append(json.loads(line))
assert len(records) == 1000 == settings.initial
streams = np.random.SeedSequence(0).spawn(3)
initial_rng, proposal_rng, explore_rng = [np.random.default_rng(s) for s in streams]
assert np.array_equal(sample(settings.initial, initial_rng), [r['x'] for r in records])
out = {'kind': 'saved_truth_first_fit_and_recommendation_not_full_scan',
       'source_records': str(a.records.resolve()), 'source_records_sha256':
       hashlib.sha256(a.records.read_bytes()).hexdigest(),
       'learning_sha256': hashlib.sha256((a.project/'src/mssm_mls/learning.py').read_bytes()).hexdigest(),
       'record_count': len(records), 'statuses': {k:sum(r['status']==k for r in records)
               for k in ('ok','unphysical','tool_error')},
       'initial_points_match_seed_0': True, 'settings': settings.__dict__,
       'new_truth_evaluations': 0, 'full_scan_completed': False}
a.output.parent.mkdir(parents=True, exist_ok=True)
def save():
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
model = Surrogates(epochs=settings.epochs, learning_rate=settings.learning_rate,
                   batch_size=settings.batch_size, seed=0)
started = time.perf_counter()
try:
    print('Fitting five networks on saved initial 1000 true records', flush=True)
    out['fit'] = model.fit(records)
    out['fit_seconds'] = time.perf_counter() - started
    save()
    print('Fit complete; beginning unchanged rejection sampler', flush=True)
    selection_started = time.perf_counter()
    try:
        batch, selection = rejection_sample(model, settings.guided, settings, proposal_rng, iteration=1)
        out['recommendation_status'] = 'completed'
        out['recommendations'] = [{'x':r['x'].tolist(), 'predicted':r['predicted']} for r in batch]
        out['selection'] = selection
    except ProposalExhausted as exc:
        out['recommendation_status'] = 'proposal_exhausted'
        out['selection'] = exc.diagnostics
    out['selection_seconds'] = time.perf_counter() - selection_started
    # Descriptive check on an independent candidate pool, never used to select
    # a setting or modify any model. This RNG is separate from the sampler.
    x = sample(10000, np.random.default_rng(20261023))
    pred, prob = model.predict(x)
    log_l, hits = score(pred)
    out['independent_pool_diagnostic'] = {'count':len(x),
        'classifier_survivors':int(np.sum(prob>=.5)),
        'probability_quantiles':np.quantile(prob, [0,.25,.5,.75,1]).tolist(),
        'predicted_eq7_pass':int(hits.sum())}
except Exception as exc:
    out['validation_error'] = type(exc).__name__ + ': ' + str(exc)
    save()
    raise
save()
print(json.dumps({k:v for k,v in out.items() if k not in ('recommendations',)}, ensure_ascii=False, indent=2))
