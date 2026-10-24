"""Algorithm-contract tests. None pretends to validate a real SUSY spectrum."""
import json
import numpy as np
import pytest
import torch
from torch import nn
from mssm_mls.domain import (BOUNDS, FIXED_PARAMETERS, OMEGA_INTERVAL,
                             parameter_dict, score, validate_observables)
from mssm_mls.learning import Surrogates, TrainingDataError, make_network, train_network
from mssm_mls.baseline import ProposalExhausted, ScanSettings, rejection_sample, run_scan


def physical(mh=10., mH=126., chi2=10., omega=.1186):
    return {"mh": mh, "mH": mH, "chi2_hs": chi2, "omega": omega}


def record(status, x=None, observables=None):
    return {"status": status, "x": [10., 8000., 6.] if x is None else x,
            "observables": physical() if observables is None else observables,
            # Legacy flags must NOT silently remove any physical training rows.
            "validation": True}


class ConstantModels:
    def __init__(self, probability=1., mh=1.):
        self.probability, self.mh, self.fit_sizes = probability, mh, []

    def fit(self, records):
        self.fit_sizes.append(len(records))
        return {"training_count": len(records)}

    def predict(self, x, threshold):
        return np.tile([self.mh, 126., 10., .1186], (len(x), 1)), np.full(len(x), self.probability)


def good_backend(params, eval_id):
    return {"status": "ok", "observables": physical(), "provenance": {"fixture": True},
            "diagnostics": {"hb_result": 1}}


def test_eq6_all_fixed_parameters_and_domains():
    assert BOUNDS.tolist() == [[2., 20.], [7000., 9000.], [2., 10.]]
    assert len(FIXED_PARAMETERS) == 21
    assert FIXED_PARAMETERS["mHpm_GeV"] == 155.
    assert FIXED_PARAMETERS["mu3_GeV"] == 1000.
    assert FIXED_PARAMETERS["mL3_GeV"] == FIXED_PARAMETERS["me3_GeV"] == 1500.
    assert all(FIXED_PARAMETERS[n] == -100. for n in ("At_GeV", "Ab_GeV", "Atau_GeV"))
    assert parameter_dict([2, 7000, 2]) == {"M1": 2., "mu": 7000., "tan_beta": 2.}
    with pytest.raises(ValueError):
        parameter_dict([2, 1000, 2])


def test_eq7_h_and_H_roles_and_log_weight():
    y = [[10, 126, 20, .1186], [20, 126, 20, .1186], [126, 10, 20, .1186]]
    logs, targets = score(y)
    assert targets.tolist() == [True, True, False]
    np.testing.assert_allclose(logs[:2], [-1.5, -3.])
    assert np.isneginf(logs[2])


@pytest.mark.parametrize("column,endpoint", [(1, 123.), (1, 129.), (2, 112.7273),
                                              (3, .08209), (3, .15511)])
def test_hard_cut_exact_decimal_endpoints_are_excluded(column, endpoint):
    row = [10., 126., 10., .1186]
    row[column] = endpoint
    assert not score([row])[1][0]
    # One float64 step inside is accepted: no ad-hoc epsilon or rounding.
    row[column] = np.nextafter(endpoint, [10., 126., 10., .1186][column])
    assert score([row])[1][0]


@pytest.mark.parametrize("row", [[-1, 126, 10, .1186], [0, 126, 10, .1186],
                                  [10, 126, -1, .1186], [10, 126, 10, 0],
                                  [10, np.nan, 10, .1186]])
def test_bad_predictions_have_zero_weight(row):
    logs, target = score([row])
    assert np.isneginf(logs[0]) and not target[0]


def test_validation_does_not_mean_experimentally_allowed():
    assert validate_observables(physical(mH=140.)).tolist() == [10., 140., 10., .1186]
    with pytest.raises(ValueError):
        validate_observables(physical(omega=0.))


def test_five_networks_are_independent_four_by_sixty():
    models = [make_network(seed) for seed in range(5)]
    for model in models:
        linear = [layer for layer in model if isinstance(layer, nn.Linear)]
        assert [(m.in_features, m.out_features) for m in linear] == [
            (3, 60), (60, 60), (60, 60), (60, 60), (60, 1)]
        assert sum(p.numel() for p in model.parameters()) == 11281
        assert sum(isinstance(m, nn.ReLU) for m in model) == 4
        assert not any(isinstance(m, nn.Dropout) for m in model)
    pointers = [p.data_ptr() for model in models for p in model.parameters()]
    assert len(pointers) == len(set(pointers))


def test_all_cumulative_truth_rows_train_and_log_is_natural(monkeypatch):
    calls = []
    def capture(model, x, y, **kwargs):
        calls.append((x.clone(), y.clone(), kwargs))
        return {"training_count": len(x), "epochs_executed": kwargs["epochs"]}
    monkeypatch.setattr("mssm_mls.learning.train_network", capture)
    observations = [record("ok"), record("ok", [11., 8100., 7.], physical(mH=140., omega=.5)),
                    record("unphysical"), record("tool_error")]
    model = Surrogates(epochs=2)
    report = model.fit(observations)
    assert [len(x) for x, _, _ in calls] == [2, 2, 2, 2, 3]
    assert calls[-1][1].tolist() == [1., 1., 0.]
    assert report["holdout_count"] == 0
    assert report["classifier"]["negative_count"] == 1
    np.testing.assert_allclose(calls[3][1].numpy(), np.log([.1186, .5]), rtol=1e-6)
    before = [id(m) for m in model.regressors]
    model.fit(observations + [record("ok")])
    assert report["physical_count"] == 2
    assert calls[-1][0].shape[0] == 4
    assert before != [id(m) for m in model.regressors]


def test_fixed_epoch_training_visits_every_row_without_validation():
    model = make_network(1)
    visits = []
    def record_input(module, args):
        if module.training:
            visits.extend(args[0][:, 0].tolist())
    hook = model.register_forward_pre_hook(record_input)
    x = torch.tensor([[0., 0, 0], [1., 0, 0], [2., 0, 0], [3., 0, 0]])
    result = train_network(model, x, torch.ones(4), epochs=3, learning_rate=.001,
                           batch_size=3, seed=1)
    hook.remove()
    assert result["epochs_executed"] == 3
    assert {i: visits.count(i) for i in range(4)} == {0: 3, 1: 3, 2: 3, 3: 3}


def test_no_physical_training_data_stops_explicitly():
    with pytest.raises(TrainingDataError):
        Surrogates(epochs=1).fit([record("unphysical"), record("tool_error")])


def test_classifier_filters_before_regression():
    seen = []
    class Classifier(nn.Module):
        def forward(self, x):
            seen.append(("classifier", len(x)))
            return 100 * (x[:, :1] - 11.)
    class Regressor(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value
        def forward(self, x):
            seen.append(("regressor", len(x)))
            assert bool((x[:, 0] > 11.).all())
            return torch.full((len(x), 1), self.value)
    model = Surrogates(epochs=1)
    model.classifier = Classifier()
    model.regressors = [Regressor(v) for v in (10., 126., 10., float(np.log(.1186)))]
    values, probabilities = model.predict([[3, 8000, 6], [19, 8000, 6]])
    assert seen == [("classifier", 2)] + [("regressor", 1)] * 4
    assert np.isnan(values[0]).all()
    assert values[1, 3] == pytest.approx(.1186, rel=1e-6)
    assert probabilities[0] < .5 < probabilities[1]


def test_rejection_uses_absolute_probability_and_handles_u_zero():
    class Rng:
        def uniform(self, low, high, size):
            return np.tile([10., 8000., 6.], (size[0], 1))
        def random(self, size):
            return np.array([0., .1, .9, .2])
    class Predictions:
        def predict(self, x, threshold):
            return np.array([[10., 140., 10., .1186],
                             [-np.log(.2)/.15, 126., 10., .1186],
                             [-np.log(.2)/.15, 126., 10., .1186],
                             [-np.log(.8)/.15, 126., 10., .1186]]), np.ones(4)
    settings = ScanSettings(guided=2, candidate_chunk=4, max_proposals=4)
    accepted, stats = rejection_sample(Predictions(), 2, settings, Rng())
    assert stats["accepted"] == 2
    assert stats["proposals"] == 4
    np.testing.assert_allclose([r["predicted"][0] for r in accepted],
                               [-np.log(.2)/.15, -np.log(.8)/.15])


def test_rejection_sample_matches_known_two_region_distribution():
    class TwoRegion:
        def predict(self, x, threshold):
            weights = np.where(x[:, 0] < 11., .2, .8)
            y = np.tile([1., 126., 10., .1186], (len(x), 1))
            y[:, 0] = -np.log(weights) / .15
            return y, np.ones(len(x))
    settings = ScanSettings(guided=4000, candidate_chunk=2048, max_proposals=20000)
    accepted, stats = rejection_sample(TwoRegion(), 4000, settings, np.random.default_rng(40))
    high_fraction = np.mean([r["x"][0] >= 11. for r in accepted])
    assert .77 < high_fraction < .83  # theory .8; generous deterministic tolerance
    assert stats["proposals"] > 4000


def test_candidate_cap_has_no_random_fallback():
    settings = ScanSettings(guided=2, candidate_chunk=8, max_proposals=16)
    with pytest.raises(ProposalExhausted) as captured:
        rejection_sample(ConstantModels(probability=0.), 2, settings, np.random.default_rng(1))
    assert captured.value.diagnostics["accepted"] == 0
    assert captured.value.diagnostics["proposals"] == 16


def test_full_paper_batch_100_guided_plus_5_random_and_cumulative_fit():
    model = ConstantModels()
    settings = ScanSettings(budget=1210, epochs=1, candidate_chunk=256, max_proposals=10000)
    result = run_scan(settings, good_backend, models=model, seed=4)
    assert model.fit_sizes == [1000, 1105]
    assert sum(r["origin"] == "initial" for r in result["records"]) == 1000
    assert sum(r["origin"] == "guided" for r in result["records"]) == 200
    explore = [r for r in result["records"] if r["origin"] == "explore"]
    assert len(explore) == 10 and all(r["predicted"] is None for r in explore)
    assert result["summary"]["attempts"] == 1210
    assert all(r["provenance"] == {"fixture": True} for r in result["records"])


def test_final_budget_truncation_and_truth_overrides_good_prediction():
    def excluded(params, eval_id):
        return {"status": "ok", "observables": physical(mH=140.)}
    settings = ScanSettings(budget=1010, epochs=1, candidate_chunk=256)
    result = run_scan(settings, excluded, models=ConstantModels(), seed=9)
    assert result["history"][0]["guided_count"] == 5
    assert result["history"][0]["random_count"] == 5
    assert result["summary"]["targets"] == 0
    assert result["summary"]["ok"] == 1010


def test_random_exploration_is_an_independent_stream():
    settings = ScanSettings(budget=1105, epochs=1, candidate_chunk=256)
    fast = run_scan(settings, good_backend, models=ConstantModels(mh=1.), seed=33)
    slow = run_scan(settings, good_backend, models=ConstantModels(mh=8.), seed=33)
    for origin in ("initial", "explore"):
        assert [r["x"] for r in fast["records"] if r["origin"] == origin] == [
            r["x"] for r in slow["records"] if r["origin"] == origin]
    assert fast["history"][0]["selection"]["proposals"] != slow["history"][0]["selection"]["proposals"]


def test_technical_failure_is_not_an_unphysical_label_and_costs_budget():
    def mixed(params, eval_id):
        if eval_id == 1:
            raise TimeoutError("fixture timeout")
        if eval_id == 2:
            return {"status": "unphysical", "reason": "fixture physical failure"}
        return good_backend(params, eval_id)
    settings = ScanSettings(initial=3, budget=3, epochs=1)
    result = run_scan(settings, mixed, seed=0)
    assert [r["status"] for r in result["records"]] == ["tool_error", "unphysical", "ok"]
    assert result["summary"]["attempts"] == 3
    assert result["summary"]["tool_error"] == 1


def test_exhaustion_preserves_partial_truth_and_failure_state(tmp_path):
    settings = ScanSettings(initial=4, budget=7, guided=2, random_extra=1,
                            epochs=1, candidate_chunk=8, max_proposals=8)
    with pytest.raises(ProposalExhausted) as captured:
        run_scan(settings, good_backend, tmp_path, models=ConstantModels(probability=0.))
    result = captured.value.scan_result
    assert len(result["records"]) == 4
    assert result["summary"]["status"] == "proposal_exhausted"
    assert json.loads((tmp_path / "summary.json").read_text())["attempts"] == 4
    assert len((tmp_path / "records.jsonl").read_text().splitlines()) == 4


def test_real_network_raw_input_training_and_predictions_are_reproducible():
    # Real-network contract, not a SUSY accuracy claim. With raw mu ~8000,
    # two training epochs do not establish usable rejection-sampling accuracy.
    observations = [record("ok", [v, 8000., 6.]) for v in (3., 6., 10., 15., 19.)]
    first, second = Surrogates(epochs=2, seed=12), Surrogates(epochs=2, seed=12)
    assert first.fit(observations) == second.fit(observations)
    x = [[4., 8100., 5.], [18., 7500., 7.]]
    for a, b in zip(first.predict(x), second.predict(x)):
        np.testing.assert_equal(a, b)
    for a, b in zip(first.regressors, second.regressors):
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert torch.equal(pa, pb)


def test_random_reference_shares_initial_points_and_never_trains():
    class NoTraining:
        def fit(self, records):
            raise AssertionError("Random scanning must not train")
    settings = ScanSettings(initial=4, budget=7, guided=2, random_extra=1, epochs=1)
    random_result = run_scan(settings, good_backend, method="random", models=NoTraining(), seed=3)
    mls_result = run_scan(settings, good_backend, models=ConstantModels(), seed=3)
    assert [r["x"] for r in random_result["records"][:4]] == [r["x"] for r in mls_result["records"][:4]]
    assert random_result["summary"]["attempts"] == 7
    assert all(r["origin"] == "random" for r in random_result["records"][4:])
