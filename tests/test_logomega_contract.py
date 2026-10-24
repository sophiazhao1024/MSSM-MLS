"""Numerical contracts specific to the explicit raw-input/log-omega variant."""
import json
import numpy as np
import pytest
import torch
from torch import nn
from mssm_mls.baseline import ScanSettings, run_scan
from mssm_mls.cli import parser
from mssm_mls.learning import Surrogates


def test_inputs_preserve_physical_numbers_and_validate_domain():
    physical = np.array([[2., 7000., 2.], [11., 8000., 6.], [20., 9000., 10.]])
    tensor = Surrogates.inputs(physical)
    assert tensor.dtype == torch.float32 and tuple(tensor.shape) == (3, 3)
    np.testing.assert_array_equal(tensor.numpy(), physical.astype(np.float32))
    np.testing.assert_array_equal(physical[1], [11., 8000., 6.])
    with pytest.raises(ValueError):
        Surrogates.inputs([[11., 0., 6.]])


def test_fit_only_logs_omega_and_never_scales_classifier_inputs(monkeypatch):
    calls = []
    def capture(model, x, y, **kwargs):
        calls.append((x.numpy().copy(), y.numpy().copy(), kwargs))
        return {"training_count": len(x)}
    monkeypatch.setattr("mssm_mls.learning.train_network", capture)
    rows = [
        {"status": "ok", "x": [3., 7100., 3.], "observables":
         {"mh": 10., "mH": 126., "chi2_hs": 20., "omega": .1}},
        {"status": "ok", "x": [19., 8900., 9.], "observables":
         {"mh": 30., "mH": 140., "chi2_hs": 80., "omega": 10.}},
        {"status": "unphysical", "x": [12., 8200., 5.], "observables": None},
        {"status": "tool_error", "x": [8., 7500., 4.], "observables": None},
    ]
    model = Surrogates(epochs=1)
    model.fit(rows)
    assert len(calls) == 5
    for x, _, _ in calls[:4]:
        np.testing.assert_array_equal(x, np.array([r["x"] for r in rows[:2]], dtype=np.float32))
    labels = np.column_stack([y for _, y, _ in calls[:4]])
    expected = [[10., 126., 20., np.log(.1)], [30., 140., 80., np.log(10.)]]
    np.testing.assert_allclose(labels, expected, rtol=1e-6)
    np.testing.assert_array_equal(calls[4][0], np.array([r["x"] for r in rows[:3]], dtype=np.float32))
    np.testing.assert_array_equal(calls[4][1], [1., 1., 0.])
    assert calls[4][2]["classifier"] is True
    assert not hasattr(model, "mean") and not hasattr(model, "scale")
    exported = model.export()
    assert "mean" not in exported and "scale" not in exported
    assert exported["input_scaling"] == "none"
    assert exported["output_scaling"] == "logomega only"


def test_predict_only_exponentiates_fourth_scalar_without_scaling_state():
    seen = []
    class Constant(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value
        def forward(self, x):
            seen.append(x.numpy().copy())
            return torch.full((len(x), 1), self.value, dtype=torch.float32)
    model = Surrogates(epochs=1)
    model.classifier = Constant(10.)
    model.regressors = [Constant(v) for v in [17., 125., 93., np.log(.12)]]
    assert not hasattr(model, "mean") and not hasattr(model, "scale")
    physical = [[7.3, 7600., 7.725]]
    y, probability = model.predict(physical)
    np.testing.assert_allclose(y, [[17., 125., 93., .12]], rtol=1e-6)
    assert probability.shape == (1,) and probability[0] > .5
    assert len(seen) == 5
    for x in seen:
        np.testing.assert_array_equal(x, np.array(physical, dtype=np.float32))


def test_manifest_distinguishes_variant_and_evidence():
    result = run_scan(ScanSettings(initial=1, budget=1),
                      lambda p, i: {"status": "unphysical"}, method="random")
    manifest = result["manifest"]
    assert manifest["variant"] == "logomega-only"
    assert manifest["input_scaling"] == "none"
    assert manifest["output_scaling"] == "logomega only"
    assert set(manifest["evidence"]) == {"paper", "user", "xBIT", "unreported"}
    assert manifest["settings"]["max_proposals"] == 20_000_000
    json.dumps(manifest, allow_nan=False)


@pytest.mark.parametrize("removed", [["--algorithm", "optimized"], ["--workers", "2"]])
def test_cli_has_no_optimized_or_parallel_algorithm_switch(removed):
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["scan", "--output", "unused"] + removed)
    assert error.value.code == 2
