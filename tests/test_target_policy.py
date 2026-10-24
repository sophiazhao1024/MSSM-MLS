"""Acceptance semantics with artificial fixtures, never SUSY validation data."""
import json
import pytest
from mssm_mls.baseline import ScanSettings, run_scan, target_decision
from mssm_mls.cli import compare_runs, main, parser


def answer(hb=1, mH=126.):
    return {"status": "ok", "observables": {"mh": 10., "mH": mH,
            "chi2_hs": 20., "omega": .1186}, "diagnostics": {"hb_result": hb}}


@pytest.mark.parametrize("hb,allowed", [(1, True), (0, False), (-1, False),
                                      (None, False), (True, False), (1., False), ("1", False)])
def test_only_explicit_hb_allowed_passes_default(hb, allowed):
    result = target_decision(answer(hb), True, "paper-higgs")
    assert result["target_eq7"] is True and result["target"] is allowed
    assert result["hb_allowed"] is allowed
    assert target_decision(answer(hb), False, "paper-higgs")["target"] is False
    assert target_decision(answer(hb), True, "eq7-only")["target"] is True


def test_missing_hb_and_errors_never_establish_final_acceptance():
    assert not target_decision({"status": "ok"}, True, "paper-higgs")["target"]
    for status in ("tool_error", "unphysical"):
        result = target_decision(answer() | {"status": status}, True, "eq7-only")
        assert result["target"] is False and result["target_eq7"] is False
    legacy = {"status": "ok", "provenance": {"higgs": {"hb_result": 1}}}
    assert target_decision(legacy, True, "paper-higgs")["target"] is True
    # Do not override an explicit unusable result with a stale provenance copy.
    assert not target_decision(legacy | {"diagnostics": {"hb_result": -1}}, True,
                               "paper-higgs")["target"]


@pytest.mark.parametrize("settings", [ScanSettings])
def test_policy_defaults_and_invalid_values(settings):
    assert settings().target_policy == "paper-higgs"
    with pytest.raises(ValueError, match="target_policy"):
        settings(target_policy="silently-ignore-hb")


@pytest.mark.parametrize("settings,run", [(ScanSettings, run_scan)])
@pytest.mark.parametrize("policy,accepted", [("paper-higgs", 1), ("eq7-only", 4)])
def test_baseline_keeps_eq7_hb_and_physical_status_separate(settings, run, policy, accepted):
    def fixture(params, eval_id):
        rows = [answer(1), answer(0), answer(-1), answer(None), answer(1, mH=140.)]
        return rows[eval_id - 1]
    result = run(settings(initial=5, budget=5, epochs=1, target_policy=policy), fixture,
                 method="random")
    assert all(r["status"] == "ok" and r["observables"] for r in result["records"])
    assert [r["target_eq7"] for r in result["records"]] == [True] * 4 + [False]
    assert result["records"][1]["hb_result"] == 0  # still a valid regression/classifier-positive row
    summary = result["summary"]
    assert summary["targets_eq7"] == 4 and summary["targets"] == accepted
    assert summary["eq7_hb_excluded"] == 1 and summary["eq7_hb_unresolved"] == 2
    assert summary["target_fraction_of_attempts"] == accepted / 5
    assert result["manifest"]["settings"]["target_policy"] == policy


def test_hb_exclusion_does_not_filter_training_rows(monkeypatch):
    from mssm_mls.learning import Surrogates
    counts = []
    def capture(model, x, y, **kwargs):
        counts.append((len(x), y.tolist()))
        return {"epochs_executed": 1, "training_count": len(x)}
    monkeypatch.setattr("mssm_mls.learning.train_network", capture)
    rows = run_scan(ScanSettings(initial=2, budget=2, epochs=1),
                    lambda p, i: answer(i - 1), method="random")["records"]
    report = Surrogates(epochs=1).fit(rows)
    assert report["physical_count"] == 2
    assert [n for n, y in counts] == [2] * 5
    assert counts[-1][1] == [1., 1.]


@pytest.mark.parametrize("extra,expected", [([], False), (["--target-policy", "eq7-only"], True)])
def test_point_cli_defaults_to_higgs_consistency(monkeypatch, tmp_path, capsys, extra, expected):
    class Backend:
        def __init__(self, *args, **kwargs):
            pass
        def doctor(self):
            return {"ready": True}
        def evaluate(self, params, eval_id):
            return answer(0)
    monkeypatch.setattr("mssm_mls.cli.PhysicsBackend", Backend)
    out = tmp_path / "point"
    main(["point", "--output", str(out), "--M1", "10", "--mu", "7600", "--tan-beta", "7.8"] + extra)
    result = json.loads((out / "point.json").read_text())
    assert result["status"] == "ok" and result["eq7_target"] is True
    assert result["target_eq7"] is True and result["target"] is expected
    assert parser().parse_args(["scan", "--output", str(tmp_path / "scan"),
                               "--target-policy", "eq7-only"]).target_policy == "eq7-only"


def test_comparison_rejects_mixed_policy_without_rewriting_old_runs(tmp_path):
    paths = []
    for name, config in [("old", {}), ("new", {"target_policy": "paper-higgs"})]:
        path = tmp_path / name
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({"settings": config, "method": "mls", "seed": 0}))
        (path / "summary.json").write_text("{}")
        (path / "records.jsonl").write_text('{"target": true}\n')
        paths.append(path)
    with pytest.raises(ValueError, match="different target policies"):
        compare_runs(paths, tmp_path / "compare")
    assert not (tmp_path / "compare").exists()
    assert json.loads((paths[0] / "manifest.json").read_text())["settings"] == {}
