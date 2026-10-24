"""Contract tests with explicit fake executables; NOT MSSM validation."""
import json
from pathlib import Path
import sys
import pytest
from mssm_mls.backend import PhysicsBackend
from mssm_mls.cards import write_fh_card, write_options


def backend(tmp_path, higgs=None, omega=None):
    higgs = higgs or {"status": "ok", "mh": 5., "mH": 125., "chi2_hs": 90., "hb_result": 0}
    omega = omega or {"status": "ok", "omega_h2": .12, "source_version": "4.3.5"}
    for name, body in {
        "fh": f"Path(sys.argv[3]).write_text({json.dumps(higgs)!r})\nPath(sys.argv[4]).write_text('BLOCK MASS\\n25 5')",
        "mo": f"assert Path(os.environ['MO_WORK_DIR']).is_dir()\nPath(sys.argv[2]).write_text({json.dumps(omega)!r})",
    }.items():
        p = tmp_path / name
        p.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nimport os, sys\n{body}\n")
        p.chmod(0o755)
    (tmp_path / "template").mkdir()
    config = {"fh_driver": "fh", "omega_driver": "mo", "mo_work_template": "template",
              "versions": {"FeynHiggs": "2.13.0", "HiggsBounds": "4.3.1", "HiggsSignals": "1.4.0", "micrOMEGAs": "4.3.5"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return PhysicsBackend(tmp_path / "config.json", tmp_path / "points", allow_version_substitution=True)


PARAMS = {"M1": 10., "mu": 8000., "tan_beta": 5.}


def test_version_substitution_is_opt_in(tmp_path):
    backend(tmp_path)
    with pytest.raises(ValueError, match="versions differ"):
        PhysicsBackend(tmp_path / "config.json", tmp_path / "points")


def test_higgsbounds_exclusion_does_not_mean_unphysical(tmp_path):
    result = backend(tmp_path).evaluate(PARAMS, 1)
    assert result["status"] == "ok"
    assert result["diagnostics"]["hb_result"] == 0
    assert result["observables"]["omega"] == .12
    assert not result["provenance"]["paper_version_match"]


def test_existing_evaluation_is_not_overwritten(tmp_path):
    b = backend(tmp_path)
    first = b.evaluate(PARAMS, 1)
    p = tmp_path / "points/point_000001/evaluation.json"
    original = p.read_bytes()
    repeated = b.evaluate(PARAMS, 1)
    assert first["status"] == "ok" and repeated["status"] == "tool_error"
    assert p.read_bytes() == original


def test_nan_is_technical_error_not_negative_physics_label(tmp_path):
    result = backend(tmp_path, omega={"status": "ok", "omega_h2": float("nan"), "source_version": "4.3.5"}).evaluate(PARAMS, 1)
    assert result["status"] == "tool_error"
    assert "Non-finite" in result["reason"]


def test_explicit_non_neutralino_lsp_is_unphysical(tmp_path):
    result = backend(tmp_path, omega={"status": "unphysical", "reason": "non_neutralino_lsp", "source_version": "4.3.5"}).evaluate(PARAMS, 1)
    assert result["status"] == "unphysical" and not result["observables"]


def test_paper_soft_mass_mapping_and_cp_real_input(tmp_path):
    path = tmp_path / "input"
    write_fh_card(path, PARAMS)
    rows = dict(line.split() for line in path.read_text().splitlines() if not line.startswith("#"))
    assert float(rows["M3SU"]) == 1000. and float(rows["Re(MUE)"]) == 8000.
    assert float(rows["M1SU"]) == 2000. and float(rows["M3SE"]) == 1500.
    assert float(rows["MHp"]) == 155. and float(rows["MA0"]) == -1.
    assert float(rows["Im(MUE)"]) == 0.


def test_options_do_not_accept_namelist_injection(tmp_path):
    with pytest.raises(ValueError):
        write_options(tmp_path / "opts", {"hs_dataset": "foo' / &bad"})
    write_options(tmp_path / "opts", {"hs_dmass": [1e-5, 0., 2.]})
    assert "1e-05d0" not in (tmp_path / "opts").read_text()


@pytest.mark.parametrize("flags", ["4002420110", "4002020110"])
def test_direct_charged_higgs_input_rejects_inconsistent_real_two_loop_flag(tmp_path, flags):
    with pytest.raises(ValueError, match="MHp.*tlCplxApprox=0"):
        write_options(tmp_path / "opts", {"fh_flags": flags})
