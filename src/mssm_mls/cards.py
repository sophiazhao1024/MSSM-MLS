"""Write one native FeynHiggs input card, with every MSSM soft mass explicit.

The native record syntax is verified against FeynHiggs 2.13.0's manual and
src/Record/Defaults.h. It is NOT a high-scale CMSSM/SPheno input card.
"""
import math
import re
from pathlib import Path
from .domain import FIXED_PARAMETERS, PARAMETER_NAMES, parameter_dict

DEFAULT_OPTIONS = {
    "fh_flags": "4002420111",
    "hs_dataset": "latestresults-1.4.0",
    "hs_pdf": 2,
    "hs_runmode": 1,
    "hs_dmass": [0., 2., 0.],
    "hb_dmass": [0., 0., 0.],
    "hb_charged_dmass": 0.,
    "hb_analyses": "LandH",
}


def write_fh_card(path, parameters, *, top_mass=172.):
    """Write Eq.(6) parameters. All dimensional inputs below are in GeV.

    MT=172 and first/second-generation A_f=-100 are reconstructed choices,
    inherited from the official example/default pattern, NOT paper claims.
    Qt=Qb=Qtau=0 is FeynHiggs's on-shell input convention. No mass is squared.
    The paper fixes charged-Higgs mass; MA0=-1 selects MHp as the input.
    """
    p = parameter_dict([parameters[name] for name in PARAMETER_NAMES])
    top_mass = float(top_mass)
    if not math.isfinite(top_mass) or top_mass <= 0:
        raise ValueError("top_mass must be finite and positive")
    f = FIXED_PARAMETERS
    rows = [("MT", top_mass), ("TB", p["tan_beta"]),
            ("MHp", f["mHpm_GeV"]), ("MA0", -1.)]
    for generation in (1, 2, 3):
        for fh_name, paper_name in (("SL", "mL"), ("SE", "me"),
                                     ("SQ", "mQ"), ("SU", "mu"), ("SD", "md")):
            rows.append((f"M{generation}{fh_name}", f[f"{paper_name}{generation}_GeV"]))
    complex_parameters = {
        "MUE": p["mu"], "M_1": p["M1"], "M_2": f["M2_GeV"], "M_3": f["M3_GeV"],
        "At": f["At_GeV"], "Ab": f["Ab_GeV"], "Atau": f["Atau_GeV"],
        "Au": -100., "Ac": -100., "Ad": -100., "As": -100., "Ae": -100., "Amu": -100.,
    }
    for name, value in complex_parameters.items():
        rows.extend([(f"Re({name})", value), (f"Im({name})", 0.)])
    rows.extend([(name, 0.) for name in ("Qt", "Qb", "Qtau")])
    rows.append(("scalefactor", 1.))
    text = "# Reconstructed Model-4 input; see docs/reconstruction_assumptions.txt\n"
    text += "# SM parameters except MT use FeynHiggs 2.13.0 built-in defaults.\n"
    text += "\n".join(f"{name:<16} {value:.17g}" for name, value in rows) + "\n"
    Path(path).write_text(text, encoding="ascii")


def write_options(path, overrides=None):
    """Write a restricted Fortran namelist; arbitrary namelist text is rejected."""
    overrides = {} if overrides is None else dict(overrides)
    unknown = set(overrides) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown physics options: {sorted(unknown)}")
    options = dict(DEFAULT_OPTIONS) | overrides
    if not re.fullmatch(r"[0-9]{10}", str(options["fh_flags"])):
        raise ValueError("fh_flags must contain ten digits")
    if options["fh_flags"][3] != "2":
        raise ValueError("This CP-conserving alignment profile requires higgsmix=2 (fourth flag)")
    if options["fh_flags"][9] == "0":
        raise ValueError("Direct MHp input is inconsistent with tlCplxApprox=0; use a validated nonzero setting")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(options["hs_dataset"])):
        raise ValueError("Invalid HiggsSignals dataset name")
    if options["hb_analyses"] not in {"LandH", "onlyL", "onlyH"}:
        raise ValueError("Unsupported HiggsBounds analysis group")
    for name, allowed in (("hs_pdf", {1, 2, 3}), ("hs_runmode", {1, 2})):
        if type(options[name]) is not int or options[name] not in allowed:
            raise ValueError(f"Invalid {name}")
    for name in ("hs_dmass", "hb_dmass"):
        values = options[name]
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"{name} needs three neutral-Higgs uncertainties")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"{name} must contain finite nonnegative values")
    charged = options["hb_charged_dmass"]
    if isinstance(charged, bool) or not isinstance(charged, (int, float)) or not math.isfinite(charged) or charged < 0:
        raise ValueError("hb_charged_dmass must be finite and nonnegative")
    rows = ["&driver_options"]
    for name, value in options.items():
        if isinstance(value, str):
            rendered = "'" + value + "'"
        elif isinstance(value, (list, tuple)):
            rendered = ", ".join((f"{v:.17g}".replace("e", "d") if "e" in f"{v:.17g}"
                                  else f"{v:.17g}d0") for v in value)
        else:
            rendered = str(value)
        rows.append(f"  {name} = {rendered}")
    rows.append("/")
    Path(path).write_text("\n".join(rows) + "\n", encoding="ascii")
    return options
