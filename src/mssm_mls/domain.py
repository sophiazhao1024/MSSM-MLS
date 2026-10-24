"""The historical Model-4 domain and Eq. (7), not today's exclusion limits.

Source: Ren et al., Nuclear Physics B 943 (2019) 114613, pp. 8-10.
All masses are in GeV. ``mu`` is the Higgsino parameter; ``mu3_GeV``
is a right-handed squark soft mass and is a different quantity.
"""
from numbers import Real
import numpy as np

PARAMETER_NAMES = ("M1", "mu", "tan_beta")
BOUNDS = np.array([[2., 20.], [7000., 9000.], [2., 10.]])
FIXED_PARAMETERS = {
    "M2_GeV": 2000., "M3_GeV": 2000., "mHpm_GeV": 155.,
    "mL3_GeV": 1500., "me3_GeV": 1500.,
    "mQ1_GeV": 2000., "mQ2_GeV": 2000.,
    "mu1_GeV": 2000., "mu2_GeV": 2000.,
    "md1_GeV": 2000., "md2_GeV": 2000.,
    "mQ3_GeV": 1000., "mu3_GeV": 1000., "md3_GeV": 1000.,
    "mL1_GeV": 1000., "mL2_GeV": 1000.,
    "me1_GeV": 1000., "me2_GeV": 1000.,
    "At_GeV": -100., "Ab_GeV": -100., "Atau_GeV": -100.,
}
OBSERVABLE_NAMES = ("mh", "mH", "chi2_hs", "omega")
REGRESSION_LABELS = ("mh", "mH", "chi2_hs", "ln_omega")
EXPECTED_TOOL_VERSIONS = {
    "FeynHiggs": "2.13.0", "micrOMEGAs": "4.3.2",
    "HiggsBounds": "4.3.1", "HiggsSignals": "1.4.0",
}
SOURCE = "Ren et al., NPB 943 (2019) 114613, Model-4, pp.8-10, Eqs.(6)-(7)"

# These are the exact decimal interval endpoints implied by Eq. (7).
# Direct interval comparisons avoid the floating-point subtraction artefact
# abs(0.08209 - 0.1186) < 0.03651 incorrectly accepting a nominal endpoint.
# theta(0)=0 is OUR explicit convention; the paper does not define theta(0).
HIGGS_INTERVAL = (123., 129.)
OMEGA_INTERVAL = (0.08209, 0.15511)
CHI2_MAX = 112.7273


def parameter_array(values):
    """Validate an N x 3 array in the Eq. (6) box; return float64 values."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("Expected a finite N x 3 parameter array")
    if np.any(x < BOUNDS[:, 0]) or np.any(x > BOUNDS[:, 1]):
        raise ValueError("Parameters are outside the Eq. (6) scan box")
    return x


def parameter_dict(x):
    """Turn one (M1, mu, tan_beta) row into the backend input mapping."""
    row = parameter_array(np.asarray(x).reshape(1, -1))[0]
    return dict(zip(PARAMETER_NAMES, map(float, row)))


def sample(count, rng):
    """Uniform box proposals: an explicit reconstruction choice for Model-4."""
    if type(count) is not int or count < 0:
        raise ValueError("count must be a non-negative integer")
    return rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(count, 3))


def validate_observables(values):
    """Accept complete physical outputs, even if they fail experimental cuts.

    A numerical/format error is not proof that a point is unphysical. The
    caller should record such an error as tool_error, never class label 0.
    ``omega`` is Omega h^2 itself; the natural logarithm is taken in learning.
    """
    if not isinstance(values, dict):
        raise ValueError("observables must be a mapping")
    row = []
    for name in OBSERVABLE_NAMES:
        value = values.get(name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"Missing or non-numeric observable: {name}")
        value = float(value)
        if not np.isfinite(value) or value < 0 or (name != "chi2_hs" and value == 0):
            raise ValueError(f"Invalid physical domain for observable: {name}")
        row.append(value)
    return np.array(row)


def score(values):
    """Return (log_L, target) for N x 4 values in physical units.

    Eq. (7): three hard cuts times exp(-0.15*mh). The mass cut is on H;
    the preference factor is on h. chi2_hs is the TOTAL HiggsSignals chi^2.
    Non-positive masses/relic density, negative chi^2 and non-finite surrogate
    predictions receive zero weight. This domain guard makes envelope M=1
    valid, and is an explicit reconstruction convention.

    Exact decimal cut endpoints are excluded. No epsilon is added and no
    rounding is performed. IEEE float64 still cannot represent every decimal
    exactly: predictions near a cut remain sensitive to numerical precision.
    The paper leaves both theta(0) and such numerical tolerances unspecified.
    """
    y = np.asarray(values, dtype=float)
    if y.ndim != 2 or y.shape[1] != 4:
        raise ValueError("Expected an N x 4 observable array")
    mh, mH, chi2, omega = y.T
    valid = (np.isfinite(y).all(axis=1) & (mh > 0) & (mH > 0)
             & (chi2 >= 0) & (omega > 0))
    target = (valid & (mH > HIGGS_INTERVAL[0]) & (mH < HIGGS_INTERVAL[1])
              & (chi2 < CHI2_MAX) & (omega > OMEGA_INTERVAL[0])
              & (omega < OMEGA_INTERVAL[1]))
    log_likelihood = np.full(len(y), -np.inf)
    log_likelihood[target] = -0.15 * mh[target]
    return log_likelihood, target
