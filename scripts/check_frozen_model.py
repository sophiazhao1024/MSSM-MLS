"""Independent exact affine and conditional float32 error-bound check.

Reads frozen weights only. No fit, optimizer, candidate scan or physics call.
Both ideal-network algebra AND the gamma error propagation use Fraction;
float conversion occurs only when presenting the final report.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import torch
import argparse

PROJECT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("runs/frozen-check.json"))
args = parser.parse_args()
WEIGHTS = PROJECT / "validation/first_round/frozen_first_fit.pt"
EXPECTED_SHA256 = "144fc555304e75affda15ab9f47e464d013e1d7971a9bd24b7a45ee3ab3f343b"
if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != EXPECTED_SHA256:
    raise SystemExit("Frozen reference weights checksum mismatch")
STATE = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
LO = [F(2), F(7000), F(2)]
HI = [F(20), F(9000), F(10)]
UNIT_ROUNDOFF = F(1, 2**24)


def bounds(coefficients, bias):
    return [
        (
            bb + sum(min(c*l, c*h) for c, l, h in zip(row, LO, HI)),
            bb + sum(max(c*l, c*h) for c, l, h in zip(row, LO, HI)),
        )
        for row, bb in zip(coefficients, bias)
    ]


def rational(value):
    return str(value.numerator) + "/" + str(value.denominator)


def check(index, label):
    coefficients = [[F(int(i == j)) for j in range(3)] for i in range(3)]
    bias = [F(0)] * 3
    errors = [F(0)] * 3
    gate_layers = []
    for key in (0, 2, 4, 6, 8):
        weight = [[F(float(v)) for v in row]
                  for row in STATE["regressors"][index][f"{key}.weight"].numpy()]
        intercept = [F(float(v)) for v in
                     STATE["regressors"][index][f"{key}.bias"].numpy()]
        absolute_inputs = [max(abs(low), abs(high))
                           for low, high in bounds(coefficients, bias)]
        count = len(coefficients)
        gamma = ((2 * count + 1) * UNIT_ROUNDOFF) / (1 - (2 * count + 1) * UNIT_ROUNDOFF)
        next_errors = [
            sum(abs(w)*err for w, err in zip(row, errors))
            + gamma * (sum(abs(w)*(value+err) for w, value, err
                           in zip(row, absolute_inputs, errors)) + abs(bb))
            for row, bb in zip(weight, intercept)
        ]
        next_coefficients = [
            [sum(w*row[j] for w, row in zip(weight_row, coefficients))
             for j in range(3)]
            for weight_row in weight
        ]
        next_bias = [bb + sum(w*bi for w, bi in zip(row, bias))
                     for row, bb in zip(weight, intercept)]
        next_bounds = bounds(next_coefficients, next_bias)
        if key == 8:
            low, high = next_bounds[0]
            error = next_errors[0]
            return {
                "label": label,
                "ideal_box_range": [float(low), float(high)],
                "exact_ideal_box_range": [rational(low), rational(high)],
                "float32_absolute_error_bound": float(error),
                "exact_float32_absolute_error_bound": rational(error),
                "float32_enclosing_range": [float(low-error), float(high+error)],
                "exact_float32_enclosing_range": [rational(low-error), rational(high+error)],
                "all_gates_safe": True,
                "gate_layers": gate_layers,
                "exact_final_coefficients": [rational(v) for v in next_coefficients[0]],
                "exact_final_bias": rational(next_bias[0]),
            }
        masks = []
        safety_margins = []
        for (low, high), error in zip(next_bounds, next_errors):
            if low > error:
                masks.append(True)
                safety_margins.append(low-error)
            elif high < -error:
                masks.append(False)
                safety_margins.append(-high-error)
            else:
                raise AssertionError(f"Uncertified gate: {label}, layer {key}")
        gate_layers.append({
            "linear_key": key,
            "always_on": sum(masks),
            "always_off": len(masks)-sum(masks),
            "minimum_exact_margin_after_error": rational(min(safety_margins)),
            "minimum_margin_after_error": float(min(safety_margins)),
        })
        coefficients = [row if active else [F(0)]*3
                        for row, active in zip(next_coefficients, masks)]
        bias = [v if active else F(0) for v, active in zip(next_bias, masks)]
        errors = [v if active else F(0) for v, active in zip(next_errors, masks)]
    raise AssertionError("Missing output layer")


def main():
    report = {
        "kind": "independent_exact_fraction_affine_and_float32_bound_check",
        "new_training": False,
        "new_physics_calls": 0,
        "weight_file": str(WEIGHTS.relative_to(PROJECT)),
        "weights_sha256": hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "box": [[float(l), float(h)] for l, h in zip(LO, HI)],
        "method": "Exact dyadic parameter Fractions; exact affine extrema; exact gamma_(2n+1) error propagation; no floating bound inflation",
        "float32_scope": "Conditional bound for ordinary IEEE float32 nearest-rounded multiply/add or FMA, no overflow or underflow, no reduced-precision transforms; float32-rounded inputs stay inside the exactly representable box endpoints",
        "causal_scope": "Certifies the final frozen functions on this box, not the training trajectory or a unique cause of failed learning; being affine alone does not imply exclusion, the certified output ranges do",
        "networks": [check(index, label)
                     for index, label in ((1, "mH"), (2, "chi2_hs"), (3, "ln_omega"))],
    }
    chi2 = report["networks"][1]
    logomega = report["networks"][2]
    assert F(chi2["exact_float32_enclosing_range"][0]) > F("112.7273")
    # A positive log-omega implies omega>1, already greater than 0.15511.
    # This avoids relying on rounded ln/exp constants in the exact exclusion check.
    assert F(logomega["exact_float32_enclosing_range"][0]) > 0
    report["chi2_independently_blocks_eq7"] = True
    report["positive_lnomega_independently_blocks_omega_upper_bound"] = True
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "networks": [
        {k: item[k] for k in ("label", "ideal_box_range", "float32_absolute_error_bound", "float32_enclosing_range", "all_gates_safe")}
        for item in report["networks"]]}, indent=2))


if __name__ == "__main__":
    main()
