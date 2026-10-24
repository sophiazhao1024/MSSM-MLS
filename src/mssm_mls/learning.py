"""Paper-style cumulative training, without a held-out set or early stopping.

Model-4 explicitly specifies four independent scalar regressors, one physical
classifier, four hidden layers of 60 neurons, Adam, and at most 2000 epochs.
This variant follows the user-requested raw inputs and log-omega-only labels.
"""
import numpy as np
import torch
from torch import nn
from .domain import REGRESSION_LABELS, parameter_array, validate_observables

RECONSTRUCTION_ASSUMPTIONS = {
    "hidden_activation": "ReLU; explicit for Model-1, not restated for Model-4",
    "regression_loss": "MSE with linear output; inherited reconstruction choice",
    "learning_rate": "constant 0.001; not stated specifically for Model-4",
    "batch_size": "256 by default; paper does not specify",
    "training_duration": "fixed configured epochs, default 2000; paper says up to 2000",
    "reinitialization": "fresh independent networks and Adam on every cumulative fit",
    "input_scaling": "none; validated physical inputs cast directly to float32",
    "output_scaling": "logomega only; mh, mH and chi2_hs unchanged; natural log omega",
    "classifier_threshold": "0.5 by default; paper does not specify",
    "precision": "CPU float32 network, float64 domain/score; 1 torch CPU thread",
    "one_class_data": "train classifier even if only one physical-label class is present",
    "no_physical_data": "stop explicitly, rather than fabricate regression labels",
}


class TrainingDataError(RuntimeError):
    """The truth data cannot train the required paper-style models."""


def make_network(seed):
    """One independent 3 -> 60 -> 60 -> 60 -> 60 -> 1 network."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layers = []
        width = 3
        for _ in range(4):
            layers.extend([nn.Linear(width, 60), nn.ReLU()])
            width = 60
        layers.append(nn.Linear(60, 1))
        return nn.Sequential(*layers)


def train_network(model, x, y, *, epochs, learning_rate, batch_size, seed,
                  classifier=False):
    """Use EVERY supplied row each epoch; no validation or regularization.

    BCEWithLogitsLoss combines sigmoid and BCE in one numerically stable
    expression. At inference we apply sigmoid explicitly to obtain P(physical).
    """
    loss_fn = nn.BCEWithLogitsLoss() if classifier else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.)
    generator = torch.Generator().manual_seed(seed)
    target = y.reshape(-1, 1)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[indices]), target[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        # This is descriptive TRAINING loss, never a stopping/selection rule.
        final_loss = float(loss_fn(model(x), target))
    if not np.isfinite(final_loss):
        raise FloatingPointError("Non-finite final training loss")
    return {"epochs_executed": epochs, "training_count": len(x),
            "training_loss": final_loss, "row_visits": epochs * len(x)}


class Surrogates:
    """Four separate observable models and one physical/unphysical model."""

    def __init__(self, *, epochs=2000, learning_rate=0.001, batch_size=256, seed=0):
        if type(epochs) is not int or epochs < 1 or epochs > 2000:
            raise ValueError("epochs must be an integer in [1, 2000]")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.epochs, self.learning_rate = epochs, learning_rate
        self.batch_size, self.seed, self.fit_count = batch_size, seed, 0
        self.regressors, self.classifier = [], None
        torch.set_num_threads(1)

    @staticmethod
    def inputs(x):
        """Validate physical inputs, then cast to float32 without rescaling."""
        return torch.tensor(parameter_array(x), dtype=torch.float32)

    def fit(self, records):
        """All ok rows regress; all ok/unphysical rows classify.

        Experimentally excluded but physical points remain positive classifier
        labels and regression data. tool_error rows never train either model.
        No previous surrogate prediction is used as a truth label.
        """
        physical = [r for r in records if r["status"] == "ok"]
        classified = [r for r in records if r["status"] in {"ok", "unphysical"}]
        if not physical:
            raise TrainingDataError("No physical truth observations; regressors cannot be trained")
        x = self.inputs([r["x"] for r in physical])
        raw = np.array([validate_observables(r["observables"]) for r in physical])
        labels = raw.copy()
        labels[:, 3] = np.log(labels[:, 3])  # natural logarithm, not log10
        y = torch.tensor(labels, dtype=torch.float32)
        self.fit_count += 1
        self.regressors, self.classifier = [], None
        report = {"fit": self.fit_count, "physical_count": len(physical),
                  "classified_count": len(classified), "regressors": [],
                  "holdout_count": 0, "regularization": False}
        for j, name in enumerate(REGRESSION_LABELS):
            seed = self.seed + 100 * self.fit_count + j
            model = make_network(seed)
            stats = train_network(model, x, y[:, j], epochs=self.epochs,
                                  learning_rate=self.learning_rate, batch_size=self.batch_size,
                                  seed=seed)
            stats["label"] = name
            self.regressors.append(model)
            report["regressors"].append(stats)
        seed = self.seed + 100 * self.fit_count + 99
        self.classifier = make_network(seed)
        class_labels = torch.tensor([r["status"] == "ok" for r in classified],
                                    dtype=torch.float32)
        report["classifier"] = train_network(
            self.classifier, self.inputs([r["x"] for r in classified]), class_labels,
            epochs=self.epochs, learning_rate=self.learning_rate,
            batch_size=self.batch_size, seed=seed, classifier=True)
        report["classifier"].update(positive_count=len(physical),
                                    negative_count=len(classified) - len(physical))
        return report

    def predict(self, x, threshold=0.5):
        """Classifier FIRST; regress ONLY survivors; return physical units.

        Returns an N x 4 prediction matrix and N probabilities. Filtered rows
        are NaN predictions, not new unphysical truth records.
        """
        if len(self.regressors) != 4 or self.classifier is None:
            raise TrainingDataError("All five networks must be fitted before recommendation")
        if not 0 < threshold < 1:
            raise ValueError("classifier threshold must lie in (0,1)")
        tensor = self.inputs(x)
        with torch.inference_mode():
            probability = torch.sigmoid(self.classifier(tensor)).squeeze(1).numpy()
            keep = np.isfinite(probability) & (probability >= threshold)
            predictions = np.full((len(tensor), 4), np.nan)
            if keep.any():
                columns = [model(tensor[keep]).squeeze(1).numpy() for model in self.regressors]
                predictions[keep] = np.column_stack(columns)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            predictions[:, 3] = np.exp(predictions[:, 3])
        return predictions, probability

    def export(self):
        """Inference state only; this is not an exact resumable scan checkpoint."""
        return {"regressors": [m.state_dict() for m in self.regressors],
                "classifier": None if self.classifier is None else self.classifier.state_dict(),
                "fit_count": self.fit_count,
                "input_scaling": "none", "output_scaling": "logomega only",
                "assumptions": RECONSTRUCTION_ASSUMPTIONS}
