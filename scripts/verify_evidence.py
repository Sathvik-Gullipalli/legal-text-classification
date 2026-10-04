"""Reconcile saved predictions and partitions without reevaluating fitted models."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
metrics = json.loads((ROOT / "results/metrics.json").read_text())
summary = json.loads((ROOT / "results/dataset_summary.json").read_text())
selection = json.loads((ROOT / "results/selection.json").read_text())
manifest = pd.read_csv(ROOT / "results/split_manifest.csv", keep_default_na=False)
predictions = pd.read_csv(ROOT / "results/test_predictions.csv")
assert manifest.groupby("group")["split"].nunique().max() == 1
assert manifest.groupby("text_hash")["split"].nunique().max() == 1
assert set(predictions.id) == set(manifest.loc[manifest.split == "test", "id"])
labels = summary["present_labels"]
for name in ("tfidf", "lda", "doc2vec"):
    y, pred = predictions.true_label, predictions[f"{name}_label"]
    expected = {"accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
                "weighted_f1": f1_score(y, pred, labels=labels, average="weighted", zero_division=0)}
    for metric, value in expected.items():
        assert np.isclose(metrics[name][metric], value, atol=1e-12), (name, metric)
    per_class = pd.read_csv(ROOT / f"results/{name}_per_category.csv")
    assert per_class.support.sum() == len(predictions)
    matrix = pd.read_csv(ROOT / f"results/{name}_confusion_matrix.csv").iloc[:,1:].to_numpy()
    assert matrix.sum() == len(predictions)
    assert np.isclose(matrix.trace() / matrix.sum(), metrics[name]["accuracy"])
winner = max(selection["models"], key=lambda n: (selection["models"][n]["macro_f1"],
              selection["models"][n]["accuracy"], -selection["models"][n]["validation_ms_per_document"]))
assert winner == selection["winner"]
assert len(manifest) == summary["retained_records"]
print("Verified split isolation, saved prediction metrics, class support, confusion matrices, and validation-only winner.")
