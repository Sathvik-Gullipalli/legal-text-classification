"""Run validation selection, freeze models, then evaluate once on held-out data."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from .data import ISSUE_AREAS, clean_and_group, download, dump_json, make_splits, read_raw
from .features import TfidfFeatures, LdaFeatures, Doc2VecFeatures


def score(y, pred, labels):
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", labels=labels, zero_division=0))}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(project_root, force=False):
    root = Path(project_root).resolve()
    out = root / "artifacts"
    results = root / "results"
    if (results / "metrics.json").exists() and not force:
        raise RuntimeError("This run already evaluated the test set. Use saved artifacts for the demo. --force starts a separately documented rerun.")
    out.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.perf_counter()
    path, provenance = download(root / "data" / "raw")
    rows, cleaning = clean_and_group(list(read_raw(path)))
    splits = make_splits(rows)
    labels = cleaning["present_labels"]
    label_names = [ISSUE_AREAS[i] for i in labels]
    summary = {**cleaning, "splits": {n: {"records": len(rs),
                "groups": len({r["group"] for r in rs}),
                "class_counts": dict(Counter(r["category"] for r in rs))} for n, rs in splits.items()},
                "date_range": [min(r["decision_date"] for r in rows), max(r["decision_date"] for r in rows)]}
    dump_json(results / "dataset_summary.json", summary)
    dump_json(results / "provenance.json", provenance)
    manifest = [{k: r[k] for k in ("id", "group", "text_hash", "citation", "label", "category")} | {"split": name}
                for name, rs in splits.items() for r in rs]
    write_csv(results / "split_manifest.csv", manifest)
    print(json.dumps(summary, indent=2), flush=True)
    train = splits["train"]
    val = splits["validation"]
    xtrain = [r["text"] for r in train]
    xval = [r["text"] for r in val]
    ytrain = np.array([r["label"] for r in train])
    yval = np.array([r["label"] for r in val])
    candidates = []
    chosen = {}
    doc_vectors = None
    configs = [("tfidf", None), ("lda", 20), ("lda", 40), ("doc2vec", None)]
    for name, topics in configs:
        print(f"Fitting {name} topics={topics} on {len(train)} training opinions", flush=True)
        before = time.perf_counter()
        feat = {"tfidf": lambda: TfidfFeatures(), "lda": lambda: LdaFeatures(topics),
                "doc2vec": lambda: Doc2VecFeatures()}[name]()
        feat.fit(xtrain)
        ft = feat.transform(xtrain)
        fv = feat.transform(xval)
        feature_seconds = time.perf_counter() - before
        for c in (0.1, 1.0, 10.0):
            t = time.perf_counter()
            clf = LogisticRegression(C=c, max_iter=1500, solver="lbfgs", random_state=42)
            clf.fit(ft, ytrain)
            fit_seconds = time.perf_counter() - t
            t = time.perf_counter()
            pred = clf.predict(fv)
            elapsed = time.perf_counter() - t
            sm = score(yval, pred, labels)
            config = {"model": name, "topics": topics, "C": c,
                      "feature_fit_and_transform_seconds": feature_seconds,
                      "classifier_fit_seconds": fit_seconds,
                      "validation_classifier_ms_per_document": elapsed * 1000 / len(val), **sm}
            candidates.append(config)
            rank = (sm["macro_f1"], sm["accuracy"], -elapsed / len(val))
            if name not in chosen or rank > chosen[name]["rank"]:
                chosen[name] = {"rank": rank, "config": config, "features": feat, "classifier": clf}
                joblib.dump({"features": feat, "classifier": clf, "config": config}, out / f"{name}.joblib", compress=3)
            print(f"  {name} topics={topics} C={c}: validation macro-F1={sm['macro_f1']:.4f}", flush=True)
        if name == "doc2vec":
            doc_vectors = ft
        # Checkpoint candidates so a later failed experiment does not lose results.
        write_csv(results / "validation_candidates.csv", candidates)
    # Validation inference time includes text transformation (not just LR).
    for name, item in chosen.items():
        t = time.perf_counter()
        item["classifier"].predict(item["features"].transform(xval))
        ms = (time.perf_counter() - t) * 1000 / len(val)
        item["config"]["validation_ms_per_document"] = ms
        item["rank"] = (item["config"]["macro_f1"], item["config"]["accuracy"], -ms)
        joblib.dump({"features": item["features"], "classifier": item["classifier"], "config": item["config"]}, out / f"{name}.joblib", compress=3)
    winner = max(chosen, key=lambda n: chosen[n]["rank"])
    frozen = {"winner": winner, "selection_metric": "validation macro-F1, accuracy, then end-to-end inference latency",
              "models": {n: item["config"] for n, item in chosen.items()}}
    dump_json(results / "selection.json", frozen)
    dump_json(out / "selection.json", frozen)
    # No tuning below this point. Test documents enter feature transforms only.
    test = splits["test"]
    xtest = [r["text"] for r in test]
    ytest = np.array([r["label"] for r in test])
    majority = Counter(ytrain).most_common(1)[0][0]
    metrics = {"majority": score(ytest, np.full(len(test), majority), labels)}
    predictions = [{"id": r["id"], "true_label": r["label"], "true_category": r["category"]} for r in test]
    for name, item in chosen.items():
        t = time.perf_counter()
        features = item["features"].transform(xtest)
        pred = item["classifier"].predict(features)
        elapsed = time.perf_counter() - t
        metrics[name] = {**score(ytest, pred, labels), "test_ms_per_document": elapsed * 1000 / len(test),
                         "train_seconds": item["config"]["feature_fit_and_transform_seconds"] + item["config"]["classifier_fit_seconds"]}
        report = classification_report(ytest, pred, labels=labels, target_names=label_names, output_dict=True, zero_division=0)
        dump_json(results / f"{name}_classification_report.json", report)
        write_csv(results / f"{name}_per_category.csv", [{"category": category, **report[category]} for category in label_names])
        cm = confusion_matrix(ytest, pred, labels=labels)
        write_csv(results / f"{name}_confusion_matrix.csv", [{"true_category": cat, **dict(zip(label_names, map(int, cm[i])))} for i, cat in enumerate(label_names)])
        fig, ax = plt.subplots(figsize=(11, 9))
        normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), label_names, rotation=75, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), label_names, fontsize=9)
        ax.set_xlabel("Predicted category")
        ax.set_ylabel("True category")
        ax.set_title(f"{name}: row-normalized test confusion matrix")
        fig.colorbar(im, ax=ax, label="Fraction within true category")
        fig.tight_layout()
        fig.savefig(results / f"{name}_confusion_matrix.png", dpi=160)
        plt.close(fig)
        for p, v in zip(predictions, pred):
            p[f"{name}_label"] = int(v)
        print(f"TEST {name}: {json.dumps(metrics[name])}", flush=True)
    write_csv(results / "test_predictions.csv", predictions)
    norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    np.save(out / "retrieval_vectors.npy", doc_vectors / np.maximum(norms, 1e-12))
    dump_json(out / "retrieval_cases.json", [{k: r[k] for k in ("id", "case_name", "decision_date", "category", "citation")} | {"excerpt": r["text"][:1600]} for r in train])
    examples = []
    for label in labels:
        r = next(r for r in test if r["label"] == label)
        examples.append({"name": f"Held-out: {r['category']}", "case_name": r["case_name"], "text": r["text"][:20000], "true_category": r["category"]})
    dump_json(out / "examples.json", examples)
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "gensim", "joblib")}
    bundle_manifest = {"format_version": 1, "seed": 42, "winner": winner,
                       "label_mapping": {str(i): ISSUE_AREAS[i] for i in labels},
                       "versions": versions, "python": platform.python_version(),
                       "dataset_sha256": provenance["sha256"],
                       "elapsed_seconds": time.perf_counter() - started,
                       "test_evaluation": "Frozen model comparison, not used for selection"}
    dump_json(out / "manifest.json", bundle_manifest)
    dump_json(results / "metrics.json", metrics)
    for file in ("metrics.json", "dataset_summary.json", "provenance.json", "split_manifest.csv", "validation_candidates.csv"):
        (out / file).write_bytes((results / file).read_bytes())
    names = list(metrics)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xx = np.arange(len(names))
    ax.bar(xx - 0.18, [metrics[n]["accuracy"] for n in names], width=.36, label="Accuracy", color="#246B83")
    ax.bar(xx + 0.18, [metrics[n]["macro_f1"] for n in names], width=.36, label="Macro-F1", color="#CE9C46")
    ax.set_xticks(xx, ["Majority", "TF-IDF + LR", "LDA + LR", "Doc2Vec + LR"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out test score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / "model_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Completed. Validation-selected model: {winner}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.project_root, args.force)


if __name__ == "__main__":
    main()
