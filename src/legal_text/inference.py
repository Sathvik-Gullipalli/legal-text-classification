"""Saved-model classification and training-corpus similarity retrieval."""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from .data import tokens


class Demo:
    def __init__(self, artifact_dir="artifacts"):
        self.path = Path(artifact_dir)
        required = ["manifest.json", "retrieval_vectors.npy", "retrieval_cases.json",
                    "tfidf.joblib", "lda.joblib", "doc2vec.joblib"]
        missing = [f for f in required if not (self.path / f).exists()]
        if missing:
            raise FileNotFoundError("Model bundle is missing: " + ", ".join(missing) +
                ". Extract team9-model-bundle.zip into the project folder, or run the training notebook.")
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.models = {n: joblib.load(self.path / f"{n}.joblib") for n in ("tfidf", "lda", "doc2vec")}
        self.vectors = np.load(self.path / "retrieval_vectors.npy")
        self.cases = json.loads((self.path / "retrieval_cases.json").read_text())
        self.examples = json.loads((self.path / "examples.json").read_text())
        if len(self.vectors) != len(self.cases):
            raise ValueError("Retrieval index and metadata have different row counts. Use the original complete bundle.")

    @staticmethod
    def _validate(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a legal opinion or select an example before predicting.")
        if not tokens(text):
            raise ValueError("The input contains no recognizable English words.")

    def predict(self, text, model_name=None):
        self._validate(text)
        name = model_name or self.manifest["winner"]
        if name not in self.models:
            raise ValueError("Choose one of tfidf, lda, or doc2vec.")
        model = self.models[name]
        feat, clf = model["features"], model["classifier"]
        if not feat.recognized(text):
            raise ValueError("No words in this input match the selected model's training vocabulary. Try a longer English legal passage.")
        with threadpool_limits(limits=1):
            x = feat.transform([text])
            probabilities = clf.predict_proba(x)[0]
        if not np.isfinite(probabilities).all():
            raise ValueError("The model returned non-finite probabilities.")
        order = np.argsort(-probabilities, kind="stable")[:3]
        classes = clf.classes_
        mapping = self.manifest["label_mapping"]
        top = [{"label": int(classes[i]), "category": mapping[str(classes[i])],
                "probability": float(probabilities[i])} for i in order]
        warnings = ["Model probabilities are uncalibrated estimates."]
        if len(tokens(text)) < 200:
            warnings.append("Short passage: training examples are full opinions, so this prediction may be less representative.")
        terms = []
        if name == "tfidf":
            predicted_index = int(order[0])
            # sklearn uses one coefficient row for binary LR, two class labels.
            weights = clf.coef_[predicted_index] if len(classes) > 2 else clf.coef_[0] * (1 if predicted_index else -1)
            indices = x.indices
            contributions = weights[indices] * x.data
            vocab = feat.vectorizer.get_feature_names_out()
            for j in np.argsort(-contributions)[:10]:
                if contributions[j] > 0:
                    terms.append({"term": str(vocab[indices[j]]), "contribution": float(contributions[j])})
        return {"model": name, "label": top[0]["label"], "category": top[0]["category"],
                "top_three": top, "warnings": warnings, "influential_terms": terms}

    def retrieve_similar(self, text, k=5):
        self._validate(text)
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        feat = self.models["doc2vec"]["features"]
        if not feat.recognized(text):
            raise ValueError("No query words occur in the Doc2Vec vocabulary.")
        with threadpool_limits(limits=1):
            vector = feat.transform([text])[0]
            norm = np.linalg.norm(vector)
            if norm < 1e-12:
                raise ValueError("Could not produce a meaningful vector for this query.")
            scores = self.vectors @ (vector / norm)
        order = np.argsort(-scores, kind="stable")
        # Multiple records belonging to a single citation should not occupy all
        # five results. Return one opinion per non-empty citation.
        seen = set()
        results = []
        for i in order:
            case = self.cases[int(i)]
            key = case["citation"] or case["id"]
            if key in seen:
                continue
            seen.add(key)
            results.append({**case, "similarity": float(scores[i])})
            if len(results) == k:
                break
        return results


_demo = None


def load(artifact_dir="artifacts"):
    global _demo
    _demo = Demo(artifact_dir)
    return _demo


def predict(text, model_name=None):
    global _demo
    if _demo is None:
        load()
    return _demo.predict(text, model_name)


def retrieve_similar(text, k=5):
    global _demo
    if _demo is None:
        load()
    return _demo.retrieve_similar(text, k)

