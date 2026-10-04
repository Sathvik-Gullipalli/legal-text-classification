import json
from pathlib import Path
import numpy as np
import pytest
from legal_text.inference import Demo


ROOT = Path(__file__).resolve().parents[1]


def test_missing_bundle_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model bundle is missing"):
        Demo(tmp_path)


@pytest.fixture(scope="module")
def demo():
    if not (ROOT / "artifacts/manifest.json").exists():
        pytest.skip("Run training or extract the model bundle to verify saved models")
    return Demo(ROOT / "artifacts")


@pytest.mark.parametrize("name", ["tfidf", "lda", "doc2vec"])
def test_predictions_finite_and_labels_mapped(demo, name):
    a = demo.predict(demo.examples[0]["text"], name)
    b = demo.predict(demo.examples[0]["text"], name)
    assert a == b
    assert len(a["top_three"]) == 3
    assert a["category"] == demo.manifest["label_mapping"][str(a["label"])]
    assert all(np.isfinite(r["probability"]) and 0 <= r["probability"] <= 1 for r in a["top_three"])
    assert [r["probability"] for r in a["top_three"]] == sorted([r["probability"] for r in a["top_three"]], reverse=True)


@pytest.mark.parametrize("text", ["", "   ", "1234 !!"])
def test_invalid_input(demo, text):
    with pytest.raises(ValueError):
        demo.predict(text)


@pytest.mark.parametrize("name", ["tfidf", "lda", "doc2vec"])
def test_out_of_vocabulary(demo, name):
    with pytest.raises(ValueError, match="No words"):
        demo.predict("qxzvptzzqqqqqq", name)


def test_short_passage_warning(demo):
    assert any("Short passage" in w for w in demo.predict("criminal evidence court trial")['warnings'])


def test_retrieval_sorted_unique_training_only(demo):
    results = demo.retrieve_similar(demo.examples[0]["text"])
    assert len(results) == 5
    assert all(np.isfinite(r["similarity"]) for r in results)
    assert [r["similarity"] for r in results] == sorted([r["similarity"] for r in results], reverse=True)
    ids = {r["id"] for r in demo.cases}
    assert all(r["id"] in ids for r in results)
    assert len({r["citation"] or r["id"] for r in results}) == 5
    with pytest.raises(ValueError):
        demo.retrieve_similar("criminal evidence", 0)

