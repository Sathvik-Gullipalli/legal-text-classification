import joblib
import numpy as np
from legal_text.features import TfidfFeatures, LdaFeatures, Doc2VecFeatures


TRAIN = ["criminal court evidence accused trial" for _ in range(6)] + ["taxation income revenue federal statute" for _ in range(6)]


def test_vectorizers_cannot_learn_validation_vocabulary():
    for feat in (TfidfFeatures(), LdaFeatures(2)):
        feat.fit(TRAIN)
        before = dict(feat.vectorizer.vocabulary_)
        feat.transform(["validationonlysentinel income"])
        assert "validationonlysentinel" not in feat.vectorizer.vocabulary_
        assert before == feat.vectorizer.vocabulary_
        assert not feat.recognized("validationonlysentinel")


def test_doc2vec_inference_repeatable_and_save_load_consistent(tmp_path):
    feat = Doc2VecFeatures().fit(TRAIN)
    a = feat.transform([TRAIN[0]])
    b = feat.transform([TRAIN[0]])
    np.testing.assert_array_equal(a, b)
    path = tmp_path / "doc.joblib"
    joblib.dump(feat, path)
    np.testing.assert_array_equal(a, joblib.load(path).transform([TRAIN[0]]))
    feat.transform(["validationonlysentinel"])
    assert "validationonlysentinel" not in feat.model.wv.key_to_index

