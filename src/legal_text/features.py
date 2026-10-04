"""Model-specific features. Every fit method receives training texts only."""
from __future__ import annotations

import hashlib
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from .data import tokens


def stable_hash(value):
    if isinstance(value, str):
        value = value.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


class TfidfFeatures:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                                         min_df=3, max_df=0.95, sublinear_tf=True,
                                         stop_words="english", dtype=np.float32)
    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self
    def transform(self, texts):
        return self.vectorizer.transform(texts)
    def recognized(self, text):
        return self.transform([text]).nnz > 0


class LdaFeatures:
    def __init__(self, n_topics, seed=42):
        self.vectorizer = CountVectorizer(max_features=15000, min_df=5,
                                         max_df=0.95, stop_words="english")
        self.model = LatentDirichletAllocation(n_components=n_topics,
                      learning_method="online", max_iter=5, batch_size=128,
                      max_doc_update_iter=30, random_state=seed, n_jobs=1,
                      evaluate_every=-1)
    def fit(self, texts):
        counts = self.vectorizer.fit_transform(texts)
        self.model.fit(counts)
        return self
    def transform(self, texts):
        return self.model.transform(self.vectorizer.transform(texts))
    def recognized(self, text):
        return self.vectorizer.transform([text]).nnz > 0


class Doc2VecFeatures:
    def __init__(self, seed=42):
        self.seed = seed
        self.model = Doc2Vec(vector_size=100, dm=0, dbow_words=0, epochs=30,
                            min_count=5, negative=5, sample=1e-4,
                            workers=1, seed=seed, hashfxn=stable_hash,
                            max_vocab_size=100000)
    def fit(self, texts):
        documents = [TaggedDocument(tokens(text), [i]) for i, text in enumerate(texts)]
        self.model.build_vocab(documents)
        self.model.train(documents, total_examples=len(documents), epochs=30)
        return self
    def transform(self, texts):
        vectors = []
        for text in texts:
            words = tokens(text)
            # infer_vector is stochastic. Reset to a text-specific seed so saved
            # models, repeat queries, and notebook runs return the same vector.
            self.model.random.seed(stable_hash(" ".join(words)) ^ self.seed)
            vectors.append(self.model.infer_vector(words, epochs=30))
        return np.asarray(vectors, dtype=np.float32)
    def recognized(self, text):
        return any(word in self.model.wv.key_to_index for word in tokens(text))

