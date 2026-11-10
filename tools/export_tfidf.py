"""Fit the tuned TF-IDF model (Week 4 settings) on the fixed training split and save it for the service ensemble.

Usage (from project root):  python tools/export_tfidf.py   -> models/tfidf_tuned.joblib
"""
import os, sys

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data_prep import LABELS, RANDOM_STATE, load_prepared  # noqa

df, _ = load_prepared(save=False)
train = df[df.split == "train"]
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode").fit(train.text)
c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=60000).fit(train.text)
A = hstack([w.transform(train.text), c.transform(train.text)]).tocsr()
models = [LogisticRegression(C=4.0, class_weight="balanced", solver="liblinear", max_iter=3000,
                             random_state=RANDOM_STATE).fit(A, train[l].values) for l in LABELS]
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
path = os.path.join(ROOT, "models", "tfidf_tuned.joblib")
joblib.dump({"word": w, "char": c, "models": models, "labels": LABELS}, path, compress=3)
test = df[df.split == "test"]
At = hstack([w.transform(test.text), c.transform(test.text)]).tocsr()
P = np.column_stack([m.predict_proba(At)[:, 1] for m in models])
print("saved", path, f"{os.path.getsize(path) / 1e6:.1f} MB", "| check vs saved test probabilities:",
      np.abs(P - __import__("pandas").read_csv(os.path.join(ROOT, "results", "tfidf_tuned_test_predictions.csv"))[[f"proba_{l}" for l in LABELS]].values).max())
