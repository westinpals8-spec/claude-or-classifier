"""Week 6: 5-fold cross-validation and leave-one-app-out testing for DistilBERT and tuned TF-IDF.

Usage (from the project root):  python src/cv_evaluate.py
Runtime: about 2.5 hours on a 2-core CPU (8 DistilBERT trainings).

* Folds: 5-fold stratified (label combination + app) over ALL 5,846 cleaned reviews, seed 42.
  Each review is predicted exactly once by a model that never saw it (out-of-fold predictions).
* DistilBERT uses the Week 4 configuration fixed in advance (lr 5e-5, 3 epochs, batch 16, max_len 128);
  inside each fold, 10% of the fold's training data is held out to pick the best epoch.
* Tuned TF-IDF uses the Week 4 configuration (word 1-2 + char 2-5 n-grams, C=4, balanced weights).
* Leave-one-app-out: train on two apps, test on the third (generalization to an unseen app).
Outputs: results/cv_oof_predictions.csv, results/cv_fold_metrics.json, results/cross_app_metrics.json
"""
from __future__ import annotations

import json, os, sys, time

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import APPS, LABELS, RANDOM_STATE, load_prepared, project_root  # noqa: E402
from train_distilbert import MODEL_NAME, metrics, predict_proba, train_one  # noqa: E402

OUT = os.path.join(project_root(), "results")
CFG = {"name": "lr5e-5", "lr": 5e-5, "pos_weight": False}


def tfidf_proba(tr_text, tr_y, te_text):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=60000)
    A_tr = hstack([w.fit_transform(tr_text), c.fit_transform(tr_text)]).tocsr()
    A_te = hstack([w.transform(te_text), c.transform(te_text)]).tocsr()
    P = np.zeros((len(te_text), len(LABELS)))
    for j in range(len(LABELS)):
        m = LogisticRegression(C=4.0, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=RANDOM_STATE)
        P[:, j] = m.fit(A_tr, tr_y[:, j]).predict_proba(A_te)[:, 1]
    return P


def distilbert_proba(tok, tr: pd.DataFrame, te: pd.DataFrame):
    sub, va = train_test_split(tr, test_size=0.1, random_state=RANDOM_STATE, stratify=tr["label_combo"])
    model, best, _ = train_one(CFG, tok, sub["text"].tolist(), sub[LABELS].values, va["text"].tolist(), va[LABELS].values)
    return predict_proba(model, tok, te["text"].tolist()), best["epoch"]


def main():
    df, _ = load_prepared(save=False)
    df = df.reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    strat = df["label_combo"] + "|" + df["app"]
    strat = strat.where(strat.map(strat.value_counts()) >= 5, df["label_combo"])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(df, strat))

    oof = {m: np.zeros((len(df), len(LABELS))) for m in ("tfidf", "distilbert")}
    fold_id = np.zeros(len(df), dtype=int)
    fold_rows = []
    for k, (tr_i, te_i) in enumerate(folds):
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        fold_id[te_i] = k
        t0 = time.perf_counter()
        oof["tfidf"][te_i] = tfidf_proba(tr["text"].values, tr[LABELS].values, te["text"].values)
        oof["distilbert"][te_i], ep = distilbert_proba(tok, tr, te)
        for m in oof:
            s, _ = metrics(te[LABELS].values, oof[m][te_i], 0.5)
            fold_rows.append({"fold": k, "model": m, **s, **({"best_epoch": ep} if m == "distilbert" else {})})
        print(f"fold {k} done in {(time.perf_counter() - t0) / 60:.1f} min",
              {r["model"]: round(r["macro_f1"], 4) for r in fold_rows[-2:]}, flush=True)
        json.dump(fold_rows, open(os.path.join(OUT, "cv_fold_metrics.json"), "w"), indent=2, default=float)

    out = df[["review_id", "app", "text"] + LABELS].copy()
    out["fold"] = fold_id
    for m in oof:
        for j, l in enumerate(LABELS):
            out[f"{m}_proba_{l}"] = oof[m][:, j]
    out.to_csv(os.path.join(OUT, "cv_oof_predictions.csv"), index=False)

    cross = []
    for held in APPS:
        tr, te = df[df.app != held], df[df.app == held]
        t0 = time.perf_counter()
        P_tf = tfidf_proba(tr["text"].values, tr[LABELS].values, te["text"].values)
        P_db, ep = distilbert_proba(tok, tr, te)
        for m, P in (("tfidf", P_tf), ("distilbert", P_db)):
            s, _ = metrics(te[LABELS].values, P, 0.5)
            cross.append({"held_out_app": held, "model": m, "n_test": len(te), **s})
        print(f"cross-app {held} done in {(time.perf_counter() - t0) / 60:.1f} min",
              {r["model"]: round(r["macro_f1"], 4) for r in cross[-2:]}, flush=True)
        json.dump(cross, open(os.path.join(OUT, "cross_app_metrics.json"), "w"), indent=2, default=float)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
