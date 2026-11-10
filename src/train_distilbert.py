"""Week 4: fine-tune DistilBERT for multi-label app feedback triage.

Usage (from the project root):
    python src/train_distilbert.py

What it does
  * Uses the same cleaned data and fixed train/test split as every other model (src/data_prep.py).
  * Holds out 10% of the training set as a validation set for model selection.
  * Runs a small hyperparameter search (learning rate, positive-class weighting),
    picks the best epoch by validation macro F1, tunes one decision threshold per label
    on the validation set, then scores the held-out test set once.
  * Saves metrics, run logs, test-set probabilities, and the best model.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import LABELS, RANDOM_STATE, load_prepared, project_root  # noqa: E402

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH = 16
EPOCHS = 3
RUNS = [  # hyperparameter configurations compared on the validation set (CPU-only budget)
    {"name": "lr3e-5", "lr": 3e-5, "pos_weight": False},
    {"name": "lr5e-5", "lr": 5e-5, "pos_weight": False},
]
OUT = os.path.join(project_root(), "results")
MODEL_DIR = os.path.join(project_root(), "models", "distilbert_best")


def set_seed(seed=RANDOM_STATE):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def batches(tok, texts, labels=None, shuffle=False):
    """Length-bucketed batches: reviews of similar length are batched together to cut padding
    (a big speed-up on CPU). Batch order is shuffled each epoch when shuffle=True."""
    order = np.argsort([len(t) for t in texts], kind="stable")
    chunks = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    rng = np.random.default_rng(RANDOM_STATE)

    class Loader:
        def __len__(self):
            return len(chunks)

        def __iter__(self):
            seq = rng.permutation(len(chunks)) if shuffle else range(len(chunks))
            for c in seq:
                idx = chunks[c]
                enc = tok([texts[i] for i in idx], truncation=True, max_length=MAX_LEN,
                          padding=True, return_tensors="pt")
                if labels is not None:
                    enc["labels"] = torch.tensor(labels[idx], dtype=torch.float)
                yield enc

    return Loader()


@torch.no_grad()
def predict_proba(model, tok, texts):
    model.eval(); out = np.zeros((len(texts), len(LABELS)))
    order = np.argsort([len(t) for t in texts], kind="stable")
    for i in range(0, len(order), 64):
        idx = order[i:i + 64]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        out[idx] = torch.sigmoid(model(**enc).logits).numpy()
    return out


def metrics(Y, P, thr=0.5):
    pred = (P >= np.asarray(thr)).astype(int)
    p, r, f, _ = precision_recall_fscore_support(Y, pred, zero_division=0)
    m = {"macro_f1": f1_score(Y, pred, average="macro", zero_division=0),
         "micro_f1": f1_score(Y, pred, average="micro", zero_division=0),
         "subset_acc": accuracy_score(Y, pred), "hamming": hamming_loss(Y, pred)}
    for i, l in enumerate(LABELS):
        m.update({f"{l}_p": p[i], f"{l}_r": r[i], f"{l}_f1": f[i]})
    return m, pred


def train_one(cfg, tok, tr_x, tr_y, va_x, va_y):
    set_seed()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), problem_type="multi_label_classification")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    loader = batches(tok, tr_x, tr_y, shuffle=True)
    steps = len(loader) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    if cfg["pos_weight"]:
        pos = tr_y.sum(0); pw = torch.tensor((len(tr_y) - pos) / pos, dtype=torch.float)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    best = {"val_macro_f1": -1}
    history = []
    for ep in range(1, EPOCHS + 1):
        model.train(); t0 = time.perf_counter(); total = 0.0
        for step, b in enumerate(loader):
            y = b.pop("labels")
            loss = loss_fn(model(**b).logits, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); total += loss.item()
            if step % 50 == 0:
                print(f"  [{cfg['name']}] epoch {ep} step {step}/{len(loader)} loss {loss.item():.4f}", flush=True)
        Pv = predict_proba(model, tok, va_x)
        vm, _ = metrics(va_y, Pv)
        rec = {"epoch": ep, "train_loss": total / len(loader), "val_macro_f1": vm["macro_f1"],
               "val_feature_request_f1": vm["feature_request_f1"], "epoch_seconds": time.perf_counter() - t0}
        history.append(rec); print(" ", rec, flush=True)
        if vm["macro_f1"] > best["val_macro_f1"]:
            best = {"val_macro_f1": vm["macro_f1"], "epoch": ep,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}, "Pv": Pv}
    model.load_state_dict(best["state"])
    return model, best, history


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    df, _ = load_prepared(save=False)
    train = df[df.split == "train"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    tr, va = train_test_split(train, test_size=0.1, random_state=RANDOM_STATE, stratify=train["label_combo"])
    tr_x, tr_y = tr["text"].tolist(), tr[LABELS].values
    va_x, va_y = va["text"].tolist(), va[LABELS].values
    te_x, te_y = test["text"].tolist(), test[LABELS].values
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    lens = [len(tok(t)["input_ids"]) for t in train["text"]]
    print(f"train {len(tr)} | val {len(va)} | test {len(test)} | tokens median {np.median(lens):.0f}, "
          f"95th pct {np.percentile(lens, 95):.0f}, truncated at {MAX_LEN}: {np.mean(np.array(lens) > MAX_LEN):.2%}", flush=True)

    runs, best_run = [], None
    for cfg in RUNS:
        cfg = dict(cfg)
        t0 = time.perf_counter()
        model, best, hist = train_one(cfg, tok, tr_x, tr_y, va_x, va_y)
        rec = {**cfg, "best_epoch": best["epoch"], "best_val_macro_f1": best["val_macro_f1"],
               "train_minutes": (time.perf_counter() - t0) / 60, "history": hist}
        runs.append(rec)
        json.dump(runs, open(os.path.join(OUT, "distilbert_runs.json"), "w"), indent=2, default=float)
        if best_run is None or rec["best_val_macro_f1"] > best_run[0]["best_val_macro_f1"]:
            best_run = (rec, model, best["Pv"])
        else:
            del model

    rec, model, Pv = best_run
    grid = np.round(np.arange(0.2, 0.81, 0.025), 3)
    thr = [float(grid[int(np.argmax([f1_score(va_y[:, j], (Pv[:, j] >= t).astype(int)) for t in grid]))])
           for j in range(len(LABELS))]

    t0 = time.perf_counter(); Pt = predict_proba(model, tok, te_x); inf = time.perf_counter() - t0
    m05, _ = metrics(te_y, Pt, 0.5)
    mt, pred = metrics(te_y, Pt, thr)
    summary = {"model": MODEL_NAME, "max_len": MAX_LEN, "batch": BATCH, "epochs": EPOCHS,
               "selected_run": {k: v for k, v in rec.items() if k != "history"},
               "thresholds": dict(zip(LABELS, thr)), "test_thr05": m05, "test_tuned_thr": mt,
               "inference_seconds_per_1000_cpu": 1000 * inf / len(te_x), "api_cost_usd_per_1000": 0.0,
               "hardware": f"CPU only, {torch.get_num_threads()} threads"}
    json.dump(summary, open(os.path.join(OUT, "distilbert_metrics.json"), "w"), indent=2, default=float)
    out = test[["review_id", "app", "text"] + LABELS].copy()
    for j, l in enumerate(LABELS):
        out[f"proba_{l}"] = Pt[:, j]; out[f"pred_{l}"] = pred[:, j]
    out.to_csv(os.path.join(OUT, "distilbert_test_predictions.csv"), index=False)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR); tok.save_pretrained(MODEL_DIR)
    json.dump({"thresholds": dict(zip(LABELS, thr))}, open(os.path.join(MODEL_DIR, "thresholds.json"), "w"))
    print("DONE", json.dumps({"test_macro_f1_tuned": mt["macro_f1"], "test_macro_f1_0.5": m05["macro_f1"]}), flush=True)


if __name__ == "__main__":
    main()
