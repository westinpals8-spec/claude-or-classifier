"""Week 6 end-to-end check: run the actual service code on the fixed test set and confirm it reproduces
the offline analysis (same labels, same escalations), and record latency.

Usage (from project root):  python tools/week6_service_check.py
"""
import json, os, sys, time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.pop("ANTHROPIC_API_KEY", None)
from data_prep import LABELS, load_prepared  # noqa
from src.service.predictor import TriageService, THRESHOLDS  # noqa

df, _ = load_prepared(save=False)
test = df[df.split == "test"].reset_index(drop=True)
svc = TriageService(os.path.join(ROOT, "models", "distilbert_best"))
t0 = time.perf_counter(); preds = svc.classify(test.text.tolist()); sec = time.perf_counter() - t0
Y = test[LABELS].values
pred = np.array([[int(l in p.labels) for l in LABELS] for p in preds])
routes = pd.Series([p.route for p in preds]).value_counts().to_dict()
# offline recomputation from saved probabilities, including the service's "at least one label" rule
cal = json.load(open(os.path.join(ROOT, "models", "distilbert_best", "calibration.json")))["temperatures"]
Tm = np.array([cal[l] for l in LABELS]); D = np.array([THRESHOLDS[l] for l in LABELS])
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
db = pd.read_csv(os.path.join(ROOT, "results", "distilbert_test_predictions.csv"))
tf = pd.read_csv(os.path.join(ROOT, "results", "tfidf_tuned_test_predictions.csv"))
E = svc.weight / (1 + np.exp(-lg(db[[f"proba_{l}" for l in LABELS]].values) / Tm)) + (1 - svc.weight) * tf[[f"proba_{l}" for l in LABELS]].values
off = (E >= D).astype(int); empty = off.sum(1) == 0; off[empty, E[empty].argmax(1)] = 1
res = {"service_version": preds[0].model, "ensemble_weight": svc.weight, "band": list(svc.band),
       "macro_f1": f1_score(Y, pred, average="macro"), "subset_acc": float((pred == Y).all(1).mean()),
       "missed_problem_reports": int(((pred[:, 0] == 0) & (Y[:, 0] == 1)).sum()),
       "routes": routes, "escalation_rate": routes.get("human_review", 0) / len(preds),
       "reviews_given_fallback_label": int(empty.sum()),
       "identical_to_offline_recomputation": bool((off == pred).all()),
       "seconds_per_1000_reviews_cpu": 1000 * sec / len(test)}
print(json.dumps(res, indent=2))
json.dump(res, open(os.path.join(ROOT, "results", "week6_service_check.json"), "w"), indent=2, default=float)
