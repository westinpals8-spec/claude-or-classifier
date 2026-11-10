"""Week 6 analysis: A/B test, calibration, error analysis, and reliability checks.

Usage (from project root, after src/cv_evaluate.py has finished):  python tools/week6_analysis.py
Outputs: results/week6_analysis.json, results/week6_error_samples.csv, results/figures/week6_*.png
"""
import json, os, sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest, ttest_rel
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from data_prep import LABELS, APPS, RANDOM_STATE, load_prepared  # noqa
R = os.path.join(ROOT, "results"); FIG = os.path.join(R, "figures")
DEPLOY = np.array([0.50, 0.475, 0.50])          # service thresholds (Week 5)
TF_THR = np.array([0.475, 0.525, 0.475])         # tuned TF-IDF thresholds (Week 4)
BAND = (0.25, 0.75)
rng = np.random.default_rng(RANDOM_STATE)
out = {}

# ---------------------------------------------------------------- 1. A/B test on the fixed test set
db = pd.read_csv(os.path.join(R, "distilbert_test_predictions.csv"))
tf = pd.read_csv(os.path.join(R, "tfidf_tuned_test_predictions.csv"))
assert (db.review_id.values == tf.review_id.values).all()
Y = db[LABELS].values
P_db = db[[f"proba_{l}" for l in LABELS]].values
P_tf = tf[[f"proba_{l}" for l in LABELS]].values
pred_db, pred_tf = (P_db >= DEPLOY).astype(int), (P_tf >= TF_THR).astype(int)
ok_db, ok_tf = (pred_db == Y).all(1), (pred_tf == Y).all(1)

b, c = int((ok_db & ~ok_tf).sum()), int((~ok_db & ok_tf).sum())   # discordant pairs
mcnemar_p = binomtest(b, b + c, 0.5).pvalue                          # exact McNemar test
diffs = []
for _ in range(2000):
    i = rng.integers(0, len(Y), len(Y))
    diffs.append(f1_score(Y[i], pred_db[i], average="macro", zero_division=0) - f1_score(Y[i], pred_tf[i], average="macro", zero_division=0))
esc = ((P_db > BAND[0]) & (P_db < BAND[1])).any(1)
pred_casc = pred_db.copy(); pred_casc[esc] = Y[esc]      # SIMULATED: a human corrects every escalated review
out["ab_test"] = {
    "n": len(Y),
    "macro_f1": {"A_tfidf_tuned": f1_score(Y, pred_tf, average="macro"), "B_distilbert": f1_score(Y, pred_db, average="macro"),
                 "C_distilbert_plus_human_review_simulated": f1_score(Y, pred_casc, average="macro")},
    "subset_acc": {"A_tfidf_tuned": ok_tf.mean(), "B_distilbert": ok_db.mean(), "C_simulated": (pred_casc == Y).all(1).mean()},
    "mcnemar": {"distilbert_right_tfidf_wrong": b, "tfidf_right_distilbert_wrong": c, "p_value": mcnemar_p},
    "macro_f1_diff_bootstrap_95ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    "escalation_rate": float(esc.mean()), "human_reviews_per_1000": float(1000 * esc.mean()),
}
print("A/B", json.dumps(out["ab_test"], default=float, indent=1))

# ---------------------------------------------------------------- 2. Cross-validation (paired by fold)
fm = pd.DataFrame(json.load(open(os.path.join(R, "cv_fold_metrics.json"))))
cv = fm.pivot(index="fold", columns="model", values="macro_f1")
t = ttest_rel(cv["distilbert"], cv["tfidf"])
out["cross_validation"] = {
    "per_fold_macro_f1": cv.round(4).to_dict(),
    "mean_sd": {m: [float(cv[m].mean()), float(cv[m].std(ddof=1))] for m in cv},
    "paired_t_test": {"t": float(t.statistic), "p": float(t.pvalue)},
    "per_label_f1_mean": fm.groupby("model")[[f"{l}_f1" for l in LABELS]].mean().round(4).to_dict(),
    "distilbert_best_epochs": fm[fm.model == "distilbert"]["best_epoch"].tolist(),
}
ca = pd.DataFrame(json.load(open(os.path.join(R, "cross_app_metrics.json"))))
out["leave_one_app_out"] = ca.pivot(index="held_out_app", columns="model", values="macro_f1").round(4).to_dict()
out["leave_one_app_out_feature_request_f1"] = ca.pivot(index="held_out_app", columns="model", values="feature_request_f1").round(4).to_dict()
print("CV", out["cross_validation"]["mean_sd"], out["cross_validation"]["paired_t_test"], out["leave_one_app_out"])

# ---------------------------------------------------------------- 3. Error analysis on out-of-fold predictions (all 5,846 reviews)
oof = pd.read_csv(os.path.join(R, "cv_oof_predictions.csv"))
Yo = oof[LABELS].values
Po = oof[[f"distilbert_proba_{l}" for l in LABELS]].values
Pt = oof[[f"tfidf_proba_{l}" for l in LABELS]].values
pr = (Po >= DEPLOY).astype(int)
err = ~(pr == Yo).all(1)
words = oof.text.str.split().str.len()
conf_wrong = ((pr != Yo) & ((Po > 0.95) | (Po < 0.05))).any(1)            # model very sure, label disagrees
both_wrong_conf = conf_wrong & ~((Pt >= TF_THR) == Yo).all(1)               # and TF-IDF disagrees with the label too
nonascii = oof.text.map(lambda s: sum(ord(ch) > 127 for ch in s) / max(len(s), 1) > 0.2)
cats = pd.DataFrame({
    "mixed_problem_and_request": (Yo[:, 0] == 1) & (Yo[:, 1] == 1),
    "predicted_mixed_but_single": (pr[:, 0] == 1) & (pr[:, 1] == 1) & ~((Yo[:, 0] == 1) & (Yo[:, 1] == 1)),
    "problem_vs_request_confusion": ((Yo[:, 0] != pr[:, 0]) & (Yo[:, 1] != pr[:, 1])),
    "missed_problem_report": (Yo[:, 0] == 1) & (pr[:, 0] == 0),
    "false_feature_request": (Yo[:, 1] == 0) & (pr[:, 1] == 1),
    "missed_feature_request": (Yo[:, 1] == 1) & (pr[:, 1] == 0),
    "very_short_<=3_words": words <= 3,
    "long_>50_words": words > 50,
    "emoji_or_non_english": nonascii,
    "confident_disagreement_both_models_(likely_label_noise)": both_wrong_conf,
})
summary = []
for name in cats:
    m = cats[name].values
    summary.append({"category": name, "reviews": int(m.sum()), "errors": int((err & m).sum()),
                    "share_of_all_errors": float((err & m).sum() / err.sum()),
                    "error_rate_in_category": float((err & m).sum() / max(m.sum(), 1))})
out["error_analysis"] = {"n_reviews": len(oof), "n_errors": int(err.sum()), "error_rate": float(err.mean()),
                         "categories": summary,
                         "error_rate_by_app": {a: float(err[(oof.app == a).values].mean()) for a in APPS}}
for s in summary: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
samp = oof[err].assign(**{f"pred_{l}": pr[err][:, j] for j, l in enumerate(LABELS)},
                        **{f"p_{l}": Po[err][:, j].round(3) for j, l in enumerate(LABELS)},
                        likely_label_noise=both_wrong_conf[err])
samp.sample(60, random_state=RANDOM_STATE).to_csv(os.path.join(R, "week6_error_samples.csv"), index=False)
oof[both_wrong_conf][["review_id", "app", "text"] + LABELS].to_csv(os.path.join(R, "week6_suspected_label_noise.csv"), index=False)

# ---------------------------------------------------------------- 4. Calibration: per-label temperature scaling
# fit on out-of-fold predictions from folds 0-3, checked on fold 4 and on the fixed test set
def ece(P, Y, bins=10):
    e = 0.0
    for j in range(P.shape[1]):
        p, y = P[:, j], Y[:, j]
        edges = np.linspace(0, 1, bins + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
            if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e / P.shape[1]

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def scale(P, T): return 1 / (1 + np.exp(-logit(P) / np.asarray(T)))
def nll(p, y): p = np.clip(p, 1e-6, 1 - 1e-6); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
fit, chk = (oof.fold != 4).values, (oof.fold == 4).values
grid = np.round(np.arange(0.5, 3.01, 0.05), 2)
T = [float(grid[np.argmin([nll(scale(Po[fit, j], g), Yo[fit, j]) for g in grid])]) for j in range(len(LABELS))]
Po_c, P_db_c = scale(Po, T), scale(P_db, T)
json.dump({"temperatures": dict(zip(LABELS, T)), "fit_on": "out-of-fold DistilBERT predictions, CV folds 0-3 (Week 6)"},
          open(os.path.join(ROOT, "models", "distilbert_best", "calibration.json"), "w"), indent=2)
out["calibration"] = {"temperatures": dict(zip(LABELS, T)),
    "ece_oof_fold4": {"before": ece(Po[chk], Yo[chk]), "after": ece(Po_c[chk], Yo[chk])},
    "ece_test": {"before": ece(P_db, Y), "after": ece(P_db_c, Y)},
    "test_macro_f1_after_scaling": f1_score(Y, (P_db_c >= DEPLOY).astype(int), average="macro"),
    "test_escalation_rate_after_scaling": float(((P_db_c > BAND[0]) & (P_db_c < BAND[1])).any(1).mean())}
print("calibration", out["calibration"])

# ---------------------------------------------------------------- 5. Reliability fix: calibrated DistilBERT + TF-IDF ensemble
# weight chosen on out-of-fold predictions (never on the test set), then confirmed once on the fixed test set
wgrid = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
blend = lambda Pd, Pt_, w: w * Pd + (1 - w) * Pt_
oof_scores = {w: f1_score(Yo, (blend(Po_c, Pt, w) >= DEPLOY).astype(int), average="macro") for w in wgrid}
fold_scores = {w: [f1_score(Yo[(oof.fold == k).values], (blend(Po_c, Pt, w)[(oof.fold == k).values] >= DEPLOY).astype(int), average="macro") for k in range(5)] for w in wgrid}
w_best = max(oof_scores, key=oof_scores.get)
P_ens = blend(P_db_c, P_tf, w_best)
pred_ens = (P_ens >= DEPLOY).astype(int)
ok_ens = (pred_ens == Y).all(1)
b2, c2 = int((ok_ens & ~ok_db).sum()), int((~ok_ens & ok_db).sum())
esc_ens = ((P_ens > BAND[0]) & (P_ens < BAND[1])).any(1)
pred_ens_h = pred_ens.copy(); pred_ens_h[esc_ens] = Y[esc_ens]
json.dump({"weight_distilbert": w_best, "tfidf_model": "models/tfidf_tuned.joblib"},
          open(os.path.join(ROOT, "models", "distilbert_best", "ensemble.json"), "w"), indent=2)
out["ensemble"] = {"weight_distilbert": w_best, "oof_macro_f1_by_weight": oof_scores, "oof_fold_macro_f1_by_weight": fold_scores,
    "test": {"macro_f1": f1_score(Y, pred_ens, average="macro"), "subset_acc": ok_ens.mean(),
             "per_label_f1": dict(zip(LABELS, f1_score(Y, pred_ens, average=None))),
             "problem_report_recall": float(pred_ens[Y[:, 0] == 1, 0].mean()),
             "missed_problem_reports": int(((pred_ens[:, 0] == 0) & (Y[:, 0] == 1)).sum()),
             "escalation_rate": float(esc_ens.mean()), "ece": ece(P_ens, Y),
             "macro_f1_with_human_review_simulated": f1_score(Y, pred_ens_h, average="macro")},
    "test_distilbert_only": {"macro_f1": f1_score(Y, pred_db, average="macro"), "subset_acc": ok_db.mean(),
             "per_label_f1": dict(zip(LABELS, f1_score(Y, pred_db, average=None))),
             "missed_problem_reports": int(((pred_db[:, 0] == 0) & (Y[:, 0] == 1)).sum())},
    "mcnemar_vs_distilbert": {"ensemble_right_distilbert_wrong": b2, "distilbert_right_ensemble_wrong": c2,
                              "p_value": binomtest(b2, b2 + c2, 0.5).pvalue if b2 + c2 else 1.0}}
# averaging two models pulls more probabilities toward the middle, so the escalation band is re-chosen
# on out-of-fold data to keep the human-review load near 10% (Week 5 DistilBERT band escalated 6-8%)
band_rows = []
for bd in [(0.25, 0.75), (0.3, 0.7), (0.35, 0.65), (0.4, 0.6), (0.45, 0.55)]:
    eo = ((blend(Po_c, Pt, w_best) > bd[0]) & (blend(Po_c, Pt, w_best) < bd[1])).any(1)
    et = ((P_ens > bd[0]) & (P_ens < bd[1])).any(1)
    band_rows.append({"band": list(bd), "oof_escalation": float(eo.mean()), "test_escalation": float(et.mean()),
                      "test_subset_acc_kept": float(ok_ens[~et].mean()), "test_subset_acc_escalated": float(ok_ens[et].mean())})
ENS_BAND = (0.4, 0.6)
out["ensemble"]["band_options"] = band_rows
out["ensemble"]["chosen_band"] = list(ENS_BAND)
e_ch = ((P_ens > ENS_BAND[0]) & (P_ens < ENS_BAND[1])).any(1)
ph = pred_ens.copy(); ph[e_ch] = Y[e_ch]
out["ensemble"]["test"].update({"escalation_rate": float(e_ch.mean()),
    "macro_f1_with_human_review_simulated": f1_score(Y, ph, average="macro"), "subset_acc_with_human_review_simulated": float((ph == Y).all(1).mean())})
print("ensemble w", w_best, json.dumps(out["ensemble"]["test"], default=float), out["ensemble"]["mcnemar_vs_distilbert"])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (title, PP) in zip(axes, [("before temperature scaling", P_db), ("after per-label temperature scaling", P_db_c)]):
    p, y = PP.ravel(), Y.ravel(); edges = np.linspace(0, 1, 11); xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum() >= 5: xs.append(p[m].mean()); ys.append(y[m].mean())
    ax.plot([0, 1], [0, 1], "--", color="gray"); ax.plot(xs, ys, "o-", color="#4C72B0")
    ax.set_title(f"DistilBERT reliability, {title}", fontsize=9); ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "week6_calibration.png"), dpi=150)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.boxplot([cv["tfidf"], cv["distilbert"]], tick_labels=["Tuned TF-IDF", "DistilBERT"])
for k, m in enumerate(["tfidf", "distilbert"], 1): ax.scatter([k] * len(cv), cv[m], color="#DD8452", zorder=3)
ax.set_ylabel("macro F1 (5 folds)"); ax.set_title("5-fold cross-validation, 5,846 reviews", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "week6_cv_boxplot.png"), dpi=150)

json.dump(out, open(os.path.join(R, "week6_analysis.json"), "w"), indent=2, default=float)
print("saved")
