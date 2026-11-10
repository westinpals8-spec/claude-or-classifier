import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []
C.append(md("""# Week 4 (Part B): Fine-Tuned DistilBERT, Model Comparison, and SHAP
**Project:** Claude or Classifier? Cost-Aware Triage of App User Feedback, Westin Pals, MSAI-699-B01

DistilBERT (`distilbert-base-uncased`) is fine-tuned by `src/train_distilbert.py` (CPU only), which logs every run to `results/distilbert_runs.json` and saves test-set probabilities. Because training takes about an hour on a CPU, this notebook **loads those saved results** instead of retraining, then:
1. Shows the hyperparameter search (learning rate, epochs) on the validation set.
2. Compares the Week 3 baseline, the tuned TF-IDF model, and DistilBERT on the same 1,170-review test set.
3. Checks fairness and risk slices.
4. Explains DistilBERT predictions with SHAP (token-level attributions).

To retrain from scratch: `python src/train_distilbert.py` from the project root."""))

C.append(code("""import json, os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score, hamming_loss
ROOT = os.path.abspath('..'); sys.path.insert(0, os.path.join(ROOT, 'src'))
from data_prep import LABELS, APPS
R = os.path.join(ROOT, 'results')
runs = json.load(open(os.path.join(R, 'distilbert_runs.json')))
dbm = json.load(open(os.path.join(R, 'distilbert_metrics.json')))
tfm = json.load(open(os.path.join(R, 'tfidf_tuned_metrics.json')))"""))

C.append(md("## 1. DistilBERT hyperparameter search (validation set, 468 reviews)"))
C.append(code("""hist = pd.DataFrame([{'run': r['name'], 'lr': r['lr'], **h} for r in runs for h in r['history']])
display(hist[['run', 'epoch', 'train_loss', 'val_macro_f1', 'val_feature_request_f1', 'epoch_seconds']].round(3))
print('selected:', dbm['selected_run']['name'], '| best epoch', dbm['selected_run']['best_epoch'],
      '| val macro F1', round(dbm['selected_run']['best_val_macro_f1'], 3), '| thresholds', dbm['thresholds'])
fig, ax = plt.subplots(figsize=(6, 3.5))
for name, g in hist.groupby('run'): ax.plot(g.epoch, g.val_macro_f1, marker='o', label=name)
ax.set_xlabel('epoch'); ax.set_ylabel('validation macro F1'); ax.set_xticks(sorted(hist.epoch.unique())); ax.legend(); ax.set_title('DistilBERT learning-rate search')
plt.tight_layout(); plt.savefig(os.path.join(R, 'figures', 'distilbert_lr_search.png'), dpi=150); plt.show()"""))

C.append(md("## 2. Head-to-head on the held-out test set (n = 1,170)"))
C.append(code("""def row(name, m, cost, sec):
    return {'model': name, 'macro_f1': m['macro_f1'], 'micro_f1': m['micro_f1'], 'subset_acc': m['subset_acc'],
            'problem_report_f1': m['problem_report_f1'], 'feature_request_f1': m['feature_request_f1'],
            'feature_request_p': m['feature_request_p'], 'feature_request_r': m['feature_request_r'],
            'irrelevant_f1': m['irrelevant_f1'], 'sec_per_1000_cpu': sec, 'api_usd_per_1000': cost}
base = json.load(open(os.path.join(R, 'baseline_metrics.json')))
bt = base['test']
base_m = {'macro_f1': bt['macro_f1'], 'micro_f1': bt['micro_f1'], 'subset_acc': bt['subset_accuracy'],
          **{f'{l}_f1': bt[l]['f1'] for l in LABELS}, **{f'{l}_p': bt[l]['precision'] for l in LABELS}, **{f'{l}_r': bt[l]['recall'] for l in LABELS}}
cmp = pd.DataFrame([row('Week 3 baseline TF-IDF', base_m, 0, base['inference_seconds_per_1000']),
                    row('Tuned TF-IDF (word+char, C=4)', tfm['test_tuned'], 0, tfm['inference_seconds_per_1000']),
                    row('DistilBERT (thr 0.5)', dbm['test_thr05'], 0, dbm['inference_seconds_per_1000_cpu']),
                    row('DistilBERT (tuned thr)', dbm['test_tuned_thr'], 0, dbm['inference_seconds_per_1000_cpu'])]).set_index('model')
cmp.round(3).T"""))

C.append(code("""ax = cmp[['problem_report_f1', 'feature_request_f1', 'irrelevant_f1', 'macro_f1']].drop('DistilBERT (thr 0.5)').T.plot.bar(
    figsize=(8, 4), rot=0, ylim=(0.6, 1.0), color=['#9E9E9E', '#4C72B0', '#DD8452'])
ax.set_ylabel('F1 (test set)'); ax.set_title('Model comparison on the held-out test set'); ax.legend(loc='upper left', fontsize=8, frameon=False)
for c in ax.containers: ax.bar_label(c, fmt='%.2f', fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(R, 'figures', 'model_comparison_week4.png'), dpi=150); plt.show()"""))

C.append(md("## 3. Fairness and risk slices\nThe costliest error is a **missed problem report** (a real user issue silently ignored), so its miss rate is shown per slice for both optimized models."))
C.append(code("""db = pd.read_csv(os.path.join(R, 'distilbert_test_predictions.csv'))
tf = pd.read_csv(os.path.join(R, 'tfidf_tuned_test_predictions.csv'))
assert (db.review_id.values == tf.review_id.values).all()
words = db.text.str.split().str.len()
nonascii = db.text.map(lambda s: sum(ord(c) > 127 for c in s) / max(len(s), 1) > 0.2)
slices = {**{f'app: {a}': db.app == a for a in APPS}, 'length <= 3 words': words <= 3,
          'length 4-20 words': (words > 3) & (words <= 20), 'length > 20 words': words > 20, 'emoji / non-English heavy': nonascii}
rows = []
for name, m in slices.items():
    m = m.values; Y = db.loc[m, LABELS].values; pos = Y[:, 0] == 1
    r = {'slice': name, 'n': int(m.sum()), 'n_problem_reports': int(pos.sum())}
    for tag, d in [('tfidf', tf), ('distilbert', db)]:
        P = d.loc[m, [f'pred_{l}' for l in LABELS]].values
        r[f'{tag}_macro_f1'] = f1_score(Y, P, average='macro', zero_division=0)
        r[f'{tag}_PR_miss_rate'] = float(((P[:, 0] == 0) & pos).sum() / max(pos.sum(), 1))
    rows.append(r)
fair = pd.DataFrame(rows).round(3); fair"""))

C.append(md("### Threshold choice is an ethical lever\nThe validation-tuned DistilBERT thresholds (problem report 0.80) maximize F1 but trade away problem-report recall. Lowering the problem-report threshold catches more real issues at the cost of more false alarms, which a human reviewer can dismiss cheaply."))
C.append(code("""Pdb = db[[f'proba_{l}' for l in LABELS]].values; Ydb = db[LABELS].values
rows = []
for t in [0.8, 0.5, 0.3]:
    pred = (Pdb[:, 0] >= t).astype(int); pos = Ydb[:, 0] == 1
    p, r, f1, _ = precision_recall_fscore_support(Ydb[:, 0], pred, average='binary', zero_division=0)
    rows.append({'problem_report_threshold': t, 'precision': p, 'recall': r, 'f1': f1,
                 'missed_problem_reports': int(((pred == 0) & pos).sum()), 'false_alarms': int(((pred == 1) & ~pos).sum())})
pd.DataFrame(rows).round(3)"""))
C.append(md("## 4. SHAP explanations for DistilBERT\nSHAP's text explainer masks tokens and measures how each one moves the model's probability for a label. Shown for a sample of test reviews, with the summed token attributions plotted for the feature-request output."))
C.append(code("""import torch, shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MD = os.path.join(ROOT, 'models', 'distilbert_best')
tok = AutoTokenizer.from_pretrained(MD); model = AutoModelForSequenceClassification.from_pretrained(MD).eval()
def f(texts):
    enc = tok(list(texts), truncation=True, max_length=128, padding=True, return_tensors='pt')
    with torch.no_grad(): return torch.sigmoid(model(**enc).logits).numpy()
explainer = shap.Explainer(f, shap.maskers.Text(tok), output_names=LABELS)
fr = LABELS.index('feature_request'); thr = dbm['thresholds']
pred_fr, y_fr = db['pred_feature_request'].values, db['feature_request'].values
pick = {'correct feature request': np.where((y_fr == 1) & (pred_fr == 1) & (words.between(8, 30).values))[0][0],
        'correct problem report': np.where((db.problem_report.values == 1) & (db.pred_problem_report.values == 1) & (words.between(8, 30).values))[0][0],
        'false-positive feature request': np.where((y_fr == 0) & (pred_fr == 1))[0][0],
        'missed feature request': np.where((y_fr == 1) & (pred_fr == 0))[0][0]}
sv = explainer([db.text[i] for i in pick.values()], max_evals=600, batch_size=32, silent=True)
for k, (name, i) in enumerate(pick.items()):
    lab = 'problem_report' if 'problem report' in name else 'feature_request'; j = LABELS.index(lab)
    toks, vals = np.array(sv[k].data), sv[k].values[:, j]
    top = np.argsort(np.abs(vals))[-6:][::-1]
    print(f'[{name}] p({lab})={f([db.text[i]])[0, j]:.2f}  "{db.text[i][:140]}"')
    print('   top tokens:', ', '.join(f'{toks[t].strip()!r} {vals[t]:+.2f}' for t in top), '\\n')"""))

C.append(code("""# Global view: average SHAP value per token across 60 test reviews, feature_request output
rng = np.random.default_rng(42); idx = rng.choice(len(db), 60, replace=False)
sv60 = explainer([db.text[i] for i in idx], max_evals=400, batch_size=32, silent=True)
agg = {}
for k in range(len(idx)):
    for t, v in zip(sv60[k].data, sv60[k].values[:, fr]):
        t = t.strip().lower()
        if t and t.isalpha(): agg.setdefault(t, []).append(v)
g = pd.Series({t: np.mean(v) for t, v in agg.items() if len(v) >= 3}).sort_values()
top = pd.concat([g.head(8), g.tail(10)])
ax = top.plot.barh(figsize=(6, 5), color=['#C62828' if v < 0 else '#2E7D32' for v in top])
ax.set_title('DistilBERT: mean SHAP toward feature_request (tokens seen 3+ times)', fontsize=9); ax.set_xlabel('mean SHAP value')
plt.tight_layout(); plt.savefig(os.path.join(R, 'figures', 'shap_distilbert_feature_request.png'), dpi=150); plt.show()
fair.to_csv(os.path.join(R, 'week4_fairness_slices.csv'), index=False); cmp.round(4).to_csv(os.path.join(R, 'week4_model_comparison.csv'))"""))

nb['cells'] = C
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, 'notebooks/03_distilbert_comparison_shap.ipynb'); print('written')
