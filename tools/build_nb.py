import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md("""# Week 3: Baseline Model, TF-IDF + Logistic Regression
**Project:** Claude or Classifier? Cost-Aware Triage of App User Feedback
**Author:** Westin Pals, MSAI-699-B01 Capstone

**Task.** Multi-label classification of English app reviews into three labels: `problem_report`, `feature_request`, `irrelevant` (Mini-BAR dataset, Wei et al., 2023).

**Baseline choice.** A TF-IDF bag of word unigrams and bigrams feeding one logistic regression classifier per label (one-vs-rest). It is the standard, transparent starting point for text classification: fast on a CPU, essentially free to run, and its weights can be read directly. Every later model (fine-tuned DistilBERT, Claude via the Anthropic API, and the cascade) has to beat this number to justify its extra cost.

**Evaluation protocol** (reused for all later models):
1. Fixed, stratified 80/20 train/test split built in `src/data_prep.py` after cleaning and de-duplication.
2. 5-fold cross-validation on the training set for a stable estimate.
3. Held-out test set metrics: per-label precision, recall, F1, plus macro F1 (primary metric because the labels are imbalanced).
4. Cross-app test: train on two apps, test on the third.
5. Runtime and cost per 1,000 reviews."""))

cells.append(code("""import json, os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             hamming_loss, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

ROOT = os.path.abspath('..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from data_prep import LABELS, APPS, RANDOM_STATE, load_prepared

RESULTS = os.path.join(ROOT, 'results')
os.makedirs(os.path.join(RESULTS, 'figures'), exist_ok=True)
np.random.seed(RANDOM_STATE)"""))

cells.append(md("## 1. Load, clean, and split"))
cells.append(code("""df, clean_log = load_prepared(save=True)
print(clean_log)
train, test = df[df.split == 'train'], df[df.split == 'test']
print('train', len(train), '| test', len(test))
df.groupby('split')[LABELS].mean().round(3)"""))

cells.append(md("## 2. Model definition"))
cells.append(code("""def build_baseline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                  lowercase=True, strip_accents='unicode')),
        ('clf', OneVsRestClassifier(LogisticRegression(
            C=1.0, class_weight='balanced', solver='liblinear', max_iter=2000,
            random_state=RANDOM_STATE))),
    ])

def evaluate(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    out = {lab: {'precision': p[i], 'recall': r[i], 'f1': f[i], 'support': int(s[i])}
           for i, lab in enumerate(LABELS)}
    out['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    out['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    out['subset_accuracy'] = accuracy_score(y_true, y_pred)
    out['hamming_loss'] = hamming_loss(y_true, y_pred)
    return out

X_train, Y_train = train['text'].values, train[LABELS].values
X_test, Y_test = test['text'].values, test[LABELS].values"""))

cells.append(md("## 3. Reference points (no learning)\nA model that always predicts the most common value for each label, and one that guesses at random with the training label rates. The baseline has to clear both."))
cells.append(code("""ref = {}
for name, strategy in [('majority', 'most_frequent'), ('random_stratified', 'stratified')]:
    dummy = OneVsRestClassifier(DummyClassifier(strategy=strategy, random_state=RANDOM_STATE))
    dummy.fit(X_train.reshape(-1, 1), Y_train)
    ref[name] = evaluate(Y_test, dummy.predict(X_test.reshape(-1, 1)))
    print(f"{name:18s} macro F1 = {ref[name]['macro_f1']:.3f}")"""))

cells.append(md("## 4. Five-fold cross-validation on the training set"))
cells.append(code("""skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []
for k, (tr, va) in enumerate(skf.split(X_train, train['label_combo'])):
    m = build_baseline().fit(X_train[tr], Y_train[tr])
    cv_scores.append(f1_score(Y_train[va], m.predict(X_train[va]), average='macro'))
    print(f'fold {k+1}: macro F1 = {cv_scores[-1]:.3f}')
print(f'CV macro F1 = {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}')"""))

cells.append(md("## 5. Train on the full training set and test on the held-out set"))
cells.append(code("""model = build_baseline()
t0 = time.perf_counter(); model.fit(X_train, Y_train); train_sec = time.perf_counter() - t0
t0 = time.perf_counter(); Y_pred = model.predict(X_test); infer_sec = time.perf_counter() - t0
test_metrics = evaluate(Y_test, Y_pred)
print(classification_report(Y_test, Y_pred, target_names=LABELS, digits=3, zero_division=0))
print(f"macro F1 {test_metrics['macro_f1']:.3f} | micro F1 {test_metrics['micro_f1']:.3f} | "
      f"subset accuracy {test_metrics['subset_accuracy']:.3f} | hamming loss {test_metrics['hamming_loss']:.3f}")
print(f'train time {train_sec:.2f}s | inference {1000*infer_sec/len(X_test):.3f}s per 1,000 reviews | API cost $0')"""))

cells.append(md("### Results by app and by review length"))
cells.append(code("""rows = []
for app in APPS:
    m = (test['app'] == app).values
    rows.append({'slice': app, 'n': int(m.sum()), 'macro_f1': f1_score(Y_test[m], Y_pred[m], average='macro', zero_division=0)})
short = (test['text'].str.split().str.len() <= 3).values
for name, m in [('<= 3 words', short), ('> 3 words', ~short)]:
    rows.append({'slice': name, 'n': int(m.sum()), 'macro_f1': f1_score(Y_test[m], Y_pred[m], average='macro', zero_division=0)})
slices = pd.DataFrame(rows).round(3); slices"""))

cells.append(md("## 6. Cross-app generalization: train on two apps, test on the third"))
cells.append(code("""cross = []
for held in APPS:
    tr, te = df[df.app != held], df[df.app == held]
    m = build_baseline().fit(tr['text'].values, tr[LABELS].values)
    pred = m.predict(te['text'].values)
    res = evaluate(te[LABELS].values, pred)
    cross.append({'held_out_app': held, 'n_test': len(te), 'macro_f1': res['macro_f1'],
                  **{f'{l}_f1': res[l]['f1'] for l in LABELS}})
cross = pd.DataFrame(cross).round(3); cross"""))

cells.append(md("## 7. What the model learned: top-weighted n-grams per label"))
cells.append(code("""vocab = np.array(model.named_steps['tfidf'].get_feature_names_out())
top = {}
for i, lab in enumerate(LABELS):
    coefs = model.named_steps['clf'].estimators_[i].coef_.ravel()
    top[lab] = list(vocab[np.argsort(coefs)[-12:][::-1]])
pd.DataFrame(top)"""))

cells.append(md("## 8. Error analysis: missed feature requests"))
cells.append(code("""fr = LABELS.index('feature_request')
missed = test[(Y_test[:, fr] == 1) & (Y_pred[:, fr] == 0)]
print(f'missed feature requests: {len(missed)} of {int(Y_test[:, fr].sum())}')
pd.set_option('display.max_colwidth', 160)
missed[['app', 'text']].head(8)"""))

cells.append(md("## 9. Save results for later comparison"))
cells.append(code("""per = pd.DataFrame({l: test_metrics[l] for l in LABELS}).T[['precision', 'recall', 'f1']]
ax = per.plot.bar(figsize=(7, 4), rot=0, ylim=(0, 1.15), color=['#4C72B0', '#DD8452', '#55A868'])
ax.set_title('Baseline (TF-IDF + logistic regression): held-out test set')
ax.set_ylabel('score'); ax.legend(loc='upper center', ncol=3, frameon=False)
for c in ax.containers: ax.bar_label(c, fmt='%.2f', fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS, 'figures', 'baseline_test_metrics.png'), dpi=150); plt.show()

summary = {
    'model': 'tfidf_word_1-2gram + one_vs_rest_logreg (class_weight=balanced, C=1.0)',
    'cleaning': clean_log,
    'n_train': len(train), 'n_test': len(test),
    'reference': {k: v['macro_f1'] for k, v in ref.items()},
    'cv_macro_f1_mean': float(np.mean(cv_scores)), 'cv_macro_f1_std': float(np.std(cv_scores)),
    'test': test_metrics,
    'cross_app': cross.to_dict(orient='records'),
    'slices': slices.to_dict(orient='records'),
    'train_seconds': train_sec, 'inference_seconds_per_1000': 1000 * infer_sec / len(X_test),
    'api_cost_usd_per_1000': 0.0,
}
with open(os.path.join(RESULTS, 'baseline_metrics.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=float)
pred_df = test[['review_id', 'app', 'text'] + LABELS].copy()
for i, l in enumerate(LABELS): pred_df[f'pred_{l}'] = Y_pred[:, i]
pred_df.to_csv(os.path.join(RESULTS, 'baseline_test_predictions.csv'), index=False)
print('saved')"""))

nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, 'notebooks/01_baseline_tfidf_logreg.ipynb')
print('notebook written')
