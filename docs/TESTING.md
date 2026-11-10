# Testing and Reliability Guide (Week 6)

## How the model is tested
1. **5-fold cross-validation** over all 5,846 cleaned reviews (`src/cv_evaluate.py`), stratified by label combination and app, seed 42. Every review is predicted once by a model that never saw it. DistilBERT keeps the Week 4 settings (lr 5e-5, 3 epochs, batch 16, max length 128); each fold holds out 10% of its training data to pick the epoch.
2. **Leave-one-app-out**: train on two apps, test on the third (generalization to an unseen app).
3. **A/B test on the fixed 1,170-review test set** (`tools/week6_analysis.py`): exact McNemar test on whether each review is fully correct, plus a 2,000-sample bootstrap 95% CI for the macro F1 difference.
4. **End-to-end service check** (`tools/week6_service_check.py`): runs the real service on the test set and confirms it is identical to an offline recomputation.
5. **Unit and API tests** (`tests/test_service.py`, 21 tests): auth, input limits, PII masking, prompt-injection text, determinism, human-review routing, strict validation of LLM output (fake client, no paid API calls).

## Key results
| Test | Tuned TF-IDF | DistilBERT |
|---|---|---|
| 5-fold CV macro F1, mean (SD) | 0.858 (0.005) | 0.880 (0.011), paired t p = .004 |
| Unseen app: Garmin / Huawei / Samsung | 0.813 / 0.806 / 0.837 | 0.858 / 0.841 / 0.865 |
| Test set macro F1 | 0.846 | 0.867 (McNemar p < .001) |

Served model (calibrated DistilBERT + TF-IDF, 50/50): test macro F1 0.883, 29 missed problem reports (DistilBERT alone: 47), 9.6% of reviews routed to human review with band (0.40, 0.60).

## Common failure modes (out-of-fold errors, DistilBERT)
- Problem report vs. feature request confusion: 35.5% of errors.
- Reviews labeled both problem and request: 81.6% mislabeled (model rarely predicts both).
- Long reviews (> 50 words): 22.9% error rate, about double the average.
- Huawei Health: 15.1% error rate vs. 9.5% Garmin Connect.
- 218 confident disagreements with the human label; a manual check of 25 found 7 annotation errors, 7 model errors, 11 ambiguous (`results/week6_manual_review_25.csv`). Full list for relabeling: `results/week6_suspected_label_noise.csv`.

## Reliability fixes now in the service
- Per-label temperature scaling (`models/distilbert_best/calibration.json`): test ECE 0.048 -> 0.028, F1 unchanged.
- Calibrated DistilBERT + TF-IDF ensemble (`models/distilbert_best/ensemble.json`, `models/tfidf_tuned.joblib`), weight chosen on cross-validation predictions.
- Escalation band re-chosen on cross-validation data for the ensemble: (0.40, 0.60).
- Every review is guaranteed at least one label.

## Best practices
1. One fixed split and one shared data module (`src/data_prep.py`) for every model.
2. Make every tuning choice on validation or cross-validation data, never on the test set.
3. Report variability and significance (fold SD, paired tests, bootstrap CIs), not a single score.
4. Test on unseen apps and per-slice error rates, not only overall metrics.
5. Calibrate probabilities before using them for thresholds or routing.
6. Keep a human fallback; log routes and hashes, never raw review text.
7. Maintain a suspected-label-error list and fix data problems in the data.

Rerun all checks whenever the model, data, or thresholds change: `./run_all.sh` (or the Week 6 steps at its end).

## Not tested
The Claude fallback was not run against the paid API: by the project owner's choice, no money was spent on API usage. It is implemented and covered by tests with a simulated client; uncertain reviews currently go to human review.
