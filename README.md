# Claude or Classifier? Cost-Aware Triage of App User Feedback

Multi-label classification of English app store reviews into `problem_report`, `feature_request`, and `irrelevant`, built end to end: baseline, optimization, fine-tuning, explainability, deployment as a FastAPI service, and rigorous testing. The design question is a practical one: how accurately can a small, free, locally run model triage feedback, and under what conditions is paying for a large language model such as Claude actually worth it?

MS in Artificial Intelligence capstone (MSAI-699), University of the Cumberlands. Author: Westin Pals.

**Demo video:** [`demo/project_demonstration.mp4`](demo/project_demonstration.mp4) shows the deployed service running live: real classifications, uncertainty routing, security checks, and a 100-review batch in 1.65 seconds. Every response in the video is a real capture.

## Headline results

All models are scored on the same fixed 1,170-review held-out test set.

| Model | Macro F1 | Subset accuracy | Seconds per 1,000 reviews | API cost |
|---|---|---|---|---|
| Baseline TF-IDF + logistic regression | 0.821 | 0.761 | 0.03 | $0 |
| Tuned TF-IDF (word + char n-grams) | 0.846 | 0.815 | 0.6 | $0 |
| Fine-tuned DistilBERT (CPU-trained) | 0.867 | 0.878 | 16.5 | $0 |
| **Served ensemble (calibrated DistilBERT + TF-IDF)** | **0.883** | **0.885** | 17.3 | $0 |

Robustness: DistilBERT beat tuned TF-IDF in 5 of 5 cross-validation folds (0.880 vs 0.858 mean macro F1, paired t p = .004), on every unseen app in leave-one-app-out testing, and significantly on the test set (McNemar p < .001). The served ensemble cut missed problem reports from 47 to 29.

The service answers about 90% of reviews itself (fully correct on 92.6% of those) and routes the 9.6% it is least sure about to human review, where Claude can optionally plug in as a fallback. The Claude path is fully implemented and unit tested with a simulated client, but was never run against the paid API: no money was spent on API usage for this project, by choice. Published prices put the cascade at well under one cent per 1,000 reviews versus roughly $0.02 to $0.12 for sending everything to Claude.

## Architecture

```
new reviews --> Triage API (FastAPI, CPU)
                  calibrated DistilBERT + tuned TF-IDF, 50/50 blend
                  |
                  |-- ~90% confident --> problem_report -> issue tracker
                  |                      feature_request -> product backlog
                  |                      irrelevant -> archived
                  |
                  '-- 9.6% uncertain (prob in 0.40-0.60) --> human review
                                                             (Claude optional)
```

Reliability features: per-label temperature calibration (test ECE 0.048 to 0.028), API key auth, request size caps, PII masking before any text leaves the process, privacy-safe logging (hashes only), strict validation of any LLM output, prompt-injection hardening, and an int8-quantized 139 MB variant for edge or on-premises use.

## Data

English portion of the Mini-BAR dataset (Wei et al., 2023): 6,000 reviews, 2,000 each from Garmin Connect, Huawei Health, and Samsung Health. Source: https://github.com/Jl-wei/mini-bar (MIT license, see `data/raw/MINI-BAR_LICENSE.txt`), also archived at https://zenodo.org/records/11066414. Cleaning normalizes Unicode, repairs 2 unlabeled reviews, and removes 154 cross-app duplicates before splitting, leaving 5,846 reviews with a fixed stratified 80/20 split (seed 42) shared by every model.

## Repository layout

- `src/data_prep.py` shared loading, cleaning, de-duplication, and splitting used by every model
- `src/profile_dataset.py` dataset profiling
- `notebooks/01_baseline_tfidf_logreg.ipynb` baseline model
- `notebooks/02_tfidf_optimization_shap.ipynb` TF-IDF feature engineering, hyperparameter and threshold tuning, rating ablation, SHAP, fairness slices
- `src/train_distilbert.py` DistilBERT fine-tuning (learning-rate search, epoch selection, threshold tuning); writes `results/distilbert_*` and `models/distilbert_best/`
- `notebooks/03_distilbert_comparison_shap.ipynb` model comparison, fairness slices, threshold tradeoff, SHAP for DistilBERT
- `src/cv_evaluate.py` 5-fold cross-validation (5,846 reviews) and leave-one-app-out tests
- `src/service/` the FastAPI triage service: `predictor.py` (ensemble + confidence cascade to Claude or human review, PII masking, output validation) and `app.py` (API key auth, input limits, privacy-safe logging)
- `tools/` benchmarks, TF-IDF export, A/B test, error analysis, calibration, ensemble search, end-to-end service check
- `tests/test_service.py` 21 unit and API tests (`python -m pytest -q tests/`)
- `docs/TESTING.md` testing methodology, results, failure modes, and best practices
- `demo/project_demonstration.mp4` the recorded live demo
- `run_all.sh` reproduces every result in order; `results/environment.txt` records versions and data/model hashes
- `Dockerfile`, `requirements-service.txt` container build for the service
- `data/`, `results/` cleaned data with the fixed split, metrics JSON, predictions, figures

## Model weights

`models/distilbert_best/` ships the config, tokenizer, thresholds, calibration, and ensemble settings, but not `model.safetensors` (about 255 MB, over GitHub's file limit). Recreate it exactly with:

```
pip install -r requirements.txt
python src/train_distilbert.py    # about 40 minutes on a 2-core CPU, seed fixed
```

The sha256 of the trained weights is recorded in `results/environment.txt` for verification. `models/tfidf_tuned.joblib` (1.1 MB) is included.

## Run it

```
pip install -r requirements.txt
./run_all.sh                      # reproduce every experiment in order
python -m pytest -q tests/        # 21 tests
```

Serve:

```
export TRIAGE_API_KEY=change-me          # clients send it as the X-API-Key header
uvicorn src.service.app:app --port 8000  # optional: ANTHROPIC_API_KEY enables the Claude fallback, TRIAGE_QUANTIZE=1 uses int8
curl -X POST localhost:8000/v1/classify -H "X-API-Key: change-me" -H "Content-Type: application/json" -d '{"reviews":[{"text":"App crashes after the update"}]}'
```

## Responsible AI

Design follows the NIST AI Risk Management Framework and the OWASP Top 10 for LLM applications. The problem-report threshold favors recall because a missed bug costs more than a dismissible false alarm; error rates are tracked per app and review length; reviews are processed locally with PII masked and only hashes logged; prompt injection text is treated as data (and tested); SHAP explains any flagged decision; every response carries a model version. A list of 218 suspected label errors found during error analysis is included for relabeling (`results/week6_suspected_label_noise.csv`).

## AI use disclosure

Generative AI (Claude, Anthropic) assisted with the code and documentation. All reported numbers come from actually running the code in this repository.
