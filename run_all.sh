#!/usr/bin/env bash
# Reproduce every result in the project, in order (CPU only). Seeds are fixed (42) everywhere.
# Approximate runtimes on a 2-core CPU are noted. Outputs land in results/ and models/.
set -euo pipefail
cd "$(dirname "$0")"
# Week 2 profiling ran on the original mini-bar repo: python src/profile_dataset.py <path-to-mini-bar>
jupyter nbconvert --to notebook --execute --inplace notebooks/01_baseline_tfidf_logreg.ipynb   # Week 3, ~1 min
jupyter nbconvert --to notebook --execute --inplace notebooks/02_tfidf_optimization_shap.ipynb # Week 4, ~10 min
python src/train_distilbert.py                                             # Week 4, ~40 min
jupyter nbconvert --to notebook --execute --inplace notebooks/03_distilbert_comparison_shap.ipynb # Week 4, ~5 min
python tools/week5_benchmark.py                                            # Week 5, ~5 min
python src/cv_evaluate.py                                                  # Week 6, ~2 h (8 DistilBERT trainings)
python tools/export_tfidf.py                                               # Week 6, ~1 min
python tools/week6_analysis.py                                             # Week 6, ~1 min (writes calibration.json, ensemble.json)
python tools/week6_service_check.py                                        # Week 6, ~1 min
python -m pytest -q tests/                                                 # Week 6, ~10 s
