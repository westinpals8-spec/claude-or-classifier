"""Shared data loading, cleaning, and splitting for the capstone.

Every model in the project (TF-IDF baseline, DistilBERT, Claude, cascade) uses this
module so they are trained and tested on exactly the same reviews.

Cleaning steps follow the Week 2 preprocessing plan:
  1. Trim whitespace and normalize Unicode (NFKC).
  2. Fix the two reviews that shipped with no label (both describe a failure, so they
     are labeled problem_report after manual review).
  3. Drop exact duplicate texts BEFORE splitting so no text appears in both train and test.
  4. Keep text only as the model input (star rating is excluded to avoid a shortcut).
"""
from __future__ import annotations

import os
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["problem_report", "feature_request", "irrelevant"]
APPS = ["Garmin Connect", "Huawei Health", "Samsung Health"]
RANDOM_STATE = 42

# The two reviews released with all three labels set to 0 (found in Week 2 profiling).
UNLABELED_FIXES = {
    "Nothing working, throwing out all my Huawei device in garbage. Bad support. "
    "God speed to closed down permanently.": "problem_report",
    "Upgrades my mobile to the pixel 6pro and tried to connect my huawei watch 3pro "
    "no joy what so ever": "problem_report",
}


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_raw(raw_dir: str | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or os.path.join(project_root(), "data", "raw")
    frames = [pd.read_csv(os.path.join(raw_dir, f"{app}.en.csv")) for app in APPS]
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"rows_raw": len(df)}
    df = df.copy()
    df["text"] = df["data"].astype(str).map(lambda s: unicodedata.normalize("NFKC", s).strip())

    fixed = 0
    for text, label in UNLABELED_FIXES.items():
        mask = (df["text"] == text) & (df[LABELS].sum(axis=1) == 0)
        df.loc[mask, label] = 1
        fixed += int(mask.sum())
    log["unlabeled_fixed"] = fixed
    log["still_unlabeled"] = int((df[LABELS].sum(axis=1) == 0).sum())

    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    log["duplicates_removed"] = before - len(df)
    log["rows_clean"] = len(df)

    df["label_combo"] = df[LABELS].astype(str).agg("".join, axis=1)
    df = df[["text", "app", "score"] + LABELS + ["label_combo"]]
    df.insert(0, "review_id", range(len(df)))
    return df, log


def make_split(df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
    """Fixed, stratified train/test split reused by every model in the project.

    Stratifies on label combination plus app so each app and label mix is represented.
    """
    strat = df["label_combo"] + "|" + df["app"]
    # merge very rare strata so stratification is possible
    counts = strat.value_counts()
    strat = strat.where(strat.map(counts) >= 5, df["label_combo"])
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=RANDOM_STATE, stratify=strat
    )
    out = df.copy()
    out["split"] = "train"
    out.loc[test_idx, "split"] = "test"
    return out


def load_prepared(save: bool = True) -> tuple[pd.DataFrame, dict]:
    df, log = clean(load_raw())
    df = make_split(df)
    if save:
        out_dir = os.path.join(project_root(), "data", "processed")
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, "reviews_clean_split.csv"), index=False)
    return df, log
