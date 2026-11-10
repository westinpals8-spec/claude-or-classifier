"""Week 2 dataset profile for the MSAI-699 capstone.

Usage: python -I profile_dataset.py <path-to-mini-bar-repo>
Reads the six English/French CSVs shipped in Jl-wei/mini-bar (dataset/for_classification)
and prints the statistics reported in the Week 2 Literature Review & Data Report.
"""
import glob
import os
import sys

import pandas as pd

LABELS = ["problem_report", "feature_request", "irrelevant"]


def main(repo):
    files = sorted(glob.glob(os.path.join(repo, "dataset/for_classification/*.en.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    text = df["data"].astype(str)
    words = text.str.split().str.len()

    print("shape", df.shape)
    print("columns", list(df.columns))
    print("missing values", df.isna().sum().to_dict())
    print("\nlabel counts by app\n", df.groupby("app")[LABELS].sum())
    print("label totals", df[LABELS].sum().to_dict())
    print("labels per review", df[LABELS].sum(axis=1).value_counts().to_dict())
    print("\nwords per review", words.describe().round(1).to_dict())
    print("reviews with <=3 words", int((words <= 3).sum()),
          "label mix", df[words <= 3][LABELS].sum().to_dict())
    dup_mask = text.duplicated(keep=False)
    groups = df[dup_mask].groupby("data")["app"].nunique()
    print("\nduplicate rows (beyond first)", int(text.duplicated().sum()),
          "duplicate groups", len(groups), "groups spanning >1 app", int((groups > 1).sum()))
    print("star rating distribution", df["score"].value_counts().sort_index().to_dict())
    print("mean rating by label", {l: round(df[df[l] == 1]["score"].mean(), 2) for l in LABELS})
    print("possible emails", int(text.str.contains(r"[\w.+-]+@[\w-]+\.\w+").sum()),
          "urls", int(text.str.contains(r"http|www\.").sum()),
          "phone-like", int(text.str.contains(r"\+?\d[\d \-]{8,}\d").sum()))
    heavy = text.apply(lambda s: sum(ord(c) > 127 for c in s) / max(len(s), 1)) > 0.2
    print("emoji/non-ASCII-heavy reviews", int(heavy.sum()))


if __name__ == "__main__":
    main(sys.argv[1])
