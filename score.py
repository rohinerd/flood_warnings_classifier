import glob
import os
import sys
import urllib.request
import zipfile

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA_URL = "https://github.com/rohinerd/flood_warnings_classifier/releases/download/data-v1/Historic_Flood_Warnings.zip"
DATA_DIR = "data"
CUTOFF = 2019
THRESHOLD = float(os.environ.get("AUC_THRESHOLD", "0.30"))


def fetch():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "flood.zip")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    return glob.glob(os.path.join(DATA_DIR, "*EA Only.ods"))[0]


def load(ods_path):
    df = pd.read_excel(ods_path, engine="odf", sheet_name="tblAllData")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["TYPE_CLEAN"] = df["TYPE"].str.strip().str.lower()
    df = df[~df["TYPE_CLEAN"].str.startswith("update")]
    df = df[df["TYPE_CLEAN"].isin(
        ["flood alert", "flood warning", "severe flood warning"]
    )].copy()
    df["SEVERE"] = (df["TYPE_CLEAN"] == "severe flood warning").astype(int)
    df["YEAR"] = df["DATE"].dt.year
    return df


def add_weekly_count(df):
    df = df.sort_values("DATE").copy()
    df["WEEK"] = df["DATE"].dt.to_period("W").dt.start_time
    weekly = df.groupby("WEEK").size()
    weekly = weekly.reindex(
        pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-MON"),
        fill_value=0,
    )
    df["PREV_WEEK_COUNT"] = df["WEEK"].map(weekly.shift(1).fillna(0)).fillna(0)
    return df


def main():
    df = add_weekly_count(load(fetch()))
    train = df[df["YEAR"] <= CUTOFF]
    test = df[df["YEAR"] > CUTOFF]

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(train[["PREV_WEEK_COUNT"]], train["SEVERE"])
    probs = model.predict_proba(test[["PREV_WEEK_COUNT"]])[:, 1]
    auc = roc_auc_score(test["SEVERE"], probs)

    print(f"AUC, test years after {CUTOFF}: {auc:.3f} (threshold {THRESHOLD})")
    if auc < THRESHOLD:
        print("Regression gate failed")
        sys.exit(1)
    print("Regression gate passed")


if __name__ == "__main__":
    main()
