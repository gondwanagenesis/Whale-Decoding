"""
Feature extraction for sperm whale codas.

Implements the four independent features from Sharma et al. 2024:
  - Rhythm: relative pattern of inter-click intervals
  - Tempo: absolute speed (mean ICI)
  - Rubato: deviation from metronomic timing within a coda
  - Ornamentation: extra clicks appended to a base pattern
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "vendor" / "sw-combinatoriality" / "data"


def load_codas(path=None):
    path = path or DATA_DIR / "DominicaCodas.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df


def load_dialogues(path=None):
    path = path or DATA_DIR / "sperm-whale-dialogues.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def get_icis(row, max_clicks=29):
    cols = [f"ICI{i}" for i in range(1, max_clicks + 1) if f"ICI{i}" in row.index]
    icis = row[cols].values.astype(float)
    n = int(row["nClicks"]) - 1
    return icis[:n]


def extract_rhythm(icis):
    if len(icis) < 2:
        return icis.copy()
    total = icis.sum()
    if total == 0:
        return icis.copy()
    return icis / total


def extract_tempo(icis):
    if len(icis) == 0:
        return 0.0
    return float(np.mean(icis))


def extract_rubato(icis):
    if len(icis) < 3:
        return 0.0
    mean_ici = np.mean(icis)
    if mean_ici == 0:
        return 0.0
    deviations = np.abs(icis - mean_ici) / mean_ici
    return float(np.mean(deviations))


def extract_ornamentation(icis, threshold_ratio=0.5):
    if len(icis) < 2:
        return 0
    median_ici = np.median(icis)
    if median_ici == 0:
        return 0
    ornament_count = 0
    for i in range(len(icis) - 1, -1, -1):
        if icis[i] < median_ici * threshold_ratio:
            ornament_count += 1
        else:
            break
    return ornament_count


def extract_all_features(df):
    records = []
    for idx, row in df.iterrows():
        icis = get_icis(row)
        if len(icis) == 0:
            continue
        records.append({
            "index": idx,
            "n_clicks": int(row["nClicks"]),
            "duration": float(row["Duration"]),
            "tempo": extract_tempo(icis),
            "rubato": extract_rubato(icis),
            "ornamentation": extract_ornamentation(icis),
            "rhythm_vector": extract_rhythm(icis).tolist(),
            "coda_type": row.get("CodaType", ""),
            "clan": row.get("Clan", ""),
            "unit": row.get("Unit", ""),
            "whale_id": row.get("IDN", 0),
        })
    return pd.DataFrame(records)


def rhythm_distance(r1, r2):
    n = max(len(r1), len(r2))
    a = np.zeros(n)
    b = np.zeros(n)
    a[:len(r1)] = r1
    b[:len(r2)] = r2
    return float(np.sqrt(np.sum((a - b) ** 2)))
