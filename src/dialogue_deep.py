"""
Deep dialogue analysis: call-response patterns, sequential structure,
exchange typology, and imitation dynamics.
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import product
from .coda_features import get_icis, extract_tempo, extract_rubato, extract_rhythm


def load_and_enrich_dialogues(dialogues_df):
    """Parse dialogues and compute per-coda features inline."""
    records = []
    for _, row in dialogues_df.iterrows():
        icis = get_icis(row)
        records.append({
            "rec": row["REC"],
            "whale": int(row["Whale"]),
            "timestamp": float(row["TsTo"]),
            "n_clicks": int(row["nClicks"]),
            "duration": float(row["Duration"]),
            "tempo": extract_tempo(icis),
            "rubato": extract_rubato(icis),
            "rhythm": extract_rhythm(icis).tolist(),
        })
    df = pd.DataFrame(records)
    df = df.sort_values(["rec", "timestamp"]).reset_index(drop=True)

    # classify each coda by click count and tempo band
    df["tempo_band"] = pd.cut(
        df["tempo"],
        bins=[0, 0.06, 0.1, 0.15, 0.25, 0.4, 1.0],
        labels=["vfast", "fast", "medium", "slow", "vslow", "crawl"],
    )
    df["coda_class"] = df["n_clicks"].astype(str) + "-" + df["tempo_band"].astype(str)
    return df


def call_response_matrix(df):
    """Build a transition matrix: P(response type | prompt type) across whale switches."""
    pairs = []
    for rec, group in df.groupby("rec"):
        rows = group.values
        cols = group.columns.tolist()
        whale_idx = cols.index("whale")
        class_idx = cols.index("coda_class")
        for i in range(1, len(rows)):
            if rows[i][whale_idx] != rows[i - 1][whale_idx]:
                pairs.append((rows[i - 1][class_idx], rows[i][class_idx]))

    pair_counts = Counter(pairs)
    classes = sorted(set(c for pair in pairs for c in pair))

    matrix = pd.DataFrame(0, index=classes, columns=classes, dtype=float)
    for (prompt, response), count in pair_counts.items():
        matrix.loc[prompt, response] = count

    # normalize rows to get conditional probabilities
    row_sums = matrix.sum(axis=1)
    prob_matrix = matrix.div(row_sums, axis=0).fillna(0)
    return matrix, prob_matrix, pair_counts


def coda_bigrams(df):
    """Sequential bigrams within each whale's stream (not cross-whale)."""
    bigrams = Counter()
    for (rec, whale), group in df.groupby(["rec", "whale"]):
        classes = group["coda_class"].values
        for i in range(1, len(classes)):
            bigrams[(classes[i - 1], classes[i])] += 1
    return bigrams


def cross_whale_bigrams(df):
    """Bigrams that cross whale boundaries (true call-response pairs)."""
    bigrams = Counter()
    for rec, group in df.groupby("rec"):
        classes = group["coda_class"].values
        whales = group["whale"].values
        for i in range(1, len(classes)):
            if whales[i] != whales[i - 1]:
                bigrams[(classes[i - 1], classes[i])] += 1
    return bigrams


def echo_vs_contrast(df):
    """For each whale switch: is the response an echo (same class) or contrast?"""
    results = []
    for rec, group in df.groupby("rec"):
        rows = list(group.itertuples())
        for i in range(1, len(rows)):
            if rows[i].whale != rows[i - 1].whale:
                same_clicks = rows[i].n_clicks == rows[i - 1].n_clicks
                same_band = rows[i].tempo_band == rows[i - 1].tempo_band
                same_class = rows[i].coda_class == rows[i - 1].coda_class

                t1, t2 = rows[i - 1].tempo, rows[i].tempo
                tempo_ratio = t2 / t1 if t1 > 0 else 0
                gap = rows[i].timestamp - (rows[i - 1].timestamp + rows[i - 1].duration)

                results.append({
                    "rec": rec,
                    "same_clicks": same_clicks,
                    "same_tempo_band": same_band,
                    "same_class": same_class,
                    "tempo_ratio": tempo_ratio,
                    "gap": gap,
                    "prompt_clicks": rows[i - 1].n_clicks,
                    "response_clicks": rows[i].n_clicks,
                    "prompt_class": rows[i - 1].coda_class,
                    "response_class": rows[i].coda_class,
                })
    return pd.DataFrame(results)


def exchange_phases(df):
    """Split each exchange into opening, middle, and closing phases."""
    results = []
    for rec, group in df.groupby("rec"):
        n = len(group)
        if n < 6:
            continue
        third = n // 3
        phases = ["opening"] * third + ["middle"] * (n - 2 * third) + ["closing"] * third
        group = group.copy()
        group["phase"] = phases

        for phase in ["opening", "middle", "closing"]:
            sub = group[group["phase"] == phase]
            results.append({
                "rec": rec,
                "phase": phase,
                "n_codas": len(sub),
                "mean_tempo": sub["tempo"].mean(),
                "mean_rubato": sub["rubato"].mean(),
                "mean_clicks": sub["n_clicks"].mean(),
                "tempo_std": sub["tempo"].std(),
                "dominant_class": sub["coda_class"].mode().iloc[0] if len(sub) > 0 else "",
            })
    return pd.DataFrame(results)


def turn_run_lengths(df):
    """How many consecutive codas does each whale produce before yielding?"""
    runs = []
    for rec, group in df.groupby("rec"):
        whales = group["whale"].values
        current_whale = whales[0]
        run_len = 1
        for i in range(1, len(whales)):
            if whales[i] == current_whale:
                run_len += 1
            else:
                runs.append({"rec": rec, "whale": current_whale, "run_length": run_len})
                current_whale = whales[i]
                run_len = 1
        runs.append({"rec": rec, "whale": current_whale, "run_length": run_len})
    return pd.DataFrame(runs)


def tempo_convergence_over_time(df):
    """Track how tempo difference between whales evolves within an exchange."""
    results = []
    for rec, group in df.groupby("rec"):
        rows = list(group.itertuples())
        switches = []
        for i in range(1, len(rows)):
            if rows[i].whale != rows[i - 1].whale:
                t1, t2 = rows[i - 1].tempo, rows[i].tempo
                diff = abs(t2 - t1) / max(t1, t2) if max(t1, t2) > 0 else 0
                switches.append({
                    "rec": rec,
                    "switch_idx": len(switches),
                    "tempo_diff_pct": diff,
                    "timestamp": rows[i].timestamp,
                })
        if len(switches) >= 4:
            first_half = switches[: len(switches) // 2]
            second_half = switches[len(switches) // 2 :]
            results.append({
                "rec": rec,
                "n_switches": len(switches),
                "first_half_diff": np.mean([s["tempo_diff_pct"] for s in first_half]),
                "second_half_diff": np.mean([s["tempo_diff_pct"] for s in second_half]),
                "convergence": (
                    np.mean([s["tempo_diff_pct"] for s in first_half])
                    - np.mean([s["tempo_diff_pct"] for s in second_half])
                ),
            })
    return pd.DataFrame(results)
