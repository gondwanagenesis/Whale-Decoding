"""
Spectral proxy analysis: extract formant-like features from ICI patterns.

Based on Begus et al. 2026 finding that sperm whale codas contain vowel-like
formant structures visible in ICI ratio patterns. Without raw audio, we derive
proxy spectral features from the ICI timing data:

1. ICI ratio contours (formant proxy)
2. ICI acceleration (second derivative)
3. Contour shape classification (a-coda vs i-coda proxy)
4. Integration with expert CodaType labels
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from math import log2
from scipy import stats


def ici_ratios(icis):
    """Compute consecutive ICI ratios: ICI[n+1]/ICI[n].
    These are proxies for formant transitions in the spectral domain.
    A ratio > 1 = deceleration, < 1 = acceleration."""
    ratios = []
    for i in range(1, len(icis)):
        if icis[i - 1] > 0.01:
            ratios.append(icis[i] / icis[i - 1])
    return ratios


def ici_acceleration(icis):
    """Second derivative of ICI sequence: d^2(ICI)/dt^2.
    Positive = convex (speeding up then slowing), negative = concave."""
    if len(icis) < 3:
        return []
    return [icis[i + 1] - 2 * icis[i] + icis[i - 1] for i in range(1, len(icis) - 1)]


def classify_contour(icis):
    """Classify ICI contour shape as proxy for Begus a-coda/i-coda distinction.

    a-coda (one formant): monotonic or near-flat ICI contour
    i-coda (two formants): ICI contour with inflection point (changes direction)
    """
    if len(icis) < 3:
        return "short"

    diffs = [icis[i + 1] - icis[i] for i in range(len(icis) - 1)]

    sign_changes = 0
    for i in range(1, len(diffs)):
        if diffs[i] * diffs[i - 1] < 0 and abs(diffs[i]) > 0.005 and abs(diffs[i - 1]) > 0.005:
            sign_changes += 1

    if sign_changes == 0:
        if all(d >= -0.005 for d in diffs):
            return "a-rising"  # monotonically increasing ICI = decelerating clicks
        elif all(d <= 0.005 for d in diffs):
            return "a-falling"  # monotonically decreasing = accelerating
        else:
            return "a-flat"
    elif sign_changes == 1:
        return "i-single"  # one inflection = two-formant proxy
    else:
        return "i-multi"  # multiple inflections = complex spectral shape


def extract_spectral_features(df):
    """Extract formant-proxy features from the main codas dataset.
    df must have ICI1..ICI9, nClicks, Duration columns."""
    records = []
    for _, row in df.iterrows():
        n = int(row["nClicks"])
        icis = [row[f"ICI{i}"] for i in range(1, min(n, 10)) if row[f"ICI{i}"] > 0]

        if len(icis) < 2:
            records.append({
                "codaNUM2018": row.get("codaNUM2018", None),
                "n_clicks": n,
                "contour_type": "short",
                "mean_ratio": None,
                "ratio_variance": None,
                "max_acceleration": None,
                "spectral_complexity": 0,
                "initial_ratio": None,
                "final_ratio": None,
            })
            continue

        ratios = ici_ratios(icis)
        accel = ici_acceleration(icis)
        contour = classify_contour(icis)

        mean_r = np.mean(ratios) if ratios else None
        var_r = np.var(ratios) if len(ratios) > 1 else 0
        max_a = max(abs(a) for a in accel) if accel else 0

        # spectral complexity: number of sign changes in acceleration
        complexity = 0
        if len(accel) > 1:
            for i in range(1, len(accel)):
                if accel[i] * accel[i - 1] < 0:
                    complexity += 1

        records.append({
            "codaNUM2018": row.get("codaNUM2018", None),
            "n_clicks": n,
            "contour_type": contour,
            "mean_ratio": mean_r,
            "ratio_variance": var_r,
            "max_acceleration": max_a,
            "spectral_complexity": complexity,
            "initial_ratio": ratios[0] if ratios else None,
            "final_ratio": ratios[-1] if ratios else None,
        })

    return pd.DataFrame(records)


def contour_by_coda_type(codas_df, spectral_df):
    """Map contour types to expert CodaType labels."""
    merged = codas_df[["codaNUM2018", "CodaType"]].merge(
        spectral_df[["codaNUM2018", "contour_type", "mean_ratio", "spectral_complexity"]],
        on="codaNUM2018",
    )

    results = {}
    for ctype, group in merged.groupby("CodaType"):
        if len(group) < 10:
            continue
        contour_dist = group["contour_type"].value_counts(normalize=True).to_dict()
        results[ctype] = {
            "n": len(group),
            "dominant_contour": group["contour_type"].mode().iloc[0],
            "contour_distribution": contour_dist,
            "mean_ratio": group["mean_ratio"].mean(),
            "mean_complexity": group["spectral_complexity"].mean(),
            "is_a_type": contour_dist.get("a-rising", 0) + contour_dist.get("a-falling", 0) + contour_dist.get("a-flat", 0) > 0.5,
        }

    return results


def formant_transition_matrix(codas_df, spectral_df):
    """Build transition probabilities between contour types.
    If a-codas and i-codas have different transition patterns,
    the contour distinction is functionally relevant."""
    merged = codas_df[["codaNUM2018", "CodaType", "Date"]].merge(
        spectral_df[["codaNUM2018", "contour_type"]],
        on="codaNUM2018",
    )
    merged = merged.sort_values(["Date", "codaNUM2018"])

    transitions = Counter()
    prev_contour = None
    prev_date = None
    for _, row in merged.iterrows():
        if prev_contour is not None and row["Date"] == prev_date:
            transitions[(prev_contour, row["contour_type"])] += 1
        prev_contour = row["contour_type"]
        prev_date = row["Date"]

    types = sorted(set(c for pair in transitions for c in pair))
    matrix = {}
    for t in types:
        row_total = sum(transitions.get((t, t2), 0) for t2 in types)
        if row_total > 0:
            matrix[t] = {t2: transitions.get((t, t2), 0) / row_total for t2 in types}

    return matrix, transitions


def spectral_entropy_contribution(codas_df, spectral_df):
    """How much additional entropy does the contour dimension add
    beyond click-count + tempo?"""
    merged = codas_df.merge(
        spectral_df[["codaNUM2018", "contour_type", "spectral_complexity"]],
        on="codaNUM2018",
    )

    # entropy of CodaType alone
    ct_counts = Counter(merged["CodaType"].dropna())
    h_type = -sum((c / sum(ct_counts.values())) * log2(c / sum(ct_counts.values()))
                   for c in ct_counts.values() if c > 0)

    # joint entropy of CodaType + contour
    joint = Counter(zip(merged["CodaType"].fillna("?"), merged["contour_type"].fillna("?")))
    total = sum(joint.values())
    h_joint = -sum((c / total) * log2(c / total) for c in joint.values() if c > 0)

    # conditional entropy H(contour | CodaType)
    h_contour_given_type = h_joint - h_type

    # entropy of contour alone
    cc = Counter(merged["contour_type"].dropna())
    h_contour = -sum((c / sum(cc.values())) * log2(c / sum(cc.values()))
                      for c in cc.values() if c > 0)

    # mutual information
    mi = h_contour - h_contour_given_type

    return {
        "H_CodaType": h_type,
        "H_contour": h_contour,
        "H_joint": h_joint,
        "H_contour_given_CodaType": h_contour_given_type,
        "MI_CodaType_contour": mi,
        "contour_distribution": dict(cc.most_common()),
        "bits_added_by_contour": h_contour_given_type,
    }


def ici_pattern_clustering(spectral_df, n_clusters=6):
    """Cluster codas by their ICI ratio profile to find natural spectral groups."""
    from scipy.cluster.hierarchy import linkage, fcluster

    valid = spectral_df.dropna(subset=["mean_ratio", "ratio_variance", "max_acceleration"]).copy()
    if len(valid) < 50:
        return None

    features = valid[["mean_ratio", "ratio_variance", "max_acceleration", "spectral_complexity"]].values
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    Z = linkage(features, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    valid["spectral_cluster"] = labels

    cluster_profiles = {}
    for cl in range(1, n_clusters + 1):
        sub = valid[valid["spectral_cluster"] == cl]
        if len(sub) == 0:
            continue
        cluster_profiles[cl] = {
            "n": len(sub),
            "mean_ratio": sub["mean_ratio"].mean(),
            "mean_variance": sub["ratio_variance"].mean(),
            "mean_complexity": sub["spectral_complexity"].mean(),
            "dominant_contour": sub["contour_type"].mode().iloc[0] if len(sub) > 0 else "?",
            "click_distribution": sub["n_clicks"].value_counts().to_dict(),
        }

    return cluster_profiles, valid
