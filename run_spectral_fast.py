"""Run spectral proxy analysis - vectorized version."""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from collections import Counter
from math import log2

print("Loading data...")
codas = pd.read_csv("vendor/sw-combinatoriality/data/DominicaCodas.csv")
print("Loaded %d codas" % len(codas))

# Vectorized ICI extraction
ici_cols = ["ICI1", "ICI2", "ICI3", "ICI4", "ICI5", "ICI6", "ICI7", "ICI8", "ICI9"]
icis = codas[ici_cols].values  # shape (N, 9)
n_clicks = codas["nClicks"].values

# ICI ratios: ICI[i+1] / ICI[i]
ratios = np.zeros((len(codas), 8))
for i in range(8):
    mask = (icis[:, i] > 0.01) & (icis[:, i+1] > 0.01) if i < 8 else np.zeros(len(codas), dtype=bool)
    if i < 8:
        mask = (icis[:, i] > 0.01) & (icis[:, i+1] > 0.01)
        ratios[:, i] = np.where(mask, icis[:, i+1] / icis[:, i], 0)

# ICI differences for contour classification
diffs = np.zeros((len(codas), 8))
for i in range(8):
    diffs[:, i] = icis[:, i+1] - icis[:, i]

# Classify contours
contour_types = []
for idx in range(len(codas)):
    nc = n_clicks[idx]
    active_icis = [icis[idx, i] for i in range(min(nc-1, 9)) if icis[idx, i] > 0]

    if len(active_icis) < 3:
        contour_types.append("short")
        continue

    d = [active_icis[i+1] - active_icis[i] for i in range(len(active_icis)-1)]

    sign_changes = 0
    for i in range(1, len(d)):
        if d[i] * d[i-1] < 0 and abs(d[i]) > 0.005 and abs(d[i-1]) > 0.005:
            sign_changes += 1

    if sign_changes == 0:
        if all(x >= -0.005 for x in d):
            contour_types.append("a-rising")
        elif all(x <= 0.005 for x in d):
            contour_types.append("a-falling")
        else:
            contour_types.append("a-flat")
    elif sign_changes == 1:
        contour_types.append("i-single")
    else:
        contour_types.append("i-multi")

codas["contour_type"] = contour_types

# Compute per-coda features
mean_ratios = []
ratio_vars = []
max_accels = []
complexities = []

for idx in range(len(codas)):
    nc = n_clicks[idx]
    active_icis = [icis[idx, i] for i in range(min(nc-1, 9)) if icis[idx, i] > 0]

    r = [active_icis[i+1]/active_icis[i] for i in range(len(active_icis)-1) if active_icis[i] > 0.01]
    a = [active_icis[i+1] - 2*active_icis[i] + active_icis[i-1] for i in range(1, len(active_icis)-1)] if len(active_icis) >= 3 else []

    mean_ratios.append(np.mean(r) if r else 0)
    ratio_vars.append(np.var(r) if len(r) > 1 else 0)
    max_accels.append(max(abs(x) for x in a) if a else 0)

    cx = 0
    if len(a) > 1:
        for i in range(1, len(a)):
            if a[i] * a[i-1] < 0:
                cx += 1
    complexities.append(cx)

codas["mean_ratio"] = mean_ratios
codas["ratio_variance"] = ratio_vars
codas["max_acceleration"] = max_accels
codas["spectral_complexity"] = complexities

print("\n=== CONTOUR TYPE DISTRIBUTION ===")
cc = codas["contour_type"].value_counts()
for ct, n in cc.items():
    print("  %-12s: %5d (%.1f%%)" % (ct, n, 100*n/len(codas)))

# By expert CodaType
print("\n=== CONTOUR BY EXPERT CODA TYPE (top 12) ===")
for ctype, group in sorted(codas.groupby("CodaType"), key=lambda x: -len(x[1]))[:12]:
    if len(group) < 10:
        continue
    vc = group["contour_type"].value_counts(normalize=True)
    a_pct = 100 * sum(vc.get(t, 0) for t in ["a-rising", "a-falling", "a-flat"])
    i_pct = 100 * sum(vc.get(t, 0) for t in ["i-single", "i-multi"])
    s_pct = 100 * vc.get("short", 0)
    cx = group["spectral_complexity"].mean()
    print("  %-6s (n=%4d): a-type=%.0f%%, i-type=%.0f%%, short=%.0f%%, complexity=%.2f" % (
        ctype, len(group), a_pct, i_pct, s_pct, cx))

# Entropy contribution
print("\n=== SPECTRAL ENTROPY CONTRIBUTION ===")
ct_counts = Counter(codas["CodaType"].dropna())
total_ct = sum(ct_counts.values())
h_type = -sum((c/total_ct) * log2(c/total_ct) for c in ct_counts.values() if c > 0)

joint = Counter(zip(codas["CodaType"].fillna("?"), codas["contour_type"]))
total_j = sum(joint.values())
h_joint = -sum((c/total_j) * log2(c/total_j) for c in joint.values() if c > 0)

h_contour_given_type = h_joint - h_type

contour_counts = Counter(codas["contour_type"])
total_c = sum(contour_counts.values())
h_contour = -sum((c/total_c) * log2(c/total_c) for c in contour_counts.values() if c > 0)

mi = h_contour - h_contour_given_type

print("  H(CodaType):                 %.3f bits" % h_type)
print("  H(contour):                  %.3f bits" % h_contour)
print("  H(contour | CodaType):       %.3f bits" % h_contour_given_type)
print("  MI(CodaType; contour):       %.3f bits" % mi)
print("  -> Contour adds %.3f bits beyond CodaType" % h_contour_given_type)

# Transition patterns between contour types
print("\n=== CONTOUR TRANSITIONS (within same day) ===")
codas_sorted = codas.sort_values(["Date", "codaNUM2018"])
transitions = Counter()
prev_ct = None
prev_date = None
for _, row in codas_sorted[["Date", "contour_type"]].iterrows():
    if prev_ct is not None and row["Date"] == prev_date:
        transitions[(prev_ct, row["contour_type"])] += 1
    prev_ct = row["contour_type"]
    prev_date = row["Date"]

types = sorted(set(c for pair in transitions for c in pair))
for from_t in types:
    total = sum(transitions.get((from_t, to_t), 0) for to_t in types)
    if total < 20:
        continue
    top = sorted([(to_t, transitions.get((from_t, to_t), 0)/total) for to_t in types], key=lambda x: -x[1])[:3]
    parts = ", ".join(["%s=%.0f%%" % (t, p*100) for t, p in top])
    print("  %-12s -> %s (n=%d)" % (from_t, parts, total))

# Spectral clustering
print("\n=== SPECTRAL CLUSTERS ===")
from scipy.cluster.hierarchy import linkage, fcluster

valid = codas.dropna(subset=["mean_ratio"]).copy()
valid = valid[valid["mean_ratio"] > 0].copy()
features = valid[["mean_ratio", "ratio_variance", "max_acceleration", "spectral_complexity"]].values
mu = features.mean(axis=0)
std = features.std(axis=0) + 1e-8
features_norm = (features - mu) / std

Z = linkage(features_norm[:2000], method="ward")  # subsample for speed
labels = fcluster(Z, t=6, criterion="maxclust")
valid_sub = valid.iloc[:2000].copy()
valid_sub["spectral_cluster"] = labels

print("  (subsample of 2000 codas)")
for cl in range(1, 7):
    sub = valid_sub[valid_sub["spectral_cluster"] == cl]
    if len(sub) == 0:
        continue
    dom_ct = sub["contour_type"].mode().iloc[0]
    dom_clicks = sub["nClicks"].mode().iloc[0]
    print("  Cluster %d (n=%d): ratio=%.2f, complexity=%.1f, contour=%s, clicks=%d" % (
        cl, len(sub), sub["mean_ratio"].mean(), sub["spectral_complexity"].mean(),
        dom_ct, dom_clicks))

# Key finding: do a-type and i-type codas have different FUNCTIONS?
print("\n=== FUNCTIONAL DIFFERENCE: a-type vs i-type ===")
a_types = codas[codas["contour_type"].str.startswith("a-")]
i_types = codas[codas["contour_type"].str.startswith("i-")]

print("  a-type codas: n=%d" % len(a_types))
print("    Mean clicks: %.1f, Mean duration: %.3f" % (a_types["nClicks"].mean(), a_types["Duration"].mean()))
print("    Top CodaTypes: %s" % ", ".join(["%s(%d)" % (t, c) for t, c in a_types["CodaType"].value_counts().head(5).items()]))

print("  i-type codas: n=%d" % len(i_types))
print("    Mean clicks: %.1f, Mean duration: %.3f" % (i_types["nClicks"].mean(), i_types["Duration"].mean()))
print("    Top CodaTypes: %s" % ", ".join(["%s(%d)" % (t, c) for t, c in i_types["CodaType"].value_counts().head(5).items()]))

# Statistical test
from scipy.stats import mannwhitneyu
stat, p = mannwhitneyu(a_types["Duration"], i_types["Duration"], alternative="two-sided")
print("  Duration difference: U=%.0f, p=%.2e" % (stat, p))

stat2, p2 = mannwhitneyu(a_types["nClicks"], i_types["nClicks"], alternative="two-sided")
print("  Click count difference: U=%.0f, p=%.2e" % (stat2, p2))

# Unit/clan distribution
print("\n  a-type by clan:")
for clan, group in a_types.groupby("Clan"):
    print("    %s: %d (%.1f%%)" % (clan, len(group), 100*len(group)/len(a_types)))
print("  i-type by clan:")
for clan, group in i_types.groupby("Clan"):
    print("    %s: %d (%.1f%%)" % (clan, len(group), 100*len(group)/len(i_types)))

# Save
results = {
    "contour_distribution": cc.to_dict(),
    "entropy": {
        "H_CodaType": round(h_type, 3),
        "H_contour": round(h_contour, 3),
        "H_contour_given_CodaType": round(h_contour_given_type, 3),
        "MI": round(mi, 3),
    },
    "a_vs_i": {
        "a_count": int(len(a_types)),
        "i_count": int(len(i_types)),
        "a_mean_clicks": round(float(a_types["nClicks"].mean()), 2),
        "i_mean_clicks": round(float(i_types["nClicks"].mean()), 2),
        "a_mean_duration": round(float(a_types["Duration"].mean()), 3),
        "i_mean_duration": round(float(i_types["Duration"].mean()), 3),
        "duration_p": float(p),
        "clicks_p": float(p2),
    },
}
with open("results/spectral_proxy.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str, ensure_ascii=False)

print("\nDONE")
