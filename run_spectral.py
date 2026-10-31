"""Run spectral proxy analysis."""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from src.coda_features import load_codas
from src.spectral_proxy import (
    extract_spectral_features, contour_by_coda_type,
    formant_transition_matrix, spectral_entropy_contribution,
    ici_pattern_clustering
)

codas = load_codas()
print("Loaded %d codas" % len(codas))

# Extract spectral features
spectral = extract_spectral_features(codas)
print("Extracted spectral features for %d codas" % len(spectral))

# Contour distribution
contour_counts = spectral["contour_type"].value_counts()
print("\n=== CONTOUR TYPE DISTRIBUTION ===")
for ct, n in contour_counts.items():
    print("  %-12s: %5d (%.1f%%)" % (ct, n, 100*n/len(spectral)))

# Map to expert CodaType
print("\n=== CONTOUR BY EXPERT CODA TYPE ===")
ct_map = contour_by_coda_type(codas, spectral)
for ctype in sorted(ct_map.keys(), key=lambda x: -ct_map[x]["n"])[:12]:
    info = ct_map[ctype]
    a_pct = 100 * (info["contour_distribution"].get("a-rising", 0) +
                   info["contour_distribution"].get("a-falling", 0) +
                   info["contour_distribution"].get("a-flat", 0))
    i_pct = 100 * (info["contour_distribution"].get("i-single", 0) +
                   info["contour_distribution"].get("i-multi", 0))
    print("  %-6s (n=%4d): a-type=%.0f%%, i-type=%.0f%%, complexity=%.2f" % (
        ctype, info["n"], a_pct, i_pct, info["mean_complexity"]))

# Transition matrix
print("\n=== CONTOUR TRANSITION PROBABILITIES ===")
matrix, trans_counts = formant_transition_matrix(codas, spectral)
for from_type in sorted(matrix.keys()):
    row = matrix[from_type]
    top = sorted(row.items(), key=lambda x: -x[1])[:3]
    parts = ", ".join(["%s=%.0f%%" % (t, p*100) for t, p in top])
    print("  %-12s -> %s" % (from_type, parts))

# Entropy contribution
print("\n=== SPECTRAL ENTROPY CONTRIBUTION ===")
entropy = spectral_entropy_contribution(codas, spectral)
for k, v in entropy.items():
    if isinstance(v, float):
        print("  %-30s: %.3f bits" % (k, v))
    elif isinstance(v, dict):
        print("  %s:" % k)
        for kk, vv in list(v.items())[:6]:
            print("    %-12s: %s" % (kk, vv))

# Spectral clustering
print("\n=== SPECTRAL CLUSTERS ===")
cluster_result = ici_pattern_clustering(spectral)
if cluster_result:
    profiles, clustered = cluster_result
    for cl, info in sorted(profiles.items()):
        print("  Cluster %d (n=%d): ratio=%.2f, var=%.4f, complexity=%.1f, contour=%s" % (
            cl, info["n"], info["mean_ratio"], info["mean_variance"],
            info["mean_complexity"], info["dominant_contour"]))
        clicks = info["click_distribution"]
        top_clicks = sorted(clicks.items(), key=lambda x: -x[1])[:3]
        print("    clicks: %s" % ", ".join(["%d=%d" % (k, v) for k, v in top_clicks]))

# Save results
results = {
    "contour_distribution": contour_counts.to_dict(),
    "contour_by_coda_type": {k: {kk: vv for kk, vv in v.items() if kk != "contour_distribution"}
                             for k, v in ct_map.items()},
    "entropy_contribution": {k: v for k, v in entropy.items() if not isinstance(v, dict)},
    "spectral_clusters": profiles if cluster_result else {},
}

with open("results/spectral_proxy.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str, ensure_ascii=False)

print("\nDONE - saved to results/spectral_proxy.json")
