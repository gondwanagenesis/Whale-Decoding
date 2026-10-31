"""Run advanced analyses: prediction, temporal evolution, HMM."""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from src.coda_features import load_codas, load_dialogues
from src.dialogue_deep import load_and_enrich_dialogues
from src.advanced_analysis import (
    predictive_model, cross_whale_prediction,
    temporal_evolution, vocabulary_drift, fit_hmm
)

# Load dialogue data
print("Loading dialogues...")
raw = load_dialogues()
df = load_and_enrich_dialogues(raw)
df = df[df['coda_class'].notna()].copy()
print("  %d dialogue codas" % len(df))

# Load main codas dataset
print("Loading main codas...")
codas = pd.read_csv("vendor/sw-combinatoriality/data/DominicaCodas.csv")
print("  %d codas" % len(codas))

# 1. PREDICTIVE MODEL
print("\n=== PREDICTIVE MODEL ===")
pred = predictive_model(df)
for r in pred:
    extras = ""
    if "unseen_contexts" in r:
        extras = " (unseen=%d, contexts=%d)" % (r["unseen_contexts"], r["n_unique_contexts"])
    print("  Order %d: accuracy=%.1f%% [%s]%s" % (
        r["order"], 100*r["accuracy"], r["strategy"], extras))

# 2. CROSS-WHALE PREDICTION
print("\n=== CROSS-WHALE PREDICTION ===")
xw = cross_whale_prediction(df)
print("  Baseline (most common):   %.1f%%" % (100*xw["baseline_accuracy"]))
print("  Echo model (repeat):      %.1f%%" % (100*xw["echo_accuracy"]))
print("  Conditional (best guess): %.1f%%" % (100*xw["conditional_accuracy"]))
print("  Echo lift over baseline:  %.1fx" % xw["echo_lift_over_baseline"])
print("  n=%d pairs" % xw["n_pairs"])

# 3. TEMPORAL EVOLUTION
print("\n=== TEMPORAL EVOLUTION ===")
temp = temporal_evolution(codas)
for _, row in temp.iterrows():
    print("  %d: n=%4d, types=%2d, H=%.2f, dominant=%s (%.0f%%), clicks=%.1f" % (
        row["year"], row["n_codas"], row["n_types"], row["entropy"],
        row["dominant_type"], 100*row["dominant_pct"], row["mean_clicks"]))

# Overall trends
if len(temp) > 3:
    from scipy.stats import linregress
    years = temp["year"].values
    entropies = temp["entropy"].values
    slope, intercept, r, p, se = linregress(years, entropies)
    print("\n  Entropy trend: slope=%.4f bits/year, R2=%.3f, p=%.3f" % (slope, r**2, p))

    n_types = temp["n_types"].values
    slope2, _, r2, p2, _ = linregress(years, n_types)
    print("  Vocabulary trend: slope=%.2f types/year, R2=%.3f, p=%.3f" % (slope2, r2**2, p2))

    clicks = temp["mean_clicks"].values
    slope3, _, r3, p3, _ = linregress(years, clicks)
    print("  Click count trend: slope=%.4f clicks/year, R2=%.3f, p=%.3f" % (slope3, r3**2, p3))

# 4. VOCABULARY DRIFT
print("\n=== VOCABULARY DRIFT ===")
drift = vocabulary_drift(codas)
for _, row in drift.iterrows():
    print("  %d->%d: Jaccard=%.2f, JSD=%.3f, +%d/-%d types" % (
        row["year_from"], row["year_to"], row["jaccard_similarity"],
        row["jensen_shannon_divergence"], row["new_types"], row["lost_types"]))

mean_jsd = drift["jensen_shannon_divergence"].mean()
mean_jaccard = drift["jaccard_similarity"].mean()
print("  Mean JSD: %.3f, Mean Jaccard: %.2f" % (mean_jsd, mean_jaccard))

# 5. HIDDEN MARKOV MODEL
print("\n=== HIDDEN MARKOV MODEL (3 states) ===")
hmm3 = fit_hmm(df, n_states=3, n_iter=30)
if hmm3:
    for sp in hmm3["state_profiles"]:
        top = ", ".join(["%s(%.0f%%)" % (c, 100*p) for c, p in sp["top_emissions"][:3]])
        print("  State %d: n=%d (%.1f%%), self_trans=%.2f, top=[%s]" % (
            sp["state"], sp["n_observations"], 100*sp["fraction"],
            sp["self_transition"], top))

    print("\n  Transition matrix:")
    A = hmm3["transition_matrix"]
    for i, row in enumerate(A):
        print("    S%d -> %s" % (i, ", ".join(["S%d=%.2f" % (j, p) for j, p in enumerate(row)])))

print("\n=== HMM (5 states) ===")
hmm5 = fit_hmm(df, n_states=5, n_iter=30)
if hmm5:
    for sp in hmm5["state_profiles"]:
        top = ", ".join(["%s(%.0f%%)" % (c, 100*p) for c, p in sp["top_emissions"][:3]])
        print("  State %d: n=%d (%.1f%%), self=%.2f, top=[%s]" % (
            sp["state"], sp["n_observations"], 100*sp["fraction"],
            sp["self_transition"], top))

# Save
results = {
    "predictive_model": pred,
    "cross_whale_prediction": xw,
    "temporal_evolution": temp.to_dict(orient="records"),
    "vocabulary_drift": drift.to_dict(orient="records"),
    "hmm_3_states": hmm3,
    "hmm_5_states": hmm5,
}
with open("results/advanced_analysis.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str, ensure_ascii=False)

print("\nDONE")
