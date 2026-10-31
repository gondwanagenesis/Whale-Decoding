"""
Main analysis pipeline for the Dominica sperm whale coda dataset.

Runs feature extraction, clustering, dialogue analysis, and outputs
a structured findings report.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.coda_features import load_codas, load_dialogues, extract_all_features
from src.coda_analyzer import (
    build_feature_matrix,
    cluster_codas,
    evaluate_clustering,
    reduce_dimensions,
    coda_type_statistics,
    unit_repertoire,
    information_capacity,
)
from src.dialogue_analyzer import (
    parse_exchanges,
    turn_taking_stats,
    tempo_matching,
    rhythm_matching,
    exchange_structure,
    coda_coordination_score,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    section("1. LOADING DATA")
    codas = load_codas()
    dialogues = load_dialogues()
    print(f"Codas: {len(codas)} rows")
    print(f"Dialogues: {len(dialogues)} rows")
    print(f"Date range: {codas['Date'].min()} to {codas['Date'].max()}")
    print(f"Coda types: {codas['CodaType'].nunique()}")
    print(f"Social units: {codas['Unit'].nunique()}")
    print(f"Click counts: {sorted(codas['nClicks'].unique())}")

    section("2. FEATURE EXTRACTION")
    features = extract_all_features(codas)
    print(f"Features extracted for {len(features)} codas")
    print(f"\nTempo (mean ICI in seconds):")
    print(f"  mean={features['tempo'].mean():.4f}, std={features['tempo'].std():.4f}")
    print(f"  range=[{features['tempo'].min():.4f}, {features['tempo'].max():.4f}]")
    print(f"\nRubato (timing deviation):")
    print(f"  mean={features['rubato'].mean():.4f}, std={features['rubato'].std():.4f}")
    print(f"\nOrnamentation:")
    orn_counts = features["ornamentation"].value_counts().sort_index()
    for orn, count in orn_counts.items():
        print(f"  {int(orn)} extra clicks: {count} codas ({100*count/len(features):.1f}%)")
    print(f"\nClicks per coda:")
    click_counts = features["n_clicks"].value_counts().sort_index()
    for nc, count in click_counts.items():
        print(f"  {nc} clicks: {count} ({100*count/len(features):.1f}%)")

    section("3. INFORMATION CAPACITY")
    info = information_capacity(features)
    print(f"Distinguishable coda types: {info['n_distinguishable_types']}")
    print(f"Entropy: {info['entropy_bits']:.2f} bits")
    print(f"Max entropy: {info['max_entropy_bits']:.2f} bits")
    print(f"Efficiency (H/Hmax): {info['efficiency']:.2%}")

    section("4. CLUSTERING vs LABELED TYPES")
    X = build_feature_matrix(features)
    labels_auto, Z, scaler = cluster_codas(X, n_clusters=info["n_distinguishable_types"])
    eval_result = evaluate_clustering(
        X, labels_auto,
        true_labels=pd.Categorical(features["coda_type"]).codes,
    )
    print(f"Auto-clusters: {eval_result['n_clusters']}")
    print(f"Silhouette score: {eval_result['silhouette']:.3f}")
    print(f"Adjusted Rand Index vs labeled types: {eval_result['ari']:.3f}")

    section("5. CODA TYPE STATISTICS")
    type_stats = coda_type_statistics(features)
    print(f"\nTop 20 coda types by frequency:")
    print(type_stats.head(20).to_string(index=False))

    section("6. SOCIAL UNIT REPERTOIRES")
    reps = unit_repertoire(features)
    for unit, info_u in sorted(reps.items(), key=lambda x: -x[1]["total_codas"]):
        print(f"\n  Unit {unit}: {info_u['total_codas']} codas, "
              f"{info_u['n_types']} types, entropy={info_u['type_entropy']:.2f} bits")
        top3 = info_u["top_types"][:3]
        top_str = ", ".join(f"{t}({c})" for t, c in top3)
        print(f"    Top types: {top_str}")

    section("7. DIALOGUE ANALYSIS")
    exchanges = parse_exchanges(dialogues)
    print(f"Parsed {len(exchanges)} exchanges")

    tt_stats = turn_taking_stats(exchanges)
    print(f"\nTurn-taking:")
    print(f"  Total turn switches: {tt_stats['n_turn_switches']}")
    print(f"  Mean gap: {tt_stats['mean_gap_seconds']:.2f}s")
    print(f"  Median gap: {tt_stats['median_gap_seconds']:.2f}s")
    print(f"  Overlap rate: {tt_stats['overlap_rate']:.2%}")

    section("8. TEMPO COORDINATION")
    tm = tempo_matching(exchanges)
    if len(tm) > 0:
        print(f"Turn switches analyzed: {len(tm)}")
        print(f"Mean tempo ratio (response/prompt): {tm['tempo_ratio'].mean():.3f}")
        print(f"Median tempo ratio: {tm['tempo_ratio'].median():.3f}")
        print(f"Mean tempo difference: {tm['tempo_diff'].mean():.4f}s")
        close_match = (tm["tempo_ratio"].between(0.8, 1.2)).mean()
        print(f"Tempo within 20% match: {close_match:.1%}")

    section("9. RHYTHM COORDINATION")
    rm = rhythm_matching(exchanges)
    if len(rm) > 0:
        print(f"Mean rhythm distance: {rm['rhythm_distance'].mean():.4f}")
        print(f"Same click count rate: {rm['same_click_count'].mean():.1%}")

    section("10. EXCHANGE STRUCTURE")
    es = exchange_structure(exchanges)
    print(f"Mean turns per exchange: {es['total_turns'].mean():.1f}")
    print(f"Mean balance (min/max whale turns): {es['balance'].mean():.2f}")
    print(f"Mean exchange duration: {es['exchange_duration'].mean():.1f}s")
    print(f"Mean turn switches per exchange: {es['turn_switches'].mean():.1f}")

    section("11. COORDINATION SCORES")
    cs = coda_coordination_score(exchanges)
    if len(cs) > 0:
        print(f"Exchanges scored: {len(cs)}")
        print(f"Mean tempo convergence: {cs['mean_tempo_convergence'].mean():.3f}")
        print(f"Mean click-count match rate: {cs['click_count_match_rate'].mean():.1%}")
        high_coord = cs[cs["mean_tempo_convergence"] > 0.8]
        print(f"Highly coordinated exchanges (>0.8): {len(high_coord)} ({100*len(high_coord)/len(cs):.1f}%)")

    section("12. DIMENSIONALITY REDUCTION")
    coords_pca = reduce_dimensions(X, method="pca")
    coords_tsne = reduce_dimensions(X, method="tsne", perplexity=min(30, len(X) - 1))
    print(f"PCA and t-SNE computed for {len(X)} codas")

    pca_df = pd.DataFrame({
        "pca_x": coords_pca[:, 0],
        "pca_y": coords_pca[:, 1],
        "tsne_x": coords_tsne[:, 0],
        "tsne_y": coords_tsne[:, 1],
        "coda_type": features["coda_type"].values,
        "unit": features["unit"].values,
        "n_clicks": features["n_clicks"].values,
        "tempo": features["tempo"].values,
        "rubato": features["rubato"].values,
        "cluster": labels_auto,
    })
    pca_df.to_csv(OUTPUT_DIR / "embeddings_2d.csv", index=False)
    print(f"Saved to results/embeddings_2d.csv")

    section("13. SAVING RESULTS")
    features.to_csv(OUTPUT_DIR / "features.csv", index=False)
    type_stats.to_csv(OUTPUT_DIR / "coda_type_stats.csv", index=False)
    if len(tm) > 0:
        tm.to_csv(OUTPUT_DIR / "tempo_matching.csv", index=False)
    if len(rm) > 0:
        rm.to_csv(OUTPUT_DIR / "rhythm_matching.csv", index=False)
    es.to_csv(OUTPUT_DIR / "exchange_structure.csv", index=False)
    if len(cs) > 0:
        cs.to_csv(OUTPUT_DIR / "coordination_scores.csv", index=False)

    findings = {
        "dataset": {
            "n_codas": len(features),
            "n_coda_types": int(features["coda_type"].nunique()),
            "n_social_units": int(features["unit"].nunique()),
            "date_range": [str(codas["Date"].min()), str(codas["Date"].max())],
        },
        "information_capacity": info,
        "clustering": eval_result,
        "turn_taking": tt_stats,
        "coordination": {
            "mean_tempo_convergence": float(cs["mean_tempo_convergence"].mean()) if len(cs) else 0,
            "click_count_match_rate": float(cs["click_count_match_rate"].mean()) if len(cs) else 0,
        },
    }
    with open(OUTPUT_DIR / "findings.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)

    print(f"\nAll results saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
