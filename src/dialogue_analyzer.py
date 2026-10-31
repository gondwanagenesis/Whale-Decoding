"""
Dialogue and turn-taking analysis for sperm whale exchanges.

Analyzes the sperm-whale-dialogues dataset: timing, coda matching
between whale pairs, conversational structure, and coordination patterns.
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from .coda_features import get_icis, extract_tempo, extract_rubato, extract_rhythm


def parse_exchanges(df):
    exchanges = []
    for rec_id, group in df.groupby("REC"):
        group = group.sort_values("TsTo")
        turns = []
        for _, row in group.iterrows():
            icis = get_icis(row)
            turns.append({
                "rec": rec_id,
                "whale": int(row["Whale"]),
                "timestamp": float(row["TsTo"]),
                "n_clicks": int(row["nClicks"]),
                "duration": float(row["Duration"]),
                "icis": icis,
                "tempo": extract_tempo(icis),
                "rubato": extract_rubato(icis),
                "rhythm": extract_rhythm(icis),
            })
        exchanges.append({"rec": rec_id, "turns": turns, "n_turns": len(turns)})
    return exchanges


def turn_taking_stats(exchanges):
    gaps = []
    overlaps = 0
    total_pairs = 0

    for ex in exchanges:
        turns = ex["turns"]
        for i in range(1, len(turns)):
            if turns[i]["whale"] != turns[i - 1]["whale"]:
                gap = turns[i]["timestamp"] - (
                    turns[i - 1]["timestamp"] + turns[i - 1]["duration"]
                )
                gaps.append(gap)
                total_pairs += 1
                if gap < 0:
                    overlaps += 1

    gaps = np.array(gaps)
    return {
        "n_turn_switches": total_pairs,
        "mean_gap_seconds": float(np.mean(gaps)) if len(gaps) else 0,
        "median_gap_seconds": float(np.median(gaps)) if len(gaps) else 0,
        "std_gap_seconds": float(np.std(gaps)) if len(gaps) else 0,
        "overlap_rate": overlaps / total_pairs if total_pairs else 0,
        "min_gap": float(np.min(gaps)) if len(gaps) else 0,
        "max_gap": float(np.max(gaps)) if len(gaps) else 0,
    }


def tempo_matching(exchanges):
    results = []
    for ex in exchanges:
        turns = ex["turns"]
        for i in range(1, len(turns)):
            if turns[i]["whale"] != turns[i - 1]["whale"]:
                t1 = turns[i - 1]["tempo"]
                t2 = turns[i]["tempo"]
                if t1 > 0:
                    ratio = t2 / t1
                    results.append({
                        "rec": ex["rec"],
                        "tempo_prev": t1,
                        "tempo_next": t2,
                        "tempo_ratio": ratio,
                        "tempo_diff": abs(t2 - t1),
                    })
    return pd.DataFrame(results)


def rhythm_matching(exchanges):
    results = []
    for ex in exchanges:
        turns = ex["turns"]
        for i in range(1, len(turns)):
            if turns[i]["whale"] != turns[i - 1]["whale"]:
                r1 = turns[i - 1]["rhythm"]
                r2 = turns[i]["rhythm"]
                n = max(len(r1), len(r2))
                a, b = np.zeros(n), np.zeros(n)
                a[:len(r1)] = r1
                b[:len(r2)] = r2
                dist = float(np.sqrt(np.sum((a - b) ** 2)))
                same_n = turns[i]["n_clicks"] == turns[i - 1]["n_clicks"]
                results.append({
                    "rec": ex["rec"],
                    "rhythm_distance": dist,
                    "same_click_count": same_n,
                    "clicks_prev": turns[i - 1]["n_clicks"],
                    "clicks_next": turns[i]["n_clicks"],
                })
    return pd.DataFrame(results)


def exchange_structure(exchanges):
    structures = []
    for ex in exchanges:
        turns = ex["turns"]
        whale_sequence = [t["whale"] for t in turns]
        click_sequence = [t["n_clicks"] for t in turns]

        w1_count = whale_sequence.count(1)
        w2_count = whale_sequence.count(2)
        switches = sum(
            1 for i in range(1, len(whale_sequence))
            if whale_sequence[i] != whale_sequence[i - 1]
        )

        structures.append({
            "rec": ex["rec"],
            "total_turns": len(turns),
            "whale1_turns": w1_count,
            "whale2_turns": w2_count,
            "turn_switches": switches,
            "balance": min(w1_count, w2_count) / max(w1_count, w2_count)
            if max(w1_count, w2_count) > 0
            else 0,
            "exchange_duration": turns[-1]["timestamp"] - turns[0]["timestamp"]
            if len(turns) > 1
            else 0,
            "click_types_used": len(set(click_sequence)),
            "whale_sequence": whale_sequence,
        })
    return pd.DataFrame(structures)


def coda_coordination_score(exchanges):
    scores = []
    for ex in exchanges:
        turns = ex["turns"]
        if len(turns) < 4:
            continue
        tempo_diffs = []
        for i in range(1, len(turns)):
            if turns[i]["whale"] != turns[i - 1]["whale"]:
                t1, t2 = turns[i - 1]["tempo"], turns[i]["tempo"]
                if t1 > 0 and t2 > 0:
                    tempo_diffs.append(abs(t2 - t1) / max(t1, t2))

        click_matches = sum(
            1
            for i in range(1, len(turns))
            if turns[i]["whale"] != turns[i - 1]["whale"]
            and turns[i]["n_clicks"] == turns[i - 1]["n_clicks"]
        )
        total_switches = sum(
            1
            for i in range(1, len(turns))
            if turns[i]["whale"] != turns[i - 1]["whale"]
        )

        scores.append({
            "rec": ex["rec"],
            "mean_tempo_convergence": 1 - np.mean(tempo_diffs) if tempo_diffs else 0,
            "click_count_match_rate": click_matches / total_switches
            if total_switches
            else 0,
            "n_switches": total_switches,
        })
    return pd.DataFrame(scores)
