"""
Formal grammar extraction from whale dialogue data.
Extracts the minimal rule set that describes observed dialogue patterns.
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from math import log2


def two_state_machine(df):
    """Model dialogue as a two-state (Echo/Contrast) Markov chain."""
    mode_sequences = []
    for rec, group in df.groupby("rec"):
        rows = list(group.itertuples())
        modes = []
        for i in range(1, len(rows)):
            if rows[i].whale != rows[i - 1].whale:
                is_echo = rows[i].coda_class == rows[i - 1].coda_class
                modes.append("E" if is_echo else "C")
        if len(modes) >= 2:
            mode_sequences.append(modes)

    ee, ec, ce, cc = 0, 0, 0, 0
    for seq in mode_sequences:
        for i in range(1, len(seq)):
            if seq[i - 1] == "E" and seq[i] == "E": ee += 1
            elif seq[i - 1] == "E" and seq[i] == "C": ec += 1
            elif seq[i - 1] == "C" and seq[i] == "E": ce += 1
            else: cc += 1

    p_leave_echo = ec / (ee + ec) if (ee + ec) > 0 else 0
    p_leave_contrast = ce / (ce + cc) if (ce + cc) > 0 else 0
    mean_echo_run = 1 / p_leave_echo if p_leave_echo > 0 else float("inf")
    mean_contrast_run = 1 / p_leave_contrast if p_leave_contrast > 0 else float("inf")
    pi_echo = p_leave_contrast / (p_leave_echo + p_leave_contrast) if (p_leave_echo + p_leave_contrast) > 0 else 0.5
    pi_contrast = 1 - pi_echo

    return {
        "transitions": {"EE": ee, "EC": ec, "CE": ce, "CC": cc},
        "P_stay_echo": ee / (ee + ec) if (ee + ec) > 0 else 0,
        "P_stay_contrast": cc / (ce + cc) if (ce + cc) > 0 else 0,
        "mean_echo_run": mean_echo_run,
        "mean_contrast_run": mean_contrast_run,
        "stationary_echo": pi_echo,
        "stationary_contrast": pi_contrast,
    }


def class_selection_rules(df):
    """What classes are selected in echo vs contrast mode."""
    echo_classes = Counter()
    contrast_pairs = Counter()
    contrast_prompt_shift = {"tempo_shift": 0, "click_shift": 0, "both": 0}

    for rec, group in df.groupby("rec"):
        rows = list(group.itertuples())
        for i in range(1, len(rows)):
            if rows[i].whale != rows[i - 1].whale:
                if rows[i].coda_class == rows[i - 1].coda_class:
                    echo_classes[rows[i].coda_class] += 1
                else:
                    contrast_pairs[(rows[i - 1].coda_class, rows[i].coda_class)] += 1
                    same_clicks = rows[i].n_clicks == rows[i - 1].n_clicks
                    same_tempo = rows[i].tempo_band == rows[i - 1].tempo_band
                    if not same_clicks and not same_tempo:
                        contrast_prompt_shift["both"] += 1
                    elif not same_tempo:
                        contrast_prompt_shift["tempo_shift"] += 1
                    else:
                        contrast_prompt_shift["click_shift"] += 1

    total_echo = sum(echo_classes.values())
    echo_dist = {k: v / total_echo for k, v in echo_classes.most_common(10)} if total_echo > 0 else {}
    total_contrast = sum(contrast_prompt_shift.values())
    contrast_mechanism = {k: v / total_contrast for k, v in contrast_prompt_shift.items()} if total_contrast > 0 else {}

    return {
        "echo_class_distribution": echo_dist,
        "contrast_top_pairs": [(p, r, c) for (p, r), c in contrast_pairs.most_common(10)],
        "contrast_mechanism": contrast_mechanism,
        "n_echo": total_echo,
        "n_contrast": sum(contrast_pairs.values()),
    }


def turn_structure_rules(df):
    """Extract the turn-taking rules."""
    runs = []
    for rec, group in df.groupby("rec"):
        whales = group["whale"].values
        run = 1
        for i in range(1, len(whales)):
            if whales[i] == whales[i - 1]:
                run += 1
            else:
                runs.append(run)
                run = 1
        runs.append(run)

    rc = Counter(runs)
    total_runs = sum(rc.values())
    dist = {length: rc.get(length, 0) / total_runs for length in range(1, 9)}
    p_yield = rc[1] / total_runs if total_runs > 0 else 0

    return {
        "turn_length_distribution": dist,
        "P_yield_per_coda": p_yield,
        "mean_turn_length": sum(r * c for r, c in rc.items()) / total_runs if total_runs > 0 else 0,
        "max_observed_turn": max(rc.keys()) if rc else 0,
    }


def opening_protocol(df):
    """What opener class predicts about the exchange mode."""
    opener_into_mode = defaultdict(Counter)
    for rec, group in df.groupby("rec"):
        rows = list(group.itertuples())
        if len(rows) < 3:
            continue
        opener_class = rows[0].coda_class
        for i in range(1, len(rows)):
            if rows[i].whale != rows[i - 1].whale:
                mode = "E" if rows[i].coda_class == rows[i - 1].coda_class else "C"
                opener_into_mode[opener_class][mode] += 1
                break

    rules = {}
    for cls, mc in opener_into_mode.items():
        total = sum(mc.values())
        if total >= 5:
            rules[cls] = {
                "P_echo": mc.get("E", 0) / total,
                "P_contrast": mc.get("C", 0) / total,
                "n": total,
                "predicted_mode": "echo" if mc.get("E", 0) > mc.get("C", 0) else "contrast",
            }

    return dict(sorted(rules.items(), key=lambda x: -x[1]["n"]))


def grammar_completeness(df):
    """How much of the data does the grammar explain?"""
    state_machine = two_state_machine(df)
    trans = state_machine["transitions"]
    total_trans = sum(trans.values())
    correct_by_majority = (trans["EE"] + trans["CC"]) / total_trans if total_trans > 0 else 0

    turn_rules = turn_structure_rules(df)
    single_turn_pct = turn_rules["turn_length_distribution"].get(1, 0)

    class_rules = class_selection_rules(df)
    total_switches = class_rules["n_echo"] + class_rules["n_contrast"]

    echo_top5 = sum(list(class_rules["echo_class_distribution"].values())[:5])
    echo_coverage = echo_top5 * class_rules["n_echo"] / total_switches if total_switches > 0 else 0

    return {
        "state_persistence_accuracy": correct_by_majority,
        "single_coda_turn_pct": single_turn_pct,
        "echo_top5_coverage": echo_top5,
        "total_switches_explained": total_switches,
        "grammar_bits_per_symbol": _grammar_entropy(df),
    }


def _grammar_entropy(df):
    """Residual entropy after applying the grammar rules."""
    classes = [str(c) for c in df["coda_class"].values.tolist()]
    n = len(classes)
    if n < 4:
        return 0.0

    trigrams = Counter()
    bigrams = Counter()
    for i in range(2, n):
        trigrams[(classes[i - 2], classes[i - 1], classes[i])] += 1
        bigrams[(classes[i - 2], classes[i - 1])] += 1

    total = sum(trigrams.values())
    if total == 0:
        return 0.0

    h = 0.0
    for (pp, prev, curr), count in trigrams.items():
        p_joint = count / total
        p_cond = count / bigrams[(pp, prev)]
        h -= p_joint * log2(p_cond)
    return h


def compare_communication_systems():
    """Compare whale codas to other known animal communication systems."""
    return {
        "sperm_whale_codas": {
            "vocabulary_size": 156,
            "entropy_rate_bits": 0.95,
            "zipf_exponent": 1.72,
            "has_combinatorial": True,
            "has_turn_taking": True,
            "has_syntax": "supra-Markov (z=-28)",
            "semantic_bound_bits": 19,
            "channel_capacity_bits": 1.79,
            "confirmed_hockett": 8,
            "notes": "ICI-timing based; 4 independent features (rhythm, tempo, rubato, ornamentation)",
        },
        "bottlenose_dolphin_whistles": {
            "vocabulary_size": "~200 signature whistles per population",
            "entropy_rate_bits": "~2.5 (estimated)",
            "zipf_exponent": "~1.5",
            "has_combinatorial": False,
            "has_turn_taking": True,
            "has_syntax": "sequential (Markov)",
            "semantic_bound_bits": "unknown",
            "channel_capacity_bits": "unknown",
            "confirmed_hockett": 5,
            "notes": "Signature whistles are identity signals; frequency-modulated",
        },
        "humpback_whale_song": {
            "vocabulary_size": "~30 sound units per year",
            "entropy_rate_bits": "~2.0",
            "zipf_exponent": "~1.8",
            "has_combinatorial": True,
            "has_turn_taking": False,
            "has_syntax": "hierarchical (phrase > theme > song)",
            "semantic_bound_bits": "unknown (broadcast, not dialogic)",
            "channel_capacity_bits": "N/A (one-way)",
            "confirmed_hockett": 4,
            "notes": "Cultural transmission; songs evolve annually; no dialogue",
        },
        "vervet_monkey_alarm_calls": {
            "vocabulary_size": 3,
            "entropy_rate_bits": "~1.58",
            "zipf_exponent": "N/A (too few types)",
            "has_combinatorial": False,
            "has_turn_taking": False,
            "has_syntax": "none (single-unit)",
            "semantic_bound_bits": "~1.58 (eagle/snake/leopard)",
            "channel_capacity_bits": "~1.58",
            "confirmed_hockett": 3,
            "notes": "Referential but fixed; no combination or dialogue",
        },
        "bee_waggle_dance": {
            "vocabulary_size": "continuous (angle + duration)",
            "entropy_rate_bits": "~4-5 (analog encoding)",
            "zipf_exponent": "N/A (continuous)",
            "has_combinatorial": True,
            "has_turn_taking": False,
            "has_syntax": "fixed template (angle + waggle duration)",
            "semantic_bound_bits": "~12 (distance + direction)",
            "channel_capacity_bits": "~12",
            "confirmed_hockett": 5,
            "notes": "Displacement + semantics but no dialogue; fixed grammar",
        },
        "bird_song_zebra_finch": {
            "vocabulary_size": "~5-10 syllables",
            "entropy_rate_bits": "~1.5",
            "zipf_exponent": "~1.3",
            "has_combinatorial": True,
            "has_turn_taking": True,
            "has_syntax": "sequential (finite-state)",
            "semantic_bound_bits": "~3-5 (mate quality + territory)",
            "channel_capacity_bits": "~3",
            "confirmed_hockett": 4,
            "notes": "Learned; syntax is finite-state automaton; limited semantics",
        },
        "human_language": {
            "vocabulary_size": "~50,000+ (active)",
            "entropy_rate_bits": "~1.0 (English text)",
            "zipf_exponent": 1.07,
            "has_combinatorial": True,
            "has_turn_taking": True,
            "has_syntax": "recursive (context-free+)",
            "semantic_bound_bits": "unbounded",
            "channel_capacity_bits": "~40-60 per second",
            "confirmed_hockett": 12,
            "notes": "The benchmark; recursive grammar, displacement, meta-language",
        },
    }
