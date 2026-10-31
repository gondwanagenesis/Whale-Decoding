"""Run grammar extraction and cross-species comparison."""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from src.coda_features import load_dialogues
from src.dialogue_deep import load_and_enrich_dialogues
from src.grammar_extraction import (
    two_state_machine, class_selection_rules, turn_structure_rules,
    opening_protocol, grammar_completeness, compare_communication_systems
)

raw = load_dialogues()
df = load_and_enrich_dialogues(raw)
df = df[df['coda_class'].notna()].copy()

state = two_state_machine(df)
classes = class_selection_rules(df)
turns = turn_structure_rules(df)
openers = opening_protocol(df)
completeness = grammar_completeness(df)
comparison = compare_communication_systems()

results = {
    'two_state_machine': state,
    'class_selection': {
        'echo_distribution': {k: round(v, 3) for k, v in classes['echo_class_distribution'].items()},
        'contrast_top_pairs': [(p, r, c) for p, r, c in classes['contrast_top_pairs'][:8]],
        'contrast_mechanism': {k: round(v, 3) for k, v in classes['contrast_mechanism'].items()},
        'n_echo': classes['n_echo'],
        'n_contrast': classes['n_contrast'],
    },
    'turn_structure': {
        'P_yield': round(turns['P_yield_per_coda'], 3),
        'mean_turn': round(turns['mean_turn_length'], 2),
        'max_turn': turns['max_observed_turn'],
    },
    'opening_rules': {
        k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()}
        for k, v in list(openers.items())[:8]
    },
    'completeness': {k: round(v, 3) if isinstance(v, float) else v for k, v in completeness.items()},
    'comparison': comparison,
}

with open('results/grammar_rules.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, default=str, ensure_ascii=False)

# Print summary
print("=== TWO-STATE MACHINE ===")
s = state
print("  P(stay Echo):     %.3f" % s["P_stay_echo"])
print("  P(stay Contrast): %.3f" % s["P_stay_contrast"])
print("  Mean echo run:    %.1f turns" % s["mean_echo_run"])
print("  Mean contrast run:%.1f turns" % s["mean_contrast_run"])
print("  Stationary: Echo=%.1f%%, Contrast=%.1f%%" % (s["stationary_echo"]*100, s["stationary_contrast"]*100))
print()
print("=== CLASS SELECTION ===")
print("  Echo codas: n=%d" % classes["n_echo"])
for k, v in list(classes['echo_class_distribution'].items())[:5]:
    print("    %-15s: %.1f%%" % (k, v*100))
print("  Contrast: n=%d" % classes["n_contrast"])
m = classes["contrast_mechanism"]
print("    Mechanism: tempo=%.1f%%, clicks=%.1f%%, both=%.1f%%" % (m["tempo_shift"]*100, m["click_shift"]*100, m["both"]*100))
print()
print("=== TURN STRUCTURE ===")
print("  P(yield per coda): %.3f" % turns["P_yield_per_coda"])
print("  Mean turn length:  %.2f" % turns["mean_turn_length"])
print("  Max observed:      %d" % turns["max_observed_turn"])
print()
print("=== OPENING PROTOCOL ===")
for cls, rule in list(openers.items())[:6]:
    print("  %-12s: %-8s (P=%.0f%%, n=%d)" % (cls, rule["predicted_mode"], max(rule["P_echo"], rule["P_contrast"])*100, rule["n"]))
print()
print("=== GRAMMAR COMPLETENESS ===")
for k, v in completeness.items():
    if isinstance(v, float):
        print("  %s: %.3f" % (k, v))
    else:
        print("  %s: %s" % (k, v))
print()
print("=== CROSS-SPECIES COMPARISON ===")
for species, data in comparison.items():
    print("  %s:" % species)
    print("    vocab=%s, Hockett=%s/12, syntax=%s" % (data["vocabulary_size"], data["confirmed_hockett"], data["has_syntax"]))

print("\nDONE - saved to results/grammar_rules.json")
