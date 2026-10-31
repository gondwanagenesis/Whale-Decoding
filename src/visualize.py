"""
Visualization utilities for whale coda analysis.
Generates charts and diagrams from analysis results.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def generate_svg_coda_map(embeddings_csv, output_path=None, method="tsne", color_by="coda_type"):
    df = pd.read_csv(embeddings_csv)
    x_col = f"{method}_x"
    y_col = f"{method}_y"
    x = df[x_col].values
    y = df[y_col].values

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    pad = 0.05
    xrange = xmax - xmin
    yrange = ymax - ymin
    xmin -= pad * xrange
    xmax += pad * xrange
    ymin -= pad * yrange
    ymax += pad * yrange

    w, h = 800, 600

    def tx(v):
        return 60 + (v - xmin) / (xmax - xmin) * (w - 80)

    def ty(v):
        return 30 + (v - ymin) / (ymax - ymin) * (h - 60)

    categories = df[color_by].unique()
    palette = _generate_palette(len(categories))
    color_map = dict(zip(categories, palette))

    lines = [
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{w}" height="{h}" fill="var(--bg, #f7f8fa)"/>',
    ]

    for _, row in df.iterrows():
        cx = tx(row[x_col])
        cy = ty(row[y_col])
        c = color_map.get(row[color_by], "#888")
        lines.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.5" fill="{c}" opacity="0.6"/>'
        )

    lines.append("</svg>")
    svg = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(svg)
    return svg


def _generate_palette(n):
    base_hues = [210, 35, 150, 10, 270, 100, 330, 50, 190, 0]
    colors = []
    for i in range(n):
        h = base_hues[i % len(base_hues)]
        s = 65 - (i // len(base_hues)) * 10
        l = 55
        colors.append(f"hsl({h}, {s}%, {l}%)")
    return colors


def findings_summary(findings_json):
    with open(findings_json) as f:
        d = json.load(f)

    lines = []
    lines.append("# Whale Coda Analysis Findings\n")

    ds = d["dataset"]
    lines.append(f"- **{ds['n_codas']}** codas analyzed")
    lines.append(f"- **{ds['n_coda_types']}** distinguishable coda types")
    lines.append(f"- **{ds['n_social_units']}** social units")
    lines.append(f"- Date range: {ds['date_range'][0]} to {ds['date_range'][1]}\n")

    ic = d["information_capacity"]
    lines.append(f"- Information entropy: **{ic['entropy_bits']:.2f}** bits")
    lines.append(f"- Max possible: {ic['max_entropy_bits']:.2f} bits")
    lines.append(f"- Efficiency: {ic['efficiency']:.1%}\n")

    cl = d["clustering"]
    lines.append(f"- Clustering silhouette: {cl['silhouette']:.3f}")
    lines.append(f"- ARI vs labeled types: {cl['ari']:.3f}\n")

    tt = d["turn_taking"]
    lines.append(f"- Mean turn gap: {tt['mean_gap_seconds']:.2f}s")
    lines.append(f"- Overlap rate: {tt['overlap_rate']:.1%}\n")

    co = d["coordination"]
    lines.append(f"- Tempo convergence: {co['mean_tempo_convergence']:.3f}")
    lines.append(f"- Click-count match rate: {co['click_count_match_rate']:.1%}")

    return "\n".join(lines)
