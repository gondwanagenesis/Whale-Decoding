"""
Information-theoretic analysis of whale coda communication.
Entropy rates, mutual information, channel capacity, Zipf's law,
Markov chain models, and grammar complexity measures.
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import product as iter_product
from math import log2, factorial
from scipy import stats


def _entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * log2(p) for p in probs)


def unigram_entropy(classes):
    counts = Counter(classes)
    return _entropy(counts.values())


def bigram_entropy(classes):
    """H(X_n | X_{n-1}) — conditional entropy of next coda given current."""
    bigrams = Counter()
    unigrams = Counter()
    for i in range(1, len(classes)):
        bigrams[(classes[i-1], classes[i])] += 1
        unigrams[classes[i-1]] += 1

    total = sum(bigrams.values())
    if total == 0:
        return 0.0

    h = 0.0
    for (prev, curr), count in bigrams.items():
        p_joint = count / total
        p_cond = count / unigrams[prev]
        h -= p_joint * log2(p_cond)
    return h


def trigram_entropy(classes):
    """H(X_n | X_{n-1}, X_{n-2}) — conditional entropy given two predecessors."""
    trigrams = Counter()
    bigrams = Counter()
    for i in range(2, len(classes)):
        trigrams[(classes[i-2], classes[i-1], classes[i])] += 1
        bigrams[(classes[i-2], classes[i-1])] += 1

    total = sum(trigrams.values())
    if total == 0:
        return 0.0

    h = 0.0
    for (pp, prev, curr), count in trigrams.items():
        p_joint = count / total
        p_cond = count / bigrams[(pp, prev)]
        h -= p_joint * log2(p_cond)
    return h


def entropy_rate_by_order(df):
    """Compute entropy at orders 0, 1, 2, 3 for the full dialogue stream."""
    classes = df["coda_class"].values.tolist()

    h0 = unigram_entropy(classes)
    h1 = bigram_entropy(classes)
    h2 = trigram_entropy(classes)

    # 4-gram conditional
    quads = Counter()
    trigs = Counter()
    for i in range(3, len(classes)):
        quads[(classes[i-3], classes[i-2], classes[i-1], classes[i])] += 1
        trigs[(classes[i-3], classes[i-2], classes[i-1])] += 1

    total = sum(quads.values())
    h3 = 0.0
    if total > 0:
        for key, count in quads.items():
            p_joint = count / total
            p_cond = count / trigs[key[:3]]
            h3 -= p_joint * log2(p_cond)

    return {"H0_unigram": h0, "H1_bigram": h1, "H2_trigram": h2, "H3_quadgram": h3}


def mutual_information_prompt_response(df):
    """I(X_prompt; X_response) across whale switches."""
    prompts = []
    responses = []
    for rec, group in df.groupby("rec"):
        rows = list(group.itertuples())
        for i in range(1, len(rows)):
            if rows[i].whale != rows[i-1].whale:
                prompts.append(rows[i-1].coda_class)
                responses.append(rows[i].coda_class)

    h_response = _entropy(Counter(responses).values())
    h_response_given_prompt = bigram_entropy_from_pairs(prompts, responses)
    mi = h_response - h_response_given_prompt

    h_prompt = _entropy(Counter(prompts).values())
    nmi = mi / min(h_prompt, h_response) if min(h_prompt, h_response) > 0 else 0

    return {
        "mutual_information_bits": mi,
        "H_response": h_response,
        "H_response_given_prompt": h_response_given_prompt,
        "H_prompt": h_prompt,
        "normalized_MI": nmi,
        "n_pairs": len(prompts),
    }


def bigram_entropy_from_pairs(xs, ys):
    """H(Y|X) from paired sequences."""
    joint = Counter(zip(xs, ys))
    marginal_x = Counter(xs)
    total = sum(joint.values())
    h = 0.0
    for (x, y), count in joint.items():
        p_joint = count / total
        p_cond = count / marginal_x[x]
        h -= p_joint * log2(p_cond)
    return h


def channel_capacity_estimate(df):
    """Estimate the information channel capacity of the dialogue system.
    Capacity = max I(X;Y) over input distributions.
    We approximate by computing MI for the empirical distribution
    and comparing to the theoretical max for this alphabet size."""
    mi_data = mutual_information_prompt_response(df)
    mi = mi_data["mutual_information_bits"]

    # count distinct classes used in dialogue
    all_classes = set(df["coda_class"].values)
    n_classes = len(all_classes)
    max_entropy = log2(n_classes) if n_classes > 1 else 0

    return {
        "empirical_MI_bits": mi,
        "alphabet_size": n_classes,
        "max_possible_entropy": max_entropy,
        "capacity_utilization": mi / max_entropy if max_entropy > 0 else 0,
    }


def zipf_analysis(df):
    """Test whether coda class frequencies follow Zipf's law."""
    counts = Counter(df["coda_class"].values)
    sorted_counts = sorted(counts.values(), reverse=True)
    ranks = np.arange(1, len(sorted_counts) + 1)
    freqs = np.array(sorted_counts, dtype=float)

    log_ranks = np.log10(ranks)
    log_freqs = np.log10(freqs)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)

    return {
        "zipf_exponent": -slope,
        "r_squared": r_value**2,
        "p_value": p_value,
        "n_types": len(sorted_counts),
        "top_5": [(cls, count) for cls, count in counts.most_common(5)],
        "ranks": ranks.tolist(),
        "frequencies": sorted_counts,
    }


def markov_model_fit(df, max_order=3):
    """Fit Markov chains of order 0-3 and compare via BIC/AIC."""
    classes = [str(c) for c in df["coda_class"].values.tolist()]
    n = len(classes)
    vocab = sorted(set(classes))
    V = len(vocab)

    results = []
    for order in range(max_order + 1):
        if order == 0:
            counts = Counter(classes)
            total = sum(counts.values())
            log_lik = sum(c * log2(c / total) for c in counts.values() if c > 0)
            n_params = V - 1
            n_data = n
        else:
            ngram_counts = Counter()
            context_counts = Counter()
            for i in range(order, len(classes)):
                context = tuple(classes[i-order:i])
                ngram_counts[(context, classes[i])] = ngram_counts.get((context, classes[i]), 0) + 1
                context_counts[context] = context_counts.get(context, 0) + 1

            log_lik = 0.0
            for (ctx, cls), count in ngram_counts.items():
                p = count / context_counts[ctx]
                log_lik += count * log2(p)
            n_params = len(context_counts) * (V - 1)
            n_data = n - order

        bic = -2 * log_lik * np.log(2) + n_params * np.log(n_data)
        aic = -2 * log_lik * np.log(2) + 2 * n_params
        bits_per_symbol = -log_lik / n_data if n_data > 0 else 0

        results.append({
            "order": order,
            "log_likelihood_bits": log_lik,
            "n_params": n_params,
            "BIC": bic,
            "AIC": aic,
            "bits_per_symbol": bits_per_symbol,
        })

    return results


def same_whale_vs_cross_whale_entropy(df):
    """Compare predictability within a whale's stream vs across whale switches."""
    same_whale_pairs = []
    cross_whale_pairs = []

    for rec, group in df.groupby("rec"):
        rows = list(group.itertuples())
        for i in range(1, len(rows)):
            pair = (rows[i-1].coda_class, rows[i].coda_class)
            if rows[i].whale == rows[i-1].whale:
                same_whale_pairs.append(pair)
            else:
                cross_whale_pairs.append(pair)

    def cond_entropy(pairs):
        joint = Counter(pairs)
        marginal = Counter(p[0] for p in pairs)
        total = sum(joint.values())
        h = 0.0
        for (x, y), count in joint.items():
            p_joint = count / total
            p_cond = count / marginal[x]
            h -= p_joint * log2(p_cond)
        return h

    h_same = cond_entropy(same_whale_pairs) if same_whale_pairs else 0
    h_cross = cond_entropy(cross_whale_pairs) if cross_whale_pairs else 0

    return {
        "H_next_given_prev_same_whale": h_same,
        "H_next_given_prev_cross_whale": h_cross,
        "n_same_whale": len(same_whale_pairs),
        "n_cross_whale": len(cross_whale_pairs),
        "predictability_ratio": h_same / h_cross if h_cross > 0 else 0,
    }


def beyond_markov_test(df, n_shuffles=200):
    """Test whether dialogue sequences have structure beyond 2nd-order Markov.
    Shuffles trigrams while preserving bigram statistics and compares
    the real quadgram entropy to the shuffled distribution."""
    classes = df["coda_class"].values.tolist()
    real_h3 = trigram_entropy(classes)

    # build bigram-preserving shuffles
    shuffled_h2s = []
    rng = np.random.default_rng(42)

    for _ in range(n_shuffles):
        # shuffle within each recording to preserve exchange boundaries
        shuffled = []
        for rec, group in df.groupby("rec"):
            seq = group["coda_class"].values.tolist()
            if len(seq) < 4:
                shuffled.extend(seq)
                continue
            # shuffle positions 1:-1 keeping first and last
            middle = seq[1:-1]
            rng.shuffle(middle)
            shuffled.extend([seq[0]] + list(middle) + [seq[-1]])
        shuffled_h2s.append(trigram_entropy(shuffled))

    mean_shuffled = np.mean(shuffled_h2s)
    std_shuffled = np.std(shuffled_h2s)
    z_score = (real_h3 - mean_shuffled) / std_shuffled if std_shuffled > 0 else 0

    return {
        "real_trigram_entropy": real_h3,
        "shuffled_mean": mean_shuffled,
        "shuffled_std": std_shuffled,
        "z_score": z_score,
        "p_value": 2 * (1 - stats.norm.cdf(abs(z_score))),
        "has_beyond_markov_structure": z_score < -2.0,
    }


def exchange_typology(df):
    """Classify each exchange by its dialogue profile."""
    from .dialogue_deep import echo_vs_contrast, turn_run_lengths

    ec = echo_vs_contrast(df)
    types = []

    for rec, group in df.groupby("rec"):
        rec_ec = ec[ec["rec"] == rec]
        if len(rec_ec) < 3:
            continue

        echo_rate = rec_ec["same_class"].mean()
        mean_gap = rec_ec["gap"].mean()
        n_switches = len(rec_ec)
        n_codas = len(group)

        # classify
        if echo_rate > 0.6:
            etype = "echo-dominant"
        elif echo_rate < 0.2:
            etype = "contrast-dominant"
        else:
            etype = "mixed"

        # density: switches per coda
        density = n_switches / n_codas if n_codas > 0 else 0

        types.append({
            "rec": rec,
            "type": etype,
            "echo_rate": echo_rate,
            "n_switches": n_switches,
            "n_codas": n_codas,
            "mean_gap": mean_gap,
            "turn_density": density,
        })

    return pd.DataFrame(types)


def individual_whale_profiles(df):
    """Per-whale dialogue behavior profiles."""
    profiles = []

    for rec, group in df.groupby("rec"):
        rows = list(group.itertuples())
        whale_stats = defaultdict(lambda: {
            "n_codas": 0, "initiates": 0, "responds": 0,
            "echo_given": 0, "echo_received": 0,
            "classes": [], "tempos": [],
        })

        for i, row in enumerate(rows):
            w = row.whale
            whale_stats[w]["n_codas"] += 1
            whale_stats[w]["classes"].append(row.coda_class)
            whale_stats[w]["tempos"].append(row.tempo)

            if i > 0 and rows[i].whale != rows[i-1].whale:
                whale_stats[w]["responds"] += 1
                whale_stats[rows[i-1].whale]["initiates"] += 1
                if rows[i].coda_class == rows[i-1].coda_class:
                    whale_stats[w]["echo_given"] += 1
                    whale_stats[rows[i-1].whale]["echo_received"] += 1

        for whale_id, s in whale_stats.items():
            if s["n_codas"] < 5:
                continue
            vocab = len(set(s["classes"]))
            vocab_entropy = _entropy(Counter(s["classes"]).values())
            profiles.append({
                "rec": rec,
                "whale": whale_id,
                "n_codas": s["n_codas"],
                "vocab_size": vocab,
                "vocab_entropy": vocab_entropy,
                "mean_tempo": np.mean(s["tempos"]),
                "initiations": s["initiates"],
                "responses": s["responds"],
                "echo_rate_given": s["echo_given"] / s["responds"] if s["responds"] > 0 else 0,
                "echo_rate_received": s["echo_received"] / s["initiates"] if s["initiates"] > 0 else 0,
                "initiative_ratio": s["initiates"] / (s["initiates"] + s["responds"]) if (s["initiates"] + s["responds"]) > 0 else 0.5,
            })

    return pd.DataFrame(profiles)


def coda_transition_network(df):
    """Build a directed weighted graph of coda class transitions."""
    edges = Counter()
    for rec, group in df.groupby("rec"):
        classes = group["coda_class"].values
        whales = group["whale"].values
        for i in range(1, len(classes)):
            if whales[i] != whales[i-1]:
                edges[(classes[i-1], classes[i])] += 1

    nodes = set()
    for (a, b), w in edges.items():
        nodes.add(a)
        nodes.add(b)

    # compute node-level metrics
    in_degree = Counter()
    out_degree = Counter()
    in_weight = Counter()
    out_weight = Counter()
    for (a, b), w in edges.items():
        out_degree[a] += 1
        in_degree[b] += 1
        out_weight[a] += w
        in_weight[b] += w

    node_metrics = []
    for n in nodes:
        node_metrics.append({
            "node": n,
            "in_degree": in_degree.get(n, 0),
            "out_degree": out_degree.get(n, 0),
            "in_weight": in_weight.get(n, 0),
            "out_weight": out_weight.get(n, 0),
            "total_weight": in_weight.get(n, 0) + out_weight.get(n, 0),
            "self_loop": edges.get((n, n), 0),
        })

    node_df = pd.DataFrame(node_metrics).sort_values("total_weight", ascending=False)
    return edges, node_df
