"""
Advanced analyses: predictive modeling, temporal evolution, Hidden Markov Model.
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from math import log2
from scipy import stats


def predictive_model(df, context_orders=[0, 1, 2, 3]):
    """Build and evaluate n-gram predictive models for next coda prediction.
    Returns accuracy at each context order."""
    classes = df["coda_class"].values.tolist()
    n = len(classes)
    results = []

    for order in context_orders:
        if order == 0:
            # predict most common class always
            mc = Counter(classes).most_common(1)[0][0]
            correct = sum(1 for c in classes if c == mc)
            accuracy = correct / n
            results.append({
                "order": 0,
                "accuracy": accuracy,
                "strategy": "most_common (%s)" % mc,
                "n_predictions": n,
            })
        else:
            # build conditional model, predict argmax P(next | context)
            ngram_counts = defaultdict(Counter)
            for i in range(order, n):
                context = tuple(classes[i - order:i])
                ngram_counts[context][classes[i]] += 1

            correct = 0
            total = 0
            unseen_contexts = 0
            for i in range(order, n):
                context = tuple(classes[i - order:i])
                if context in ngram_counts:
                    prediction = ngram_counts[context].most_common(1)[0][0]
                    if prediction == classes[i]:
                        correct += 1
                else:
                    unseen_contexts += 1
                total += 1

            results.append({
                "order": order,
                "accuracy": correct / total if total > 0 else 0,
                "strategy": "argmax P(next | %d-gram context)" % order,
                "n_predictions": total,
                "unseen_contexts": unseen_contexts,
                "n_unique_contexts": len(ngram_counts),
            })

    return results


def cross_whale_prediction(df):
    """Predict the response coda given the prompt coda (across whale switches).
    Tests how much information the prompt carries about the response."""
    prompts = []
    responses = []
    for rec, group in df.groupby("rec"):
        rows = list(group.itertuples())
        for i in range(1, len(rows)):
            if rows[i].whale != rows[i - 1].whale:
                prompts.append(rows[i - 1].coda_class)
                responses.append(rows[i].coda_class)

    if not prompts:
        return {}

    # baseline: predict most common response
    mc = Counter(responses).most_common(1)[0][0]
    baseline_acc = sum(1 for r in responses if r == mc) / len(responses)

    # echo model: predict response = prompt
    echo_acc = sum(1 for p, r in zip(prompts, responses) if p == r) / len(prompts)

    # conditional model: predict argmax P(response | prompt)
    cond = defaultdict(Counter)
    for p, r in zip(prompts, responses):
        cond[p][r] += 1

    correct = 0
    for p, r in zip(prompts, responses):
        prediction = cond[p].most_common(1)[0][0]
        if prediction == r:
            correct += 1
    cond_acc = correct / len(prompts)

    return {
        "n_pairs": len(prompts),
        "baseline_accuracy": baseline_acc,
        "echo_accuracy": echo_acc,
        "conditional_accuracy": cond_acc,
        "echo_lift_over_baseline": echo_acc / baseline_acc if baseline_acc > 0 else 0,
        "conditional_lift_over_echo": cond_acc / echo_acc if echo_acc > 0 else 0,
    }


def temporal_evolution(codas_df):
    """Analyze how coda usage patterns change over the dataset's time span."""
    codas_df = codas_df.copy()
    codas_df["year"] = pd.to_datetime(codas_df["Date"], format="%m/%d/%Y", errors="coerce").dt.year

    yearly = []
    for year, group in codas_df.groupby("year"):
        if pd.isna(year) or len(group) < 20:
            continue
        type_counts = Counter(group["CodaType"])
        total = sum(type_counts.values())
        n_types = len(type_counts)
        h = -sum((c / total) * log2(c / total) for c in type_counts.values() if c > 0)

        # dominant type
        dom_type, dom_count = type_counts.most_common(1)[0]
        dom_pct = dom_count / total

        # click distribution
        mean_clicks = group["nClicks"].mean()
        mean_duration = group["Duration"].mean()

        yearly.append({
            "year": int(year),
            "n_codas": len(group),
            "n_types": n_types,
            "entropy": h,
            "dominant_type": dom_type,
            "dominant_pct": dom_pct,
            "mean_clicks": mean_clicks,
            "mean_duration": mean_duration,
        })

    return pd.DataFrame(yearly)


def vocabulary_drift(codas_df, window_years=3):
    """Measure how much the coda vocabulary changes over time windows."""
    codas_df = codas_df.copy()
    codas_df["year"] = pd.to_datetime(codas_df["Date"], format="%m/%d/%Y", errors="coerce").dt.year

    years = sorted(codas_df["year"].dropna().unique())
    drift = []

    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        set1 = set(codas_df[codas_df["year"] == y1]["CodaType"])
        set2 = set(codas_df[codas_df["year"] == y2]["CodaType"])

        jaccard = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 1
        new_types = set2 - set1
        lost_types = set1 - set2

        # frequency-weighted Jensen-Shannon divergence
        freq1 = Counter(codas_df[codas_df["year"] == y1]["CodaType"])
        freq2 = Counter(codas_df[codas_df["year"] == y2]["CodaType"])
        all_types = set(freq1.keys()) | set(freq2.keys())
        t1 = sum(freq1.values())
        t2 = sum(freq2.values())

        p = np.array([freq1.get(t, 0) / t1 for t in all_types])
        q = np.array([freq2.get(t, 0) / t2 for t in all_types])
        m = (p + q) / 2
        # avoid log(0)
        kl_pm = sum(p[i] * log2(p[i] / m[i]) for i in range(len(p)) if p[i] > 0 and m[i] > 0)
        kl_qm = sum(q[i] * log2(q[i] / m[i]) for i in range(len(q)) if q[i] > 0 and m[i] > 0)
        jsd = (kl_pm + kl_qm) / 2

        drift.append({
            "year_from": int(y1),
            "year_to": int(y2),
            "jaccard_similarity": jaccard,
            "new_types": len(new_types),
            "lost_types": len(lost_types),
            "jensen_shannon_divergence": jsd,
        })

    return pd.DataFrame(drift)


def fit_hmm(df, n_states=3, n_iter=50):
    """Fit a Hidden Markov Model to dialogue sequences.
    Discovers latent conversational states from observed coda classes."""
    # encode coda classes as integers
    classes = df["coda_class"].values.tolist()
    vocab = sorted(set(classes))
    class_to_idx = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)

    # split into exchange sequences
    sequences = []
    for rec, group in df.groupby("rec"):
        seq = [class_to_idx[c] for c in group["coda_class"].values if c in class_to_idx]
        if len(seq) >= 3:
            sequences.append(seq)

    if not sequences:
        return None

    # Baum-Welch (simplified: single long sequence concatenation)
    all_obs = []
    for seq in sequences:
        all_obs.extend(seq)
    T = len(all_obs)
    K = n_states

    rng = np.random.default_rng(42)

    # initialize
    pi = np.ones(K) / K
    A = rng.dirichlet(np.ones(K), size=K)  # transition
    B = rng.dirichlet(np.ones(V), size=K)  # emission

    for iteration in range(n_iter):
        # Forward
        alpha = np.zeros((T, K))
        alpha[0] = pi * B[:, all_obs[0]]
        alpha[0] /= alpha[0].sum() + 1e-300

        for t in range(1, T):
            alpha[t] = alpha[t - 1] @ A * B[:, all_obs[t]]
            s = alpha[t].sum()
            if s > 0:
                alpha[t] /= s

        # Backward
        beta = np.zeros((T, K))
        beta[T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = A @ (B[:, all_obs[t + 1]] * beta[t + 1])
            s = beta[t].sum()
            if s > 0:
                beta[t] /= s

        # Gamma and Xi
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        # re-estimate
        pi = gamma[0]
        pi /= pi.sum()

        for j in range(K):
            for k in range(K):
                num = sum(alpha[t, j] * A[j, k] * B[k, all_obs[t + 1]] * beta[t + 1, k]
                          for t in range(T - 1))
                denom = sum(alpha[t, j] * beta[t, j] for t in range(T - 1))
                A[j, k] = num / (denom + 1e-300)
            A[j] /= A[j].sum() + 1e-300

        for k in range(K):
            for v in range(V):
                B[k, v] = sum(gamma[t, k] for t in range(T) if all_obs[t] == v)
            B[k] /= B[k].sum() + 1e-300

    # decode most likely states using Viterbi
    log_pi = np.log(pi + 1e-300)
    log_A = np.log(A + 1e-300)
    log_B = np.log(B + 1e-300)

    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = log_pi + log_B[:, all_obs[0]]

    for t in range(1, T):
        for j in range(K):
            scores = delta[t - 1] + log_A[:, j]
            psi[t, j] = np.argmax(scores)
            delta[t, j] = scores[psi[t, j]] + log_B[j, all_obs[t]]

    # backtrack
    states = np.zeros(T, dtype=int)
    states[T - 1] = np.argmax(delta[T - 1])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    # characterize states
    state_profiles = []
    for k in range(K):
        mask = states == k
        n_in_state = mask.sum()
        top_emissions = sorted(enumerate(B[k]), key=lambda x: -x[1])[:5]
        top_codas = [(vocab[idx], round(float(prob), 3)) for idx, prob in top_emissions]

        # what whale switches look like in this state
        echo_count = 0
        contrast_count = 0
        obs_in_state = [all_obs[t] for t in range(T) if states[t] == k]

        state_profiles.append({
            "state": k,
            "n_observations": int(n_in_state),
            "fraction": n_in_state / T,
            "top_emissions": top_codas,
            "self_transition": float(A[k, k]),
        })

    return {
        "n_states": K,
        "state_profiles": state_profiles,
        "transition_matrix": A.tolist(),
        "initial_distribution": pi.tolist(),
        "vocab_size": V,
    }
