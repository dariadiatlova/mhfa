import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def compute_cosine_similarity(embedding_a, embedding_b):
    return cosine_similarity(
        [normalize(embedding_a.reshape(1, -1))[0]],
        [normalize(embedding_b.reshape(1, -1))[0]],
    )[0][0]


def compute_p_target(labels):
    target_trials = np.sum(labels)
    total_trials = len(labels)
    p_target = target_trials / total_trials if total_trials > 0 else 0

    return p_target


# Generate target and impostor scores
def generate_scores(embeddings1, embeddings2, labels):
    tar_scores = []
    imp_scores = []
    num_pairs = len(labels)
    for i in range(num_pairs):
        score = compute_cosine_similarity(embeddings1[i], embeddings2[i])
        if labels[i] == 1:
            tar_scores.append(score)
        else:
            imp_scores.append(score)

    return np.array(tar_scores), np.array(imp_scores)


def compute_frr_far(tar, imp):
    # Combine target and impostor scores and find unique thresholds
    thresholds = np.unique(np.hstack((tar, imp)))

    # Initialize arrays to store FRR and FAR
    frr = np.zeros_like(thresholds, dtype=float)
    far = np.zeros_like(thresholds, dtype=float)

    # Compute FRR and FAR for each threshold
    for i, threshold in enumerate(thresholds):
        frr[i] = np.sum(tar < threshold) / len(tar)  # False Rejection Rate
        far[i] = np.sum(imp >= threshold) / len(imp)  # False Acceptance Rate

    # Extend thresholds to ensure it covers all scores
    thresholds = np.hstack((thresholds, thresholds[-1] + 1e-6))
    frr = np.hstack((frr, frr[-1]))
    far = np.hstack((far, far[-1]))

    return thresholds, frr, far


def compute_min_c(pt, tar, imp, c_miss=1, c_fa=1):
    tar_imp, fnr, fpr = compute_frr_far(tar, imp)

    beta = c_fa * (1 - pt) / (c_miss * pt)
    log_beta = np.log(beta)
    act_c = fnr + beta * fpr
    index_min = np.argmin(act_c)
    min_c = act_c[index_min]
    threshold = tar_imp[index_min]

    return min_c, threshold, log_beta


def get_min_c(embeddings1, embeddings2, labels, c_miss=1, c_fa=1):
    p_target = compute_p_target(labels)
    tar, imp = generate_scores(embeddings1, embeddings2, labels)
    min_c = compute_min_c(p_target, tar, imp, c_miss, c_fa)[0]

    return min_c


def compute_eer(tar, imp):
    tar_imp, fr, fa = compute_frr_far(tar, imp)

    index_min = np.argmin(np.abs(fr - fa))
    eer = 100.0 * np.mean((fr[index_min], fa[index_min]))
    threshold = tar_imp[index_min]

    return eer, threshold


def get_eer(embeddings1, embeddings2, labels):
    tar, imp = generate_scores(embeddings1, embeddings2, labels)
    return compute_eer(tar, imp)[0]
