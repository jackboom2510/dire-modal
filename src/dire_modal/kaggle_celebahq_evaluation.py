"""Pure-Python test metrics and paired, stratified bootstrap for CelebA-HQ SDv2.

Evaluation is deliberately downstream of checkpoint selection. The primary
summary is macro balanced-prior AP across the four predeclared held-out
families; SDv2 is reported only as a seen-generator control.
"""

from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

PRIMARY_HOLDOUT_FAMILIES = ("dalle2", "if", "midjourney", "stargan")
SEEN_CONTROL_FAMILY = "sdv2"
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 3407
ACC_THRESHOLD = 0.5


def _validate_scores(real_scores: Sequence[float], fake_scores: Sequence[float]) -> None:
    if not real_scores or not fake_scores:
        raise ValueError("each generator comparison needs non-empty real and fake scores")
    if any(not math.isfinite(float(score)) for score in (*real_scores, *fake_scores)):
        raise ValueError("scores must all be finite probabilities/logits")


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-aware non-interpolated AP, matching sklearn's threshold grouping."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must have the same non-zero length")
    positives = sum(int(label) for label in labels)
    if positives == 0:
        raise ValueError("AP is undefined without positive examples")
    order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
    tp = fp = 0
    previous_recall = 0.0
    result = 0.0
    i = 0
    while i < len(order):
        j = i
        threshold = float(scores[order[i]])
        while j < len(order) and float(scores[order[j]]) == threshold:
            if int(labels[order[j]]) == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        result += (recall - previous_recall) * precision
        previous_recall = recall
        i = j
    return result


def balanced_prior_average_precision(
    real_scores: Sequence[float], fake_scores: Sequence[float]
) -> float:
    """Weighted AP with total real/fake sample weight each fixed at 0.5."""
    _validate_scores(real_scores, fake_scores)
    n_real, n_fake = len(real_scores), len(fake_scores)
    rows = [
        (float(score), 0, 0.5 / n_real) for score in real_scores
    ] + [
        (float(score), 1, 0.5 / n_fake) for score in fake_scores
    ]
    rows.sort(key=lambda row: -row[0])
    true_positive_weight = false_positive_weight = 0.0
    previous_recall = 0.0
    result = 0.0
    i = 0
    while i < len(rows):
        j = i
        score = rows[i][0]
        while j < len(rows) and rows[j][0] == score:
            if rows[j][1]:
                true_positive_weight += rows[j][2]
            else:
                false_positive_weight += rows[j][2]
            j += 1
        recall = true_positive_weight / 0.5
        precision = true_positive_weight / (true_positive_weight + false_positive_weight)
        result += (recall - previous_recall) * precision
        previous_recall = recall
        i = j
    return result


def binary_auroc(real_scores: Sequence[float], fake_scores: Sequence[float]) -> float:
    """Mann–Whitney AUROC with half-credit for tied real/fake scores."""
    _validate_scores(real_scores, fake_scores)
    rows = sorted((float(score), 0) for score in real_scores)
    rows.extend((float(score), 1) for score in fake_scores)
    rows.sort(key=lambda row: row[0])
    negatives_before = 0
    wins = 0.0
    i = 0
    while i < len(rows):
        j = i
        score = rows[i][0]
        group_negative = group_positive = 0
        while j < len(rows) and rows[j][0] == score:
            if rows[j][1]:
                group_positive += 1
            else:
                group_negative += 1
            j += 1
        wins += group_positive * negatives_before
        wins += 0.5 * group_positive * group_negative
        negatives_before += group_negative
        i = j
    return wins / (len(real_scores) * len(fake_scores))


def _family_metrics(
    real_scores: Sequence[float], fake_scores: Sequence[float], threshold: float = ACC_THRESHOLD
) -> dict[str, float | int]:
    _validate_scores(real_scores, fake_scores)
    labels = [0] * len(real_scores) + [1] * len(fake_scores)
    scores = [float(x) for x in real_scores] + [float(x) for x in fake_scores]
    correct = sum(score < threshold for score in real_scores)
    correct += sum(score >= threshold for score in fake_scores)
    return {
        "n_real": len(real_scores),
        "n_fake": len(fake_scores),
        "fake_prevalence": len(fake_scores) / (len(fake_scores) + len(real_scores)),
        "ap_raw_prevalence": average_precision(labels, scores),
        "ap_balanced_prior_50_50": balanced_prior_average_precision(real_scores, fake_scores),
        "auroc": binary_auroc(real_scores, fake_scores),
        "acc_at_0_5": correct / (len(real_scores) + len(fake_scores)),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate percentile from no bootstrap replicates")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_stratified_bootstrap(
    *,
    real_ids: Sequence[str],
    real_scores: Sequence[float],
    fake_scores_by_family: Mapping[str, Sequence[float]],
    families: Sequence[str] = PRIMARY_HOLDOUT_FAMILIES,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Paired bootstrap: share each real-ID resample across families, sample fakes separately.

    The same sampled index vector is used for the real examples in every family
    on each replicate. Each family's fake indices are independently drawn with
    replacement within that generator. Point estimates and percentile CIs are
    returned for raw AP, 50:50-prior AP, AUROC and ACC; the macro primary is the
    equal-weight mean of 50:50-prior AP for ``families``.
    """
    if len(real_ids) != len(real_scores) or len(set(real_ids)) != len(real_ids):
        raise ValueError("real IDs must be unique and align one-to-one with real scores")
    if not real_ids or replicates < 100:
        raise ValueError("need real records and at least 100 bootstrap replicates")
    if not families or any(family not in fake_scores_by_family for family in families):
        raise ValueError("all requested bootstrap families need fake-score vectors")
    for family in families:
        _validate_scores(real_scores, fake_scores_by_family[family])

    point_by_family = {
        family: _family_metrics(real_scores, fake_scores_by_family[family])
        for family in fake_scores_by_family
    }
    point_macro = sum(
        float(point_by_family[family]["ap_balanced_prior_50_50"]) for family in families
    ) / len(families)
    keys = ("ap_raw_prevalence", "ap_balanced_prior_50_50", "auroc", "acc_at_0_5")
    samples: dict[str, dict[str, list[float]]] = {
        family: {key: [] for key in keys} for family in fake_scores_by_family
    }
    macro_samples: list[float] = []
    rng = random.Random(seed)
    n_real = len(real_scores)
    for _ in range(replicates):
        real_indices = [rng.randrange(n_real) for _ in range(n_real)]
        sampled_real = [float(real_scores[i]) for i in real_indices]
        macro_value = 0.0
        for family, fake_scores in fake_scores_by_family.items():
            n_fake = len(fake_scores)
            fake_indices = [rng.randrange(n_fake) for _ in range(n_fake)]
            sampled_fake = [float(fake_scores[i]) for i in fake_indices]
            metrics = _family_metrics(sampled_real, sampled_fake)
            for key in keys:
                samples[family][key].append(float(metrics[key]))
            if family in families:
                macro_value += float(metrics["ap_balanced_prior_50_50"])
        macro_samples.append(macro_value / len(families))

    def ci(values: Sequence[float]) -> list[float]:
        return [_percentile(values, 0.025), _percentile(values, 0.975)]

    return {
        "replicates": replicates,
        "seed": seed,
        "ci_method": "paired stratified percentile bootstrap, 95%",
        "pairing": "shared real IDs across generator families; fake IDs resampled independently by family",
        "point_macro_ap_balanced_prior_50_50": point_macro,
        "macro_ap_balanced_prior_50_50_ci95": ci(macro_samples),
        "point_by_family": point_by_family,
        "ci95_by_family": {
            family: {key: ci(values) for key, values in family_samples.items()}
            for family, family_samples in samples.items()
        },
        "primary_families": list(families),
    }
