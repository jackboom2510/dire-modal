from __future__ import annotations

import pytest

from dire_modal.kaggle_celebahq_evaluation import (
    BOOTSTRAP_REPLICATES,
    PRIMARY_HOLDOUT_FAMILIES,
    balanced_prior_average_precision,
    binary_auroc,
    paired_stratified_bootstrap,
)


def test_balanced_prior_ap_and_auroc_are_perfect_for_separated_scores() -> None:
    real = [0.01, 0.2, 0.3]
    fake = [0.7, 0.9]
    assert balanced_prior_average_precision(real, fake) == pytest.approx(1.0)
    assert binary_auroc(real, fake) == pytest.approx(1.0)


def test_balanced_prior_ap_and_auroc_give_half_credit_for_all_ties() -> None:
    real = [0.5, 0.5]
    fake = [0.5, 0.5, 0.5]
    assert balanced_prior_average_precision(real, fake) == pytest.approx(0.5)
    assert binary_auroc(real, fake) == pytest.approx(0.5)


def test_balanced_prior_ap_is_invariant_to_empirical_fake_prevalence() -> None:
    real = [0.1, 0.2]
    fake_small = [0.3]
    fake_repeated = [0.3, 0.3, 0.3, 0.3]
    assert balanced_prior_average_precision(real, fake_small) == pytest.approx(
        balanced_prior_average_precision(real, fake_repeated)
    )


def test_paired_bootstrap_is_deterministic_and_reports_all_family_metrics() -> None:
    real_ids = [f"real/{i}.png" for i in range(8)]
    real_scores = [0.05, 0.15, 0.2, 0.35, 0.3, 0.4, 0.1, 0.25]
    fakes = {
        family: [0.6 + (i / 100) for i in range(8)]
        for family in PRIMARY_HOLDOUT_FAMILIES
    }
    fakes["sdv2"] = [0.55 + i / 100 for i in range(8)]
    first = paired_stratified_bootstrap(
        real_ids=real_ids, real_scores=real_scores, fake_scores_by_family=fakes,
        replicates=100, seed=3407,
    )
    second = paired_stratified_bootstrap(
        real_ids=real_ids, real_scores=real_scores, fake_scores_by_family=fakes,
        replicates=100, seed=3407,
    )
    assert first == second
    assert first["replicates"] == 100
    assert first["primary_families"] == list(PRIMARY_HOLDOUT_FAMILIES)
    assert first["point_macro_ap_balanced_prior_50_50"] == pytest.approx(1.0)
    assert set(first["ci95_by_family"]) == {*PRIMARY_HOLDOUT_FAMILIES, "sdv2"}
    assert len(first["macro_ap_balanced_prior_50_50_ci95"]) == 2
    assert BOOTSTRAP_REPLICATES == 2_000


def test_bootstrap_rejects_misaligned_ids_or_missing_generator_scores() -> None:
    with pytest.raises(ValueError, match="align one-to-one"):
        paired_stratified_bootstrap(
            real_ids=["duplicate", "duplicate"], real_scores=[0.1, 0.2],
            fake_scores_by_family={"dalle2": [0.8]}, families=("dalle2",), replicates=100,
        )
    with pytest.raises(ValueError, match="requested bootstrap families"):
        paired_stratified_bootstrap(
            real_ids=["r0"], real_scores=[0.1], fake_scores_by_family={},
            families=("dalle2",), replicates=100,
        )
