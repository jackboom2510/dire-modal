import pytest

torch = pytest.importorskip("torch")

from dire_zero.experiments import (
    EXPERIMENT_TEST_GENERATORS,
    ROBUSTNESS_LEVELS,
    TABLE4_STEPS,
    TABLE5_REPRESENTATIONS,
    corrupt_image,
    image_variant_metadata,
    make_representation,
    validate_run_id,
    validate_steps,
)


def test_paper_experiment_grids_are_complete_and_explicit():
    assert TABLE4_STEPS == (5, 10, 20, 50)
    assert TABLE5_REPRESENTATIONS == ("REC", "RGB", "RGB_DIRE", "DIRE")
    assert [x["value"] for x in ROBUSTNESS_LEVELS[1:4]] == [1.0, 2.0, 3.0]
    assert [x["value"] for x in ROBUSTNESS_LEVELS[4:]] == [65, 30]
    assert EXPERIMENT_TEST_GENERATORS == ("adm", "ddpm", "iddpm", "pndm", "sdv2")


def test_table4_requires_all_paper_steps_without_duplicates():
    assert validate_steps([50, 20, 5, 10]) == TABLE4_STEPS
    with pytest.raises(ValueError, match="đủ S"):
        validate_steps([5, 10, 20])
    with pytest.raises(ValueError, match="trùng"):
        validate_steps([5, 10, 20, 20])


def test_table5_representation_shapes_ranges_and_channel_order():
    rgb = torch.zeros(3, 8, 8)
    rec = torch.ones(3, 8, 8) * -1
    dire = torch.ones(3, 8, 8) * 1.5
    assert torch.equal(make_representation(rgb, rec, dire, "RGB"), torch.full((3, 8, 8), 0.5))
    assert torch.equal(make_representation(rgb, rec, dire, "REC"), torch.zeros(3, 8, 8))
    assert torch.equal(make_representation(rgb, rec, dire, "DIRE"), dire)
    both = make_representation(rgb, rec, dire, "RGB & DIRE")
    assert both.shape == (6, 8, 8)
    assert torch.equal(both[:3], torch.full((3, 8, 8), 0.5))
    assert torch.equal(both[3:], dire)
    assert image_variant_metadata("RGB & DIRE")["channels"] == 6


def test_representation_rejects_bad_data():
    good = torch.zeros(3, 8, 8)
    with pytest.raises(ValueError, match="representation"):
        make_representation(good, good, good, "HSV")
    with pytest.raises(ValueError, match="cùng kích thước"):
        make_representation(good, torch.zeros(3, 4, 4), good, "DIRE")


def test_robustness_corruption_is_deterministic_and_does_not_mutate_source():
    source = torch.zeros(3, 16, 16)
    before = source.clone()
    clean = corrupt_image(source, ROBUSTNESS_LEVELS[0])
    blur = corrupt_image(source, ROBUSTNESS_LEVELS[1])
    jpeg = corrupt_image(source, ROBUSTNESS_LEVELS[4])
    assert torch.equal(clean, source)
    assert torch.equal(blur, corrupt_image(source, ROBUSTNESS_LEVELS[1]))
    assert torch.equal(jpeg, corrupt_image(source, ROBUSTNESS_LEVELS[4]))
    assert torch.equal(source, before)
    with pytest.raises(ValueError, match="sigma"):
        corrupt_image(source, {"kind": "gaussian_blur", "value": 4})


def test_run_id_is_safe_for_experiment_paths():
    assert validate_run_id("table5-2026.11") == "table5-2026.11"
    with pytest.raises(ValueError):
        validate_run_id("../overwrite")
