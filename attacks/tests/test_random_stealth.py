import pandas as pd

from attacks.attacks.random_stealth import apply_random_stealth
from attacks.config import STEALTH_STD


def baseline_dataset(sensor_id=201, rows=30):
    return pd.DataFrame(
        {
            "observation_id": range(1, rows + 1),
            "country": ["ET"] * rows,
            "location_id": [101] * rows,
            "sensor_id": [sensor_id] * rows,
            "timestamp_utc": pd.date_range(
                "2026-01-01T00:00:00Z",
                periods=rows,
                freq="h",
            ),
            "value": [10.0] * rows,
        }
    )


def random_stealth_campaign(sensor_id=201, attack_id=1):
    return pd.DataFrame(
        [
            {
                "random_seed": 42,
                "attack_id": attack_id,
                "country": "ET",
                "location_id": 101,
                "sensor_id": sensor_id,
                "attack_type": "Random Stealth",
            }
        ]
    )


def test_apply_random_stealth_initializes_attack_columns():
    attacked = apply_random_stealth(
        baseline_dataset(),
        random_stealth_campaign(),
    )

    assert {
        "original_value",
        "attacked_value",
        "attack_id",
        "attack_type",
        "attack_active",
        "attack_strength",
        "attack_start",
        "attack_end",
    }.issubset(attacked.columns)
    assert attacked["original_value"].eq(10.0).all()


def test_apply_random_stealth_marks_attacked_rows():
    attacked = apply_random_stealth(
        baseline_dataset(rows=30),
        random_stealth_campaign(attack_id=7),
    )

    stealth_rows = attacked[attacked["attack_type"] == "Random Stealth"]

    assert len(stealth_rows) == 24
    assert stealth_rows["attack_active"].all()
    assert stealth_rows["attack_id"].eq(7).all()
    assert stealth_rows["attack_strength"].gt(0).all()
    assert stealth_rows["attack_start"].nunique() == 1
    assert stealth_rows["attack_end"].nunique() == 1


def test_apply_random_stealth_applies_bounded_noise_metadata():
    attacked = apply_random_stealth(
        baseline_dataset(rows=24),
        random_stealth_campaign(),
    )

    stealth_rows = attacked[attacked["attack_type"] == "Random Stealth"]
    noise = stealth_rows["value"].to_numpy() - stealth_rows["original_value"].to_numpy()

    assert len(stealth_rows) == 24
    assert abs(noise.std() - stealth_rows["attack_strength"].iloc[0]) < 1e-12
    assert stealth_rows["attack_strength"].iloc[0] < STEALTH_STD * 2
    assert stealth_rows["attacked_value"].equals(stealth_rows["value"])


def test_apply_random_stealth_ignores_other_attack_types():
    campaign = random_stealth_campaign()
    campaign.loc[0, "attack_type"] = "Constant Bias"

    attacked = apply_random_stealth(
        baseline_dataset(rows=30),
        campaign,
    )

    assert attacked["value"].eq(10.0).all()
    assert attacked["attack_id"].eq(-1).all()
    assert attacked["attack_type"].eq("None").all()
    assert attacked["attack_active"].eq(False).all()
    assert attacked["attack_strength"].eq(0.0).all()


def test_apply_random_stealth_only_targets_matching_sensor():
    df = pd.concat(
        [
            baseline_dataset(sensor_id=201, rows=30),
            baseline_dataset(sensor_id=202, rows=30).assign(
                observation_id=range(31, 61),
            ),
        ],
        ignore_index=True,
    )

    attacked = apply_random_stealth(
        df,
        random_stealth_campaign(sensor_id=201),
    )

    targeted = attacked[attacked["sensor_id"] == 201]
    untouched = attacked[attacked["sensor_id"] == 202]

    assert (targeted["attack_type"] == "Random Stealth").sum() == 24
    assert untouched["value"].eq(10.0).all()
    assert untouched["attack_type"].eq("None").all()


def test_apply_random_stealth_is_deterministic_for_same_input():
    df = baseline_dataset(rows=30)
    campaign = random_stealth_campaign()

    first = apply_random_stealth(df, campaign)
    second = apply_random_stealth(df, campaign)

    pd.testing.assert_frame_equal(first, second)
