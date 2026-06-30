import pandas as pd

from attacks.attacks.constant_bias import apply_constant_bias
from attacks.config import BIAS_MAX, BIAS_MIN


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
            "attack_id": [-1] * rows,
            "attack_type": ["None"] * rows,
            "attack_active": [False] * rows,
            "attack_strength": [0.0] * rows,
        }
    )


def constant_bias_campaign(sensor_id=201, attack_id=1):
    return pd.DataFrame(
        [
            {
                "random_seed": 42,
                "attack_id": attack_id,
                "country": "ET",
                "location_id": 101,
                "sensor_id": sensor_id,
                "attack_type": "Constant Bias",
            }
        ]
    )


def test_apply_constant_bias_adds_original_and_attacked_value_columns():
    attacked = apply_constant_bias(
        baseline_dataset(),
        constant_bias_campaign(),
    )

    assert "original_value" in attacked.columns
    assert "attacked_value" in attacked.columns
    assert attacked["original_value"].eq(10.0).all()
    assert attacked["attacked_value"].equals(attacked["value"])


def test_apply_constant_bias_marks_attacked_rows():
    attacked = apply_constant_bias(
        baseline_dataset(rows=30),
        constant_bias_campaign(attack_id=7),
    )

    bias_rows = attacked[attacked["attack_type"] == "Constant Bias"]

    assert len(bias_rows) == 24
    assert bias_rows["attack_active"].all()
    assert bias_rows["attack_id"].eq(7).all()
    assert bias_rows["attack_strength"].between(BIAS_MIN, BIAS_MAX).all()
    assert (bias_rows["value"] > bias_rows["original_value"]).all()


def test_apply_constant_bias_uses_constant_strength_within_attack_window():
    attacked = apply_constant_bias(
        baseline_dataset(rows=30),
        constant_bias_campaign(),
    )

    bias_rows = attacked[attacked["attack_type"] == "Constant Bias"]
    deltas = bias_rows["value"] - bias_rows["original_value"]

    assert deltas.nunique() == 1
    assert bias_rows["attack_strength"].nunique() == 1
    assert deltas.iloc[0] == bias_rows["attack_strength"].iloc[0]


def test_apply_constant_bias_does_not_modify_non_constant_bias_campaigns():
    campaign = constant_bias_campaign()
    campaign.loc[0, "attack_type"] = "Gradual Drift"

    attacked = apply_constant_bias(
        baseline_dataset(rows=30),
        campaign,
    )

    assert attacked["value"].eq(10.0).all()
    assert attacked["attack_id"].eq(-1).all()
    assert attacked["attack_type"].eq("None").all()
    assert attacked["attack_active"].eq(False).all()
    assert attacked["attack_strength"].eq(0.0).all()


def test_apply_constant_bias_only_targets_matching_sensor():
    df = pd.concat(
        [
            baseline_dataset(sensor_id=201, rows=30),
            baseline_dataset(sensor_id=202, rows=30).assign(
                observation_id=range(31, 61),
            ),
        ],
        ignore_index=True,
    )

    attacked = apply_constant_bias(
        df,
        constant_bias_campaign(sensor_id=201),
    )

    targeted = attacked[attacked["sensor_id"] == 201]
    untouched = attacked[attacked["sensor_id"] == 202]

    assert (targeted["attack_type"] == "Constant Bias").sum() == 24
    assert untouched["value"].eq(10.0).all()
    assert untouched["attack_type"].eq("None").all()


def test_apply_constant_bias_is_deterministic_for_same_input():
    df = baseline_dataset(rows=30)
    campaign = constant_bias_campaign()

    first = apply_constant_bias(df, campaign)
    second = apply_constant_bias(df, campaign)

    pd.testing.assert_frame_equal(first, second)
