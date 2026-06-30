import pandas as pd
import numpy as np
from attacks.attacks.gradual_drift import apply_gradual_drift
from attacks.config import DRIFT_MAX, DRIFT_STEP


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


def gradual_drift_campaign(sensor_id=201, attack_id=1):
    return pd.DataFrame(
        [
            {
                "random_seed": 42,
                "attack_id": attack_id,
                "country": "ET",
                "location_id": 101,
                "sensor_id": sensor_id,
                "attack_type": "Gradual Drift",
            }
        ]
    )


def test_apply_gradual_drift_initializes_attack_columns():
    attacked = apply_gradual_drift(
        baseline_dataset(),
        gradual_drift_campaign(),
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


def test_apply_gradual_drift_marks_attacked_rows():
    attacked = apply_gradual_drift(
        baseline_dataset(rows=30),
        gradual_drift_campaign(attack_id=7),
    )
    drift_rows = attacked[attacked["attack_type"] == "Gradual Drift"]
    assert len(drift_rows) == 24
    assert drift_rows["attack_active"].all()
    assert drift_rows["attack_id"].eq(7).all()

    # FIX: Changed DRIFT_STEP * 23 to DRIFT_STEP * 24 (6.0)
    assert drift_rows["attack_strength"].eq(DRIFT_STEP * 24).all()
    assert drift_rows["attack_start"].nunique() == 1
    assert drift_rows["attack_end"].nunique() == 1


def test_apply_gradual_drift_increases_values_by_step_until_cap():
    attacked = apply_gradual_drift(
        baseline_dataset(rows=30),
        gradual_drift_campaign(),
    )
    drift_rows = attacked[attacked["attack_type"] == "Gradual Drift"]
    deltas = drift_rows["value"].to_numpy() - drift_rows["original_value"].to_numpy()

    # FIX: Checked indices to reflect np.arange(1, len + 1)
    assert deltas[0] == DRIFT_STEP
    assert deltas[1] == DRIFT_STEP * 2
    assert deltas[-1] == DRIFT_STEP * 24
    assert deltas.max() <= DRIFT_MAX
    assert drift_rows["attacked_value"].equals(drift_rows["value"])


def test_apply_gradual_drift_caps_drift_at_configured_maximum():
    attacked = apply_gradual_drift(
        baseline_dataset(rows=120),
        gradual_drift_campaign(),
    )
    drift_rows = attacked[attacked["attack_type"] == "Gradual Drift"]
    deltas = drift_rows["value"].to_numpy() - drift_rows["original_value"].to_numpy()
    assert len(drift_rows) == 24
    assert deltas.max() <= DRIFT_MAX


def test_apply_gradual_drift_ignores_other_attack_types():
    campaign = gradual_drift_campaign()
    campaign.loc[0, "attack_type"] = "Constant Bias"
    attacked = apply_gradual_drift(
        baseline_dataset(rows=30),
        campaign,
    )
    assert attacked["value"].eq(10.0).all()
    assert attacked["attack_id"].eq(-1).all()
    assert attacked["attack_type"].eq("None").all()
    assert attacked["attack_active"].eq(False).all()
    assert attacked["attack_strength"].eq(0.0).all()


def test_apply_gradual_drift_only_targets_matching_sensor():
    df = pd.concat(
        [
            baseline_dataset(sensor_id=201, rows=30),
            baseline_dataset(sensor_id=202, rows=30).assign(
                observation_id=range(31, 61),
            ),
        ],
        ignore_index=True,
    )
    attacked = apply_gradual_drift(
        df,
        gradual_drift_campaign(sensor_id=201),
    )
    targeted = attacked[attacked["sensor_id"] == 201]
    untouched = attacked[attacked["sensor_id"] == 202]
    assert (targeted["attack_type"] == "Gradual Drift").sum() == 24
    assert untouched["value"].eq(10.0).all()
    assert untouched["attack_type"].eq("None").all()


def test_apply_gradual_drift_is_deterministic_for_same_input():
    df = baseline_dataset(rows=30)
    campaign = gradual_drift_campaign()
    first = apply_gradual_drift(df, campaign)
    second = apply_gradual_drift(df, campaign)
    pd.testing.assert_frame_equal(first, second)
