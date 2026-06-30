import pandas as pd

from attacks.attacks.spike_suppression import apply_spike_suppression
from attacks.config import SUPPRESSION_FACTOR


def baseline_dataset(sensor_id=201, rows=24, values=None):
    if values is None:
        values = [float(i) for i in range(rows)]

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
            "value": values,
        }
    )


def spike_suppression_campaign(sensor_id=201, attack_id=1):
    return pd.DataFrame(
        [
            {
                "random_seed": 42,
                "attack_id": attack_id,
                "country": "ET",
                "location_id": 101,
                "sensor_id": sensor_id,
                "attack_type": "Spike Suppression",
            }
        ]
    )


def test_apply_spike_suppression_initializes_attack_columns():
    attacked = apply_spike_suppression(
        baseline_dataset(),
        spike_suppression_campaign(),
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
    assert attacked["original_value"].tolist() == [float(i) for i in range(24)]


def test_apply_spike_suppression_marks_only_window_spikes():
    attacked = apply_spike_suppression(
        baseline_dataset(rows=24),
        spike_suppression_campaign(attack_id=7),
    )

    spike_rows = attacked[attacked["attack_type"] == "Spike Suppression"]

    assert len(spike_rows) == 3
    assert spike_rows["attack_active"].all()
    assert spike_rows["attack_id"].eq(7).all()
    assert spike_rows["attack_strength"].eq(SUPPRESSION_FACTOR).all()
    assert spike_rows["attack_start"].nunique() == 1
    assert spike_rows["attack_end"].nunique() == 1


def test_apply_spike_suppression_multiplies_spikes_by_factor():
    attacked = apply_spike_suppression(
        baseline_dataset(rows=24),
        spike_suppression_campaign(),
    )

    spike_rows = attacked[attacked["attack_type"] == "Spike Suppression"]
    expected = spike_rows["original_value"] * SUPPRESSION_FACTOR

    assert spike_rows["value"].equals(expected)
    assert spike_rows["attacked_value"].equals(spike_rows["value"])
    assert spike_rows["original_value"].tolist() == [21.0, 22.0, 23.0]


def test_apply_spike_suppression_leaves_non_spikes_unchanged():
    attacked = apply_spike_suppression(
        baseline_dataset(rows=24),
        spike_suppression_campaign(),
    )

    unchanged = attacked[attacked["attack_type"] == "None"]

    assert len(unchanged) == 21
    assert unchanged["value"].equals(unchanged["original_value"])
    assert unchanged["attack_active"].eq(False).all()


def test_apply_spike_suppression_ignores_other_attack_types():
    campaign = spike_suppression_campaign()
    campaign.loc[0, "attack_type"] = "Random Stealth"

    attacked = apply_spike_suppression(
        baseline_dataset(rows=24),
        campaign,
    )

    assert attacked["value"].equals(attacked["original_value"])
    assert attacked["attack_id"].eq(-1).all()
    assert attacked["attack_type"].eq("None").all()
    assert attacked["attack_active"].eq(False).all()
    assert attacked["attack_strength"].eq(0.0).all()


def test_apply_spike_suppression_only_targets_matching_sensor():
    df = pd.concat(
        [
            baseline_dataset(sensor_id=201, rows=24),
            baseline_dataset(sensor_id=202, rows=24).assign(
                observation_id=range(25, 49),
            ),
        ],
        ignore_index=True,
    )

    attacked = apply_spike_suppression(
        df,
        spike_suppression_campaign(sensor_id=201),
    )

    targeted = attacked[attacked["sensor_id"] == 201]
    untouched = attacked[attacked["sensor_id"] == 202]

    assert (targeted["attack_type"] == "Spike Suppression").sum() == 3
    assert untouched["value"].equals(untouched["original_value"])
    assert untouched["attack_type"].eq("None").all()


def test_apply_spike_suppression_is_deterministic_for_same_input():
    df = baseline_dataset(rows=24)
    campaign = spike_suppression_campaign()

    first = apply_spike_suppression(df, campaign)
    second = apply_spike_suppression(df, campaign)

    pd.testing.assert_frame_equal(first, second)
