import pandas as pd

from attacks.campaign import ATTACK_TYPES, create_campaign
from attacks.config import RANDOM_SEED


def baseline_dataset(sensor_count=10):
    return pd.DataFrame(
        {
            "observation_id": range(1, sensor_count + 1),
            "country": ["ET"] * sensor_count,
            "location_id": list(range(100, 100 + sensor_count)),
            "sensor_id": list(range(200, 200 + sensor_count)),
            "timestamp_utc": pd.to_datetime(
                ["2026-01-01T00:00:00Z"] * sensor_count,
                utc=True,
            ),
            "value": [10.0] * sensor_count,
        }
    )


def test_create_campaign_returns_expected_columns():
    campaign = create_campaign(baseline_dataset(), attack_fraction=0.2)

    assert list(campaign.columns) == [
        "random_seed",
        "attack_id",
        "country",
        "location_id",
        "sensor_id",
        "attack_type",
    ]


def test_create_campaign_selects_requested_sensor_fraction():
    campaign = create_campaign(baseline_dataset(sensor_count=10), attack_fraction=0.2)

    assert len(campaign) == 2
    assert campaign["attack_id"].tolist() == [1, 2]
    assert campaign[["country", "location_id", "sensor_id"]].duplicated().sum() == 0


def test_create_campaign_selects_at_least_one_sensor():
    campaign = create_campaign(baseline_dataset(sensor_count=3), attack_fraction=0.0)

    assert len(campaign) == 1


def test_create_campaign_uses_allowed_attack_types_and_seed():
    campaign = create_campaign(baseline_dataset(), attack_fraction=0.5)

    assert set(campaign["attack_type"]).issubset(set(ATTACK_TYPES))
    assert campaign["random_seed"].eq(RANDOM_SEED).all()


def test_create_campaign_is_deterministic_for_same_input():
    df = baseline_dataset(sensor_count=10)

    first = create_campaign(df, attack_fraction=0.3)
    second = create_campaign(df, attack_fraction=0.3)

    pd.testing.assert_frame_equal(first, second)
