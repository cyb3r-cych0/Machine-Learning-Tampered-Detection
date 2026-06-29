"""
Attack simulation pipeline.
"""

from .campaign import create_campaign

from .attacks import ATTACK_REGISTRY


def run_attack_pipeline(df):

    campaign = create_campaign(df)

    attacked = df.copy()

    executed = []

    for attack_name, attack_function in ATTACK_REGISTRY.items():

        attacked = attack_function(

            attacked,

            campaign

        )

        executed.append(

            attack_name

        )

    return (

        attacked,

        campaign,

        executed

    )