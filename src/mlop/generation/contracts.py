"""Pure contract-frame builders.

Each function takes a single calibration row and returns the unpriced feature
frame for that date. Pricing and IO live elsewhere, so these are cheap to test
without QuantLib or a filesystem.
"""

from itertools import product

import numpy as np
import pandas as pd

from ..spec import (
    ASIAN,
    ASIAN_COLUMNS,
    BARRIER,
    BARRIER_COLUMNS,
    HESTON_PARAMETERS,
    AsianSpec,
    BarrierSpec,
)


def _asian_block(spot, strikes, maturity, n_fixings, frequency, spec, row):
    return pd.DataFrame(
        product(
            [spot], strikes, [maturity], [n_fixings], [frequency],
            [spec.past_fixings], spec.averaging_types, spec.w,
            [row['risk_free_rate']], [row['dividend_rate']], [row['calculation_date']],
            *([row[p]] for p in HESTON_PARAMETERS),
        ),
        columns=ASIAN_COLUMNS,
    )


def asian_contracts(row: pd.Series, spec: AsianSpec = ASIAN) -> pd.DataFrame:
    """Build the Asian option grid for one calibration row.

    For the i-th maturity, fixing frequencies are taken from the first ``i + 1``
    entries of the frequency tuple, so longer maturities admit coarser averaging.
    """
    spot = row['spot_price']
    strikes = spec.strikes(spot)
    blocks = []

    for i, maturity in enumerate(spec.maturities):
        for frequency in spec.fixing_frequencies[:i + 1]:
            blocks.append(
                _asian_block(spot, strikes, maturity, maturity / frequency, frequency, spec, row)
            )

    if spec.include_single_fixing_block:
        for frequency in spec.fixing_frequencies:
            blocks.append(
                _asian_block(spot, strikes, frequency, 1, frequency, spec, row)
            )

    return pd.concat(blocks, ignore_index=True)


def barrier_contracts(row: pd.Series, spec: BarrierSpec = BARRIER) -> pd.DataFrame:
    """Build the barrier option grid for one calibration row."""
    spot = row['spot_price']
    strikes = spec.strikes(spot)

    sides = [
        pd.DataFrame(
            product(
                [spot], strikes, spec.barriers(spot, updown),
                spec.maturities, [updown], spec.outin, spec.w,
            ),
            columns=BARRIER_COLUMNS,
        )
        for updown in ('Down', 'Up')
    ]

    features = pd.concat(sides, ignore_index=True)
    features['barrier_type_name'] = features['updown'] + features['outin']
    features['rebate'] = spec.rebate
    features['risk_free_rate'] = row['risk_free_rate']
    features['dividend_rate'] = row['dividend_rate']

    heston = pd.Series(row[list(HESTON_PARAMETERS)]).astype(float)
    features[heston.index] = np.tile(heston, (features.shape[0], 1))
    features['calculation_date'] = row['calculation_date']

    return features
