"""Generation drivers.

Loading calibrations is isolated here because it depends on ``model_settings``,
which carries global state; the contract builders and resume logic do not.
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .contracts import asian_contracts, barrier_contracts
from .resume import output_filename, pending
from .spec import ASIAN, BARRIER, SHORT_TERM_ASIAN


def load_spx_calibrations(root: Path) -> pd.DataFrame:
    """Load SPX Heston calibrations, newest first."""
    from model_settings import ms

    ms.find_root(Path(root).resolve())
    ms.collect_spx_calibrations()

    df = ms.spx_calibrations.copy()
    df['calculation_date'] = pd.to_datetime(df['calculation_date'], format='mixed')
    return df.sort_values('calculation_date', ascending=False).reset_index(drop=True)


def _generate(calibrations, output_dir, build, price, suffix, spec):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    todo = pending(calibrations, output_dir)
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc=suffix):
        features = build(row, spec)
        features['date'] = pd.Timestamp(row['calculation_date']).floor('D')
        features[price.column] = price(features)
        features.to_csv(
            output_dir / output_filename(row['calculation_date'], row['spot_price'], suffix),
            index=False,
        )
    return len(todo)


class _AsianPricer:
    column = 'asian_price'

    def __call__(self, features):
        from quantlib_pricers import asian_option_pricer

        return asian_option_pricer().df_asian_option_price(features)


class _BarrierPricer:
    column = 'barrier_price'

    def __call__(self, features):
        from quantlib_pricers import barrier_option_pricer

        return barrier_option_pricer().df_barrier_price(features)


def generate_asians(calibrations, output_dir, tag='SPX', short_term=False):
    """Price the Asian grid for every calibration date not yet on disk."""
    spec = SHORT_TERM_ASIAN if short_term else ASIAN
    suffix = f'{tag} short-term asian options' if short_term else f'{tag} asian options'
    return _generate(calibrations, output_dir, asian_contracts, _AsianPricer(), suffix, spec)


def generate_barriers(calibrations, output_dir, tag='SPX'):
    """Price the barrier grid for every calibration date not yet on disk."""
    return _generate(
        calibrations, output_dir, barrier_contracts, _BarrierPricer(),
        f'{tag} barrier options', BARRIER,
    )
