"""Contract grid specifications.

These constants are the domain spec recovered from the original generation
scripts in ``testing/historical_contracts_generation``. They define which
contracts are priced for each calibration date; everything else is plumbing.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AsianSpec:
    """Grid for arithmetic/geometric Asian options."""

    strike_lower: float = 0.5
    strike_upper: float = 1.5
    n_strikes: int = 5
    #: truncate strikes to whole numbers (the original used ``int(s * ratio)``)
    integer_strikes: bool = True
    fixing_frequencies: tuple[int, ...] = (1, 7, 28, 84, 168, 336)
    maturities: tuple[int, ...] = (7, 28, 84, 168, 336)
    averaging_types: tuple[str, ...] = ('geometric', 'arithmetic')
    w: tuple[str, ...] = ('call', 'put')
    past_fixings: int = 0
    #: emit an extra block where days_to_maturity == fixing_frequency, n_fixings == 1
    include_single_fixing_block: bool = True

    def strikes(self, spot: float) -> np.ndarray:
        lower, upper = spot * self.strike_lower, spot * self.strike_upper
        if self.integer_strikes:
            lower, upper = int(lower), int(upper)
        return np.linspace(lower, upper, self.n_strikes)


#: Short-dated variant: wider strike grid, no single-fixing block.
SHORT_TERM_ASIAN = AsianSpec(
    strike_lower=0.7,
    strike_upper=1.3,
    n_strikes=9,
    integer_strikes=False,
    fixing_frequencies=(7, 28, 84),
    maturities=(7, 28, 84),
    include_single_fixing_block=False,
)

ASIAN = AsianSpec()


@dataclass(frozen=True)
class BarrierSpec:
    """Grid for up/down in/out barrier options."""

    strike_lower: float = 0.9
    strike_upper: float = 1.1
    n_strikes: int = 9
    maturities: tuple[int, ...] = (60, 90, 180, 360, 540, 720)
    outin: tuple[str, ...] = ('Out', 'In')
    w: tuple[str, ...] = ('call', 'put')
    rebate: float = 0.0
    n_barriers: int = 5
    #: down barriers span [spot * 0.5, spot * 0.99]; up barriers [spot * 1.01, spot * 1.5]
    down_barrier_range: tuple[float, float] = (0.5, 0.99)
    up_barrier_range: tuple[float, float] = (1.01, 1.5)

    def strikes(self, spot: float) -> np.ndarray:
        return np.linspace(spot * self.strike_lower, spot * self.strike_upper, self.n_strikes)

    def barriers(self, spot: float, updown: str) -> list[float]:
        lo, hi = self.down_barrier_range if updown == 'Down' else self.up_barrier_range
        return np.linspace(spot * lo, spot * hi, self.n_barriers).astype(float).tolist()


BARRIER = BarrierSpec()

HESTON_PARAMETERS = ('kappa', 'theta', 'rho', 'eta', 'v0')
