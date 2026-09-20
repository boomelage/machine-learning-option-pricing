"""Keyed, idempotent resumption.

The original scripts resumed by *counting* files in the output directory and
slicing the calibration frame by that offset::

    computed_outputs = len([f for f in os.listdir(output_dir) if f.endswith('.csv')])
    df = df.iloc[computed_outputs:]

That offset is positional, so an extra, missing or renamed file -- or any change
to the calibration set between runs -- silently shifts the window and yields
skipped or duplicated calculation dates with no error. Resumption here is keyed
on the dates actually present on disk, which makes it order-independent and safe
to re-run.
"""

import re
from pathlib import Path

import pandas as pd

#: ``2024-11-17_175055269926_573412 <tag> asian options.csv`` -> the date prefix.
#: ``%H%M%S%f`` contributes exactly 12 digits.
_STAMP = re.compile(r'^(\d{4}-\d{2}-\d{2}_\d{12})_')

_STAMP_FORMAT = '%Y-%m-%d_%H%M%S%f'


def output_filename(calculation_date, spot_price, suffix: str) -> str:
    """Build the canonical output filename for one calibration date."""
    stamp = pd.Timestamp(calculation_date).strftime(_STAMP_FORMAT)
    spot_tag = str(int(spot_price * 100)).replace('_', '')
    return f'{stamp}_{spot_tag} {suffix}.csv'


def completed_dates(output_dir: Path) -> set[pd.Timestamp]:
    """Return the calculation dates already written to ``output_dir``.

    Files that do not carry a parseable stamp are ignored rather than counted,
    so stray CSVs cannot shift the resume point.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return set()

    done = set()
    for path in output_dir.glob('*.csv'):
        match = _STAMP.match(path.name)
        if match:
            done.add(pd.to_datetime(match.group(1), format=_STAMP_FORMAT))
    return done


def pending(calibrations: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Rows of ``calibrations`` with no corresponding output file yet."""
    done = completed_dates(output_dir)
    if not done:
        return calibrations.copy()
    remaining = calibrations[~calibrations['calculation_date'].isin(done)]
    return remaining.reset_index(drop=True)
