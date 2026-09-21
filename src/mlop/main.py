"""Command line entry point for contract generation.

Run it as a module, never as a file path::

    python -m mlop.main asian --dates 5

``python src/mlop/main.py`` puts ``src/mlop`` on ``sys.path`` instead of ``src``. The
subpackage then imports as a top-level ``generation``, so the ``from ..spec import``
inside it has nothing above it to reach and raises "attempted relative import beyond
top-level package". Running with ``-m`` imports ``mlop`` by name, which roots the
package where its relative imports expect it.
"""

import argparse
from pathlib import Path

from .generation import (
    asian_contracts,
    barrier_contracts,
    generate_asians,
    generate_barriers,
    load_spx_calibrations,
)
from .spec import ASIAN, BARRIER, SHORT_TERM_ASIAN

#: Default output root. Resolved from this file so the default holds wherever the
#: command is run from, matching ``generation.runner.CALIBRATIONS_CSV``.
GENERATED_DIR = Path(__file__).resolve().parents[2] / 'data' / 'generated'

#: One subdirectory of the output root per contract type. The two Asian grids are
#: separated as well: they differ in strikes, maturities and fixing frequencies, so a
#: single folder would hold two incompatible schemas under one name.
SUBDIRS = {
    'asian': 'asian',
    'asian_short_term': 'asian_short_term',
    'barrier': 'barrier',
}


def output_dir(root: Path, kind: str, short_term: bool = False) -> Path:
    """The directory a given contract type is written to, under ``root``."""
    key = 'asian_short_term' if kind == 'asian' and short_term else kind
    return Path(root) / SUBDIRS[key]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m mlop.main',
        description='Price contract grids for each Heston calibration date.',
    )
    parser.add_argument(
        'contracts', choices=('asian', 'barrier', 'both'),
        help='which grid to price',
    )
    parser.add_argument(
        '--dates', type=int, default=None, metavar='N',
        help='only the N most recent calibration dates (default: all of them)',
    )
    parser.add_argument(
        '--output', type=Path, default=GENERATED_DIR, metavar='DIR',
        help='output root; each contract type gets its own subfolder '
             f'({"/, ".join(SUBDIRS.values())}/) (default: {GENERATED_DIR})',
    )
    parser.add_argument(
        '--calibrations', type=Path, default=None, metavar='CSV',
        help='calibration file to read (default: the one under data/price_dynamics)',
    )
    parser.add_argument('--tag', default='SPX', help='underlying tag in output filenames')
    parser.add_argument(
        '--short-term', action='store_true',
        help='use the short-dated Asian grid (ignored for barriers)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='report the grid size per date without pricing anything',
    )
    return parser


def _dry_run(calibrations, contracts, root, short_term):
    """Report what a real run would price. Building a grid is cheap; pricing is not."""
    row = calibrations.iloc[0]
    for kind in contracts:
        if kind == 'asian':
            spec = SHORT_TERM_ASIAN if short_term else ASIAN
            width = len(asian_contracts(row, spec))
        else:
            width = len(barrier_contracts(row, BARRIER))
        print(f'{kind}: {width} contracts per date x {len(calibrations)} dates '
              f'= {width * len(calibrations)} rows '
              f'-> {output_dir(root, kind, short_term)}')


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    calibrations = load_spx_calibrations(args.calibrations)
    if args.dates is not None:
        calibrations = calibrations.head(args.dates)

    if calibrations.empty:
        print('no calibrations to price')
        return 1

    kinds = ('asian', 'barrier') if args.contracts == 'both' else (args.contracts,)
    newest = calibrations['calculation_date'].iloc[0].date()
    oldest = calibrations['calculation_date'].iloc[-1].date()
    print(f'{len(calibrations)} calibration dates, {oldest} to {newest}')

    if args.dry_run:
        _dry_run(calibrations, kinds, args.output, args.short_term)
        return 0

    # Each type writes to its own subfolder. The generators still filter on the
    # filename suffix when deciding which dates are done, so pointing two of them at
    # one directory stays safe -- the split is for legibility, not correctness.
    for kind in kinds:
        target = output_dir(args.output, kind, args.short_term)
        if kind == 'asian':
            written = generate_asians(
                calibrations, target, tag=args.tag, short_term=args.short_term)
        else:
            written = generate_barriers(calibrations, target, tag=args.tag)
        print(f'{kind}: priced {written} date(s) into {target}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
