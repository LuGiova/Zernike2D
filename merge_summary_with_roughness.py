#!/usr/bin/env python3
"""Merge decoys summary CSV files with ring roughness values.

The merge key is the complex name in ``decoys_summary`` and the prefix before
the first underscore in ``decoys_roughness_summary``.
Only complexes present in both inputs are kept.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROUGHNESS_RING_COLUMNS = [f'roughness_ring{i}' for i in range(1, 11)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Merge a decoys_summary CSV with roughness ring columns from a decoys_roughness_summary CSV.'
    )
    parser.add_argument('decoys_summary_csv', type=Path, help='Path to the decoys_summary CSV file')
    parser.add_argument('decoys_roughness_summary_csv', type=Path, help='Path to the decoys_roughness_summary CSV file')
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        help='Output CSV path. Defaults to <decoys_summary_stem>_with_roughness.csv next to the summary file.',
    )
    return parser


def _extract_base_complex_name(value: object) -> str:
    text = '' if pd.isna(value) else str(value)
    return text.split('_', 1)[0]


def merge_summary_with_roughness(decoys_summary_csv: Path, decoys_roughness_summary_csv: Path) -> pd.DataFrame:
    summary_df = pd.read_csv(decoys_summary_csv)
    roughness_df = pd.read_csv(decoys_roughness_summary_csv)

    if 'complex_name' not in summary_df.columns:
        raise ValueError(f"Missing 'complex_name' column in {decoys_summary_csv}")
    if 'complex_name' not in roughness_df.columns:
        raise ValueError(f"Missing 'complex_name' column in {decoys_roughness_summary_csv}")

    missing_roughness_columns = [column for column in ROUGHNESS_RING_COLUMNS if column not in roughness_df.columns]
    if missing_roughness_columns:
        raise ValueError(
            f"Missing roughness columns in {decoys_roughness_summary_csv}: {', '.join(missing_roughness_columns)}"
        )

    summary_df = summary_df.copy()
    summary_df['_merge_key'] = summary_df['complex_name'].astype(str)

    roughness_df = roughness_df.copy()
    roughness_df['_merge_key'] = roughness_df['complex_name'].map(_extract_base_complex_name)
    roughness_df = roughness_df[['_merge_key'] + ROUGHNESS_RING_COLUMNS].drop_duplicates(subset=['_merge_key'])

    merged_df = summary_df.merge(roughness_df, on='_merge_key', how='inner')
    merged_df = merged_df.drop(columns=['_merge_key'])
    return merged_df


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = args.decoys_summary_csv.with_name(f'{args.decoys_summary_csv.stem}_with_roughness.csv')

    merged_df = merge_summary_with_roughness(args.decoys_summary_csv, args.decoys_roughness_summary_csv)
    merged_df.to_csv(output_path, index=False)
    print(f'Wrote {len(merged_df)} rows to {output_path}')


if __name__ == '__main__':
    main()