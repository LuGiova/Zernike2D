#!/usr/bin/env python3
"""
Confronta profili radiali g(r) native vs decoy da due CSV summary.

Assunzioni:
- Nel CSV native la colonna complex_name contiene il nome base del complesso, es. 1A0H.
- Nel CSV decoy la colonna complex_name contiene nomebase-NUMERODECOY, es. 1A0H-3.
- I CSV possono essere in ordine diverso.
- Non tutti i native devono avere un decoy: lo script usa solo l'intersezione.

Output:
- un solo file grafico, ad esempio ./output_files/correlations.pdf
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calcola e plotta Pearson e RMSE tra g(r) native e decoy in un unico file."
    )
    parser.add_argument("native_csv", help="CSV summary dei complessi native")
    parser.add_argument("decoy_csv", help="CSV summary dei complessi decoy")
    parser.add_argument(
        "output_plot",
        help="Percorso del file grafico finale, es. ./output_files/correlations.pdf"
    )
    parser.add_argument(
        "--curve",
        choices=["physical", "zernike", "both"],
        default="both",
        help="Quale g(r) analizzare: physical, zernike o both. Default: both."
    )
    parser.add_argument(
        "--complex-col",
        default="complex_name",
        help="Nome colonna con ID complesso. Default: complex_name."
    )
    parser.add_argument(
        "--min-valid-rings",
        type=int,
        default=5,
        help="Numero minimo di anelli validi per calcolare Pearson/RMSE. Default: 5."
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Numero di bin negli istogrammi. Default: 30."
    )
    parser.add_argument(
        "--decoy-suffix-regex",
        default=r"-\d+$",
        help=(
            "Regex da rimuovere dal complex_name decoy per ottenere il nome native. "
            "Default: '-\\d+$', cioè rimuove suffissi tipo -1, -23, -004."
        )
    )
    return parser.parse_args()


def read_csv_clean(path):
    df = pd.read_csv(
        path,
        na_values=["", " ", "NA", "NaN", "nan", "None", "null", "NULL"],
        keep_default_na=True,
    )
    df.columns = df.columns.str.strip()
    return df


def get_base_complex_name(name, suffix_regex):
    name = str(name).strip()
    return re.sub(suffix_regex, "", name)


def check_unique(df, col, label):
    duplicated = df[col].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, col].astype(str).unique()[:10]
        raise ValueError(
            f"Nel dataset {label} ci sono ID complesso duplicati dopo il parsing. "
            f"Esempi: {examples}. "
            "Se hai davvero piu decoys per native, devi scegliere quale usare oppure aggregarli."
        )


def ring_columns(prefix):
    return [f"{prefix}_ring{i}_mean" for i in range(1, 11)]


def validate_columns(df, cols, dataset_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Nel dataset {dataset_name} mancano queste colonne: {missing}"
        )


def pearson_corr(x, y):
    if len(x) < 2:
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def compute_pair_metrics(row, native_cols, decoy_cols, min_valid):
    x = row[native_cols].to_numpy(dtype=float)
    y = row[decoy_cols].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = int(valid.sum())

    if n_valid < min_valid:
        return pd.Series({
            "pearson": np.nan,
            "rmse": np.nan,
            "n_valid_rings": n_valid,
        })

    x = x[valid]
    y = y[valid]

    pearson = pearson_corr(x, y)
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))

    return pd.Series({
        "pearson": pearson,
        "rmse": rmse,
        "n_valid_rings": n_valid,
    })


def fisher_mean(corr_values):
    values = pd.Series(corr_values).dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    eps = 1e-12
    values = np.clip(values, -1 + eps, 1 - eps)
    return float(np.tanh(np.mean(np.arctanh(values))))


def describe_values(values, is_correlation=False):
    values = pd.Series(values).dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "variance": np.nan,
            "median": np.nan,
            "fisher_mean": np.nan,
        }

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "variance": float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "fisher_mean": fisher_mean(values) if is_correlation else np.nan,
    }


def stats_label(stats, include_fisher=True):
    lines = [
        f"n = {stats['n']}",
        f"media = {stats['mean']:.4g}",
        f"varianza = {stats['variance']:.4g}",
        f"mediana = {stats['median']:.4g}",
    ]
    if include_fisher:
        lines.append(f"media Fisher = {stats['fisher_mean']:.4g}")
    return "\n".join(lines)


def plot_hist(ax, values, stats, title, xlabel, bins, include_fisher=True):
    values = pd.Series(values).dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    ax.hist(values, bins=bins, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Numero di complessi")

    if len(values) > 0:
        ax.axvline(stats["mean"], linestyle="--", label="media")
        ax.axvline(stats["median"], linestyle=":", label="mediana")
        if include_fisher and np.isfinite(stats["fisher_mean"]):
            ax.axvline(stats["fisher_mean"], linestyle="-.", label="media Fisher")

    # Entry invisibile con le statistiche richieste in legenda.
    ax.plot([], [], " ", label=stats_label(stats, include_fisher=include_fisher))
    ax.legend(frameon=True)


def main():
    args = parse_args()

    native = read_csv_clean(args.native_csv)
    decoy = read_csv_clean(args.decoy_csv)

    if args.complex_col not in native.columns:
        raise ValueError(f"Colonna {args.complex_col!r} non trovata nel CSV native.")
    if args.complex_col not in decoy.columns:
        raise ValueError(f"Colonna {args.complex_col!r} non trovata nel CSV decoy.")

    native = native.copy()
    decoy = decoy.copy()

    native["base_complex_name"] = native[args.complex_col].astype(str).str.strip()
    decoy["base_complex_name"] = decoy[args.complex_col].apply(
        lambda x: get_base_complex_name(x, args.decoy_suffix_regex)
    )

    check_unique(native, "base_complex_name", "native")
    check_unique(decoy, "base_complex_name", "decoy")

    n_native_total = len(native)
    n_decoy_total = len(decoy)

    merged = native.merge(
        decoy,
        on="base_complex_name",
        how="inner",
        suffixes=("_native", "_decoy"),
    )

    n_matched = len(merged)
    native_matched = set(merged["base_complex_name"])
    native_all = set(native["base_complex_name"])
    decoy_all = set(decoy["base_complex_name"])

    unmatched_native = sorted(native_all - decoy_all)
    unmatched_decoy = sorted(decoy_all - native_all)

    print(f"Native totali: {n_native_total}")
    print(f"Decoy totali: {n_decoy_total}")
    print(f"Complessi matched native-decoy: {n_matched}")
    print(f"Native senza decoy: {len(unmatched_native)}")
    print(f"Decoy senza native: {len(unmatched_decoy)}")

    curves = ["physical", "zernike"] if args.curve == "both" else [args.curve]

    output_plot = Path(args.output_plot)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    plot_data = {}

    for curve in curves:
        native_cols_original = ring_columns(curve)
        decoy_cols_original = ring_columns(curve)

        validate_columns(native, native_cols_original, "native")
        validate_columns(decoy, decoy_cols_original, "decoy")

        native_cols = [f"{c}_native" for c in native_cols_original]
        decoy_cols = [f"{c}_decoy" for c in decoy_cols_original]

        pair_metrics = merged.apply(
            compute_pair_metrics,
            axis=1,
            native_cols=native_cols,
            decoy_cols=decoy_cols,
            min_valid=args.min_valid_rings,
        )

        pearson_stats = describe_values(pair_metrics["pearson"], is_correlation=True)
        rmse_stats = describe_values(pair_metrics["rmse"], is_correlation=False)
        plot_data[curve] = {
            "pearson": {
                "values": pair_metrics["pearson"],
                "stats": pearson_stats,
                "title": f"Pearson native-decoy ({curve})",
                "xlabel": "Pearson correlation",
                "include_fisher": True,
            },
            "rmse": {
                "values": pair_metrics["rmse"],
                "stats": rmse_stats,
                "title": f"RMSE native-decoy ({curve})",
                "xlabel": "RMSE",
                "include_fisher": False,
            },
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_specs = [
        (axes[0, 0], "physical", "pearson"),
        (axes[0, 1], "physical", "rmse"),
        (axes[1, 0], "zernike", "pearson"),
        (axes[1, 1], "zernike", "rmse"),
    ]

    for ax, curve, metric in plot_specs:
        if curve not in plot_data:
            ax.axis("off")
            continue
        spec = plot_data[curve][metric]
        plot_hist(
            ax,
            spec["values"],
            spec["stats"],
            title=spec["title"],
            xlabel=spec["xlabel"],
            bins=args.bins,
            include_fisher=spec["include_fisher"],
        )

    fig.suptitle("Confronto native vs decoy", y=1.02)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFile salvato:")
    print(f"- {output_plot}")


if __name__ == "__main__":
    main()
