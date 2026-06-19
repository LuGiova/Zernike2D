#!/usr/bin/env python3
"""CLI builder for the complementary-plane workflow."""

import argparse


def build_cli_complementary_plane2():
    des = (
        "Given two surfaces to compare, compute the complementary plane between them. "
        "It can run either on one pair of files or in batch mode on a directory/zip of PDB complexes."
    )
    parser = argparse.ArgumentParser(description=des, add_help=False)

    req_grp = parser.add_argument_group(title="input/output arguments")
    req_grp.add_argument("-sf1", "--surface1",
                         required=False,
                         help="single-complex mode: first surface/PDB/CSV file",
                         metavar="")
    req_grp.add_argument("-sf2", "--surface2",
                         required=False,
                         help="single-complex mode: second surface/PDB/CSV file",
                         metavar="")
    req_grp.add_argument("--batch-dir",
                         type=str,
                         help="batch mode: directory containing paired PDB files named COMPLEX_A.pdb, COMPLEX_B.pdb, etc.",
                         metavar="")
    req_grp.add_argument("--batch-zip",
                         type=str,
                         help="batch mode: zip archive containing paired PDB files named COMPLEX_A.pdb, COMPLEX_B.pdb, etc.",
                         metavar="")
    req_grp.add_argument("-o", "--output",
                         required=True,
                         help="output path; destination folder of output files",
                         metavar="")

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("-t", "--threshold",
                          type=float,
                          default=5.0,
                          help="float; distance threshold in angstroms (default: 5.0)",
                          metavar="")
    optional.add_argument("-n", "--points",
                          type=int,
                          default=100,
                          help="int; number of points per ring (default: 100)",
                          metavar="")
    optional.add_argument("--output-name",
                          type=str,
                          help="single-complex mode: custom output file name without extension",
                          metavar="")
    optional.add_argument("--batch-summary",
                          type=str,
                          default=None,
                          help="batch mode: path/name of the unique summary CSV. Default: OUTPUT/batch_summary.csv",
                          metavar="")
    optional.add_argument("--workers",
                          type=int,
                          default=1,
                          help="batch mode: number of worker processes. Use 1 for minimum RAM; >1 for parallel execution (default: 1)",
                          metavar="")
    optional.add_argument("--force",
                          action='store_true',
                          help="batch mode: ignore existing summary and recompute all complexes")
    optional.add_argument("--verbose",
                          action='store_true',
                          help="boolean; enable verbose output")
    optional.add_argument("--csv",
                          action='store_true',
                          help="boolean; save detailed complementary CSV for each complex (default: False)")
    optional.add_argument("-p", "--plot",
                          action='store_true',
                          help="boolean; generate plots")
    optional.add_argument("--sampling-strategy",
                          type=str,
                          choices=['default', 'angular_cells', 'kmeans'],
                          default='default',
                          help="string; sampling strategy: 'default' (current), 'angular_cells', or 'kmeans' (default: default)",
                          metavar="")
    optional.add_argument("-h", "--help",
                          action="help",
                          help="show this help message and exit")

    args = parser.parse_args()

    batch_inputs = [bool(args.batch_dir), bool(args.batch_zip)]
    is_batch = any(batch_inputs)

    if sum(batch_inputs) > 1:
        parser.error("Use only one batch input: either --batch-dir or --batch-zip.")

    if is_batch:
        if args.surface1 or args.surface2:
            parser.error("In batch mode do not pass --surface1/--surface2.")
        if args.output_name:
            parser.error("--output-name is only for single-complex mode; batch output names are based on complex_name.")
        if args.workers < 1:
            parser.error("--workers must be >= 1.")
    else:
        if not args.surface1 or not args.surface2:
            parser.error("Single-complex mode requires --surface1 and --surface2, or use --batch-dir/--batch-zip.")

    return args
