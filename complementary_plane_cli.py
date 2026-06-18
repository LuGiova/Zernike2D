#!/usr/bin/env python3
"""CLI builder for the complementary-plane workflow."""

import argparse


def build_cli_complementary_plane2():
    des = ("Given two surfaces to compare, compute the complementary plane between them using binding site matching")
    parser = argparse.ArgumentParser(description=des, add_help=False)
    req_grp = parser.add_argument_group(title="positional arguments")
    req_grp.add_argument("-sf1", "--surface1",
                         required=True,
                         help="surface1 file; full path of the surface1 csv file",
                         metavar="")
    req_grp.add_argument("-sf2", "--surface2",
                         required=True,
                         help="surface2 file; full path of the surface2 csv file",
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
                          help="string; custom output file name (without extension)",
                          metavar="")
    optional.add_argument("--verbose",
                          action='store_true',
                          help="boolean; enable verbose output")
    optional.add_argument("--csv",
                          action='store_true',
                          help="boolean; save detailed complementary CSV (default: False)")
    optional.add_argument("-p", "--plot",
                          action='store_true',
                          help="boolean; generate plots")
    optional.add_argument("-h", "--help",
                          action="help",
                          help="show this help message and exit")
    return parser.parse_args()