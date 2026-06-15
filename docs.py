#!/usr/bin/env python3
"""
Command-line interface builders for all scripts in the Zernike2D project.
"""

import argparse


def build_cli_molecular_surface():
    des = ("Given as input a `.pdb` file obtains a molecular surface using dms software")
    parser = argparse.ArgumentParser(description=des, add_help=False)
    req_grp = parser.add_argument_group(title="positional arguments")
    req_grp.add_argument("-pdb", "--pdb",
                         required=True,
                         help="pdb file; the pdb file name (ex: 1a1u_A.pdb)",
                         metavar="")
    req_grp.add_argument("-i", "--input",
                         required=True,
                         help="input path; folder with pdb files",
                         metavar="")
    req_grp.add_argument("-o", "--output",
                         required=True,
                         help="output path; destination folder of output files",
                         metavar="")
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-a', 
                          action='store_true', 
                          help="boolean; use all atom in .pdb file to calculate dms surface")
    optional.add_argument("-h", "--help",
                          action="help",
                          help="show this help message and exit")
    return parser.parse_args()


def build_cli_zernike_invariants():
    des = (f"Given a surface of which coordinates and versors of each point are defined, "
           f"calculate the 2D Zernike invariants")
    parser = argparse.ArgumentParser(description=des, add_help=False)
    req_grp = parser.add_argument_group(title="positional arguments")
    req_grp.add_argument("-sf", "--surface",
                         required=True,
                         help="surface file; the csv file name (ex: 1a1u_A.csv)",
                         metavar="")
    req_grp.add_argument("-i", "--input",
                         required=True,
                         help="input path; folder with surface file",
                         metavar="")
    req_grp.add_argument("-o", "--output",
                         required=True,
                         help="output path; destination folder of output files",
                         metavar="")
    req_grp.add_argument("-v", "--verso",
                         required=True,
                         help="verso; direction of the patch with respect to the z axis; 1 positive, -1 negative",
                         metavar="")
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("-h", "--help",
                          action="help",
                          help="show this help message and exit")
    return parser.parse_args()


def build_cli_binding_propensity():
    des = ("Given two surfaces to compare, compute the binding propensity")
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
    optional.add_argument("-h", "--help",
                          action="help",
                          help="show this help message and exit")
    return parser.parse_args()


def build_cli_binding_site():
    des = ("Given two surfaces to compare, compute the binding sites between them")
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
    optional.add_argument("-p", "--plot",
                          action='store_true',
                          help="boolean; generate 3D visualization plots")
    optional.add_argument("-h", "--help",
                          action="help",
                          help="show this help message and exit")
    return parser.parse_args()


def build_cli_complementary_plane():
    des = ("Given two surfaces to compare, compute the complementary plane between them")
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
    optional.add_argument("-s", "--sample-every",
                          type=int,
                          default=1,
                          help="int; sample every Nth point from the surface (default: 1)",
                          metavar="")
    optional.add_argument("--use-surface-normals",
                          action='store_true',
                          help="boolean; use surface normals in calculations")
    optional.add_argument("--output-name",
                          type=str,
                          help="string; custom output file name (without extension)",
                          metavar="")
    optional.add_argument("-p", "--plot",
                          action='store_true',
                          help="boolean; generate plots")
    optional.add_argument("-h", "--help",
                          action="help",
                          help="show this help message and exit")
    return parser.parse_args()


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
                          help="int; number of sample points (default: 100)",
                          metavar="")
    optional.add_argument("--output-name",
                          type=str,
                          help="string; custom output file name (without extension)",
                          metavar="")
    optional.add_argument("--verbose",
                          action='store_true',
                          help="boolean; enable verbose output")
    optional.add_argument("-p", "--plot",
                          action='store_true',
                          help="boolean; generate plots")
    optional.add_argument("-h", "--help",
                          action="help",
                          help="show this help message and exit")
    return parser.parse_args()


def build_cli_flatness():
    des = ("Given a surface of which coordinates and normals are defined, compute the flatness")
    parser = argparse.ArgumentParser(description=des, add_help=False)
    req_grp = parser.add_argument_group(title="positional arguments")
    req_grp.add_argument("-s", "--surface",
                         required=True,
                         help="surface file; the csv file name (ex: 1a1u_A.csv)",
                         metavar="")
    req_grp.add_argument("-i", "--input",
                         required=True,
                         help="input path; folder with surface file",
                         metavar="")
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("-h", "--help",
                          action="help",
                          help="show this help message and exit")
    return parser.parse_args()


def build_cli_gyration_radius():
    des = ("Calculate radius of gyration from a PDB structure")
    parser = argparse.ArgumentParser(description=des, add_help=False)
    req_grp = parser.add_argument_group(title="positional arguments")
    req_grp.add_argument("--pdb",
                         required=True,
                         help="pdb file; path to PDB file",
                         metavar="")
    req_grp.add_argument("-i", "--input",
                         required=True,
                         help="input path; folder with pdb files",
                         metavar="")
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("-h", "--help",
                          action="help",
                          help="show this help message and exit")
    return parser.parse_args()


def build_cli_interface_correlation():
    des = ("Analyze correlation between physical and Zernike distances on an interface")
    parser = argparse.ArgumentParser(description=des, add_help=False)
    req_grp = parser.add_argument_group(title="positional arguments")
    req_grp.add_argument("-i", "--input",
                         required=True,
                         help="input file; path to input CSV file",
                         metavar="")
    req_grp.add_argument("-o", "--output",
                         required=True,
                         help="output path; path to output figure file",
                         metavar="")
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("-r", "--radius",
                          type=float,
                          default=6.0,
                          help="float; smoothing radius in angstroms (default: 6.0)",
                          metavar="")
    optional.add_argument("--topo",
                          action='store_true',
                          help="boolean; generate a 2x2 kriging topographic figure")
    optional.add_argument("--save-csv",
                          action='store_true',
                          help="boolean; save smoothed data to CSV")
    optional.add_argument("-h", "--help",
                          action="help",
                          help="show this help message and exit")
    return parser.parse_args()


def build_cli_plot_complementary_plane():
    des = ("Given a complementary plane CSV, generate 2D scatter plots")
    parser = argparse.ArgumentParser(description=des, add_help=False)
    req_grp = parser.add_argument_group(title="positional arguments")
    req_grp.add_argument("-i", "--input",
                         required=True,
                         help="input file; path to complementary plane CSV file",
                         metavar="")
    req_grp.add_argument("-o", "--output",
                         required=True,
                         help="output path; destination folder for output image",
                         metavar="")
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("-h", "--help",
                          action="help",
                          help="show this help message and exit")
    return parser.parse_args()


if __name__ == '__main__':
    pass
