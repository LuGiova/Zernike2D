#!/usr/bin/env python3
"""Split dataset2 complex PDB files into per-protein PDBs and a summary CSV."""

from __future__ import annotations

import argparse
import csv
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split all PDB files in dataset2.tar.gz into one PDB per protein and store them in a zip "
            "together with a CSV summary."
        )
    )
    parser.add_argument("archive", help="Path to dataset2.tar.gz")
    parser.add_argument("-o", "--output", required=True, help="Output zip path")
    return parser.parse_args()


def parse_pdb_name(pdb_name: str) -> tuple[str, str]:
    stem = Path(pdb_name).stem
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid PDB name {pdb_name!r}: expected NOMECOMPLESSO_CATENE_NUMERODECOY.pdb")
    complex_name, _chain_spec, decoy = parts
    if not decoy.isdigit():
        raise ValueError(f"Invalid decoy number in {pdb_name!r}")
    return complex_name, decoy


def parse_remarks(lines: list[str]) -> tuple[str, str, str]:
    decoy = ""
    contact = ""
    score = ""
    rmsd = ""

    for line in lines:
        if line.startswith("REMARK Number:"):
            decoy = line.split(":", 1)[1].strip()
        elif line.startswith("REMARK Contact:"):
            contact = line.split(":", 1)[1].strip()
        elif line.startswith("REMARK Score:"):
            score = line.split(":", 1)[1].strip()
        elif line.startswith("REMARK RMSD:"):
            rmsd = line.split(":", 1)[1].strip()

    if not decoy:
        raise ValueError("Missing REMARK Number")

    return decoy, contact, score, rmsd


def split_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            current.append(line.rstrip("\n"))
            continue
        if line.startswith("TER"):
            if current:
                blocks.append(current)
                current = []
            continue
        if line.startswith(("ENDMDL", "END")):
            break

    if current:
        blocks.append(current)

    if len(blocks) != 2:
        raise ValueError(f"Expected exactly 2 protein blocks separated by TER, found {len(blocks)}")

    return blocks


def chain_id_from_atom_line(line: str) -> str:
    if len(line) > 21:
        chain_id = line[21].strip()
        if chain_id:
            return chain_id
    return "_"


def choose_chain_label(block: list[str]) -> str:
    chain_ids = [chain_id_from_atom_line(line) for line in block]
    counts = Counter(chain_ids)
    ordered = []
    seen = set()
    for chain_id in chain_ids:
        if chain_id not in seen:
            ordered.append(chain_id)
            seen.add(chain_id)

    return max(ordered, key=lambda chain_id: (counts[chain_id], -ordered.index(chain_id)))


def build_clean_pdb(block: list[str]) -> str:
    return "\n".join(block + ["TER", "END"]) + "\n"


def process_archive(archive_path: Path) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    pdb_outputs: list[tuple[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    with tarfile.open(archive_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".pdb")]
        if not members:
            raise ValueError(f"No PDB files found in {archive_path}")

        for member in tqdm.tqdm(sorted(members, key=lambda item: item.name), desc="Splitting complexes", unit="pdb"):
            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            text = extracted.read().decode("utf-8", errors="replace")
            lines = text.splitlines()

            complex_name, fallback_decoy = parse_pdb_name(member.name)
            decoy, contact, score, rmsd = parse_remarks(lines)
            if not decoy:
                decoy = fallback_decoy

            blocks = split_blocks(lines)
            output_names: list[str] = []

            for block in blocks:
                chain_label = choose_chain_label(block)
                output_name = f"{complex_name}-{decoy}_{chain_label}.pdb"
                pdb_outputs.append((output_name, build_clean_pdb(block)))
                output_names.append(output_name)

            summary_rows.append(
                {
                    "complex_name": complex_name,
                    "decoy": decoy,
                    "contact": contact,
                    "score": score,
                    "rmsd": rmsd,
                    "protein_1_pdb": output_names[0],
                    "protein_2_pdb": output_names[1],
                }
            )

    return pdb_outputs, summary_rows


def write_zip(output_zip: Path, pdb_outputs: list[tuple[str, str]], summary_rows: list[dict[str, str]]) -> None:
    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        summary_path = tmpdir_path / "dataset2_split_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "complex_name",
                    "decoy",
                    "contact",
                    "score",
                    "rmsd",
                    "protein_1_pdb",
                    "protein_2_pdb",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for output_name, content in pdb_outputs:
                zf.writestr(output_name, content)
            zf.write(summary_path, arcname=summary_path.name)


def main() -> None:
    args = parse_args()
    archive_path = Path(args.archive)
    output_zip = Path(args.output)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    pdb_outputs, summary_rows = process_archive(archive_path)
    write_zip(output_zip, pdb_outputs, summary_rows)


if __name__ == "__main__":
    main()