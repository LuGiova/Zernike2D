#!/usr/bin/env python3
"""Split dataset2 complex PDB files into per-protein PDBs and a summary CSV."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split all PDB files in a tar.gz or zip archive into one PDB per protein and store them "
            "in a zip together with a CSV summary."
        )
    )
    parser.add_argument("archive", help="Path to the input archive (.tar.gz or .zip)")
    parser.add_argument("-o", "--output", required=True, help="Output zip path")
    parser.add_argument("-j", "--workers", type=int, default=1, help="Number of worker processes to use")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
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
    raw_segments: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            current.append(line.rstrip("\n"))
            continue
        if line.startswith("TER"):
            if current:
                raw_segments.append(current)
                current = []
            continue
        if line.startswith(("ENDMDL", "END")):
            break

    if current:
        raw_segments.append(current)

    if not raw_segments:
        raise ValueError("No ATOM/HETATM segments found")

    # Merge adjacent TER-separated segments that continue the same chain,
    # preserving the internal TER marker in the merged output block.
    blocks: list[list[str]] = []
    block_chain_labels: list[str] = []
    for segment in raw_segments:
        segment_chain = choose_chain_label(segment)
        if blocks and block_chain_labels[-1] == segment_chain:
            blocks[-1].append("TER")
            blocks[-1].extend(segment)
        else:
            blocks.append(segment[:])
            block_chain_labels.append(segment_chain)

    if len(blocks) != 2:
        raise ValueError(
            f"Expected exactly 2 protein blocks after chain-aware TER merge, found {len(blocks)} "
            f"(raw segments: {len(raw_segments)})"
        )

    return blocks


def chain_id_from_atom_line(line: str) -> str:
    if len(line) > 21:
        chain_id = line[21].strip()
        if chain_id:
            return chain_id
    return "_"


def choose_chain_label(block: list[str]) -> str:
    chain_ids = [chain_id_from_atom_line(line) for line in block if line.startswith(("ATOM", "HETATM"))]
    if not chain_ids:
        return "_"
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
    # This function will be run in the main process and will dispatch per-member
    # work to worker processes when requested.
    raise NotImplementedError("process_archive should be called with workers via process_archive_with_workers")


@contextmanager
def _suppress_worker_output(enabled: bool):
    if not enabled:
        yield
        return
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def _read_member_text(archive_path: Path, member_name: str) -> str:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            with zf.open(member_name) as handle:
                return handle.read().decode("utf-8", errors="replace")

    with tarfile.open(archive_path, "r:*") as tar:
        member = tar.getmember(member_name)
        extracted = tar.extractfile(member)
        if extracted is None:
            raise ValueError(f"Could not extract {member_name}")
        return extracted.read().decode("utf-8", errors="replace")


def _list_archive_members(archive_path: Path) -> list[str]:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            members = [info.filename for info in zf.infolist() if not info.is_dir() and info.filename.lower().endswith(".pdb")]
    else:
        with tarfile.open(archive_path, "r:*") as tar:
            members = [m.name for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".pdb")]

    if not members:
        raise ValueError(f"No PDB files found in {archive_path}")

    return sorted(members)


def _run_member_task(member_name: str, archive_path: str, verbose: bool):
    """Worker: process one PDB member inside a tar or zip archive."""
    try:
        with _suppress_worker_output(not verbose):
            text = _read_member_text(Path(archive_path), member_name)
            lines = text.splitlines()

            complex_name, fallback_decoy = parse_pdb_name(member_name)
            decoy, contact, score, rmsd = parse_remarks(lines)
            if not decoy:
                decoy = fallback_decoy

            blocks = split_blocks(lines)
            pdb_outs: list[tuple[str, str]] = []
            output_names: list[str] = []
            for block in blocks:
                chain_label = choose_chain_label(block)
                output_name = f"{complex_name}-{decoy}_{chain_label}.pdb"
                pdb_outs.append((output_name, build_clean_pdb(block)))
                output_names.append(output_name)

            summary_row = {
                "complex_name": complex_name,
                "decoy": decoy,
                "contact": contact,
                "score": score,
                "rmsd": rmsd,
                "protein_1_pdb": output_names[0],
                "protein_2_pdb": output_names[1],
            }

        return {"ok": True, "member": member_name, "pdb_outputs": pdb_outs, "summary_row": summary_row}
    except Exception as exc:
        return {"ok": False, "member": member_name, "error": repr(exc)}


def process_archive_with_workers(archive_path: Path, workers: int = 1, verbose: bool = False) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    pdb_outputs: list[tuple[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    sorted_members = _list_archive_members(archive_path)

    bar = tqdm.tqdm(total=len(sorted_members), desc="Splitting complexes", unit="pdb", disable=verbose)

    bar = tqdm.tqdm(total=len(sorted_members), desc="Splitting complexes", unit="pdb", disable=verbose)

    if workers == 1:
        for member_name in sorted_members:
            result = _run_member_task(member_name, str(archive_path), verbose)
            if result.get("ok"):
                pdb_outputs.extend(result["pdb_outputs"])
                summary_rows.append(result["summary_row"])
            else:
                raise RuntimeError(f"Failed processing {member_name}: {result.get('error')}")
            bar.update(1)
        bar.close()
        return pdb_outputs, summary_rows

    try:
        try:
            executor = ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1)
        except TypeError:
            executor = ProcessPoolExecutor(max_workers=workers)

        futures = {executor.submit(_run_member_task, member_name, str(archive_path), verbose): member_name for member_name in sorted_members}
        try:
            for future in as_completed(futures):
                res = future.result()
                member_name = futures[future]
                if res.get("ok"):
                    pdb_outputs.extend(res["pdb_outputs"])
                    summary_rows.append(res["summary_row"])
                else:
                    raise RuntimeError(f"Failed processing {member_name}: {res.get('error')}")
                bar.update(1)
        finally:
            for f in futures:
                f.cancel()
            executor.shutdown(wait=True)
    finally:
        bar.close()

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

    pdb_outputs, summary_rows = process_archive_with_workers(archive_path, workers=getattr(args, 'workers', 1), verbose=getattr(args, 'verbose', False))
    write_zip(output_zip, pdb_outputs, summary_rows)


if __name__ == "__main__":
    main()