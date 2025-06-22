#!/usr/bin/env python3
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import click
import pandas as pd
from Bio import SeqIO
from loguru import logger as log
from pyfaidx import Fasta
from tqdm import tqdm


def run_command(command: str, shell: bool = False):
    """Runs an external command and handles errors robustly."""
    command_str = command if shell else " ".join(map(str, command))
    log.info(f"Command: {command_str}")
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            shell=shell,
        )
    except FileNotFoundError:
        log.error(
            f"Command not found: '{command[0]}'. Is it installed and in your PATH?"
        )
        raise
    except subprocess.CalledProcessError as e:
        log.error(f"Command '{command_str}' failed with exit code {e.returncode}:")
        log.error(e.stderr)
        raise


def parse_junction_file_to_bed(
    sj_file_path: Path, support_threshold: int, half_window: int
) -> Iterator[Tuple[str, str]]:
    """
    Parses a single STAR SJ.out.tab file and yields BED-formatted strings.
    This function replaces the previous awk logic.
    """
    with open(sj_file_path, "r") as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 7:
                continue

            if int(fields[6]) < support_threshold:
                continue

            strand_code = int(fields[3])
            if strand_code == 0:
                continue

            chrom = fields[0]
            start_pos = int(fields[1])
            end_pos = int(fields[2])
            strand = "+" if strand_code == 1 else "-"

            donor_pos, acceptor_pos = (
                (start_pos, end_pos) if strand == "+" else (end_pos, start_pos)
            )

            donor_start, donor_end = donor_pos - half_window, donor_pos + half_window
            if donor_start > 0:
                yield ("donor", f"{chrom}\t{donor_start}\t{donor_end}\t.\t0\t{strand}")

            acceptor_start, acceptor_end = (
                acceptor_pos - half_window,
                acceptor_pos + half_window,
            )
            if acceptor_start > 0:
                yield (
                    "acceptor",
                    f"{chrom}\t{acceptor_start}\t{acceptor_end}\t.\t0\t{strand}",
                )


def extract_junctions(
    star_output_dir: Path,
    tmp_dir: Path,
    support_threshold: int,
    window_size: int,
) -> Tuple[Path, Path]:
    """Extracts, filters, and saves splice junctions from all STAR output files."""
    donor_bed_path = tmp_dir / "donors.bed"
    acceptor_bed_path = tmp_dir / "acceptors.bed"
    half_window = window_size // 2

    star_sj_files = list(star_output_dir.glob("*SJ.out.tab"))
    if not star_sj_files:
        raise click.FileError(f"No *SJ.out.tab files found in '{star_output_dir}'")

    log.info(f"Processing {len(star_sj_files)} SJ.out.tab files...")

    with open(donor_bed_path, "w") as f_donor, open(
        acceptor_bed_path, "w"
    ) as f_acceptor:
        for sj_file in tqdm(
            star_sj_files,
            desc="Parsing junction files",
            unit="file",
            ascii=True,
            leave=False,
            ncols=100,
        ):
            for site_type, bed_line in parse_junction_file_to_bed(
                sj_file, support_threshold, half_window
            ):
                if site_type == "donor":
                    f_donor.write(bed_line + "\n")
                else:
                    f_acceptor.write(bed_line + "\n")

    log.info("Removing duplicate entries from BED files...")
    for bed_path in [donor_bed_path, acceptor_bed_path]:
        sorted_path = bed_path.with_suffix(".sorted.bed")
        run_command(f"sort -k1,1 -k2,2n {bed_path} | uniq > {sorted_path}", shell=True)
        sorted_path.rename(bed_path)

    return donor_bed_path, acceptor_bed_path


def parse_chromosome_from_fasta_header(header: str) -> int:
    """Parses chromosome number from a FASTA header for data splitting."""
    match = re.search(r"::(chr|CHR|Chr)?([0-9]+):", header)
    if match:
        return int(match.group(2))

    match = re.match(r"chr(\d+)_neg_", header)
    if match:
        return int(match.group(1))

    return 99


def assign_split(chr_num: int) -> str:
    """Assigns dataset split (train/validation/test) based on chromosome number."""
    if chr_num in [1, 2, 3]:
        return "train"
    elif chr_num == 4:
        return "validation"
    else:
        return "test"


def load_fasta_and_assign_splits(fasta_path: Path, label: int) -> List[Dict]:
    """Reads a FASTA file and returns a list of dictionaries with assigned splits."""
    records = []
    log.info(f"Loading FASTA records from {fasta_path.name}...")
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        chr_num = parse_chromosome_from_fasta_header(rec.id)
        records.append(
            {
                "name": rec.id,
                "sequence": str(rec.seq).upper(),
                "label": label,
                "chr_num": chr_num,
                "split": assign_split(chr_num),
            }
        )
    return records


def generate_negative_samples(
    reference_path: Path, num_negatives: int, seq_len: int, label: int, seed: int
) -> List[Dict]:
    """Generates negative samples by random sampling from the reference genome."""
    log.info(f"Generating {num_negatives} negative samples of length {seq_len}...")
    ref = Fasta(str(reference_path), sequence_always_upper=True)

    chroms_to_sample = [f"{i}" for i in range(1, 6)]

    negative_list = []
    random.seed(seed)

    with tqdm(
        total=num_negatives,
        desc="Generating negative samples",
        unit="sequence",
        ascii=True,
        leave=False,
        ncols=100,
    ) as pbar:
        while len(negative_list) < num_negatives:
            chrom_name = random.choice(chroms_to_sample)
            chrom_len = len(ref[chrom_name])

            if chrom_len < seq_len:
                continue

            start = random.randint(0, chrom_len - seq_len)
            seq = str(ref[chrom_name][start : start + seq_len].seq)

            if "N" in seq:
                continue

            chr_num = int(chrom_name)
            name = f"chr{chr_num}_neg_{len(negative_list)}"

            negative_list.append(
                {
                    "name": name,
                    "sequence": seq,
                    "label": label,
                    "chr_num": chr_num,
                    "split": assign_split(chr_num),
                }
            )
            pbar.update(1)

    return negative_list


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--star_output_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing STAR's SJ.out.tab files.",
)
@click.option(
    "--reference_fasta",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Reference genome FASTA file.",
)
@click.option(
    "--output_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to save output files.",
)
@click.option(
    "--label-donor", default=1, show_default=True, help="Integer label for donor sites."
)
@click.option(
    "--label-acceptor",
    default=2,
    show_default=True,
    help="Integer label for acceptor sites.",
)
@click.option(
    "--label-negative",
    default=0,
    show_default=True,
    help="Integer label for negative samples.",
)
@click.option(
    "--support-threshold",
    default=10,
    show_default=True,
    help="Minimum read support for a junction.",
)
@click.option(
    "--neg-factor",
    default=2.0,
    show_default=True,
    help="Ratio of negative samples to total positive samples.",
)
@click.option(
    "--seq-len",
    default=160,
    show_default=True,
    help="Length of sequences to extract/sample.",
)
@click.option(
    "--seed", default=42, show_default=True, help="Random seed for reproducibility."
)
@click.option(
    "--keep-intermediate", is_flag=True, help="Keep intermediate BED and FASTA files."
)
def main(
    star_output_dir,
    reference_fasta,
    output_dir,
    label_donor,
    label_acceptor,
    label_negative,
    support_threshold,
    neg_factor,
    seq_len,
    seed,
    keep_intermediate,
):
    # Configure Loguru for rich console logging
    log.info("Starting dataset creation pipeline...")
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "intermediate_files"
    tmp_dir.mkdir(exist_ok=True)

    try:
        donor_bed_path, acceptor_bed_path = extract_junctions(
            star_output_dir, tmp_dir, support_threshold, seq_len
        )
        log.success("Extracted and deduplicated splice junctions.")

        donor_fa_path = tmp_dir / "donors.fa"
        acceptor_fa_path = tmp_dir / "acceptors.fa"

        run_command(
            [
                "bedtools",
                "getfasta",
                "-fi",
                str(reference_fasta),
                "-bed",
                str(donor_bed_path),
                "-fo",
                str(donor_fa_path),
                "-s",
                "-name",
            ]
        )
        run_command(
            [
                "bedtools",
                "getfasta",
                "-fi",
                str(reference_fasta),
                "-bed",
                str(acceptor_bed_path),
                "-fo",
                str(acceptor_fa_path),
                "-s",
                "-name",
            ]
        )
        log.success("Created FASTA files for positive samples.")

        donor_records = load_fasta_and_assign_splits(donor_fa_path, label_donor)
        acceptor_records = load_fasta_and_assign_splits(
            acceptor_fa_path, label_acceptor
        )

        num_positives = len(donor_records) + len(acceptor_records)
        num_negatives = int(num_positives * neg_factor)

        negative_records = generate_negative_samples(
            reference_fasta, num_negatives, seq_len, label_negative, seed
        )

        all_records = donor_records + acceptor_records + negative_records
        df = pd.DataFrame(all_records)

        log.info("Splitting data and saving to CSV files...")
        for split_name in ["train", "validation", "test"]:
            df_split = df[df["split"] == split_name].drop(columns=["chr_num", "split"])
            df_split = df_split.sample(frac=1, random_state=seed).reset_index(drop=True)

            output_path = output_dir / f"{split_name}.csv"
            df_split.to_csv(output_path, index=False)
            log.info(f"Wrote {len(df_split)} records to {output_path}")

        # output dataest statistics
        # log.info(
        #    "\n===== Final Dataset Statistics =====\n"
        #    + "\n".join(
        #        [
        #            f"{s.capitalize()} Set\n"
        #            + f"Total records: {len(df[df['split'] == s])}\n"
        #            + df[df["split"] == s]["label"]
        #            .value_counts()
        #            .sort_index()
        #            .to_string()
        #            for s in ["train", "validation", "test"]
        #        ]
        #    )
        #    + "\n===================================="
        # )

    finally:
        if not keep_intermediate and tmp_dir.exists():
            log.info("Removing intermediate directory...")
            shutil.rmtree(tmp_dir)

    log.success("Pipeline finished successfully!")


if __name__ == "__main__":
    main()
