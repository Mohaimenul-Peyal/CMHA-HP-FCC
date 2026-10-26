"""Cross-check the R-suite manifest against the original published sequences."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_pdf", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("fcc_r1_r3_utlc_benchmarks.csv"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    extracted = subprocess.check_output([
        "pdftotext", "-f", "22", "-l", "22", "-layout", str(args.original_pdf), "-"
    ]).decode("utf-8")
    table = extracted.split("Table 1 Example runtimes", 1)[1]
    chunks = re.findall(r"^\s*([HP]{50})(?=\s|$)", table, re.MULTILINE)
    if len(chunks) != 12:
        raise ValueError(f"Expected 12 printed 50-residue lines, got {len(chunks)}")
    sequences = ["".join(chunks[i:i + 4]) for i in range(0, 12, 4)]
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    checks = []
    for i, (row, seq) in enumerate(zip(rows, sequences), 1):
        assert row["benchmark"] == f"R{i}FCC"
        assert seq == row["hp_sequence"], row["benchmark"]
        assert len(seq) == 200 and seq.count("H") == 100
        checks.append({"benchmark": row["benchmark"], "length": len(seq),
                       "H_count": seq.count("H"), "exact_sequence_match": True,
                       "sequence_sha256": hashlib.sha256(seq.encode("ascii")).hexdigest()})
    report = {
        "original_source_doi": "10.1007/s10601-006-6848-8",
        "original_source_table": "Table 1, printed page 26, PDF page 22",
        "original_pdf_sha256": hashlib.sha256(args.original_pdf.read_bytes()).hexdigest(),
        "author_supplement_url": "https://bioinformatics.bc.edu/clotelab/FCCproteinStructure/",
        "author_supplement_section": "Instances R, visually cross-checked in the web page",
        "target_source_doi": "10.1103/PhysRevE.90.042715",
        "target_source_table": "Table IV, page 042715-6 (Native E. and IWL columns)",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "checks": checks,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
