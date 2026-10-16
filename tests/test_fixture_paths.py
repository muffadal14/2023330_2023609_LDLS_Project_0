#!/usr/bin/env python3
"""Cover the two paths the five corpus documents never reach.

The corpus has no real cross-references and no section small enough to merge,
so both features were unexercised there and one of them turned out to be
broken.  tests/fixtures/document_xref_and_small_blocks.tex is a small document
with real \\label/\\ref pairs and a deliberately tiny section.

Run with:

    .\\latex_efficient_attention_dataset\\.venv\\Scripts\\python.exe tests\\test_fixture_paths.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"
TEX = FIXTURE_DIR / "document_xref_and_small_blocks.tex"
JSON = FIXTURE_DIR / "out" / "document_xref_and_small_blocks.json"

MODEL = ROOT / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
DRIVER = ROOT / "driver" / "build" / "Release" / "llama-latex-driver.exe"

sys.path.insert(0, str(ROOT / "latex_efficient_attention_dataset" / "parser"))
sys.path.insert(0, str(ROOT / "tests"))
import parse_latex as P  # noqa: E402


def build_fixture():
    """Parse the fixture with the settings the fixture is designed for."""
    source = TEX.read_text(encoding="utf-8")
    return P.build_document(source, TEX.stem, TEX.name, MODEL, 4096, 0,
                            seq_mode="blocks", max_sequences=16,
                            min_block_tokens=40, xref_links=True)


def check_structure(data, failures):
    assignment = data["metadata"]["sequence_assignment"]
    blocks = assignment["blocks"]

    merged = [b for b in blocks if b["merged"]]
    if not merged:
        failures.append("the tiny section was not merged into a neighbour")
    print("  merged blocks: %d of %d" % (len(merged), len(blocks)))
    for record in merged:
        print("    seq %d holds node ids %s"
              % (record["seq_id"], record["node_ids"]))

    if not assignment["cross_references"]:
        failures.append("no cross-reference was resolved")
    for link in assignment["cross_references"]:
        if link["from_seq"] == link["target_seq"]:
            failures.append("a cross-reference points inside one sequence")
        print("  xref: seq %d cites seq %d [%s]"
              % (link["from_seq"], link["target_seq"], link["label"]))

    extra = sum(1 for n in data["nodes"] if n.get("extra_seq_ids"))
    if extra:
        failures.append("%d node(s) carry a second sequence, which would "
                        "make that sequence non-contiguous" % extra)

    # The invariant the driver depends on.
    owned = {}
    for node in data["nodes"]:
        for position, seq in zip(node["positions"], node["seq_ids"]):
            owned.setdefault(seq, []).append(position)
    for seq, positions in owned.items():
        ordered = sorted(positions)
        if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
            failures.append("sequence %s is not contiguous" % seq)
    print("  every sequence owns a contiguous position range")

    # Merged sequences must still cover a contiguous range.
    for record in blocks:
        span = range(record["pos_min"], record["pos_max"] + 1)
        covered = set()
        for node in data["nodes"]:
            if record["seq_id"] in node["seq_ids"]:
                covered.update(node["positions"])
        if covered != set(span):
            failures.append("sequence %d does not cover [%d,%d] exactly"
                            % (record["seq_id"], record["pos_min"],
                               record["pos_max"]))


def run_driver(extra):
    command = [
        str(DRIVER), "-m", str(MODEL), "--doc", str(JSON),
        "--mode", "ask", "-n", "16", "-t", "4",
        "--focus-seq", "3",
        "-q", "Which earlier sections does the cache layout refer to?",
    ] + extra
    proc = subprocess.run(command, capture_output=True, text=True,
                          errors="replace")
    return (proc.stdout + proc.stderr), proc.returncode


def check_driver(failures):
    if not DRIVER.exists():
        failures.append("driver not built: %s" % DRIVER)
        return

    with_xref, code = run_driver([])
    if code != 0:
        failures.append("driver failed with cross-references enabled")
        print(with_xref[-1500:])
        return

    joined = [l for l in with_xref.splitlines() if "gather_context" in l]
    print("  with cross-references: %s" % (joined[0].split("I ")[-1] if joined else "?"))
    if not any("holds 2 block" in l for l in joined):
        failures.append("the citing block did not pull in the block it cites")

    without, _ = run_driver(["--no-follow-xrefs"])
    alone = [l for l in without.splitlines() if "gather_context" in l]
    print("  without cross-references: %s"
          % (alone[0].split("I ")[-1] if alone else "?"))
    if not any("holds 1 block" in l for l in alone):
        failures.append("--no-follow-xrefs did not restrict the context")


def main():
    if not TEX.exists():
        raise SystemExit("fixture missing: %s" % TEX)

    print("=== parsing the fixture ===")
    data, _total = build_fixture()
    failures = []
    check_structure(data, failures)

    print("\n=== driver on the fixture ===")
    check_driver(failures)

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print("  - %s" % failure)
        return 1
    print("\nboth paths behave as intended")
    return 0


if __name__ == "__main__":
    sys.exit(main())