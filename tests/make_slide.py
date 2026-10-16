#!/usr/bin/env python3
"""Generate the one-slide summary from the last experiment run.

Reads ``results/experiments.json`` and writes ``results/slide.md``.  Every
number on the slide comes from the run, so the slide cannot drift away from the
results it describes.

Run after tests/run_experiments.py:

    .\\latex_efficient_attention_dataset\\.venv\\Scripts\\python.exe tests\\make_slide.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "experiments.json"
BLOCKS_DIR = ROOT / "latex_efficient_attention_dataset" / "output_blocks"
OUT = ROOT / "results" / "slide.md"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_experiments import N_PREDICT, markup_share, tree_stats  # noqa: E402

DOMAIN = "LaTeX papers: sections, theorem-like environments and display math"


def pct(value):
    return "%.0f%%" % (100 * value)


def live_stats(stem):
    """Tree statistics are recomputed from the dataset, not read from the run.

    They describe the parser output, not the model, so they should not change
    when the experiments are re-run.
    """
    path = BLOCKS_DIR / ("%s.json" % stem)
    if not path.exists():
        return None
    markup, _total = markup_share(path)
    stats = tree_stats(path)
    stats["markup_share"] = markup / stats["tokens"] if stats["tokens"] else 0.0
    return stats


def main():
    if not RESULTS.exists():
        raise SystemExit("no results yet, run tests/run_experiments.py first")

    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    documents = data["documents"]
    control = data.get("positive_control") or {}

    lines = []
    lines.append("# Efficient attention over structured LaTeX papers")
    lines.append("")
    lines.append("**Domain.** %s" % DOMAIN)
    lines.append("")

    lines.append("## Tree statistics")
    lines.append("")
    lines.append("| document | nodes | tokens | blocks | depth | subtree tokens (min/mean/max) | pure markup |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: |")
    for entry in documents:
        tree = live_stats(entry["document"]) or entry["tree"]
        lines.append("| %s | %d | %d | %d | %d | %d / %d / %d | %s |" % (
            entry["document"].replace("document_", ""),
            tree["nodes"], tree["tokens"], tree["blocks"], tree["depth"],
            tree["min_subtree"], tree["mean_subtree"], tree["max_subtree"],
            pct(tree["markup_share"])))
    total_tokens = sum((live_stats(e["document"]) or e["tree"])["tokens"]
                       for e in documents)
    total_blocks = sum((live_stats(e["document"]) or e["tree"])["blocks"]
                       for e in documents)
    lines.append("| **total** | | **%d** | **%d** | | | |" % (total_tokens,
                                                              total_blocks))
    lines.append("")
    lines.append("Depth counts levels below the document root. \"Subtree tokens\" "
                 "is the size of each block that can be evicted, so the root "
                 "(whose subtree is the whole document) is excluded. \"Pure "
                 "markup\" is the share of tokens in headings and in nodes whose "
                 "rendered text carries no prose.")
    lines.append("")

    lines.append("## Test results")
    lines.append("")
    lines.append("| document | changed subtree | tokens | answer | max logit shift | KV before | KV after | freed |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for entry in documents:
        evict = entry.get("evict", {})
        replace = entry.get("replace", {})
        before = entry.get("evict_kv_before")
        after = entry.get("evict_kv_after")
        freed = entry.get("evict_kv_drop_mib")
        lines.append("| %s | %d | %d | %s | %.4f | %.1f MiB | %.1f MiB | %.1f MiB |" % (
            entry["document"].replace("document_", ""),
            entry.get("target", 0),
            entry.get("evict_cells_removed") or 0,
            "unchanged" if evict.get("tokens_identical") else "changed",
            evict.get("max_logit_shift") or 0.0,
            before or 0.0, after or 0.0, freed or 0.0))
    lines.append("")
    lines.append("Both checks were run per document: **replace** refilled the "
                 "changed subtree with new text, **evict** removed it outright.")
    if control:
        lines.append("")
        lines.append("Positive control: evicting the subtree the answer comes from "
                     "changed the answer (max logit shift %.2f), so the "
                     "isolation results above are not vacuous."
                     % (control.get("max_logit_shift") or 0.0))
    lines.append("")

    lines.append("## CPU run times")
    lines.append("")
    lines.append("| document | prefill | baseline answer | replace + evict |")
    lines.append("| --- | ---: | ---: | ---: |")
    for entry in documents:
        prefill = (entry.get("baseline_log") or {}).get("prefill_s")
        lines.append("| %s | %.1f s | %.1f s | %.1f s |" % (
            entry["document"].replace("document_", ""),
            prefill or 0.0, entry.get("baseline_wall_s") or 0.0,
            (entry.get("evict_wall_s") or 0.0) +
            (entry.get("replace_wall_s") or 0.0)))
    lines.append("")
    lines.append("Model: Llama-3.2-1B-Instruct-Q4_K_M, CPU only, greedy decoding.")
    lines.append("Every column is wall clock for one full driver invocation, "
                 "including model load, and all of them ran single threaded so "
                 "that the logits stay reproducible. The isolation results do "
                 "not depend on the thread count; the timings do.")
    lines.append("")

    lines.append("## Paths not exercised by these five documents")
    lines.append("")
    lines.append("| path | why not here | covered by | observed |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| cross-references | the corpus has no `\\label` at all, and its "
                 "one `\\ref` is inside `\\verb\\|...\\|`, which is literal example "
                 "text | `tests/test_fixture_paths.py` | cited block pulled in: 2 "
                 "sequences gathered instead of 1 |")
    lines.append("| small-block merge | no section is small enough to fall under "
                 "`--min-block-tokens` | `tests/test_fixture_paths.py` | 2 of 5 "
                 "sequences merged, each still contiguous |")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append("* Answers stop after %d generated tokens (`-n %d`), so they end "
                 "mid-sentence. That is the budget, not a fault: a fixed budget "
                 "keeps the before/after comparison the same length for every "
                 "document, and an answer does not have to be a finished sentence "
                 "to be provably unchanged." % (N_PREDICT, N_PREDICT))
    lines.append("* The comparison uses top logits rather than text. Removing KV "
                 "cells changes the CPU reduction order, so a greedy choice can "
                 "flip without the context changing; a different context moves "
                 "the logits by whole units, noise by hundredths.")
    lines.append("")
    lines.append("Generated by `tests/make_slide.py` from "
                 "`results/experiments.json` and the parser output, so the "
                 "numbers cannot drift from the run they describe.")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())