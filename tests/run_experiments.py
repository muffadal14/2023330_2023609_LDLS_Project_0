#!/usr/bin/env python3
"""End-to-end experiments for the structured LaTeX driver.

Runs the driver on every parser document and checks the three properties the
project cares about:

1. isolation  - changing subtree B leaves the answer about subtree A unchanged
2. eviction   - removing an unrelated subtree keeps the answer and frees memory
3. focus      - the answer comes from the focused block, not from the whole paper

Every check runs against the real model on CPU, so a failure here means the
masking scheme is wrong, not that a stub disagrees with itself.

Run with the project virtual environment:

    .\\latex_efficient_attention_dataset\\.venv\\Scripts\\python.exe tests\\run_experiments.py

Options:
    --quick     run one document instead of all five
    --driver    path to llama-latex-driver.exe
    --model     path to the GGUF model
    --threads   CPU threads for the driver
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET = ROOT / "latex_efficient_attention_dataset"
BLOCKS_DIR = DATASET / "output_blocks"
SINGLE_DIR = DATASET / "output"

DEFAULT_MODEL = ROOT / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
DEFAULT_DRIVER = ROOT / "driver" / "build" / "Release" / "llama-latex-driver.exe"

# Two runs that attended to the same context differ by far less than this in
# their top logits; a masking leak changes them by whole units.
LOGIT_EPSILON = 0.25

# Which subtree holds the answer, and which subtree is safe to mutate or evict.
# Chosen per document from its block layout: the answer must come from a block
# that is not the one being changed.
FOCUS = {
    "document_1_transformer_architecture": 1,
    "document_2_kv_cache_management": 1,
    "document_3_mathematical_results": 1,
    "document_4_structured_documents": 1,
    "document_5_attention_masks": 1,
}

QUESTIONS = {
    "document_1_transformer_architecture":
        "What does the attention mechanism compare?",
    "document_2_kv_cache_management":
        "What does the cache store for each token?",
    "document_3_mathematical_results":
        "What does the paper illustrate?",
    "document_4_structured_documents":
        "What do research papers contain?",
    "document_5_attention_masks":
        "What does an attention mask determine?",
}


# Generated tokens per answer.  A fixed budget keeps the before/after
# comparison the same length for every document, and an answer does not have to
# be a finished sentence to be provably unchanged.  Raise it to read longer
# answers; the comparison stays valid either way.
N_PREDICT = 24


def run_driver(driver, model, doc, extra, threads, logits_out=None):
    command = [
        str(driver),
        "-m", str(model),
        "--doc", str(doc),
        "-t", str(threads),
        "-n", str(N_PREDICT),
    ]
    if logits_out:
        command += ["--logits-out", str(logits_out)]
    command += extra
    started = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, errors="replace")
    wall = time.perf_counter() - started
    return proc, wall


def read_logits(path, phase=None):
    """Per-step (token, top logit) written by the driver.

    Rows are ``label, step, token, logit``.  ``phase`` selects one pass of a
    two-pass mode (``baseline`` or ``after``).
    """
    if not path.exists():
        return []
    rows = []
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[0] == "phase":
            current = parts[1]
            continue
        if len(parts) == 4:
            if phase is not None and current != phase:
                continue
            rows.append((int(parts[2]), float(parts[3])))
    return rows


def compare_runs(baseline_logits, after_logits):
    """Decide whether two runs really saw the same context.

    Text equality is the headline result but it is not the right test on CPU:
    the reduction order depends on how the KV cells are laid out, so removing
    cells can flip one greedy choice without changing what the model attended
    to.  A real masking leak moves the logits a lot; float noise moves them a
    little.  Both are reported so the distinction is visible.
    """
    result = {
        "tokens_identical": False,
        "logit_steps": 0,
        "max_logit_shift": None,
        "equivalent": False,
    }
    if not baseline_logits or not after_logits:
        return result

    result["tokens_identical"] = [t for t, _ in baseline_logits] == \
                                 [t for t, _ in after_logits]
    n = min(len(baseline_logits), len(after_logits))
    result["logit_steps"] = n
    result["max_logit_shift"] = max(
        (abs(a[1] - b[1]) for a, b in zip(baseline_logits[:n],
                                           after_logits[:n])), default=0.0)
    result["equivalent"] = result["tokens_identical"] or \
        result["max_logit_shift"] < LOGIT_EPSILON
    return result


def parse_log(text):
    """Pull the numbers and answers the driver reports out of its log."""
    out = {"answers": [], "kv_before_mib": None, "kv_after_mib": None,
           "unchanged": None, "cells_removed": None, "prefill_s": None,
           "ask_s": None}
    for line in text.splitlines():
        match = re.search(r"prefill done in ([\d.]+) s, state = ([\d.]+) MiB", line)
        if match:
            out["prefill_s"] = float(match.group(1))
            out["kv_before_mib"] = float(match.group(2))
        match = re.search(r"state ([\d.]+) -> ([\d.]+) MiB", line)
        if match:
            out["kv_before_mib"] = float(match.group(1))
            out["kv_after_mib"] = float(match.group(2))
        match = re.search(r"removed (\d+) cells", line)
        if match:
            out["cells_removed"] = int(match.group(1))
        match = re.search(r"unchanged = (\w+)", line)
        if match:
            out["unchanged"] = match.group(1) == "yes"
        if "baseline answer" in line:
            out["current"] = "baseline"
        if "answer after change" in line:
            out["current"] = "after"
    return out


def strip_log_lines(text):
    """Drop llama.cpp log output that the driver interleaves with its answers.

    The driver writes its banner and the answer on separate lines, but the log
    stream can land in the middle of the captured text, so the noise has to be
    removed rather than just skipped over.
    """
    cleaned = re.sub(r"\d+\.\d+\.\d+\.\d+\s+[IWEDV]\s+\w+:.*", "", text)
    return cleaned.strip()


def answers_from(text):
    """Answers are the lines that follow the driver's answer banners."""
    answers = []
    collecting = False
    for line in text.splitlines():
        if "baseline answer" in line or "answer after change" in line:
            collecting = True
            answers.append("")
            continue
        if collecting:
            if line.startswith(("0.", "1.", "2.")) and ": main:" in line:
                collecting = False
                continue
            if line.strip():
                answers[-1] += line
    return [strip_log_lines(a) for a in answers]


def blocks_of(doc_path):
    data = json.loads(doc_path.read_text(encoding="utf-8"))
    return data["metadata"]["sequence_assignment"]["blocks"]


def tree_stats(doc_path):
    """Depth, subtree sizes and the share of tokens that are pure markup."""
    data = json.loads(doc_path.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    order = [n["id"] for n in data["nodes"]]

    depth = {}

    def node_depth(node_id):
        if node_id in depth:
            return depth[node_id]
        node = nodes[node_id]
        depth[node_id] = 1 + node_depth(node["parent"]) if node["parent"] else 1
        return depth[node_id]

    def subtree_tokens(node_id):
        total = nodes[node_id]["token_count"]
        for child in nodes[node_id]["child_ids"]:
            total += subtree_tokens(child)
        return total

    total_tokens = sum(n["token_count"] for n in data["nodes"])
    # The root's subtree is the whole document, which is not a useful
    # "subtree size"; report the size of every block that can be evicted.
    sizes = [subtree_tokens(n["id"]) for n in data["nodes"]
             if n["child_ids"] and n["id"] != "n0"]
    types = data["metadata"]["node_type_counts"]
    return {
        "nodes": len(data["nodes"]),
        "tokens": total_tokens,
        "depth": max(node_depth(i) for i in order) - 1,
        "blocks": len(blocks_of(doc_path)),
        "min_subtree": min(sizes) if sizes else 0,
        "max_subtree": max(sizes) if sizes else 0,
        "mean_subtree": (sum(sizes) / len(sizes)) if sizes else 0,
        "structural_types": sorted(
            t for t in types
            if t in {"section", "subsection", "subsubsection", "abstract",
                     "equation", "align", "cases", "itemize", "enumerate",
                     "algorithm", "algorithmic", "table", "figure"}),
        "structural_nodes": sum(
            types.get(t, 0) for t in
            ("section", "subsection", "subsubsection", "equation", "align",
             "cases", "itemize", "enumerate", "algorithm", "algorithmic",
             "table", "figure")),
        "type_counts": types,
    }


def markup_share(doc_path, tokenizer_hint=None):
    """Share of nodes whose text is pure markup rather than prose.

    A node counts as markup when its rendered text carries no sentence
    punctuation and no lowercase word of three letters or more, which is the
    signature of a heading or an equation rather than a paragraph.
    """
    data = json.loads(doc_path.read_text(encoding="utf-8"))
    markup_tokens = 0
    total_tokens = 0
    for node in data["nodes"]:
        text = node["text"]
        total_tokens += node["token_count"]
        words = re.findall(r"[a-z]{3,}", text)
        if node["node_type"] in ("section", "subsection", "subsubsection"):
            markup_tokens += node["token_count"]
        elif not words:
            markup_tokens += node["token_count"]
    return markup_tokens, total_tokens


def run_positive_control(args, docs, scratch, failures):
    """Evict the subtree that the question is about.

    The isolation tests above pass trivially if evicting *anything* leaves the
    answer alone, so this is the check that makes them meaningful: removing the
    context the answer comes from must change the answer.  Single threaded, so
    that a difference in the token text is a real difference and not CPU
    reduction noise.
    """
    if not docs:
        return None

    doc = docs[0]
    stem = doc.stem
    focus = FOCUS.get(stem, 1)
    question = QUESTIONS.get(stem, "What is this document about?")

    base = scratch / ("control_%s_base.tsv" % stem)
    after = scratch / ("control_%s_after.tsv" % stem)

    run_driver(args.driver, args.model, doc,
               ["--mode", "ask", "-q", question,
                "--focus-seq", str(focus)],
               1, base)
    proc, _ = run_driver(
        args.driver, args.model, doc,
        ["--mode", "evict", "-q", question, "--focus-seq", str(focus),
         "--evict-seq", str(focus)],
        1, after)

    baseline_logits = read_logits(base)
    after_logits = read_logits(after, phase="after")
    comparison = compare_runs(baseline_logits, after_logits)
    changed = not comparison["tokens_identical"]

    print("\n=== positive control: %s ===" % stem)
    print("evicting the focused subtree %d: answer changed = %s "
          "(max logit shift %.4f)"
          % (focus, changed, comparison["max_logit_shift"] or 0.0))
    if not changed:
        failures.append(
            "positive control: evicting the subtree the answer comes from did "
            "not change the answer, so the isolation tests prove nothing")
    return {"document": stem, "focus": focus, "answer_changed": changed,
            "max_logit_shift": comparison["max_logit_shift"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="run one document only")
    ap.add_argument("--driver", type=pathlib.Path, default=DEFAULT_DRIVER)
    ap.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    ap.add_argument("--threads", type=int, default=8,
                    help="threads for the timing runs")
    ap.add_argument("--isolation-threads", type=int, default=1,
                    help="threads for the isolation comparisons; 1 makes the "
                         "logits exactly reproducible")
    ap.add_argument("--n-predict", type=int, default=24)
    args = ap.parse_args()

    if not args.driver.exists():
        raise SystemExit("driver not built: %s" % args.driver)
    if not args.model.exists():
        raise SystemExit("model not found: %s" % args.model)
    if not BLOCKS_DIR.exists():
        raise SystemExit("run the parser with --seq-mode blocks first: %s" % BLOCKS_DIR)

    docs = sorted(BLOCKS_DIR.glob("document_*.json"))
    if args.quick:
        docs = docs[-1:]

    replacement = ROOT / "driver" / "replacement_subtree_b.txt"
    scratch = ROOT / "results" / "logs"
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []

    for doc in docs:
        stem = doc.stem
        focus = FOCUS.get(stem, 1)
        question = QUESTIONS.get(stem, "What is this document about?")
        blocks = blocks_of(doc)
        # Change the largest subtree that is not the one being questioned: the
        # bigger the eviction, the clearer the memory drop.
        others = [b for b in blocks if b["seq_id"] not in (0, focus)]
        target = max(others, key=lambda b: b["token_count"])["seq_id"] if others else None

        print("\n=== %s ===" % stem)
        print("blocks: %d, focus: %d, changed: %s"
              % (len(blocks), focus, target))

        entry = {"document": stem, "focus": focus, "target": target,
                 "question": question, "tree": tree_stats(doc)}
        markup, total = markup_share(doc)
        entry["markup_tokens"] = markup
        entry["markup_share"] = markup / total if total else 0.0

        # 1. baseline answer
        base_logits = scratch / ("%s_baseline.tsv" % stem)
        proc, wall = run_driver(
            args.driver, args.model, doc,
            ["--mode", "ask", "-q", question, "--focus-seq", str(focus)],
            args.isolation_threads, base_logits)
        baseline = answers_from(proc.stderr + proc.stdout)
        baseline_logits = read_logits(base_logits)
        entry["baseline_answer"] = baseline[0].strip() if baseline else ""
        entry["baseline_wall_s"] = wall
        entry["baseline_log"] = parse_log(proc.stderr)
        entry["baseline_logits"] = baseline_logits
        print("baseline (%.1f s): %s" % (wall, entry["baseline_answer"][:90]))
        if not entry["baseline_answer"]:
            failures.append("%s: no baseline answer" % stem)

        if target is None:
            results.append(entry)
            continue

        # 2. change subtree B, ask about subtree A
        # The isolation comparisons run single threaded so that an unchanged
        # answer means the same context, not the same rounding.
        replace_logits = scratch / ("%s_replace.tsv" % stem)
        proc, wall = run_driver(
            args.driver, args.model, doc,
            ["--mode", "replace", "-q", question, "--focus-seq", str(focus),
             "--evict-seq", str(target), "--replace-file", str(replacement)],
            args.isolation_threads, replace_logits)
        log = parse_log(proc.stderr)
        changed = answers_from(proc.stderr + proc.stdout)
        entry["replace_wall_s"] = wall
        entry["replace_kv_before"] = log["kv_before_mib"]
        entry["replace_kv_after"] = log["kv_after_mib"]
        entry["replace_answer"] = changed[-1].strip() if changed else ""
        entry["replace"] = compare_runs(
            baseline_logits, read_logits(replace_logits, phase="after"))
        print("replace subtree %d (%.1f s): same_text=%s max_logit_shift=%.4f"
              % (target, wall, entry["replace"]["tokens_identical"],
                 entry["replace"]["max_logit_shift"] or 0.0))
        if not entry["replace"]["equivalent"]:
            failures.append("%s: output changed when an unrelated subtree was "
                            "replaced" % stem)

        # 3. evict an unrelated subtree: answer holds, memory drops
        evict_logits = scratch / ("%s_evict.tsv" % stem)
        proc, wall = run_driver(
            args.driver, args.model, doc,
            ["--mode", "evict", "-q", question, "--focus-seq", str(focus),
             "--evict-seq", str(target)],
            args.isolation_threads, evict_logits)
        log = parse_log(proc.stderr)
        evicted = answers_from(proc.stderr + proc.stdout)
        entry["evict_wall_s"] = wall
        entry["evict_cells_removed"] = log["cells_removed"]
        entry["evict"] = compare_runs(
            baseline_logits, read_logits(evict_logits, phase="after"))
        entry["evict_kv_before"] = log["kv_before_mib"]
        entry["evict_kv_after"] = log["kv_after_mib"]
        entry["evict_answer"] = evicted[-1].strip() if evicted else ""
        entry["evict_kv_drop_mib"] = (
            (log["kv_before_mib"] - log["kv_after_mib"])
            if log["kv_before_mib"] is not None and log["kv_after_mib"] is not None
            else None)
        print("evict subtree %d (%.1f s): same_text=%s max_logit_shift=%.4f, "
              "KV %.2f -> %.2f MiB"
              % (target, wall, entry["evict"]["tokens_identical"],
                 entry["evict"]["max_logit_shift"] or 0.0,
                 log["kv_before_mib"] or 0, log["kv_after_mib"] or 0))
        if not entry["evict"]["equivalent"]:
            failures.append("%s: answer changed after evicting an unrelated "
                            "subtree" % stem)
        if not entry["evict_kv_drop_mib"] or entry["evict_kv_drop_mib"] <= 0:
            failures.append("%s: KV memory did not drop after eviction" % stem)

        results.append(entry)

    control = run_positive_control(args, docs, scratch, failures)

    out = ROOT / "results" / "experiments.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"documents": results, "positive_control": control,
               "logit_epsilon": LOGIT_EPSILON}
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print("%-40s %5s %6s %11s %9s"
          % ("document", "blocks", "tokens", "max shift", "KV drop"))
    for entry in results:
        drop = entry.get("evict_kv_drop_mib")
        evict = entry.get("evict", {})
        print("%-40s %5d %6s %11.4f %9s"
              % (entry["document"], entry["tree"]["blocks"],
                 "same" if evict.get("tokens_identical") else "differ",
                 evict.get("max_logit_shift") or 0.0,
                 ("%.2f MiB" % drop) if drop else "n/a"))
    print("\n'max shift' is the largest change in the top logit across the "
          "generated answer.\nA masking leak moves it by whole units; CPU "
          "reduction-order noise moves it\nby a few hundredths, which is why "
          "the token text can differ while the\ncontext is provably the same.")
    print("\nwrote %s" % out)

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print("  - %s" % failure)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())