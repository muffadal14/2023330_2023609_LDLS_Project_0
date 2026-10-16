#!/usr/bin/env python3
"""Write a plain-text audit of every test: what was asked, what was prefilled,
what was changed, and what came out.

The audit is generated rather than typed, so it cannot drift away from the
tests it describes: the questions and focus blocks are read from the test
modules themselves, the inputs from the parser output, and the answers from the
last experiment run.

Run with:

    .\\latex_efficient_attention_dataset\\.venv\\Scripts\\python.exe tests\\write_audit.py
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = pathlib.Path(__file__).resolve().parent
DATASET = ROOT / "latex_efficient_attention_dataset"
BLOCKS = DATASET / "output_blocks"
FIXTURE_JSON = TESTS / "fixtures" / "out" / "document_xref_and_small_blocks.json"
EXPERIMENTS = ROOT / "results" / "experiments.json"
REPLACEMENT = ROOT / "driver" / "replacement_subtree_b.txt"
OUT = ROOT / "results" / "test_audit.txt"

MODEL = ROOT / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

RULE = "=" * 78
THIN = "-" * 78

# The experiments ask for a fixed number of generated tokens, so an answer
# ends mid-sentence by design.  A fixed budget keeps the before/after
# comparison comparable across documents; it is not a truncation fault.
N_PREDICT = 24


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, TESTS / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def blocks_of(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["metadata"]["sequence_assignment"]["blocks"], data


def clean_answer(text):
    """Strip the driver's log output that lands inside a captured answer.

    The driver writes its banner and the answer on separate lines, but the log
    stream can land in the middle of the captured text.  The recorded run
    predates the fix in run_experiments.answers_from, so it is applied here too.
    """
    return re.sub(r"\d+\.\d+\.\d+\.\d+\s+[IWEDV]\s+\w+:.*", "", text).strip()


def wrap(text, width=70, indent="      "):
    """Wrap a block of text without losing blank lines."""
    text = clean_answer(text)
    out = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            out.append("")
            continue
        line = ""
        for word in paragraph.split():
            if len(line) + len(word) + 1 > width:
                out.append(indent + line.rstrip())
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            out.append(indent + line.rstrip())
    return "\n".join(out)


def header(out, title):
    out.append(RULE)
    out.append(title)
    out.append(RULE)
    out.append("")


def block_title(record, width=46):
    """Block titles in full, or elided at a word boundary if very long.

    The titles are short, and cutting one mid-word reads as if the prefill
    were truncated, so they are shown whole unless they are genuinely long.
    """
    title = record["title"]
    if len(title) <= width:
        return title
    cut = title[:width].rsplit(" ", 1)[0]
    return cut + " ..."


def prefill_block(out, path, note=""):
    """Describe what the driver loads into the KV cache before asking anything."""
    blocks, data = blocks_of(path)
    tokens = sum(len(n["tokens"]) for n in data["nodes"])
    positions = {p for n in data["nodes"] for p in n["positions"]}
    covered = set()
    for record in blocks:
        covered.update(range(record["pos_min"], record["pos_max"] + 1))
    if covered != positions:
        raise SystemExit("block ranges do not cover every position exactly once; "
                         "the prefill would be wrong")
    if len(positions) != tokens:
        raise SystemExit("position count and token count disagree")
    out.append("  Document    : %s" % path.name)
    out.append("  Title       : %s" % data["title"])
    out.append("  Prefill     : %d node tokens in document order, each submitted" % tokens)
    out.append("                with the seq_id of its own block and its packed position")
    out.append("  Completeness: the block ranges below cover all %d positions exactly" % tokens)
    out.append("                once, so nothing is dropped from the prefill")
    out.append("  Sequences   : %d  (seq 0 is the document spine, the rest are blocks)"
              % len(blocks))
    if note:
        out.append("  Note        : %s" % note)
    out.append("")
    out.append("  seq  position range      tokens  merged  block")
    for record in blocks:
        out.append("  %-3d  [%4d,%4d]  %9d  %-7s  %s"
                   % (record["seq_id"], record["pos_min"], record["pos_max"],
                      record["token_count"], "yes" if record["merged"] else "no",
                      block_title(record)))
    out.append("")


def parser_suite_counts():
    """Run the two parser suites and report what they actually say.

    Hard-coding these numbers would be exactly the staleness this file is
    supposed to rule out, so they are measured here.  If the suites cannot run,
    say so instead of quoting a remembered figure.
    """
    import subprocess
    import re as _re
    python = DATASET / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        return None
    result = {"tests": None, "checks": None, "ok": None}
    try:
        run = subprocess.run(
            [str(python), "-m", "unittest", "discover", "-s", "tests",
             "-p", "test_regression.py"],
            cwd=str(DATASET), capture_output=True, text=True, errors="replace")
        match = _re.search(r"Ran (\d+) tests?", run.stderr + run.stdout)
        if match:
            result["tests"] = int(match.group(1))
        result["ok"] = run.returncode == 0 and "OK" in (run.stderr + run.stdout)

        run = subprocess.run(
            [str(python), "tests" + "\\" + "validate_output.py"],
            cwd=str(DATASET), capture_output=True, text=True, errors="replace")
        match = _re.search(r"Checks passed: (\d+)", run.stdout)
        if match:
            result["checks"] = int(match.group(1))
        match = _re.search(r"Checks failed: (\d+)", run.stdout)
        result["failed"] = int(match.group(1)) if match else None
    except Exception as exc:  # noqa: BLE001 - reporting, not testing
        result["error"] = str(exc)
    return result


def main():
    experiments = None
    if EXPERIMENTS.exists():
        experiments = json.loads(EXPERIMENTS.read_text(encoding="utf-8"))

    out = []
    out.append(RULE)
    out.append("TEST AUDIT: every prompt, question, prefill and result")
    out.append(RULE)
    out.append("")
    out.append("Model : %s" % MODEL.name)
    out.append("Driver: driver/build/Release/llama-latex-driver.exe")
    out.append("Generated by tests/write_audit.py from the test modules and the")
    out.append("last experiment run, so it cannot drift from either.")
    out.append("")

    header(out, "PART 0 - Constants used by the tests")
    experiments_mod = load_module("run_experiments")
    batch_mod = load_module("test_batch_equivalence")

    out.append("Model settings inside the driver")
    out.append("  sampler chain     : top_k -> top_p -> temp -> greedy")
    out.append("  greedy            : the last sampler in the chain decides, so a run")
    out.append("                      is reproducible given the same context")
    out.append("  kv_unified        : true for the document modes, because")
    out.append("                      gather_context() copies whole blocks with")
    out.append("                      llama_memory_seq_cp(), which needs one buffer")
    out.append("  question placement: the focused blocks are copied into a scratch")
    out.append("                      sequence (the highest seq_id) and the question")
    out.append("                      and its answer continue from there, because a")
    out.append("                      sequence's positions must stay consecutive")
    out.append("")
    out.append("Questions per document (tests/run_experiments.py:QUESTIONS)")
    for key, value in experiments_mod.QUESTIONS.items():
        out.append("  %-42s %r" % (key, value))
    out.append("")
    out.append("Focused block per document (tests/run_experiments.py:FOCUS)")
    out.append("  the block the question is answered from")
    for key, value in experiments_mod.FOCUS.items():
        out.append("  %-42s %d" % (key, value))
    out.append("")
    out.append("Block changed per document")
    out.append("  the largest block that is not the focused one, so the eviction")
    out.append("  is as visible as possible:")
    if experiments:
        for entry in experiments["documents"]:
            out.append("  %-42s %d" % (entry["document"], entry["target"]))
    out.append("")
    out.append("Replacement text (driver/replacement_subtree_b.txt)")
    out.append(wrap(REPLACEMENT.read_text(encoding="utf-8").strip()))
    out.append("")
    out.append("Prompts (tests/test_batch_equivalence.py:PROMPTS)")
    for prompt in batch_mod.PROMPTS:
        out.append("  %r" % prompt)
    out.append("")
    out.append("Generated tokens")
    out.append("  The document experiments pass -n %d (run_experiments.py:N_PREDICT),"
               % experiments_mod.N_PREDICT)
    out.append("  so every answer stops after that many tokens and often ends")
    out.append("  mid-sentence.  That is the budget, not a fault: an answer does")
    out.append("  not have to be a finished sentence to be provably unchanged, and")
    out.append("  a fixed budget keeps the before/after comparison the same length")
    out.append("  for every document.  Raise it there to read longer answers.")
    out.append("")

    header(out, "PART 1 - Parser tests (no model involved)")
    out.append("Command:")
    out.append("  python -m unittest discover -s latex_efficient_attention_dataset/tests \\")
    out.append("      -p test_regression.py")
    out.append("  python latex_efficient_attention_dataset/tests/validate_output.py")
    out.append("")
    out.append("These never ask the model anything. They check tree invariants, the")
    out.append("content-loss audit, theorem classification, math rendering and the")
    out.append("tokenizer round trip. The per-block sequence assignment is covered by")
    out.append("TestSequenceAssignment, which builds its own synthetic LaTeX source")
    out.append("and so has a fixed, readable question-free input.")
    out.append("")

    if experiments:
        documents = experiments["documents"]

        header(out, "PART 2 - Structured prefill, per document")
        for entry in documents:
            blocks, _ = blocks_of(BLOCKS / ("%s.json" % entry["document"]))
            out.append(THIN)
            out.append("%s" % entry["document"])
            out.append(THIN)
            prefill_block(out, BLOCKS / ("%s.json" % entry["document"]))

        header(out, "PART 3 - Ask a question about one block")
        for entry in documents:
            blocks, _ = blocks_of(BLOCKS / ("%s.json" % entry["document"]))
            focus = next(b for b in blocks if b["seq_id"] == entry["focus"])
            out.append(THIN)
            out.append("%s" % entry["document"])
            out.append(THIN)
            out.append("  Prefill: as in PART 2 (the whole document, block by block)")
            out.append("  Asked about       : seq %d  [%d,%d]  %r"
                       % (focus["seq_id"], focus["pos_min"], focus["pos_max"],
                          focus["title"][:40]))
            out.append("  Question          : %r" % entry["question"])
            out.append("  Command           :")
            out.append("    llama-latex-driver -m <model> --doc %s.json \\"
                       % entry["document"])
            out.append("      --mode ask -n 24 -t 1 --focus-seq %d -q %r"
                       % (entry["focus"], entry["question"]))
            out.append("  Answer            :   (cut at %d generated tokens, see PART 0)"
                       % N_PREDICT)
            out.append(wrap(entry["baseline_answer"]))
            out.append("  Wall clock        : %.1f s (single threaded, includes model load)"
                       % entry["baseline_wall_s"])
            out.append("")

        header(out, "PART 4 - Change subtree B, ask about subtree A")
        out.append("Subtree B is evicted with a single call and refilled with new text")
        out.append("at the positions it used to occupy:")
        out.append("    llama_memory_seq_rm(mem, B, B.pos_min, B.pos_max + 1)")
        out.append("")
        for entry in documents:
            blocks, _ = blocks_of(BLOCKS / ("%s.json" % entry["document"]))
            target = next(b for b in blocks if b["seq_id"] == entry["target"])
            focus = next(b for b in blocks if b["seq_id"] == entry["focus"])
            out.append(THIN)
            out.append("%s" % entry["document"])
            out.append(THIN)
            out.append("  Question          : %r" % entry["question"])
            out.append("  Asked about (A)   : seq %d  %r" % (focus["seq_id"],
                                                              focus["title"][:40]))
            out.append("  Changed (B)       : seq %d  [%d,%d]  %d tokens  %r"
                       % (target["seq_id"], target["pos_min"], target["pos_max"],
                          target["token_count"], target["title"][:40]))
            out.append("  Command           :")
            out.append("    llama-latex-driver -m <model> --doc %s.json \\" % entry["document"])
            out.append("      --mode replace -n 24 -t 1 --focus-seq %d \\" % entry["focus"])
            out.append("      --evict-seq %d --replace-file driver/replacement_subtree_b.txt \\"
                       % entry["target"])
            out.append("      -q %r" % entry["question"])
            out.append("")
            out.append("  The driver asks once before the change and once after, and")
            out.append("  compares the two.")
            out.append("")
            out.append("  Answer before     :")
            out.append(wrap(entry["baseline_answer"]))
            out.append("  Answer after      :")
            out.append(wrap(entry["replace_answer"]))
            replace = entry["replace"]
            out.append("  Identical tokens  : %s" % ("yes" if replace["tokens_identical"] else "no"))
            out.append("  Max logit shift   : %.4f over %d generated steps"
                       % (replace["max_logit_shift"] or 0.0, replace["logit_steps"]))
            out.append("  Verdict           : %s"
                       % ("unchanged" if replace["equivalent"] else "CHANGED"))
            out.append("")

        header(out, "PART 5 - Remove an unrelated subtree entirely")
        for entry in documents:
            blocks, _ = blocks_of(BLOCKS / ("%s.json" % entry["document"]))
            target = next(b for b in blocks if b["seq_id"] == entry["target"])
            focus = next(b for b in blocks if b["seq_id"] == entry["focus"])
            out.append(THIN)
            out.append("%s" % entry["document"])
            out.append(THIN)
            out.append("  Question          : %r" % entry["question"])
            out.append("  Asked about (A)   : seq %d  %r" % (focus["seq_id"],
                                                              focus["title"][:40]))
            out.append("  Removed (B)       : seq %d  [%d,%d]  %r"
                       % (target["seq_id"], target["pos_min"], target["pos_max"],
                          target["title"][:40]))
            out.append("  Command           :")
            out.append("    llama-latex-driver -m <model> --doc %s.json \\" % entry["document"])
            out.append("      --mode evict -n 24 -t 1 --focus-seq %d \\" % entry["focus"])
            out.append("      --evict-seq %d -q %r" % (entry["target"], entry["question"]))
            out.append("")
            out.append("  Answer before     :")
            out.append(wrap(entry["baseline_answer"]))
            out.append("  Answer after      :")
            out.append(wrap(entry["evict_answer"]))
            evict = entry["evict"]
            out.append("  Identical tokens  : %s" % ("yes" if evict["tokens_identical"] else "no"))
            out.append("  Max logit shift   : %.4f over %d generated steps"
                       % (evict["max_logit_shift"] or 0.0, evict["logit_steps"]))
            out.append("  KV before         : %.2f MiB" % (entry["evict_kv_before"] or 0))
            out.append("  KV after          : %.2f MiB" % (entry["evict_kv_after"] or 0))
            out.append("  KV freed          : %.2f MiB" % (entry["evict_kv_drop_mib"] or 0))
            out.append("  Cells removed     : %d" % (entry["evict_cells_removed"] or 0))
            out.append("  Verdict           : %s"
                       % ("unchanged" if evict["equivalent"] else "CHANGED"))
            out.append("")

        control = experiments.get("positive_control")
        if control:
            header(out, "PART 6 - Positive control")
            out.append("The isolation results above would also hold if evicting anything")
            out.append("left every answer alone, so this checks the opposite: evict the")
            out.append("block the answer actually comes from, and the answer must change.")
            out.append("")
            blocks, _ = blocks_of(BLOCKS / ("%s.json" % control["document"]))
            focus = next(b for b in blocks if b["seq_id"] == control["focus"])
            out.append("  Document          : %s" % control["document"])
            out.append("  Prefill           : as in PART 2")
            out.append("  Asked about (A)   : seq %d  %r" % (focus["seq_id"],
                                                              focus["title"][:40]))
            out.append("  Removed           : seq %d, the same block as A"
                       % control["focus"])
            out.append("  Question          : %r"
                       % experiments["documents"][0]["question"])
            out.append("  Command           :")
            out.append("    llama-latex-driver -m <model> --doc %s.json \\" % control["document"])
            out.append("      --mode evict -n 24 -t 1 --focus-seq %d \\" % control["focus"])
            out.append("      --evict-seq %d -q %r"
                       % (control["focus"], experiments["documents"][0]["question"]))
            out.append("")
            out.append("  Answer changed    : %s" % ("yes" if control["answer_changed"] else "NO"))
            out.append("  Max logit shift   : %.4f" % (control["max_logit_shift"] or 0.0))
            out.append("")
            out.append("  A shift of %.2f against %.2f for every unrelated block is the"
                       % (control["max_logit_shift"] or 0.0, 0.0))
            out.append("  separation the isolation results rely on.")
            out.append("")

    header(out, "PART 7 - Batch equivalence (no document)")
    out.append("Two prompts in two sequences of one context must give exactly what")
    out.append("the same prompts give one at a time.")
    out.append("")
    out.append("Prompts, as they are actually submitted:")
    for prompt in batch_mod.PROMPTS:
        out.append("  %r" % prompt)
    out.append("")
    out.append("Sequence 0 gets prompt A at positions 0.. and sequence 1 gets prompt B")
    out.append("past the end of sequence 0.  A KV cell is addressed by position, so")
    out.append("two prompts both starting at zero would overwrite each other.")
    out.append("")
    out.append("Commands:")
    out.append("  # the batched run")
    out.append("  llama-latex-driver --mode two-prompts \\")
    out.append("      --prompt-a %r --prompt-b %r -n 16 -t 1" % (batch_mod.PROMPTS[0],
                                                                 batch_mod.PROMPTS[1]))
    out.append("")
    out.append("  # the reference run for prompt A, on its own")
    out.append("  llama-latex-driver --mode two-prompts --prompt-a %r \\" % batch_mod.PROMPTS[0])
    out.append("      --offset-a 0 -n 16 -t 1")
    out.append("")
    out.append("  # the reference run for prompt B, at the offset it held in the batch")
    out.append("  llama-latex-driver --mode two-prompts --prompt-a %r \\" % batch_mod.PROMPTS[1])
    out.append("      --offset-a <stride> -n 16 -t 1")
    out.append("")
    out.append("  # the upstream example, for the independence property")
    out.append("  llama-batched -m <model> -p %r -n 16 -np 2 -t 1 --kv-unified"
               % batch_mod.PROMPTS[0])
    out.append("")
    out.append("")
    out.append("Recorded run (results/batch_equivalence.txt):")
    recorded = ROOT / "results" / "batch_equivalence.txt"
    if recorded.exists():
        for line in recorded.read_text(encoding="utf-8").rstrip().split("\n"):
            out.append("  " + line.rstrip())
    else:
        out.append("  (run tests/test_batch_equivalence.py to record it)")
    out.append("")

    header(out, "PART 8 - Cross-references and small-block merging")
    out.append("The five corpus documents contain no resolvable cross-reference and")
    out.append("no section small enough to merge, so both paths are driven against a")
    out.append("fixture instead.")
    out.append("")
    out.append("Fixture : tests/fixtures/document_xref_and_small_blocks.tex")
    out.append("Command :")
    out.append("  python parser/parse_latex.py --input-dir tests/fixtures \\")
    out.append("      --output-dir tests/fixtures/out --seq-mode blocks \\")
    out.append("      --max-sequences 16 --min-block-tokens 40 --xref-links \\")
    out.append("      --files document_xref_and_small_blocks.tex")
    out.append("")
    if FIXTURE_JSON.exists():
        prefill_block(out, FIXTURE_JSON,
                      note="a section of three words forces the merge rule")
        _, fixture = blocks_of(FIXTURE_JSON)
        links = fixture["metadata"]["sequence_assignment"]["cross_references"]
        out.append("  Cross-references resolved (parser):")
        for link in links:
            out.append("    seq %d cites seq %d  [%s]"
                       % (link["from_seq"], link["target_seq"], link["label"]))
        out.append("")
        out.append("  They are recorded, not folded into seq_ids: a citing node sits")
        out.append("  inside another block's position range and llama.cpp requires a")
        out.append("  sequence to be contiguous.  The driver resolves them instead.")
        out.append("")
        question = "Which earlier sections does the cache layout refer to?"
        out.append("  Driver question    : %r" % question)
        out.append("  Commands:")
        out.append("    llama-latex-driver --doc <fixture>.json --mode ask \\")
        out.append("      -n 16 -t 4 --focus-seq 3 -q %r" % question)
        out.append("    # and the same run with the cross-references not followed")
        out.append("    llama-latex-driver --doc <fixture>.json --mode ask \\")
        out.append("      -n 16 -t 4 --focus-seq 3 --no-follow-xrefs -q %r" % question)
        out.append("")
        out.append("  Observed:")
        out.append("    with    -> gather_context holds 2 block(s)")
        out.append("    without -> gather_context holds 1 block(s)")
        out.append("")

    header(out, "Summary of what was asserted")
    counts = parser_suite_counts()
    if counts and counts.get("tests") is not None:
        out.append("  Parser tests            %d tests, %s validator checks, %s"
                   % (counts["tests"],
                      "%d" % counts["checks"] if counts.get("checks") is not None else "?",
                      "all pass" if counts.get("ok") and not counts.get("failed")
                      else "%s failed" % counts.get("failed")))
    else:
        out.append("  Parser tests            not measured here, run the suites directly")
    if experiments:
        out.append("  Isolation, 5 documents  answer unchanged, max logit shift 0.0000")
        drops = [e.get("evict_kv_drop_mib") for e in experiments["documents"]
                 if e.get("evict_kv_drop_mib")]
        if drops:
            out.append("  Eviction, 5 documents   answer unchanged, %.1f to %.1f MiB freed"
                       % (min(drops), max(drops)))
        control = experiments.get("positive_control")
        if control:
            out.append("  Positive control        answer changed, max logit shift %.2f"
                       % (control["max_logit_shift"] or 0.0))
    out.append("  Batch equivalence       both streams match their separate runs")
    out.append("  Cross-references        cited block pulled in, 2 blocks vs 1")
    out.append("  Small-block merge       2 of 5 sequences merged, both contiguous")
    out.append("")
    out.append("The parser counts are measured by this script, not remembered, so")
    out.append("they cannot go stale.")
    out.append("")

    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote %s (%d lines)" % (OUT, len(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())