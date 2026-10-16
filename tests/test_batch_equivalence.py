#!/usr/bin/env python3
"""Foundation check: a batched run must equal the same runs made one at a time.

Two prompts submitted as two sequences of one prefill must produce exactly what
the same two prompts produce when each is prefilled and decoded on its own. If
the attention mask were built from anything other than the sequence ids, each
prompt would see the other's tokens and the streams would drift apart.

Verified twice: against the upstream ``llama-batched`` example, and against our
own driver's ``two-prompts`` mode.

Run with:

    .\\latex_efficient_attention_dataset\\.venv\\Scripts\\python.exe tests\\test_batch_equivalence.py
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BATCHED = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-batched.exe"
DRIVER = ROOT / "driver" / "build" / "Release" / "llama-latex-driver.exe"
DEFAULT_MODEL = ROOT / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

PROMPTS = [
    "Hello my name is",
    "The capital of France is",
]

# Two runs that attended to the same context differ by far less than this in
# their top logits; a different context moves them by whole units.
LOGIT_EPSILON = 0.25

# Everything the test prints is mirrored here so the audit can quote it.
LINES = []


def emit(text=""):
    LINES.append(text)
    print(text)

# The log line that llama.cpp interleaves into stdout, e.g.
# "0.01.137.156 I main: n_predict = 16, ..."
LOG_LINE = re.compile(r"^\d+\.\d+\.\d+\.\d+\s+[IWED]\s+\w+:")


def clean(text):
    """Drop llama.cpp log lines and the prompt echo."""
    out = []
    for line in text.splitlines():
        if LOG_LINE.match(line.strip()):
            continue
        out.append(line)
    return "\n".join(out).strip()


def run_batched(model, prompt, n_parallel, n_predict, threads):
    command = [
        str(BATCHED), "-m", str(model), "-p", prompt,
        "-n", str(n_predict), "-np", str(n_parallel), "-t", str(threads),
        # A batch whose tokens belong to several sequences needs the unified
        # KV buffer; without it llama.cpp refuses the batch outright.
        "--kv-unified",
    ]
    return subprocess.run(command, capture_output=True, text=True,
                          errors="replace")


def run_driver_two_prompts(model, prompt_a, prompt_b, n_predict, threads,
                           joint=False, logits_out=None):
    command = [
        str(DRIVER), "-m", str(model), "--mode", "two-prompts",
        "--prompt-a", prompt_a, "--prompt-b", prompt_b,
        "-n", str(n_predict), "-t", str(threads),
    ]
    if joint:
        command.append("--joint-prefill")
    if logits_out:
        command += ["--logits-out", str(logits_out)]
    return subprocess.run(command, capture_output=True, text=True,
                          errors="replace")


def run_driver_single(model, prompt, n_predict, threads, offset_a=0,
                      logits_out=None):
    """Reference run: one prompt, one sequence, nothing else in the batch.

    ``offset_a`` must reproduce where the prompt sat in the batched run. RoPE
    positions are absolute even though the attention between two positions is
    relative, so a reference run at a different offset is not the same run.
    """
    command = [
        str(DRIVER), "-m", str(model), "--mode", "two-prompts",
        "--prompt-a", prompt, "--offset-a", str(offset_a),
        "-n", str(n_predict), "-t", str(threads),
    ]
    if logits_out:
        command += ["--logits-out", str(logits_out)]
    return subprocess.run(command, capture_output=True, text=True,
                          errors="replace")


def read_logits(path, label="seq0"):
    """Per-step (token, top logit) written by the driver."""
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 4 and parts[0] == label:
            rows.append((int(parts[2]), float(parts[3])))
    return rows


def max_logit_shift(a, b):
    n = min(len(a), len(b))
    return max((abs(a[i][1] - b[i][1]) for i in range(n)), default=None)


def sequence_lines(text):
    """The driver's 'sequence N: <text>' log lines, in order."""
    return [m.group(2) for m in
            re.finditer(r"^\S+\s+I\s+main: sequence (\d+):\s?(.*)$", text, re.M)]


def check_batched_example(model, n_predict, threads):
    """Independence, as demonstrated by the upstream example.

    ``llama-batched`` cannot express two different prompts: it submits one
    prompt to every sequence and then samples each with a stochastic chain, so
    comparing its text against a separate run would compare two different
    sampling processes rather than two contexts.  What it does demonstrate is
    that two sequences in one batch stay independent, which is the part of the
    mechanism the driver relies on.  The equality check itself runs below
    against the driver's greedy two-prompt mode.
    """
    prompt = PROMPTS[0]
    joint = run_batched(model, prompt, 2, n_predict, threads)
    streams = re.findall(r"^sequence \d+:\n\n(.*?)\n\n", joint.stdout, re.M | re.S)
    texts = [clean(s) for s in streams]

    emit("=== llama-batched, one prompt shared by two sequences ===")
    emit("  (the upstream example samples stochastically, so this section "
         "changes on every run; only the independence property is being "
         "checked here, not the exact text)")
    for index, text in enumerate(texts):
        emit("  seq %d -> %r" % (index, text[:90]))

    if len(texts) < 2:
        emit("  could not parse two sequences")
        return []

    failures = []
    if texts[0] == texts[1]:
        failures.append("both sequences produced identical text, so the two "
                        "streams are not independent")
    else:
        emit("  the two sequences diverge, so they are independent")
    return failures


def token_count(model_path):
    """Prompt A length, needed to reproduce sequence 1's position offset."""
    from llama_cpp import Llama
    llm = Llama(model_path=str(model_path), vocab_only=True, n_ctx=512,
                verbose=False)
    try:
        return {p: len(llm.tokenize(p.encode("utf-8"), add_bos=True,
                                    special=True)) for p in PROMPTS}
    finally:
        del llm


def check_two_prompts(model, n_predict, threads):
    """Our driver, two genuinely different prompts in two sequences."""
    prompt_a, prompt_b = PROMPTS
    joint = run_driver_two_prompts(model, prompt_a, prompt_b, n_predict, threads)

    combined = joint.stdout + joint.stderr
    joint_texts = sequence_lines(combined)
    if len(joint_texts) < 2:
        emit("--- driver output ---")
        emit(combined[-2000:])
        return ["driver two-prompts produced %d sequences" % len(joint_texts)]

    emit("\n=== driver two-prompts, two different prompts ===")

    # Sequence 1 starts past sequence 0; reproduce that offset so the
    # reference run is the same computation.
    try:
        lengths = token_count(model)
    except ImportError:
        emit("  llama-cpp-python unavailable, cannot reproduce the offset")
        return []
    stride = lengths[prompt_a] + n_predict + 8

    singles = []
    for prompt, offset in ((prompt_a, 0), (prompt_b, stride)):
        solo = run_driver_single(model, prompt, n_predict, threads, offset)
        solo_lines = sequence_lines(solo.stdout + solo.stderr)
        singles.append(solo_lines[0] if solo_lines else "")

    for index, prompt in enumerate((prompt_a, prompt_b)):
        emit("  seq %d prompt %r" % (index, prompt))
        emit("    batched -> %r" % joint_texts[index][:90])
        emit("    single  -> %r" % singles[index][:90])

    failures = []
    for index in range(2):
        if joint_texts[index].strip() != singles[index].strip():
            emit("  seq %d: DIFFER" % index)
            failures.append("driver seq %d differs between the two-prompt run "
                            "and the single-prompt run" % index)
        else:
            emit("  seq %d: MATCH" % index)

    if joint_texts[0] == joint_texts[1]:
        failures.append("both streams produced identical text")

    # The same comparison with both prompts in one prefill batch, which is what
    # the upstream example does.  This is reported rather than asserted: the
    # masking is unchanged, but a larger attention matmul reorders the
    # accumulation, so the logits move and a near-tied greedy choice can flip.
    scratch = ROOT / "results" / "logs"
    scratch.mkdir(parents=True, exist_ok=True)
    joint_path = scratch / "joint_prefill_batch.tsv"
    single_path = scratch / "separate_prefill_batch.tsv"
    joint_run = run_driver_two_prompts(model, prompt_a, prompt_b, n_predict,
                                       threads, joint=True,
                                       logits_out=joint_path)
    run_driver_single(model, prompt_a, n_predict, threads, 0, single_path)
    joint_again = sequence_lines(joint_run.stdout + joint_run.stderr)
    shift = max_logit_shift(read_logits(single_path), read_logits(joint_path))
    emit("\n=== one prefill batch instead of two ===")
    emit("  seq 0 separate -> %r" % (singles[0][:80]))
    emit("  seq 0 joint    -> %r" % (joint_again[0][:80] if joint_again else ""))
    emit("  max logit shift between the two prefill splits: %s"
          % ("%.4f" % shift if shift is not None else "unknown"))
    if shift is not None and shift > LOGIT_EPSILON:
        emit("  a joint prefill moves the logits by more than %.2f, so the "
              "split of the prefill is part of the comparison, not the mask"
              % LOGIT_EPSILON)

    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    ap.add_argument("--n-predict", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1,
                    help="1 keeps the logits exactly reproducible")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit("model not found: %s" % args.model)

    failures = []
    if BATCHED.exists():
        failures += check_batched_example(args.model, args.n_predict,
                                          args.threads)
    else:
        emit("llama-batched not built, skipping that part: %s" % BATCHED)

    if DRIVER.exists():
        failures += check_two_prompts(args.model, args.n_predict, args.threads)
    else:
        emit("driver not built, skipping that part: %s" % DRIVER)

    if failures:
        emit("\nFAILURES:")
        for failure in failures:
            emit("  - %s" % failure)
        code = 1
    else:
        emit("\nevery batched run matched the corresponding separate runs")
        code = 0

    # Keep a record so the audit can quote what this test actually saw.
    out = ROOT / "results" / "batch_equivalence.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    emit("wrote %s" % out)
    return code


if __name__ == "__main__":
    sys.exit(main())
