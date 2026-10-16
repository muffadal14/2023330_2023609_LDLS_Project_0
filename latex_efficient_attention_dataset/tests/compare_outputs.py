"""Compare two output directories and report the meaningful differences.

Reports node/token count deltas, node-type count deltas, content-word recovery
and math-symbol retention, plus per-node text diffs.

Both arguments may be relative to the project root or absolute paths.  A
baseline is not kept inside the project, so point the first argument at any
directory holding the previous outputs (for example a copy taken with
``parse_latex.py --backup-dir``).

Run with:
    .\\.venv\\Scripts\\python.exe tests\\compare_outputs.py C:\\temp\\baseline_output output
"""
from __future__ import annotations

import difflib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def resolve(directory):
    path = pathlib.Path(directory)
    return path if path.is_absolute() else ROOT / path


def load(directory):
    base = resolve(directory)
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(base.glob("document_*.json"))}


def summarise(label, data):
    print(f"\n=== {label} ===")
    total_nodes = total_tokens = 0
    for name, document in sorted(data.items()):
        nodes = document["nodes"]
        tokens = sum(len(n["tokens"]) for n in nodes)
        total_nodes += len(nodes)
        total_tokens += tokens
        counts = document.get("metadata", {}).get("node_type_counts") or {}
        if not counts:
            counts = {}
            for node in nodes:
                counts[node["node_type"]] = counts.get(node["node_type"], 0) + 1
        print(f"{name}: nodes={len(nodes)} tokens={tokens} types={json.dumps(counts, sort_keys=True)}")
    print(f"TOTAL: nodes={total_nodes} tokens={total_tokens}")
    return total_nodes, total_tokens


def diff_texts(old, new):
    old_nodes = {n["id"]: n for n in old["nodes"]}
    new_nodes = {n["id"]: n for n in new["nodes"]}
    print("\n  per-node text differences:")
    shown = 0
    for node_id in sorted(set(old_nodes) | set(new_nodes),
                          key=lambda i: int(i[1:])):
        a = old_nodes.get(node_id)
        b = new_nodes.get(node_id)
        if a is None:
            print(f"    + {node_id} added as {b['node_type']}: {b['text'][:90]!r}")
            shown += 1
            continue
        if b is None:
            print(f"    - {node_id} removed (was {a['node_type']}): {a['text'][:90]!r}")
            shown += 1
            continue
        if a["node_type"] != b["node_type"]:
            print(f"    ~ {node_id} type {a['node_type']} -> {b['node_type']}")
            shown += 1
        if a["text"] != b["text"]:
            print(f"    ~ {node_id} text:")
            for line in difflib.unified_diff(
                    [a["text"] + "\n"], [b["text"] + "\n"],
                    fromfile="old", tofile="new", n=0):
                if line.startswith(("---", "+++", "@@")):
                    continue
                print("        " + line.rstrip("\n")[:200])
            shown += 1
        if shown > 60:
            print("    ... (truncated)")
            break


WORD_RE = re.compile(r"[A-Za-z]{3,}")
MATH_GLYPHS = set("≤≥≠≈≡∼→←⇒⇔×÷±∓∈∉∪∩∧∨∀∃¬∑∏∫∞εαβγδθλμσπφψωℝℕℤℚℂℓ")


def source_words(name):
    """Content words of the .tex source, with LaTeX command names removed."""
    source = (ROOT / (name + ".tex")).read_text(encoding="utf-8")
    source = re.sub(r"(?<!\\)%.*", " ", source)
    source = re.sub(r"\\[A-Za-z@]+\*?", " ", source)
    return {w.lower() for w in WORD_RE.findall(source)}


def extracted_words(document):
    blob = " ".join(n["text"] for n in document["nodes"])
    words = {w.lower() for w in WORD_RE.findall(blob)}
    words |= {c for c in blob if c in MATH_GLYPHS}
    return words


def compare_coverage(old, new):
    """How much of the source's vocabulary actually survives extraction."""
    print("\n" + "=" * 70)
    print("CONTENT RECOVERY (source words present in extracted text)")
    print("=" * 70)
    for name in sorted(set(old) & set(new)):
        wanted = source_words(name)
        before = extracted_words(old[name])
        after = extracted_words(new[name])
        kept_before = wanted & before
        kept_after = wanted & after
        recovered = sorted(kept_after - kept_before)
        lost = sorted(kept_before - kept_after)
        print(f"\n{name}: {len(kept_before)}/{len(wanted)} "
              f"({100.0 * len(kept_before) / max(len(wanted), 1):.1f}%) -> "
              f"{len(kept_after)}/{len(wanted)} "
              f"({100.0 * len(kept_after) / max(len(wanted), 1):.1f}%)")
        if recovered:
            print(f"  recovered  : {' '.join(recovered)}")
        if lost:
            print(f"  lost       : {' '.join(lost)}")
        missing = sorted(wanted - kept_after)
        if missing:
            print(f"  still missing ({len(missing)}): {' '.join(missing[:60])}")


MATH_GLYPHS = "≠≤≥≈≡∼→←⇒⇔×÷±∓∈∉⊂⊆⊃⊇∪∩∧∨∀∃¬∑∏∫∞ℓℝℕℤℚℂεαβγδθλμσπφψω√⊤‖…𝒞"
RAW_CMD_RE = re.compile(r"\\([A-Za-z@]+)")


def glyph_inventory(document):
    blob = "".join(n["text"] for n in document["nodes"])
    return {g: blob.count(g) for g in MATH_GLYPHS if g in blob}


def surviving_raw_commands(name, document):
    """Distinct source commands that still appear unrendered in the text."""
    source = (ROOT / (name + ".tex")).read_text(encoding="utf-8")
    source = re.sub(r"(?<!\\)%.*", " ", source)
    begin = source.find(r"\begin{document}")
    if begin != -1:
        end = source.rfind(r"\end{document}")
        source = source[begin:end]
    in_source = {c for c in RAW_CMD_RE.findall(source)} - {"begin", "end"}
    blob = " ".join(n["text"] for n in document["nodes"])
    return sorted(c for c in in_source if ("\\" + c) in blob)


def compare_symbols(old, new):
    print("\n" + "=" * 70)
    print("MATH SYMBOL RETENTION  (glyphs in extracted text / raw commands left)")
    print("=" * 70)
    print("Note: a glyph count that DROPS here is usually a fix, not a regression -")
    print("the previous renderer replaced commands without a word boundary, so")
    print("\\frac -> '<=ft', \\gets -> '>=ts', \\ell -> '<=', \\top -> '->p'.")
    for name in sorted(set(old) & set(new)):
        before, after = glyph_inventory(old[name]), glyph_inventory(new[name])
        keys = sorted(set(before) | set(after))
        gains = {k: (before.get(k, 0), after.get(k, 0))
                 for k in keys if before.get(k, 0) != after.get(k, 0)}
        print(f"\n{name}: glyphs {sum(before.values())} -> {sum(after.values())}"
              f" | unrendered commands {len(surviving_raw_commands(name, old[name]))}"
              f" -> {len(surviving_raw_commands(name, new[name]))}")
        for glyph, (was, now) in gains.items():
            print(f"  {'+' if now > was else '-'} {glyph}: {was} -> {now}")
        leftover = surviving_raw_commands(name, new[name])
        if leftover:
            print(f"  unrendered: {' '.join(leftover)}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    old_dir, new_dir = sys.argv[1], sys.argv[2]
    if not resolve(old_dir).is_dir():
        print(f"baseline directory not found: {resolve(old_dir)}", file=sys.stderr)
        raise SystemExit(2)
    old, new = load(old_dir), load(new_dir)
    if not old:
        print(f"no document_*.json found in {resolve(old_dir)}", file=sys.stderr)
        raise SystemExit(2)

    summarise(old_dir, old)
    summarise(new_dir, new)
    compare_coverage(old, new)
    compare_symbols(old, new)

    print("\n" + "=" * 70)
    print("NODE-LEVEL DIFFERENCES (node ids are positional, so shifts are expected)")
    print("=" * 70)
    for name in sorted(set(old) | set(new)):
        if name not in old:
            print(f"\n{name}: only in {new_dir}")
            continue
        if name not in new:
            print(f"\n{name}: only in {old_dir}")
            continue
        print(f"\n{name}:")
        diff_texts(old[name], new[name])


if __name__ == "__main__":
    main()
