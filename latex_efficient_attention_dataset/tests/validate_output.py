"""Structural validation of the generated JSON dataset.

This is the original basic checker, extended to the schema version produced by
``parser/parse_latex.py``.  The deeper fidelity and tokenizer regression suite
lives in ``tests/test_regression.py``.

Run with:  .\\.venv\\Scripts\\python.exe tests\\validate_output.py
"""
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
SCHEMA = ROOT / "schema" / "document_schema.json"
SOURCES = sorted(ROOT.glob("document_*.tex"))
JSON_FILES = sorted(OUTPUT.glob("document_*.json"))
MAX_TOKENS = 2000

errors = []
passed = 0


def check(condition, message):
    global passed
    if condition:
        passed += 1
        print(f"PASS: {message}")
    else:
        errors.append(message)
        print(f"FAIL: {message}")


if not SCHEMA.exists():
    raise SystemExit(f"Schema not found: {SCHEMA}")

schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
validator = Draft202012Validator(schema)

check(len(SOURCES) == 5, f"Found 5 source files (found {len(SOURCES)})")
check(len(JSON_FILES) == 5, f"Found 5 JSON files (found {len(JSON_FILES)})")

for path in JSON_FILES:
    print(f"\n--- Checking {path.name} ---")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{path.name}: invalid JSON: {exc}")
        print(f"FAIL: Invalid JSON: {exc}")
        continue

    schema_errors = list(validator.iter_errors(data))
    check(not schema_errors, f"{path.name}: JSON Schema validation")
    for error in schema_errors:
        print(f"  Schema error: {error.message}")

    nodes = data.get("nodes", [])
    meta = data.get("metadata", {})
    ids = [node.get("id") for node in nodes]
    id_set = set(ids)

    check(bool(nodes), f"{path.name}: contains nodes")
    check(len(ids) == len(id_set), f"{path.name}: node IDs are unique")
    check(bool(nodes) and nodes[0].get("id") == "n0", f"{path.name}: first node is root n0")

    if not nodes:
        continue

    check(nodes[0].get("parent") is None, f"{path.name}: root parent is null")

    assignment = meta.get("sequence_assignment", {})
    sequence_count = assignment.get("sequence_count", 1)
    valid_seqs = set(range(sequence_count))
    check(sequence_count >= 1, f"{path.name}: sequence_count >= 1")

    blocks = assignment.get("blocks", [])
    check(len(blocks) == sequence_count,
          f"{path.name}: one record per declared sequence ({len(blocks)})")
    check([b["seq_id"] for b in blocks] == list(range(sequence_count)),
          f"{path.name}: block seq_ids are 0..sequence_count-1")

    for node in nodes:
        node_id = node.get("id")
        parent = node.get("parent")

        if node_id != "n0":
            check(parent in id_set, f"{path.name}: parent exists for {node_id}")

        tokens = node.get("tokens", [])
        positions = node.get("positions", [])
        seq_ids = node.get("seq_ids", [])

        check(len(tokens) == len(positions) == len(seq_ids),
              f"{path.name}: aligned arrays for {node_id}")
        check(node.get("token_count") == len(tokens),
              f"{path.name}: token_count matches for {node_id}")
        check(all(isinstance(t, int) and t >= 0 for t in tokens),
              f"{path.name}: valid token IDs for {node_id}")
        check(all(isinstance(p, int) and p >= 0 for p in positions),
              f"{path.name}: valid positions for {node_id}")
        check(set(seq_ids) <= valid_seqs,
              f"{path.name}: seq_ids within the declared convention for {node_id}")
        check(isinstance(node.get("latex"), str) and isinstance(node.get("text"), str),
              f"{path.name}: latex and text present for {node_id}")
        span = node.get("source_span")
        check(isinstance(span, list) and len(span) == 2 and span[0] <= span[1],
              f"{path.name}: valid source_span for {node_id}")
        for child in node.get("child_ids", []):
            check(child in id_set, f"{path.name}: child {child} of {node_id} exists")

    all_positions = [p for node in nodes for p in node.get("positions", [])]
    check(all_positions == sorted(all_positions), f"{path.name}: positions are ordered")
    check(len(all_positions) == len(set(all_positions)), f"{path.name}: positions are unique")
    check(all_positions == list(range(len(all_positions))),
          f"{path.name}: positions are packed 0..N-1")

    token_count = sum(len(n.get("tokens", [])) for n in nodes)
    check(token_count <= MAX_TOKENS,
          f"{path.name}: token count <= {MAX_TOKENS} ({token_count})")

    check(meta.get("node_count") == len(nodes), f"{path.name}: metadata node_count")
    check(meta.get("tokenization", {}).get("total_tokens") == token_count,
          f"{path.name}: metadata total_tokens")
    check(meta.get("content_loss_audit", {}).get("lost") == [],
          f"{path.name}: no control sequence lost during extraction")

    total_positions = [p for node in nodes for p in node.get("positions", [])]
    for block in blocks:
        owned = [p for node in nodes if block["seq_id"] in node.get("seq_ids", [])
                 for p in node.get("positions", [])]
        check(block["pos_min"] == min(owned, default=block["pos_min"])
              and block["pos_max"] == max(owned, default=block["pos_max"]),
              f"{path.name}: seq {block['seq_id']} covers positions "
              f"[{block['pos_min']}, {block['pos_max']}]")
        check(block["pos_min"] < total_positions[-1],
              f"{path.name}: seq {block['seq_id']} range is non-empty")

    check(all(isinstance(n.get("text"), str) for n in nodes),
          f"{path.name}: every node has text")

    type_counts = {}
    for node in nodes:
        type_counts[node["node_type"]] = type_counts.get(node["node_type"], 0) + 1
    print(f"Summary: {len(nodes)} nodes, {token_count} tokens, "
          f"{sum(bool(n.get('text')) for n in nodes)} nonempty nodes")
    print(f"Types: {json.dumps(type_counts, sort_keys=True)}")
    print(f"Sequences: {sequence_count} ({assignment.get('mode')}, "
          f"xref_links={assignment.get('xref_links')})")

print(f"\nChecks passed: {passed}")
print(f"Checks failed: {len(errors)}")

if errors:
    for error in errors:
        print(f"  - {error}")
    raise SystemExit(1)

print("Basic output validation completed.")
