#!/usr/bin/env python3
"""Automated regression tests for the LaTeX parser and the generated dataset.

Run with either of:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
    .\\.venv\\Scripts\\python.exe tests\\test_regression.py

The suite covers four layers:

1. structure      - tree invariants, packed positions, source-span containment
2. fidelity       - mathematical operators, relations, subscripts and equations
                     survive extraction, and nothing is silently deleted
3. classification - theorem / proposition / lemma / definition / proof /
                     equation / align / cases / algorithm / algorithmic mapping
4. tokenizer      - token IDs reproduce exactly under the documented scheme

The tokenizer tests need ``llama-cpp-python`` and the GGUF model; they are
skipped (not silently passed) when the environment cannot provide them.
"""
from __future__ import annotations

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "parser"))

import parse_latex as P  # noqa: E402

SOURCES = sorted(ROOT.glob("document_*.tex"))
OUTPUT_DIR = ROOT / "output"
SCHEMA_PATH = ROOT / "schema" / "document_schema.json"
MODEL_PATH = pathlib.Path(
    r"C:\Users\muffa\LDLS Project 0\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf")

MAX_TOKENS = 2000

# Commands that carry mathematical content.  They must be *rendered*, never
# survive as raw "\cmd" in a node's text, because a raw copy means the node was
# not understood.  Commands quoted verbatim on purpose (\ref, \begin, \section
# inside \verb|...|, and \frac which has no Unicode equivalent) are excluded and
# handled by the "preserved verbatim" tests instead.
MATH_COMMANDS = [
    "in", "notin", "geq", "leq", "ne", "implies", "iff", "max", "min",
    "varepsilon", "epsilon", "alpha", "beta", "gamma", "lambda", "sigma",
    "times", "to", "leftarrow", "gets", "top", "sqrt", "mathbb", "mathcal",
    "cup", "cap", "sum", "prod", "int", "infty", "partial", "cdot", "ldots",
    "operatorname", "left", "right", "forall", "exists", "subseteq",
    "frac", "dfrac", "tfrac", "cfrac",
]

# Commands quoted verbatim inside \verb|...| are content the author wrote
# literally, so they must NOT be reported as unresolved.  Any command that also
# occurs outside a \verb body is still reported (see
# TestVerbatimIsContent.test_verb_quoted_command_is_still_reported_outside_verb).
ALLOWED_VERBATIM_COMMANDS: set = set()


def load_outputs():
    data = {}
    for path in sorted(OUTPUT_DIR.glob("document_*.json")):
        data[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return data


def node_by_type(document, node_type):
    return [n for n in document["nodes"] if n["node_type"] == node_type]


def node_texts(document):
    return [n["text"] for n in document["nodes"]]


def all_text(document):
    return "\n".join(node_texts(document))


OUTPUTS = load_outputs()


def sequence_count(document):
    """Number of logical sequences the document declares."""
    return document["metadata"].get("sequence_assignment", {}).get("sequence_count", 1)


def sequence_mode(document):
    return document["metadata"].get("sequence_assignment", {}).get("mode", "single")


class BaseCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if len(SOURCES) != 5:
            raise AssertionError("expected 5 .tex sources, found %d" % len(SOURCES))
        if len(OUTPUTS) != 5:
            raise AssertionError("expected 5 output JSON files in %s, found %d"
                                 % (OUTPUT_DIR, len(OUTPUTS)))


# ---------------------------------------------------------------------------
# 1. Structure and metadata
# ---------------------------------------------------------------------------
class TestCorpusShape(BaseCase):
    def test_five_sources_and_five_outputs(self):
        self.assertEqual(len(SOURCES), 5)
        self.assertEqual(len(OUTPUTS), 5)
        for source in SOURCES:
            self.assertIn(source.stem, OUTPUTS)

    def test_schema_validates_every_document(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        try:
            import jsonschema
        except ImportError:  # pragma: no cover
            self.skipTest("jsonschema not installed")
        validator = jsonschema.Draft202012Validator(schema)
        for name, document in sorted(OUTPUTS.items()):
            errors = sorted(validator.iter_errors(document), key=lambda e: e.path)
            self.assertEqual(errors, [], "%s: %s" % (name, [e.message for e in errors]))

    def test_root_and_ids(self):
        for name, document in sorted(OUTPUTS.items()):
            nodes = document["nodes"]
            self.assertEqual(nodes[0]["id"], "n0", name)
            self.assertEqual(nodes[0]["node_type"], "document", name)
            self.assertIsNone(nodes[0]["parent"], name)
            ids = [n["id"] for n in nodes]
            self.assertEqual(len(ids), len(set(ids)), "%s: duplicate ids" % name)
            lookup = {n["id"]: n for n in nodes}
            for node in nodes:
                if node["id"] == "n0":
                    continue
                self.assertIn(node["parent"], lookup, "%s: %s" % (name, node["id"]))
                # no cycles
                seen, cursor = set(), node
                while cursor["parent"] is not None:
                    self.assertNotIn(cursor["id"], seen)
                    seen.add(cursor["id"])
                    cursor = lookup[cursor["parent"]]
                # child_ids agree with parent links
                for child in node["child_ids"]:
                    self.assertIn(child, lookup)
                    self.assertEqual(lookup[child]["parent"], node["id"])

    def test_nodes_are_in_document_order(self):
        """Pre-order traversal must follow the source, so packed positions do too."""
        for name, document in sorted(OUTPUTS.items()):
            lookup = {n["id"]: n for n in document["nodes"]}
            spans = [tuple(n["source_span"]) for n in document["nodes"] if not n.get("is_heading")]
            for earlier, later in zip(spans, spans[1:]):
                self.assertLessEqual(earlier[0], later[0],
                                     "%s: nodes out of source order" % name)

    def test_source_spans_point_at_real_source(self):
        for name, document in sorted(OUTPUTS.items()):
            source = (ROOT / (name + ".tex")).read_text(encoding="utf-8")
            lookup = {n["id"]: n for n in document["nodes"]}
            for node in document["nodes"]:
                start, end = node["source_span"]
                self.assertLessEqual(start, end, "%s/%s" % (name, node["id"]))
                self.assertLessEqual(end, len(source), "%s/%s" % (name, node["id"]))
                parent = lookup[node["parent"]] if node["parent"] else None
                if parent is not None:
                    self.assertGreaterEqual(start, parent["source_span"][0])
                    self.assertLessEqual(end, parent["source_span"][1],
                                         "%s/%s escapes parent" % (name, node["id"]))
                # Every chunk of LaTeX a node claims to own must really occur
                # inside that node's span in the original file.
                for chunk in node["latex"].split("\n"):
                    self.assertIn(chunk, source[start:end],
                                  "%s/%s: latex not found at its span"
                                  % (name, node["id"]))

    def test_metadata_is_self_consistent(self):
        for name, document in sorted(OUTPUTS.items()):
            meta = document["metadata"]
            source = (ROOT / (name + ".tex")).read_text(encoding="utf-8")
            self.assertEqual(meta["source_file"], name + ".tex")
            self.assertEqual(meta["source_length"], len(source), name)
            self.assertEqual(meta["node_count"], len(document["nodes"]), name)
            self.assertEqual(meta["nonempty_node_count"],
                             sum(1 for n in document["nodes"] if n["text"].strip()), name)
            counts = {}
            for node in document["nodes"]:
                counts[node["node_type"]] = counts.get(node["node_type"], 0) + 1
            self.assertEqual(meta["node_type_counts"], counts, name)
            self.assertEqual(meta["tokenization"]["total_tokens"],
                             sum(len(n["tokens"]) for n in document["nodes"]), name)


# ---------------------------------------------------------------------------
# 2. Token / position / seq_id consistency
# ---------------------------------------------------------------------------
class TestTokenMetadata(BaseCase):
    def test_parallel_arrays_and_packed_positions(self):
        for name, document in sorted(OUTPUTS.items()):
            positions = []
            for node in document["nodes"]:
                length = len(node["tokens"])
                self.assertEqual(length, len(node["positions"]), "%s/%s" % (name, node["id"]))
                self.assertEqual(length, len(node["seq_ids"]), "%s/%s" % (name, node["id"]))
                self.assertEqual(node["token_count"], length, "%s/%s" % (name, node["id"]))
                self.assertTrue(all(isinstance(t, int) and t >= 0 for t in node["tokens"]))
                count = sequence_count(document)
                self.assertEqual(set(node["seq_ids"]) - set(range(count)), set(),
                                 "%s/%s: seq_ids outside the declared convention"
                                 % (name, node["id"]))
                positions.extend(node["positions"])
            self.assertEqual(positions, list(range(len(positions))),
                             "%s: positions are not packed 0..N-1" % name)
            self.assertEqual(len(positions), len(set(positions)), name)

    def test_token_counts_are_real_tokenizer_counts(self):
        """Re-tokenize every node and require an exact match with the stored IDs."""
        try:
            from llama_cpp import Llama
        except ImportError:  # pragma: no cover
            self.skipTest("llama-cpp-python not installed in this interpreter")
        if not MODEL_PATH.exists():
            self.skipTest("GGUF model not available: %s" % MODEL_PATH)
        llm = Llama(model_path=str(MODEL_PATH), vocab_only=True, n_ctx=512, verbose=False)
        try:
            for name, document in sorted(OUTPUTS.items()):
                total = 0
                for node in document["nodes"]:
                    expected = ([int(t) for t in
                                 llm.tokenize(node["text"].encode("utf-8"),
                                              add_bos=False, special=True)]
                                if node["text"] else [])
                    self.assertEqual(node["tokens"], expected,
                                     "%s/%s: stored tokens do not reproduce"
                                     % (name, node["id"]))
                    total += len(expected)
                self.assertEqual(total,
                                 document["metadata"]["tokenization"]["total_tokens"], name)
                self.assertLessEqual(total, MAX_TOKENS,
                                     "%s: %d tokens exceeds the %d limit"
                                     % (name, total, MAX_TOKENS))
        finally:
            del llm

    def test_declared_scheme_matches_metadata(self):
        for document in OUTPUTS.values():
            tok = document["metadata"]["tokenization"]
            self.assertEqual(tok["granularity"], "per_node")
            self.assertEqual(tok["input_field"], "text")
            self.assertFalse(tok["add_bos"])
            self.assertTrue(tok["special"])
            self.assertIn("model tokenizer", tok["kind"])
            self.assertEqual(document["metadata"]["position_convention"], "packed_global")
            mode = sequence_mode(document)
            self.assertIn(mode, ("single", "blocks"))
            self.assertEqual(
                document["metadata"]["sequence_convention"],
                "single_sequence_per_document" if mode == "single"
                else "one_sequence_per_top_level_block")
            self.assertGreaterEqual(sequence_count(document), 1)


# ---------------------------------------------------------------------------
# 2b. Structured sequence assignment
# ---------------------------------------------------------------------------
class TestSequenceAssignment(BaseCase):
    """Tests for the per-block seq_ids used by the structured-prefill driver.

    These build documents on the fly rather than reading ``output/``, so they
    cover both conventions regardless of which one was last generated.
    """

    SOURCE = (r"\documentclass{article}"
              r"\begin{document}"
              r"\begin{abstract}Short abstract body here.\end{abstract}"
              r"\section{First}"
              r"Body of the first section with several words in it."
              r"\subsection{Nested}"
              r"Body of the nested subsection."
              r"\label{sec:nested}"
              r"\section{Second}"
              r"Body of the second section, which cites \ref{sec:nested}."
              r"\end{document}")

    def build(self, **kwargs):
        """Parse the synthetic source and assign sequences to it.

        The tests do not load the GGUF model, so each node is given a
        deterministic stand-in token count (one token per word) and packed
        positions.  Sequence assignment only reads positions and token counts,
        so this exercises the real code path.
        """
        title, nodes, _ = P.extract_structure(P.strip_comments(self.SOURCE),
                                              "synthetic")
        position = 0
        for node in nodes:
            count = max(1, len(node["text"].split()))
            node["tokens"] = list(range(position, position + count))
            node["positions"] = list(range(position, position + count))
            node["token_count"] = count
            position += count
        assignment = P.assign_sequences(nodes, **kwargs)
        return nodes, {n["id"]: n for n in nodes}, assignment

    def test_single_mode_puts_every_token_in_sequence_zero(self):
        nodes, _, assignment = self.build(mode="single")
        seq_ids = [s for node in nodes for s in node["seq_ids"]]
        self.assertTrue(seq_ids)
        self.assertEqual(set(seq_ids), {0})

        # The one record must describe the whole document, not just the title.
        record = assignment["blocks"][0]
        self.assertEqual(record["seq_id"], 0)
        positions = [p for node in nodes for p in node["positions"]]
        self.assertEqual(record["pos_min"], min(positions))
        self.assertEqual(record["pos_max"], max(positions))
        self.assertEqual(record["token_count"], len(positions))

    def test_blocks_mode_gives_each_top_level_block_its_own_sequence(self):
        nodes, lookup, _ = self.build(mode="blocks")
        blocks = [lookup["n1"], lookup["n4"]]
        sequences = set()
        for block in blocks:
            members = P._subtree(lookup, block)
            for member in members:
                sequences.add(member["seq_ids"][0])
        self.assertEqual(len(sequences), len(blocks))

    def test_a_block_only_covers_its_own_subtree(self):
        nodes, lookup, _ = self.build(mode="blocks")
        first = lookup["n1"]
        own = set()
        for member in P._subtree(lookup, first):
            own.update(member["positions"])
        for node in nodes:
            for position, seq in zip(node["positions"], node["seq_ids"]):
                if seq == first["seq_ids"][0]:
                    self.assertIn(position, own)

    def test_every_token_carries_at_least_one_sequence(self):
        for mode in ("single", "blocks"):
            nodes, _, _ = self.build(mode=mode)
            for node in nodes:
                self.assertEqual(len(node["seq_ids"]), len(node["positions"]), mode)
                self.assertTrue(all(isinstance(s, int) and s >= 0
                                    for s in node["seq_ids"]), mode)

    def test_small_blocks_are_merged_when_below_the_threshold(self):
        # Every section is below the threshold, so sections must be folded
        # together instead of each keeping its own sequence.
        _, _, assignment = self.build(mode="blocks", min_block_tokens=1000)
        records = {r["seq_id"]: r for r in assignment["blocks"]}
        self.assertTrue(any(record["merged"] for record in records.values()))
        self.assertLess(assignment["sequence_count"], 4)

    def test_sequence_budget_forces_adjacent_blocks_to_merge(self):
        _, _, assignment = self.build(mode="blocks", max_sequences=3)
        self.assertLessEqual(assignment["sequence_count"], 3)
        records = sorted(assignment["blocks"], key=lambda r: r["pos_min"])
        # Every sequence must own one contiguous position range, because that
        # is what makes a single seq+position eviction possible.
        previous = -1
        for record in records:
            self.assertGreater(record["pos_min"], previous,
                               "sequence ranges must be disjoint and ordered")
            previous = record["pos_max"]

    def test_block_records_carry_their_position_range(self):
        _, _, assignment = self.build(mode="blocks")
        for record in assignment["blocks"]:
            self.assertLessEqual(record["pos_min"], record["pos_max"])

    def test_recorded_ranges_match_the_tokens_that_carry_the_sequence(self):
        nodes, _, assignment = self.build(mode="blocks")
        for record in assignment["blocks"]:
            owned = [p for node in nodes
                     if record["seq_id"] in node["seq_ids"]
                     for p in node["positions"]]
            self.assertEqual(record["pos_min"], min(owned), record["seq_id"])
            self.assertEqual(record["pos_max"], max(owned), record["seq_id"])
            self.assertEqual(record["token_count"], len(owned), record["seq_id"])

    def test_xref_links_are_recorded_for_the_caller_to_resolve(self):
        _, _, assignment = self.build(mode="blocks", xref_links=True)
        self.assertTrue(assignment["cross_references"],
                        "the synthetic source cites sec:nested")
        for link in assignment["cross_references"]:
            self.assertNotEqual(link["from_seq"], link["target_seq"])
            self.assertTrue(link["label"])

    def test_a_token_belongs_to_exactly_one_sequence(self):
        """Giving a citing token a second seq_id would break the cache.

        llama.cpp rejects a sequence whose positions are not contiguous, and a
        citing node usually sits in the middle of another block's range.  The
        parser therefore records the link and leaves seq_ids single valued.
        """
        for xref in (True, False):
            nodes, _, _ = self.build(mode="blocks", xref_links=xref)
            for node in nodes:
                self.assertNotIn("extra_seq_ids", node,
                                 "extra memberships are not emitted (%s)" % xref)
            self.assertEqual(len({s for node in nodes
                                  for s in node["seq_ids"]}) > 0, True)

    def test_every_sequence_owns_a_contiguous_position_range(self):
        """The invariant the driver relies on when it copies and evicts."""
        for xref in (True, False):
            nodes, _, _ = self.build(mode="blocks", xref_links=xref)
            owned = {}
            for node in nodes:
                for position, seq in zip(node["positions"], node["seq_ids"]):
                    owned.setdefault(seq, []).append(position)
            self.assertTrue(owned)
            for seq, positions in owned.items():
                ordered = sorted(positions)
                self.assertEqual(ordered,
                                 list(range(ordered[0], ordered[0] + len(ordered))),
                                 "sequence %s is not contiguous" % seq)

    def test_validate_tree_rejects_a_non_contiguous_sequence(self):
        nodes, _, _ = self.build(mode="blocks")
        lookup = {n["id"]: n for n in nodes}
        victim = [n for n in nodes if n["id"] != "n0"][0]
        victim["seq_ids"] = [lookup["n0"]["seq_ids"][0]] * len(victim["positions"])
        with self.assertRaises(ValueError):
            P.validate_tree(nodes, None, 2)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.build(mode="nonsense")


# ---------------------------------------------------------------------------
# 3. Environment classification
# ---------------------------------------------------------------------------
class TestEnvironmentClassification(BaseCase):
    def test_statement_environments(self):
        doc = OUTPUTS["document_3_mathematical_results"]
        counts = doc["metadata"]["node_type_counts"]
        self.assertEqual(counts.get("theorem"), 1)
        self.assertEqual(counts.get("proposition"), 1)
        self.assertEqual(counts.get("lemma"), 2)
        self.assertEqual(counts.get("definition"), 2)
        self.assertEqual(counts.get("proof"), 4)

    def test_proposition_is_not_labelled_theorem(self):
        doc = OUTPUTS["document_3_mathematical_results"]
        propositions = node_by_type(doc, "proposition")
        self.assertEqual(len(propositions), 1)
        proposition = propositions[0]
        self.assertEqual(proposition["environment"], "proposition")
        self.assertIn("cx_n", proposition["text"])
        self.assertNotEqual(proposition["node_type"], "theorem")

    def test_proof_follows_the_proposition_it_closes(self):
        """'This proves the proposition.' must be a proof after a proposition."""
        doc = OUTPUTS["document_3_mathematical_results"]
        siblings = {}
        for node in doc["nodes"]:
            siblings.setdefault(node["parent"], []).append(node)
        closing = [n for n in doc["nodes"]
                   if "This proves the proposition" in n["text"]]
        self.assertEqual(len(closing), 1)
        proof = closing[0]
        self.assertEqual(proof["node_type"], "proof")
        self.assertEqual(proof["environment"], "proof")
        previous = siblings[proof["parent"]][siblings[proof["parent"]].index(proof) - 1]
        self.assertEqual(previous["node_type"], "proposition")

    def test_math_environments(self):
        doc1 = OUTPUTS["document_1_transformer_architecture"]
        self.assertEqual(len(node_by_type(doc1, "align")), 1)
        self.assertEqual(len(node_by_type(doc1, "equation")), 3)
        self.assertEqual(node_by_type(doc1, "align")[0]["environment"], "align")

        doc3 = OUTPUTS["document_3_mathematical_results"]
        self.assertEqual(len(node_by_type(doc3, "align")), 1)
        # \begin{equation} plus five \[ ... \] display blocks
        self.assertEqual(len(node_by_type(doc3, "equation")), 6)

        doc4 = OUTPUTS["document_4_structured_documents"]
        self.assertEqual(len(node_by_type(doc4, "equation")), 3)

        doc5 = OUTPUTS["document_5_attention_masks"]
        self.assertEqual(len(node_by_type(doc5, "equation")), 4)

    def test_cases_environments_are_not_generic_other(self):
        doc = OUTPUTS["document_5_attention_masks"]
        cases = node_by_type(doc, "cases")
        self.assertEqual(len(cases), 2)
        for node in cases:
            self.assertEqual(node["environment"], "cases")
            self.assertNotEqual(node["node_type"], "other")
            # a cases block lives inside the equation that uses it
            parent = next(n for n in doc["nodes"] if n["id"] == node["parent"])
            self.assertEqual(parent["node_type"], "equation")

    def test_algorithm_environments_are_not_generic_other(self):
        doc = OUTPUTS["document_4_structured_documents"]
        algorithms = node_by_type(doc, "algorithm")
        listings = node_by_type(doc, "algorithmic")
        self.assertEqual(len(algorithms), 1)
        self.assertEqual(len(listings), 1)
        self.assertEqual(algorithms[0]["environment"], "algorithm")
        self.assertEqual(listings[0]["environment"], "algorithmic")
        self.assertEqual(listings[0]["parent"], algorithms[0]["id"])
        self.assertEqual(algorithms[0].get("caption"),
                         "Collect tokens for a selected subtree")

    def test_generic_other_is_only_used_when_warranted(self):
        """Every remaining 'other' node must still carry its environment name."""
        for name, document in sorted(OUTPUTS.items()):
            for node in node_by_type(document, "other"):
                self.assertTrue(node.get("environment"),
                                "%s/%s: 'other' node lost its environment name"
                                % (name, node["id"]))

    def test_section_hierarchy(self):
        doc = OUTPUTS["document_4_structured_documents"]
        sections = node_by_type(doc, "section")
        subsections = node_by_type(doc, "subsection")
        self.assertEqual(len(sections), 12)
        self.assertEqual(len(subsections), 3)
        for node in sections + subsections:
            self.assertTrue(node.get("is_heading"), node["id"])
        # every subsection nests under a section
        section_ids = {n["id"] for n in sections}
        for node in subsections:
            self.assertIn(node["parent"], section_ids)


# ---------------------------------------------------------------------------
# 4. Mathematical fidelity (the confirmed Document 3 defects)
# ---------------------------------------------------------------------------
class TestMathFidelityDocument3(BaseCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.doc = OUTPUTS["document_3_mathematical_results"]
        cls.text = all_text(cls.doc)

    def test_neq_is_not_reduced_to_nothing(self):
        """Regression: "c\\ne 0" used to come out as "c 0"."""
        self.assertNotIn("c 0", self.text)
        proofs = node_by_type(self.doc, "proof")
        closing = next(n for n in proofs if "This proves the proposition" in n["text"])
        self.assertIn("c ≠ 0", closing["text"])

    def test_implication_survives(self):
        # doc 3 contains exactly three \implies: the convergence definition
        # and the two epsilon/2 display equations.
        self.assertEqual(self.text.count("⇒"), 3,
                         "\\implies/\\Rightarrow must not be deleted")

    def test_max_survives_in_the_sum_theorem_proof(self):
        proofs = node_by_type(self.doc, "proof")
        sums = next(n for n in proofs if "Let N = max" in n["text"])
        self.assertIn("max(N_1, N_2)", sums["text"])
        # the word "maximum" elsewhere must not be mistaken for the operator
        self.assertNotIn("max(N", next(
            n["text"] for n in proofs if "finite maximum" in n["text"]))

    def test_definition_keeps_its_conditions(self):
        definition = node_by_type(self.doc, "definition")[0]
        self.assertIn("(x_n)_{n ≥ 1}", definition["text"])
        self.assertIn("∈ ℝ", definition["text"])
        self.assertIn("ε > 0", definition["text"])
        equation = next(n for n in self.doc["nodes"]
                        if n["parent"] == definition["id"] and n["node_type"] == "equation")
        self.assertIn("n ≥ N ⇒ |x_n - L| < ε", equation["text"])

    def test_display_equations_are_their_own_nodes(self):
        displays = [n for n in node_by_type(self.doc, "equation")
                    if n.get("environment") is None]
        self.assertEqual(len(displays), 5)
        joined = " ".join(n["text"] for n in displays)
        self.assertIn("|x_n - c| = |c - c| = 0 < ε", joined)
        self.assertIn("ε / 2", joined)

    def test_align_rows_and_implication_chain(self):
        align = node_by_type(self.doc, "align")[0]
        self.assertIn("&", align["text"])
        self.assertGreaterEqual(align["text"].count("\n"), 2)
        self.assertIn("(x_n - a) + (y_n - b)", align["text"])

    def test_cross_reference_is_preserved_and_recorded(self):
        refs = [n for n in self.doc["nodes"] if n["references"]]
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["references"], ["lem:bounded"])
        self.assertIn("\\ref{lem:bounded}", refs[0]["text"])

    def test_no_math_command_survives_raw(self):
        for command in MATH_COMMANDS:
            self.assertNotIn("\\" + command + " ", self.text,
                             "\\%s leaked into text" % command)


class TestMathFidelityAllDocuments(BaseCase):
    def test_operators_and_scripts_are_rendered(self):
        expectations = {
            "document_1_transformer_architecture": ["∈ ℝ^{n × d}", "softmax",
                                                     "Attention(Q, K, V) = softmax ((QK^⊤) / (√{d_k}) + M)V",
                                                     "Concat(H_1, …, H_h)W_O",
                                                     "Q & = XW_Q"],
            "document_2_kv_cache_management": ["𝒞_n = {(K_t^ℓ, V_t^ℓ): 1 ≤ t ≤ n",
                                               "M(n) = Lnm."],
            "document_3_mathematical_results": ["≠", "⇒", "max(N_1, N_2)", "≤"],
            "document_4_structured_documents": ["S(v) = ∑_{u ∈ Desc(v) ∪ {v}} |tokens_u|",
                                                "v = (id, parent, type, tokens, positions, seq_ids)"],
            "document_5_attention_masks": ["S = (QK^⊤) / (√{d_k})",
                                           "A = softmax(S + M)",
                                           "M_{ij} =", "0, & s_i = s_j and p_j ≤ p_i"],
        }
        for name, needles in expectations.items():
            text = all_text(OUTPUTS[name])
            for needle in needles:
                self.assertIn(needle, text, "%s: missing %r" % (name, needle))

    def test_no_math_command_survives_raw_anywhere(self):
        for name, document in sorted(OUTPUTS.items()):
            text = all_text(document)
            for command in MATH_COMMANDS:
                self.assertNotIn("\\" + command + " ", text,
                                 "%s: \\%s leaked into text" % (name, command))

    def test_only_allow_listed_commands_survive_raw(self):
        for name, document in sorted(OUTPUTS.items()):
            reported = set(document["metadata"]["unresolved_commands"]["commands"])
            self.assertTrue(reported <= ALLOWED_VERBATIM_COMMANDS,
                            "%s: unexpected unresolved commands %s"
                            % (name, sorted(reported - ALLOWED_VERBATIM_COMMANDS)))
            self.assertEqual(reported,
                             {c for n in document["nodes"]
                              for c in n["unresolved_commands"]},
                            "%s: per-node and metadata unresolved lists disagree" % name)
            for command in reported:
                self.assertIn("\\" + command, all_text(document),
                              "%s: %s reported unresolved but not present"
                              % (name, command))
            self.assertEqual(reported, set(),
                             "%s: nothing in this corpus is genuinely unrendered; "
                             "quoted \\verb examples must not be reported" % name)
            self.assertEqual(document["metadata"]["unresolved_commands"]["count"], 0,
                             name)
            for node in document["nodes"]:
                self.assertEqual(node["unresolved_commands"], [],
                                 "%s/%s" % (name, node["id"]))

    def test_content_loss_audit_is_clean(self):
        for name, document in sorted(OUTPUTS.items()):
            audit = document["metadata"]["content_loss_audit"]
            self.assertEqual(audit["lost"], [], "%s lost %s" % (name, audit["lost"]))
            self.assertEqual(audit["invented"], [], name)
            for command in audit["dropped"]:
                self.assertIn(command, P.DROPPED_COMMANDS, name)

    def test_every_document_has_latex_and_text(self):
        for name, document in sorted(OUTPUTS.items()):
            for node in document["nodes"]:
                self.assertIsInstance(node["latex"], str)
                self.assertIsInstance(node["text"], str)
                if node["id"] != "n0":
                    self.assertTrue(node["latex"], "%s/%s has no LaTeX" % (name, node["id"]))


class TestVerbatimIsContent(BaseCase):
    def test_verb_quoted_commands_are_not_structure(self):
        doc = OUTPUTS["document_4_structured_documents"]
        text = all_text(doc)
        self.assertIn("\\section", text)
        self.assertIn("\\begin{...}", text)
        self.assertIn("\\end{...}", text)
        # the sentence must be one paragraph, not three fragments
        fragments = [t for t in node_texts(doc) if t.strip().startswith("A LaTeX parser")]
        self.assertEqual(len(fragments), 1)
        self.assertIn("environments delimited by \\begin{...} and \\end{...}",
                      fragments[0])

    def test_verb_quoted_command_is_not_a_heading(self):
        doc = OUTPUTS["document_4_structured_documents"]
        headings = [n for n in doc["nodes"] if n.get("is_heading")]
        for node in headings:
            self.assertNotIn("\\", node["text"].replace("\\begin{", "").replace("\\end{", ""),
                             "%s: heading text still contains markup" % node["id"])

    def test_verb_quoted_commands_are_not_reported_unresolved(self):
        """Quoted markup is content, not a parser failure (regression)."""
        doc = OUTPUTS["document_4_structured_documents"]
        text = all_text(doc)
        for quoted in ("\\section", "\\subsection", "\\begin{...}", "\\end{...}"):
            self.assertIn(quoted, text, "%s should survive as content" % quoted)
            self.assertNotIn(quoted.lstrip("\\"),
                             doc["metadata"]["unresolved_commands"]["commands"],
                             "%s quoted verbatim must not be reported unresolved" % quoted)
        doc3 = OUTPUTS["document_3_mathematical_results"]
        self.assertIn("\\ref{lem:bounded}", all_text(doc3))
        self.assertEqual(doc3["metadata"]["unresolved_commands"], {"count": 0, "commands": []})

    def test_no_document_reports_any_unresolved_command(self):
        for name, document in sorted(OUTPUTS.items()):
            self.assertEqual(document["metadata"]["unresolved_commands"]["count"], 0,
                             "%s: %s" % (name, document["metadata"]["unresolved_commands"]))


class TestFractionRendering(BaseCase):
    def frac(self, source, math=True):
        return P.convert_text(source, math=math).text

    def test_simple_fraction(self):
        self.assertEqual(self.frac(r"$S=\frac{QK^\top}{\sqrt{d_k}}$"),
                         "S = (QK^⊤) / (√{d_k})")

    def test_fraction_variants(self):
        for command in P.FRACTION_COMMANDS:
            self.assertEqual(self.frac(r"$\%s{a}{b}$" % command), "(a) / (b)")

    def test_nested_fraction(self):
        self.assertEqual(self.frac(r"$\frac{a}{\frac{b}{c}}$"),
                         "(a) / ((b) / (c))")

    def test_fraction_without_argument_groups(self):
        """A malformed \\frac stays visible and is reported, never guessed."""
        result = P.convert_text(r"$\frac{a}$", math=True)
        self.assertEqual(result.text, r"\frac{a}")
        self.assertEqual(result.unresolved, ["frac"])

    def test_no_raw_frac_survives_in_the_dataset(self):
        for name, document in sorted(OUTPUTS.items()):
            self.assertNotIn("\\frac", all_text(document), name)
            self.assertNotIn("\\dfrac", all_text(document), name)

    def test_document_fractions(self):
        self.assertIn("Attention(Q, K, V) = softmax ((QK^⊤) / (√{d_k}) + M)V",
                      all_text(OUTPUTS["document_1_transformer_architecture"]))
        self.assertIn("S = (QK^⊤) / (√{d_k})",
                      all_text(OUTPUTS["document_5_attention_masks"]))


# ---------------------------------------------------------------------------
# 5. Renderer unit tests (no model, no files)
# ---------------------------------------------------------------------------
class TestRenderer(unittest.TestCase):
    def render(self, text, math=False, code=False):
        return P.convert_text(text, math=math, code=code).text

    def test_word_boundary_protects_top_from_to(self):
        """Regression: substring replacement turned \\top into an arrow."""
        self.assertEqual(self.render(r"$QK^\top$", math=True), "QK^⊤")

    def test_braces_are_kept(self):
        out = self.render(r"$(x_n)_{n\geq 1}$")
        self.assertEqual(out, "(x_n)_{n ≥ 1}")
        self.assertNotEqual(out, "(x_n)_ n≥ 1")

    def test_operators_are_mapped(self):
        self.assertEqual(self.render(r"$c\ne 0$"), "c ≠ 0")
        self.assertEqual(self.render(r"$a\implies b$"), "a ⇒ b")
        self.assertEqual(self.render(r"$N=\max(N_1,N_2)$"), "N = max(N_1, N_2)")
        self.assertEqual(self.render(r"$\varepsilon \to \infty$"), "ε → ∞")
        self.assertEqual(r"$\mathcal{C}_n \in \mathbb{R}$" and
                         self.render(r"$\mathcal{C}_n \in \mathbb{R}$"),
                         "𝒞_n ∈ ℝ")

    def test_unknown_command_is_preserved_and_reported(self):
        result = P.convert_text(r"$S=\weirdop{a}{b}$", math=True)
        self.assertIn(r"\weirdop{a}{b}", result.text)
        self.assertIn("weirdop", result.unresolved)

    def test_command_outside_verb_is_still_reported(self):
        """A genuinely unrendered command must not be silently excused."""
        result = P.convert_text(r"Use \verb|\section| but \weirdcmd also appears.")
        self.assertEqual(result.unresolved, ["weirdcmd"])
        self.assertIn("\\section", result.text)

    def test_command_mixed_in_verb_and_outside_is_reported(self):
        """If a command occurs outside a \\verb body too, it must be reported."""
        result = P.convert_text(r"\verb|\section| and a bare \section here")
        self.assertEqual(result.unresolved, ["section"])

    def test_fraction_is_never_reported_unresolved(self):
        result = P.convert_text(r"$\frac{a}{b}$", math=True)
        self.assertEqual(result.unresolved, [])
        self.assertEqual(result.text, "(a) / (b)")

    def test_content_free_command_is_dropped(self):
        result = P.convert_text(r"\maketitle Some text.")
        self.assertEqual(result.text, "Some text.")
        self.assertIn("maketitle", result.dropped)

    def test_math_spacing_does_not_leak_into_prose(self):
        self.assertEqual(self.render("a multi-head architecture."),
                         "a multi-head architecture.")
        self.assertEqual(self.render(r"$x_n-c$", math=True), "x_n - c")

    def test_alignment_rows_become_hard_breaks(self):
        out = self.render("a &= b\\\\\nc &= d", math=True)
        self.assertIn("\n", out)
        self.assertEqual(len(out.split("\n")), 2)

    def test_labels_and_references_are_recorded(self):
        result = P.convert_text(r"See \ref{eq:a} and \label{eq:b} here.")
        self.assertEqual(result.references, ["eq:a"])
        self.assertEqual(result.labels, ["eq:b"])
        self.assertIn("\\ref{eq:a}", result.text)

    def test_verb_body_survives_begin_end_cleanup(self):
        out = self.render(r"delimited by \verb|\begin{...}| and \verb|\end{...}|.")
        self.assertIn("\\begin{...}", out)
        self.assertIn("\\end{...}", out)

    def test_strip_comments_preserves_length_and_lines(self):
        source = "a % comment\nb\\%\nc"
        stripped = P.strip_comments(source)
        self.assertEqual(len(stripped), len(source))
        self.assertEqual(stripped.count("\n"), source.count("\n"))
        # the real comment is blanked; the escaped "\%" is not a comment
        self.assertEqual(stripped, "a          \nb\\%\nc")

    def test_escaped_percent_is_not_a_comment(self):
        stripped = P.strip_comments("x is 100\\% sure % not a comment\n")
        self.assertIn("100\\%", stripped)
        self.assertNotIn("not a comment", stripped)

    def test_comment_inside_inline_verb_is_kept(self):
        source = r"\verb|100%| done"
        self.assertIn("100%", P.strip_comments(source))

    def test_renderer_is_deterministic(self):
        sample = r"Mixed $\alpha_{ij}\leq\beta$ and \texttt{prose} plus \verb|\cmd|."
        first = P.convert_text(sample, math=False, code=False)
        second = P.convert_text(sample, math=False, code=False)
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.unresolved, second.unresolved)

    def test_scan_emits_events_for_display_math_and_headings(self):
        body = ("\\section{Intro}\ntext\n\\[\nx=1\n\\]\n"
                "$$y=3$$\n"
                "\\begin{equation}z=4\\end{equation}\n"
                "\\begin{itemize}\\item one\\item two\\end{itemize}")
        events = P.scan(body)
        kinds = [e["kind"] for e in events]
        # \[...\] and $$...$$ are display-math events; an equation environment is
        # an environment whose body arrives as a text run.
        self.assertEqual(kinds.count("math"), 2)
        self.assertIn("heading", kinds)
        self.assertEqual(kinds.count("item"), 2)
        envs = [e["name"] for e in events if e["kind"].startswith("env")]
        self.assertIn("itemize", envs)
        self.assertIn("equation", envs)

    def test_scan_marks_math_and_code_environments(self):
        body = ("plain prose\n"
                "\\begin{align}a&=b\\end{align}\n"
                "\\begin{algorithmic}\\State x\\end{algorithmic}")
        by_kind = {}
        for event in P.scan(body):
            if event["kind"] == "text":
                by_kind[(event["math"], event["code"])] = event["raw"]
        self.assertIn((False, False), by_kind)
        self.assertTrue(any(math_flag and not code for math_flag, code in by_kind))
        self.assertTrue(any(code for _, code in by_kind))
        self.assertTrue(by_kind[(False, False)].strip().startswith("plain prose"))

    def test_scan_does_not_treat_verb_content_as_heading(self):
        events = P.scan("text \\verb|\\section{Fake}| more text\n")
        self.assertEqual([e for e in events if e["kind"] == "heading"], [])

    def test_newtheorem_declarations_drive_classification(self):
        mapping = P.parse_newtheorem(
            "\\newtheorem{proposition}[definition]{Proposition}\n"
            "\\newtheorem{theorem}[definition]{Theorem}")
        self.assertEqual(P.classify_environment("proposition", mapping), "proposition")
        self.assertEqual(P.classify_environment("theorem", mapping), "theorem")
        # without a declaration, an unknown environment stays generic but named
        self.assertEqual(P.classify_environment("mysteryenv", mapping), "other")


class TestRobustnessOnStructuresTheCorpusLacks(unittest.TestCase):
    """The five documents never exercise these; the schema still promises them."""

    def build(self, source):
        title, nodes, info = P.extract_structure(source, "probe")
        P.validate_tree(nodes, len(source))
        self.assertEqual(P.audit_content_loss(source, nodes)["lost"], [])
        self.assertEqual(info["reparented"], [])
        return nodes

    def types(self, source):
        return {n["node_type"] for n in self.build(source)}

    def test_all_sectioning_levels_nest_and_validate(self):
        """Regression: \\part..\\paragraph on one line used to break containment."""
        nodes = self.build(r"\begin{document}\part{P}\chapter{C}\section{S}"
                           r"\subsection{SS}\subsubsection{SSS}\paragraph{PP}"
                           r"text\end{document}")
        self.assertEqual(
            {n["node_type"] for n in nodes},
            {"document", "part" if False else "section", "subsection",
             "subsubsection", "paragraph"})
        lookup = {n["id"]: n for n in nodes}
        for node in nodes:
            parent = lookup.get(node["parent"])
            if parent is not None and parent.get("is_heading"):
                self.assertGreaterEqual(node["source_span"][0], parent["source_span"][0])
                self.assertLessEqual(node["source_span"][1], parent["source_span"][1])

    def test_list_environments_produce_item_nodes(self):
        types = self.types(r"\begin{document}\begin{itemize}\item one\item two"
                           r"\end{itemize}\begin{enumerate}\item first"
                           r"\end{enumerate}\end{document}")
        self.assertEqual(types, {"document", "itemize", "enumerate", "item"})
        nodes = self.build(r"\begin{document}\begin{itemize}\item one\item two"
                           r"\end{itemize}\end{document}")
        items = [n for n in nodes if n["node_type"] == "item"]
        self.assertEqual(len(items), 2)
        self.assertEqual([i["text"] for i in items], ["one", "two"])

    def test_figure_and_table(self):
        types = self.types(r"\begin{document}\begin{figure}\caption{A fig}\end{figure}"
                           r"\begin{table}\caption{A tab}\begin{tabular}{ll}a&b"
                           r"\end{tabular}\end{table}\end{document}")
        self.assertEqual(types, {"document", "figure", "table"})

    def test_deprecated_envelope_nesting(self):
        self.assertEqual(self.types(r"\begin{document}\begin{theorem}"
                                    r"\begin{lemma}inner\end{lemma} outer"
                                    r"\end{theorem}\end{document}"),
                         {"document", "theorem", "lemma"})

    def test_labels_and_refs_are_recognised_not_unresolved(self):
        nodes = self.build(r"\begin{document}\section{S}\label{sec:s}"
                           r"See \ref{sec:s} and \eqref{eq:e}.\end{document}")
        joined = [n for n in nodes if "sec:s" in n["text"]]
        self.assertEqual(joined[0]["labels"], ["sec:s"])
        self.assertEqual(sorted(joined[0]["references"]), ["eq:e", "sec:s"])
        for node in nodes:
            self.assertEqual(node["unresolved_commands"], [], node["id"])

    def test_degenerate_inputs_do_not_crash(self):
        for source in [r"\section{A} text $x^2$",
                       r"\begin{document}\begin{quote}unclosed\end{document}",
                       r"\begin{document}text \end{nosuchenv} more\end{document}",
                       r"\begin{document}\end{document}",
                       "\\begin{document}\\$a\nb$ tail\\end{document}",
                       r"\begin{document}text $unclosed math.\end{document}",
                       r"\begin{document}\section{A} $\frac{a}{$ text.\end{document}"]:
            nodes = self.build(source)
            self.assertIsInstance(nodes, list)

    def test_verbatim_environment_is_opaque(self):
        source = ("\\begin{document}\\begin{verbatim}\\$x$ % not a comment"
                  "\\end{verbatim} after\\end{document}")
        nodes = self.build(source)
        verbatim = [n for n in nodes if n.get("environment") == "verbatim"]
        self.assertEqual(len(verbatim), 1)
        self.assertIn("$x$", verbatim[0]["text"])
        self.assertIn("% not a comment", verbatim[0]["text"])

    def test_malformed_command_is_reported_not_guessed(self):
        result = P.convert_text(r"$\frac{a}$", math=True)
        self.assertEqual(result.text, r"\frac{a}")
        self.assertEqual(result.unresolved, ["frac"])


class TestParserDeterminism(BaseCase):
    def test_reparsing_reproduces_identical_structure(self):
        for name in sorted(OUTPUTS):
            source = (ROOT / (name + ".tex")).read_text(encoding="utf-8")
            title, nodes, _ = P.extract_structure(source, name)
            stored = OUTPUTS[name]
            self.assertEqual(title, stored["title"], name)
            self.assertEqual(len(nodes), len(stored["nodes"]), name)
            for fresh, old in zip(nodes, stored["nodes"]):
                self.assertEqual(fresh["id"], old["id"], name)
                self.assertEqual(fresh["node_type"], old["node_type"], name)
                self.assertEqual(fresh["text"], old["text"], "%s/%s" % (name, old["id"]))
                self.assertEqual(fresh["source_span"], old["source_span"], name)

    def test_audit_raises_nothing_for_the_corpus(self):
        for name, document in sorted(OUTPUTS.items()):
            source = (ROOT / (name + ".tex")).read_text(encoding="utf-8")
            audit = P.audit_content_loss(source, document["nodes"])
            self.assertEqual(audit["lost"], [], "%s: %s" % (name, audit["lost"]))

    def test_validate_tree_accepts_every_stored_document(self):
        for name, document in sorted(OUTPUTS.items()):
            source_length = len((ROOT / (name + ".tex")).read_text(encoding="utf-8"))
            P.validate_tree(document["nodes"], source_length)


if __name__ == "__main__":
    unittest.main(verbosity=2)
