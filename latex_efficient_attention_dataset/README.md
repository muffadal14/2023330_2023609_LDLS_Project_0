# LaTeX dataset for efficient-attention project

This folder contains five self-contained LaTeX source documents plus the parser,
JSON schema, generated dataset and tests used for parser development and
structured-document experiments.

## Documents

1. `document_1_transformer_architecture.tex` — Transformer components and attention
2. `document_2_kv_cache_management.tex` — KV cache and eviction
3. `document_3_mathematical_results.tex` — definitions, lemmas, theorems, proofs
4. `document_4_structured_documents.tex` — document trees and parsing
5. `document_5_attention_masks.tex` — masks, sequence IDs, and batching

## Layout

| Path | Purpose |
| --- | --- |
| `parser/parse_latex.py` | The parser (structure, renderer, audit, tokenizer, sequence assignment, CLI) |
| `schema/document_schema.json` | JSON Schema for the emitted documents |
| `output/*.json` | Generated dataset, single-sequence convention |
| `output_blocks/*.json` | Generated dataset, one sequence per top-level block |
| `tests/test_regression.py` | Full regression suite (structure, fidelity, tokenizer, sequences) |
| `tests/validate_output.py` | Basic schema + structural validation |
| `tests/compare_outputs.py` | Reports differences between two output directories |

## Reproducing everything

Run from this directory with the project virtual environment (it is the only
interpreter that has `llama-cpp-python` and `jsonschema` installed):

```powershell
# single-sequence convention, the baseline the driver is compared against
.\.venv\Scripts\python.exe parser\parse_latex.py --output-dir output --seq-mode single
# one sequence per top-level block, what the driver consumes
.\.venv\Scripts\python.exe parser\parse_latex.py --output-dir output_blocks --seq-mode blocks --max-sequences 16 --min-block-tokens 16 --xref-links

.\.venv\Scripts\python.exe tests\validate_output.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_regression.py" -v
```

Both output directories must be regenerated whenever the parser changes, and
`validate_output.py` checks `output/`; pass it a different directory by
running it from a copy, or regenerate `output/` first.

The token limit is `--max-tokens` (default 2000).

`--backup-dir <dir>` copies the existing `output/*.json` somewhere before
overwriting them, and never replaces a backup file that already exists (pass
`--force-backup` to override). No baseline is kept inside the project, so point
the backup at a location outside it:

```powershell
.\.venv\Scripts\python.exe parser\parse_latex.py --output-dir output --backup-dir C:\temp\baseline
.\.venv\Scripts\python.exe tests\compare_outputs.py C:\temp\baseline output
```

`tests/compare_outputs.py` accepts either absolute paths or paths relative to
the project root, and prints node/token/type deltas, content-word recovery and
math-symbol retention.

## Tokenization and position convention

This is stated here and repeated in the `metadata` block of every emitted file,
so a reader never has to guess.

* **Tokenizer** — the Llama 3.2 GGUF vocabulary
  (`models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`) loaded through
  `llama-cpp-python` with `vocab_only=True`. `tokens` are therefore **real model
  token IDs**, not word counts or character offsets. `tests/test_regression.py`
  re-tokenizes every node and requires an exact match with the stored IDs.
* **Granularity** — each node's `text` is tokenized independently with
  `add_bos=False`, `special=True`. Consequently, concatenating the node token
  lists can differ by a token or two from tokenizing the whole document in one
  call. This is a deliberate consequence of node ownership, and it is recorded
  in `metadata.tokenization.note`.
* **`positions`** — 0-based, contiguous, and packed over the flattened pre-order
  node list. The document is one logical sequence, so `positions` is exactly
  `range(total_tokens)`.
* **`seq_ids`** — which logical sequence each token belongs to. Two
  conventions, selected by `--seq-mode` and recorded in
  `metadata.sequence_convention`:
  * `single_sequence_per_document` (default, and what `output/` holds) — every
    entry is `0` and the whole document is one sequence.
  * `one_sequence_per_top_level_block` (`--seq-mode blocks`, what
    `output_blocks/` holds) — every top-level block gets its own `seq_id`, so
    `llama.cpp` masks each block to itself and a block can be evicted with a
    single `llama_memory_seq_rm`. Each sequence owns exactly one contiguous
    position range, because llama.cpp rejects a sequence with a gap
    (`src/llama-batch.cpp`). `metadata.sequence_assignment.blocks` lists every
    sequence with its `pos_min`, `pos_max` and `token_count`.

  `positions` are packed in both modes: `position_convention` is always
  `packed_global` and `positions` is exactly `range(total_tokens)`.

  See `assign_sequences` in `parser/parse_latex.py` for the merge rules, and
  `../README.md` for how the driver consumes this.
* **Ownership** — a node owns only its own text. Descendant text is never
  copied into an ancestor, so every token belongs to exactly one node and
  subtree token sums never double-count. A newline in a node's `text` marks a
  point where a child node (for example an equation) was removed from the flow.

These counts are *not* counts of the raw `.tex` source. If an experiment feeds
the raw LaTeX to the model instead, tokenize the `.tex` file itself with the
same tokenizer and report that separately.

## Node content fields

Every node carries both:

* `latex` — the verbatim LaTeX owned by that node (its own text runs only).
* `text` — a deterministic, meaning-preserving linear rendering of `latex`.

Rendering rules:

* Grouping braces are **kept**, so subscripts, superscripts and grouping stay
  explicit: `(x_n)_{n\geq 1}` renders as `(x_n)_{n ≥ 1}`, never `(x_n)_ n ≥ 1`.
* Only an explicit allow-list of content-free commands
  (`DROPPED_COMMANDS` in the parser: `\maketitle`, `\left`, `\hline`, …) is
  removed. Everything else is mapped to a documented Unicode equivalent or
  **preserved verbatim**.
* `\frac`, `\dfrac`, `\tfrac` and `\cfrac` render as a parenthesised slash
  fraction, e.g. `\frac{QK^\top}{\sqrt{d_k}}` → `(QK^⊤) / (√{d_k})`. Nesting is
  handled (`\frac{a}{\frac{b}{c}}` → `(a) / ((b) / (c))`); a malformed `\frac`
  is left visible and reported rather than guessed at.
* Anything preserved verbatim is listed in the node's `unresolved_commands` and
  in `metadata.unresolved_commands`. **All five documents currently report zero
  unresolved commands.**
* Markup quoted on purpose inside `\verb|…|` — such as the `\section` and
  `\begin{...}` that Documents 3 and 4 use as teaching examples — is content the
  author wrote literally. It is preserved in `text` but deliberately **not**
  reported as unresolved, because that would be a false "the parser did not
  understand this" warning. A command that also occurs outside a `\verb` body
  is still reported.
* `audit_content_loss()` re-scans the source and fails the run if a control
  sequence is missing from the output for any reason other than the allow-list.
  The result is stored in `metadata.content_loss_audit` (currently `lost: []`
  for all five documents).

Other node fields: `environment` (original LaTeX environment name),
`source_span` (character offsets into the real `.tex` file), `child_ids`,
`labels`, `references`, `token_count`, and `is_heading` (distinguishes a
`\paragraph` heading from a text paragraph).

## Node types

`document`, `abstract`, `section`, `subsection`, `subsubsection`, `paragraph`,
`equation`, `align`, `cases`, `theorem`, `proposition`, `lemma`, `corollary`,
`conjecture`, `claim`, `remark`, `example`, `definition`, `proof`, `figure`,
`table`, `item`, `itemize`, `enumerate`, `description`, `algorithm`,
`algorithmic`, `other`.

* `align` is used for `align`, `alignat`, `aligned`, `split`, `flalign`,
  `gathered`; `equation` for `equation`, `gather`, `multline`, `displaymath`
  and for `\[ … \]` / `$$ … $$` display blocks.
* `cases`, `algorithm` and `algorithmic` have their own types; the original
  environment name is always kept in `environment`.
* Theorem-like environments take their kind from `\newtheorem` declarations in
  the document, so `proposition` is a `proposition` and not a `theorem`.
* `other` is used only when no dedicated type applies, and such nodes still
  carry their `environment` name.

## Current counts

| Document | Nodes | Nonempty | Packed tokens |
| --- | ---: | ---: | ---: |
| 1 | 39 | 39 | 1324 |
| 2 | 41 | 41 | 1481 |
| 3 | 28 | 28 | 842 |
| 4 | 50 | 50 | 1639 |
| 5 | 51 | 51 | 1729 |
| **Total** | **209** | **209** | **7015** |

All five stay under the 2000-token limit, all five validate against the schema,
and all five report zero unresolved commands. `tests/compare_outputs.py` prints the
node/token/type deltas and the math-symbol retention against a previous run
produced with `--backup-dir`.
