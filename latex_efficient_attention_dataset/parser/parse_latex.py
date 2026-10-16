#!/usr/bin/env python3
"""LaTeX-to-tree JSON parser for the efficient-attention dataset.

Design summary
--------------
Structure
    A hand-written scanner (:func:`scan`) walks the source once and emits a
    flat event stream (text runs, headings, environment boundaries, list items,
    display-math blocks).  :func:`extract_structure` turns that stream into a
    tree of nodes and emits them in document (pre-order) order.

    Scanning happens at the character/command level rather than with one big
    regex, so inline verbatim (``\\verb|\\section|``), inline math and balanced
    argument groups can never be mistaken for structural markup.  The previous
    parser split sentences in half at every ``\\verb|\\section|``.

Content fidelity
    Every node carries two content fields:

    ``latex``
        The verbatim LaTeX source owned by that node: its own text runs only.
        Text owned by a nested environment lives on that child node.
    ``text``
        A deterministic, meaning-preserving linear rendering of ``latex``.

    The renderer drops only an explicit allow-list of *content-free* commands
    (:data:`DROPPED_COMMANDS` - layout and preamble directives).  Every other
    control sequence is either mapped to a documented Unicode equivalent or
    preserved verbatim and reported in ``unresolved_commands``.  Grouping
    braces are deliberately kept so that subscripts, superscripts and grouping
    stay explicit: ``(x_n)_{n>=1}`` can never collapse into ``(x_n) n>=1``.

    :func:`audit_content_loss` re-scans the source and fails the run if a
    control sequence is absent from the output for any reason other than that
    allow-list, and the parser refuses to write a document that fails the
    audit.

Token ownership
    A node owns its own text only.  Descendant text is never copied into an
    ancestor, so every token is assigned to exactly one node and subtree token
    sums never double-count.

Tokenization scheme (also recorded in ``metadata`` of every emitted file)
    ``tokens``    Llama-3.2 GGUF vocabulary token IDs obtained through
                  llama-cpp-python (``Llama(vocab_only=True)``) by tokenizing
                  each node's ``text`` independently with ``add_bos=False`` and
                  ``special=True``.  These are real model-tokenizer counts, not
                  word or character estimates.
    ``positions`` 0-based, contiguous and packed over the flattened pre-order
                  node list, i.e. the whole document is one logical sequence and
                  ``positions[-1] == total_tokens - 1``.
    ``seq_ids``   all ``0``: every token belongs to logical sequence 0.

    Because nodes are tokenized independently, concatenating node tokens can
    differ by a token or two from tokenizing the document in one call.  That is
    a deliberate consequence of node ownership and is stated in the output.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

try:
    import jsonschema
except ImportError:  # pragma: no cover - optional dependency
    jsonschema = None

SCHEMA_VERSION = "2.0.0"

DEFAULT_INPUTS = [
    "document_1_transformer_architecture.tex",
    "document_2_kv_cache_management.tex",
    "document_3_mathematical_results.tex",
    "document_4_structured_documents.tex",
    "document_5_attention_masks.tex",
]

NODE_TYPES = {
    "document", "abstract", "section", "subsection", "subsubsection", "paragraph",
    "equation", "align", "cases", "theorem", "proposition", "lemma", "corollary",
    "conjecture", "claim", "remark", "example", "definition", "proof",
    "figure", "table", "item", "itemize", "enumerate", "description",
    "algorithm", "algorithmic", "other",
}

SECTION_LEVELS = {
    "part": 1, "chapter": 2, "section": 3,
    "subsection": 4, "subsubsection": 5, "paragraph": 6,
}

# Statement kind for theorem-like environments.  A document may override these
# through \newtheorem; see :func:`parse_newtheorem`.
STATEMENT_ENVIRONMENTS = {
    "theorem": "theorem", "theorem*": "theorem",
    "proposition": "proposition", "proposition*": "proposition",
    "lemma": "lemma", "lemma*": "lemma",
    "corollary": "corollary", "corollary*": "corollary",
    "conjecture": "conjecture", "conjecture*": "conjecture",
    "claim": "claim", "claim*": "claim",
    "definition": "definition", "definition*": "definition",
    "example": "example", "example*": "example",
    "remark": "remark", "remark*": "remark",
    "proof": "proof", "qedsymbol": "other",
}

ENVIRONMENT_TYPES = {
    "abstract": "abstract",
    "equation": "equation", "equation*": "equation",
    "displaymath": "equation", "eqnarray": "equation", "eqnarray*": "equation",
    "gather": "equation", "gather*": "equation",
    "multline": "equation", "multline*": "equation",
    "math": "equation", "IEEEeqnarray": "equation", "IEEEeqnarray*": "equation",
    "align": "align", "align*": "align", "alignat": "align", "alignat*": "align",
    "aligned": "align", "alignedat": "align", "gathered": "align",
    "flalign": "align", "flalign*": "align", "split": "align",
    "cases": "cases", "dcases": "cases", "rcases": "cases",
    "figure": "figure", "figure*": "figure", "wrapfigure": "figure",
    "table": "table", "table*": "table", "tabular": "table", "tabular*": "table",
    "array": "table", "subequations": "other",
    "itemize": "itemize", "enumerate": "enumerate", "description": "description",
    "algorithm": "algorithm", "algorithm*": "algorithm",
    "algorithmic": "algorithmic", "algorithmicx": "algorithmic",
}

SMALL_NODE_TYPES = {"paragraph", "item", "other"}

VERBATIM_ENVIRONMENTS = {"verbatim", "Verbatim", "lstlisting", "alltt", "comment"}

# --------------------------------------------------------------------------
# Content-free commands: the *only* commands the renderer may remove.
# --------------------------------------------------------------------------
DROPPED_COMMANDS = {
    # document structure and pagination
    "maketitle", "makeindex", "printindex", "tableofcontents", "listoffigures",
    "listoftables", "newpage", "clearpage", "cleardoublepage", "pagebreak",
    "linebreak", "nobreak", "onecolumn", "twocolumn", "appendix",
    # pure delimiter sizing (the braces are kept, the size keyword is not)
    "left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl",
    "Bigr", "biggl", "biggr", "Biggl", "Biggr", "bigm", "Bigm", "middle",
    # typography switches and spacing
    "noindent", "indent", "par", "centering", "raggedright", "raggedleft",
    "smallskip", "medskip", "bigskip", "vskip", "hskip", "hspace", "vspace",
    "hfill", "vfill", "hrulefill", "normalfont", "rmfamily", "sffamily",
    "ttfamily", "bfseries", "mdseries", "itshape", "upshape", "scshape",
    "slshape", "bf", "it", "sf", "tt", "em", "rm", "sl", "sc", "protect",
    "relax", "unskip", "sloppy", "raggedbottom", "flushbottom",
    "thispagestyle", "pagestyle", "justifying", "and",
    # definitions and counters
    "renewcommand", "newcommand", "providecommand", "DeclareMathOperator",
    "newtheorem", "DeclareTheorem", "newcounter", "setcounter", "addtocounter",
    "stepcounter", "refstepcounter", "setlength", "addtolength", "settowidth",
    "settoheight", "arraystretch", "linespread", "item", "itemsep", "topsep",
    # preamble and package hooks
    "documentclass", "usepackage", "RequirePackage", "bibliographystyle",
    "bibliography", "printbibliography", "keywords", "IEEEpubitem",
    "indexspace", "let", "def", "global", "long", "outer", "edef", "xdef",
    # purely visual table rules
    "toprule", "midrule", "bottomrule", "cmidrule", "cline", "hdashline",
    "hline", "captionsetup",
}

# Commands whose braced argument *is* the content: unwrap and keep the argument.
WRAPPER_COMMANDS = {
    "textbf", "textit", "emph", "textsl", "textrm", "texttt", "textsc",
    "textmd", "textup", "textnormal", "mbox", "makebox", "text",
    "underline", "overline", "textsuperscript", "textsubscript",
    "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "mathnormal",
    "boldsymbol", "pmb", "bm", "operatorname", "operatorname*",
    "operatornamewithlimits", "footnote", "footnotemark", "footnotetext",
    "index", "glossary", "caption", "captionof", "subcaption",
}

SPACING_COMMANDS = {
    ",", ";", ":", "!", "quad", "qquad", "enspace", "thinspace",
    "medspace", "thickspace", "negthinspace", "space", "nobreakspace",
}

# Fraction commands, rendered as a parenthesised slash fraction because no
# Unicode glyph exists for an arbitrary numerator over denominator.
FRACTION_COMMANDS = ("dfrac", "tfrac", "cfrac", "frac")

LABEL_COMMANDS = {"label"}
REFERENCE_COMMANDS = {
    "ref", "eqref", "autoref", "pageref", "cref", "Cref", "crefrange",
    "Crefrange", "vref", "nameref", "cite", "citep", "citet", "citealp",
    "citealt", "citeauthor", "citeyear", "nocite", "footcite",
}

ALGORITHM_KEYWORDS = {
    "Require": "Require:", "Requires": "Requires:",
    "Ensure": "Ensure:", "Ensures": "Ensures:", "Ensureleft": "Ensure:",
    "State": "State:", "Call": "Call:", "Return": "Return:",
    "Comment": "Comment:", "Cmt": "Comment:",
    "ForAll": "For all", "For": "For", "While": "While", "If": "If",
    "Else": "Else", "Repeat": "Repeat", "RepeatUntil": "Repeat until",
    "Procedure": "Procedure", "Function": "Function",
    "EndFor": "end for", "EndForAll": "end for all", "EndWhile": "end while",
    "EndIf": "end if", "EndRepeat": "end repeat", "EndProcedure": "end procedure",
    "EndFunction": "end function",
}

_KEYWORD_LINE_RE = re.compile(
    r"^(?:State:|Call:|Return:|Require:|Ensure[s]?:|Comment:|For all\b|For\b|"
    r"While\b|Repeat\b|Repeat until\b|If\b|Else\b|Procedure\b|Function\b|"
    r"end for\b|end while\b|end if\b|end repeat\b)")

MATH_ENVIRONMENTS = {
    "equation", "equation*", "align", "align*", "aligned", "alignedat",
    "alignat", "alignat*", "gather", "gather*", "gathered", "multline",
    "multline*", "displaymath", "eqnarray", "eqnarray*", "split", "flalign",
    "flalign*", "cases", "dcases", "rcases", "math", "IEEEeqnarray",
    "IEEEeqnarray*",
}

CODE_ENVIRONMENTS = {"algorithm", "algorithm*", "algorithmic", "algorithmicx"}

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "varpi": "ϖ",
    "rho": "ρ", "varrho": "ϱ", "sigma": "σ", "varsigma": "ς", "tau": "τ",
    "upsilon": "υ", "phi": "φ", "varphi": "ϕ", "chi": "χ", "psi": "ψ",
    "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ",
    "Psi": "Ψ", "Omega": "Ω",
}

SYMBOLS = {
    # relations
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠",
    "equiv": "≡", "approx": "≈", "sim": "∼", "simeq": "≃", "cong": "≅",
    "propto": "∝", "prec": "≺", "preceq": "⪯", "succ": "≻", "succeq": "⪰",
    "subset": "⊂", "supset": "⊃", "subseteq": "⊆", "supseteq": "⊇",
    "in": "∈", "ni": "∋", "notin": "∉", "perp": "⊥", "parallel": "∥",
    "doteq": "≐", "asymp": "≍", "models": "⊨", "vdash": "⊢",
    "mid": "∣", "top": "⊤", "bot": "⊥", "ll": "≪", "gg": "≫",
    # arrows and logic
    "to": "→", "rightarrow": "→", "longrightarrow": "→", "shortrightarrow": "→",
    "leftarrow": "←", "gets": "←", "longleftarrow": "←", "shortleftarrow": "←",
    "leftrightarrow": "↔", "uparrow": "↑", "downarrow": "↓",
    "Rightarrow": "⇒", "implies": "⇒", "Longrightarrow": "⇒", "shortimplies": "⇒",
    "Leftarrow": "⇐", "Leftrightarrow": "⇔", "iff": "⇔",
    "Longleftrightarrow": "⇔", "mapsto": "↦", "hookrightarrow": "↪",
    "forall": "∀", "exists": "∃", "nexists": "∄", "neg": "¬", "lnot": "¬",
    "land": "∧", "wedge": "∧", "lor": "∨", "vee": "∨",
    # operators and set notation
    "times": "×", "div": "÷", "pm": "±", "mp": "∓", "cdot": "·",
    "ast": "∗", "star": "⋆", "sum": "∑", "prod": "∏", "coprod": "∐",
    "int": "∫", "iint": "∬", "iiint": "∭", "oint": "∮",
    "cup": "∪", "cap": "∩", "setminus": "∖", "oplus": "⊕", "otimes": "⊗",
    "circ": "∘", "dagger": "†", "ddagger": "‡", "checkmark": "✓",
    "bigoplus": "⨁", "bigotimes": "⨂", "infty": "∞", "partial": "∂",
    "nabla": "∇", "emptyset": "∅", "varnothing": "∅", "aleph": "ℵ",
    "hbar": "ℏ", "ell": "ℓ", "imath": "ı", "jmath": "ȷ",
    "Re": "ℜ", "Im": "ℑ", "prime": "′", "degree": "°",
    # punctuation and delimiters
    "dots": "…", "ldots": "…", "cdots": "…", "vdots": "⋮", "ddots": "⋱",
    "angle": "∠", "triangle": "△", "square": "□", "diamond": "⋄",
    "sqrt": "√", "backslash": "\\",
    "langle": "⟨", "rangle": "⟩", "lfloor": "⌊", "rfloor": "⌋",
    "lceil": "⌈", "rceil": "⌉", "lbrace": "{", "rbrace": "}",
    "vert": "|", "Vert": "‖", "lVert": "‖", "rVert": "‖",
    # function-like operators, rendered as readable names
    "max": "max", "min": "min", "sup": "sup", "inf": "inf", "lim": "lim",
    "limsup": "lim sup", "liminf": "lim inf", "log": "log", "ln": "ln",
    "exp": "exp", "sin": "sin", "cos": "cos", "tan": "tan", "det": "det",
    "dim": "dim", "ker": "ker", "deg": "deg", "gcd": "gcd", "hom": "hom",
    "arg": "arg", "Pr": "Pr",
}
SYMBOLS.update(GREEK)

BLACKBOARD = {
    "R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ",
    "E": "\U0001D53C", "P": "\U0001D53E", "F": "\U0001D53B",
}

CALLIGRAPHIC = {
    "A": "\U0001D49C", "C": "\U0001D49E", "D": "\U0001D49F", "E": "ℰ",
    "F": "ℱ", "G": "\U0001D4A2", "H": "ℋ", "I": "ℐ", "J": "\U0001D4A5",
    "K": "\U0001D4A6", "L": "ℒ", "M": "ℳ", "N": "\U0001D4A9", "O": "\U0001D4AA",
    "P": "\U0001D4AB", "Q": "\U0001D4AC", "R": "ℛ", "S": "\U0001D4AE",
    "T": "\U0001D4AF", "U": "\U0001D4B0", "V": "\U0001D4B1", "W": "\U0001D4B2",
    "X": "\U0001D4B3", "Y": "\U0001D4B4", "Z": "\U0001D4B5",
}

ESCAPED_LITERALS = {
    "%": "%", "&": "&", "#": "#", "$": "$", "_": "_", "{": "{", "}": "}",
    " ": " ", "`": "`", "^": "^", "-": "",
}

OPERATOR_SPACING = "=<>≤≥≠≈≡∼→←⇒⇔×÷±∓∈∉⊂⊆⊃⊇∪∩∧∨∀∃¬∪∩⋃⋂"
# Extra glyphs that may follow an operator: Unicode letters already match \w.
VALUE_START = r"[\w(\[{∞±∓×÷…′√∂∇‖∣|≪≫∘⊕⊗∑∏∫∮⊤∅⋆∗∠△□⋄]"
# Characters (not regex) added to the operator set inside math.  Applying the
# set to prose would turn "multi-head" into "multi - head".
MATH_EXTRA_SPACING = r"-+*/"
# Sentinel for a LaTeX row separator ("\\"), preserved until whitespace
# normalisation so that alignment rows survive as hard line breaks.
ROW_BREAK = "\x00"
# Sentinels marking the body of an inline \verb span.
VERB_OPEN = "\x02"
VERB_CLOSE = "\x03"

_CMD_WORD_RE = re.compile(r"\\([A-Za-z@]+)(\*?)")
_ENV_RE = re.compile(r"\\(begin|end)\s*\{([A-Za-z*]+)\}")
_INLINE_VERB_RE = re.compile(r"\\(?:verb\*?|lstinline|Verb\*?)\s*(.)", re.S)
_NEWTHEOREM_RE = re.compile(
    r"\\newtheorem\*?\s*\{([^}]*)\}\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")
_TITLE_RE = re.compile(r"\\title\s*(?:\[[^\]]*\])?")
_CAPTION_RE = re.compile(r"\\caption\*?\s*\{([^{}]*)\}")
_LABEL_RE = re.compile(r"\\label\*?\s*\{([^{}]*)\}")
_REFERENCE_RE = re.compile(
    r"\\(?:ref|eqref|autoref|pageref|cref|Cref|crefrange|Crefrange|vref|"
    r"nameref|cite|citep|citet|citealp|citealt|citeauthor|citeyear|nocite|"
    r"footcite)\*?\s*\{([^{}]*)\}")
_ANY_COMMAND_RE = re.compile(r"\\([A-Za-z@]+)\*?")


# ---------------------------------------------------------------------------
# Low level helpers
# ---------------------------------------------------------------------------
def strip_comments(source: str) -> str:
    """Replace LaTeX comments with spaces, preserving length and line count.

    Keeping the length identical means every offset computed on the result is
    also a valid offset into the original file, so ``source_span`` values index
    the real ``.tex`` file.
    """
    chars = list(source)
    n = len(source)
    i = 0
    verb_delim = None
    verb_resume = 0
    while i < n:
        if verb_delim is not None:
            if i >= verb_resume and source[i] == verb_delim and source[i - 1] != "\\":
                verb_delim = None
            i += 1
            continue
        ch = source[i]
        if ch == "%":
            # "\%" is an escaped literal percent, not a comment introducer.
            backslashes = 0
            j = i - 1
            while j >= 0 and source[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 1:
                i += 1
                continue
            j = i
            while j < n and source[j] != "\n":
                chars[j] = " "
                j += 1
            i = j
            continue
        if ch == "\\":
            match = _INLINE_VERB_RE.match(source, i)
            if match:
                verb_delim = match.group(1)
                verb_resume = match.end()
                i = verb_resume
                continue
            match = _ENV_RE.match(source, i)
            if match and match.group(2) in VERBATIM_ENVIRONMENTS:
                # Skip to the matching \end so that '%' inside stays literal.
                closing = re.compile(r"\\end\s*\{" + re.escape(match.group(2)) + r"\}").search(source, match.end())
                i = closing.end() if closing else match.end()
                continue
            i += 1
            continue
        i += 1
    return "".join(chars)


def read_group(source: str, start: int):
    """Read a balanced ``{...}`` group at/after ``start``.

    Returns ``(content, next_index, content_start, content_end)``; ``content``
    is ``None`` when there is no group.
    """
    i = start
    while i < len(source) and source[i].isspace():
        i += 1
    if i >= len(source) or source[i] != "{":
        return None, i, None, None
    depth = 0
    j = i
    while j < len(source):
        ch = source[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[i + 1:j], j + 1, i + 1, j
        j += 1
    return None, len(source), None, None


def read_optional_arg(source: str, start: int):
    """Read a ``[...]`` optional argument; returns ``(content, next_index)``."""
    i = start
    while i < len(source) and source[i].isspace():
        i += 1
    if i >= len(source) or source[i] != "[":
        return None, i
    depth = 0
    j = i
    while j < len(source):
        ch = source[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return source[i + 1:j], j + 1
        j += 1
    return None, len(source)


# ---------------------------------------------------------------------------
# LaTeX -> meaning-preserving text
# ---------------------------------------------------------------------------
def _replace_symbols(text: str, table: dict) -> str:
    """Substitute ``\\name`` -> value using a word-boundary-safe regex.

    The look-ahead is what stops the entry for ``\\to`` from rewriting the
    inside of ``\\top`` (the previous parser produced ``QK^->p`` that way).
    """
    if not table:
        return text
    names = sorted(table, key=len, reverse=True)
    pattern = re.compile(
        r"\\(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z@])")

    def sub(match):
        return table[match.group(1)]

    return pattern.sub(sub, text)


def _unwrap_commands(text: str, names) -> str:
    """Replace ``\\cmd{arg}`` with ``arg`` using balanced-brace scanning."""
    if not names:
        return text
    pattern = re.compile(
        r"\\(" + "|".join(sorted(names, key=len, reverse=True)) + r")\*?[ \t]*")
    out = []
    pos = 0
    while True:
        match = pattern.search(text, pos)
        if match is None:
            out.append(text[pos:])
            return "".join(out)
        out.append(text[pos:match.start()])
        content, nxt, _, _ = read_group(text, match.end())
        if content is None:
            out.append(match.group(0))
            pos = match.end()
            continue
        out.append(content)
        pos = nxt


def _consume_groups(text: str, start: int):
    """Consume consecutive balanced ``{...}``/``[...]`` groups after ``start``.

    Only spaces and tabs are skipped, never newlines, so that a command which
    takes a group does not swallow the line break that follows it.
    """
    i = start
    parts = []
    while True:
        content, nxt, _, _ = read_group(text, i)
        if content is None:
            content, nxt = read_optional_arg(text, i)
            if content is None:
                break
        parts.append(content)
        i = nxt
        j = i
        while j < len(text) and text[j] in " \t":
            j += 1
        if j == i:
            break
        i = j
    return parts, i


def _apply_algorithm_keywords(text: str) -> str:
    """Turn ``\\State x`` / ``\\ForAll{...}`` into readable pseudocode lines."""
    names = sorted(ALGORITHM_KEYWORDS, key=len, reverse=True)
    pattern = re.compile(
        r"\\(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z@])")
    out = []
    pos = 0
    while True:
        match = pattern.search(text, pos)
        if match is None:
            out.append(text[pos:])
            return "".join(out)
        out.append(text[pos:match.start()])
        groups, nxt = _consume_groups(text, match.end())
        following = text[nxt] if nxt < len(text) else ""
        if groups:
            out.append(ALGORITHM_KEYWORDS[match.group(1)] + " " + " ".join(groups))
        else:
            out.append(ALGORITHM_KEYWORDS[match.group(1)]
                       + ("" if following in "\n\r" or following == "" else " "))
        pos = nxt


_FRACTION_RE = re.compile(r"\\(?:%s)(?![A-Za-z@])" % "|".join(FRACTION_COMMANDS))


def _render_fractions(text: str) -> str:
    """Render ``\\frac{a}{b}`` as the readable slash form ``(a)/(b)``.

    A parenthesised slash fraction is used because it survives plain text,
    nesting and tokenizer input unchanged, and because no Unicode glyph exists
    for an arbitrary numerator over denominator.  Nesting is handled by
    recursing into both arguments; a malformed ``\\frac`` is left visible
    rather than guessed at.
    """
    out = []
    pos = 0
    while True:
        match = _FRACTION_RE.search(text, pos)
        if match is None:
            out.append(text[pos:])
            return "".join(out)
        numerator, after_num, _, _ = read_group(text, match.end())
        denominator, after_den = (None, None)
        if numerator is not None:
            denominator, after_den, _, _ = read_group(text, after_num)
        if numerator is None or denominator is None:
            out.append(text[pos:match.end()])
            pos = match.end()
            continue
        out.append(text[pos:match.start()])
        out.append("(%s)/(%s)" % (_render_fractions(numerator.strip()),
                                  _render_fractions(denominator.strip())))
        pos = after_den


# Fraction commands rendered as a parenthesised slash fraction.
def _replace_blackboard_and_calligraphic(text: str) -> str:
    def repl(match):
        style, letter = match.group(1), match.group(2)
        table = BLACKBOARD if style == "mathbb" else CALLIGRAPHIC
        if letter in table:
            return table[letter]
        return match.group(0)  # unmapped font stays visible rather than lost

    return re.sub(r"\\(mathbb|mathcal)\s*\{\s*([A-Za-z])\s*\}", repl, text)


def _char_class(chars: str) -> str:
    """Build a regex character class from literal characters.

    ``re.escape`` must not be used here: escaping ``\\-+*/`` inside ``[...]``
    adds the backslash itself as a class member, which makes every LaTeX
    control sequence match the class.
    """
    return "[" + re.sub(r"([\\\]\^\-])", r"\\\1", chars) + "]"


def _apply_spacing_repair(line: str, math: bool = False) -> str:
    """Insert deterministic spaces around relation/operator glyphs.

    ``\\w`` is Unicode-aware in Python 3, so Unicode letters such as the Greek
    letters and blackboard bold count as word characters: ``X\\in\\mathbb{R}``
    renders as ``X ∈ ℝ`` rather than ``X ∈ℝ``.  Inside math the operator set also
    includes ``-``, ``+``, ``*`` and ``/``; applying it to prose would turn
    "multi-head" into "multi - head".
    """
    classes = _char_class(OPERATOR_SPACING + (MATH_EXTRA_SPACING if math else ""))
    line = re.sub(r"(?<=[\w)\]\|}])\s*(?=" + classes + r")\s*", " ", line)
    line = re.sub(r"(?<=" + classes + r")\s*(?=" + VALUE_START + r")", " ", line)
    line = re.sub(r",(?=[^\s)\]])", ", ", line)
    # A stripped math delimiter can leave a space in front of sentence
    # punctuation: "$d$. Learned" must not become "width d . Learned".
    line = re.sub(r"[ \t]+([.,;:!?])(?=\s|$)", r"\1", line)
    # "\left(" / "\right)" are dropped as sizing keywords; tidy the gap they leave.
    return re.sub(r"([(\[{])[ \t]+", r"\1", re.sub(r"[ \t]+([)\]}])", r"\1", line))


def _space_math_spans(text: str) -> str:
    """Apply math spacing inside inline/display math delimiters."""
    def repl(match):
        return _apply_spacing_repair(match.group(0), math=True)

    text = re.sub(r"(?<![\\$])\$\$[\s\S]*?(?<!\\)\$\$", repl, text)
    text = re.sub(r"(?<![\\$])\$(?!\$)[\s\S]*?(?<!\\)\$", repl, text)
    text = re.sub(r"\\\[([\s\S]*?)\\\]", lambda m: "\\[" + _apply_spacing_repair(m.group(1), math=True) + "\\]", text)
    text = re.sub(r"\\\(([\s\S]*?)\\\)", lambda m: "\\(" + _apply_spacing_repair(m.group(1), math=True) + "\\)", text)
    return text


def _normalise_whitespace(text: str, code: bool = False) -> str:
    """Collapse soft line wraps but keep explicit structure.

    LaTeX source wraps prose across lines; joining those with spaces is what a
    reader expects.  A newline is kept when the previous line carried a LaTeX
    row separator (``\\\\``, marked by :data:`ROW_BREAK`) and, inside
    pseudocode, when either neighbouring line starts with an algorithmic
    keyword, so listing line structure survives.
    """
    blocks = []
    current = []
    pending_hard = False
    previous_raw = ""
    for raw_line in text.split("\n"):
        hard_after = ROW_BREAK in raw_line
        stripped = raw_line.replace(ROW_BREAK, "")
        converted = _apply_spacing_repair(re.sub(r"[ \t]+", " ", stripped)).strip()
        if not converted:
            if current:
                blocks.append(current)
                current = []
            pending_hard = False
            previous_raw = ""
            continue
        hard = pending_hard or bool(current) and code and (
            bool(_KEYWORD_LINE_RE.match(converted))
            or bool(_KEYWORD_LINE_RE.match(_apply_spacing_repair(previous_raw).strip()))
        )
        current.append((hard, converted))
        pending_hard = hard_after
        previous_raw = stripped
    if current:
        blocks.append(current)

    out_blocks = []
    for block in blocks:
        parts = []
        for hard, value in block:
            parts.append(("\n" if parts and hard else " ") + value)
        out_blocks.append("".join(parts).strip())
    return "\n".join(b for b in out_blocks if b)


class NormalisationResult:
    __slots__ = ("text", "labels", "references", "captions", "dropped", "unresolved")

    def __init__(self, text, labels, references, captions, dropped, unresolved):
        self.text = text
        self.labels = labels
        self.references = references
        self.captions = captions
        self.dropped = dropped
        self.unresolved = unresolved


_VERB_PLACEHOLDER_RE = re.compile(VERB_OPEN + r"(\d+)" + VERB_CLOSE)


def _extract_inline_verb(text: str):
    """Hide ``\\verb|<delim>body<delim>`` bodies behind placeholders.

    Two things must happen before any other rule runs:

    * ``\\verb|\\section|`` must not be read as a heading, and
    * ``\\verb|\\begin{...}|`` must not be deleted by the ``\\begin``/``\\end``
      cleanup below.

    Replacing the body with an opaque placeholder (``\\x02<index>\\x03``, which
    contains no backslash or brace) guarantees that no later regex can reach
    it.  :func:`_restore_inline_verb` puts the literal bodies back at the very
    end, so quoted markup survives into ``text`` instead of being deleted.
    """
    bodies = []
    out = []
    pos = 0
    for match in _INLINE_VERB_RE.finditer(text):
        out.append(text[pos:match.start()])
        delim = match.group(1)
        start = match.end()
        end = text.find(delim, start)
        if end == -1:
            out.append(text[start:])
            return "".join(out), bodies
        bodies.append(text[start:end])
        out.append("%s%d%s" % (VERB_OPEN, len(bodies) - 1, VERB_CLOSE))
        pos = end + 1
    out.append(text[pos:])
    return "".join(out), bodies


def _restore_inline_verb(text: str, bodies) -> str:
    def repl(match):
        return bodies[int(match.group(1))]

    return _VERB_PLACEHOLDER_RE.sub(repl, text)


def _unresolved_commands(text: str, verb_bodies):
    """Commands left unrendered in ``text``, excluding recognised ones.

    Three kinds of command are *not* unresolved:

    * commands quoted on purpose inside ``\\verb|...|`` - such as the
      ``\\section`` and ``\\begin{...}`` that Documents 3 and 4 use as teaching
      examples.  That is content the author wrote literally, not something the
      parser failed to understand.  A command is only excused this way when
      every one of its occurrences lies inside a verbatim body.
    * ``\\label`` and the ``\\ref``/``\\cite`` family, which stay visible in
      ``text`` but are recognised and recorded in the node's ``labels`` and
      ``references``.
    * the content-free allow-list and pure-spacing commands, which are removed
      on purpose.
    """
    text_counts = Counter(_ANY_COMMAND_RE.findall(text))
    verb_counts = Counter()
    for body in verb_bodies:
        verb_counts.update(_ANY_COMMAND_RE.findall(body))
    excused = {c for c in verb_counts if text_counts[c] == verb_counts[c]}
    excused |= LABEL_COMMANDS | REFERENCE_COMMANDS
    return sorted(c for c in text_counts
                  if c not in excused
                  and c not in DROPPED_COMMANDS
                  and c not in SPACING_COMMANDS)


def convert_text(text: str, math: bool = False, code: bool = False) -> NormalisationResult:
    """Render LaTeX into a deterministic, meaning-preserving linear form.

    ``math`` marks text that the source typesets in math mode (the body of an
    equation-like environment); ``code`` marks pseudocode, where line structure
    is significant.
    """
    # 1. Hide inline verbatim first: quoted markup is content, not structure.
    text, verb_bodies = _extract_inline_verb(text)

    # 2. Identifiers become machine-readable metadata (and stay visible).
    labels = _LABEL_RE.findall(text)
    references = _REFERENCE_RE.findall(text)
    captions = list(_CAPTION_RE.findall(text))

    # 3. Math spacing is applied before the delimiters disappear.
    text = _space_math_spans(text)

    # 4. Structural markers must not leak into node text.
    text = re.sub(r"\\(begin|end)\s*\{[^{}]*\}", " ", text)
    text = re.sub(r"\\\[|\\\]|\\\(|\\\)", " ", text)
    text = re.sub(r"\$\$|\$", " ", text)

    # 5. Commands whose argument is the content.
    text = _apply_algorithm_keywords(text)
    text = _unwrap_commands(text, WRAPPER_COMMANDS)
    text = _replace_blackboard_and_calligraphic(text)

    # 6. Drop only the allow-listed content-free commands and pure spacing.
    dropped = sorted({c for c in _ANY_COMMAND_RE.findall(text)
                      if c in DROPPED_COMMANDS})
    text = _replace_symbols(text, {name: " " for name in DROPPED_COMMANDS})
    text = re.sub(
        r"\\(?:[,;:!]|quad|qquad|enspace|thinspace|medspace|thickspace|"
        r"negthinspace|hspace\*?|vspace\*?)[ \t]*", " ", text)

    # 7. Escaped literals and the double-bar delimiter.
    text = text.replace("\\|", "‖")
    for old, new in ESCAPED_LITERALS.items():
        text = re.sub(r"\\" + re.escape(old), new, text)

    # 8. Symbols (word-boundary safe).
    text = _replace_symbols(text, SYMBOLS)
    text = _render_fractions(text)

    # 9. Alignment: rows become hard breaks, columns are marked with ' & '.
    text = re.sub(r"\\\\\*?[ \t]*(\[[^\]]*\])?", ROW_BREAK, text)
    text = re.sub(r"[ \t]*&[ \t]*", " & ", text)

    # 10. Whitespace and spacing repair.  Braces are intentionally preserved.
    text = _normalise_whitespace(text.replace("~", " "), code=code or math)
    if math:
        text = "\n".join(_apply_spacing_repair(line, math=True)
                         for line in text.split("\n"))
    # Put the literal \verb bodies back, then pick up any identifiers they quote.
    text = _restore_inline_verb(text, verb_bodies)
    labels = sorted(set(labels) | set(_LABEL_RE.findall(text)))
    references = sorted(set(references) | set(_REFERENCE_RE.findall(text)))
    unresolved = _unresolved_commands(text, verb_bodies)
    return NormalisationResult(text, labels, references, captions, dropped, unresolved)


def render(text: str) -> str:
    """Convenience wrapper returning only the rendered text."""
    return convert_text(text).text


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------
def scan(source: str, body_offset: int = 0):
    """Yield structural events for ``source``.

    The scanner never looks inside inline verbatim, inline math, display math
    or verbatim environments, so markup quoted as content cannot be mistaken
    for structure.
    """
    n = len(source)
    i = 0
    events = []
    buf = []
    buf_start = 0
    env_stack = []

    def flush(end):
        raw = "".join(buf)
        buf.clear()
        if raw.strip():
            innermost = env_stack[-1][0] if env_stack else ""
            events.append({"kind": "text", "raw": raw,
                           "start": body_offset + buf_start, "end": body_offset + end,
                           "math": innermost in MATH_ENVIRONMENTS,
                           "code": innermost in CODE_ENVIRONMENTS})

    def emit(event):
        nonlocal buf_start
        flush(event["start"] - body_offset)
        events.append(event)
        buf_start = event["end"] - body_offset

    while i < n:
        ch = source[i]

        if ch == "\\":
            if not buf:
                buf_start = i
            # Display math must be recognised before the generic command path,
            # otherwise "\[" is consumed as an escaped literal and the equation
            # silently disappears into the surrounding prose.
            if source.startswith("\\[", i):
                end = source.find("\\]", i + 2)
                end = n if end == -1 else end + 2
                emit({"kind": "math", "raw": source[i:end],
                      "start": body_offset + i, "end": body_offset + end})
                i = end
                continue

            verb = _INLINE_VERB_RE.match(source, i)
            if verb:
                end = source.find(verb.group(1), verb.end())
                end = n if end == -1 else end + 1
                buf.append(source[i:end])
                i = end
                continue

            env = _ENV_RE.match(source, i)
            if env:
                name = env.group(2)
                j = env.end()
                if name in VERBATIM_ENVIRONMENTS:
                    closing = re.compile(
                        r"\\end\s*\{" + re.escape(name) + r"\}").search(source, j)
                    stop = closing.start() if closing else n
                    emit({"kind": "verbatim", "name": name, "raw": source[j:stop],
                          "start": body_offset + j, "end": body_offset + stop})
                    if closing:
                        emit({"kind": "env_end", "name": name,
                              "start": body_offset + closing.start(),
                              "end": body_offset + closing.end()})
                        i = closing.end()
                    else:
                        i = stop
                    continue
                # \begin{align}[t] / \begin{algorithmic}[1]: placement info, not content.
                k = j
                while k < n and source[k].isspace():
                    k += 1
                if env.group(1) == "begin" and k < n and source[k] == "[":
                    _, j = read_optional_arg(source, j)
                is_begin = env.group(1) == "begin"
                # The stack is updated *after* emit so that the text run closing
                # the previous environment is still tagged with that environment.
                emit({"kind": "env_begin" if is_begin else "env_end",
                      "name": name, "start": body_offset + i, "end": body_offset + j})
                if is_begin:
                    env_stack.append((name, i))
                else:
                    for index in range(len(env_stack) - 1, -1, -1):
                        if env_stack[index][0] == name:
                            env_stack = env_stack[:index]
                            break
                i = j
                continue

            cmd = _CMD_WORD_RE.match(source, i)
            if cmd:
                name = cmd.group(1)
                if name in SECTION_LEVELS:
                    arg, j, _, _ = read_group(source, cmd.end())
                    if arg is not None:
                        emit({"kind": "heading", "command": name,
                              "level": SECTION_LEVELS[name], "arg": arg,
                              "raw": source[i:j],
                              "start": body_offset + i, "end": body_offset + j})
                        i = j
                        continue
                elif name == "item":
                    label, j = read_optional_arg(source, cmd.end())
                    emit({"kind": "item", "label": label, "raw": source[i:j],
                          "start": body_offset + i, "end": body_offset + j})
                    i = j
                    continue
                # Any other command stays in the text buffer together with its
                # balanced arguments, so nested markup is never re-scanned.
                j = cmd.end()
                groups, j = _consume_groups(source, j)
                buf.append(source[i:cmd.end()]
                           + "".join("{%s}" % g for g in groups))
                i = j
                continue

            # Escaped literal such as \{ \} \% \\ \& \$ \|
            buf.append(source[i:i + 2])
            i += 2
            continue

        if ch == "$" and source.startswith("$$", i):
            end = source.find("$$", i + 2)
            end = n if end == -1 else end + 2
            emit({"kind": "math", "raw": source[i:end],
                  "start": body_offset + i, "end": body_offset + end})
            i = end
            continue

        if ch == "$":
            j = i + 1
            while j < n:
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == "\n" and source[j - 1] != "\\":
                    break
                if source[j] == "$":
                    j += 1
                    break
                j += 1
            if not buf:
                buf_start = i
            buf.append(source[i:j])
            i = j
            continue

        if not buf:
            buf_start = i
        buf.append(ch)
        i += 1

    flush(n)
    return events


# ---------------------------------------------------------------------------
# Theorem-like environment declarations
# ---------------------------------------------------------------------------
def parse_newtheorem(source: str) -> dict:
    """Map theorem-like environment names to their declared statement kind."""
    mapping = dict(STATEMENT_ENVIRONMENTS)
    for match in _NEWTHEOREM_RE.finditer(source):
        env_name, label = match.group(1).strip(), match.group(2).strip()
        kind = label.split()[-1].lower() if label else "theorem"
        mapping[env_name] = kind if kind in NODE_TYPES else "theorem"
        mapping[env_name + "*"] = mapping[env_name]
    return mapping


def classify_environment(name: str, statements: dict) -> str:
    """Node type for an environment; ``other`` only when nothing fits."""
    if name in statements:
        kind = statements[name]
        return kind if kind in NODE_TYPES else "other"
    if name in ENVIRONMENT_TYPES:
        return ENVIRONMENT_TYPES[name]
    return "other"


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------
def _new_node(node_id, node_type, parent, span, environment=None):
    return {
        "id": node_id,
        "node_type": node_type,
        "parent": parent,
        "text": "",
        "latex": "",
        "tokens": [],
        "positions": [],
        "seq_ids": [],
        "token_count": 0,
        "source_span": [span[0], span[1]],
        "child_ids": [],
        "labels": [],
        "references": [],
        "unresolved_commands": [],
        **({"environment": environment} if environment else {}),
    }


def parse_title(source: str, fallback: str):
    """Return (title, raw latex of the \\title argument)."""
    match = _TITLE_RE.search(source)
    if not match:
        return fallback, ""
    title, _, cs, ce = read_group(source, match.end())
    return render(title or "").strip() or fallback, (title or "")


def _split_paragraphs(raw: str, abs_start: int):
    """Split a raw text run into blank-line separated paragraphs with offsets."""
    spans = []
    n = len(raw)
    i = 0
    start = None
    while i < n:
        j = raw.find("\n", i)
        j = n if j == -1 else j
        if raw[i:j].strip():
            if start is None:
                start = i
        elif start is not None:
            spans.append((start, i))
            start = None
        i = j + 1
    if start is not None:
        spans.append((start, n))
    return [(abs_start + a, abs_start + b, raw[a:b])
            for a, b in spans if raw[a:b].strip()]


def extract_structure(source: str, document_id: str):
    """Build the node tree for one document.  Returns ``(title, nodes, info)``."""
    cleaned = strip_comments(source)
    statements = parse_newtheorem(cleaned)
    title, title_raw = parse_title(cleaned, document_id)

    begin = re.search(r"\\begin\s*\{document\}", cleaned)
    end = re.search(r"\\end\s*\{document\}", cleaned)
    if begin:
        body_start = begin.end()
        body_end = end.start() if end and end.start() > body_start else len(cleaned)
    else:
        body_start, body_end = 0, len(cleaned)
    body = cleaned[body_start:body_end]

    events = scan(body, body_offset=body_start)

    root = _new_node("n0", "document", None, (0, len(source)))
    root["latex"] = title_raw
    root["text"] = title
    nodes = [root]
    counter = [0]
    own_parts = {}
    own_span = {}
    own_mode_by_id = {}
    dropped_seen = set()
    reparented = []

    def add(node_type, parent, span, environment=None, text="", latex="", result=None,
            is_heading=False):
        counter[0] += 1
        node = _new_node("n%d" % counter[0], node_type, parent["id"], span, environment)
        node["text"] = text
        node["latex"] = latex
        if is_heading:
            node["is_heading"] = True
        nodes.append(node)
        parent["child_ids"].append(node["id"])
        own_span[node["id"]] = [span[0], span[1]]
        if result is not None:
            record(node, result)
        return node

    def record(node, result):
        for label in result.labels:
            if label not in node["labels"]:
                node["labels"].append(label)
        for ref in result.references:
            if ref not in node["references"]:
                node["references"].append(ref)
        node["unresolved_commands"] = sorted(
            set(node["unresolved_commands"]) | set(result.unresolved))

    def render_node_text(raw, math=False, code=False):
        result = convert_text(raw, math=math, code=code)
        dropped_seen.update(result.dropped)
        return result

    current = root      # node that receives new children
    owner = None        # node that accumulates its own text (None -> paragraphs)
    owner_mode = (False, False)   # (math, code) of the owning environment
    section_stack = [(-1, root)]
    env_frames = []     # [(current, owner, section_depth, owner_mode)]
    open_envs = []      # [(name, node)]

    def own(node, raw, start, end):
        own_parts.setdefault(node["id"], []).append(raw)
        own_mode_by_id[node["id"]] = owner_mode
        span = own_span.setdefault(node["id"], [start, end])
        span[0] = min(span[0], start)
        span[1] = max(span[1], end)

    def route_text(raw, start, math=False, code=False):
        if not raw.strip():
            return
        if owner is not None:
            own(owner, raw, start, start + len(raw))
            return
        for a, b, chunk in _split_paragraphs(raw, start):
            result = render_node_text(chunk, math, code)
            if not result.text.strip():
                continue  # e.g. a lone \maketitle: no content, no node
            add("paragraph", current, (a, b), text=result.text, latex=chunk)
            record(nodes[-1], result)

    for event in events:
        kind = event["kind"]

        if kind == "text":
            route_text(event["raw"], event["start"], event.get("math"), event.get("code"))

        elif kind == "verbatim":
            add("other", current, (event["start"], event["end"]),
                environment=event["name"], text=event["raw"], latex=event["raw"])

        elif kind == "math":
            result = render_node_text(event["raw"], math=True)
            add("equation", current, (event["start"], event["end"]),
                text=result.text, latex=event["raw"], result=result)

        elif kind == "heading":
            level = event["level"]
            while len(section_stack) > 1 and section_stack[-1][0] >= level:
                section_stack.pop()
            parent = section_stack[-1][1] if section_stack else root
            command = event["command"]
            node_type = ("section" if command in {"part", "chapter", "section"}
                         else "subsection" if command == "subsection"
                         else "subsubsection" if command == "subsubsection"
                         else "paragraph")
            result = render_node_text(event["arg"])
            heading = add(node_type, parent, (event["start"], event["end"]),
                          text=result.text, latex=event["raw"], result=result,
                          is_heading=True)
            heading["_heading_level"] = level
            section_stack.append((level, heading))
            current = heading
            owner = None

        elif kind == "item":
            result = render_node_text(event["label"] or "")
            node = add("item", current, (event["start"], event["end"]),
                       text=result.text, latex=event["raw"], result=result)
            owner = node
            owner_mode = (False, False)
            own_span[node["id"]] = [event["end"], event["end"]]

        elif kind == "env_begin":
            name = event["name"]
            node = add(classify_environment(name, statements), current,
                       (event["start"], event["start"]), environment=name)
            own_span[node["id"]] = [event["end"], event["end"]]
            env_frames.append((current, owner, len(section_stack), owner_mode))
            open_envs.append((name, node))
            current, owner = node, node
            owner_mode = (name in MATH_ENVIRONMENTS, name in CODE_ENVIRONMENTS)

        elif kind == "env_end":
            name = event["name"]
            for index in range(len(open_envs) - 1, -1, -1):
                if open_envs[index][0] == name:
                    node = open_envs[index][1]
                    open_envs = open_envs[:index]
                    break
            else:
                continue
            node["source_span"][1] = max(node["source_span"][1], event["end"])
            while len(env_frames) > len(open_envs):
                current, owner, depth, owner_mode = env_frames.pop()
                del section_stack[depth:]

    # Close environments that were never explicitly closed.
    while open_envs:
        open_envs.pop()
        if env_frames:
            current, owner, depth, owner_mode = env_frames.pop()
            del section_stack[depth:]

    # Materialise accumulated own-text.
    for node in nodes:
        parts = [p for p in own_parts.get(node["id"], []) if p.strip()]
        if parts:
            node["latex"] = "\n".join(parts)
            math_mode, code_mode = own_mode_by_id.get(node["id"], (False, False))
            result = render_node_text(node["latex"], math_mode, code_mode)
            node["text"] = result.text
            record(node, result)
            dropped_seen.update(result.dropped)
            span = own_span.get(node["id"])
            if span:
                node["source_span"] = [node["source_span"][0], max(node["source_span"][1], span[1])]
            if result.captions and node["node_type"] == "algorithm":
                node["caption"] = render(result.captions[0])
        if node["id"] != "n0" and not node["latex"].strip():
            node["latex"] = node["text"]

    # Bottom-up "has content" pass (nodes are in pre-order, so reverse works).
    has_content = {}
    for node in reversed(nodes):
        has_content[node["id"]] = bool(node["text"].strip()) or any(
            has_content.get(c, False) for c in node["child_ids"])

    kept = [n for n in nodes if n["id"] == "n0" or has_content.get(n["id"])]
    kept_ids = {n["id"] for n in kept}
    for node in kept:
        node["child_ids"] = [c for c in node["child_ids"] if c in kept_ids]
        if node["id"] != "n0" and node["parent"] not in kept_ids:
            reparented.append(node["id"])
            node["parent"] = "n0"
            root["child_ids"].append(node["id"])

    # Emit in document order (pre-order DFS) so packed positions follow source.
    lookup = {n["id"]: n for n in kept}
    order = []
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(reversed([lookup[c] for c in node["child_ids"]]))

    _close_heading_spans(order, lookup, body_end)
    for node in order:
        node.pop("_heading_level", None)

    info = {"dropped": sorted(dropped_seen), "reparented": sorted(reparented),
            "body_span": [body_start, body_end]}
    return title, order, info


def _close_heading_spans(order, lookup, body_end):
    """Give every heading node the source extent of its whole section body.

    The scanner can only see the ``\\section{...}`` marker itself; a section's
    real extent runs until the next heading of the same or a higher level.
    Without this a section's span would be shorter than its children's, which
    would break the containment invariant checked by :func:`validate_tree`.

    Sections still open at the end of the document all run to the end of their
    container.  That bound must be resolved from the *outermost* section first:
    capping an inner section by its parent's span would use the parent's
    unextended marker span and cut the inner section off from its own children.
    """
    open_sections = []
    for node in order:
        level = node.get("_heading_level")
        if level is None:
            continue
        while open_sections and open_sections[-1][0] >= level:
            _, open_node = open_sections.pop()
            open_node["source_span"][1] = node["source_span"][0]
        open_sections.append((level, node))
    if not open_sections:
        return
    _, outermost = open_sections[0]
    parent = lookup.get(outermost["parent"])
    bound = parent["source_span"][1] if parent else body_end
    bound = min(bound, body_end)
    for _, node in open_sections:      # outermost first: all share the bound
        node["source_span"][1] = bound


def extract_document(source: str, document_id: str):
    """Backwards-compatible wrapper: returns ``(title, nodes)``."""
    title, nodes, _ = extract_structure(source, document_id)
    return title, nodes


# ---------------------------------------------------------------------------
# Content-loss audit
# ---------------------------------------------------------------------------
def audit_content_loss(source: str, nodes) -> dict:
    """Verify that no control sequence was silently dropped.

    Every ``\\command`` in the document body must either appear in the ``latex``
    of some node or be an explicitly allow-listed content-free command.
    """
    cleaned = strip_comments(source)
    begin = re.search(r"\\begin\s*\{document\}", cleaned)
    body = cleaned[begin.end():] if begin else cleaned

    def commands(text):
        return {m.group(1) for m in _ANY_COMMAND_RE.finditer(text)} - {"begin", "end"}

    source_commands = commands(body)
    node_latex = "\n".join(n.get("latex", "") for n in nodes)
    kept_commands = commands(node_latex)

    return {
        "lost": sorted(c for c in source_commands - kept_commands
                       if c not in DROPPED_COMMANDS),
        "invented": sorted(kept_commands - source_commands),
        "dropped": sorted(source_commands - kept_commands),
    }


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def tokenize_nodes(nodes, model_path, n_ctx, vocab_size=None):
    """Tokenize each node's text independently with the GGUF vocabulary.

    Returns ``(total_tokens, tokenizer_metadata)``.
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "llama-cpp-python is not installed in this Python environment; run the "
            "parser with .\\.venv\\Scripts\\python.exe"
        ) from exc
    llm = Llama(model_path=str(model_path), vocab_only=True, n_ctx=n_ctx, verbose=False)
    position = 0
    try:
        for node in nodes:
            text = node["text"]
            toks = llm.tokenize(text.encode("utf-8"), add_bos=False, special=True) if text else []
            node["tokens"] = [int(t) for t in toks]
            node["positions"] = list(range(position, position + len(toks)))
            node["seq_ids"] = [0] * len(toks)
            node["token_count"] = len(toks)
            position += len(toks)
    finally:
        del llm
    metadata = {
        "kind": "llama-cpp GGUF vocabulary (model tokenizer)",
        "model_file": str(model_path),
        "model_file_name": Path(model_path).name,
        "vocab_size": int(vocab_size) if vocab_size else 128256,
        "granularity": "per_node",
        "input_field": "text",
        "add_bos": False,
        "special": True,
        "total_tokens": position,
        "note": ("Nodes are tokenized independently, so concatenating node tokens "
                 "can differ by a token or two from tokenizing the whole document "
                 "in one call."),
    }
    return position, metadata


# ---------------------------------------------------------------------------
# Sequence assignment
# ---------------------------------------------------------------------------

# Node types that open a top-level block.  Each block becomes one llama.cpp
# sequence, so a question about one section can be masked to that section.
BLOCK_ROOT_TYPES = ("abstract", "section", "subsection", "subsubsection")


def _subtree(lookup, node):
    """Pre-order list of the node and all of its descendants."""
    out = [node]
    stack = list(reversed(node["child_ids"]))
    while stack:
        current = lookup[stack.pop()]
        out.append(current)
        stack.extend(reversed(current["child_ids"]))
    return out


def _position_span(nodes):
    positions = [p for node in nodes for p in node["positions"]]
    if not positions:
        return [0, -1]
    return [min(positions), max(positions)]


def _group_members(roots, lookup, spine):
    """Nodes owned by one group.

    A block owns its whole subtree, but the document spine owns only its own
    tokens: its subtree is the entire document, which would make every group
    overlap every other one.
    """
    members = []
    for root in roots:
        if root is spine:
            members.append(root)
        else:
            members.extend(_subtree(lookup, root))
    return members


def _group_record(roots, lookup, spine):
    members = _group_members(roots, lookup, spine)
    span = _position_span(members)
    return {
        "node_ids": [root["id"] for root in roots],
        "title": roots[0]["text"][:80].strip(),
        "node_count": len(members),
        "token_count": sum(node["token_count"] for node in members),
        "pos_min": span[0],
        "pos_max": span[1],
        "merged": len(roots) > 1,
    }


def _group_token_count(roots, lookup, spine):
    return sum(node["token_count"] for node in _group_members(roots, lookup, spine))


def _merge_groups(group_roots, max_sequences, lookup, spine):
    """Fold adjacent groups together until the sequence budget is met.

    Adjacency matters: a merged sequence must cover a contiguous position
    range, otherwise the driver cannot evict it with a single
    ``llama_memory_seq_rm`` call.
    """
    while len(group_roots) > max_sequences:
        # Fold the adjacent pair with the smallest combined size, so the
        # largest blocks keep their own sequence.
        best, best_size = 0, None
        for i in range(len(group_roots) - 1):
            size = _group_token_count(group_roots[i] + group_roots[i + 1],
                                      lookup, spine)
            if best_size is None or size < best_size:
                best, best_size = i, size
        group_roots[best] = group_roots[best] + group_roots[best + 1]
        del group_roots[best + 1]
    return group_roots


def assign_sequences(nodes, mode="single", max_sequences=16,
                     min_block_tokens=0, xref_links=False):
    """Assign a logical ``seq_id`` to every token.

    ``single`` keeps the whole document in sequence 0.  ``blocks`` gives every
    top-level block its own sequence so that a block can be evicted, mutated or
    masked independently, and records the result in ``metadata.blocks``.

    A token belongs to exactly one sequence.  Cross-references are *not*
    expressed by giving a citing node a second sequence: llama.cpp requires a
    sequence's positions to be contiguous, and a citing node usually sits in the
    middle of another block's range, so a second membership would produce a
    non-contiguous sequence.  The links are instead recorded in
    ``metadata.sequence_assignment.cross_references`` and the caller resolves
    them when it assembles the context it asks about.

    Returns a metadata dict for ``metadata.sequence_assignment``.
    """
    if mode not in ("single", "blocks"):
        raise ValueError("unknown seq mode: %s" % mode)

    lookup = {node["id"]: node for node in nodes}
    root = lookup["n0"]
    for node in nodes:
        node["seq_ids"] = [0] * len(node["tokens"])

    if mode == "single":
        # In single mode sequence 0 holds the whole document, so its record
        # spans every position rather than just the title.
        spine = _group_record([root], lookup, root)
        spine["seq_id"] = 0
        spine["title"] = "whole document"
        spine["node_count"] = len(nodes)
        spine["token_count"] = sum(node["token_count"] for node in nodes)
        positions = _position_span(nodes)
        spine["pos_min"], spine["pos_max"] = positions
        return {
            "mode": "single",
            "sequence_count": 1,
            "max_sequences": max_sequences,
            "min_block_tokens": min_block_tokens,
            "xref_links": False,
            "cross_references": [],
            "blocks": [spine],
        }

    blocks = [lookup[cid] for cid in root["child_ids"]
              if lookup[cid]["node_type"] in BLOCK_ROOT_TYPES]

    # A very small block is not worth its own sequence: fold it into the
    # preceding block.  The first block always starts its own group, so there is
    # always something to fold a tiny leading block into (the document spine).
    group_roots = [[root]]
    for block in blocks:
        size = _group_token_count([block], lookup, root)
        if len(group_roots) > 1 and 0 < min_block_tokens and size < min_block_tokens:
            group_roots[-1].append(block)
        else:
            group_roots.append([block])

    group_roots = _merge_groups(group_roots, max_sequences, lookup, root)
    records = [_group_record(roots, lookup, root) for roots in group_roots]
    for seq_id, record in enumerate(records):
        record["seq_id"] = seq_id

    seq_of_node = {}
    for record in records:
        for node_id in record["node_ids"]:
            members = _group_members([lookup[node_id]], lookup, root)
            for member in members:
                seq_of_node[member["id"]] = record["seq_id"]

    # ``seq_ids`` stays one integer per token (the token's own block, which is
    # what the schema promises).  Cross-references are recorded rather than
    # folded into the sequence ids: a citing node usually sits in the middle of
    # another block's range, and llama.cpp requires a sequence's positions to be
    # contiguous, so a second membership would produce an invalid sequence.
    cross_references = []
    label_owner = {}
    for node in nodes:
        for label in node.get("labels", []):
            label_owner.setdefault(label, node)

    for node in nodes:
        own_seq = seq_of_node.get(node["id"], 0)
        for ref in node.get("references", []):
            target = label_owner.get(ref)
            if target is None:
                continue
            target_seq = seq_of_node.get(target["id"])
            if target_seq is None or target_seq == own_seq:
                continue
            link = {
                "from_node": node["id"],
                "from_seq": own_seq,
                "target_node": target["id"],
                "target_seq": target_seq,
                "label": ref,
            }
            if xref_links and link not in cross_references:
                cross_references.append(link)
        node["seq_ids"] = [own_seq] * len(node["tokens"])

    return {
        "mode": "blocks",
        "sequence_count": len(records),
        "max_sequences": max_sequences,
        "min_block_tokens": min_block_tokens,
        "xref_links": bool(xref_links),
        "cross_references": cross_references,
        "blocks": records,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_tree(nodes, source_length=None, sequence_count=1):
    if not nodes or nodes[0]["id"] != "n0":
        raise ValueError("First node must be root n0.")
    ids = [n["id"] for n in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate node IDs.")
    lookup = {n["id"]: n for n in nodes}
    if nodes[0]["parent"] is not None or nodes[0]["node_type"] != "document":
        raise ValueError("Root must be a document node with null parent.")
    for node in nodes:
        if node["id"] != "n0" and node["parent"] not in lookup:
            raise ValueError("Missing parent for %s: %s" % (node["id"], node["parent"]))
        if not (len(node["tokens"]) == len(node["positions"]) == len(node["seq_ids"])):
            raise ValueError("Token/position/seq_id lengths differ for %s" % node["id"])
        if node.get("token_count") != len(node["tokens"]):
            raise ValueError("token_count disagrees with tokens for %s" % node["id"])
        if set(node["seq_ids"]) - set(range(sequence_count)):
            raise ValueError("seq_ids outside [0, %d) for %s"
                             % (sequence_count, node["id"]))
        seen = set()
        cursor = node
        while cursor["parent"] is not None:
            if cursor["id"] in seen:
                raise ValueError("Cycle detected at %s" % node["id"])
            seen.add(cursor["id"])
            cursor = lookup[cursor["parent"]]
        for child in node["child_ids"]:
            if child not in lookup:
                raise ValueError("child_ids of %s points at missing node %s" % (node["id"], child))
            if lookup[child]["parent"] != node["id"]:
                raise ValueError("child_ids of %s disagrees with parent of %s" % (node["id"], child))

    positions = [p for node in nodes for p in node["positions"]]
    if positions != sorted(positions):
        raise ValueError("Packed positions are not ordered by node-list traversal.")

    # llama.cpp refuses a sequence whose positions are not contiguous, so a
    # token may not join a second sequence unless doing so keeps that sequence
    # contiguous.  One token per sequence is always contiguous.
    by_seq = {}
    for node in nodes:
        for position, seq in zip(node["positions"], node["seq_ids"]):
            by_seq.setdefault(seq, []).append(position)
    for seq, seq_positions in by_seq.items():
        ordered = sorted(seq_positions)
        if len(set(ordered)) != len(ordered):
            raise ValueError("sequence %d owns a position twice" % seq)
        expected = list(range(ordered[0], ordered[0] + len(ordered)))
        if ordered != expected:
            raise ValueError("positions of sequence %d are not contiguous" % seq)
    if len(positions) != len(set(positions)):
        raise ValueError("Token positions are not unique.")
    if positions != list(range(len(positions))):
        raise ValueError("Packed positions are not contiguous from 0.")

    if source_length is not None:
        for node in nodes:
            span = node.get("source_span")
            if not span:
                continue
            if not (0 <= span[0] <= span[1] <= source_length):
                raise ValueError("source_span out of range for %s" % node["id"])
            parent = lookup[node["parent"]] if node["parent"] else None
            if parent is not None and parent.get("source_span"):
                pspan = parent["source_span"]
                if not (pspan[0] <= span[0] and span[1] <= pspan[1]):
                    raise ValueError("source_span of %s escapes parent %s"
                                     % (node["id"], node["parent"]))
        for node in nodes:
            # Heading nodes span their whole section, so a sibling heading can
            # legitimately start inside an earlier sibling's extent.
            if node.get("is_heading"):
                continue
            previous_end = None
            for child_id in node["child_ids"]:
                span = lookup[child_id].get("source_span")
                if span and previous_end is not None and span[0] < previous_end:
                    raise ValueError("overlapping sibling spans under %s (%s)"
                                     % (node["id"], child_id))
                if span:
                    previous_end = span[1]


def validate_schema(data, schema_path):
    if not schema_path:
        return
    if jsonschema is None:
        raise RuntimeError("jsonschema is required when --schema is supplied.")
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    jsonschema.validate(instance=data, schema=schema)


def merge_small_siblings(nodes, threshold):
    """Optional, lossy merge of adjacent small sibling nodes (off by default)."""
    if threshold <= 0:
        return nodes
    lookup = {n["id"]: n for n in nodes}
    removed = set()
    for parent in nodes:
        group = []

        def flush(group):
            if len(group) >= 2 and sum(len(n["text"].split()) for n in group) <= threshold:
                first = group[0]
                first["text"] = " ".join(n["text"] for n in group if n["text"]).strip()
                first["latex"] = "\n".join(n.get("latex", "") for n in group)
                first["source_span"] = [min(n["source_span"][0] for n in group),
                                        max(n["source_span"][1] for n in group)]
                first["merged_from"] = [n["id"] for n in group]
                for extra in group[1:]:
                    for child in extra.get("child_ids", []):
                        if child not in parent["child_ids"]:
                            parent["child_ids"].append(child)
                        lookup[child]["parent"] = parent["id"]
                    first["labels"] = sorted(set(first["labels"]) | set(extra["labels"]))
                    first["references"] = sorted(
                        set(first["references"]) | set(extra["references"]))
                removed.update(n["id"] for n in group[1:])
            return []

        for child_id in list(parent.get("child_ids", [])):
            node = lookup[child_id]
            if (node["node_type"] in SMALL_NODE_TYPES
                    and len(node["text"].split()) <= threshold):
                group.append(node)
            else:
                group = flush(group)
        flush(group)

    for parent in nodes:
        parent["child_ids"] = [c for c in parent.get("child_ids", []) if c not in removed]
    result = [n for n in nodes if n["id"] not in removed]
    retained = {n["id"] for n in result}
    for node in result:
        if node["id"] != "n0" and node["parent"] not in retained:
            node["parent"] = "n0"
    return result


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------
def build_document(source: str, document_id: str, source_name: str, model_path,
                   n_ctx: int = 4096, merge_threshold: int = 0,
                   seq_mode: str = "single", max_sequences: int = 16,
                   min_block_tokens: int = 0, xref_links: bool = False):
    """Parse, audit, (optionally merge), tokenize and assemble one document.

    Merging happens *before* tokenization so that tokens, packed positions and
    the reported total always describe exactly the node list that is emitted.
    Sequence assignment happens *after* tokenization because it needs the
    packed positions and token counts of every node.
    """
    title, nodes, info = extract_structure(source, document_id)
    audit = audit_content_loss(source, nodes)
    if audit["lost"]:
        raise ValueError("content loss detected, refusing to emit: %s"
                         % ", ".join(audit["lost"]))
    if info["reparented"]:
        raise ValueError("nodes were orphaned and re-parented to the root: %s"
                         % ", ".join(info["reparented"]))
    if merge_threshold > 0:
        nodes = merge_small_siblings(nodes, merge_threshold)
    total, token_metadata = tokenize_nodes(nodes, model_path, n_ctx)
    assignment = assign_sequences(nodes, seq_mode, max_sequences,
                                  min_block_tokens, xref_links)
    unresolved = sorted({c for node in nodes for c in node.get("unresolved_commands", [])})
    sequence_convention = ("single_sequence_per_document" if seq_mode == "single"
                           else "one_sequence_per_top_level_block")
    data = {
        "document_id": document_id,
        "title": title,
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "source_file": source_name,
            "source_length": len(source),
            "node_count": len(nodes),
            "nonempty_node_count": sum(1 for n in nodes if n["text"].strip()),
            "node_type_counts": _count_types(nodes),
            "tokenization": token_metadata,
            "position_convention": "packed_global",
            "sequence_convention": sequence_convention,
            "sequence_assignment": assignment,
            "ownership_convention": (
                "Each node owns only its own text; descendant text is never copied "
                "into ancestors, so every token is assigned to exactly one node."),
            "text_convention": (
                "text is a deterministic meaning-preserving linear rendering of the "
                "node's latex field. Grouping braces are retained so subscripts, "
                "superscripts and grouping stay explicit; unmapped control sequences "
                "are preserved verbatim."),
            "dropped_content_free_commands": info["dropped"],
            "unresolved_commands": {"count": len(unresolved), "commands": unresolved},
            "content_loss_audit": audit,
        },
        "nodes": nodes,
    }
    return data, total


def _count_types(nodes) -> dict:
    counts = {}
    for node in nodes:
        counts[node["node_type"]] = counts.get(node["node_type"], 0) + 1
    return dict(sorted(counts.items()))


def main():
    ap = argparse.ArgumentParser(
        description="Parse LaTeX documents into schema-valid JSON node trees.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=".")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--schema", default="schema/document_schema.json")
    ap.add_argument("--model",
                    default=r"C:\Users\muffa\LDLS Project 0\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--merge-threshold", type=int, default=0)
    ap.add_argument("--seq-mode", choices=("single", "blocks"), default="single",
                    help="single = every token is seq 0; blocks = one sequence "
                         "per top-level block for independent eviction")
    ap.add_argument("--max-sequences", type=int, default=16,
                    help="sequence budget; adjacent blocks are merged until it holds")
    ap.add_argument("--min-block-tokens", type=int, default=0,
                    help="blocks smaller than this are merged into the previous block")
    ap.add_argument("--xref-links", action="store_true",
                    help="give a node the seq_id of every block it cites, so "
                         "cross-references can attend outside their own block")
    ap.add_argument("--backup-dir", default=None,
                    help="copy existing outputs here before overwriting them")
    ap.add_argument("--force-backup", action="store_true",
                    help="overwrite files that already exist in --backup-dir")
    ap.add_argument("--files", nargs="*", default=DEFAULT_INPUTS)
    args = ap.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    if not model_path.exists():
        ap.error("GGUF model not found: %s" % model_path)

    if args.backup_dir:
        backup = Path(args.backup_dir)
        backup.mkdir(parents=True, exist_ok=True)
        for existing in sorted(output_dir.glob("document_*.json")):
            target = backup / existing.name
            if target.exists() and not args.force_backup:
                # Never clobber a baseline that already exists: re-running the
                # parser must not silently destroy the pre-change reference.
                print("[BACKUP] kept existing %s" % target)
                continue
            shutil.copy2(existing, target)
            print("[BACKUP] %s -> %s" % (existing.name, target))

    failures = 0
    for filename in args.files:
        source_path = input_dir / filename
        if not source_path.exists():
            print("[FAIL] Missing input: %s" % source_path, file=sys.stderr)
            failures += 1
            continue
        try:
            source = source_path.read_text(encoding="utf-8")
            data, total = build_document(source, source_path.stem, filename,
                                         model_path, args.n_ctx,
                                         args.merge_threshold, args.seq_mode,
                                         args.max_sequences,
                                         args.min_block_tokens, args.xref_links)
            if total > args.max_tokens:
                raise ValueError("%d packed node tokens exceeds limit %d"
                                 % (total, args.max_tokens))
            validate_tree(data["nodes"], len(source),
                          data["metadata"]["sequence_assignment"]["sequence_count"])
            validate_schema(data, args.schema)
            out = output_dir / ("%s.json" % source_path.stem)
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
            print("[OK] %s: %d nodes, %d packed node tokens, %d sequence(s) -> %s"
                  % (source_path.name, len(data["nodes"]), total,
                     data["metadata"]["sequence_assignment"]["sequence_count"], out))
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print("[FAIL] %s: %s" % (source_path.name, exc), file=sys.stderr)
    if failures:
        raise SystemExit(1)
    print("Successfully parsed %d document(s)." % len(args.files))


if __name__ == "__main__":
    main()
