// Structured-prefill driver for LaTeX subtree experiments.
//
// Loads a parser JSON document, prefills it with one llama.cpp sequence per
// top-level block, asks a question, and can evict or replace a single subtree
// by its position range.
//
// Run with:
//   llama-latex-driver -m model.gguf -f doc.json --mode ask -n 32
//
// The JSON files are produced by latex_efficient_attention_dataset/parser.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::json;

// Room kept for a second prompt in mode two-prompts, so that the batched run
// and the single-prompt reference run allocate the same cache.
static const int PROMPT_RESERVE = 32;

// One sequence as described by metadata.sequence_assignment.blocks.
struct block {
    int         seq_id    = 0;
    std::string title;
    llama_pos   pos_min   = 0;
    llama_pos   pos_max   = 0;
    int         tokens    = 0;
    bool        merged    = false;
};

// A resolved \ref: from_seq refers to target_seq.
struct cross_reference {
    int from_seq   = 0;
    int target_seq = 0;
    std::string label;
};

// A token as the parser emitted it: real model token ID, packed position and
// the sequence it belongs to.
struct doc_token {
    llama_token          id    = 0;
    llama_pos            pos   = 0;
    std::vector<llama_seq_id> seq_ids;
};

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("    %s -m model.gguf --doc doc.json --mode ask -n 32\n", argv[0]);
    LOG("\n    options:\n");
    LOG("      --doc <file>          parser JSON document (required)\n");
    LOG("      -q <text>             the question to ask\n");
    LOG("      --focus-seq <n>       answer from this block only, default: all\n");
    LOG("      --evict-seq <n>       block to evict or replace\n");
    LOG("      --replace-file <file> replacement text for mode replace\n");
    LOG("      --prompt-a <text>     first prompt for mode two-prompts\n");
    LOG("      --prompt-b <text>     second prompt for mode two-prompts\n");
    LOG("      --logits-out <file>   write the per-step top logits as TSV\n");
    LOG("      --offset-a <n>       where prompt A starts in mode two-prompts\n");
    LOG("      --no-follow-xrefs    do not gather the blocks a block cites\n");
    LOG("      --joint-prefill      submit both prompts in one batch\n");
    LOG("      --kv-unified         use one shared KV buffer\n");
    LOG("      --debug-logits       print the per-step top logit to the log\n");
    LOG("\n    modes:\n");
    LOG("      stats    report the tree and sequence layout, run nothing\n");
    LOG("      two-prompts  run two prompts as two sequences and print both\n");
    LOG("                   (omit --prompt-b for a single-prompt reference run)\n");
    LOG("      prefill  load the document into the KV cache, print nothing\n");
    LOG("      ask      ask a question and generate greedily\n");
    LOG("      evict    prefill, evict one block, ask again\n");
    LOG("      replace  prefill, evict one block, refill it, ask again\n");
    LOG("\n");
}

// ---------------------------------------------------------------------------
// JSON loading
// ---------------------------------------------------------------------------
static bool load_document(const std::string & path,
                          std::vector<doc_token> & tokens,
                          std::vector<block> & blocks,
                          std::vector<cross_reference> & xrefs,
                          std::string & title,
                          int & n_seq_max_needed) {
    std::ifstream file(path);
    if (!file) {
        LOG_ERR("%s: cannot open %s\n", __func__, path.c_str());
        return false;
    }
    json data = json::parse(file);

    title = data.value("title", "");

    const json & assignment = data.at("metadata").at("sequence_assignment");
    for (const auto & entry : assignment.at("blocks")) {
        block b;
        b.seq_id  = entry.at("seq_id");
        b.title   = entry.value("title", "");
        b.pos_min = entry.at("pos_min");
        b.pos_max = entry.at("pos_max");
        b.tokens  = entry.at("token_count");
        b.merged  = entry.value("merged", false);
        blocks.push_back(b);
        n_seq_max_needed = std::max(n_seq_max_needed, b.seq_id + 1);
    }

    if (assignment.contains("cross_references")) {
        for (const auto & link : assignment.at("cross_references")) {
            cross_reference x;
            x.from_seq   = link.at("from_seq");
            x.target_seq = link.at("target_seq");
            x.label      = link.value("label", "");
            xrefs.push_back(x);
        }
    }

    for (const auto & node : data.at("nodes")) {
        const json & toks = node.at("tokens");
        const json & poss = node.at("positions");
        const json & seqs = node.at("seq_ids");

        // A token belongs to exactly one sequence: llama.cpp requires a
        // sequence's positions to be contiguous, so a token cannot join a
        // second one.  Cross-references are resolved by gather_context().
        for (size_t i = 0; i < toks.size(); ++i) {
            doc_token t;
            t.id       = toks[i].get<llama_token>();
            t.pos      = poss[i].get<llama_pos>();
            t.seq_ids.push_back(seqs[i].get<llama_seq_id>());
            tokens.push_back(t);
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------------

static bool     g_debug_logits = false;
static FILE *   g_logits_file  = NULL;

// Per-step trace of a generated token: the token chosen and its top logit.
// Comparing traces tells a real difference in context apart from float noise,
// because noise moves the logits slightly while a different context moves them
// a lot.
static void trace_step(int i, llama_token token, const char * label,
                       llama_context * ctx, int32_t idx) {
    if (g_debug_logits == false && g_logits_file == NULL) {
        return;
    }
    const float * logits = llama_get_logits_ith(ctx, idx);
    float top = 0.0f;
    if (logits != NULL) {
        top = logits[0];
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(
                llama_get_model(ctx)));
        for (int v = 1; v < n_vocab; ++v) {
            top = std::max(top, logits[v]);
        }
    }
    if (g_debug_logits) {
        LOG_INF("%s: %s step %d token %d top_logit %.6f\n",
                __func__, label, i, token, top);
    }
    if (g_logits_file != NULL) {
        fprintf(g_logits_file, "%s\t%d\t%d\t%.6f\n", label, i, token, top);
    }
}

// Sample a token and record it together with the logit that produced it.
// Tracing inside the sampling call keeps the two aligned; reading the logits
// before or after the call reports a neighbouring step instead.
static llama_token sample_traced(llama_sampler * smpl, llama_context * ctx,
                                 int32_t idx, int step, const char * label) {
    llama_token token = llama_sampler_sample(smpl, ctx, idx);
    trace_step(step, token, label, ctx, idx);
    return token;
}

// Modes evict and replace ask twice, so the trace needs a separator between
// the baseline pass and the pass that follows the change.
static void trace_phase(const char * label) {
    if (g_logits_file != NULL) {
        fprintf(g_logits_file, "phase\t%s\n", label);
        fflush(g_logits_file);
    }
}

// Decode two prompts that live in two different sequences, and report what
// each one produced.
//
// This is the foundation check of the whole approach: if the attention mask
// were built from anything other than the sequence ids, each prompt would see
// the other's tokens and the two streams would differ from two separate runs.
// The reference implementation for this check is llama-batched; see
// tests/test_batch_equivalence.py.
//
// Each stream is decoded with its own single-token call, so the batch index
// that sampling reads from is simply the index of the token just submitted.
// An empty prompt_b means a single-prompt reference run: one sequence, one
// prompt, nothing else in the batch.
static bool run_two_prompts(llama_context * ctx,
                            llama_model * model,
                            llama_sampler * smpl_a,
                            llama_sampler * smpl_b,
                            common_batch & batch,
                            const std::vector<llama_token> & toks_a,
                            const std::vector<llama_token> & toks_b,
                            int n_predict,
                            llama_pos stride,
                            llama_pos offset_a,
                            bool joint_prefill,
                            std::string & out_a,
                            std::string & out_b) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool two = !toks_b.empty();
    llama_token next_a = 0;
    llama_token next_b = 0;

    // Prefill both prompts.
    //
    // The two prompts cannot both start at position zero: a KV cell is
    // addressed by position, so sequence 1 is placed past sequence 0 using the
    // stride the caller passes in.
//
// By default each prompt is submitted as its own batch.  That is what makes
// the comparison against a single-prompt run exact: with both prompts in one
// batch the attention of prompt A is computed inside a larger matmul, the
// accumulation order changes, and a near-tied greedy choice can flip.  Nothing
// leaks between the sequences either way - the mask is per sequence - but the
// numbers move, so the split has to be held fixed to compare.
    //
    // --joint-prefill submits both in one batch, which is what the upstream
    // example does, and is kept so the difference can be measured rather than
    // assumed.
    if (joint_prefill) {
        batch.clear();
        for (size_t i = 0; i < toks_a.size(); ++i) {
            batch.add(toks_a[i], offset_a + (llama_pos) i,
                      std::vector<llama_seq_id>{ 0 }, false);
        }
        for (size_t i = 0; i < toks_b.size(); ++i) {
            batch.add(toks_b[i], stride + (llama_pos) i,
                      std::vector<llama_seq_id>{ 1 }, false);
        }
        batch.set_output((int) toks_a.size() - 1, true);
        if (two) {
            batch.set_output((int) toks_a.size() + (int) toks_b.size() - 1, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: prompt prefill failed\n", __func__);
            return false;
        }
        next_a = sample_traced(smpl_a, ctx, (int32_t) toks_a.size() - 1,
                               -1, "seq0");
        next_b = two ? sample_traced(smpl_b, ctx,
                    (int32_t) (toks_a.size() + toks_b.size()) - 1, -1, "seq1")
                    : 0;
    } else {
        batch.clear();
        for (size_t i = 0; i < toks_a.size(); ++i) {
            batch.add(toks_a[i], offset_a + (llama_pos) i,
                      std::vector<llama_seq_id>{ 0 }, false);
        }
        batch.set_output((int) toks_a.size() - 1, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: prefill of prompt A failed\n", __func__);
            return false;
        }
        next_a = sample_traced(smpl_a, ctx, (int32_t) toks_a.size() - 1, -1, "seq0");

        if (two) {
            batch.clear();
            for (size_t i = 0; i < toks_b.size(); ++i) {
                batch.add(toks_b[i], stride + (llama_pos) i,
                          std::vector<llama_seq_id>{ 1 }, false);
            }
            batch.set_output((int) toks_b.size() - 1, true);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                LOG_ERR("%s: prefill of prompt B failed\n", __func__);
                return false;
            }
            next_b = sample_traced(smpl_b, ctx, (int32_t) toks_b.size() - 1, -1, "seq1");
        }
    }

    for (int step = 0; step < n_predict; ++step) {
        const bool done_a = llama_vocab_is_eog(vocab, next_a);
        const bool done_b = two && llama_vocab_is_eog(vocab, next_b);
        if (done_a && done_b) {
            break;
        }

        // Each stream is submitted on its own, so the decode that follows it
        // produces exactly the logits needed to sample that stream's next
        // token.  The two samplers are separate because sampler state is
        // per stream.
        if (!done_a) {
            out_a += common_token_to_piece(ctx, next_a);
            batch.clear();
            batch.add(next_a, offset_a + (llama_pos) toks_a.size() + step,
                      std::vector<llama_seq_id>{ 0 }, true);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                LOG_ERR("%s: decode of sequence 0 failed at step %d\n", __func__, step);
                return false;
            }
            next_a = sample_traced(smpl_a, ctx, 0, step, "seq0");
        }

        if (!done_b && two) {
            out_b += common_token_to_piece(ctx, next_b);
            batch.clear();
            batch.add(next_b, stride + (llama_pos) toks_b.size() + step,
                      std::vector<llama_seq_id>{ 1 }, true);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                LOG_ERR("%s: decode of sequence 1 failed at step %d\n", __func__, step);
                return false;
            }
            next_b = sample_traced(smpl_b, ctx, 0, step, "seq1");
        }
    }
    return true;
}

// Put every document token into the cache.  A token keeps its own block's
// seq_id, so the attention mask lets it see only that block; tokens that carry
// an extra seq_id (a cross-reference) can also reach the block they cite.
static bool prefill_document(llama_context * ctx, common_batch & batch,
                             const std::vector<doc_token> & tokens,
                             const std::vector<block> & blocks) {
    batch.clear();
    for (const auto & t : tokens) {
        batch.add(t.id, t.pos, t.seq_ids, false);
    }
    if (batch.size() == 0) {
        LOG_ERR("%s: document has no tokens\n", __func__);
        return false;
    }

    // Only the final token needs logits: it is the first token of a question.
    batch.set_output(batch.size() - 1, true);

    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        LOG_ERR("%s: prefill decode failed\n", __func__);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Question and generation
// ---------------------------------------------------------------------------

// Collect the context the question should see into one scratch sequence.
//
// A sequence's positions must stay consecutive, and the blocks each end at a
// different position, so the question cannot simply join them.  Instead the
// wanted blocks are copied into a dedicated answer sequence, and the question
// and its answer continue from there.  A token that carried an extra seq_id
// (a cross-reference) is copied along with every block that can see it.
static bool gather_context(llama_context * ctx,
                           llama_seq_id answer_seq,
                           int focus,
                           const std::vector<block> & blocks,
                           const std::vector<cross_reference> & xrefs,
                           bool follow_xrefs,
                           llama_pos & next_pos) {
    llama_memory_t mem = llama_get_memory(ctx);

    // The answer sequence must start empty and contiguous.
    llama_memory_seq_rm(mem, answer_seq, -1, -1);

    std::vector<int> wanted;
    auto add = [&](int seq_id) {
        if (std::find(wanted.begin(), wanted.end(), seq_id) == wanted.end()) {
            wanted.push_back(seq_id);
        }
    };

    for (const auto & b : blocks) {
        if (focus < 0 || b.seq_id == focus) {
            add(b.seq_id);
        }
    }
    if (wanted.empty()) {
        LOG_ERR("%s: no block selected for focus %d\n", __func__, focus);
        return false;
    }

    // A block that cites another block cannot answer without it, so the cited
    // block is pulled in as well.  This is the "group by cited block"
    // approximation: it is resolved here rather than by giving the citing
    // tokens a second seq_id, because that would make a sequence
    // non-contiguous and llama.cpp would refuse the batch.
    if (follow_xrefs) {
        for (const auto & x : xrefs) {
            const bool from_wanted = std::find(wanted.begin(), wanted.end(),
                                               x.from_seq) != wanted.end();
            const bool to_wanted = std::find(wanted.begin(), wanted.end(),
                                             x.target_seq) != wanted.end();
            if (from_wanted && !to_wanted) {
                add(x.target_seq);
            } else if (to_wanted && !from_wanted) {
                add(x.from_seq);
            }
        }
    }

    // Copy in position order, so the answer sequence stays contiguous.
    std::sort(wanted.begin(), wanted.end());
    next_pos = 0;
    for (int seq_id : wanted) {
        const block * found = nullptr;
        for (const auto & b : blocks) {
            if (b.seq_id == seq_id) {
                found = &b;
            }
        }
        if (found == NULL) {
            continue;
        }
        llama_memory_seq_cp(mem, found->seq_id, answer_seq,
                            found->pos_min, found->pos_max + 1);
        next_pos = std::max(next_pos, found->pos_max + 1);
    }

    LOG_INF("%s: answer sequence %d holds %zu block(s), %lld positions\n",
            __func__, answer_seq, wanted.size(), (long long) next_pos);
    return true;
}

// Read a whole file, used for prompts and replacements.
static bool read_file(const std::string & path, std::string & out) {
    std::ifstream file(path);
    if (!file) {
        return false;
    }
    out.assign((std::istreambuf_iterator<char>(file)),
               std::istreambuf_iterator<char>());
    return true;
}

static bool run_question(llama_context * ctx,
                         llama_model * model,
                         llama_sampler * smpl,
                         common_batch & batch,
                         const std::vector<llama_token> & question,
                         llama_pos start_pos,
                         llama_seq_id answer_seq,
                         int n_predict,
                         std::string & answer,
                         int & n_decoded) {
    const std::vector<llama_seq_id> seq_ids { answer_seq };

    // The vocabulary is needed for the end-of-generation check.
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (question.empty()) {
        LOG_ERR("%s: empty question\n", __func__);
        return false;
    }

    int32_t last = -1;
    for (size_t i = 0; i < question.size(); ++i) {
        batch.clear();
        batch.add(question[i], start_pos + (llama_pos) i, seq_ids, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: question decode failed at %d\n", __func__, (int) i);
            return false;
        }
        last = 0;
    }

    for (int i = 0; i < n_predict; ++i) {
        llama_token next = sample_traced(smpl, ctx, last, i, "answer");
        if (llama_vocab_is_eog(llama_model_get_vocab(model), next)) {
            break;
        }
        answer += common_token_to_piece(ctx, next);

        batch.clear();
        batch.add(next, start_pos + (llama_pos) question.size() + i, seq_ids, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: generation decode failed at %d\n", __func__, i);
            return false;
        }
        last = 0;
        n_decoded += 1;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Eviction
// ---------------------------------------------------------------------------

// Drop one block from the cache.  The block is a contiguous position range
// owned by one sequence, so a single seq+position removal is enough.
static bool evict_block(llama_context * ctx, const block & b, int & tokens_removed) {
    llama_memory_t mem = llama_get_memory(ctx);

    llama_pos p_min = llama_memory_seq_pos_min(mem, b.seq_id);
    llama_pos p_max = llama_memory_seq_pos_max(mem, b.seq_id);
    if (p_max < p_min) {
        LOG_ERR("%s: sequence %d is empty, nothing to evict\n", __func__, b.seq_id);
        return false;
    }

    if (!llama_memory_seq_rm(mem, b.seq_id, b.pos_min, b.pos_max + 1)) {
        LOG_ERR("%s: partial removal refused for sequence %d\n", __func__, b.seq_id);
        return false;
    }

    tokens_removed = (int) (p_max - p_min + 1);
    return true;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------
static void print_layout(const std::vector<block> & blocks,
                         const std::string & title) {
    LOG("%s\n", title.c_str());
    LOG("%-4s %-8s %-8s %-7s %-6s %s\n", "seq", "pos", "count", "tokens", "merged", "title");
    for (const auto & b : blocks) {
        LOG("%-4d [%lld,%lld] %-7d %-6d %-6s %s\n",
            b.seq_id, (long long) b.pos_min, (long long) b.pos_max,
            (int) (b.pos_max - b.pos_min + 1), b.tokens,
            b.merged ? "yes" : "no", b.title.c_str());
    }
}

// Bytes the whole context state occupies.  This is the number that must drop
// after an eviction.
static size_t state_bytes(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

#ifdef _MSC_VER
// The driver writes a small diagnostic TSV with fprintf.
#pragma warning(disable : 4996)
#endif

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_predict = 32;

    std::string mode = "ask";
    std::string doc_path;
    std::string prompt_a;
    std::string prompt_b;
    std::string question = "Question:";
    std::string replacement_path;
    std::string logits_path;
    int  focus = -1;
    int  evict_seq = -1;
    bool kv_unified = false;
    bool joint_prefill = false;
    bool follow_xrefs = true;
    // Where prompt A starts.  The reference run of a single prompt must place
    // it exactly where it sat in the two-prompt run, because RoPE positions are
    // absolute even though attention between them is relative.
    llama_pos offset_a = 0;

    common_init();

    // The driver's own options are removed before the common parser runs, so
    // that unknown-option handling and the usage text stay in one place.
    std::vector<char *> common_argv;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        bool consumed = false;
        if (a == "--debug-logits") {
            g_debug_logits = true;
            consumed = true;
        } else if (a == "--no-follow-xrefs") {
            follow_xrefs = false;
            consumed = true;
        } else if (a == "--joint-prefill") {
            joint_prefill = true;
            consumed = true;
        } else if (a == "--kv-unified") {
            kv_unified = true;
            consumed = true;
        } else if (a == "--offset-a") {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --offset-a needs a value\n", __func__);
                return 1;
            }
            offset_a = (llama_pos) std::stol(argv[i + 1]);
            i += 1;
            consumed = true;
        } else if (a == "--prompt-a" || a == "--prompt-b") {
            if (i + 1 >= argc) {
                LOG_ERR("%s: %s needs a value\n", __func__, a.c_str());
                return 1;
            }
            if (a == "--prompt-a") {
                prompt_a = argv[i + 1];
            } else {
                prompt_b = argv[i + 1];
            }
            i += 1;
            consumed = true;
        } else if (a == "--mode" || a == "--focus-seq" || a == "--evict-seq" ||
            a == "--replace-file" || a == "--doc" || a == "--logits-out") {
            if (i + 1 >= argc) {
                LOG_ERR("%s: %s needs a value\n", __func__, a.c_str());
                return 1;
            }
            const std::string value = argv[i + 1];
            if      (a == "--mode")          mode = value;
            else if (a == "--doc")           doc_path = value;
            else if (a == "--focus-seq")     focus = std::stoi(value);
            else if (a == "--evict-seq")     evict_seq = std::stoi(value);
            else if (a == "--replace-file")  replacement_path = value;
            else if (a == "--logits-out")    logits_path = value;
            i += 1;
            consumed = true;
        } else if (a == "-q") {
            if (i + 1 >= argc) {
                LOG_ERR("%s: -q needs a value\n", __func__);
                return 1;
            }
            question = argv[i + 1];
            i += 1;
            consumed = true;
        }
        if (!consumed) {
            common_argv.push_back(argv[i]);
        }
    }
    common_argv.push_back(nullptr);

    if (!common_params_parse((int) common_argv.size() - 1, common_argv.data(),
                             params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    // Mode two-prompts needs no document: it is the foundation check that a
    // batch of independent sequences behaves like separate runs.
    const bool two_prompts = mode == "two-prompts";
    if (!two_prompts && doc_path.empty()) {
        LOG_ERR("%s: --doc <document.json> is required\n", __func__);
        print_usage(0, argv);
        return 1;
    }
    // --prompt-b is optional: omitting it gives the single-prompt reference
    // run that the two-prompt run is compared against.
    if (two_prompts && prompt_a.empty()) {
        LOG_ERR("%s: --prompt-a is required\n", __func__);
        print_usage(0, argv);
        return 1;
    }

    // The trace file is opened after the options are read and before any
    // decoding, so every generated step is recorded.
    if (!logits_path.empty()) {
        g_logits_file = fopen(logits_path.c_str(), "w");
        if (g_logits_file == NULL) {
            LOG_ERR("%s: cannot write %s\n", __func__, logits_path.c_str());
            return 1;
        }
    }

    std::vector<doc_token> tokens;
    std::vector<block> blocks;
    std::string title;
    // Even the single-prompt reference run allocates two sequences and the same
    // context size, so that the two runs differ only in whether the second
    // prompt is present.  A different cache shape changes the reduction order
    // on CPU, which would otherwise show up as a difference that has nothing
    // to do with masking.
    std::vector<cross_reference> xrefs;
    int n_seq_needed = two_prompts ? 2 : 1;

    if (two_prompts) {
        LOG_INF("%s: two prompts in two sequences\n", __func__);
    } else {
        if (!load_document(doc_path, tokens, blocks, xrefs, title,
                           n_seq_needed)) {
            return 1;
        }
        LOG_INF("%s: %s, %zu tokens, %zu sequences\n",
                __func__, title.c_str(), tokens.size(), blocks.size());

        if (mode == "stats") {
            print_layout(blocks, title);
            return 0;
        }
    }

    // Sequence 0..n_seq_needed-1 hold the document blocks.  The last sequence is
    // the scratch sequence the question and the answer live in.
    const uint32_t n_seq_max = (uint32_t) n_seq_needed + (two_prompts ? 0 : 1);
    const llama_seq_id answer_seq = n_seq_needed;

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Tokenize up front: the context has to be large enough for both prompts
    // before anything can be submitted.
    const std::vector<llama_token> toks_a =
        two_prompts ? common_tokenize(vocab, prompt_a, true)
                    : std::vector<llama_token>();
    const std::vector<llama_token> toks_b =
        (two_prompts && !prompt_b.empty())
            ? common_tokenize(vocab, prompt_b, true)
            : std::vector<llama_token>();

    llama_context_params ctx_params = common_context_params_to_llama(params);
    // A block is evicted as a position range, so each sequence needs room for
    // the whole document plus the question and the answer.
    // Both prompts need room, whether or not the second one is present: the
    // context shape is held constant across the two runs on purpose.  A cache
    // of a different size changes the attention kernel's reduction order on
    // CPU, which would appear as a difference that has nothing to do with
    // masking.
    const int prompt_room = (int) offset_a + (int) toks_a.size() +
        std::max((int) toks_b.size(), PROMPT_RESERVE) +
        2 * params.n_predict + 16;
    const int tokens_needed = two_prompts
        ? prompt_room
        : (int) tokens.size() + params.n_predict + 8;
    ctx_params.n_ctx      = n_seq_max * (uint32_t) tokens_needed;
    ctx_params.n_batch    = std::max((uint32_t) params.n_batch, (uint32_t) tokens.size());
    ctx_params.n_seq_max  = n_seq_max;
    // gather_context() copies whole blocks into the scratch sequence, and seq_cp()
    // needs one buffer, so the document modes use the unified KV cache.  Mode
    // two-prompts never copies, so it can keep a stream per sequence, which is
    // what makes its two contexts provably independent.
    ctx_params.kv_unified = two_prompts ? kv_unified : true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        LOG_ERR("%s: failed to create the context\n", __func__);
        return 1;
    }

    LOG_INF("%s: n_ctx = %u, n_seq_max = %u, n_ctx/seq = %u\n", __func__,
            llama_n_ctx(ctx), llama_n_seq_max(ctx), llama_n_ctx_seq(ctx));

    common_batch batch(ctx);

    // Greedy decoding, so that two runs of the same context agree exactly.
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    auto make_sampler = [&]() {
        llama_sampler * chain = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(chain,
            llama_sampler_init_top_k(params.sampling.top_k));
        llama_sampler_chain_add(chain,
            llama_sampler_init_top_p(params.sampling.top_p, params.sampling.min_keep));
        llama_sampler_chain_add(chain,
            llama_sampler_init_temp(params.sampling.temp));
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        return chain;
    };
    llama_sampler * smpl   = make_sampler();
    llama_sampler * smpl_b = toks_b.empty() ? NULL : make_sampler();

    if (two_prompts) {
        std::string out_a, out_b;
        const auto t = ggml_time_us();
        // Sequence 1 starts past sequence 0: a unified KV cell is addressed by
    // position, so two prompts both starting at zero would collide.
    const llama_pos stride = (llama_pos) (toks_a.size() + params.n_predict + 8);
        if (!run_two_prompts(ctx, model, smpl, smpl_b, batch,
                             toks_a, toks_b, params.n_predict, stride,
                             offset_a, joint_prefill, out_a, out_b)) {
            return 1;
        }
        LOG_INF("%s: two prompts in %.2f s\n", __func__,
                (ggml_time_us() - t) / 1000000.0);
        LOG("%s\n", out_a.c_str());
        LOG("%s\n", out_b.c_str());
        LOG_INF("%s: sequence 0: %s\n", __func__, out_a.c_str());
        if (!prompt_b.empty()) {
            LOG_INF("%s: sequence 1: %s\n", __func__, out_b.c_str());
        }
        llama_sampler_free(smpl);
        if (smpl_b != NULL) {
            llama_sampler_free(smpl_b);
        }
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    std::vector<llama_token> question_tokens = common_tokenize(vocab, question, true);

    const auto t_start = ggml_time_us();
    if (!prefill_document(ctx, batch, tokens, blocks)) {
        return 1;
    }
    const size_t state_after_prefill = state_bytes(ctx);
    LOG_INF("%s: prefill done in %.2f s, state = %.2f MiB\n", __func__,
            (ggml_time_us() - t_start) / 1000000.0, state_after_prefill / 1048576.0);

    std::string answer;
    int n_decoded = 0;

    if (mode == "prefill") {
        LOG("%s\n", answer.c_str());
    } else if (mode == "ask" || mode == "evict" || mode == "replace") {
        // Ask before any change, so the driver can compare against it.
        std::string before;
        int decoded_before = 0;
        llama_pos question_pos = 0;
        trace_phase("baseline");
        const auto t_before = ggml_time_us();
        if (!gather_context(ctx, answer_seq, focus, blocks, xrefs,
                           follow_xrefs, question_pos) ||
            !run_question(ctx, model, smpl, batch, question_tokens, question_pos,
                          answer_seq, params.n_predict, before, decoded_before)) {
            return 1;
        }
        const double before_s = (ggml_time_us() - t_before) / 1000000.0;
        LOG_INF("%s: baseline answer (%.2f s):\n%s\n", __func__, before_s, before.c_str());

        if (mode == "ask") {
            answer = before;
            n_decoded = decoded_before;
        } else {
            if (evict_seq < 0) {
                LOG_ERR("%s: --evict-seq is required for mode %s\n", __func__, mode.c_str());
                return 1;
            }
            const block * target = nullptr;
            for (const auto & b : blocks) {
                if (b.seq_id == evict_seq) {
                    target = &b;
                }
            }
            if (target == NULL) {
                LOG_ERR("%s: no block with seq_id %d\n", __func__, evict_seq);
                return 1;
            }

            int removed = 0;
            if (!evict_block(ctx, *target, removed)) {
                return 1;
            }
            const size_t state_after_evict = state_bytes(ctx);
            LOG_INF("%s: evicted sequence %d ([%lld,%lld]), removed %d cells, state %.2f -> %.2f MiB\n",
                    __func__, target->seq_id, (long long) target->pos_min,
                    (long long) target->pos_max, removed,
                    state_after_prefill / 1048576.0, state_after_evict / 1048576.0);

            if (mode == "replace") {
                if (replacement_path.empty()) {
                    LOG_ERR("%s: --replace-file is required for mode replace\n", __func__);
                    return 1;
                }
                std::string replacement;
                if (!read_file(replacement_path, replacement)) {
                    LOG_ERR("%s: cannot open %s\n", __func__, replacement_path.c_str());
                    return 1;
                }
                std::vector<llama_token> new_tokens = common_tokenize(vocab, replacement, false);

                batch.clear();
                for (size_t i = 0; i < new_tokens.size(); ++i) {
                    batch.add(new_tokens[i], target->pos_min + (llama_pos) i,
                              std::vector<llama_seq_id>{ target->seq_id }, false);
                }
                if (batch.size() > 0) {
                    batch.set_output(batch.size() - 1, true);
                    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                        LOG_ERR("%s: replacement decode failed\n", __func__);
                        return 1;
                    }
                    LOG_INF("%s: refilled sequence %d with %zu tokens at positions [%lld,%lld]\n",
                            __func__, target->seq_id, new_tokens.size(),
                            (long long) target->pos_min,
                            (long long) (target->pos_min + (llama_pos) new_tokens.size() - 1));
                }
            }

            const auto t_after = ggml_time_us();
            std::string after;
            int decoded_after = 0;
            llama_pos after_pos = 0;
            trace_phase("after");
            if (!gather_context(ctx, answer_seq, focus, blocks, xrefs,
                               follow_xrefs, after_pos) ||
                !run_question(ctx, model, smpl, batch, question_tokens, after_pos,
                              answer_seq, params.n_predict, after, decoded_after)) {
                return 1;
            }
            LOG_INF("%s: answer after change (%.2f s):\n%s\n", __func__,
                    (ggml_time_us() - t_after) / 1000000.0, after.c_str());
            LOG_INF("%s: unchanged = %s\n", __func__,
                    before == after ? "yes" : "NO");
            answer = after;
            n_decoded = decoded_after;
        }
    } else {
        LOG_ERR("%s: unknown mode %s\n", __func__, mode.c_str());
        return 1;
    }

    LOG_INF("%s: mode=%s focus=%d decoded=%d state=%.2f MiB\n",
            __func__, mode.c_str(), focus, n_decoded, state_bytes(ctx) / 1048576.0);

    llama_perf_context_print(ctx);

    if (g_logits_file != NULL) {
        fclose(g_logits_file);
    }
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}