// Pipeline version of anniemap:
//   - Reads are k-merized, searched and (optionally) aligned in parallel batches.
//   - Each worker thread is responsible for an entire "batch pipeline":
//         k-mer -> FAISS search -> (optional) WFA2 alignment -> TSV line formatting.
//   - Completed batches are pushed into a shared output queue where any worker
//     can temporarily become a "writer" and flush ready batches to the output
//     stream in order (opportunistic writer pattern).
//
// The goal is to remove a single dedicated writer thread as a bottleneck:
// writing is spread across workers, but the final TSV remains strictly
// ordered by batch id.
//
// Usage (roughly mirroring anniemap.cpp):
//
//   ./anniemap_pipeline <fastq_r1> <ref_name> [fastq_r2]
//       [--max-length N] [--step-size N] [--nlist N] [--nprobe N] [--threads N]
//       [--canonical] [--align] [--secondary] [--sam] [--output out.tsv]
//       [--banded] [--adaptive] [--zdrop]
//   WFA2 heuristics (--align only): enable any combination; numeric params are
//   hard-coded in configure_wfa_heuristics() below.
//   With --sam and --secondary: emit one SAM record per read; align primary and
//   best separated secondary (same rule as TSV), keep the higher WFA2 score.
//   With --sam and --align: POS and CIGAR are corrected for the padded local reference
//   (ends-free I/S in the padded flank are dropped, not turned into S); append optional
//   AS:i:<WFA2 score> on each mapped record (after QUAL), matching TSV alignment_score.

#include <faiss/IndexBinary.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/index_io.h>

#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <set>
#include <string>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <zlib.h>
#include "kseq.h"
KSEQ_INIT(gzFile, gzread)

#include "kmer_vector.hpp"
#include "bindings/cpp/WFAligner.hpp"

// ----- Shared helpers (same as anniemap.cpp) -----

// Load all FASTA records from `path` into `out_seqs`.
// Key = header line after '>' (as stored in metadata by faiss_create_kmer).
// Value = sequence (uppercase, no whitespace).
// Returns false on I/O failure or if no sequence content is found.
static bool load_fasta_all_sequences(const std::string& path,
                                     std::unordered_map<std::string, std::string>& out_seqs) {
    std::ifstream in(path);
    if (!in) return false;
    out_seqs.clear();
    std::string line;
    std::string current_name;
    std::string current_seq;
    auto flush = [&]() {
        if (current_name.empty() || current_seq.empty()) return;
        out_seqs[std::move(current_name)] = std::move(current_seq);
        current_seq.clear();
    };
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            flush();
            current_name = line.substr(1);
            continue;
        }
        for (char c : line) {
            if (std::isspace(static_cast<unsigned char>(c))) continue;
            current_seq += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        }
    }
    flush();
    return !out_seqs.empty();
}

// DNA complement helper used for reverse-complement operations.
// Non-ACGT characters are mapped to 'N' to keep alignment safe.
static char complement(char c) {
    switch (std::toupper(static_cast<unsigned char>(c))) {
        case 'A': return 'T';
        case 'T': return 'A';
        case 'C': return 'G';
        case 'G': return 'C';
        case 'N': return 'N';
        default:  return 'N';
    }
}

// In-place reverse-complement of a DNA string.
// The function walks from both ends toward the middle and swaps
// complemented bases; odd-length strings have their middle base complemented.
static void reverse_complement_inplace(std::string& s) {
    const std::size_t n = s.size();
    for (std::size_t i = 0; i < n / 2; ++i) {
        std::size_t j = n - 1 - i;
        char a = complement(s[i]);
        char b = complement(s[j]);
        s[i] = b;
        s[j] = a;
    }
    if (n % 2) s[n / 2] = complement(s[n / 2]);
}

// Build a local reference segment around `target_start` suitable for ends-free
// alignment with the read.
//
// - A padding proportional to read_len (at least 15bp) is taken on both sides.
// - Out-of-bounds regions at the genome edges are filled with 'N'.
// - If strand == '-', the segment is reverse-complemented so that alignment
//   is always performed in the forward orientation relative to the read.
static std::string build_padded_ref(const std::string& ref,
                                    int target_start,
                                    std::size_t align_read_len,
                                    char strand) {
    const std::size_t ref_len = ref.size();
    int padding = static_cast<int>(align_read_len / 10);
    if (align_read_len % 10) padding += 1;
    if (padding < 15) padding = 15;
    int64_t padded_start = static_cast<int64_t>(target_start) - padding;
    int64_t padded_end   = static_cast<int64_t>(target_start) + static_cast<int64_t>(align_read_len) + padding;
    std::size_t actual_start = padded_start < 0 ? 0u : static_cast<std::size_t>(padded_start);
    std::size_t actual_end   = (padded_end > static_cast<int64_t>(ref_len)) ? ref_len : static_cast<std::size_t>(padded_end);
    if (actual_end <= actual_start) actual_end = actual_start + 1;
    std::string seg = ref.substr(actual_start, actual_end - actual_start);
    if (padded_start < 0) seg.insert(0, static_cast<std::size_t>(-padded_start), 'N');
    if (padded_end > static_cast<int64_t>(ref_len)) seg.append(static_cast<std::size_t>(padded_end - static_cast<int64_t>(ref_len)), 'N');
    if (strand == '-') reverse_complement_inplace(seg);
    return seg;
}

// Hard-coded WFA2 heuristic parameters (CLI flags only select which apply).
static constexpr int kWfaBandMinK = -80;
static constexpr int kWfaBandMaxK = 80;
static constexpr int kWfaAdaptiveMinWfLen = 10;
static constexpr int kWfaAdaptiveMaxDist = 50;
static constexpr int kWfaAdaptiveSteps = 1;
static constexpr int kWfaZdrop = 100;
static constexpr int kWfaZdropSteps = 1;

// Reset heuristics then OR in enabled strategies (WFA2 applies wf-adaptive, then
// z-drop, then static band on each heuristic pass when multiple are enabled).
static void configure_wfa_heuristics(
    wfa::WFAlignerGapAffine* aligner,
    bool banded,
    bool adaptive,
    bool zdrop) {
    aligner->setHeuristicNone();
    if (!banded && !adaptive && !zdrop) return;
    if (banded) aligner->setHeuristicBandedStatic(kWfaBandMinK, kWfaBandMaxK);
    if (adaptive) aligner->setHeuristicWFadaptive(
        kWfaAdaptiveMinWfLen, kWfaAdaptiveMaxDist, kWfaAdaptiveSteps);
    if (zdrop) aligner->setHeuristicZDrop(kWfaZdrop, kWfaZdropSteps);
}

// Escape a single TSV field into `s`.
// Ensures that the final TSV file is strictly "one record per line" by:
//   - replacing '\t' with "\t", '\n' with "\n", '\r' with "\r",
//   - escaping literal backslashes as "\\".
// This keeps the output robust when read names or sequences contain control characters.
static void append_tsv_field(std::string& s, const std::string& value) {
    s.reserve(s.size() + value.size() + 16);
    for (char c : value) {
        if (c == '\t') { s += '\\'; s += 't'; }
        else if (c == '\n') { s += '\\'; s += 'n'; }
        else if (c == '\r') { s += '\\'; s += 'r'; }
        else if (c == '\\') { s += '\\'; s += '\\'; }
        else { s += c; }
    }
}

// Normalize paired-end read names for SAM output (option B):
// input headers are assumed to end in "/1" or "/2".
static void normalize_pair_qname_inplace(std::string& qname) {
    if (qname.size() >= 2) {
        const std::string suffix = qname.substr(qname.size() - 2);
        if (suffix == "/1" || suffix == "/2") {
            qname.resize(qname.size() - 2);
        }
    }
}

// Normalize reference sequence names for SAM output.
// Example input: "NC_001477.1 |Dengue virus 1, complete genome"
// Output:        "NC_001477.1"
static std::string normalize_ref_sequence_name(const std::string& ref_header) {
    // Trim leading whitespace (defensive).
    std::size_t begin = 0;
    while (begin < ref_header.size() && std::isspace(static_cast<unsigned char>(ref_header[begin])))
        ++begin;

    // Find first whitespace or '|' (whichever comes first).
    std::size_t pos_ws = ref_header.find_first_of(" \t\r\n", begin);
    std::size_t pos_pipe = ref_header.find('|', begin);

    std::size_t cut;
    if (pos_ws == std::string::npos && pos_pipe == std::string::npos) cut = ref_header.size();
    else if (pos_ws == std::string::npos) cut = pos_pipe;
    else if (pos_pipe == std::string::npos) cut = pos_ws;
    else cut = std::min(pos_ws, pos_pipe);

    std::string token = ref_header.substr(begin, cut - begin);
    // Trim trailing whitespace if we cut at whitespace.
    while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back()))) token.pop_back();
    return token.empty() ? ref_header : token;
}

// Compute how many reference bases the CIGAR consumes.
// Reference-consuming ops in SAM: M, D, N, =, X (I and S do not consume reference).
static int cigar_ref_consumed_len(const std::string& cigar) {
    if (cigar.empty() || cigar == "*") return 0;
    int total = 0;
    int num = 0;
    for (char c : cigar) {
        if (c >= '0' && c <= '9') {
            num = num * 10 + (c - '0');
            continue;
        }
        if (num == 0) continue; // be defensive against malformed CIGAR
        switch (c) {
            case 'M':
            case 'D':
            case 'N':
            case '=':
            case 'X':
                total += num;
                break;
            default:
                break; // I, S, H, P: ignore
        }
        num = 0;
    }
    return total;
}

// ----- Padded-ref geometry (must match build_padded_ref) + SAM POS/CIGAR adjustment -----

static int ref_padding_for_read_len(std::size_t read_len) {
    int padding = static_cast<int>(read_len / 10);
    if (read_len % 10) padding += 1;
    if (padding < 15) padding = 15;
    return padding;
}

// Virtual 0-based start coordinate of the padded window (may be negative before N-fill).
static int64_t padded_ref_virtual_start(int target_start, std::size_t read_len) {
    return static_cast<int64_t>(target_start) - ref_padding_for_read_len(read_len);
}

// Length of the padded string returned by build_padded_ref (before/after RC; same length).
static std::size_t padded_ref_segment_len(int target_start, std::size_t read_len, std::size_t ref_len) {
    int pad = ref_padding_for_read_len(read_len);
    int64_t padded_start = static_cast<int64_t>(target_start) - pad;
    int64_t padded_end = static_cast<int64_t>(target_start) + static_cast<int64_t>(read_len) + pad;
    std::size_t actual_start = padded_start < 0 ? 0u : static_cast<std::size_t>(padded_start);
    std::size_t actual_end = (padded_end > static_cast<int64_t>(ref_len))
        ? ref_len
        : static_cast<std::size_t>(padded_end);
    if (actual_end <= actual_start) actual_end = actual_start + 1;
    std::size_t core_len = actual_end - actual_start;
    std::size_t n_prefix = padded_start < 0 ? static_cast<std::size_t>(-padded_start) : 0u;
    std::size_t n_suffix = padded_end > static_cast<int64_t>(ref_len)
        ? static_cast<std::size_t>(padded_end - static_cast<int64_t>(ref_len))
        : 0u;
    return core_len + n_prefix + n_suffix;
}

static std::vector<std::pair<int, char>> cigar_to_ops(const std::string& cigar) {
    std::vector<std::pair<int, char>> ops;
    if (cigar.empty() || cigar == "*") return ops;
    int num = 0;
    for (char c : cigar) {
        if (c >= '0' && c <= '9') {
            num = num * 10 + (c - '0');
            continue;
        }
        if (num > 0) ops.push_back({num, c});
        num = 0;
    }
    return ops;
}

static bool cigar_op_is_query_clip(char op) {
    return op == 'I' || op == 'S';
}

static bool cigar_op_is_ref_gap(char op) {
    return op == 'D' || op == 'N';
}

static std::string cigar_ops_to_string(const std::vector<std::pair<int, char>>& ops) {
    std::string s;
    s.reserve(ops.size() * 8);
    for (const auto& pr : ops) {
        s += std::to_string(pr.first);
        s += pr.second;
    }
    return s;
}

static void cigar_merge_adjacent(std::vector<std::pair<int, char>>& ops) {
    if (ops.empty()) return;
    std::vector<std::pair<int, char>> out;
    out.reserve(ops.size());
    for (const auto& pr : ops) {
        if (!out.empty() && out.back().second == pr.second)
            out.back().first += pr.first;
        else
            out.push_back(pr);
    }
    ops.swap(out);
}

// Sum of query-consuming CIGAR ops (M, I, S, =, X, H).
static int cigar_ops_query_consumed(const std::vector<std::pair<int, char>>& ops) {
    int total = 0;
    for (const auto& pr : ops) {
        switch (pr.second) {
            case 'M':
            case 'I':
            case 'S':
            case '=':
            case 'X':
            case 'H':
                total += pr.first;
                break;
            default:
                break;
        }
    }
    return total;
}

static int cigar_ops_ref_consumed(const std::vector<std::pair<int, char>>& ops) {
    int total = 0;
    for (const auto& pr : ops) {
        switch (pr.second) {
            case 'M':
            case 'D':
            case 'N':
            case '=':
            case 'X':
                total += pr.first;
                break;
            default:
                break;
        }
    }
    return total;
}

// WFA2 aligns to a padded local reference. FAISS POS is the window start on the forward genome;
// shift POS using ends-free clip lengths vs expected padding, then drop leading/trailing I/S that
// lie in the padded flank (do not re-emit them as S — POS already places the alignment).
// If the trimmed ops do not consume the full read length (rare), prepend S for the deficit so SAM
// stays valid — that case is real clipping / indels, not padding bookkeeping.
static void sam_adjust_pos_strip_padding_cigar(
    char strand,
    int meta_start,
    std::size_t align_read_len,
    std::size_t ref_len,
    const std::string& cigar,
    int& pos_1based,
    std::string& cigar_out,
    int& ref_span,
    int& end_1based) {
    cigar_out = cigar;
    if (cigar.empty() || cigar == "*") return;

    const int64_t vs = padded_ref_virtual_start(meta_start, align_read_len);
    const std::size_t seg_len = padded_ref_segment_len(meta_start, align_read_len, ref_len);
    const int fidx_meta = static_cast<int>(static_cast<int64_t>(meta_start) - vs);
    const int64_t last_read_0 = static_cast<int64_t>(meta_start) + static_cast<int64_t>(align_read_len) - 1;
    const int fidx_end = static_cast<int>(last_read_0 - vs);

    const int exp_lead = (strand == '+')
        ? fidx_meta
        : (static_cast<int>(seg_len) - 1 - fidx_end);

    std::vector<std::pair<int, char>> ops = cigar_to_ops(cigar);
    if (ops.empty()) return;

    int lead_q = 0;
    while (!ops.empty() && cigar_op_is_query_clip(ops.front().second)) {
        lead_q += ops.front().first;
        ops.erase(ops.begin());
    }
    while (!ops.empty() && cigar_op_is_query_clip(ops.back().second)) {
        ops.pop_back();
    }

    // Terminal D/N can also be pure padded-flank artifacts with ends-free
    // alignment. Remove them conservatively from both ends.
    while (!ops.empty() && cigar_op_is_ref_gap(ops.front().second)) {
        ops.erase(ops.begin());
    }
    while (!ops.empty() && cigar_op_is_ref_gap(ops.back().second)) {
        ops.pop_back();
    }
    if (ops.empty()) return;

    if (strand == '+')
        pos_1based = meta_start + 1 + (lead_q - exp_lead);
    else
        pos_1based = meta_start + 1 - (lead_q - exp_lead);

    cigar_merge_adjacent(ops);
    const int q_mid = cigar_ops_query_consumed(ops);
    const int r_mid = cigar_ops_ref_consumed(ops);
    const int rl = static_cast<int>(align_read_len);
    if (q_mid < rl) {
        std::vector<std::pair<int, char>> with_s;
        with_s.reserve(ops.size() + 1u);
        with_s.push_back({rl - q_mid, 'S'});
        with_s.insert(with_s.end(), ops.begin(), ops.end());
        cigar_merge_adjacent(with_s);
        cigar_out = cigar_ops_to_string(with_s);
    } else {
        cigar_out = cigar_ops_to_string(ops);
    }
    ref_span = r_mid;
    end_1based = ref_span > 0 ? (pos_1based + ref_span - 1) : 0;
}

static constexpr int DEFAULT_THREADS = 4;
static constexpr int DEFAULT_MAX_LENGTH = 100;
static constexpr int DEFAULT_STEP_SIZE = 1;

struct ReadRecord {
    std::string id;
    std::string seq;
    std::string qual;
    // If the read was length-capped for FAISS search, keep the original
    // sequence/qualities here so downstream WFA2 alignment can use full-length reads.
    std::string full_seq;
    std::string full_qual;
    // 0-based offset in full_seq/full_qual where the FAISS search slice starts.
    std::size_t search_offset = 0;
};

struct SearchBundle {
    int search_len = 0;
    faiss::IndexBinary* index = nullptr;
    std::vector<std::string> meta_name;
    std::vector<int> meta_start;
    std::vector<char> meta_strand;
    std::vector<std::uint8_t> ref_strand;
    faiss::idx_t k = 1;
};

// Read all records from a gzipped FASTQ file into `out`.
// Read names are kept as-is (no R1/R2 suffix).
// Sequences are kept as-is; length filtering is performed later.
// Returns false on I/O failure.
static bool read_fastq_one(const std::string& path, std::vector<ReadRecord>& out) {
    gzFile fp = gzopen(path.c_str(), "rb");
    if (!fp) { std::cerr << "Failed to open " << path << "\n"; return false; }
    kseq_t* seq = kseq_init(fp);
    if (!seq) { gzclose(fp); return false; }
    while (true) {
        int64_t len = kseq_read(seq);
        if (len < 0) break;
        std::string name = seq->name.s;
        out.push_back({
            std::move(name),
            std::string(seq->seq.s, static_cast<std::size_t>(len)),
            std::string(seq->qual.s, static_cast<std::size_t>(len)),
            std::string(),
            std::string(),
            0
        });
    }
    kseq_destroy(seq);
    gzclose(fp);
    return true;
}

static bool load_metadata_tsv(const std::string& path,
    std::vector<std::string>& sequence_name, std::vector<int>& start_pos, std::vector<char>& strand) {
    // Metadata format (non-canonical index):
    //   header line (ignored)
    //   sequence_name \t start_pos \t strand
    // One row per reference window stored in the FAISS index.
    std::ifstream in(path);
    if (!in) { std::cerr << "Failed to open metadata: " << path << "\n"; return false; }
    std::string line;
    if (!std::getline(in, line)) { std::cerr << "Empty metadata file\n"; return false; }
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::size_t first_tab = line.find('\t');
        if (first_tab == std::string::npos) continue;
        sequence_name.push_back(line.substr(0, first_tab));
        std::string rest = line.substr(first_tab + 1);
        std::size_t second_tab = rest.find('\t');
        if (second_tab == std::string::npos) continue;
        start_pos.push_back(std::stoi(rest.substr(0, second_tab)));
        std::string third = rest.substr(second_tab + 1);
        strand.push_back(third.empty() ? '+' : third[0]);
    }
    return true;
}

static bool load_metadata_tsv_canonical(const std::string& path,
    std::vector<std::string>& sequence_name, std::vector<int>& start_pos) {
    // Metadata format (canonical index):
    //   header line (ignored)
    //   sequence_name \t start_pos
    // Strand is recovered at query time from a separate packed bit-vector file.
    std::ifstream in(path);
    if (!in) { std::cerr << "Failed to open metadata: " << path << "\n"; return false; }
    std::string line;
    if (!std::getline(in, line)) { std::cerr << "Empty metadata file\n"; return false; }
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::size_t first_tab = line.find('\t');
        if (first_tab == std::string::npos) continue;
        sequence_name.push_back(line.substr(0, first_tab));
        start_pos.push_back(std::stoi(line.substr(first_tab + 1)));
    }
    return true;
}

static bool load_strand_file(const std::string& path, std::size_t n_vectors, std::vector<std::uint8_t>& out_strand) {
    out_strand.resize(n_vectors * KMER_CANONICAL_BYTES);
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Failed to open strand file: " << path << "\n"; return false; }
    in.read(reinterpret_cast<char*>(out_strand.data()), static_cast<std::streamsize>(out_strand.size()));
    if (!in || in.gcount() != static_cast<std::streamsize>(out_strand.size())) return false;
    return true;
}

static char strand_from_overlap(const std::uint8_t* read_presence, const std::uint8_t* read_strand,
    const std::uint8_t* ref_presence, const std::uint8_t* ref_strand, std::size_t n_bytes) {
    int same = 0, total = 0;
    for (std::size_t b = 0; b < n_bytes; ++b) {
        std::uint8_t overlap = read_presence[b] & ref_presence[b];
        if (!overlap) continue;
        for (int bit = 7; bit >= 0; --bit) {
            if ((overlap >> bit) & 1) {
                ++total;
                if (((read_strand[b] >> bit) & 1) == ((ref_strand[b] >> bit) & 1)) ++same;
            }
        }
    }
    return (same >= total - same) ? '+' : '-';
}

static void set_large_stream_buffer(std::ostream& out, std::vector<char>& buf, std::size_t size = 2 * 1024 * 1024) {
    buf.resize(size);
    if (out.rdbuf()) out.rdbuf()->pubsetbuf(buf.data(), static_cast<std::streamsize>(buf.size()));
}

// ----- Output queue + opportunistic writer -----

struct OutputState {
    std::mutex mutex;
    std::queue<std::pair<int, std::vector<std::string>>> queue;
    std::unordered_map<int, std::vector<std::string>> pending;
    int next_to_write = 0;
    std::ostream* out = nullptr;

    // Enqueue a completed batch of TSV lines.
    // The batch is identified by a monotonically increasing batch_id so that
    // we can maintain global output order even when workers finish out of order.
    void push(int batch_id, std::vector<std::string> lines) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.emplace(batch_id, std::move(lines));
    }

    // Try to become the writer: drain the current queue into a "pending" map,
    // then write any contiguous range of ready batches starting from next_to_write.
    //
    // Concurrency model:
    //   - Any worker may call try_drain() after producing a batch.
    //   - At most one worker holds the mutex and writes at a time, so the
    //     underlying stream never sees interleaved bytes.
    //   - Batches are written in order of batch_id to keep the TSV stable.
    void try_drain() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!out) return;
        while (!queue.empty()) {
            auto p = std::move(queue.front());
            queue.pop();
            pending[p.first] = std::move(p.second);
        }
        while (pending.count(next_to_write)) {
            std::vector<std::string> chunk = std::move(pending[next_to_write]);
            pending.erase(next_to_write);
            ++next_to_write;
            // Keep the mutex locked while writing to ensure only one
            // thread writes to the stream at a time; this avoids
            // interleaving bytes from different lines.
            for (const auto& line : chunk) {
                out->write(line.data(), static_cast<std::streamsize>(line.size()));
            }
        }
    }
};

// ----- Worker: process one batch (kmer -> search -> align) and push; then try_drain -----

static void process_batch(
    int batch_id,
    std::size_t start,
    std::size_t end,
    const std::vector<ReadRecord>& valid,
    const std::unordered_map<int, SearchBundle>* bundles_by_len,
    int default_search_len,
    const std::unordered_map<std::string, std::string>* ref_seqs,  // null if !do_align; key = meta sequence_name
    const std::unordered_map<std::string, std::string>* ref_sam_names, // null if not normalizing; key = full meta sequence_name
    bool canonical,
    bool do_align,
    bool secondary,
    bool output_sam,
    bool paired_sam,
    int min_sep_len,
    std::size_t d_bytes,
    const faiss::idx_t max_k,
    std::vector<std::uint8_t>& packed_buf,
    std::vector<std::uint8_t>& strand_buf,
    std::vector<KmerVector>* kmer_buf,           // null if canonical
    std::vector<int32_t>& distances_buf,
    std::vector<faiss::idx_t>& labels_buf,
    std::vector<std::uint8_t>& ref_presence_buf,
    wfa::WFAlignerGapAffine* aligner,
    bool wfa_banded,
    bool wfa_adaptive,
    bool wfa_zdrop,
    OutputState& output
) {
    // Local batch size in reads (half-open interval [start, end)).
    const std::size_t batch_size = end - start;
    std::vector<std::string> lines;
    lines.reserve(output_sam ? batch_size : (secondary ? batch_size * 2 : batch_size));
    auto bundle_for_idx = [&](std::size_t idx) -> const SearchBundle& {
        int search_len = static_cast<int>(valid[idx].seq.size());
        if (bundles_by_len) {
            auto it = bundles_by_len->find(search_len);
            if (it != bundles_by_len->end()) return it->second;
            auto it_def = bundles_by_len->find(default_search_len);
            if (it_def != bundles_by_len->end()) return it_def->second;
        }
        throw std::runtime_error("Missing search bundle for read length");
    };
    auto query_seq_for_alignment = [&](std::size_t idx) -> const std::string& {
        return valid[idx].full_seq.empty() ? valid[idx].seq : valid[idx].full_seq;
    };
    auto query_qual_for_output = [&](std::size_t idx) -> const std::string& {
        return valid[idx].full_qual.empty() ? valid[idx].qual : valid[idx].full_qual;
    };
    auto query_anchor_len_for_search = [&](std::size_t idx) -> std::size_t {
        return valid[idx].seq.size();
    };
    auto query_anchor_offset_for_search = [&](std::size_t idx) -> std::size_t {
        return valid[idx].search_offset;
    };
    auto full_start_from_search_hit = [&](int hit_start, std::size_t idx, char strand_char) -> int {
        const std::size_t anchor_len = query_anchor_len_for_search(idx);
        const std::size_t full_len = query_seq_for_alignment(idx).size();
        const std::size_t anchor_off = query_anchor_offset_for_search(idx);
        if (strand_char == '-') {
            const std::size_t suffix_len = full_len - (anchor_off + anchor_len);
            return hit_start - static_cast<int>(suffix_len);
        }
        return hit_start - static_cast<int>(anchor_off);
    };

    // 1) K-mer computation for this batch.
    //    We either:
    //      - build canonical presence/strand bit-vectors directly into packed_buf/strand_buf, or
    //      - build full KmerVector objects and then pack them to bytes.
    if (canonical) {
        (void)get_canonical_map();
        CanonicalPresence pres;
        CanonicalStrand strand;
        for (std::size_t i = start; i < end; ++i) {
            std::size_t local = i - start;
            compute_kmer_vector_5_canonical(valid[i].seq.data(), valid[i].seq.size(), pres, strand);
            pack_canonical_big_endian(pres, packed_buf.data() + local * KMER_CANONICAL_BYTES);
            pack_canonical_big_endian(strand, strand_buf.data() + local * KMER_CANONICAL_BYTES);
        }
    } else {
        for (std::size_t i = start; i < end; ++i) {
            std::size_t local = i - start;
            compute_kmer_vector_5(valid[i].seq.data(), valid[i].seq.size(), (*kmer_buf)[local]);
        }
        for (std::size_t local = 0; local < batch_size; ++local)
            pack_kmer_vector_big_endian((*kmer_buf)[local], packed_buf.data() + local * d_bytes);
    }

    // 2) FAISS search for this batch.
    for (std::size_t local = 0; local < batch_size; ++local) {
        const std::size_t global_i = start + local;
        const SearchBundle& b = bundle_for_idx(global_i);
        int32_t* dptr = distances_buf.data() + local * max_k;
        faiss::idx_t* lptr = labels_buf.data() + local * max_k;
        for (faiss::idx_t j = 0; j < max_k; ++j) {
            dptr[j] = 2147483647;
            lptr[j] = -1;
        }
        b.index->search(1, packed_buf.data() + local * d_bytes, b.k, dptr, lptr, nullptr);
    }

    // ----- SAM output (paired-aware RNEXT/PNEXT) -----
    // In SAM mode we output one record per mate, even if unmapped.
    // With --secondary: align primary and best separated secondary; emit one record
    // for the hit with the higher alignment score (tie -> primary).
    if (output_sam) {
        auto norm_ref = [&](const std::string& full_ref) -> const std::string& {
            if (!ref_sam_names) return full_ref;
            auto it = ref_sam_names->find(full_ref);
            if (it == ref_sam_names->end()) return full_ref;
            return it->second;
        };

        std::vector<char> strand_char(batch_size, '+');
        std::vector<bool> mapped(batch_size, false);
        std::vector<std::size_t> meta_idx_for(batch_size, static_cast<std::size_t>(-1));
        std::vector<int> pos_for(batch_size, 0); // 1-based; 0 when unmapped
        std::vector<int> ref_span_for(batch_size, 0); // reference-consuming bases from CIGAR
        std::vector<int> end_for(batch_size, 0);      // 1-based inclusive end on reference
        std::vector<std::string> cigar_for(batch_size, "*");
        std::vector<int> align_score_for(batch_size, 0); // WFA score; SAM optional AS:i (mapped only)

        struct SamAlign {
            bool ok = false;
            int score = 0;
            char strand_char = '+';
            std::string cigar = "*";
            int pos_1based = 0;
            int ref_span = 0;
            int end_1based = 0;
            std::size_t meta_idx = 0;
        };

        auto compute_sam_align = [&](std::size_t meta_idx, faiss::idx_t lab, std::size_t local, std::size_t global_i) -> SamAlign {
            SamAlign a;
            a.meta_idx = meta_idx;
            const SearchBundle& b = bundle_for_idx(global_i);
            if (canonical && !b.ref_strand.empty()) {
                b.index->reconstruct(lab, ref_presence_buf.data());
                a.strand_char = strand_from_overlap(
                    packed_buf.data() + local * d_bytes,
                    strand_buf.data() + local * KMER_CANONICAL_BYTES,
                    ref_presence_buf.data(),
                    b.ref_strand.data() + meta_idx * KMER_CANONICAL_BYTES,
                    KMER_CANONICAL_BYTES);
            } else {
                a.strand_char = b.meta_strand[meta_idx];
            }
            const int full_start = full_start_from_search_hit(b.meta_start[meta_idx], global_i, a.strand_char);
            a.pos_1based = full_start + 1;

            if (!do_align || !ref_seqs) return a;

            auto it = ref_seqs->find(b.meta_name[meta_idx]);
            if (it == ref_seqs->end()) return a;

            std::string padded_ref = build_padded_ref(
                it->second,
                full_start,
                query_seq_for_alignment(global_i).size(),
                a.strand_char);
            configure_wfa_heuristics(aligner, wfa_banded, wfa_adaptive, wfa_zdrop);
            aligner->alignEndsFree(
                query_seq_for_alignment(global_i), 0, 0,
                padded_ref,
                static_cast<int>(padded_ref.size()),
                static_cast<int>(padded_ref.size()));

            std::string cigar_raw = aligner->getCIGAR(/*showMismatches=*/false);
            if (!cigar_raw.empty()) {
                a.cigar = std::move(cigar_raw);
                sam_adjust_pos_strip_padding_cigar(
                    a.strand_char,
                    full_start,
                    query_seq_for_alignment(global_i).size(),
                    it->second.size(),
                    a.cigar,
                    a.pos_1based,
                    a.cigar,
                    a.ref_span,
                    a.end_1based);
            } else {
                a.ref_span = 0;
                a.end_1based = 0;
            }
            a.score = aligner->getAlignmentScore();
            a.ok = true;
            return a;
        };

        for (std::size_t local = 0; local < batch_size; ++local) {
            const std::size_t global_i = start + local;
            const SearchBundle& b = bundle_for_idx(global_i);
            faiss::idx_t lab_primary = labels_buf[local * max_k];
            if (lab_primary < 0) continue;
            std::size_t idx_primary = static_cast<std::size_t>(lab_primary);
            if (idx_primary >= b.meta_name.size()) continue;

            SamAlign pri = compute_sam_align(idx_primary, lab_primary, local, global_i);

            if (!secondary) {
                if (!pri.ok) continue;
                strand_char[local] = pri.strand_char;
                meta_idx_for[local] = pri.meta_idx;
                pos_for[local] = pri.pos_1based;
                cigar_for[local] = std::move(pri.cigar);
                ref_span_for[local] = pri.ref_span;
                end_for[local] = pri.end_1based;
                align_score_for[local] = pri.score;
                mapped[local] = true;
                continue;
            }

            const int pri_start = b.meta_start[idx_primary];
            const int min_sep = min_sep_len;
            faiss::idx_t lab_secondary = -1;
            std::size_t idx_secondary = 0;
            int32_t best_sec_dist = 2147483647;
            for (faiss::idx_t j = 1; j < b.k; ++j) {
                faiss::idx_t lab = labels_buf[local * max_k + j];
                if (lab < 0) continue;
                std::size_t idx = static_cast<std::size_t>(lab);
                if (idx >= b.meta_name.size()) continue;
                int sec_start = b.meta_start[idx];
                if (std::abs(sec_start - pri_start) < min_sep) continue;
                int32_t d = distances_buf[local * max_k + j];
                if (d < best_sec_dist) {
                    best_sec_dist = d;
                    lab_secondary = lab;
                    idx_secondary = idx;
                }
            }

            SamAlign sec;
            if (lab_secondary >= 0) {
                sec = compute_sam_align(idx_secondary, lab_secondary, local, global_i);
            }

            const SamAlign* win = nullptr;
            if (pri.ok && sec.ok) {
                win = (sec.score > pri.score) ? &sec : &pri;
            } else if (pri.ok) {
                win = &pri;
            } else if (sec.ok) {
                win = &sec;
            } else {
                continue;
            }

            strand_char[local] = win->strand_char;
            meta_idx_for[local] = win->meta_idx;
            pos_for[local] = win->pos_1based;
            cigar_for[local] = win->cigar;
            ref_span_for[local] = win->ref_span;
            end_for[local] = win->end_1based;
            align_score_for[local] = win->score;
            mapped[local] = true;
        }

        // Emit SAM: one record per mate.
        for (std::size_t local = 0; local < batch_size; ++local) {
            const std::size_t global_i = start + local;
            const bool is_read1 = paired_sam && ((local % 2) == 0);
            const std::size_t mate_local = paired_sam ? (local ^ 1) : local;

            const bool mapped_i = mapped[local];
            const bool mapped_m = paired_sam ? mapped[mate_local] : false;
            const int pos_i = mapped_i ? pos_for[local] : 0;
            const int pnext = (paired_sam && mapped_m) ? pos_for[mate_local] : 0;
            int tlen = 0;

            if (paired_sam && mapped_i && mapped_m) {
                // TLEN only makes sense when both mates align to the same reference.
                const SearchBundle& b_i = bundle_for_idx(global_i);
                const SearchBundle& b_m = bundle_for_idx(start + mate_local);
                const std::string& rname_i = b_i.meta_name[meta_idx_for[local]];
                const std::string& rname_m = b_m.meta_name[meta_idx_for[mate_local]];
                if (rname_i == rname_m) {
                    const int span_i = ref_span_for[local];
                    const int span_m = ref_span_for[mate_local];
                    if (span_i > 0 && span_m > 0) {
                        const int pos_m = pos_for[mate_local];
                        const int end_i = end_for[local];
                        const int end_m = end_for[mate_local];
                        const int outer_start = std::min(pos_i, pos_m);
                        const int outer_end = std::max(end_i, end_m);
                        tlen = outer_end - outer_start + 1;
                        // Sign convention: negative when this read starts to the right.
                        if (pos_i > pos_m) tlen = -tlen;
                    }
                }
            }

            int flag = 0;
            if (paired_sam) flag |= is_read1 ? 64 : 128; // read1/read2
            if (!mapped_i) flag |= 4;                   // segment unmapped
            if (mapped_i && strand_char[local] == '-') flag |= 16; // query mapped to reverse strand

            if (paired_sam) {
                if (mapped_m && strand_char[mate_local] == '-') flag |= 32; // mate reverse
                if (mapped_i && !mapped_m) flag |= 8;                        // mate unmapped
            }

            std::string line;
            line.reserve(256);
            append_tsv_field(line, valid[global_i].id); // QNAME
            line += '\t';
            line += std::to_string(flag);
            line += '\t';
            const std::string* rname_i_norm = nullptr;
            if (mapped_i) {
                const SearchBundle& b_i = bundle_for_idx(global_i);
                rname_i_norm = &norm_ref(b_i.meta_name[meta_idx_for[local]]);
            }
            if (mapped_i) line += *rname_i_norm; else line += '*'; // RNAME
            line += '\t';
            line += std::to_string(pos_i); // POS
            line += "\t255\t";            // MAPQ
            if (mapped_i) line += cigar_for[local]; else line += '*'; // CIGAR
            line += '\t';
            // RNEXT: mate reference name (or '=' if same as RNAME); '*' if mate unmapped.
            if (paired_sam && mapped_m) {
                const SearchBundle& b_m = bundle_for_idx(start + mate_local);
                const std::string& rname_m_norm = norm_ref(b_m.meta_name[meta_idx_for[mate_local]]);
                if (mapped_i && rname_i_norm && *rname_i_norm == rname_m_norm) line += '=';
                else line += rname_m_norm;
            } else {
                line += '*';
            }
            line += '\t';
            line += std::to_string(pnext); // PNEXT
            line += '\t';
            line += std::to_string(tlen); // TLEN
            line += '\t';
            append_tsv_field(line, query_seq_for_alignment(global_i)); // SEQ
            line += '\t';
            append_tsv_field(line, query_qual_for_output(global_i)); // QUAL
            if (mapped_i && do_align) {
                line += "\tAS:i:";
                line += std::to_string(align_score_for[local]);
            }
            line += '\n';
            lines.push_back(std::move(line));
        }

        output.push(batch_id, std::move(lines));
        output.try_drain();
        return;
    }

    // Helper: append one TSV line (default) or one SAM record (when output_sam).
    auto emit_line = [&](std::size_t global_i, std::size_t meta_idx, faiss::idx_t lab_hit, int32_t dist, int rank) {
        std::string line;
        line.reserve(256);
        char strand_char;
        const SearchBundle& b = bundle_for_idx(global_i);
        if (canonical && !b.ref_strand.empty()) {
            b.index->reconstruct(lab_hit, ref_presence_buf.data());
            strand_char = strand_from_overlap(
                packed_buf.data() + (global_i - start) * d_bytes,
                strand_buf.data() + (global_i - start) * KMER_CANONICAL_BYTES,
                ref_presence_buf.data(),
                b.ref_strand.data() + meta_idx * KMER_CANONICAL_BYTES,
                KMER_CANONICAL_BYTES);
        } else {
            strand_char = b.meta_strand[meta_idx];
        }
        const int full_start = full_start_from_search_hit(b.meta_start[meta_idx], global_i, strand_char);
        int align_score = 0;
        std::string cigar;
        if (do_align && ref_seqs) {
            auto it = ref_seqs->find(b.meta_name[meta_idx]);
            if (it != ref_seqs->end()) {
                std::string padded_ref = build_padded_ref(
                    it->second,
                    full_start,
                    query_seq_for_alignment(global_i).size(),
                    strand_char);
                configure_wfa_heuristics(aligner, wfa_banded, wfa_adaptive, wfa_zdrop);
                aligner->alignEndsFree(query_seq_for_alignment(global_i), 0, 0,
                    padded_ref, static_cast<int>(padded_ref.size()), static_cast<int>(padded_ref.size()));
                align_score = aligner->getAlignmentScore();
                if (output_sam) cigar = aligner->getCIGAR(/*showMismatches=*/false);
            }
        }

        if (output_sam) {
            // SAM columns (mandatory): QNAME, FLAG, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL
            int flag = 0;
            if (strand_char == '-') flag |= 16;  // read mapped to reverse strand
            if (rank == 2) flag |= 256;         // secondary alignment

            append_tsv_field(line, valid[global_i].id);
            line += '\t';
            line += std::to_string(flag);
            line += '\t';
            append_tsv_field(line, b.meta_name[meta_idx]);
            line += '\t';
            line += std::to_string(full_start + 1); // SAM is 1-based
            line += "\t255\t";
            line += (cigar.empty() ? "*" : cigar);
            line += "\t*\t0\t0\t";
            append_tsv_field(line, query_seq_for_alignment(global_i));
            line += "\t";
            append_tsv_field(line, query_qual_for_output(global_i));
            if (do_align) {
                line += "\tAS:i:";
                line += std::to_string(align_score);
            }
            line += "\n";
            lines.push_back(std::move(line));
            return;
        }

        // TSV output (existing format).
        append_tsv_field(line, valid[global_i].id);
        line += '\t';
        line += std::to_string(query_seq_for_alignment(global_i).size());
        line += '\t';
        line += strand_char;
        line += '\t';
        append_tsv_field(line, b.meta_name[meta_idx]);
        line += '\t';
        line += std::to_string(full_start);
        line += '\t';
        line += std::to_string(static_cast<int>(dist));
        if (do_align) { line += '\t'; line += std::to_string(align_score); }
        if (secondary) { line += '\t'; line += std::to_string(rank); }
        line += "\t0\t";
        append_tsv_field(line, query_seq_for_alignment(global_i));
        line += '\n';
        lines.push_back(std::move(line));
    };

    // 3) For each query: primary hit (and optionally secondary), then format TSV line(s).
    for (std::size_t local = 0; local < batch_size; ++local) {
        const std::size_t global_i = start + local;
        const SearchBundle& b = bundle_for_idx(global_i);
        faiss::idx_t lab_primary = labels_buf[local * max_k];
        if (lab_primary < 0) continue;
        std::size_t idx_primary = static_cast<std::size_t>(lab_primary);
        if (idx_primary >= b.meta_name.size()) continue;
        const int pri_start = b.meta_start[idx_primary];
        int32_t dist_primary = distances_buf[local * max_k];

        if (!secondary) {
            emit_line(global_i, idx_primary, lab_primary, dist_primary, 1);
            continue;
        }

        // Primary line (rank 1).
        emit_line(global_i, idx_primary, lab_primary, dist_primary, 1);

        // Secondary: among remaining hits, choose lowest distance s.t. |sec_start - pri_start| >= max_length.
        const int min_sep = min_sep_len;
        faiss::idx_t lab_secondary = -1;
        std::size_t idx_secondary = 0;
        int32_t best_sec_dist = 2147483647;
        for (faiss::idx_t j = 1; j < b.k; ++j) {
            faiss::idx_t lab = labels_buf[local * max_k + j];
            if (lab < 0) continue;
            std::size_t idx = static_cast<std::size_t>(lab);
            if (idx >= b.meta_name.size()) continue;
            int sec_start = b.meta_start[idx];
            if (std::abs(sec_start - pri_start) < min_sep) continue;
            int32_t d = distances_buf[local * max_k + j];
            if (d < best_sec_dist) {
                best_sec_dist = d;
                lab_secondary = lab;
                idx_secondary = idx;
            }
        }
        if (lab_secondary >= 0) {
            emit_line(global_i, idx_secondary, lab_secondary, best_sec_dist, 2);
        }
    }

    output.push(batch_id, std::move(lines));
    output.try_drain();
}

int main(int argc, char** argv) {
#ifdef _OPENMP
    // Initialize the OpenMP runtime early so OMP_DISPLAY_ENV=TRUE can print
    // before any early exit (otherwise the first omp_* call may be too late).
    if (std::getenv("OMP_DISPLAY_ENV")) {
        (void)omp_get_num_procs();
    }
#endif
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <fastq_r1> <ref_name> [fastq_r2]"
                  << " [--max-length N] [--step-size N] [--nlist N] [--nprobe N] [--threads N]"
                  << " [--canonical] [--align] [--secondary] [--sam] [--variable] [--output out.tsv]"
                  << " [--banded] [--adaptive] [--zdrop]\n";
        return 1;
    }

    // Positional arguments:
    //   fastq_r1 : gzipped FASTQ file for read 1.
    //   ref_name : base name used to locate the FAISS index + metadata.
    // Optional positional:
    //   fastq_r2 : if present, treated as paired FASTQ (read ids kept as-is).
    std::string fastq_r1 = argv[1];
    std::string ref_name = argv[2];
    std::string fastq_r2;
    int max_length = DEFAULT_MAX_LENGTH;
    int step_size = DEFAULT_STEP_SIZE;
    int nlist = 0, nprobe = 0;
    int num_threads = DEFAULT_THREADS;
    bool canonical = false;
    bool do_align = false;
    bool secondary = false;
    bool output_sam = false;
    bool variable = false;
    std::string output_path;
    bool wfa_banded = false;
    bool wfa_adaptive = false;
    bool wfa_zdrop = false;

    int i = 3;
    if (i < argc && std::string(argv[i]).rfind("--", 0) != 0)
        fastq_r2 = argv[i++];
    // Parse optional flags controlling windowing, index selection and alignment.
    while (i < argc) {
        std::string arg = argv[i];
        if (arg == "--max-length" && i + 1 < argc) { max_length = std::stoi(argv[++i]); }
        else if (arg == "--step-size" && i + 1 < argc) { step_size = std::stoi(argv[++i]); }
        else if (arg == "--nlist" && i + 1 < argc) { nlist = std::stoi(argv[++i]); }
        else if (arg == "--nprobe" && i + 1 < argc) { nprobe = std::stoi(argv[++i]); }
        else if (arg == "--threads" && i + 1 < argc) { num_threads = std::stoi(argv[++i]); }
        else if (arg == "--canonical") { canonical = true; }
        else if (arg == "--align") { do_align = true; }
        else if (arg == "--secondary") { secondary = true; }
        else if (arg == "--sam") { output_sam = true; }
        else if (arg == "--variable") { variable = true; }
        else if (arg == "--banded") { wfa_banded = true; }
        else if (arg == "--adaptive") { wfa_adaptive = true; }
        else if (arg == "--zdrop") { wfa_zdrop = true; }
        else if (arg == "--output" && i + 1 < argc) { output_path = argv[++i]; }
        else if (arg.rfind("--", 0) == 0) { std::cerr << "Unknown option: " << arg << "\n"; return 1; }
        else { std::cerr << "Unexpected argument: " << arg << "\n"; return 1; }
        ++i;
    }

    if (output_sam && !do_align) {
        std::cerr << "--sam requires --align\n";
        return 1;
    }
    if ((wfa_banded || wfa_adaptive || wfa_zdrop) && !do_align) {
        std::cerr << "WFA flags --banded/--adaptive/--zdrop require --align\n";
        return 1;
    }
    auto choose_variable_len = [](std::size_t full_len) -> int {
        if (full_len < 50) return -1;
        if (full_len < 75) return 50;
        if (full_len < 100) return 75;
        if (full_len < 150) return 100;
        return 150;
    };

    const std::string reads_base = "/Volumes/OWC Envoy Ultra/reads";
    // Convenience resolver for read paths:
    //   - If `path` is absolute and exists, use as-is.
    //   - Otherwise, attempt to resolve under the global reads_base.
    auto resolve_fastq = [&reads_base](const std::string& path) -> std::string {
        if (path.empty()) return path;
        std::filesystem::path p(path);
        if (p.is_absolute() && std::filesystem::exists(p)) return path;
        if (!p.is_absolute()) {
            std::string under = reads_base;
            if (!under.empty() && under.back() != '/') under += '/';
            std::string candidate = under + path;
            if (std::filesystem::exists(candidate)) return candidate;
        }
        return path;
    };
    std::string r1_resolved = resolve_fastq(fastq_r1);
    std::string r2_resolved = fastq_r2.empty() ? std::string() : resolve_fastq(fastq_r2);

    // Default output path:
    //   - If the input FASTQ lives under reads_base, mirror the directory
    //     structure under map_outputs/ and name the file using the leaf folder
    //     plus a suffix describing step size / IVF params / canonical.
    //   - Otherwise, fall back to a simple filename in the CWD.
    if (output_path.empty()) {
        namespace fs = std::filesystem;
        const std::string base_dir = "/Volumes/OWC Envoy Ultra/reads";
        const std::string map_base = "/Volumes/OWC Envoy Ultra/map_outputs";
        fs::path r1_path(r1_resolved);
        std::string input_dir = r1_path.parent_path().string();
        std::string out_suffix = "_step" + std::to_string(step_size);
        if (nlist > 0 && nprobe > 0) out_suffix += "_nlist" + std::to_string(nlist) + "_nprobe" + std::to_string(nprobe);
        if (canonical) out_suffix += "_canonical";
        if (variable) out_suffix += "_variable";
        if (input_dir.size() >= base_dir.size() && input_dir.compare(0, base_dir.size(), base_dir) == 0) {
            std::string rel = input_dir.substr(base_dir.size());
            while (!rel.empty() && rel[0] == '/') rel.erase(0, 1);
            std::string folder_name = fs::path(input_dir).filename().string();
            if (folder_name.empty()) folder_name = "output";
            std::string output_dir = rel.empty() ? map_base : (map_base + "/" + rel);
            std::error_code ec;
            fs::create_directories(output_dir, ec);
            output_path = output_dir + "/" + folder_name + "_faiss" + out_suffix + (output_sam ? ".sam" : ".tsv");
        } else {
            output_path = "anniemap_faiss" + out_suffix + (output_sam ? ".sam" : ".tsv");
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    std::ostream& status_out = output_sam ? std::cerr : std::cout;
    const bool paired_sam = output_sam && !fastq_r2.empty();
    // 1) Load FASTQ(s) and apply a simple length filter/cropping policy.
    //    Default: keep reads >= max_length and center-crop to max_length.
    //    --variable: keep reads >= 50 and center-crop to one of {50,75,100,150}
    //                using the largest bucket <= full read length.
    std::vector<ReadRecord> valid;
    if (!paired_sam) {
        std::vector<ReadRecord> reads;
        if (!read_fastq_one(r1_resolved, reads)) return 1;
        if (!fastq_r2.empty() && !read_fastq_one(r2_resolved, reads)) return 1;

        valid.reserve(reads.size());
        for (auto& r : reads) {
            const std::size_t full_len = r.seq.size();
            const int target_len = variable ? choose_variable_len(full_len) : (full_len >= static_cast<std::size_t>(max_length) ? max_length : -1);
            if (target_len > 0) {
                if (full_len > static_cast<std::size_t>(target_len)) {
                    r.full_seq = r.seq;
                    r.full_qual = r.qual;
                    const std::size_t crop_len = static_cast<std::size_t>(target_len);
                    r.search_offset = (full_len - crop_len) / 2;
                    r.seq = r.full_seq.substr(r.search_offset, crop_len);
                    if (r.full_qual.size() >= r.search_offset + crop_len) {
                        r.qual = r.full_qual.substr(r.search_offset, crop_len);
                    } else if (r.qual.size() > crop_len) {
                        r.qual.resize(crop_len);
                    }
                }
                valid.push_back(std::move(r));
            }
        }
        reads.clear();
    } else {
        // Paired SAM mode: build read-pairs, normalize QNAME (option B), and keep mates adjacent.
        std::vector<ReadRecord> r1_reads;
        std::vector<ReadRecord> r2_reads;
        if (!read_fastq_one(r1_resolved, r1_reads)) return 1;
        if (!read_fastq_one(r2_resolved, r2_reads)) return 1;

        auto normalize_copy = [&](const std::string& s) -> std::string {
            std::string t = s;
            normalize_pair_qname_inplace(t);
            return t;
        };

        const bool ordering_ok = (r1_reads.size() == r2_reads.size() &&
            (r1_reads.empty() ||
             (normalize_copy(r1_reads.front().id) == normalize_copy(r2_reads.front().id) &&
              normalize_copy(r1_reads.back().id) == normalize_copy(r2_reads.back().id))));

        if (ordering_ok) {
            valid.reserve(2 * r1_reads.size());
            const std::size_t n = r1_reads.size();
            for (std::size_t idx = 0; idx < n; ++idx) {
                ReadRecord r1 = std::move(r1_reads[idx]);
                ReadRecord r2 = std::move(r2_reads[idx]);
                const std::size_t len1 = r1.seq.size();
                const std::size_t len2 = r2.seq.size();
                const int target1 = variable ? choose_variable_len(len1) : (len1 >= static_cast<std::size_t>(max_length) ? max_length : -1);
                const int target2 = variable ? choose_variable_len(len2) : (len2 >= static_cast<std::size_t>(max_length) ? max_length : -1);
                if (target1 < 0 || target2 < 0) continue;

                normalize_pair_qname_inplace(r1.id);
                normalize_pair_qname_inplace(r2.id);
                if (len1 > static_cast<std::size_t>(target1)) {
                    r1.full_seq = r1.seq;
                    r1.full_qual = r1.qual;
                    const std::size_t full_len = r1.full_seq.size();
                    const std::size_t crop_len = static_cast<std::size_t>(target1);
                    r1.search_offset = (full_len - crop_len) / 2;
                    r1.seq = r1.full_seq.substr(r1.search_offset, crop_len);
                    if (r1.full_qual.size() >= r1.search_offset + crop_len) {
                        r1.qual = r1.full_qual.substr(r1.search_offset, crop_len);
                    } else if (r1.qual.size() > crop_len) {
                        r1.qual.resize(crop_len);
                    }
                }
                if (len2 > static_cast<std::size_t>(target2)) {
                    r2.full_seq = r2.seq;
                    r2.full_qual = r2.qual;
                    const std::size_t full_len = r2.full_seq.size();
                    const std::size_t crop_len = static_cast<std::size_t>(target2);
                    r2.search_offset = (full_len - crop_len) / 2;
                    r2.seq = r2.full_seq.substr(r2.search_offset, crop_len);
                    if (r2.full_qual.size() >= r2.search_offset + crop_len) {
                        r2.qual = r2.full_qual.substr(r2.search_offset, crop_len);
                    } else if (r2.qual.size() > crop_len) {
                        r2.qual.resize(crop_len);
                    }
                }
                valid.push_back(std::move(r1));
                valid.push_back(std::move(r2));
            }
        } else {
            status_out << "R1/R2 ordering mismatch; pairing by normalized name (slower)\n";
            std::unordered_map<std::string, ReadRecord> r2_by_name;
            r2_by_name.reserve(r2_reads.size());
            for (auto& r : r2_reads) {
                normalize_pair_qname_inplace(r.id);
                r2_by_name.emplace(r.id, std::move(r));
            }
            for (auto& r1 : r1_reads) {
                normalize_pair_qname_inplace(r1.id);
                auto it = r2_by_name.find(r1.id);
                if (it == r2_by_name.end()) continue;

                ReadRecord r2 = std::move(it->second);
                r2_by_name.erase(it);

                const std::size_t len1 = r1.seq.size();
                const std::size_t len2 = r2.seq.size();
                const int target1 = variable ? choose_variable_len(len1) : (len1 >= static_cast<std::size_t>(max_length) ? max_length : -1);
                const int target2 = variable ? choose_variable_len(len2) : (len2 >= static_cast<std::size_t>(max_length) ? max_length : -1);
                if (target1 < 0 || target2 < 0) continue;

                if (len1 > static_cast<std::size_t>(target1)) {
                    r1.full_seq = r1.seq;
                    r1.full_qual = r1.qual;
                    const std::size_t full_len = r1.full_seq.size();
                    const std::size_t crop_len = static_cast<std::size_t>(target1);
                    r1.search_offset = (full_len - crop_len) / 2;
                    r1.seq = r1.full_seq.substr(r1.search_offset, crop_len);
                    if (r1.full_qual.size() >= r1.search_offset + crop_len) {
                        r1.qual = r1.full_qual.substr(r1.search_offset, crop_len);
                    } else if (r1.qual.size() > crop_len) {
                        r1.qual.resize(crop_len);
                    }
                }
                if (len2 > static_cast<std::size_t>(target2)) {
                    r2.full_seq = r2.seq;
                    r2.full_qual = r2.qual;
                    const std::size_t full_len = r2.full_seq.size();
                    const std::size_t crop_len = static_cast<std::size_t>(target2);
                    r2.search_offset = (full_len - crop_len) / 2;
                    r2.seq = r2.full_seq.substr(r2.search_offset, crop_len);
                    if (r2.full_qual.size() >= r2.search_offset + crop_len) {
                        r2.qual = r2.full_qual.substr(r2.search_offset, crop_len);
                    } else if (r2.qual.size() > crop_len) {
                        r2.qual.resize(crop_len);
                    }
                }
                valid.push_back(std::move(r1));
                valid.push_back(std::move(r2));
            }
        }
    }

    status_out << "Reads after length filter (" << (variable ? ">= 50 variable buckets" : (">= " + std::to_string(max_length))) << "): " << valid.size() << "\n";
    if (valid.empty()) { std::cerr << "No reads to process.\n"; return 1; }

    const std::size_t n_reads = valid.size();
    const std::size_t d_bytes = canonical ? KMER_CANONICAL_BYTES : (KMER_DIM / 8);

    // 2) Load FAISS index bundle(s) and metadata produced by faiss_create_kmer.
    std::set<int> active_search_lens;
    for (const auto& r : valid) active_search_lens.insert(static_cast<int>(r.seq.size()));
    if (active_search_lens.empty()) { std::cerr << "No active search lengths.\n"; return 1; }

    std::unordered_map<int, SearchBundle> bundles_by_len;
    bundles_by_len.reserve(active_search_lens.size());
    std::vector<faiss::IndexBinary*> owned_indices;
    owned_indices.reserve(active_search_lens.size());
    faiss::idx_t max_k = 1;
    for (int search_len : active_search_lens) {
        std::ostringstream suffix_oss;
        suffix_oss << "_" << search_len << "_step" << step_size;
        if (nlist > 0 && nprobe > 0) suffix_oss << "_nlist" << nlist << "_nprobe" << nprobe;
        if (canonical) suffix_oss << "_canonical";
        suffix_oss << "_binary";
        const std::string suffix = suffix_oss.str();
        const std::string index_path = "ref_vectors/" + ref_name + suffix + ".faiss";
        const std::string metadata_path = "ref_vectors/" + ref_name + suffix + "_metadata.tsv";
        const std::string strand_path = "ref_vectors/" + ref_name + suffix + "_strand.bin";

        faiss::IndexBinary* index = faiss::read_index_binary(index_path.c_str());
        if (!index) {
            std::cerr << "Failed to load FAISS index: " << index_path << "\n";
            for (faiss::IndexBinary* p : owned_indices) delete p;
            return 1;
        }
        owned_indices.push_back(index);
        if (canonical) {
            if (auto* ivf = dynamic_cast<faiss::IndexBinaryIVF*>(index)) ivf->make_direct_map(true);
        }
        {
            const size_t query_batch_size = 100000;
            if (auto* ivf = dynamic_cast<faiss::IndexBinaryIVF*>(index)) {
                if (auto* flat = dynamic_cast<faiss::IndexBinaryFlat*>(ivf->quantizer))
                    flat->query_batch_size = query_batch_size;
            } else if (auto* flat = dynamic_cast<faiss::IndexBinaryFlat*>(index)) {
                flat->query_batch_size = query_batch_size;
            }
        }

        SearchBundle b;
        b.search_len = search_len;
        b.index = index;
        if (canonical) {
            if (!load_metadata_tsv_canonical(metadata_path, b.meta_name, b.meta_start)) {
                for (faiss::IndexBinary* p : owned_indices) delete p;
                return 1;
            }
            if (!load_strand_file(strand_path, b.meta_name.size(), b.ref_strand)) {
                for (faiss::IndexBinary* p : owned_indices) delete p;
                return 1;
            }
        } else {
            if (!load_metadata_tsv(metadata_path, b.meta_name, b.meta_start, b.meta_strand)) {
                for (faiss::IndexBinary* p : owned_indices) delete p;
                return 1;
            }
        }
        if (secondary) {
            faiss::idx_t desired = static_cast<faiss::idx_t>(search_len / 2);
            if (desired < 2) desired = 2;
            b.k = std::min(desired, index->ntotal);
            if (b.k < 2) b.k = 1;
        } else {
            b.k = 1;
        }
        max_k = std::max(max_k, b.k);
        bundles_by_len.emplace(search_len, std::move(b));
    }
    if (secondary && max_k < 2) secondary = false;

    // 3) Optionally load all reference sequences for WFA2 alignment (keyed by meta sequence_name).
    std::unordered_map<std::string, std::string> ref_seqs;
    if (do_align) {
        std::string ref_path = "ref_sequences/" + ref_name + ".fasta";
        if (!load_fasta_all_sequences(ref_path, ref_seqs)) {
            std::cerr << "Failed to load reference for --align: " << ref_path << "\n";
            for (faiss::IndexBinary* p : owned_indices) delete p;
            return 1;
        }
        std::size_t total_bases = 0;
        for (const auto& kv : ref_seqs) total_bases += kv.second.size();
        status_out << "Loaded " << ref_seqs.size() << " reference sequence(s) from " << ref_path
                  << " (" << total_bases << " bp total) for WFA2.\n";
    }

    // Precompute reference names for SAM output (normalize FASTA headers).
    std::unordered_map<std::string, std::string> ref_sam_names;
    if (output_sam) {
        ref_sam_names.reserve(ref_seqs.size());
        for (const auto& kv : ref_seqs) {
            ref_sam_names.emplace(kv.first, normalize_ref_sequence_name(kv.first));
        }
    }

    // Header needs unique SN entries after normalization.
    std::unordered_map<std::string, std::size_t> sam_sn_to_len;
    if (output_sam) {
        sam_sn_to_len.reserve(ref_seqs.size());
        for (const auto& kv : ref_seqs) {
            const auto it = ref_sam_names.find(kv.first);
            const std::string& sn = (it == ref_sam_names.end()) ? kv.first : it->second;
            if (!sn.empty() && !sam_sn_to_len.count(sn)) sam_sn_to_len.emplace(sn, kv.second.size());
        }
    }

    // 4) Compute batch boundaries.
    // We typically use more batches than threads to improve load balancing
    // and to increase the chances that some thread can opportunistically
    // drain the output queue while others are still computing.
    const int num_batches = std::max(num_threads * 4, 1);
    std::vector<std::pair<std::size_t, std::size_t>> batches;
    std::size_t batch_size = 0;
    if (paired_sam) {
        const std::size_t n_pairs = n_reads / 2;
        const std::size_t batch_pairs_size = (n_pairs + num_batches - 1) / static_cast<std::size_t>(num_batches);
        batch_size = batch_pairs_size * 2; // reads per batch (2 mates)
        for (std::size_t s = 0; s < n_pairs; s += batch_pairs_size) {
            const std::size_t s_reads = 2 * s;
            const std::size_t e_pairs = std::min(s + batch_pairs_size, n_pairs);
            const std::size_t e_reads = 2 * e_pairs;
            batches.push_back({s_reads, e_reads});
        }
    } else {
        batch_size = (n_reads + num_batches - 1) / static_cast<std::size_t>(num_batches);
        for (std::size_t s = 0; s < n_reads; s += batch_size)
            batches.push_back({s, std::min(s + batch_size, n_reads)});
    }
    const int actual_batches = static_cast<int>(batches.size());

    // 5) Open output stream (file or stdout) and write header line.
    std::ofstream out_file;
    std::vector<char> file_buf;
    if (!output_path.empty()) {
        out_file.open(output_path);
        if (!out_file) {
            std::cerr << "Failed to open output: " << output_path << "\n";
            for (faiss::IndexBinary* p : owned_indices) delete p;
            return 1;
        }
        set_large_stream_buffer(out_file, file_buf);
        status_out << "Writing results to " << output_path << "\n";
    }
    std::ostream* out_ptr = output_path.empty() ? &std::cout : &out_file;
    if (output_sam) {
        *out_ptr << "@HD\tVN:1.6\tSO:unsorted\n";
        // In --align mode we already loaded full reference sequences (keyed by meta sequence_name).
        // SAM headers must use normalized reference names.
        for (const auto& kv : sam_sn_to_len) {
            *out_ptr << "@SQ\tSN:" << kv.first << "\tLN:" << kv.second << "\n";
        }
    } else if (do_align && secondary) {
        *out_ptr << "query_name\tquery_length\tstrand\ttarget_name\ttarget_start\tmapping_quality\talignment_score\trank\tsegment_start\tread\n";
    } else if (do_align) {
        *out_ptr << "query_name\tquery_length\tstrand\ttarget_name\ttarget_start\tmapping_quality\talignment_score\tsegment_start\tread\n";
    } else if (secondary) {
        *out_ptr << "query_name\tquery_length\tstrand\ttarget_name\ttarget_start\tmapping_quality\trank\tsegment_start\tread\n";
    } else {
        *out_ptr << "query_name\tquery_length\tstrand\ttarget_name\ttarget_start\tmapping_quality\tsegment_start\tread\n";
    }

    // Shared output state used by all worker threads.
    OutputState output;
    output.out = out_ptr;

    // FAISS search uses OpenMP internally. We already have num_threads workers
    // each calling index->search() on their batch; if each search also used
    // num_threads OMP threads we'd get num_threads^2 threads.
    // To avoid oversubscription, force single-threaded FAISS search per worker;
    // parallelism comes purely from our explicit worker threads.
#ifdef _OPENMP
    omp_set_num_threads(1);
    if (std::getenv("ANNIEMAP_OMP_DEBUG")) {
        std::cerr << "[anniemap_pipeline] main after omp_set_num_threads(1): omp_get_max_threads()="
                  << omp_get_max_threads() << " omp_get_num_procs()=" << omp_get_num_procs() << "\n";
    }
#endif

    std::atomic<int> next_batch{0};
    const std::unordered_map<std::string, std::string>* ref_seqs_ptr = do_align ? &ref_seqs : nullptr;

    auto t_pipeline_start = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(num_threads));
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
#ifdef _OPENMP
            // libomp keeps nthreads-var per native thread; main's omp_set_num_threads(1)
            // does not apply to std::thread workers, so FAISS could still use OMP_NUM_THREADS.
            omp_set_num_threads(1);
            if (std::getenv("ANNIEMAP_OMP_DEBUG")) {
                std::cerr << "[anniemap_pipeline] worker " << t
                          << " omp_get_max_threads()=" << omp_get_max_threads() << "\n";
            }
#endif
            std::vector<std::uint8_t> packed_buf(batch_size * d_bytes);
            std::vector<std::uint8_t> strand_buf(canonical ? batch_size * KMER_CANONICAL_BYTES : 0);
            std::vector<KmerVector> kmer_buf(canonical ? 0 : batch_size);
            std::vector<int32_t> distances_buf(batch_size * max_k);
            std::vector<faiss::idx_t> labels_buf(batch_size * max_k);
            std::vector<std::uint8_t> ref_presence_buf(KMER_CANONICAL_BYTES);
            auto aligner = std::make_unique<wfa::WFAlignerGapAffine>(
                2, 6, 1, wfa::WFAligner::Alignment, wfa::WFAligner::MemoryHigh);

            while (true) {
                int batch_id = next_batch.fetch_add(1);
                if (batch_id >= actual_batches) break;
                std::size_t start = batches[static_cast<std::size_t>(batch_id)].first;
                std::size_t end   = batches[static_cast<std::size_t>(batch_id)].second;
                process_batch(batch_id, start, end, valid, &bundles_by_len, max_length,
                    ref_seqs_ptr, output_sam ? &ref_sam_names : nullptr, canonical, do_align, secondary, output_sam, paired_sam, max_length, d_bytes, max_k,
                    packed_buf, strand_buf, canonical ? nullptr : &kmer_buf, distances_buf, labels_buf, ref_presence_buf,
                    aligner.get(), wfa_banded, wfa_adaptive, wfa_zdrop, output);
            }
            output.try_drain();
        });
    }
    for (auto& w : workers) w.join();

    output.try_drain();

    auto t_pipeline_end = std::chrono::steady_clock::now();
    double sec_pipeline = std::chrono::duration<double>(t_pipeline_end - t_pipeline_start).count();
    double sec_total = std::chrono::duration<double>(t_pipeline_end - t0).count();

    for (faiss::IndexBinary* p : owned_indices) delete p;
    if (out_file.is_open()) out_file.close();

    std::ostream& timing_out = output_sam ? std::cerr : std::cout;
    timing_out << "\n--- Timing (s) ---\n";
    timing_out << "read_fastq+filter:   " << std::chrono::duration<double>(t_pipeline_start - t0).count() << "\n";
    timing_out << "pipeline (batches):  " << sec_pipeline << "\n";
    timing_out << "total:               " << sec_total << "\n";

    return 0;
}
