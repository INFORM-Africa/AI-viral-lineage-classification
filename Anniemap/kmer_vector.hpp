// Shared k-mer vector utilities for Anniemap.
// - Fixed k = 5 (1024 possible 5-mers).
// - Representation is binary presence (0/1) for each possible k-mer, not counts.
// - Canonical mode collapses a k-mer with its reverse complement into a single bin
//   (512 canonical bins) and stores an additional binary strand flag per bin.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

// Fixed 5-mer configuration: 4^5 = 1024 dimensions.
// For now, k is compile-time fixed. Treat this as "k".
static constexpr int KMER_K = 5;
static constexpr std::size_t KMER_DIM = 1u << (2 * KMER_K); // 4^K

// Canonical: min(kmer, revcomp(kmer)) -> 512 bins (half size).
static constexpr std::size_t KMER_DIM_CANONICAL = KMER_DIM / 2u; // 512

// Binary k-mer presence vector: 0/1 for each possible k-mer.
using KmerVector = std::array<std::uint8_t, KMER_DIM>;

// Canonical: one byte per canonical bin (0/1 presence, 0/1 strand on first occurrence).
using CanonicalPresence = std::array<std::uint8_t, KMER_DIM_CANONICAL>;
using CanonicalStrand = std::array<std::uint8_t, KMER_DIM_CANONICAL>;

// Map nucleotide to 2-bit code (A,C,G,T -> 0..3).
// Returns -1 for any non-ACGT base so the caller can reset the rolling window.
inline int nt4(char c) noexcept {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}

// Reverse complement of a 5-mer encoded as 10 bits (2 bits per base).
// Encoding convention:
//   - The "leftmost" base in the k-mer occupies the most-significant 2 bits.
//   - Complement is implemented as XOR with 3 (0<->3, 1<->2) on each 2-bit base.
inline std::uint32_t reverse_complement_5mer(std::uint32_t code) noexcept {
    constexpr std::uint32_t MASK = static_cast<std::uint32_t>(KMER_DIM - 1);
    code &= MASK;
    std::uint32_t rc = 0;
    for (int i = 0; i < KMER_K; ++i) {
        std::uint32_t base = (code >> (2 * i)) & 3u;
        rc = (rc << 2) | (base ^ 3u);  // complement: 0<->3, 1<->2
    }
    return rc;
}

// Canonical bin index for each of the 1024 raw 5-mer codes.
// For a pair (k-mer, reverse-complement) we assign a single canonical bin id
// in [0, 511] and store it in both entries. This is precomputed once, lazily.
// Thread-safety:
//   - Constructed via std::call_once so all threads observe the same mapping.
inline const std::array<std::int32_t, KMER_DIM>& get_canonical_map() {
    static std::array<std::int32_t, KMER_DIM> canonical_map;
    static std::once_flag once;
    std::call_once(once, []() {
        for (std::size_t i = 0; i < KMER_DIM; ++i) {
            canonical_map[i] = -1;
        }
        int next_idx = 0;
        for (std::size_t i = 0; i < KMER_DIM; ++i) {
            if (canonical_map[i] != -1) continue;
            std::uint32_t rc = reverse_complement_5mer(static_cast<std::uint32_t>(i));
            std::size_t r = static_cast<std::size_t>(rc);
            canonical_map[i] = next_idx;
            canonical_map[r] = next_idx;
            ++next_idx;
        }
    });
    return canonical_map;
}

// Compute 5-mer binary presence vector for a single read.
// Semantics:
//   - out_vec[i] == 1 iff the i-th 5-mer (in lexical ACGT order) appears at
//     least once in the sequence; no frequency information is retained.
//   - Non-ACGT characters break the rolling window and are skipped.
inline void compute_kmer_vector_5(const char* seq, std::size_t len, KmerVector& out_vec) noexcept {
    out_vec.fill(0);
    if (len < static_cast<std::size_t>(KMER_K)) {
        return;
    }

    constexpr std::uint32_t MASK = static_cast<std::uint32_t>(KMER_DIM - 1); // keep last 2*K bits
    std::uint32_t val = 0;
    int run = 0; // number of consecutive valid bases seen

    for (std::size_t i = 0; i < len; ++i) {
        int code = nt4(seq[i]);
        if (code < 0) {
            // Break k-mer when encountering non-ACGT.
            run = 0;
            val = 0;
            continue;
        }

        val = ((val << 2) | static_cast<std::uint32_t>(code)) & MASK;
        if (run < KMER_K) {
            ++run;
            if (run < KMER_K) {
                continue; // haven't formed a full k-mer yet
            }
        }
        out_vec[val] = 1;  // binary: mark presence, no counting
    }
}

// Canonical 5-mer representation.
// For each position where a valid 5-mer can be formed:
//   - We compute the raw code and its reverse complement code.
//   - Both codes share a single canonical bin, given by get_canonical_map().
//   - out_presence[bin] is set to 1 the first time this canonical bin is seen.
//   - out_strand[bin] encodes the strand of that first occurrence:
//       0 = forward (k-mer <= revcomp(k-mer))
//       1 = reverse (k-mer >  revcomp(k-mer)).
// Subsequent occurrences of the same canonical bin do not change the strand.
inline void compute_kmer_vector_5_canonical(
    const char* seq,
    std::size_t len,
    CanonicalPresence& out_presence,
    CanonicalStrand& out_strand
) noexcept {
    out_presence.fill(0);
    out_strand.fill(0);
    if (len < static_cast<std::size_t>(KMER_K)) {
        return;
    }

    const std::array<std::int32_t, KMER_DIM>& canonical_map = get_canonical_map();
    constexpr std::uint32_t MASK = static_cast<std::uint32_t>(KMER_DIM - 1);
    std::uint32_t val = 0;
    int run = 0;

    for (std::size_t i = 0; i < len; ++i) {
        int code = nt4(seq[i]);
        if (code < 0) {
            run = 0;
            val = 0;
            continue;
        }

        val = ((val << 2) | static_cast<std::uint32_t>(code)) & MASK;
        if (run < KMER_K) {
            ++run;
            if (run < KMER_K) continue;
        }

        std::uint32_t rc = reverse_complement_5mer(val);
        int bin = canonical_map[val];
        if (out_presence[static_cast<std::size_t>(bin)] == 0) {
            out_presence[static_cast<std::size_t>(bin)] = 1;
            out_strand[static_cast<std::size_t>(bin)] = (val > rc) ? 1 : 0;
        }
    }
}

// Pack binary k-mer vector into bytes (big-endian, same layout as numpy.packbits).
// Contract:
//   - v[0] becomes the MSB of out_bytes[0]; v[7] becomes its LSB, etc.
//   - Caller must provide at least KMER_DIM/8 bytes in out_bytes.
inline void pack_kmer_vector_big_endian(const KmerVector& v, std::uint8_t* out_bytes) {
    const std::size_t d_bits = KMER_DIM;
    const std::size_t d_bytes = d_bits / 8;
    for (std::size_t i = 0; i < d_bytes; ++i) {
        out_bytes[i] = 0;
    }
    for (std::size_t bit = 0; bit < d_bits; ++bit) {
        if (!v[bit]) continue;
        std::size_t byte_idx = bit / 8;
        std::size_t bit_in_byte = bit % 8;
        out_bytes[byte_idx] |= static_cast<std::uint8_t>(1u << (7 - bit_in_byte));
    }
}

// Pack canonical presence or strand (512 bits) into 64 bytes, big-endian.
// Layout matches pack_kmer_vector_big_endian and the Python/NumPy side.
static constexpr std::size_t KMER_CANONICAL_BYTES = KMER_DIM_CANONICAL / 8;

inline void pack_canonical_big_endian(
    const std::array<std::uint8_t, KMER_DIM_CANONICAL>& v,
    std::uint8_t* out_bytes
) {
    for (std::size_t i = 0; i < KMER_CANONICAL_BYTES; ++i) {
        out_bytes[i] = 0;
    }
    for (std::size_t bit = 0; bit < KMER_DIM_CANONICAL; ++bit) {
        if (!v[bit]) continue;
        std::size_t byte_idx = bit / 8;
        std::size_t bit_in_byte = bit % 8;
        out_bytes[byte_idx] |= static_cast<std::uint8_t>(1u << (7 - bit_in_byte));
    }
}

