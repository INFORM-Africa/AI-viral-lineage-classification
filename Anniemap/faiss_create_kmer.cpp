// Build a FAISS binary index over k-mer presence vectors for reference sequences.
// This is a C++ analogue of faiss_create.py but uses binary k-mer vectors
// (from kmer_vector.hpp) instead of FCGR / chaos game representations.
//
// - Input reference FASTA is taken from: ref_sequences/<base_name>.fasta
// - Output index is written to:          ref_vectors/<base_name>_<...>.faiss
// - Metadata TSV is written to:         ref_vectors/<base_name>_<...>_metadata.tsv
//
// Only binary IVF (IndexBinaryIVF) and binary flat (IndexBinaryFlat) are supported.

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "kmer_vector.hpp"

// Metadata for a single sliding window extracted from the reference.
// - sequence_name: FASTA record identifier (header line without the '>' prefix).
// - start_pos:     0-based coordinate within the reference sequence.
// - strand:        '+' for forward, '-' for reverse complement.
//                  In canonical mode this is left as '\0' and strand is stored
//                  separately in a packed bit-vector file.
struct WindowMeta {
    std::string sequence_name;
    int start_pos = 0;
    char strand = '+'; // '+' for forward, '-' for reverse complement (unused in canonical mode)
};

// Simple reverse-complement for DNA (A,C,G,T,N and common ambiguity codes).
static char rc_base(char b) {
    switch (b) {
        case 'A': return 'T';
        case 'C': return 'G';
        case 'G': return 'C';
        case 'T': return 'A';
        case 'a': return 't';
        case 'c': return 'g';
        case 'g': return 'c';
        case 't': return 'a';
        case 'N': case 'n': return 'N';
        default: return 'N';
    }
}

static std::string reverse_complement(const std::string& seq) {
    std::string rc;
    rc.resize(seq.size());
    for (std::size_t i = 0, j = seq.size(); i < seq.size(); ++i) {
        char b = seq[seq.size() - 1 - i];
        rc[i] = rc_base(b);
    }
    return rc;
}

// Read a multi-FASTA file and generate binary k-mer vectors for overlapping windows.
//
// For each record:
//   - We slide a fixed-size window of length window_size over the sequence with stride step_size.
//   - For each window we build:
//       * One vector for the forward sequence.
//       * One vector for the reverse complement of that window.
//   - Both vectors are appended to out_vectors and described in out_meta.
//
// Output:
//   - out_vectors.size() == 2 * (total number of windows across all records).
//   - out_meta[i] describes the window that produced vectors[i].
static bool process_fasta_kmer(
    const std::string& fasta_path,
    int window_size,
    int step_size,
    std::vector<KmerVector>& out_vectors,
    std::vector<WindowMeta>& out_meta
) {
    std::ifstream in(fasta_path);
    if (!in) {
        std::cerr << "Failed to open FASTA file: " << fasta_path << "\n";
        return false;
    }

    std::string line;
    std::string current_name;
    std::string current_seq;

    auto flush_record = [&](void) {
        if (current_name.empty() || current_seq.empty()) {
            return;
        }
        const std::string& seq = current_seq;
        const int len = static_cast<int>(seq.size());
        for (int start = 0; start + window_size <= len; start += step_size) {
            std::string window = seq.substr(start, window_size);

            // Forward strand.
            KmerVector kv_fwd;
            compute_kmer_vector_5(window.data(), window.size(), kv_fwd);
            out_vectors.push_back(kv_fwd);
            out_meta.push_back(WindowMeta{current_name, start, '+'});

            // Reverse complement strand.
            std::string rc = reverse_complement(window);
            KmerVector kv_rc;
            compute_kmer_vector_5(rc.data(), rc.size(), kv_rc);
            out_vectors.push_back(kv_rc);
            out_meta.push_back(WindowMeta{current_name, start, '-'});
        }
    };

    while (std::getline(in, line)) {
        if (!line.empty() && line[0] == '>') {
            // Header line: flush previous record.
            flush_record();
            current_name = line.substr(1);
            current_seq.clear();
        } else {
            // Sequence line: append, stripping whitespace.
            for (char c : line) {
                if (!std::isspace(static_cast<unsigned char>(c))) {
                    current_seq.push_back(c);
                }
            }
        }
    }
    // Flush last record.
    flush_record();

    return true;
}

// Canonical processing:
//   - Each window is converted into a canonical presence/strand pair via
//     compute_kmer_vector_5_canonical.
//   - CanonicalPresence (512 bits) and CanonicalStrand (512 bits) are both packed
//     into big-endian bit-vectors (64 bytes each).
//   - Only the packed presence stream is stored in the FAISS index; the packed
//     strand stream is written to a sidecar binary file so we can recover
//     strand information at query time.
//
// Output invariants:
//   - out_packed_presence.size() == out_packed_strand.size().
//   - out_packed_presence.size() == meta.size() * KMER_CANONICAL_BYTES.
static bool process_fasta_kmer_canonical(
    const std::string& fasta_path,
    int window_size,
    int step_size,
    std::vector<std::uint8_t>& out_packed_presence,  // 64 bytes per vector
    std::vector<std::uint8_t>& out_packed_strand,    // 64 bytes per vector
    std::vector<WindowMeta>& out_meta
) {
    std::ifstream in(fasta_path);
    if (!in) {
        std::cerr << "Failed to open FASTA file: " << fasta_path << "\n";
        return false;
    }

    std::string line;
    std::string current_name;
    std::string current_seq;
    constexpr std::size_t row_bytes = KMER_CANONICAL_BYTES;

    auto flush_record = [&](void) {
        if (current_name.empty() || current_seq.empty()) return;
        const std::string& seq = current_seq;
        const int len = static_cast<int>(seq.size());
        for (int start = 0; start + window_size <= len; start += step_size) {
            std::string window = seq.substr(start, window_size);
            CanonicalPresence pres;
            CanonicalStrand strand;
            compute_kmer_vector_5_canonical(window.data(), window.size(), pres, strand);

            out_packed_presence.resize(out_packed_presence.size() + row_bytes);
            pack_canonical_big_endian(pres, out_packed_presence.data() + out_packed_presence.size() - row_bytes);
            out_packed_strand.resize(out_packed_strand.size() + row_bytes);
            pack_canonical_big_endian(strand, out_packed_strand.data() + out_packed_strand.size() - row_bytes);
            out_meta.push_back(WindowMeta{current_name, start, '\0'});
        }
    };

    while (std::getline(in, line)) {
        if (!line.empty() && line[0] == '>') {
            flush_record();
            current_name = line.substr(1);
            current_seq.clear();
        } else {
            for (char c : line) {
                if (!std::isspace(static_cast<unsigned char>(c)))
                    current_seq.push_back(c);
            }
        }
    }
    flush_record();
    return true;
}

static void ensure_directory(const std::string& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::cerr << "Warning: failed to ensure directory '" << dir
                  << "': " << ec.message() << "\n";
    }
}

struct CmdOptions {
    std::string base_name;     // e.g. "ref_dengue_1"
    int window_size = 50;
    int step_size = 1;
    int k = KMER_K;           // currently must be 5
    int nlist = 0;            // 0 = auto
    int nprobe = 0;           // 0 = auto
    std::string index_type = "binary"; // "binary" or "binary_flat"
    bool canonical = false;   // canonical k-mers (512 dims + external strand metadata)
};

// Parse command-line arguments into CmdOptions.
// Expected usage:
//   faiss_create_kmer <base_name> [--window-size N] [--step-size N]
//                     [--k-mer K] [--nlist N] [--nprobe N]
//                     [--index-type binary|binary_flat] [--canonical]
//
// Where:
//   - base_name determines input FASTA (ref_sequences/base_name.fasta)
//     and output prefix under ref_vectors/.
//   - index-type:
//       * "binary"      -> IVF index (IndexBinaryIVF + IndexBinaryFlat quantizer).
//       * "binary_flat" -> flat Hamming index (IndexBinaryFlat).
//   - nlist/nprobe only apply to "binary" IVF indices.
static bool parse_args(int argc, char** argv, CmdOptions& opt) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <base_name> [--window-size N] [--step-size N]"
                  << " [--k-mer K] [--nlist N] [--nprobe N]"
                  << " [--index-type binary|binary_flat] [--canonical]\n";
        return false;
    }
    opt.base_name = argv[1];

    int i = 2;
    while (i < argc) {
        std::string arg = argv[i];
        if (arg == "--window-size" && i + 1 < argc) {
            opt.window_size = std::stoi(argv[++i]);
        } else if (arg == "--step-size" && i + 1 < argc) {
            opt.step_size = std::stoi(argv[++i]);
        } else if (arg == "--k-mer" && i + 1 < argc) {
            opt.k = std::stoi(argv[++i]);
        } else if (arg == "--nlist" && i + 1 < argc) {
            opt.nlist = std::stoi(argv[++i]);
        } else if (arg == "--nprobe" && i + 1 < argc) {
            opt.nprobe = std::stoi(argv[++i]);
        } else if (arg == "--index-type" && i + 1 < argc) {
            opt.index_type = argv[++i];
        } else if (arg == "--canonical") {
            opt.canonical = true;
        } else {
            std::cerr << "Unknown or malformed option: " << arg << "\n";
            return false;
        }
        ++i;
    }

    if (opt.index_type != "binary" && opt.index_type != "binary_flat") {
        std::cerr << "index-type must be 'binary' or 'binary_flat'\n";
        return false;
    }

    if (opt.k != KMER_K) {
        std::cerr << "Currently only k=" << KMER_K
                  << " is supported (got k=" << opt.k << ")\n";
        return false;
    }

    return true;
}

int main(int argc, char** argv) {
    CmdOptions opt;
    if (!parse_args(argc, argv, opt)) {
        return 1;
    }

    ensure_directory("ref_vectors");

    const std::string fasta_path = "ref_sequences/" + opt.base_name + ".fasta";

    std::string suffix;
    {
        std::ostringstream oss;
        oss << "_" << opt.window_size << "_step" << opt.step_size;
        if (opt.index_type == "binary" && opt.nlist > 0 && opt.nprobe > 0) {
            oss << "_nlist" << opt.nlist << "_nprobe" << opt.nprobe;
        }
        if (opt.canonical) {
            oss << "_canonical";
        }
        oss << "_" << opt.index_type;
        suffix = oss.str();
    }

    const std::string index_file = "ref_vectors/" + opt.base_name + suffix + ".faiss";
    const std::string metadata_file = "ref_vectors/" + opt.base_name + suffix + "_metadata.tsv";
    const std::string strand_file = "ref_vectors/" + opt.base_name + suffix + "_strand.bin";

    std::cout << "Building FAISS " << opt.index_type
              << " index from " << fasta_path
              << " using k-mer k=" << KMER_K
              << (opt.canonical ? " (canonical, dim=512 bits)" : " (dim=" + std::to_string(KMER_DIM) + " bits)")
              << "..." << std::endl;

    auto t0 = std::chrono::steady_clock::now();

    std::size_t nb = 0;
    std::size_t d_bits = 0;
    std::size_t d_bytes = 0;
    std::vector<std::uint8_t> packed;
    std::vector<WindowMeta> meta;

    if (opt.canonical) {
        std::vector<std::uint8_t> packed_presence, packed_strand;
        if (!process_fasta_kmer_canonical(
                fasta_path, opt.window_size, opt.step_size,
                packed_presence, packed_strand, meta)) {
            return 1;
        }
        nb = meta.size();
        if (nb == 0) {
            std::cerr << "No windows generated from FASTA; nothing to index.\n";
            return 1;
        }
        d_bits = KMER_DIM_CANONICAL;
        d_bytes = KMER_CANONICAL_BYTES;
        packed = std::move(packed_presence);

        std::cout << "Vector dimension (bits): " << d_bits
                  << ", packed to " << d_bytes << " bytes per vector"
                  << ", total vectors: " << nb << " (canonical)\n";

        if (opt.index_type == "binary_flat") {
            std::cout << "Using IndexBinaryFlat (exhaustive Hamming search)\n";
            faiss::IndexBinaryFlat index(static_cast<faiss::idx_t>(d_bits));
            index.add(static_cast<faiss::idx_t>(nb), packed.data());
            faiss::write_index_binary(&index, index_file.c_str());
            std::cout << "Index saved to " << index_file << "\n";
        } else {
            int nlist = opt.nlist;
            if (nlist <= 0) {
                nlist = static_cast<int>(std::max<std::size_t>(
                    1, std::min<std::size_t>(256, static_cast<std::size_t>(std::sqrt(nb)))));
            }
            std::cout << "Using " << nlist << " clusters for IVF binary index (Hamming distance)\n";
            auto quantizer = std::make_unique<faiss::IndexBinaryFlat>(static_cast<faiss::idx_t>(d_bits));
            faiss::IndexBinaryIVF index(quantizer.get(), static_cast<faiss::idx_t>(d_bits), nlist);
            if (!index.is_trained && nb > static_cast<std::size_t>(nlist)) {
                std::cout << "Training the index...\n";
                index.train(static_cast<faiss::idx_t>(nb), packed.data());
            }
            index.add(static_cast<faiss::idx_t>(nb), packed.data());
            int nprobe = opt.nprobe <= 0 ? std::max(1, nlist / 32) : opt.nprobe;
            index.nprobe = nprobe;
            std::cout << "Using nprobe=" << nprobe << " for search\n";
            index.make_direct_map(true);  // enable reconstruct() for strand recovery in anniemap
            faiss::write_index_binary(&index, index_file.c_str());
            std::cout << "Index saved to " << index_file << "\n";
        }

        std::ofstream strand_out(strand_file, std::ios::binary);
        if (!strand_out) {
            std::cerr << "Warning: failed to open strand file for writing: " << strand_file << "\n";
        } else {
            strand_out.write(reinterpret_cast<const char*>(packed_strand.data()),
                             static_cast<std::streamsize>(packed_strand.size()));
            std::cout << "Strand metadata saved to " << strand_file << " (64 bytes per vector)\n";
        }

        std::ofstream meta_out(metadata_file);
        if (!meta_out) {
            std::cerr << "Warning: failed to open metadata file for writing: " << metadata_file << "\n";
        } else {
            meta_out << "sequence_name\tstart_pos\n";
            for (const auto& m : meta) {
                meta_out << m.sequence_name << '\t' << m.start_pos << '\n';
            }
            std::cout << "Metadata TSV saved to " << metadata_file << "\n";
        }
    } else {
        std::vector<KmerVector> vectors;
        if (!process_fasta_kmer(fasta_path, opt.window_size, opt.step_size, vectors, meta)) {
            return 1;
        }
        if (vectors.empty()) {
            std::cerr << "No windows generated from FASTA; nothing to index.\n";
            return 1;
        }
        nb = vectors.size();
        d_bits = KMER_DIM;
        d_bytes = d_bits / 8;
        packed.resize(nb * d_bytes);
        for (std::size_t i = 0; i < nb; ++i) {
            pack_kmer_vector_big_endian(vectors[i], packed.data() + i * d_bytes);
        }

        std::cout << "Vector dimension (bits): " << d_bits
                  << ", packed to " << d_bytes << " bytes per vector"
                  << ", total vectors: " << nb << "\n";

        if (opt.index_type == "binary_flat") {
            std::cout << "Using IndexBinaryFlat (exhaustive Hamming search)\n";
            faiss::IndexBinaryFlat index(static_cast<faiss::idx_t>(d_bits));
            index.add(static_cast<faiss::idx_t>(nb), packed.data());
            faiss::write_index_binary(&index, index_file.c_str());
            std::cout << "Index saved to " << index_file << "\n";
        } else {
            int nlist = opt.nlist;
            if (nlist <= 0) {
                nlist = static_cast<int>(std::max<std::size_t>(
                    1, std::min<std::size_t>(256, static_cast<std::size_t>(std::sqrt(nb)))));
            }
            std::cout << "Using " << nlist << " clusters for IVF binary index (Hamming distance)\n";
            auto quantizer = std::make_unique<faiss::IndexBinaryFlat>(static_cast<faiss::idx_t>(d_bits));
            faiss::IndexBinaryIVF index(quantizer.get(), static_cast<faiss::idx_t>(d_bits), nlist);
            if (!index.is_trained && nb > static_cast<std::size_t>(nlist)) {
                std::cout << "Training the index...\n";
                index.train(static_cast<faiss::idx_t>(nb), packed.data());
            }
            index.add(static_cast<faiss::idx_t>(nb), packed.data());
            int nprobe = opt.nprobe <= 0 ? std::max(1, nlist / 32) : opt.nprobe;
            index.nprobe = nprobe;
            std::cout << "Using nprobe=" << nprobe << " for search\n";
            faiss::write_index_binary(&index, index_file.c_str());
            std::cout << "Index saved to " << index_file << "\n";
        }

        std::ofstream meta_out(metadata_file);
        if (!meta_out) {
            std::cerr << "Warning: failed to open metadata file for writing: " << metadata_file << "\n";
        } else {
            meta_out << "sequence_name\tstart_pos\tstrand\n";
            for (const auto& m : meta) {
                meta_out << m.sequence_name << '\t' << m.start_pos << '\t' << m.strand << '\n';
            }
            std::cout << "Metadata TSV saved to " << metadata_file << "\n";
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    std::cout << "\nIndex statistics:\n";
    std::cout << "Index type: " << opt.index_type << (opt.canonical ? " (canonical)" : "") << "\n";
    std::cout << "Number of vectors: " << nb << "\n";
    std::cout << "Window size: " << opt.window_size << "\n";
    std::cout << "Step size: " << opt.step_size << "\n";
    std::cout << "k (k-mer size): " << KMER_K << "\n";
    if (opt.canonical) {
        std::cout << "Canonical: 512 dimensions, strand stored in " << strand_file << "\n";
    }
    if (opt.index_type == "binary_flat") {
        std::cout << "IndexBinaryFlat: exhaustive Hamming search (no nlist/nprobe)\n";
    } else {
        std::cout << "nlist: " << (opt.nlist > 0 ? opt.nlist : -1)
                  << " (actual used: computed above)\n";
        std::cout << "nprobe: " << (opt.nprobe > 0 ? opt.nprobe : -1)
                  << " (actual used: see log)\n";
    }
    std::cout << "Build time: " << dt.count() << " seconds\n";

    return 0;
}

