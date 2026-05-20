# Anniemap

Anniemap maps short nucleotide reads to a reference by **k-mer presence** similarity search with [FAISS](https://github.com/facebookresearch/faiss), optionally followed by **ends-free alignment** with [WFA2](https://github.com/smarco/WFA2-lib). It is designed for viral or compact references where exhaustive seed-and-extend aligners are heavy, and where a fixed **k = 5** binary fingerprint (1024 dimensions, or 512 in canonical mode) is a practical similarity signal.

The mapper loads a pre-built FAISS index over sliding windows of the reference, searches each read (or a center-cropped segment of it) for the nearest window(s) under **Hamming distance**, and writes **TSV** or **SAM** output. Alignment is optional but required for SAM.

---

## Table of contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Preparing reference data](#preparing-reference-data)
5. [Building an index](#building-an-index)
6. [Running anniemap](#running-anniemap)
7. [Command-line options](#command-line-options)
8. [Output formats](#output-formats)
9. [Index and file naming](#index-and-file-naming)
10. [Sample data (Dengue 1)](#sample-data-dengue-1)
11. [Examples](#examples)
12. [Performance notes](#performance-notes)
13. [Troubleshooting](#troubleshooting)

---

## Overview

**Indexing (`faiss_create_kmer`)** slides a fixed-size window along each reference sequence (forward and reverse complement, or canonical k-mers), encodes each window as a binary presence vector over all 5-mers, and stores vectors in a FAISS binary index (`IndexBinaryIVF` by default, or `IndexBinaryFlat`).

**Mapping (`anniemap`)** for each read:

1. Filters and optionally center-crops to a search length.
2. Builds the same k-mer vector as the index.
3. Queries FAISS for the nearest reference window(s).
4. Optionally runs WFA2 ends-free alignment on a padded local reference segment and reports an alignment score.
5. Writes ordered TSV or SAM records using a multi-threaded batch pipeline.

Parameters used at index time (`--window-size`, `--step-size`, `--canonical`, IVF `nlist`/`nprobe` when embedded in the filename) must match at mapping time.

---

## Requirements

| Component | Purpose |
|-----------|---------|
| **C++17 compiler** (`g++` or `clang++`) | Build all binaries |
| **zlib** | Gzip-compressed FASTQ (`*.fq.gz`, `*.fastq.gz`) |
| **CMake** | Build FAISS from source |
| **OpenMP** | macOS: `brew install libomp`; Linux: usually via compiler |
| **FAISS** (C++ library) | Source build (`./build_faiss.sh`) or conda/pip (`faiss-cpu`) |
| **WFA2-lib** | Linked into `anniemap` (build before `anniemap`) |
| **kseq.h** | Bundled in the repository root |

The repository includes a `faiss/` tree and `WFA2-lib/`. If you clone with submodules, initialize them; otherwise ensure both directories are present before building.

---

## Installation

All commands below assume the repository root as the working directory.

### 1. Clone and enter the project

```bash
git clone <repository-url> Anniemap
cd Anniemap
# If using submodules:
# git submodule update --init --recursive
```

### 2. Build FAISS (recommended: from source)

```bash
./build_faiss.sh
```

On **macOS**, install OpenMP first:

```bash
brew install libomp cmake
```

On **Linux**, install `cmake`, `g++`, and OpenMP development packages for your distribution.

`build_faiss.sh` writes shared libraries under `faiss/build/faiss/` (`libfaiss_avx2` on Intel x86_64, `libfaiss` on arm64).

**Alternative:** use FAISS from conda or pip when building only `anniemap` (not `faiss_create_kmer`):

```bash
conda install -c pytorch faiss-cpu
# or: pip install faiss-cpu
USE_PYTHON_FAISS=1 ./build_anniemap.sh
```

### 3. Build WFA2

```bash
cd WFA2-lib && make all && cd ..
```

This produces `WFA2-lib/lib/libwfacpp.a`, required by `build_anniemap.sh`.

### 4. Build index tool and mapper

```bash
./build_faiss_create_kmer.sh   # -> ./faiss_create_kmer
./build_anniemap.sh            # -> ./anniemap
```

`build_anniemap.sh` links WFA2 even if you only plan k-mer mapping without `--align`; the binary is always built with alignment support.

### 5. Verify

```bash
./faiss_create_kmer
./anniemap
```

Both should print usage and exit with status 1.

---

## Preparing reference data

Place reference FASTA here:

```text
ref_sequences/<ref_name>.fasta
```

- `<ref_name>` is the logical name you pass to indexing and mapping (no path, no `.fasta`).
- Headers are the text after `>` on each record; they are stored verbatim in index metadata and must match keys in the FASTA used for `--align`.
- Sequences are uppercased internally; whitespace on sequence lines is stripped.

Example:

```text
ref_sequences/viral_panel.fasta
```

---

## Building an index

Use `faiss_create_kmer`:

```text
faiss_create_kmer <ref_name> [options]
```

### `faiss_create_kmer` options

| Option | Argument | Default | Description |
|--------|----------|---------|-------------|
| `<ref_name>` | — | (required) | Base name; reads `ref_sequences/<ref_name>.fasta` |
| `--window-size` | *N* | `50` | Sliding window length (bp) on the reference |
| `--step-size` | *N* | `1` | Stride between window start positions (bp) |
| `--k-mer` | *K* | `5` | K-mer size (only **5** is supported) |
| `--nlist` | *N* | auto | IVF cluster count; `0` = `min(256, sqrt(n_vectors))` |
| `--nprobe` | *N* | auto | IVF probes at build time; `0` = `max(1, nlist/32)` |
| `--index-type` | `binary` \| `binary_flat` | `binary` | `binary` = IVF; `binary_flat` = exhaustive Hamming |
| `--canonical` | — | off | 512-dim canonical k-mers + separate strand file |

**Index type note:** `anniemap` loads indices whose filenames end with `_binary` (IVF). Indexes built with `--index-type binary_flat` use the suffix `_binary_flat` and are **not** loaded by the current mapper. Use the default IVF index for mapping.

### What gets written

For each build, files are written under `ref_vectors/`:

| File | Description |
|------|-------------|
| `<ref_name>_<L>_step<S>[_nlist<N>_nprobe<P>][_canonical]_<type>.faiss` | FAISS binary index |
| `..._metadata.tsv` | Per-vector `sequence_name`, `start_pos` [, `strand`] |
| `..._strand.bin` | Canonical mode only: packed strand bits per vector |

Non-canonical indexing stores **two vectors per window** (forward and reverse complement). Canonical mode stores one vector per window plus strand metadata.

### Index examples

Default 50 bp windows, step 1, IVF:

```bash
./faiss_create_kmer viral_panel --window-size 50 --step-size 1
```

100 bp windows for reads mapped with `--max-length 100`:

```bash
./faiss_create_kmer viral_panel --window-size 100 --step-size 1
```

Canonical k-mers:

```bash
./faiss_create_kmer viral_panel --window-size 100 --step-size 1 --canonical
```

Explicit IVF shape (suffix must match at mapping time):

```bash
./faiss_create_kmer viral_panel --window-size 100 --step-size 1 --nlist 256 --nprobe 8
```

**Variable read lengths** (`--variable` in anniemap): build one index per bucket — window sizes **50, 75, 100, and 150** — with the same `<ref_name>`, `--step-size`, and flags:

```bash
for L in 50 75 100 150; do
  ./faiss_create_kmer viral_panel --window-size $L --step-size 1
done
```

---

## Running anniemap

### Synopsis

```text
anniemap <fastq_r1> <ref_name> [fastq_r2] [options]
```

| Positional | Description |
|------------|-------------|
| `fastq_r1` | Read 1 FASTQ (gzip supported) |
| `ref_name` | Index base name (same as for `faiss_create_kmer`) |
| `fastq_r2` | Optional read 2 for paired FASTQ |

Run from the project root (or any directory where `ref_vectors/` and `ref_sequences/` resolve correctly).

### Minimal examples

K-mer mapping only, 100 bp center crop, default index:

```bash
./anniemap reads_R1.fastq.gz viral_panel \
  --max-length 100 --step-size 1 --threads 8 \
  --output results.tsv
```

K-mer mapping + WFA2 alignment:

```bash
./anniemap reads_R1.fastq.gz viral_panel \
  --max-length 100 --step-size 1 --threads 8 \
  --align --output results.tsv
```

Paired-end SAM (mates must be in sync or name-pairable; see [Paired-end reads](#paired-end-reads)):

```bash
./anniemap reads_R1.fastq.gz viral_panel reads_R2.fastq.gz \
  --max-length 100 --step-size 1 --threads 8 \
  --align --sam --output results.sam
```

---

## Command-line options

Options may appear in any order after positional arguments. Unknown flags cause a non-zero exit.

### Read processing

#### `--max-length N`

**Default:** `100`

Keep reads whose length is **≥ N**. If the read is longer than `N`, a **center crop** of length `N` is used for k-mer search (and stored as the search slice). With `--align`, WFA2 uses the **full read** when cropping occurred (`full_seq` / `full_qual` are retained).

Must match the index `--window-size` used at build time for that run (unless using `--variable`).

#### `--variable`

**Default:** off

Instead of a single `--max-length`, bucket each read by full length:

| Read length | Search length |
|-------------|---------------|
| < 50 | discarded |
| 50–74 | 50 |
| 75–99 | 75 |
| 100–149 | 100 |
| ≥ 150 | 150 |

Requires a separate index for each active bucket (see [Building an index](#building-an-index)). Appends `_variable` to the auto-generated output filename suffix when applicable.

#### `--step-size N`

**Default:** `1`

Must match the `--step-size` used when building the index (part of the index filename). Controls which reference windows exist in the index, not read k-mer stride (reads always use all valid 5-mers in the search slice).

### Index selection

#### `--canonical`

**Default:** off

Load canonical indexes (`_canonical` in the filename) and strand sidecar (`_strand.bin`). Strand for each hit is inferred by comparing read and reference strand bit-vectors over overlapping k-mer bins.

Must match index build; append `_canonical` to output suffix when auto-naming.

#### `--nlist N` and `--nprobe P`

**Default:** `0` (omit suffix)

Used only to **locate** the index file:

```text
ref_vectors/<ref_name>_<search_len>_step<step>_nlist<N>_nprobe<P>[_canonical]_binary.faiss
```

Both must be **> 0** and must match values used at index build **if** that build embedded them in the filename. The probe count used at search time is the value stored inside the index file at build time; these flags do not re-tune a loaded IVF index at runtime.

If the index was built with automatic `nlist`/`nprobe`, omit both flags when mapping.

### Search and reporting

#### `--secondary`

**Default:** off

Request **two** reference hits per read when possible:

1. **Primary:** best Hamming distance (rank 1).
2. **Secondary:** among other hits, the lowest distance among windows whose start is at least **`--max-length`** bp away from the primary start on the same coordinate system (`|sec_start - pri_start| >= max_length`).

`k` for FAISS search is set to `min(search_len/2, ntotal)` (at least 2). If the index is too small, secondary mode is disabled automatically.

In **TSV** mode, two lines are emitted per read (ranks 1 and 2) when a valid secondary exists.

In **`--sam`** mode with `--secondary`, primary and secondary are both aligned; **one** SAM record is emitted per read, choosing the hit with the higher WFA2 score (tie → primary).

#### `--align`

**Default:** off

Load `ref_sequences/<ref_name>.fasta` and run **WFA2 ends-free** alignment on a padded local reference around each hit. Adds `alignment_score` to TSV and enables accurate SAM `POS`/`CIGAR` adjustment.

**Required** for `--sam`.

Gap scoring (fixed): opening **2**, extension **6**, mismatch **1**; alignment mode ends-free; memory profile high.

#### `--sam`

**Default:** off (TSV output)

Write SAM v1.6 (`@HD`, `@SQ`, then records). Requires `--align`.

- Mapped records include optional **`AS:i:`** WFA2 score after **QUAL** when `--align` is set.
- **POS** and **CIGAR** are adjusted to remove padding artifacts from the local reference window.
- Reference names in `@SQ` and `RNAME` are normalized: first token of the FASTA header, cut at whitespace or `|`.

**Paired SAM** (`fastq_r2` + `--sam`): one record per mate; flags for read1/read2, mate info, and **TLEN** when both mates map to the same reference sequence. Read names ending in `/1` or `/2` are stripped for pairing.

#### `--output PATH`

**Default:** auto

If omitted, writes `anniemap_faiss_step<S>...tsv` in the current directory, or a path under a configured mirror tree when input paths match an internal layout. Always set explicitly for portable pipelines:

```bash
--output /path/to/out.tsv
# or
--output /path/to/out.sam
```

Status and timing go to **stdout** for TSV, **stderr** for SAM.

### Parallelism

#### `--threads N`

**Default:** `4`

Worker threads for the batch pipeline (k-mer → FAISS → align → format). FAISS OpenMP is forced to **1 thread per worker** to avoid oversubscription; use this flag for parallelism, not `OMP_NUM_THREADS`.

### WFA2 heuristics (`--align` only)

Numeric parameters are **hard-coded**; flags only enable strategies. Combinations are allowed.

| Flag | Effect (fixed parameters) |
|------|---------------------------|
| `--banded` | Static band, k ∈ [−80, 80] |
| `--adaptive` | WF-adaptive: min WF length 10, max distance 50, steps 1 |
| `--zdrop` | Z-drop: 100, steps 1 |

If none are set, alignment uses exact WFA2 (slower, more accurate).

Applying `--banded`, `--adaptive`, or `--zdrop` without `--align` is an error.

### Environment variables (optional)

| Variable | Effect |
|----------|--------|
| `OMP_DISPLAY_ENV` | Print OpenMP environment at startup |
| `ANNIEMAP_OMP_DEBUG` | Log OpenMP thread counts per worker |
| `CXX` | Compiler for build scripts |

---

## Output formats

### TSV (default)

Header depends on options:

| Mode | Columns |
|------|---------|
| k-mer only | `query_name`, `query_length`, `strand`, `target_name`, `target_start`, `mapping_quality`, `segment_start`, `read` |
| + `--align` | … + `alignment_score` before `segment_start` |
| + `--secondary` | … + `rank` (1 or 2) |

Field meanings:

- **`mapping_quality`:** FAISS **Hamming distance** to the matched reference window (lower is better). This is not a Phred MAPQ.
- **`target_start`:** 0-based start on the forward reference sequence; adjusted to full-read coordinates when the read was center-cropped.
- **`strand`:** `+` or `-`.
- **`segment_start`:** always `0` in the current implementation.
- **`read`:** query sequence used for alignment (full length if cropped for search).
- **`alignment_score`:** WFA2 score when `--align` is set.

Tab, newline, and backslash in names/sequences are escaped for one-line records.

### SAM (`--sam`)

Standard SAM columns; unsorted. Unmapped reads appear with `RNAME=*`, `POS=0`, `CIGAR=*`. Secondary alignments in non-paired TSV-style paths use flag **256**; paired `--sam --secondary` collapses to one record per read as described above.

---

## Index and file naming

### Directory layout

```text
ref_sequences/<ref_name>.fasta     # reference for --align and indexing input
ref_vectors/<ref_name>_<L>_step<S>[_nlist<N>_nprobe<P>][_canonical]_binary.faiss
ref_vectors/<ref_name>_..._metadata.tsv
ref_vectors/<ref_name>_..._strand.bin   # canonical only
```

### Matching mapping to index

At runtime, for each distinct search length present in the filtered reads, anniemap loads:

```text
ref_vectors/<ref_name>_<search_len>_step<step>[_nlist<N>_nprobe<P>][_canonical]_binary.faiss
```

Checklist:

| Setting | Index build | Mapping |
|---------|-------------|---------|
| Window / read length | `--window-size L` | `--max-length L` (or `--variable` buckets) |
| Stride | `--step-size S` | `--step-size S` |
| Canonical | `--canonical` | `--canonical` |
| IVF suffix | `--nlist N --nprobe P` (both > 0) | same `--nlist` and `--nprobe` |
| Index type | `binary` (default) | (implicit `_binary` suffix) |

### Paired-end reads

| Output | Behavior |
|--------|----------|
| **TSV**, two FASTQs | All R1 records then all R2 records are processed as independent single-end reads (no mate pairing). |
| **SAM**, two FASTQs + `--align` | True paired-end: mates adjacent, `RNEXT`/`PNEXT`/`TLEN`, QNAME normalized. |

For paired libraries, use **`--sam --align`** when mate information matters.

---

## Sample data (Dengue 1)

The repository includes paired 50 bp reads under `sample_data/`:

| File | Description |
|------|-------------|
| `Dengue_1_50bp_R1.fq` | Read 1 |
| `Dengue_1_50bp_R2.fq` | Read 2 |

### Index for `ref_dengue_1` (50 bp, canonical, IVF)

Reference: `ref_sequences/ref_dengue_1.fasta`. Build an index that matches the mapping settings below:

```bash
./faiss_create_kmer ref_dengue_1 \
  --window-size 50 \
  --step-size 1 \
  --nlist 512 \
  --nprobe 13 \
  --canonical
```

Expected index files:

```text
ref_vectors/ref_dengue_1_50_step1_nlist512_nprobe13_canonical_binary.faiss
ref_vectors/ref_dengue_1_50_step1_nlist512_nprobe13_canonical_binary_metadata.tsv
ref_vectors/ref_dengue_1_50_step1_nlist512_nprobe13_canonical_binary_strand.bin
```

### Map the sample reads

From the repository root, with **`--max-length 50`**, **`--canonical`**, **`--nlist 512`**, **`--nprobe 13`**, and **no** `--variable`:

```bash
./anniemap \
  sample_data/Dengue_1_50bp_R1.fq \
  ref_dengue_1 \
  sample_data/Dengue_1_50bp_R2.fq \
  --max-length 50 \
  --step-size 1 \
  --canonical \
  --nlist 512 \
  --nprobe 13 \
  --threads 4 \
  --output sample_data/dengue_1_50bp_mapped.tsv
```

K-mer-only mapping (faster, no reference FASTA alignment pass):

```bash
./anniemap \
  sample_data/Dengue_1_50bp_R1.fq \
  ref_dengue_1 \
  sample_data/Dengue_1_50bp_R2.fq \
  --max-length 50 \
  --step-size 1 \
  --canonical \
  --nlist 512 \
  --nprobe 13 \
  --threads 4 \
  --output sample_data/dengue_1_50bp_mapped_kmer.tsv
```

Add `--align` (and optionally `--secondary` or `--sam`) when you need WFA2 alignment scores or SAM output; that requires `ref_sequences/ref_dengue_1.fasta` as documented in [Command-line options](#command-line-options).

---

## Examples

### End-to-end workflow

```bash
# 1. Reference
cp my_virus.fasta ref_sequences/outbreak_2024.fasta

# 2. Index (100 bp windows)
./faiss_create_kmer outbreak_2024 --window-size 100 --step-size 1

# 3. Map
./anniemap sample_R1.fastq.gz outbreak_2024 \
  --max-length 100 --step-size 1 --threads 16 \
  --align --output sample_mapped.tsv
```

### Canonical index + secondary hits

```bash
./faiss_create_kmer outbreak_2024 --window-size 100 --step-size 1 --canonical

./anniemap sample_R1.fastq.gz outbreak_2024 \
  --max-length 100 --step-size 1 --canonical \
  --secondary --align --output sample_canonical.tsv
```

### SAM with heuristics

```bash
./anniemap sample_R1.fastq.gz outbreak_2024 sample_R2.fastq.gz \
  --max-length 100 --step-size 1 --threads 16 \
  --align --sam --banded --adaptive \
  --output sample.sam
```

---

## Performance notes

- Build FAISS with `./build_faiss.sh` on the same machine class you use for production mapping.
- Increase `--threads` until CPU or disk saturates; typical starting point: number of physical cores.
- IVF indexes trade accuracy for speed; increase `--nprobe` at **index** build time for higher recall (and larger index files if `nlist` grows).
- `--align` dominates runtime on long reads; use heuristics (`--banded`, `--adaptive`, `--zdrop`) for throughput.
- Use gzip FASTQ; I/O is streamed with zlib + kseq.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|----------------|-----|
| `Failed to load FAISS index` | Wrong suffix or missing index for read length | Rebuild with matching `--window-size`, `--step-size`, `--canonical`, `nlist`/`nprobe`; for `--variable`, build 50/75/100/150 indexes |
| `Failed to load reference for --align` | Missing FASTA | Add `ref_sequences/<ref_name>.fasta` |
| `WFA2-lib required` at build | Library not built | `cd WFA2-lib && make all` |
| `faiss/build/faiss not found` | FAISS not built | `./build_faiss.sh` or `USE_PYTHON_FAISS=1` |
| `No reads to process` | Reads shorter than `--max-length` or < 50 with `--variable` | Check read lengths and filtering |
| `--sam requires --align` | SAM without alignment | Add `--align` |
| Slow vs Python FAISS | Different library build | Try `USE_PYTHON_FAISS=1 ./build_anniemap.sh` |

---

## Citation and third-party software

Anniemap depends on **FAISS** and **WFA2-lib**; cite those projects when publishing work that uses this tool. K-mer indexing uses a fixed **k = 5** binary presence encoding implemented in `kmer_vector.hpp`.
