#!/usr/bin/env bash
# Build faiss_create_kmer against the FAISS built by build_faiss.sh.
# Run from Anniemap_C++: ./build_faiss_create_kmer.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Must match build_faiss.sh: build dir is $SCRIPT_DIR/faiss/build, libs in build/faiss/
FAISS_ROOT="$SCRIPT_DIR/faiss"
FAISS_LIB_DIR="$FAISS_ROOT/build/faiss"

if [[ ! -d "$FAISS_LIB_DIR" ]]; then
    echo "Error: $FAISS_LIB_DIR not found. Run ./build_faiss.sh first."
    exit 1
fi

ARCH=$(uname -m)
CXX="${CXX:-g++}"

# Include path: faiss repo root so that #include <faiss/IndexBinaryFlat.h> works
INC="-I$FAISS_ROOT"
LIB_DIR="-L$FAISS_LIB_DIR"
RPATH="-Wl,-rpath,$FAISS_LIB_DIR"

if [[ "$ARCH" == "x86_64" ]]; then
    LIB="-lfaiss_avx2"
    if [[ ! -f "$FAISS_LIB_DIR/libfaiss_avx2.dylib" ]]; then
        echo "Error: libfaiss_avx2.dylib not found. Run ./build_faiss.sh (Intel Mac)."
        exit 1
    fi
else
    LIB="-lfaiss"
    if [[ ! -f "$FAISS_LIB_DIR/libfaiss.dylib" ]]; then
        echo "Error: libfaiss.dylib not found. Run ./build_faiss.sh first."
        exit 1
    fi
fi

echo "Building faiss_create_kmer (C++17, Release)..."
$CXX -std=c++17 -O2 $INC faiss_create_kmer.cpp -o faiss_create_kmer $LIB_DIR $LIB $RPATH

echo "Done: ./faiss_create_kmer"
