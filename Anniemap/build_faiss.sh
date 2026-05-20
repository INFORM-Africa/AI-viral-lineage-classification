#!/usr/bin/env bash
# Build FAISS from source in faiss/ for macOS (Intel or Apple Silicon).
# Optimizations match what the Python wheel uses: AVX2 on Intel, generic/SVE on ARM.
# Run from the Anniemap_C++ directory: ./build_faiss.sh
#
# On macOS, OpenMP is required; install it with:  brew install libomp

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
FAISS_ROOT="$SCRIPT_DIR/faiss"
BUILD_DIR="$FAISS_ROOT/build"

if [[ ! -d "$FAISS_ROOT" ]]; then
    echo "Error: $FAISS_ROOT not found. Run from the directory that contains faiss/."
    exit 1
fi

ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Options used for all builds.
# Use -march=native on ARM so the whole library (including binary Hamming in hamdis-inl.h)
# is built for the host CPU: ensures __aarch64__ → NEON and optimal instruction selection.
RELEASE_FLAGS="-O3"
if [[ "$ARCH" == "arm64" ]]; then
    RELEASE_FLAGS="-O3 -march=native"
    echo "Using Release flags: $RELEASE_FLAGS (NEON for binary Hamming)"
fi
CMAKE_OPTS=(
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_FLAGS_RELEASE="$RELEASE_FLAGS"
    -DFAISS_ENABLE_GPU=OFF
    -DFAISS_ENABLE_PYTHON=OFF
    -DBUILD_TESTING=OFF
    -DBUILD_SHARED_LIBS=ON
    -DFAISS_ENABLE_MKL=OFF
    -DFAISS_USE_LTO=ON
)

# On macOS, Apple Clang does not ship OpenMP; use Homebrew's libomp so FindOpenMP succeeds.
if [[ "$(uname -s)" == "Darwin" ]]; then
    LIBOMP_PREFIX=""
    if command -v brew &>/dev/null; then
        LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null)"
    fi
    if [[ -z "$LIBOMP_PREFIX" || ! -d "$LIBOMP_PREFIX" ]]; then
        echo "Error: OpenMP is required to build FAISS on macOS. Install it with:"
        echo "  brew install libomp"
        echo "Then re-run this script."
        exit 1
    fi
    # Help FindOpenMP find libomp and omp.h (it looks for OpenMP_libomp_LIBRARY and omp.h).
    CMAKE_OPTS+=(
        -DCMAKE_PREFIX_PATH="${LIBOMP_PREFIX}"
        -DOpenMP_libomp_LIBRARY="${LIBOMP_PREFIX}/lib/libomp.dylib"
        -DOpenMP_CXX_INCLUDE_DIR="${LIBOMP_PREFIX}/include"
    )
    echo "Using OpenMP from: $LIBOMP_PREFIX"
fi

# BUILD_DIR is absolute ($SCRIPT_DIR/faiss/build), so cmake and make work from any cwd
if [[ "$ARCH" == "x86_64" ]]; then
    # Intel Mac: AVX2 (same as Python faiss on macOS Intel)
    echo "Configuring for Intel (AVX2)..."
    CMAKE_OPTS+=(-DFAISS_OPT_LEVEL=avx2)
    cmake "${CMAKE_OPTS[@]}" "$FAISS_ROOT"
    echo "Building faiss and faiss_avx2..."
    make -C "$BUILD_DIR" -j"$(sysctl -n hw.ncpu)" faiss faiss_avx2
    echo "Done. Use libfaiss_avx2.dylib for your C++ app:"
    echo "  FAISS_LIB=$BUILD_DIR/faiss"
    echo "  (link against libfaiss_avx2.dylib)"
elif [[ "$ARCH" == "arm64" ]]; then
    # Apple Silicon: dd = dynamic dispatch (main faiss gets NEON/SVE, like Python wheel)
    echo "Configuring for Apple Silicon (dynamic dispatch, NEON/SVE)..."
    CMAKE_OPTS+=(-DFAISS_OPT_LEVEL=dd)
    cmake "${CMAKE_OPTS[@]}" "$FAISS_ROOT"
    echo "Building faiss (with SIMD)..."
    make -C "$BUILD_DIR" -j"$(sysctl -n hw.ncpu)" faiss
    echo "Done. Use libfaiss.dylib for your C++ app (SIMD-optimized):"
    echo "  FAISS_LIB=$BUILD_DIR/faiss"
else
    echo "Unknown architecture $ARCH; building generic faiss."
    CMAKE_OPTS+=(-DFAISS_OPT_LEVEL=generic)
    cmake "${CMAKE_OPTS[@]}" "$FAISS_ROOT"
    make -C "$BUILD_DIR" -j"$(sysctl -n hw.ncpu)" faiss
fi

echo ""
echo "To build faiss_create_kmer (from Anniemap_C++):"
echo "  ./build_faiss_create_kmer.sh"
echo "Or manually (run from Anniemap_C++), for Intel (AVX2):"
echo '  g++ -std=c++17 -O2 -I faiss faiss_create_kmer.cpp -o faiss_create_kmer -L faiss/build/faiss -lfaiss_avx2 -Wl,-rpath,faiss/build/faiss'
echo "For Apple Silicon:"
echo '  g++ -std=c++17 -O2 -I faiss faiss_create_kmer.cpp -o faiss_create_kmer -L faiss/build/faiss -lfaiss -Wl,-rpath,faiss/build/faiss'
