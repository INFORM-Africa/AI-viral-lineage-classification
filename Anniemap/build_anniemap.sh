#!/usr/bin/env bash
# Build anniemap.cpp (FASTQ -> k-mer -> FAISS search -> TSV).
# By default uses FAISS built from source (./build_faiss.sh).
# To use the pre-built FAISS from your Python/conda env instead, run:
#   USE_PYTHON_FAISS=1 ./build_anniemap.sh
#   or: ./build_anniemap.sh --python-faiss
# If the C++ binary is slower than Python anniemap.py, use --python-faiss to link the same FAISS as pip/conda.
# Requires: zlib, kseq.h in project root. For source build: run ./build_faiss.sh first.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

USE_PYTHON_FAISS="${USE_PYTHON_FAISS:-0}"
[[ "$1" == "--python-faiss" ]] && USE_PYTHON_FAISS=1 && shift
[[ "$USE_PYTHON_FAISS" == "1" ]] || USE_PYTHON_FAISS=0

FAISS_ROOT="$SCRIPT_DIR/faiss"
ARCH=$(uname -m)
CXX="${CXX:-g++}"
INC_FAISS="-I$FAISS_ROOT"
INC_KSEQ="-I$SCRIPT_DIR"
LIB_DIR=""
RPATH=""
LIB=""

if [[ "$USE_PYTHON_FAISS" -eq 1 ]]; then
    # Use FAISS from Python/conda (pre-built wheel or conda package).
    echo "Using FAISS from Python/conda installation..."
    if [[ -n "$CONDA_PREFIX" ]] && [[ -d "$CONDA_PREFIX/lib" ]]; then
        # Conda: libfaiss*.dylib or .so in $CONDA_PREFIX/lib, headers in include.
        FAISS_LIB_DIR="$CONDA_PREFIX/lib"
        INC_FAISS="-I$CONDA_PREFIX/include -I$FAISS_ROOT"
        LIB_DIR="-L$FAISS_LIB_DIR"
        RPATH="-Wl,-rpath,$FAISS_LIB_DIR"
        if [[ "$ARCH" == "x86_64" ]] && [[ -f "$FAISS_LIB_DIR/libfaiss_avx2.dylib" ]]; then
            LIB="-lfaiss_avx2"
        elif [[ "$ARCH" == "x86_64" ]] && [[ -f "$FAISS_LIB_DIR/libfaiss_avx2.so" ]]; then
            LIB="-lfaiss_avx2"
        elif [[ -f "$FAISS_LIB_DIR/libfaiss.dylib" ]]; then
            LIB="-lfaiss"
        elif [[ -f "$FAISS_LIB_DIR/libfaiss.so" ]]; then
            LIB="-lfaiss"
        fi
        if [[ -z "$LIB" ]]; then
            echo "Error: No libfaiss in $FAISS_LIB_DIR. Install with: conda install -c pytorch faiss-cpu"
            exit 1
        fi
        echo "  Conda lib: $FAISS_LIB_DIR $LIB"
    else
        # Pip / system Python: use the faiss package's _swigfaiss*.so (contains FAISS C++ symbols).
        FAISS_PKG_DIR="$(python3 -c "import faiss; import os; print(os.path.dirname(faiss.__file__))" 2>/dev/null)" || true
        if [[ -z "$FAISS_PKG_DIR" || ! -d "$FAISS_PKG_DIR" ]]; then
            echo "Error: Could not find faiss package. Install with: pip install faiss-cpu (or conda install -c pytorch faiss-cpu)"
            exit 1
        fi
        EXT=".so"
        if [[ "$ARCH" == "x86_64" ]] && [[ -f "$FAISS_PKG_DIR/_swigfaiss_avx2$EXT" ]]; then
            FAISS_SO="$FAISS_PKG_DIR/_swigfaiss_avx2$EXT"
        elif [[ -f "$FAISS_PKG_DIR/_swigfaiss$EXT" ]]; then
            FAISS_SO="$FAISS_PKG_DIR/_swigfaiss$EXT"
        else
            echo "Error: No _swigfaiss*.so in $FAISS_PKG_DIR"
            exit 1
        fi
        LIB="$FAISS_SO"
        RPATH="-Wl,-rpath,$FAISS_PKG_DIR"
        echo "  Python faiss: $FAISS_SO"
    fi
else
    # Use FAISS built from source in faiss/
    FAISS_LIB_DIR="$FAISS_ROOT/build/faiss"
    if [[ ! -d "$FAISS_LIB_DIR" ]]; then
        echo "Error: $FAISS_LIB_DIR not found. Run ./build_faiss.sh first, or use USE_PYTHON_FAISS=1 ./build_anniemap.sh"
        exit 1
    fi
    LIB_DIR="-L$FAISS_LIB_DIR"
    RPATH="-Wl,-rpath,$FAISS_LIB_DIR"
    if [[ "$ARCH" == "x86_64" ]]; then
        LIB="-lfaiss_avx2"
        [[ -f "$FAISS_LIB_DIR/libfaiss_avx2.dylib" ]] || [[ -f "$FAISS_LIB_DIR/libfaiss_avx2.so" ]] || { echo "Error: libfaiss_avx2 not found."; exit 1; }
    else
        LIB="-lfaiss"
        [[ -f "$FAISS_LIB_DIR/libfaiss.dylib" ]] || [[ -f "$FAISS_LIB_DIR/libfaiss.so" ]] || { echo "Error: libfaiss not found."; exit 1; }
    fi
fi

# OpenMP: needed so FAISS search uses multiple threads (omp_set_num_threads).
# On macOS with Homebrew: brew install libomp, then we add flags.
OMP_FLAGS=""
OMP_LIBS=""
if [[ "$(uname -s)" == "Darwin" ]] && command -v brew &>/dev/null; then
    LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null)"
    if [[ -n "$LIBOMP_PREFIX" && -d "$LIBOMP_PREFIX" ]]; then
        OMP_FLAGS="-Xpreprocessor -fopenmp -I$LIBOMP_PREFIX/include"
        OMP_LIBS="-L$LIBOMP_PREFIX/lib -lomp"
    fi
else
    # Linux / others: -fopenmp defines _OPENMP and links libgomp/libomp
    OMP_FLAGS="-fopenmp"
    OMP_LIBS="-fopenmp"
fi

# WFA2: required for --align (ends-free alignment). Build WFA2-lib first: cd WFA2-lib && make all
WFA2_ROOT="$SCRIPT_DIR/WFA2-lib"
WFA2_LIB="$WFA2_ROOT/lib"
WFA2_INC="$WFA2_ROOT"
if [[ ! -d "$WFA2_LIB" || ! -f "$WFA2_LIB/libwfacpp.a" ]]; then
    echo "Error: WFA2-lib required for anniemap. Build with: cd WFA2-lib && make all"
    exit 1
fi
INC_WFA2="-I$WFA2_INC"
LIB_WFA2="-L$WFA2_LIB -lwfacpp -lm"
echo "Using WFA2 from $WFA2_ROOT (--align enabled)."

echo "Building anniemap (C++17, Release)..."
$CXX -std=c++17 -O2 $OMP_FLAGS $INC_FAISS $INC_KSEQ $INC_WFA2 anniemap.cpp -o anniemap $LIB_DIR $LIB $RPATH $LIB_WFA2 -lz $OMP_LIBS

echo "Done: ./anniemap ..."
