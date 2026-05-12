#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BUILD_SYCL_DIR="/home/rrudnick/llvm/build_sycl"
BUILD_SYCL_DIR="${BUILD_SYCL_DIR:-${DEFAULT_BUILD_SYCL_DIR}}"

if [[ -d "${BUILD_SYCL_DIR}/bin" ]]; then
  export PATH="${BUILD_SYCL_DIR}/bin:${PATH}"
fi

BUILD_SYCL_LD_PATH=""
if [[ -d "${BUILD_SYCL_DIR}/lib" ]]; then
  BUILD_SYCL_LD_PATH="${BUILD_SYCL_DIR}/lib"
fi
if [[ -d "${BUILD_SYCL_DIR}/lib64" ]]; then
  BUILD_SYCL_LD_PATH="${BUILD_SYCL_LD_PATH:+${BUILD_SYCL_LD_PATH}:}${BUILD_SYCL_DIR}/lib64"
fi
if [[ -n "${BUILD_SYCL_LD_PATH}" ]]; then
  export LD_LIBRARY_PATH="${BUILD_SYCL_LD_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

DEFAULT_ONEAPI_ROOT="/home/test-user/oneapi_2025.3.2.21"
ONEAPI_ROOT="${ONEAPI_ROOT:-${DEFAULT_ONEAPI_ROOT}}"

if ! command -v icpx >/dev/null 2>&1 && ! command -v dpcpp >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
  if [[ ! -f "${ONEAPI_ROOT}/setvars.sh" ]]; then
    echo "No SYCL compiler on PATH and setvars.sh not found at ${ONEAPI_ROOT}/setvars.sh" >&2
    exit 1
  fi

  set +u
  source "${ONEAPI_ROOT}/setvars.sh" >/dev/null
  set -u
fi

DEFAULT_SYCL_CXX="icpx"
if [[ -x "${BUILD_SYCL_DIR}/bin/clang++" ]]; then
  DEFAULT_SYCL_CXX="${BUILD_SYCL_DIR}/bin/clang++"
elif [[ -x "${BUILD_SYCL_DIR}/bin/icpx" ]]; then
  DEFAULT_SYCL_CXX="${BUILD_SYCL_DIR}/bin/icpx"
elif command -v icpx >/dev/null 2>&1; then
  DEFAULT_SYCL_CXX="icpx"
elif command -v dpcpp >/dev/null 2>&1; then
  DEFAULT_SYCL_CXX="dpcpp"
elif command -v clang++ >/dev/null 2>&1; then
  DEFAULT_SYCL_CXX="clang++"
fi

SYCL_CXX="${SYCL_CXX:-${DEFAULT_SYCL_CXX}}"
if [[ "${SYCL_CXX}" == */* ]]; then
  if [[ ! -x "${SYCL_CXX}" ]]; then
    echo "SYCL_CXX is set to '${SYCL_CXX}', but it is not executable." >&2
    exit 1
  fi
else
  if ! command -v "${SYCL_CXX}" >/dev/null 2>&1; then
    echo "No SYCL compiler found. Set SYCL_CXX to a SYCL-capable compiler (e.g. ${BUILD_SYCL_DIR}/bin/clang++ or icpx)." >&2
    exit 1
  fi
fi

cd "${SCRIPT_DIR}"

"${SYCL_CXX}" -std=c++17 -fsycl -O0 -g source_kernel_double_repro.cpp -o source_kernel_double_repro
./source_kernel_double_repro