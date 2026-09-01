#!/usr/bin/env bash
#
# build-version-libdevice.sh - Build a legacy SYCL version's device libraries
# from its OWN vendored sources, using the *new* clang, into that version's
# staged tree at lib/dpcpp-<major>/sycl/.
#
# We emit the SAME per-library .bc files the RTC device-library linker loads by
# name (DeviceCompilation.cpp: getDeviceLibraries + the bf16 device-image path),
# so the runtime needs NO change beyond its version-aware toolchain prefix: it
# finds each version's libraries in that version's subtree exactly as it finds
# the live ones. This keeps behavior identical to upstream (same libs, same
# LinkOnlyNeeded, same bf16 handling) with zero merge-semantics risk, and pins
# each libdevice implementation to its matching header version.
#
# Emitted (Linux/SPIR-V), matching the RTC's load set:
#   libsycl-crt.bc                       <- crt_wrapper.cpp
#   libsycl-cmath.bc                     <- cmath_wrapper.cpp
#   libsycl-imf.bc                       <- imf_wrapper.cpp + all imf fallback srcs
#   libsycl-itt-stubs.bc                 <- itt_stubs.cpp            (profiling only)
#   libsycl-itt-compiler-wrappers.bc     <- itt_compiler_wrappers.cpp
#   libsycl-itt-user-wrappers.bc         <- itt_user_wrappers.cpp
#   libsycl-fallback-bfloat16.bc         <- fallback-bfloat16.cpp    (bf16 kernels)
#   libsycl-native-bfloat16.bc           <- bfloat16_wrapper.cpp
# (cmath-fp64 / complex are NOT loaded by the RTC, so they are not emitted.)
#
# All path computation lives here so the generated ninja rule carries no '$'.
#
# Usage:
#   build-version-libdevice.sh <clang> <llvm-link> <version-prefix> \
#                              <latest-prefix> <vendored-libdevice-dir>

set -euo pipefail

CLANG="$1"; LINK="$2"; VPREFIX="$3"; LATEST="$4"; SRC="$5"

# Discover the libdevice subdir (lib/dpcpp-<major>/sycl) from the live tree so it
# matches getLibPathSuffix() at runtime.
LIBSUB="$(basename "$(echo "${LATEST}"/lib/dpcpp-* )")"
OUTDIR="${VPREFIX}/lib/${LIBSUB}/sycl"
mkdir -p "${OUTDIR}"

OPTS=(-Wno-sycl-strict -Wno-undefined-internal -sycl-std=2020
      --target=x86_64-unknown-linux-gnu -std=c++17
      -fsycl-device-only -fsycl-device-obj=llvmir)

# Compile one source to a device bitcode.
compile() { "${CLANG}" "${OPTS[@]}" -I "${SRC}" -I "${SRC}/imf" "$1" -o "$2"; }

# Simple one-source libraries: <output-lib-name> <source-file>
declare -A SIMPLE=(
  [libsycl-crt]=crt_wrapper.cpp
  [libsycl-cmath]=cmath_wrapper.cpp
  [libsycl-itt-stubs]=itt_stubs.cpp
  [libsycl-itt-compiler-wrappers]=itt_compiler_wrappers.cpp
  [libsycl-itt-user-wrappers]=itt_user_wrappers.cpp
  [libsycl-fallback-bfloat16]=fallback-bfloat16.cpp
  [libsycl-native-bfloat16]=bfloat16_wrapper.cpp
)
for lib in "${!SIMPLE[@]}"; do
  s="${SRC}/${SIMPLE[$lib]}"
  [[ -f "${s}" ]] || { echo "warn: missing ${s}, skipping ${lib}" >&2; continue; }
  compile "${s}" "${OUTDIR}/${lib}.bc"
done

# libsycl-imf: wrapper + all fp32/fp64/bf16 fallback sources, llvm-link'd.
IMF_SRCS=(imf_wrapper.cpp
          imf_utils/integer_misc.cpp imf_utils/half_convert.cpp
          imf_utils/float_convert.cpp imf_utils/simd_emulate.cpp
          imf_utils/fp32_round.cpp imf/imf_inline_fp32.cpp imf/imf_fp32_dl.cpp
          imf_utils/double_convert.cpp imf_utils/fp64_round.cpp
          imf/imf_inline_fp64.cpp imf/imf_fp64_dl.cpp
          imf_utils/bfloat16_convert.cpp imf/imf_inline_bf16.cpp)
TMP="$(mktemp -d)"; trap 'rm -rf "${TMP}"' EXIT
imf_bcs=()
for s in "${IMF_SRCS[@]}"; do
  [[ -f "${SRC}/${s}" ]] || continue
  o="${TMP}/$(echo "${s}" | tr '/' '_').bc"
  compile "${SRC}/${s}" "${o}"
  imf_bcs+=("${o}")
done
"${LINK}" "${imf_bcs[@]}" -o "${OUTDIR}/libsycl-imf.bc"

echo "wrote $(ls "${OUTDIR}"/libsycl-*.bc | wc -l) device libraries to ${OUTDIR}"
