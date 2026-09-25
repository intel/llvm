#!/usr/bin/env bash
#
# snapshot-version.sh - Freeze a released SYCL toolchain into a vendored,
# long-term-support (LTS) snapshot under sycl-jit/sycl-toolchains/.
#
# The RTC/JIT device compiler embedded in libsycl-jit.so can then compile user
# SYCL source against *this* frozen header set + libdevice, in addition to the
# live ("latest") one, so a single driver supports multiple SYCL versions.
#
# What it captures (source only, via `git archive` of a committed release tag):
#   include/{sycl,CL,std}   from sycl/include/{sycl,CL,std}
#   libdevice/              from libdevice/            (built to .bc by the NEW clang)
#   VERSION                 label + ref + resolved SHA (provenance record)
#
# "latest" is NOT produced here: it always tracks the current working tree and
# is assembled at build time from the live sycl-headers / libsycldevice targets.
# See README.md in this directory.
#
# ONE-TIME IMPORT: run this once per version, then `git add` + commit the result.
# From then on the snapshot is tracked source you maintain with normal git
# commits (e.g. new-clang compatibility fixes) -- git IS the patch history.
# Do NOT `--force` re-run a version you have since modified: it re-imports the
# pristine release and discards your tracked divergence. `VERSION` keeps the
# origin SHA for provenance.
#
# Usage:
#   ./snapshot-version.sh <label> [<git-ref>] [--remote <name>] [--force]
#
#   <label>    Version label, e.g. 6.2.2  -> dir sycl-v-6.2.2/
#   <git-ref>  Tag/commit to snapshot. Default: v<label> (e.g. v6.2.2).
#   --remote   Remote to fetch the ref from if missing. Default: intel.
#   --force    Overwrite an existing snapshot dir.
#
# Example:
#   ./snapshot-version.sh 6.3.0
#   ./snapshot-version.sh 6.2.2 v6.2.2 --remote intel

set -euo pipefail

REMOTE=intel
FORCE=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE="$2"; shift 2 ;;
    --force)  FORCE=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    -*) echo "error: unknown option '$1'" >&2; exit 2 ;;
    *)  POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ ${#POSITIONAL[@]} -lt 1 ]]; then
  echo "error: missing <label>. See --help." >&2
  exit 2
fi
LABEL="${POSITIONAL[0]}"
REF="${POSITIONAL[1]:-v${LABEL}}"

ROOT="$(git rev-parse --show-toplevel)"
DEST="${ROOT}/sycl-jit/sycl-toolchains/sycl-v-${LABEL}"

# The SYCL header subtrees the `sycl-headers` install component ships from
# committed source (sycl/CMakeLists.txt).
HEADER_PATHS=(sycl/include/sycl sycl/include/CL sycl/include/std)
LIBDEVICE_PATH=libdevice
# unified-runtime is a separate in-repo top-level project. Its layout differs by
# release: older tags keep headers flat under unified-runtime/include/ (and SYCL
# headers spell the include as <ur_api.h>); newer ones nest them under
# unified-runtime/include/unified-runtime/ (spelled <unified-runtime/ur_api.h>).
# We always land them under include/unified-runtime/ and rely on the RTC build
# adding BOTH `-I include` and `-I include/unified-runtime`, so either spelling
# resolves. See README.md ("Include-path contract").
UR_PATH=unified-runtime/include
# These SYCL headers are build-generated, not committed source, so they must be
# (re)generated into the snapshot:
#   device_aspect_macros.hpp  <- committed generator script, input = include/sycl
#   version.hpp, feature_test.hpp <- configure_file() of source/*.in templates
GEN_SRC_PATH=sycl/source

if [[ -e "${DEST}" ]]; then
  if [[ "${FORCE}" -eq 1 ]]; then
    echo ">> removing existing ${DEST}"
    rm -rf "${DEST}"
  else
    echo "error: ${DEST} already exists (use --force to overwrite)." >&2
    exit 1
  fi
fi

# Ensure the ref is available locally; shallow-fetch the single tag if not.
if ! git rev-parse -q --verify "${REF}^{commit}" >/dev/null 2>&1; then
  echo ">> ${REF} not present locally; fetching from '${REMOTE}'"
  git fetch --depth 1 "${REMOTE}" "refs/tags/${REF}:refs/tags/${REF}"
fi
SHA="$(git rev-parse "${REF}^{commit}")"

echo ">> snapshotting ${REF} (${SHA}) -> ${DEST}"
mkdir -p "${DEST}/include" "${DEST}/libdevice"

# Headers: strip the leading "sycl/include" so entries land under include/.
#   sycl/include/sycl/foo.hpp  ->  include/sycl/foo.hpp
# `git -C "${ROOT}"` so pathspecs resolve from the repo root regardless of cwd.
git -C "${ROOT}" archive "${REF}" -- "${HEADER_PATHS[@]}" \
  | tar -x --strip-components=2 -C "${DEST}/include"

# libdevice: strip the leading "libdevice" component.
#   libdevice/device_math.h    ->  libdevice/device_math.h
git -C "${ROOT}" archive "${REF}" -- "${LIBDEVICE_PATH}" \
  | tar -x --strip-components=1 -C "${DEST}/libdevice"

# unified-runtime headers -> include/unified-runtime/ (strip "unified-runtime/include").
mkdir -p "${DEST}/include/unified-runtime"
git -C "${ROOT}" archive "${REF}" -- "${UR_PATH}" \
  | tar -x --strip-components=2 -C "${DEST}/include/unified-runtime"

# Generated headers. Extract the release's own source/ templates + generator to a
# temp dir and run them, so the generated output matches THIS version.
GEN_TMP="$(mktemp -d)"
trap 'rm -rf "${GEN_TMP}"' EXIT
git -C "${ROOT}" archive "${REF}" -- "${GEN_SRC_PATH}" \
  | tar -x --strip-components=1 -C "${GEN_TMP}"    # -> ${GEN_TMP}/source/*

# device_aspect_macros.hpp: committed python generator, args = <in sycl dir> <out sycl dir>
python3 "${GEN_TMP}/source/device_aspect_macros_generator.py" \
  "${DEST}/include/sycl" "${DEST}/include/sycl"

# version.hpp / feature_test.hpp: real configure_file() via `cmake -P`. Version
# numbers come from the label; ext feature macros left at their template default
# (undefined -> 0). NOTE: a shipped release may enable specific backend ext
# macros; if a version's RTC tests depend on those, override them here or lift
# feature_test.hpp from that release package.
IFS='.' read -r MAJ MIN PAT _ <<< "${LABEL}"
cat > "${GEN_TMP}/configure.cmake" <<CMAKE
set(SYCL_MAJOR_VERSION "${MAJ:-0}")
set(SYCL_MINOR_VERSION "${MIN:-0}")
set(SYCL_PATCH_VERSION "${PAT:-0}")
set(__SYCL_COMPILER_VERSION "00000000")
configure_file("\${IN}" "\${OUT}")   # both \${VAR} and @VAR@ forms are substituted
CMAKE
for pair in "version.hpp.in:version.hpp" "feature_test.hpp.in:feature_test.hpp"; do
  cmake -DIN="${GEN_TMP}/source/${pair%%:*}" \
        -DOUT="${DEST}/include/sycl/${pair##*:}" \
        -P "${GEN_TMP}/configure.cmake" >/dev/null
done

# Provenance record. Deterministic (no timestamp) so re-running is a no-op diff.
cat > "${DEST}/VERSION" <<EOF
label=${LABEL}
ref=${REF}
sha=${SHA}
EOF

echo ">> done."
echo "   headers:   $(find "${DEST}/include" -type f | wc -l) files"
echo "   libdevice: $(find "${DEST}/libdevice" -type f | wc -l) files"
echo "   size:      $(du -sh "${DEST}" | cut -f1)"
echo
echo "Next: rebuild sycl-jit (CMake auto-discovers sycl-v-* dirs) and run the"
echo "RTC E2E tests against this version. Passing == promote/keep; failing =="
echo "old sources need a patch (commit the fix on top of this snapshot)."
