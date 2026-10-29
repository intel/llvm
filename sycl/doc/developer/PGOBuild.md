# Build DPC++ with profile-guided optimization

Profile-guided optimization (PGO) uses measurements from representative
compiler workloads to optimize frequently executed paths. The DPC++ PGO driver
creates separate instrumented and optimized builds so an existing development
build is not modified.

## Prerequisites

- A host Clang installation that includes the profile runtime.
- CMake and a single-configuration generator such as Ninja. Multi-configuration
  generators are rejected because their configuration-specific tool paths are
  not supported by this workflow.
- Enough disk space for three complete DPC++ build directories.

## Build

From the repository root, run:

```bash
python3 buildbot/pgo.py --parallelism "$(nproc)"
```

The driver finds `clang` and `clang++` on `PATH`. Use `--host-clang` and
`--host-clangxx` when the host toolchain is installed elsewhere. Profile data
is merged with `llvm-profdata` from the same host toolchain because raw profile
formats may differ between LLVM releases. Use `--llvm-profdata` to select an
explicit compatible tool when it is not installed beside the host Clang.

The driver always performs a clean DPC++ build using the instrumented compiler;
this is the required training workload and provides representative C, C++,
optimization, and host code-generation coverage. Before that build, the default
training workload also runs `check-clang`, `check-llvm`, and `check-sycl` for
broader compiler and SYCL coverage. Instrumentation diagnostics can make these
FileCheck-based tests fail, so each target runs independently and its exit
status is reported without stopping the remaining workloads or the required
training build. The instrumented compiler reserves additional value-profile
counters to avoid known profiler warnings that would otherwise pollute test
output. The workflow ignores an inherited `LLVM_PROFILE_FILE` so profile data
cannot escape its isolated output directory. To replace the supplemental
targets, repeat `--training-target`:

```bash
python3 buildbot/pgo.py \
  --training-target check-sycl \
  --training-target check-sycl-unittests
```

Arguments for the existing DPC++ configuration script can be forwarded with
`--configure-arg`. For example:

```bash
python3 buildbot/pgo.py \
  --configure-arg=--no-assertions \
  --configure-arg=--use-lld
```

Use `--dry-run` to inspect every configuration, build, training, and profile
merge command without changing the build tree.

The driver requires fresh instrumented, training, profile, and optimized output
paths. Choose a new `--out-dir` for another run; this prevents profiles from an
older compiler or training workload from contaminating the result.

The resulting paths are:

- `build-pgo/instrumented`: instrumented toolchain and raw profiles;
- `build-pgo/training`: clean DPC++ build compiled by the instrumented toolchain;
- `build-pgo/dpcpp.profdata`: merged profile;
- `build-pgo/optimized`: PGO-optimized toolchain.

## Measure the result

Compare a normal Release build and the optimized build using the same source
tree, build options, machine, and compiler workload. Run each measurement
multiple times and report the median. At minimum, record compiler wall time,
peak memory, output size, and the exact revision and command line. Do not use
the profile-training workload as the only benchmark; retain a separate workload
to avoid measuring overfitting.

PGO effectiveness depends on the training workload. Profiles collected only
from host C++ compilation may not represent SYCL offload compilation or other
target architectures.
