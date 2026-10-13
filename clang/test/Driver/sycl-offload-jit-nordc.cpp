/// Tests for the SYCL non-RDC (-fno-gpu-rdc) compilation flow added in LLORG by
/// llvm/llvm-project#218089, adapted to the intel/llvm SYCL driver: spir64
/// target, --offload-new-driver, and the intel/llvm SYCL device pipeline in
/// clang-linker-wrapper (llvm-link + sycl-post-link) instead of
/// clang-sycl-linker.

/// Check the phases graph in non-RDC mode.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fno-gpu-rdc %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES-NORDC %s
// CHK-PHASES-NORDC: 6: backend, {5}, ir, (device-sycl)
// CHK-PHASES-NORDC-NEXT: 7: offload, "device-sycl (spir64-unknown-unknown)" {6}, ir
// CHK-PHASES-NORDC-NEXT: 8: llvm-offload-binary, {7}, image, (device-sycl)
// CHK-PHASES-NORDC-NEXT: 9: clang-linker-wrapper, {8}, sycl-fatbin, (device-sycl)
// CHK-PHASES-NORDC-NEXT: 10: offload, "host-sycl (x86_64{{.*}})" {2}, "device-sycl (spir64{{.*}})" {9}, ir
// CHK-PHASES-NORDC-NEXT: 11: backend, {10}, assembler, (host-sycl)
// CHK-PHASES-NORDC-NEXT: 12: assembler, {11}, object, (host-sycl)
// CHK-PHASES-NORDC-NEXT: 13: clang-linker-wrapper, {12}, image, (host-sycl)

/// With multiple architectures the packaged binary holds an entry per
/// architecture, and a single fat binary is expected to reach the host.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fno-gpu-rdc \
// RUN:   -fsycl-targets=intel_gpu_pvc,intel_gpu_bmg_g21 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES-NORDC-ARCHS %s
// CHK-PHASES-NORDC-ARCHS: 7: offload, "device-sycl (spir64_gen-unknown-unknown:bmg_g21)" {6}, ir
// CHK-PHASES-NORDC-ARCHS: 12: offload, "device-sycl (spir64_gen-unknown-unknown:pvc)" {11}, ir
// CHK-PHASES-NORDC-ARCHS-NEXT: 13: llvm-offload-binary, {7, 12}, image, (device-sycl)
// CHK-PHASES-NORDC-ARCHS-NEXT: 14: clang-linker-wrapper, {13}, sycl-fatbin, (device-sycl)

/// Multiple device triples are not supported today in non-RDC mode.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fno-gpu-rdc \
// RUN:   -fsycl-targets=spir64,spir64_x86_64 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NORDC-MULTI-TRIPLE %s
// CHK-NORDC-MULTI-TRIPLE: error: '-fno-gpu-rdc' is not supported with multiple SYCL offloading targets

/// Multiple device triples are supported in RDC mode.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fgpu-rdc \
// RUN:   -fsycl-targets=spir64,spir64_x86_64 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RDC-MULTI-TRIPLE %s
// CHK-RDC-MULTI-TRIPLE-NOT: error:
// CHK-RDC-MULTI-TRIPLE-DAG: "-cc1" "-triple" "spir64-unknown-unknown"{{.*}} "-fsycl-is-device"
// CHK-RDC-MULTI-TRIPLE-DAG: "-cc1" "-triple" "spir64_x86_64-unknown-unknown"{{.*}} "-fsycl-is-device"

/// A single target repeated is one target.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fno-gpu-rdc \
// RUN:   -fsycl-targets=spir64,spir64 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NORDC-DUP-TRIPLE %s
// CHK-NORDC-DUP-TRIPLE-NOT: error:
// CHK-NORDC-DUP-TRIPLE: clang-linker-wrapper{{.*}} "--emit-fatbin-only"

/// Check that in non-RDC mode clang-linker-wrapper finalizes the packaged
/// device images into a fat binary rather than a host object, and that the
/// binary is included into the host compilation via -foffload-include-binary.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fno-gpu-rdc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NORDC-INCLUDE %s \
// RUN:     --implicit-check-not='"-fembed-offload-object='
// CHK-NORDC-INCLUDE: clang-linker-wrapper{{.*}} "--linker-path={{.*}}llvm-link" "--emit-fatbin-only" "-o" "[[FB:.*]].syclfb"
// CHK-NORDC-INCLUDE: "-cc1"{{.*}} "-fsycl-is-host"{{.*}} "-foffload-include-binary" "[[FB]].syclfb"

/// -v reaches the wrapper. Unlike LLORG, intel/llvm does not forward -v to the
/// SYCL device compiler options: those become ocloc/JIT backend options, which
/// must not receive generic Clang flags (intel/llvm#23281).
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fno-gpu-rdc -v -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NORDC-VERBOSE %s
// CHK-NORDC-VERBOSE: clang-linker-wrapper{{.*}} "--wrapper-verbose"
// CHK-NORDC-VERBOSE-SAME: "--emit-fatbin-only"

/// -flto on a SYCL command line requests *host* LTO. It must not divert the
/// per-TU device finalize to llvm-lto, which would write bitcode where a
/// finalized device image is expected; the device link is unaffected and the
/// binary is still included at compile time.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fno-gpu-rdc \
// RUN:   -flto -c %s 2>&1 | FileCheck -check-prefix=CHK-NORDC-LTO %s \
// RUN:     --implicit-check-not=llvm-lto
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fno-gpu-rdc \
// RUN:   -flto %s 2>&1 | FileCheck -check-prefix=CHK-NORDC-LTO %s \
// RUN:     --implicit-check-not=llvm-lto
// CHK-NORDC-LTO: clang-linker-wrapper{{.*}} "--linker-path={{.*}}llvm-link" "--emit-fatbin-only"
// CHK-NORDC-LTO: "-cc1"{{.*}} "-fsycl-is-host"{{.*}} "-foffload-include-binary"

/// In non-RDC mode the split does happen while compiling.
// RUN: %clang -### -c --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fno-gpu-rdc \
// RUN:   -fsycl-device-image-split=kernel %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SPLIT-KERNEL %s \
// RUN:     --implicit-check-not='argument unused during compilation'
// CHK-SPLIT-KERNEL: clang-linker-wrapper{{.*}}"--device-linker=spir64-unknown-unknown=--module-split-mode=kernel"
