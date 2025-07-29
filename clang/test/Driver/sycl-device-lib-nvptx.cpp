// Tests specific to `-fsycl-targets=nvptx64-nvidia-nvptx`
// Verify that the correct devicelib linking actions are spawned by the driver.
// Check also if the correct warnings are generated.

// UNSUPPORTED: system-windows

// Check if internal libraries are still linked against when linkage of all
// device libs is manually excluded.
// RUN: %clangxx -ccc-print-phases -std=c++11 -fsycl -fno-sycl-device-lib=all --sysroot=%S/Inputs/SYCL \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda -fsycl-instrument-device-code %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-NO-DEVLIB %s

// CHK-NO-DEVLIB-NOT: {{[0-9]+}}: input, "{{.*}}devicelib-nvptx64-nvidia-cuda.bc", ir, (device-sycl, sm_50)
// CHK-NO-DEVLIB: [[LIB1:[0-9]+]]: input, "{{.*}}libsycl-itt-user-wrappers.bc", ir, (device-sycl, sm_50)
// CHK-NO-DEVLIB-NOT: {{[0-9]+}}: input, "{{.*}}devicelib-nvptx64-nvidia-cuda.bc", ir, (device-sycl, sm_50)
// CHK-NO-DEVLIB: [[LIB2:[0-9]+]]: input, "{{.*}}libsycl-itt-compiler-wrappers.bc", ir, (device-sycl, sm_50)
// CHK-NO-DEVLIB-NOT: {{[0-9]+}}: input, "{{.*}}devicelib-nvptx64-nvidia-cuda.bc", ir, (device-sycl, sm_50)
// CHK-NO-DEVLIB: [[LIB3:[0-9]+]]: input, "{{.*}}libsycl-itt-stubs.bc", ir, (device-sycl, sm_50)
// CHK-NO-DEVLIB-NOT: {{[0-9]+}}: input, "{{.*}}devicelib-nvptx64-nvidia-cuda.bc", ir, (device-sycl, sm_50)
// CHK-NO-DEVLIB: {{[0-9]+}}: linker, {{{.*}}[[LIB1]], [[LIB2]], [[LIB3]]{{.*}}}, ir, (device-sycl, sm_50)

// Check that the -fsycl-device-lib flag has no effect when "all" is specified.
// RUN: %clangxx -ccc-print-phases -std=c++11 -fsycl -fsycl-device-lib=all --sysroot=%S/Inputs/SYCL \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ALL %s

// Check that the -fsycl-device-lib flag has no effect when subsets of libs
// are specified.
// RUN: %clangxx -ccc-print-phases -std=c++11 --sysroot=%S/Inputs/SYCL \
// RUN: -fsycl -fsycl-device-lib=libc,libm-fp32,libm-fp64,libimf-fp32,libimf-fp64,libimf-bf16,libm-bfloat16 \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ALL %s

// Check that -fno-sycl-device-lib is ignored when it does not contain "all".
// A warning should be printed that the flag got ignored.
// RUN: %clangxx -ccc-print-phases -std=c++11 -fsycl --sysroot=%S/Inputs/SYCL \
// RUN: -fno-sycl-device-lib=libc,libm-fp32,libm-fp64,libimf-fp32,libimf-fp64,libimf-bf16,libm-bfloat16 \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHK-UNUSED-WARN,CHK-ALL %s

// CHK-UNUSED-WARN: warning: argument unused during compilation: '-fno-sycl-device-lib='
// CHK-ALL: [[DEVLIB:[0-9]+]]: input, "{{.*}}devicelib-nvptx64-nvidia-cuda.bc", ir, (device-sycl, sm_50)
// CHK-ALL: {{[0-9]+}}: linker, {{{.*}}[[DEVLIB]]{{.*}}}, ir, (device-sycl, sm_50)

// Check that llvm-link uses the "-only-needed" flag.
// Not using the flag breaks kernel bundles.
// RUN: %clangxx -### -nocudalib -fno-sycl-libspirv --sysroot=%S/Inputs/SYCL -fsycl -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ONLY-NEEDED %s

// CHK-ONLY-NEEDED: llvm-link"{{.*}}"-only-needed"{{.*}}"{{.*}}devicelib-nvptx64-nvidia-cuda.bc"{{.*}}
