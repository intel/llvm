// Check if -fsycl-instrument-device-code is passed to device-side -cc1
// and if ITT device libraries are pulled in.
// The following conditions must be fulfilled:
// 1. A SPIR-V-based environment must be targetted
// 2. The option must not be explicitly disabled in the Driver call

// FIXME: Force linux targets to allow for the libraries to be found.  Dummy
// inputs for --sysroot should be updated to work better for Windows.

// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-instrument-device-code --sysroot=%S/Inputs/SYCL -fsycl-targets=spir64 -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK-SPIRV,CHECK-HOST %s
// -fno-sycl-device-lib mustn't affect the linkage of ITT libraries
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-instrument-device-code --sysroot=%S/Inputs/SYCL -fno-sycl-device-lib=all -fsycl-targets=spir64 -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK-SPIRV %s

// CHECK-SPIRV: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-instrument-device-code"
// CHECK-SPIRV: llvm-link{{.*}} "-only-needed"
// CHECK-SPIRV-SAME: "{{.*}}libsycl-itt-user-wrappers.bc"
// CHECK-SPIRV-SAME: "{{.*}}libsycl-itt-compiler-wrappers.bc"
// CHECK-SPIRV-SAME: "{{.*}}libsycl-itt-stubs.bc"
// CHECK-HOST-NOT: "-cc1"{{.*}} "-fsycl-is-host"{{.*}} "-fsycl-instrument-device-code"

// ITT annotations in device code are disabled by default. However, for SYCL
// offloading, we still link ITT annotations libraries to ensure ABI
// compatibility with previous release.
// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-targets=spir64 -### \
// RUN:   --sysroot=%S/Inputs/SYCL %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK-ITT-LINK-ONLY %s
// RUN: %clangxx -fsycl --no-offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda -nocudalib -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK-NONPASSED %s

// CHECK-ITT-LINK-ONLY-NOT: "-fsycl-instrument-device-code"
// CHECK-ITT-LINK-ONLY: llvm-link{{.*}} {{.*}}libsycl-itt-{{.*}}

// RUN: %clangxx -fsycl --no-offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:   -fno-sycl-instrument-device-code -fsycl-targets=spir64 -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK-NONPASSED %s
// RUN: %clangxx -fsycl --no-offload-new-driver --sysroot=%s/Inputs/SYCL \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda -fno-sycl-instrument-device-code \
// RUN:   -nocudalib -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK-NONPASSED %s

// CHECK-NONPASSED-NOT: "-fsycl-instrument-device-code"
// CHECK-NONPASSED-NOT: llvm-link{{.*}} {{.*}}libsycl-itt-{{.*}}.bc"
